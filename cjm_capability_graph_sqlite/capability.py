"""Capability implementation for Context Graph using SQLite.

Stage 8 (Option C / PILLAR 1c): the tool re-bases onto ToolCapability. The
graph store IS this tool's persistence, so unlike the compute tools there is
no cache bookend to lift out -- GenericGraphStorageAdapter is a pure
wire-normalize + forward adapter. The domain base
cjm_graph_capability_system.GraphCapability dissolves; the data nouns come
straight from cjm_context_graph_primitives (graph-capability-system only
re-exported them). get_plugin_metadata is gone -- name/version derive from
importlib.metadata and the default db_path from the substrate-injected
per-capability CAPABILITY_DATA_DIR.

Stage 4: the typed query surface (expressions + results from the
data-primitives lib; per-backend translation owned by THIS capability)."""

import dataclasses
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from cjm_capability_graph_sqlite.query_translation import translate_edge_query, translate_node_query
from cjm_context_graph_primitives.graph import GraphContext, GraphEdge, GraphNode
from cjm_context_graph_primitives.provenance import SourceRef
from cjm_context_graph_primitives.query import (EdgeQuery, EdgeQueryResult, NodeQuery,
                                                NodeQueryResult, RawQuery, RawQueryResult)
from cjm_substrate.core.capability import ToolCapability
from cjm_substrate.core.errors import CapabilityInputError
from cjm_substrate.utils.validation import (config_to_dict, dataclass_to_jsonschema, dict_to_config,
                                            SCHEMA_DESC, SCHEMA_TITLE)


@dataclass
class SQLiteGraphCapabilityConfig:
    """Configuration for SQLite Graph Capability."""
    db_path: Optional[str] = field(
        default=None,
        metadata={
            SCHEMA_TITLE: "Database Path",
            SCHEMA_DESC: "Absolute path to SQLite DB. If None, uses default env path."
        }
    )
    readonly: bool = field(
        default=False,
        metadata={
            SCHEMA_TITLE: "Read Only",
            SCHEMA_DESC: "Open database in read-only mode."
        }
    )


class SQLiteGraphCapability(ToolCapability):
    """Local, file-backed Context Graph implementation using SQLite.

    Stores nodes and edges in relational tables with JSON payloads for
    properties::

        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            properties JSON,
            sources JSON,
            created_at REAL,
            updated_at REAL
        );

        CREATE TABLE edges (  -- foreign keys give cascade delete
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            properties JSON,
            created_at REAL,
            updated_at REAL,
            FOREIGN KEY(source_id) REFERENCES nodes(id) ON DELETE CASCADE,
            FOREIGN KEY(target_id) REFERENCES nodes(id) ON DELETE CASCADE
        );
    """

    config_class = SQLiteGraphCapabilityConfig

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config: SQLiteGraphCapabilityConfig = None
        self._db_path: str = None

    @property
    def name(self) -> str:  # Tool identity, derived from the installed distribution (PILLAR 1c)
        """Get the tool name (the installed distribution name)."""
        from importlib.metadata import metadata, packages_distributions
        # `__package__` is None in a notebook/__main__ context, so guard the
        # ffmpeg-template derivation with the known package module name.
        pkg = __package__ or "cjm_capability_graph_sqlite"
        dist = (packages_distributions().get(pkg) or [pkg.replace("_", "-")])[0]
        return metadata(dist)["Name"]

    @property
    def version(self) -> str:  # Tool version
        """Get the tool version string."""
        from cjm_capability_graph_sqlite import __version__
        return __version__

    def get_current_config(self) -> Dict[str, Any]:  # Current configuration as dictionary
        """Return current configuration state, with db_path RESOLVED to the effective path.

        The raw config's db_path may be None/'' (manifest-default load) while the
        capability resolves the real store at initialize (config override or the
        substrate data-dir convention). Downstream cores derive run provenance and
        the SIDECAR WRITE-JOURNAL path from this echo — reporting the raw None sent
        pipeline writes UNJOURNALED (found live 2026-07-14, first journaled ingestion
        run; the 6dfe00e9 scaffolding-as-defaults family / D19 lesson)."""
        if not self.config:
            return {}
        cfg = config_to_dict(self.config)
        if getattr(self, "_db_path", None):
            cfg["db_path"] = str(self._db_path)
        return cfg

    def get_config_schema(self) -> Dict[str, Any]:  # JSON Schema for configuration
        """Return JSON Schema for UI generation."""
        return dataclass_to_jsonschema(SQLiteGraphCapabilityConfig)

    def initialize(
        self,
        config: Optional[Any] = None  # Configuration dataclass, dict, or None
    ) -> None:
        """Initialize DB connection and schema."""
        self.config = dict_to_config(SQLiteGraphCapabilityConfig, config or {})

        # Determine DB path (config override > substrate-injected default).
        # The graph store IS this tool's persistence (unlike the compute tools
        # whose cache moved to the adapter), so db_path stays a TOOL concern.
        # The substrate owns the per-capability data-dir convention: it injects
        # CAPABILITY_DATA_DIR (= CJM_CAPABILITY_DATA_DIR/<capability-name>), so the tool
        # consumes that join rather than re-deriving it (PILLAR 1c verify-(c)).
        self._db_path = self.config.db_path or self._default_db_path()

        self.logger.info(f"Initializing SQLite Graph at: {self._db_path}")
        # SG-41: a read-only graph must pre-exist; skip schema creation (the
        # read-only connection cannot run CREATE TABLE).
        if not self.config.readonly:
            self._init_db()

    def _default_db_path(self) -> str:  # Default graph-store path (substrate-owned per-capability data dir)
        """Resolve the default DB path from the substrate-injected per-capability
        `CAPABILITY_DATA_DIR` (= `CJM_CAPABILITY_DATA_DIR/<capability-name>`). Falls back
        to the same convention for direct/un-proxied use (e.g. notebook tests)."""
        data_dir = os.environ.get("CAPABILITY_DATA_DIR")
        if not data_dir:
            root = os.environ.get(
                "CJM_CAPABILITY_DATA_DIR", os.path.join(os.getcwd(), ".cjm", "data")
            )
            data_dir = os.path.join(root, "cjm-capability-graph-sqlite")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, "context_graph.db")

    def _init_db(self) -> None:
        """Create tables and indices."""
        with self._connect() as con:
            # Enable Foreign Keys
            con.execute("PRAGMA foreign_keys = ON;")

            # Nodes Table
            con.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    properties JSON,
                    sources JSON,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label);")

            # Edges Table
            con.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties JSON,
                    created_at REAL,
                    updated_at REAL,
                    FOREIGN KEY(source_id) REFERENCES nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(target_id) REFERENCES nodes(id) ON DELETE CASCADE
                )
            """)
            con.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relation_type);")
            # Stage 4 (H7): composite endpoint+type indexes — the typed
            # translation's correlated EXISTS probes are (endpoint=?,
            # relation_type=?); without these the planner picks the
            # low-cardinality relation_type index and SCANS (162s -> ms at
            # 13k-segment corpus scale). IF NOT EXISTS = existing DBs pick
            # them up on the next initialize.
            con.execute("CREATE INDEX IF NOT EXISTS idx_edges_source_type ON edges(source_id, relation_type);")
            con.execute("CREATE INDEX IF NOT EXISTS idx_edges_target_type ON edges(target_id, relation_type);")
            # H7 companion: planner statistics. Without sqlite_stat1 the
            # correlated-probe plans pick the WRONG composite index (the
            # target-side probe scans a whole document's PART_OF rows per
            # outer row: 5s; source-side probe: 8ms at corpus scale).
            # Cheap (~40ms at 13k segments); writable opens only. Guarded:
            # stats are an OPTIMIZATION — a corrupted DB must still LOAD so
            # integrity_check (the designed G3 gate) can diagnose it, never
            # fail at initialize via a stats side-effect.
            try:
                con.execute("ANALYZE;")
            except sqlite3.DatabaseError as e:
                logging.warning(f"ANALYZE skipped (run integrity_check): {e}")

    def _connect(self) -> sqlite3.Connection:  # Open DB connection (honors readonly; sets WAL + FKs)
        """Open a SQLite connection honoring the `readonly` config (SG-41).

        Read-only config opens the DB with URI `mode=ro` so any write raises at
        the SQLite layer. Read-write connections assert `journal_mode=WAL` (better
        concurrent-reader behavior; persists at the DB-file level) on each open.
        Foreign keys are enabled on every connection.
        """
        if self.config and self.config.readonly:
            con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        else:
            con = sqlite3.connect(self._db_path)
            con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA foreign_keys = ON;")
        return con

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _row_to_node(
        self,
        row: Tuple  # DB row: (id, label, properties_json, sources_json, created_at, updated_at)
    ) -> GraphNode:  # Reconstructed GraphNode
        """Convert DB row to GraphNode DTO."""
        props = json.loads(row[2]) if row[2] else {}
        sources_raw = json.loads(row[3]) if row[3] else []
        sources = [SourceRef.from_dict(s) for s in sources_raw]
        return GraphNode(
            id=row[0], label=row[1], properties=props, sources=sources,
            created_at=row[4] if len(row) > 4 else None,
            updated_at=row[5] if len(row) > 5 else None,
        )

    def _row_to_edge(
        self,
        row: Tuple  # DB row: (id, source_id, target_id, relation_type, properties_json, created_at, updated_at)
    ) -> GraphEdge:  # Reconstructed GraphEdge
        """Convert DB row to GraphEdge DTO."""
        props = json.loads(row[4]) if row[4] else {}
        return GraphEdge(
            id=row[0], source_id=row[1], target_id=row[2],
            relation_type=row[3], properties=props,
            created_at=row[5] if len(row) > 5 else None,
            updated_at=row[6] if len(row) > 6 else None,
        )

    def _dict_to_node(
        self,
        data: Dict[str, Any]  # Node data as dictionary
    ) -> GraphNode:  # Reconstructed GraphNode
        """Convert dictionary to GraphNode (nested sources via GraphNode.from_dict)."""
        return GraphNode.from_dict(data)

    def _dict_to_edge(
        self,
        data: Dict[str, Any]  # Edge data as dictionary
    ) -> GraphEdge:  # Reconstructed GraphEdge
        """Convert dictionary to GraphEdge."""
        return GraphEdge(
            id=data["id"],
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            properties=data.get("properties", {}),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    # -------------------------------------------------------------------------
    # CREATE
    # -------------------------------------------------------------------------

    def add_nodes(
        self,
        nodes: List[GraphNode]  # Nodes to create
    ) -> List[str]:  # Created node IDs
        """Bulk create nodes.

        A node arriving with created_at/updated_at already set keeps them —
        journal REPLAY restores temporal provenance this way (finding 0d50b921:
        stamping now() here clamped every rebuilt node to rebuild time)."""
        ids = []
        now = time.time()
        with self._connect() as con:
            for n in nodes:
                sources_json = json.dumps([s.to_dict() for s in n.sources])
                props_json = json.dumps(n.properties)
                created = n.created_at if n.created_at is not None else now
                updated = n.updated_at if n.updated_at is not None else created
                try:
                    con.execute(
                        "INSERT INTO nodes (id, label, properties, sources, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (n.id, n.label, props_json, sources_json, created, updated)
                    )
                    ids.append(n.id)
                except sqlite3.IntegrityError:
                    self.logger.warning(f"Node ID collision or error: {n.id}")
        return ids

    def add_edges(
        self,
        edges: List[GraphEdge]  # Edges to create
    ) -> List[str]:  # Created edge IDs
        """Bulk create edges.

        Carried created_at/updated_at are honored like add_nodes (0d50b921).

        Missing-endpoint edges (FOREIGN KEY failures) are AGGREGATED into one warning
        per call — count + a bounded sample — never one line per edge: a rebuild replays
        every journaled edge whose endpoint is absent, and per-edge logging flooded
        diagnostics.db to 1.5M rows / 633 MB (finding 0d886ffe). The skipped count is
        recoverable by the caller as `len(edges) - len(ids)`."""
        ids = []
        now = time.time()
        missing: List[Tuple[str, str, str]] = []  # (source_id, target_id, relation_type) of FK-skipped edges
        with self._connect() as con:
            con.execute("PRAGMA foreign_keys = ON;")
            for e in edges:
                props_json = json.dumps(e.properties)
                created = e.created_at if e.created_at is not None else now
                updated = e.updated_at if e.updated_at is not None else created
                try:
                    con.execute(
                        "INSERT INTO edges (id, source_id, target_id, relation_type, properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (e.id, e.source_id, e.target_id, e.relation_type, props_json, created, updated)
                    )
                    ids.append(e.id)
                except sqlite3.IntegrityError as err:
                    if "FOREIGN KEY" in str(err):
                        missing.append((e.source_id, e.target_id, e.relation_type))
                    else:
                        self.logger.warning(f"Edge creation error ({e.id}): {err}")
        if missing:
            sample = "; ".join(f"{s[:8]}->{t[:8]} {r}" for s, t, r in missing[:3])
            self.logger.warning(
                f"add_edges: {len(missing)} of {len(edges)} edge(s) skipped — endpoint node missing "
                f"(FOREIGN KEY); sample: {sample}")
        return ids

    # -------------------------------------------------------------------------
    # READ
    # -------------------------------------------------------------------------

    def get_node(
        self,
        node_id: str  # UUID of node to retrieve
    ) -> Optional[GraphNode]:  # Node or None if not found
        """Get a single node by ID."""
        with self._connect() as con:
            cur = con.execute(
                "SELECT id, label, properties, sources, created_at, updated_at FROM nodes WHERE id = ?",
                (node_id,)
            )
            row = cur.fetchone()
            return self._row_to_node(row) if row else None

    def get_edge(
        self,
        edge_id: str  # UUID of edge to retrieve
    ) -> Optional[GraphEdge]:  # Edge or None if not found
        """Get a single edge by ID."""
        with self._connect() as con:
            cur = con.execute(
                "SELECT id, source_id, target_id, relation_type, properties, created_at, updated_at FROM edges WHERE id = ?",
                (edge_id,)
            )
            row = cur.fetchone()
            return self._row_to_edge(row) if row else None

    def find_nodes_by_source(
        self,
        source_ref: SourceRef  # External resource reference
    ) -> List[GraphNode]:  # Nodes attached to this source
        """Find all nodes linked to a specific external resource."""
        # Content-hash-primary reverse index (CR-19): identity = content_hash
        # (SQL match); the locator is verified Python-side, which keeps the SQL
        # free of per-locator-kind coupling. Slice is NOT compared: the hash is
        # over the consumed (sliced) content, so hash equality already pins it.
        query = """
            SELECT DISTINCT n.id, n.label, n.properties, n.sources, n.created_at, n.updated_at
            FROM nodes n, json_each(n.sources) as src
            WHERE json_extract(src.value, '$.content_hash') = ?
        """
        results = []
        with self._connect() as con:
            cur = con.execute(query, (source_ref.content_hash,))
            for row in cur:
                node = self._row_to_node(row)
                if any(r.content_hash == source_ref.content_hash
                       and r.locator == source_ref.locator for r in node.sources):
                    results.append(node)
        return results

    def find_nodes_by_label(
        self,
        label: str,  # Node label to search for
        limit: int = 100  # Max results
    ) -> List[GraphNode]:  # Matching nodes
        """Find nodes by label."""
        results = []
        with self._connect() as con:
            cur = con.execute(
                "SELECT id, label, properties, sources, created_at, updated_at FROM nodes WHERE label = ? LIMIT ?",
                (label, limit)
            )
            for row in cur:
                results.append(self._row_to_node(row))
        return results

    def get_context(
        self,
        node_id: str,  # Starting node UUID
        depth: int = 1,  # Traversal depth (1 = immediate neighbors)
        filter_labels: Optional[List[str]] = None  # Only include nodes with these labels
    ) -> GraphContext:  # Subgraph containing node and its neighborhood
        """Get the neighborhood of a specific node."""
        # For depth=1, use simple query; for deeper, use recursive CTE
        edge_ids = []
        with self._connect() as con:
            if depth == 1:
                cur = con.execute(
                    "SELECT id FROM edges WHERE source_id = ? OR target_id = ?",
                    (node_id, node_id)
                )
            else:
                # Recursive CTE for multi-hop traversal
                query = """
                WITH RECURSIVE traversal(edge_id, node_id, depth) AS (
                    -- Base case: edges connected to the start node
                    SELECT id, 
                           CASE WHEN source_id = ? THEN target_id ELSE source_id END,
                           1
                    FROM edges
                    WHERE source_id = ? OR target_id = ?

                    UNION

                    -- Recursive step: edges connected to discovered nodes
                    SELECT e.id,
                           CASE WHEN e.source_id = t.node_id THEN e.target_id ELSE e.source_id END,
                           t.depth + 1
                    FROM edges e
                    JOIN traversal t ON (e.source_id = t.node_id OR e.target_id = t.node_id)
                    WHERE t.depth < ?
                )
                SELECT DISTINCT edge_id FROM traversal;
                """
                cur = con.execute(query, (node_id, node_id, node_id, depth))

            edge_ids = [row[0] for row in cur.fetchall()]

        # Fetch full Edge objects
        edges = []
        node_ids_in_context = {node_id}  # Always include the center

        if edge_ids:
            placeholders = ','.join('?' for _ in edge_ids)
            with self._connect() as con:
                cur = con.execute(
                    f"SELECT id, source_id, target_id, relation_type, properties, created_at, updated_at FROM edges WHERE id IN ({placeholders})",
                    tuple(edge_ids)
                )
                for row in cur:
                    e = self._row_to_edge(row)
                    edges.append(e)
                    node_ids_in_context.add(e.source_id)
                    node_ids_in_context.add(e.target_id)

        # Fetch full Node objects
        nodes = []
        if node_ids_in_context:
            placeholders = ','.join('?' for _ in node_ids_in_context)
            with self._connect() as con:
                sql = f"SELECT id, label, properties, sources, created_at, updated_at FROM nodes WHERE id IN ({placeholders})"

                # Apply optional label filtering
                params = list(node_ids_in_context)
                if filter_labels:
                    sql += f" AND label IN ({','.join('?' for _ in filter_labels)})"
                    params.extend(filter_labels)

                cur = con.execute(sql, tuple(params))
                for row in cur:
                    nodes.append(self._row_to_node(row))

        return GraphContext(
            nodes=nodes,
            edges=edges,
            metadata={"depth": depth, "center": node_id}
        )

    # -------------------------------------------------------------------------
    # UPDATE
    # -------------------------------------------------------------------------

    def update_node(
        self,
        node_id: str,  # UUID of node to update
        properties: Dict[str, Any]  # Properties to merge/update
    ) -> bool:  # True if successful
        """Partial update of node properties.

        A reserved `updated_at` key in `properties` sets the COLUMN instead of
        now() and never lands in the JSON blob — journal REPLAY restores a STATE
        op's true time this way (0d50b921 residual: later-STATE nodes clamped
        to rebuild time)."""
        properties = dict(properties)
        stamp = properties.pop("updated_at", None)
        with self._connect() as con:
            # Fetch existing to merge
            cur = con.execute("SELECT properties FROM nodes WHERE id = ?", (node_id,))
            row = cur.fetchone()
            if not row:
                return False

            existing = json.loads(row[0]) if row[0] else {}
            existing.update(properties)  # Merge

            con.execute(
                "UPDATE nodes SET properties = ?, updated_at = ? WHERE id = ?",
                (json.dumps(existing), stamp if stamp is not None else time.time(), node_id)
            )
            return True

    def update_edge(
        self,
        edge_id: str,  # UUID of edge to update
        properties: Dict[str, Any]  # Properties to merge/update
    ) -> bool:  # True if successful
        """Partial update of edge properties."""
        with self._connect() as con:
            cur = con.execute("SELECT properties FROM edges WHERE id = ?", (edge_id,))
            row = cur.fetchone()
            if not row:
                return False

            existing = json.loads(row[0]) if row[0] else {}
            existing.update(properties)

            con.execute(
                "UPDATE edges SET properties = ?, updated_at = ? WHERE id = ?",
                (json.dumps(existing), time.time(), edge_id)
            )
            return True

    # -------------------------------------------------------------------------
    # DELETE
    # -------------------------------------------------------------------------

    def delete_nodes(
        self,
        node_ids: List[str],  # UUIDs of nodes to delete
        cascade: bool = True  # Also delete connected edges
    ) -> int:  # Number of nodes deleted
        """Delete nodes (and optionally connected edges)."""
        with self._connect() as con:
            if cascade:
                con.execute("PRAGMA foreign_keys = ON;")  # Ensures cascade works
            else:
                con.execute("PRAGMA foreign_keys = OFF;")

            placeholders = ','.join('?' for _ in node_ids)
            cur = con.execute(
                f"DELETE FROM nodes WHERE id IN ({placeholders})",
                tuple(node_ids)
            )
            return cur.rowcount

    def delete_edges(
        self,
        edge_ids: List[str]  # UUIDs of edges to delete
    ) -> int:  # Number of edges deleted
        """Delete edges."""
        with self._connect() as con:
            placeholders = ','.join('?' for _ in edge_ids)
            cur = con.execute(
                f"DELETE FROM edges WHERE id IN ({placeholders})",
                tuple(edge_ids)
            )
            return cur.rowcount

    # -------------------------------------------------------------------------
    # LIFECYCLE & INTROSPECTION
    # -------------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Any]:  # Graph schema/ontology
        """Return the current ontology/schema of the graph."""
        schema = {"node_labels": [], "edge_types": [], "counts": {}}
        with self._connect() as con:
            # Labels
            cur = con.execute("SELECT DISTINCT label FROM nodes")
            schema["node_labels"] = [r[0] for r in cur.fetchall()]

            # Types
            cur = con.execute("SELECT DISTINCT relation_type FROM edges")
            schema["edge_types"] = [r[0] for r in cur.fetchall()]

            # Counts
            cur = con.execute("SELECT label, COUNT(*) FROM nodes GROUP BY label")
            for row in cur:
                schema["counts"][row[0]] = row[1]

        return schema

    def import_graph(
        self,
        graph_data: GraphContext,  # Data to import
        merge_strategy: str = "overwrite"  # "overwrite", "skip", or "merge"
    ) -> Dict[str, int]:  # Import statistics {nodes_created, edges_created, merge_strategy}
        """Bulk import a GraphContext honoring merge_strategy (SG-41).

        On id-conflict: "skip" keeps the existing row untouched; "overwrite"
        replaces its mutable fields with the incoming values; "merge" unions
        properties (incoming wins per key) and unions node sources by identity.
        Brand-new ids are always inserted. The `nodes_created`/`edges_created`
        counts report rows written (inserted or updated).
        """
        if merge_strategy not in ("overwrite", "skip", "merge"):
            raise CapabilityInputError(  # SG-47: typed input-validation
                f"Unknown merge_strategy: {merge_strategy}", fields_invalid=["merge_strategy"],
            )
        n = self._import_nodes(graph_data.nodes, merge_strategy)
        e = self._import_edges(graph_data.edges, merge_strategy)
        return {"nodes_created": n, "edges_created": e, "merge_strategy": merge_strategy}

    def _import_nodes(
        self,
        nodes: List[GraphNode],  # Nodes to import
        merge_strategy: str  # "overwrite" | "skip" | "merge"
    ) -> int:  # Count of rows written (inserted or updated)
        """Import nodes honoring merge_strategy (see import_graph)."""
        now = time.time()
        written = 0
        with self._connect() as con:
            for node in nodes:
                props_json = json.dumps(node.properties)
                sources_json = json.dumps([s.to_dict() for s in node.sources])
                row = con.execute("SELECT properties, sources FROM nodes WHERE id = ?", (node.id,)).fetchone()
                if row is None:
                    con.execute(
                        "INSERT INTO nodes (id, label, properties, sources, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (node.id, node.label, props_json, sources_json, now, now),
                    )
                    written += 1
                elif merge_strategy == "skip":
                    continue
                elif merge_strategy == "overwrite":
                    con.execute(
                        "UPDATE nodes SET label = ?, properties = ?, sources = ?, updated_at = ? WHERE id = ?",
                        (node.label, props_json, sources_json, now, node.id),
                    )
                    written += 1
                else:  # merge
                    merged_props = json.loads(row[0]) if row[0] else {}
                    merged_props.update(node.properties)
                    existing_sources = json.loads(row[1]) if row[1] else []
                    seen = {json.dumps(s, sort_keys=True) for s in existing_sources}
                    for s in node.sources:
                        sd = s.to_dict()
                        key = json.dumps(sd, sort_keys=True)
                        if key not in seen:
                            existing_sources.append(sd)
                            seen.add(key)
                    con.execute(
                        "UPDATE nodes SET label = ?, properties = ?, sources = ?, updated_at = ? WHERE id = ?",
                        (node.label, json.dumps(merged_props), json.dumps(existing_sources), now, node.id),
                    )
                    written += 1
        return written

    def _import_edges(
        self,
        edges: List[GraphEdge],  # Edges to import
        merge_strategy: str  # "overwrite" | "skip" | "merge"
    ) -> int:  # Count of rows written (inserted or updated)
        """Import edges honoring merge_strategy (see import_graph)."""
        now = time.time()
        written = 0
        with self._connect() as con:
            for edge in edges:
                props_json = json.dumps(edge.properties)
                row = con.execute("SELECT properties FROM edges WHERE id = ?", (edge.id,)).fetchone()
                if row is None:
                    try:
                        con.execute(
                            "INSERT INTO edges (id, source_id, target_id, relation_type, properties, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (edge.id, edge.source_id, edge.target_id, edge.relation_type, props_json, now, now),
                        )
                        written += 1
                    except sqlite3.IntegrityError as err:
                        self.logger.warning(f"Edge import error (likely missing node): {err}")
                elif merge_strategy == "skip":
                    continue
                elif merge_strategy == "overwrite":
                    con.execute(
                        "UPDATE edges SET source_id = ?, target_id = ?, relation_type = ?, properties = ?, updated_at = ? WHERE id = ?",
                        (edge.source_id, edge.target_id, edge.relation_type, props_json, now, edge.id),
                    )
                    written += 1
                else:  # merge
                    merged = json.loads(row[0]) if row[0] else {}
                    merged.update(edge.properties)
                    con.execute(
                        "UPDATE edges SET properties = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(merged), now, edge.id),
                    )
                    written += 1
        return written

    def export_graph(
        self,
        filter_query: Optional[NodeQuery] = None  # Typed node filter; None = whole graph (stage 4; GraphQuery dissolved)
    ) -> GraphContext:  # Exported subgraph or full graph
        """Export the whole graph, or the subgraph selected by a typed node
        query (matching nodes + the edges among them)."""
        if filter_query is not None:
            # Force full-node mode regardless of the filter's count/project.
            nq = dataclasses.replace(filter_query, count=False, project=None)
            nodes = self.query_nodes(nq).nodes or []
            ids_json = json.dumps([n.id for n in nodes])
            with self._connect() as con:
                cur = con.execute(
                    "SELECT id, source_id, target_id, relation_type, properties, created_at, updated_at "
                    "FROM edges WHERE source_id IN (SELECT value FROM json_each(?)) "
                    "AND target_id IN (SELECT value FROM json_each(?))",
                    (ids_json, ids_json))
                edges = [self._row_to_edge(row) for row in cur]
            return GraphContext(nodes=nodes, edges=edges,
                                metadata={"filtered": True, "node_count": len(nodes)})

        all_nodes = []
        all_edges = []

        with self._connect() as con:
            cur = con.execute("SELECT id, label, properties, sources, created_at, updated_at FROM nodes")
            for row in cur:
                all_nodes.append(self._row_to_node(row))

            cur = con.execute("SELECT id, source_id, target_id, relation_type, properties, created_at, updated_at FROM edges")
            for row in cur:
                all_edges.append(self._row_to_edge(row))

        return GraphContext(nodes=all_nodes, edges=all_edges)

    def query(
        self,
        sql: str,  # A single read-only SELECT (or WITH ... SELECT) statement
        params: Optional[Union[List[Any], Dict[str, Any]]] = None  # Bound parameters (positional list or named dict)
    ) -> Dict[str, Any]:  # {"columns": [...], "rows": [[...]], "row_count": int}
        """Execute a single read-only SELECT and return its rows (SG-41).

        Guards reject empty input, multiple statements, and anything not starting
        with SELECT/WITH. The statement runs on a fresh read-only connection
        (URI `mode=ro`), so even a query that slips past the prefix guard cannot
        mutate the database. Bound `params` use SQLite's qmark placeholders.
        """
        text = (sql or "").strip()
        while text.endswith(";"):
            text = text[:-1].strip()
        if not text:
            raise CapabilityInputError("query requires a non-empty SQL string", fields_invalid=["sql"])
        if ";" in text:
            raise CapabilityInputError("query accepts a single statement only", fields_invalid=["sql"])
        head = text.lstrip("(").split(None, 1)[0].lower()
        if head not in ("select", "with"):
            raise CapabilityInputError(
                "query allows read-only SELECT statements only", fields_invalid=["sql"],
            )
        con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
        try:
            # Positional (list) or named (dict) parameters both bind.
            _bind = params if isinstance(params, dict) else (tuple(params) if params else ())
            cur = con.execute(text, _bind)
            columns = [d[0] for d in cur.description] if cur.description else []
            rows = [list(r) for r in cur.fetchall()]
        finally:
            con.close()
        return {"columns": columns, "rows": rows, "row_count": len(rows)}


def query_nodes(
    self,
    query: NodeQuery,  # Typed node query (the portable, scale-shaped surface)
) -> NodeQueryResult:  # Typed result (nodes / rows / count per query mode)
    """Execute a typed node query (stage 4) — translated to parameterized SQL
    by `query_translation`, run on a fresh read-only connection (SG-41)."""
    sql, params, mode, keys = translate_node_query(query)
    con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
    try:
        cur = con.execute(sql, params)
        if mode == "count":
            return NodeQueryResult(count=cur.fetchone()[0])
        rows = cur.fetchall()
    finally:
        con.close()
    if mode == "full":
        return NodeQueryResult(nodes=[self._row_to_node(r) for r in rows])
    out = []
    for r in rows:
        d = dict(zip(keys, r))
        if "sources" in d and d["sources"]:
            d["sources"] = json.loads(d["sources"])  # ref dicts (CR-19 shape)
        out.append(d)
    return NodeQueryResult(rows=out)


def query_edges(
    self,
    query: EdgeQuery,  # Typed edge query
) -> EdgeQueryResult:  # Typed result (edges / rows / count per query mode)
    """Execute a typed edge query (stage 4) — same contract as `query_nodes`."""
    sql, params, mode, keys = translate_edge_query(query)
    con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
    try:
        cur = con.execute(sql, params)
        if mode == "count":
            return EdgeQueryResult(count=cur.fetchone()[0])
        rows = cur.fetchall()
    finally:
        con.close()
    if mode == "full":
        return EdgeQueryResult(edges=[self._row_to_edge(r) for r in rows])
    return EdgeQueryResult(rows=[dict(zip(keys, r)) for r in rows])


def raw_query(
    self,
    query: RawQuery,  # The marked, backend-coupled raw escape (backend REQUIRED)
) -> RawQueryResult:  # Tabular backend-shaped result
    """Execute the raw escape — refuses backend mismatches (non-portable by
    construction); the read-only single-statement guards are `query`'s (SG-41).
    Recurring raw patterns are promotion candidates into the typed surface."""
    if query.backend != "sqlite":
        raise CapabilityInputError(
            f"RawQuery backend {query.backend!r} refused by sqlite-graph "
            f"(this backend is 'sqlite'; raw queries are non-portable by construction)",
            fields_invalid=["backend"],
        )
    res = self.query(sql=query.text, params=query.params)
    return RawQueryResult(columns=res["columns"], rows=res["rows"],
                          row_count=res["row_count"], backend="sqlite")


def integrity_check(self) -> Dict[str, Any]:  # {"ok": bool, "errors": [...], "backend": "sqlite"}
    """Backend self-check (stage 4; the G3 corruption find institutionalized):
    `PRAGMA quick_check` on a fresh read-only connection so loop-backs and
    stress tests assert storage health cheaply and corruption FAILS LOUDLY."""
    con = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
    try:
        rows = [r[0] for r in con.execute("PRAGMA quick_check")]
    except sqlite3.DatabaseError as e:
        # Severe corruption makes quick_check itself raise — still a STRUCTURED
        # verdict (loop-backs assert ok cheaply; loud, never silent).
        rows = [f"{type(e).__name__}: {e}"]
    finally:
        con.close()
    ok = rows == ["ok"]
    return {"ok": ok, "errors": [] if ok else rows, "backend": "sqlite"}


SQLiteGraphCapability.query_nodes = query_nodes
SQLiteGraphCapability.query_edges = query_edges
SQLiteGraphCapability.raw_query = raw_query
SQLiteGraphCapability.integrity_check = integrity_check
