"""The per-backend translation of the typed query expressions (pass-2 Thread 5; stage 4): NodeQuery/EdgeQuery -> parameterized SQLite SQL over the nodes/edges schema. THIS module is what makes the typed surface portable - the expression is domain- and backend-neutral; every backend tool owns a translation like this one (the ratified stage-4 split: backend owns translation; the adapter stays generic). Pure functions, unit-tested against an in-memory DB with the production schema - no capability runtime needed.

Translation rules:

- **Predicates** compile to `json_extract(properties, $.<path>)` comparisons
  with bound parameters (values NEVER interpolated). Property paths are dotted
  (`payload.document_id`) and validated against `[A-Za-z0-9_]` segments —
  paths are interpolated into the JSON-path literal, so validation is the
  injection guard for the path position.
- **`contains` is case-INSENSITIVE by definition** (P12): `lower(col) LIKE ...
  ESCAPE` with `%`/`_`/`\\` escaped in the value. Both sides use SQLite
  `lower()` (ASCII-range folding) — consistent, but non-ASCII case folding is
  NOT performed (recorded; a unicode-folding consumer would be the promotion
  evidence).
- **Id lists** bind as ONE JSON-array parameter via `json_each(?)` — no SQL
  parameter-count limits at corpus scale (13k+ ids).
- **Relation / endpoint constraints** compile to correlated `EXISTS`
  subqueries (never JOINs) — no row multiplication, so `count` needs no
  DISTINCT and rows need no GROUP BY.
- **`SourcePredicate`** translates content-hash-primary (`json_each(sources)`);
  a locator constraint raises (unsupported — use `find_nodes_by_source`, which
  does Python-side locator equality; a recurring need here is promotion
  evidence).
- **Modes**: `count` > `project` > full (mirrors the result DTO contract).
  Projected node rows always carry `id` (+ `label`/`created_at`/`updated_at`
  projectable structurally — the storage stamps, never shadowed by a
  same-named property; `sources` projectable as parsed ref dicts); edge rows
  always carry `id`/`source_id`/`target_id` (+ `relation_type` projectable)."""

import json
import re
from typing import Any, List, Optional, Tuple

from cjm_context_graph_primitives.query import (EdgeQuery, NodeQuery, PropertyPredicate,
                                                RelationPredicate, SourcePredicate)
from cjm_substrate.core.errors import CapabilityInputError

_PROP_PATH_RE = re.compile(r"^[A-Za-z0-9_]+(\.[A-Za-z0-9_]+)*$")  # Dotted-path guard (paths are interpolated; values are bound)


def _json_path(
    prop: str,  # Property name or dotted path (e.g. "payload.document_id")
) -> str:  # SQL string literal for json_extract's path argument
    """Validate a dotted property path and render the JSON-path literal."""
    if not _PROP_PATH_RE.match(prop or ""):
        raise CapabilityInputError(
            f"Invalid property path {prop!r} (segments must be [A-Za-z0-9_])",
            fields_invalid=["prop"],
        )
    return "'$." + prop + "'"


def _escape_like(
    value: str,  # Raw substring the caller wants matched literally
) -> str:  # Value with LIKE wildcards escaped (ESCAPE '\\')
    """Escape `%` / `_` / `\\` so `contains` matches literally, never as wildcards."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _predicate_sql(
    pred: PropertyPredicate,  # One typed property predicate
    props_col: str,  # SQL expression for the properties JSON column (e.g. "n.properties")
) -> Tuple[str, List[Any]]:  # (SQL fragment, bound params)
    """Compile one property predicate to a parameterized SQL fragment."""
    col = f"json_extract({props_col}, {_json_path(pred.prop)})"
    op = pred.op
    if op == "eq":
        return f"{col} = ?", [pred.value]
    if op == "ne":
        return f"{col} != ?", [pred.value]
    if op == "lt":
        return f"{col} < ?", [pred.value]
    if op == "le":
        return f"{col} <= ?", [pred.value]
    if op == "gt":
        return f"{col} > ?", [pred.value]
    if op == "ge":
        return f"{col} >= ?", [pred.value]
    if op == "in":
        if not isinstance(pred.value, (list, tuple)):
            raise CapabilityInputError(
                f"Op 'in' requires a list value (got {type(pred.value).__name__})",
                fields_invalid=["value"],
            )
        return (f"{col} IN (SELECT value FROM json_each(?))",
                [json.dumps(list(pred.value))])
    if op == "contains":
        if not isinstance(pred.value, str):
            raise CapabilityInputError(
                f"Op 'contains' requires a string value (got {type(pred.value).__name__})",
                fields_invalid=["value"],
            )
        # Defined case-INSENSITIVE (P12). Both sides through SQLite's lower()
        # (ASCII folding); wildcards escaped so the match is literal.
        return (f"lower({col}) LIKE '%' || lower(?) || '%' ESCAPE '\\'",
                [_escape_like(pred.value)])
    if op == "is_null":
        return f"{col} IS NULL", []
    if op == "not_null":
        return f"{col} IS NOT NULL", []
    raise CapabilityInputError(
        f"Predicate op {pred.op!r} not supported by the sqlite translation",
        fields_invalid=["op"],
    )


def _source_match_sql(
    sp: SourcePredicate,  # Provenance match (content-hash-primary per CR-19)
    sources_expr: str,  # SQL expression for the sources JSON column (e.g. "n.sources")
) -> Tuple[str, List[Any]]:  # (EXISTS fragment, bound params)
    """Compile a source predicate to an EXISTS over the sources array.

    Content-hash-primary (identity field = the stable query surface, C19).
    A locator constraint RAISES — unsupported in the typed translation
    (use `find_nodes_by_source` for locator equality; recurring need here
    is the promotion evidence the raw-escape posture wants recorded).
    """
    if sp.locator is not None:
        raise CapabilityInputError(
            "SourcePredicate.locator is not supported by the sqlite typed "
            "translation (content-hash-primary); use find_nodes_by_source",
            fields_invalid=["locator"],
        )
    return (f"EXISTS (SELECT 1 FROM json_each({sources_expr}) AS src "
            f"WHERE json_extract(src.value, '$.content_hash') = ?)",
            [sp.content_hash])


def _relation_exists_sql(
    rel: RelationPredicate,  # One-hop relation constraint (+ far-end constraints)
    node_expr: str,  # SQL expression for the candidate node id (e.g. "n.id", "e.source_id")
) -> Tuple[str, List[Any]]:  # (EXISTS fragment, bound params)
    """Compile a relation predicate to a correlated EXISTS (no row multiplication).

    Far-end constraints (stage-4 promotions): `node_id` / `node_ids` pin the
    far node; `node_source` nests a provenance EXISTS on the far node.
    Subquery scoping keeps the fixed aliases (r / fn / src) collision-free.
    """
    params: List[Any] = [rel.relation_type]
    if rel.direction == "out":
        join_cond = f"r.source_id = {node_expr}"
        far = "r.target_id"
    elif rel.direction == "in":
        join_cond = f"r.target_id = {node_expr}"
        far = "r.source_id"
    else:  # both
        join_cond = f"(r.source_id = {node_expr} OR r.target_id = {node_expr})"
        far = (f"(CASE WHEN r.source_id = {node_expr} "
               f"THEN r.target_id ELSE r.source_id END)")
    conds = [join_cond, "r.relation_type = ?"]
    if rel.node_id is not None:
        conds.append(f"{far} = ?")
        params.append(rel.node_id)
    if rel.node_ids is not None:
        conds.append(f"{far} IN (SELECT value FROM json_each(?))")
        params.append(json.dumps(list(rel.node_ids)))
    if rel.node_source is not None:
        src_frag, src_params = _source_match_sql(rel.node_source, "fn.sources")
        conds.append(
            f"EXISTS (SELECT 1 FROM nodes AS fn WHERE fn.id = {far} AND {src_frag})")
        params.extend(src_params)
    return f"EXISTS (SELECT 1 FROM edges AS r WHERE {' AND '.join(conds)})", params


NODE_FULL_COLUMNS = "n.id, n.label, n.properties, n.sources, n.created_at, n.updated_at"  # Matches _row_to_node order
EDGE_FULL_COLUMNS = "e.id, e.source_id, e.target_id, e.relation_type, e.properties, e.created_at, e.updated_at"  # Matches _row_to_edge order


def _order_limit_sql(
    query,  # NodeQuery or EdgeQuery (order_by / limit / offset fields)
    props_col: str,  # Properties column expression for ORDER BY paths
    params: List[Any],  # Bound-params list (appended in place)
) -> str:  # ORDER BY / LIMIT / OFFSET tail
    """Compile the shared ordering + paging tail."""
    tail = ""
    if query.order_by is not None:
        tail += (f" ORDER BY json_extract({props_col}, "
                 f"{_json_path(query.order_by.prop)})"
                 + (" DESC" if query.order_by.descending else " ASC"))
    if query.limit is not None or query.offset:
        tail += " LIMIT ?"
        params.append(query.limit if query.limit is not None else -1)
        if query.offset:
            tail += " OFFSET ?"
            params.append(query.offset)
    return tail


def translate_node_query(
    q: NodeQuery,  # Typed node query
) -> Tuple[str, List[Any], str, Optional[List[str]]]:  # (sql, params, mode, row keys)
    """Translate a `NodeQuery` to parameterized SQLite SQL.

    mode: "count" | "rows" | "full" (count > project > full, mirroring the
    result DTO's exactly-one-populated contract). For "rows", the returned
    keys list zips against each cursor row ("id" always first; "label"
    projects structurally; "sources" projects as the raw JSON column for the
    caller to parse).
    """
    where: List[str] = []
    params: List[Any] = []
    if q.ids is not None:
        where.append("n.id IN (SELECT value FROM json_each(?))")
        params.append(json.dumps(list(q.ids)))
    if q.label is not None:
        where.append("n.label = ?")
        params.append(q.label)
    for pred in q.where:
        frag, ps = _predicate_sql(pred, "n.properties")
        where.append(frag)
        params.extend(ps)
    if q.source is not None:
        frag, ps = _source_match_sql(q.source, "n.sources")
        where.append(frag)
        params.extend(ps)
    if q.related is not None:
        frag, ps = _relation_exists_sql(q.related, "n.id")
        where.append(frag)
        params.extend(ps)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    if q.count:
        return (f"SELECT COUNT(*) FROM nodes AS n{where_sql}", params, "count", None)

    if q.project is not None:
        keys = ["id"]
        select_parts = ["n.id"]
        for name in q.project:
            if name == "id":
                continue
            if name == "label":
                select_parts.append("n.label")
                keys.append("label")
            elif name in ("created_at", "updated_at"):
                select_parts.append("n." + name)   # storage stamp, not a property
                keys.append(name)
            elif name == "sources":
                select_parts.append("n.sources")
                keys.append("sources")
            else:
                select_parts.append(f"json_extract(n.properties, {_json_path(name)})")
                keys.append(name)
        tail = _order_limit_sql(q, "n.properties", params)
        return (f"SELECT {', '.join(select_parts)} FROM nodes AS n{where_sql}{tail}",
                params, "rows", keys)

    tail = _order_limit_sql(q, "n.properties", params)
    return (f"SELECT {NODE_FULL_COLUMNS} FROM nodes AS n{where_sql}{tail}",
            params, "full", None)


def translate_edge_query(
    q: EdgeQuery,  # Typed edge query
) -> Tuple[str, List[Any], str, Optional[List[str]]]:  # (sql, params, mode, row keys)
    """Translate an `EdgeQuery` to parameterized SQLite SQL.

    Same mode contract as `translate_node_query`. Projected rows always carry
    `id`/`source_id`/`target_id`; `relation_type` projects structurally.
    Endpoint constraints (`source_related`/`target_related` — the D13
    NEXT-chain count) compile to correlated EXISTS on the endpoint node.
    """
    where: List[str] = []
    params: List[Any] = []
    if q.ids is not None:
        where.append("e.id IN (SELECT value FROM json_each(?))")
        params.append(json.dumps(list(q.ids)))
    if q.relation_type is not None:
        where.append("e.relation_type = ?")
        params.append(q.relation_type)
    if q.source_id is not None:
        where.append("e.source_id = ?")
        params.append(q.source_id)
    if q.target_id is not None:
        where.append("e.target_id = ?")
        params.append(q.target_id)
    if q.source_ids is not None:
        where.append("e.source_id IN (SELECT value FROM json_each(?))")
        params.append(json.dumps(list(q.source_ids)))
    if q.target_ids is not None:
        where.append("e.target_id IN (SELECT value FROM json_each(?))")
        params.append(json.dumps(list(q.target_ids)))
    if q.source_related is not None:
        frag, ps = _relation_exists_sql(q.source_related, "e.source_id")
        where.append(frag)
        params.extend(ps)
    if q.target_related is not None:
        frag, ps = _relation_exists_sql(q.target_related, "e.target_id")
        where.append(frag)
        params.extend(ps)
    for pred in q.where:
        frag, ps = _predicate_sql(pred, "e.properties")
        where.append(frag)
        params.extend(ps)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    if q.count:
        return (f"SELECT COUNT(*) FROM edges AS e{where_sql}", params, "count", None)

    if q.project is not None:
        keys = ["id", "source_id", "target_id"]
        select_parts = ["e.id", "e.source_id", "e.target_id"]
        for name in q.project:
            if name in ("id", "source_id", "target_id"):
                continue
            if name == "relation_type":
                select_parts.append("e.relation_type")
                keys.append("relation_type")
            else:
                select_parts.append(f"json_extract(e.properties, {_json_path(name)})")
                keys.append(name)
        tail = _order_limit_sql(q, "e.properties", params)
        return (f"SELECT {', '.join(select_parts)} FROM edges AS e{where_sql}{tail}",
                params, "rows", keys)

    tail = _order_limit_sql(q, "e.properties", params)
    return (f"SELECT {EDGE_FULL_COLUMNS} FROM edges AS e{where_sql}{tail}",
            params, "full", None)
