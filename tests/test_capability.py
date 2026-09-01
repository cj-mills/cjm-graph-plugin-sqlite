"""Tests for cjm_capability_graph_sqlite.capability.

Projected from the capability notebook's demo/test cells at the c25780e8 flip:
the CRUD/round-trip walk, the SG-41 smells (WAL / readonly / guarded query /
merge_strategy), and the stage-4 typed-surface smoke test."""
import sqlite3
import uuid

import pytest

from cjm_capability_graph_sqlite.capability import SQLiteGraphCapability
from cjm_context_graph_primitives.graph import GraphContext, GraphEdge, GraphNode
from cjm_context_graph_primitives.locators import FileRef
from cjm_context_graph_primitives.provenance import SourceRef
from cjm_context_graph_primitives.query import (EdgeQuery, NodeQuery, OrderBy,
                                                PropertyPredicate, RawQuery,
                                                RelationPredicate)
from cjm_context_graph_primitives.slices import FullContent
from cjm_substrate.core.errors import CapabilityInputError


@pytest.fixture
def cap(tmp_path):
    c = SQLiteGraphCapability()
    c.initialize({"db_path": str(tmp_path / "graph.db")})
    yield c
    c.cleanup()


def test_crud_roundtrip_and_provenance(cap, tmp_path):
    assert "db_path" in cap.get_config_schema()["properties"]

    alice_id, bob_id, ml_id = (str(uuid.uuid4()) for _ in range(3))
    transcript_content = b"Alice discussed machine learning with Bob in the podcast."
    content_hash = SourceRef.compute_hash(transcript_content)
    transcript_ref = SourceRef(locator=FileRef(path="/runs/run_demo.json"),
                               content_hash=content_hash, slice=FullContent("text"))

    created = cap.add_nodes([
        GraphNode(id=alice_id, label="Person",
                  properties={"name": "Alice", "role": "speaker"}, sources=[transcript_ref]),
        GraphNode(id=bob_id, label="Person", properties={"name": "Bob"}),
        GraphNode(id=ml_id, label="Concept",
                  properties={"name": "Machine Learning", "definition": "AI subfield"}),
    ])
    assert len(created) == 3
    created = cap.add_edges([
        GraphEdge(id=str(uuid.uuid4()), source_id=alice_id, target_id=ml_id,
                  relation_type="MENTIONS", properties={"confidence": 0.95}),
        GraphEdge(id=str(uuid.uuid4()), source_id=bob_id, target_id=ml_id, relation_type="MENTIONS"),
        GraphEdge(id=str(uuid.uuid4()), source_id=alice_id, target_id=bob_id, relation_type="KNOWS"),
    ])
    assert len(created) == 3

    alice = cap.get_node(alice_id)
    assert alice.label == "Person" and alice.properties["name"] == "Alice"

    context = cap.get_context(alice_id, depth=1)
    names = {n.properties.get("name", n.label) for n in context.nodes}
    assert {"Alice", "Bob", "Machine Learning"} <= names

    # content-hash-primary reverse index (locator must also match)
    found = cap.find_nodes_by_source(transcript_ref)
    assert [n.properties.get("name") for n in found] == ["Alice"]
    other = SourceRef(locator=FileRef(path="/elsewhere.json"), content_hash=content_hash)
    assert cap.find_nodes_by_source(other) == []

    # content hash round-trips through SQLite; verify() still works
    loaded_ref = cap.get_node(alice_id).sources[0]
    assert loaded_ref.content_hash == content_hash
    assert loaded_ref.verify(transcript_content)
    assert not loaded_ref.verify(b"tampered")

    people = cap.find_nodes_by_label("Person")
    assert sorted(p.properties["name"] for p in people) == ["Alice", "Bob"]

    schema = cap.get_schema()
    assert sorted(schema["node_labels"]) == ["Concept", "Person"]
    assert sorted(schema["edge_types"]) == ["KNOWS", "MENTIONS"]
    assert schema["counts"]["Person"] == 2

    cap.update_node(alice_id, {"role": "host", "verified": True})
    assert cap.get_node(alice_id).properties["role"] == "host"

    # export -> temp file -> import into a fresh capability
    exported = cap.export_graph()
    assert len(exported.nodes) == 3 and len(exported.edges) == 3
    temp_path = exported.to_temp_file()
    new_cap = SQLiteGraphCapability()
    new_cap.initialize({"db_path": str(tmp_path / "graph2.db")})
    stats = new_cap.import_graph(GraphContext.from_file(temp_path))
    assert stats["nodes_created"] == 3 if "nodes_created" in stats else stats
    new_cap.cleanup()
    import os
    os.unlink(temp_path)

    # delete with cascade
    assert cap.delete_nodes([alice_id], cascade=True) == 1
    assert cap.get_node(alice_id) is None


def test_sg41_wal_query_guard_merge_readonly(tmp_path):
    db = str(tmp_path / "sg.db")
    p = SQLiteGraphCapability()
    p.initialize({"db_path": db})

    # SG-41(a): WAL enabled on the DB file
    assert sqlite3.connect(db).execute("PRAGMA journal_mode;").fetchone()[0].lower() == "wal"

    sr = {"locator": {"kind": "file", "path": "/runs/sg.json"}, "content_hash": "h1", "slice": None}
    p.add_nodes([GraphNode.from_dict({"id": "n1", "label": "Doc",
                                      "properties": {"x": 1}, "sources": [sr]})])
    assert p.get_node("n1").properties == {"x": 1}

    # SG-41(c): query is a guarded read-only parameterized SELECT
    q = p.query(sql="SELECT id FROM nodes WHERE label = ?", params=["Doc"])
    assert q["row_count"] == 1 and q["columns"] == ["id"] and q["rows"][0][0] == "n1"
    for bad in ('UPDATE nodes SET label = "x"', "SELECT 1; SELECT 2", "   "):
        with pytest.raises(CapabilityInputError):
            p.query(bad)

    # SG-41(d): import_graph merge_strategy="merge" unions properties + sources
    inc = GraphContext.from_dict({
        "nodes": [{"id": "n1", "label": "Doc2", "properties": {"y": 2},
                   "sources": [{"locator": {"kind": "file", "path": "/runs/sg.json"},
                                "content_hash": "h2", "slice": None}]}],
        "edges": []})
    p.import_graph(inc, merge_strategy="merge")
    m = p.get_node("n1")
    assert m.properties == {"x": 1, "y": 2} and len(m.sources) == 2
    p.cleanup()

    # SG-41(b): readonly config honored — reads work, writes rejected
    ro = SQLiteGraphCapability()
    ro.initialize({"db_path": db, "readonly": True})
    assert ro.get_node("n1") is not None
    with pytest.raises(Exception):
        ro.add_nodes([GraphNode.from_dict({"id": "z", "label": "Z",
                                           "properties": {}, "sources": []})])
    ro.cleanup()


def test_typed_surface_smoke(cap):
    doc = str(uuid.uuid4())
    segs = [str(uuid.uuid4()) for _ in range(3)]
    cap.add_nodes([GraphNode(id=doc, label="Document", properties={"title": "T"})] + [
        GraphNode(id=s, label="Segment", properties={"index": i, "text": f"seg {i}"})
        for i, s in enumerate(segs)])
    cap.add_edges([GraphEdge(id=str(uuid.uuid4()), source_id=s, target_id=doc,
                             relation_type="PART_OF") for s in segs] +
                  [GraphEdge(id=str(uuid.uuid4()), source_id=segs[i],
                             target_id=segs[i + 1], relation_type="NEXT")
                   for i in range(2)])

    # count mode
    assert cap.query_nodes(NodeQuery(label="Segment", count=True)).count == 3
    # D13 aggregate: NEXT among the doc's segments
    assert cap.query_edges(EdgeQuery(
        relation_type="NEXT",
        source_related=RelationPredicate("PART_OF", node_id=doc), count=True)).count == 2
    # ordered projection rows carry id
    rows = cap.query_nodes(NodeQuery(
        label="Segment", related=RelationPredicate("PART_OF", node_id=doc),
        order_by=OrderBy(prop="index"), project=["index", "text"])).rows
    assert [r["index"] for r in rows] == [0, 1, 2] and rows[0]["id"] == segs[0]
    # full mode returns typed nodes
    nodes = cap.query_nodes(NodeQuery(ids=[segs[1]])).nodes
    assert len(nodes) == 1 and isinstance(nodes[0], GraphNode)
    # raw escape: backend marking enforced
    res = cap.raw_query(RawQuery(text="SELECT COUNT(*) FROM nodes", backend="sqlite"))
    assert res.rows[0][0] == 4 and res.backend == "sqlite"
    with pytest.raises(CapabilityInputError):
        cap.raw_query(RawQuery(text="SELECT 1", backend="postgres"))
    # integrity check green on a healthy DB
    assert cap.integrity_check() == {"ok": True, "errors": [], "backend": "sqlite"}
    # filtered export: nodes matching + edges among them
    ctx = cap.export_graph(NodeQuery(label="Segment",
                                     related=RelationPredicate("PART_OF", node_id=doc)))
    assert len(ctx.nodes) == 3 and len(ctx.edges) == 2  # NEXT only (PART_OF targets the doc)


def test_update_node_reserved_updated_at_sets_column(cap):
    # 0d50b921 residual: a reserved `updated_at` key in the update payload sets
    # the COLUMN (journal replay restores a STATE op's true time) and never
    # lands in the JSON properties blob; without it, now()-stamping stands.
    import json

    nid = str(uuid.uuid4())
    cap.add_nodes([GraphNode(id=nid, label="Person", properties={"name": "Ada"})])
    assert cap.update_node(nid, {"role": "host", "updated_at": 1234.5})
    con = sqlite3.connect(cap._db_path)
    row = con.execute("SELECT properties, updated_at FROM nodes WHERE id = ?",
                      (nid,)).fetchone()
    con.close()
    props = json.loads(row[0])
    assert props["role"] == "host" and "updated_at" not in props
    assert row[1] == 1234.5
    # Without the reserved key the column re-stamps to now().
    assert cap.update_node(nid, {"role": "guest"})
    con = sqlite3.connect(cap._db_path)
    row2 = con.execute("SELECT updated_at FROM nodes WHERE id = ?", (nid,)).fetchone()
    con.close()
    assert row2[0] > 1234.5


def test_add_edges_aggregates_missing_endpoint_warnings(cap, caplog):
    """Finding 0d886ffe: FK-skipped edges log ONE warning per call (count + sample),
    never one line per edge — per-edge logging flooded diagnostics.db during rebuilds."""
    import logging
    from cjm_context_graph_primitives.graph import GraphEdge, GraphNode
    cap.add_nodes([GraphNode(id="n-live", label="T", properties={}, sources=[])])
    edges = [GraphEdge(id=f"e{i}", source_id="n-live", target_id=f"n-missing-{i}",
                       relation_type="REL", properties={}) for i in range(25)]
    edges.append(GraphEdge(id="e-ok", source_id="n-live", target_id="n-live",
                           relation_type="SELF", properties={}))
    with caplog.at_level(logging.WARNING):
        ids = cap.add_edges(edges)
    assert ids == ["e-ok"]  # skipped count recoverable as len(edges) - len(ids)
    fk_lines = [r for r in caplog.records if "endpoint node missing" in r.getMessage()]
    assert len(fk_lines) == 1
    msg = fk_lines[0].getMessage()
    assert "25 of 26" in msg and "sample:" in msg
    assert not any("Edge creation error (likely missing node)" in r.getMessage() for r in caplog.records)
