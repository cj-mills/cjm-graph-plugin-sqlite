"""Tests for cjm_capability_graph_sqlite.query_translation.

Projected from the query_translation notebook's behavior-test cell at the
c25780e8 flip: the translation contract is verified against the PRODUCTION
schema in an in-memory DB — by what it returns, not by SQL strings."""
import json
import sqlite3

import pytest

from cjm_capability_graph_sqlite.query_translation import (translate_edge_query,
                                                           translate_node_query)
from cjm_context_graph_primitives.locators import FileRef
from cjm_context_graph_primitives.query import (EdgeQuery, NodeQuery, OrderBy,
                                                PropertyPredicate, RelationPredicate,
                                                SourcePredicate)
from cjm_substrate.core.errors import CapabilityInputError


@pytest.fixture(scope="module")
def con():
    con = sqlite3.connect(":memory:")
    con.execute("""CREATE TABLE nodes (
        id TEXT PRIMARY KEY, label TEXT NOT NULL, properties JSON, sources JSON,
        created_at REAL, updated_at REAL)""")
    con.execute("""CREATE TABLE edges (
        id TEXT PRIMARY KEY, source_id TEXT NOT NULL, target_id TEXT NOT NULL,
        relation_type TEXT NOT NULL, properties JSON, created_at REAL, updated_at REAL)""")

    def _node(id, label, props=None, sources=None):
        con.execute("INSERT INTO nodes VALUES (?,?,?,?,1.0,1.0)",
                    (id, label, json.dumps(props or {}), json.dumps(sources or [])))

    def _edge(id, src, tgt, rel, props=None):
        con.execute("INSERT INTO edges VALUES (?,?,?,?,?,1.0,1.0)",
                    (id, src, tgt, rel, json.dumps(props or {})))

    _node("doc-1", "Document", {"title": "Ep"})
    _node("s0", "Segment", {"index": 0, "text": "It's 100% real_deal", "start_time": 0.0})
    _node("s1", "Segment", {"index": 1, "text": "hello world", "start_time": 1.0},
          sources=[{"locator": {"kind": "file", "path": "/m.json"},
                    "content_hash": "sha256:ab", "slice": None}])
    _node("s2", "Segment", {"index": 2, "text": ""})
    _node("s3", "Segment", {"index": 3})  # text key absent -> json_extract NULL
    _node("c1", "Correction", {"correction_type": "text_content",
                               "payload": {"document_id": "doc-1", "segment_id": "s1"}})
    _node("c2", "Correction", {"correction_type": "text_content",
                               "payload": {"segment_id": "s1"}})
    _node("other", "Segment", {"index": 0, "text": "other doc"})  # NOT part of doc-1
    for s in ("s0", "s1", "s2", "s3"):
        _edge(f"po-{s}", s, "doc-1", "PART_OF")
    _edge("sw", "doc-1", "s0", "STARTS_WITH")
    _edge("n0", "s0", "s1", "NEXT")
    _edge("n1", "s1", "s2", "NEXT")
    _edge("nx", "other", "other", "NEXT")  # NEXT edge OUTSIDE doc-1
    _edge("co", "c1", "s1", "CORRECTS")
    _edge("co2", "c2", "s1", "CORRECTS")
    _edge("sup", "c2", "c1", "SUPERSEDES")
    _edge("rev", "sess-1", "s0", "REVIEWED", {"decision": "corrected"})
    yield con
    con.close()


def run(con, q):
    if isinstance(q, NodeQuery):
        sql, params, mode, keys = translate_node_query(q)
    else:
        sql, params, mode, keys = translate_edge_query(q)
    cur = con.execute(sql, params)
    if mode == "count":
        return cur.fetchone()[0]
    rows = cur.fetchall()
    if mode == "rows":
        return [dict(zip(keys, r)) for r in rows]
    return rows


def test_spine_read_ordered_projection(con):
    rows = run(con, NodeQuery(label="Segment",
                              related=RelationPredicate("PART_OF", node_id="doc-1"),
                              order_by=OrderBy(prop="index"),
                              project=["index", "text", "sources"]))
    assert [r["index"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["id"] == "s0"
    assert json.loads(rows[1]["sources"])[0]["content_hash"] == "sha256:ab"


def test_structural_timestamp_projection(con):
    # created_at/updated_at project from the storage COLUMNS (the stamps),
    # not json_extract over properties — no Segment carries them as props.
    rows = run(con, NodeQuery(label="Segment",
                              related=RelationPredicate("PART_OF", node_id="doc-1"),
                              order_by=OrderBy(prop="index"),
                              project=["index", "created_at", "updated_at"]))
    assert [r["created_at"] for r in rows] == [1.0, 1.0, 1.0, 1.0]
    assert rows[0]["updated_at"] == 1.0
    assert rows[0]["id"] == "s0"


def test_empty_filter_or_case_two_counts(con):
    empty_eq = run(con, NodeQuery(label="Segment", where=[PropertyPredicate("text", "eq", "")],
                                  related=RelationPredicate("PART_OF", node_id="doc-1"), count=True))
    empty_null = run(con, NodeQuery(label="Segment", where=[PropertyPredicate("text", "is_null")],
                                    related=RelationPredicate("PART_OF", node_id="doc-1"), count=True))
    assert (empty_eq, empty_null) == (1, 1)


def test_hash_cache_far_end_source(con):
    rows = run(con, NodeQuery(label="Correction",
                              related=RelationPredicate("CORRECTS",
                                                        node_source=SourcePredicate(content_hash="sha256:ab"))))
    assert sorted(r[0] for r in rows) == ["c1", "c2"]


def test_batch_far_end_ids(con):
    rows = run(con, NodeQuery(label="Correction",
                              related=RelationPredicate("CORRECTS", node_ids=["s1", "nope"])))
    assert sorted(r[0] for r in rows) == ["c1", "c2"]


def test_next_chain_count_scoped(con):
    n = run(con, EdgeQuery(relation_type="NEXT",
                           source_related=RelationPredicate("PART_OF", node_id="doc-1"), count=True))
    assert n == 2
    assert run(con, EdgeQuery(relation_type="NEXT", count=True)) == 3  # unscoped sees all


def test_superseded_set_target_ids(con):
    rows = run(con, EdgeQuery(relation_type="SUPERSEDES", target_ids=["c1", "zz"], project=[]))
    assert [r["target_id"] for r in rows] == ["c1"]


def test_dotted_property_path(con):
    rows = run(con, NodeQuery(label="Correction",
                              where=[PropertyPredicate("payload.document_id", "eq", "doc-1")]))
    assert [r[0] for r in rows] == ["c1"]


def test_reviewed_edge_projection_with_property(con):
    rows = run(con, EdgeQuery(relation_type="REVIEWED", source_id="sess-1", project=["decision"]))
    assert rows == [{"id": "rev", "source_id": "sess-1", "target_id": "s0",
                     "decision": "corrected"}]


def test_contains_case_insensitive_and_literal_wildcards(con):
    def hit(needle):
        return [r[0] for r in run(con, NodeQuery(label="Segment",
                                                 where=[PropertyPredicate("text", "contains", needle)]))]
    assert hit("IT'S") == ["s0"]      # case-insensitive, apostrophe-safe (bound param)
    assert hit("100% real") == ["s0"]  # literal %
    assert hit("l_dea") == ["s0"]      # literal _
    assert hit("1%l") == []            # would match via wildcard if % unescaped
    assert hit("Hello W") == ["s1"]


def test_ne_order_desc_limit_and_paging(con):
    rows = run(con, NodeQuery(label="Segment", where=[PropertyPredicate("text", "ne", "")],
                              related=RelationPredicate("PART_OF", node_id="doc-1"),
                              order_by=OrderBy(prop="index", descending=True), limit=1))
    assert rows[0][0] == "s1"  # s3 (null) + s2 ('') excluded; highest index with text
    rows = run(con, NodeQuery(label="Segment", related=RelationPredicate("PART_OF", node_id="doc-1"),
                              order_by=OrderBy(prop="index"), limit=2, offset=1))
    assert [r[0] for r in rows] == ["s1", "s2"]


def test_in_op_ids_batch_and_top_level_source(con):
    rows = run(con, NodeQuery(label="Segment", where=[PropertyPredicate("index", "in", [0, 2])],
                              related=RelationPredicate("PART_OF", node_id="doc-1")))
    assert sorted(r[0] for r in rows) == ["s0", "s2"]
    rows = run(con, NodeQuery(ids=["s1", "c1"]))
    assert sorted(r[0] for r in rows) == ["c1", "s1"]
    rows = run(con, NodeQuery(source=SourcePredicate(content_hash="sha256:ab")))
    assert [r[0] for r in rows] == ["s1"]


def test_direction_in_and_both(con):
    rows = run(con, NodeQuery(related=RelationPredicate("STARTS_WITH", direction="in", node_id="doc-1")))
    assert [r[0] for r in rows] == ["s0"]
    rows = run(con, NodeQuery(related=RelationPredicate("NEXT", direction="both", node_id="s1")))
    assert sorted(r[0] for r in rows) == ["s0", "s2"]


def test_unsupported_and_invalid_raise_loudly(con):
    bads = [
        lambda: translate_node_query(NodeQuery(source=SourcePredicate(
            content_hash="sha256:ab", locator=FileRef("/x")))),
        lambda: translate_node_query(NodeQuery(where=[PropertyPredicate("a", "in", "notalist")])),
        lambda: translate_node_query(NodeQuery(where=[PropertyPredicate("text; DROP", "eq", "x")])),
        lambda: translate_node_query(NodeQuery(order_by=OrderBy(prop="a'b"))),
    ]
    for bad in bads:
        with pytest.raises(CapabilityInputError):
            bad()
