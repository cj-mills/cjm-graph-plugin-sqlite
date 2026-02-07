import sys
import os
import json
import uuid
from pathlib import Path

# Add paths to find local libs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cjm_graph_plugin_system.core import (
    GraphNode, GraphEdge, GraphContext, SourceRef
)
from cjm_graph_plugin_sqlite.plugin import SQLiteGraphPlugin

def title(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def run_lifecycle_test():
    title("TEST 1: SQLite Graph Plugin Lifecycle")

    # 1. Setup Temporary DB Path
    base_dir = Path(__file__).parent
    temp_db_path = base_dir / "test_graph.db"

    # Cleanup previous run
    if temp_db_path.exists():
        os.remove(temp_db_path)

    print(f"[Setup] DB Path: {temp_db_path}")

    # 2. Initialize Plugin
    plugin = SQLiteGraphPlugin()
    plugin.initialize({"db_path": str(temp_db_path)})
    print(f"[Init] Plugin initialized. DB exists? {temp_db_path.exists()}")

    # 3. Create Nodes
    print("\n--- Creating Nodes ---")

    # Simulate content that was consumed from a transcription
    transcript_content = b"Sun Tzu said, The art of war is of vital importance to the state."
    content_hash = SourceRef.compute_hash(transcript_content)

    # Node 1: Transcript Source (Sun Tzu)
    ref_voxtral = SourceRef(
        plugin_name="cjm-transcription-plugin-voxtral-hf",
        table_name="transcriptions",
        row_id="job_123",
        content_hash=content_hash,
        segment_slice="timestamp:00:00-00:10"
    )

    node_a = GraphNode(
        id=str(uuid.uuid4()),
        label="Person",
        properties={"name": "Sun Tzu", "role": "General"},
        sources=[ref_voxtral]
    )

    # Node 2: Concept (The Art of War) - with its own content hash
    concept_content = b"The Art of War"
    concept_hash = SourceRef.compute_hash(concept_content)

    ref_concept = SourceRef(
        plugin_name="cjm-transcription-plugin-voxtral-hf",
        table_name="transcriptions",
        row_id="job_123",
        content_hash=concept_hash,
        segment_slice="char:0-14"
    )

    node_b = GraphNode(
        id=str(uuid.uuid4()),
        label="Concept",
        properties={"title": "The Art of War"},
        sources=[ref_concept]
    )

    created_nodes = plugin.add_nodes([node_a, node_b])
    print(f"Created Nodes: {len(created_nodes)}")
    assert len(created_nodes) == 2

    # 4. Create Edge
    print("\n--- Creating Edge ---")
    edge = GraphEdge(
        id=str(uuid.uuid4()),
        source_id=node_a.id,
        target_id=node_b.id,
        relation_type="AUTHORED",
        properties={"year": -500}
    )

    created_edges = plugin.add_edges([edge])
    print(f"Created Edges: {len(created_edges)}")
    assert len(created_edges) == 1

    # 5. Test Traversal (get_context)
    print("\n--- Testing Traversal (get_context) ---")
    ctx = plugin.get_context(node_a.id, depth=1)

    print(f"Context Nodes found: {len(ctx.nodes)}")
    print(f"Context Edges found: {len(ctx.edges)}")

    # Verify we got the neighbor
    neighbor = next((n for n in ctx.nodes if n.id == node_b.id), None)
    assert neighbor is not None, "Failed to find connected neighbor"
    print(f"  -> Found neighbor: {neighbor.properties['title']}")

    # Verify we got the edge
    found_edge = next((e for e in ctx.edges if e.id == edge.id), None)
    assert found_edge is not None, "Failed to find connecting edge"
    print(f"  -> Found edge type: {found_edge.relation_type}")

    # 6. Test Source Lookup (Federation)
    print("\n--- Testing Source Lookup ---")
    search_ref = SourceRef(
        plugin_name="cjm-transcription-plugin-voxtral-hf",
        table_name="transcriptions",
        row_id="job_123",
        content_hash=content_hash
    )
    results = plugin.find_nodes_by_source(search_ref)

    print(f"Found {len(results)} nodes linked to source.")
    assert len(results) >= 1
    found_ids = {r.id for r in results}
    assert node_a.id in found_ids
    print(f"  -> Correctly identified node: {results[0].properties.get('name', results[0].properties.get('title'))}")

    # 7. Test Schema Introspection
    print("\n--- Testing Schema Introspection ---")
    schema = plugin.get_schema()
    print(f"Labels: {schema['node_labels']}")
    print(f"Edge Types: {schema['edge_types']}")

    assert "Person" in schema['node_labels']
    assert "AUTHORED" in schema['edge_types']

    # 8. Test Export
    print("\n--- Testing Export ---")
    export_ctx = plugin.export_graph()
    print(f"Exported {len(export_ctx.nodes)} nodes and {len(export_ctx.edges)} edges.")
    assert len(export_ctx.nodes) == 2

    # 9. Test Delete (Cascade)
    print("\n--- Testing Cascade Delete ---")
    deleted_count = plugin.delete_nodes([node_a.id], cascade=True)
    print(f"Deleted Nodes: {deleted_count}")

    # Verify Edge is gone
    orphan_edge = plugin.get_edge(edge.id)
    assert orphan_edge is None, "Cascade failed: Edge still exists after Node delete"
    print("  -> Edge successfully deleted by cascade.")

    # Cleanup
    if temp_db_path.exists():
        os.remove(temp_db_path)
    print("\n[SUCCESS] All lifecycle tests passed.")


def run_content_hash_test():
    title("TEST 2: Content Hash Round-Trip Through SQLite")

    # 1. Setup
    base_dir = Path(__file__).parent
    temp_db_path = base_dir / "test_graph_hash.db"
    if temp_db_path.exists():
        os.remove(temp_db_path)

    plugin = SQLiteGraphPlugin()
    plugin.initialize({"db_path": str(temp_db_path)})

    # 2. Create nodes with content hashes
    print("\n--- Creating Nodes with Content Hashes ---")

    content_a = b"Laying Plans Sun Tzu said, The art of war is of vital importance to the state."
    hash_a = SourceRef.compute_hash(content_a)

    content_b = b"All warfare is based on deception."
    hash_b = SourceRef.compute_hash(content_b)

    ref_a = SourceRef(
        plugin_name="cjm-transcription-plugin-voxtral-hf",
        table_name="transcriptions",
        row_id="job_abc",
        content_hash=hash_a,
        segment_slice="full_text"
    )

    ref_b = SourceRef(
        plugin_name="cjm-transcription-plugin-voxtral-hf",
        table_name="transcriptions",
        row_id="job_abc",
        content_hash=hash_b,
        segment_slice="char:200-233"
    )

    node_1 = GraphNode(
        id=str(uuid.uuid4()),
        label="Document",
        properties={"title": "Laying Plans"},
        sources=[ref_a]
    )

    node_2 = GraphNode(
        id=str(uuid.uuid4()),
        label="Quote",
        properties={"text": "All warfare is based on deception."},
        sources=[ref_b]
    )

    edge_1 = GraphEdge(
        id=str(uuid.uuid4()),
        source_id=node_1.id,
        target_id=node_2.id,
        relation_type="CONTAINS",
        properties={}
    )

    plugin.add_nodes([node_1, node_2])
    plugin.add_edges([edge_1])

    # 3. Retrieve and verify hashes survived SQLite round-trip
    print("\n--- Verifying Hash Round-Trip (get_node) ---")
    loaded_1 = plugin.get_node(node_1.id)
    loaded_2 = plugin.get_node(node_2.id)

    assert loaded_1 is not None
    assert loaded_2 is not None
    assert len(loaded_1.sources) == 1
    assert len(loaded_2.sources) == 1

    assert loaded_1.sources[0].content_hash == hash_a
    assert loaded_2.sources[0].content_hash == hash_b
    assert loaded_1.sources[0].segment_slice == "full_text"
    assert loaded_2.sources[0].segment_slice == "char:200-233"
    print(f"  -> Node 1 hash: {loaded_1.sources[0].content_hash[:30]}...")
    print(f"  -> Node 2 hash: {loaded_2.sources[0].content_hash[:30]}...")
    print("  -> PASSED: Hashes survived SQLite storage.")

    # 4. Verify using SourceRef.verify()
    print("\n--- Verifying SourceRef.verify() After Round-Trip ---")
    assert loaded_1.sources[0].verify(content_a), "verify() should return True for original content"
    assert loaded_2.sources[0].verify(content_b), "verify() should return True for original content"
    assert not loaded_1.sources[0].verify(b"tampered content"), "verify() should return False for tampered content"
    print("  -> PASSED: verify() works correctly after SQLite round-trip.")

    # 5. Verify hashes survive get_context traversal
    print("\n--- Verifying Hash Through get_context ---")
    ctx = plugin.get_context(node_1.id, depth=1)

    ctx_node_1 = next(n for n in ctx.nodes if n.id == node_1.id)
    ctx_node_2 = next(n for n in ctx.nodes if n.id == node_2.id)
    assert ctx_node_1.sources[0].content_hash == hash_a
    assert ctx_node_2.sources[0].content_hash == hash_b
    print("  -> PASSED: Hashes preserved through get_context traversal.")

    # 6. Verify hashes survive export -> import cycle
    print("\n--- Verifying Hash Through Export/Import Cycle ---")
    exported = plugin.export_graph()

    # Verify in the exported GraphContext
    exp_node_1 = next(n for n in exported.nodes if n.id == node_1.id)
    assert exp_node_1.sources[0].content_hash == hash_a

    # Import into a fresh DB
    temp_db_path_2 = base_dir / "test_graph_hash_import.db"
    if temp_db_path_2.exists():
        os.remove(temp_db_path_2)

    plugin_2 = SQLiteGraphPlugin()
    plugin_2.initialize({"db_path": str(temp_db_path_2)})
    stats = plugin_2.import_graph(exported)
    print(f"  Import stats: {stats}")

    # Verify in the imported DB
    imported_node = plugin_2.get_node(node_1.id)
    assert imported_node is not None
    assert imported_node.sources[0].content_hash == hash_a
    assert imported_node.sources[0].verify(content_a)
    print("  -> PASSED: Hashes preserved through export/import cycle.")

    # 7. Verify hashes survive to_dict/from_dict cycle (GraphContext)
    print("\n--- Verifying Hash Through to_dict/from_dict ---")
    ctx_dict = exported.to_dict()
    reloaded = GraphContext.from_dict(ctx_dict)
    reloaded_node = next(n for n in reloaded.nodes if n.id == node_1.id)
    assert reloaded_node.sources[0].content_hash == hash_a
    assert reloaded_node.sources[0].verify(content_a)
    print("  -> PASSED: Hashes preserved through to_dict/from_dict cycle.")

    # 8. Verify hashes survive to_temp_file/from_file cycle
    print("\n--- Verifying Hash Through File Round-Trip ---")
    temp_file = exported.to_temp_file()
    from_file = GraphContext.from_file(temp_file)
    file_node = next(n for n in from_file.nodes if n.id == node_2.id)
    assert file_node.sources[0].content_hash == hash_b
    assert file_node.sources[0].verify(content_b)
    os.remove(temp_file)
    print("  -> PASSED: Hashes preserved through file round-trip.")

    # 9. Verify execute() dispatcher preserves hashes
    print("\n--- Verifying Hash Through execute() Dispatcher ---")
    result = plugin.execute(action="get_node", node_id=node_1.id)
    assert result["node"]["sources"][0]["content_hash"] == hash_a
    print("  -> PASSED: Hashes preserved through execute() dispatcher.")

    # Cleanup
    plugin.cleanup()
    plugin_2.cleanup()
    if temp_db_path.exists():
        os.remove(temp_db_path)
    if temp_db_path_2.exists():
        os.remove(temp_db_path_2)
    print("\n[SUCCESS] All content hash tests passed.")


if __name__ == "__main__":
    try:
        run_lifecycle_test()
        run_content_hash_test()
        title("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"\n!!! ASSERTION FAILED !!!\n{e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n!!! ERROR !!!\n{e}")
        import traceback
        traceback.print_exc()
