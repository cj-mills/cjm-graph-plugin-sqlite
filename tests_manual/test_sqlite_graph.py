import sys
import os
import json
import uuid
import shutil
from pathlib import Path

# Add paths to find local libs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cjm_graph_plugin_system.core import (
    GraphNode, GraphEdge, SourceRef
)
from cjm_graph_plugin_sqlite.plugin import SQLiteGraphPlugin

def title(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def run_lifecycle_test():
    title("TEST: SQLite Graph Plugin Lifecycle")

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
    
    # Node 1: Transcript Source (Sun Tzu)
    ref_voxtral = SourceRef(
        plugin_name="cjm-transcription-plugin-voxtral-hf",
        table_name="transcriptions",
        row_id="job_123",
        segment_slice="00:00-00:10"
    )
    
    node_a = GraphNode(
        id=str(uuid.uuid4()),
        label="Person",
        properties={"name": "Sun Tzu", "role": "General"},
        sources=[ref_voxtral]
    )
    
    # Node 2: Concept (The Art of War)
    node_b = GraphNode(
        id=str(uuid.uuid4()),
        label="Concept",
        properties={"title": "The Art of War"},
        sources=[]
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
    # Ask for context around Node A (Sun Tzu). Should find the edge and Node B.
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
    # Search for nodes linked to Voxtral job_123
    search_ref = SourceRef("cjm-transcription-plugin-voxtral-hf", "transcriptions", "job_123")
    results = plugin.find_nodes_by_source(search_ref)
    
    print(f"Found {len(results)} nodes linked to source.")
    assert len(results) == 1
    assert results[0].id == node_a.id
    print(f"  -> Correctly identified node: {results[0].properties['name']}")

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
    # Delete Node A (Sun Tzu). Should also delete the Edge to Node B.
    deleted_count = plugin.delete_nodes([node_a.id], cascade=True)
    print(f"Deleted Nodes: {deleted_count}")
    
    # Verify Edge is gone
    orphan_edge = plugin.get_edge(edge.id)
    assert orphan_edge is None, "Cascade failed: Edge still exists after Node delete"
    print("  -> Edge successfully deleted by cascade.")

    # Cleanup
    if temp_db_path.exists():
        os.remove(temp_db_path)
    print("\n[SUCCESS] All SQLite Graph tests passed.")

if __name__ == "__main__":
    try:
        run_lifecycle_test()
    except Exception as e:
        print(f"\n!!! FAILED !!!\n{e}")
        import traceback
        traceback.print_exc()