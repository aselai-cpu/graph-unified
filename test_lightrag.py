"""Test script for LightRAG dual-level retrieval strategy."""

import asyncio
from pathlib import Path

from graphunified.config.settings import Settings
from graphunified.storage.graph_store import GraphStore
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore
from graphunified.strategies.lightrag import LightRAGStrategy
from graphunified.utils.embedding_factory import create_embedding_client


async def main():
    """Test LightRAG dual-level retrieval."""

    # Load configuration
    config_path = Path("settings-local-embeddings.yaml")
    settings = Settings.load(config_path)

    print("=" * 80)
    print("Testing LightRAG Dual-Level Retrieval Strategy")
    print("=" * 80)

    # Initialize components
    print("\n1. Loading vector store...")
    vector_store_path = Path("test-output-stage5/lancedb")
    vector_store = VectorStore.from_config(
        settings.storage.vector_db,
        vector_store_path,
        settings.embedding.dimension,
    )

    print("2. Loading Parquet store...")
    parquet_store_path = Path("test-output-stage5")
    parquet_store = ParquetStore(parquet_store_path)

    print("3. Loading knowledge graph...")
    graph_store_path = parquet_store_path / "graph"
    graph_store = GraphStore(
        root_dir=graph_store_path,
        directed=True,
        graph_format="pickle",
    )

    # Load existing graph
    if (graph_store_path / "graph.pickle").exists():
        await graph_store.load()
        print(
            f"   Graph loaded: {graph_store.graph.number_of_nodes()} nodes, "
            f"{graph_store.graph.number_of_edges()} edges"
        )
    else:
        print("   ERROR: Graph not found! Run test_graphrag_global.py first.")
        return

    print("4. Creating embedding client...")
    embedding_client = create_embedding_client(settings.embedding)

    print("5. Initializing LightRAG strategy...")
    strategy = LightRAGStrategy(
        config=settings.strategies.graphrag,
        vector_store=vector_store,
        graph_store=graph_store,
        parquet_store=parquet_store,
        embedding_client=embedding_client,
    )

    # Load data
    print("6. Loading entities, relationships, chunks, and communities...")
    entities = [e async for e in parquet_store.load_entities()]
    relationships = [r async for r in parquet_store.load_relationships()]
    chunks = [c async for c in parquet_store.load_chunks()]

    # Load communities from GraphRAG Global test
    # For this test, we'll detect communities directly
    print("   Detecting communities...")
    node_to_community = await graph_store.detect_communities_louvain(resolution=1.0)

    # Convert to Community objects (simplified version)
    from collections import defaultdict
    from uuid import UUID

    from graphunified.config.models import Community

    community_to_nodes = defaultdict(list)
    for node_id, comm_id in node_to_community.items():
        community_to_nodes[comm_id].append(node_id)

    communities = []
    for comm_id, entity_id_strs in community_to_nodes.items():
        community = Community(
            level=0,
            entity_ids=[UUID(eid) for eid in entity_id_strs],
            size=len(entity_id_strs),
            title=f"Community {comm_id}",
            summary=f"Community {comm_id} with {len(entity_id_strs)} entities",
            metadata={"community_number": comm_id},
        )
        communities.append(community)

    print(
        f"   Loaded {len(entities)} entities, {len(relationships)} relationships, "
        f"{len(chunks)} chunks, {len(communities)} communities"
    )

    # Index
    print("\n7. Building LightRAG indexes...")
    await strategy.index(chunks, entities, relationships, communities)

    # Validate
    print("8. Validating indexes...")
    is_valid = await strategy.validate_index()
    print(f"   Indexes valid: {is_valid}")

    if not is_valid:
        print("   ERROR: Index validation failed!")
        return

    # Test queries with different modes
    test_cases = [
        # Local queries (entity-focused)
        ("What is GraphRAG?", "local"),
        ("Define machine learning", "local"),
        ("How does entity extraction work?", "local"),
        # Global queries (thematic)
        ("What are the main themes in this corpus?", "global"),
        ("Summarize the key topics discussed", "global"),
        ("Give me an overview of the content", "global"),
        # Hybrid queries (mixed)
        ("How does GraphRAG relate to broader AI trends?", "hybrid"),
        ("What specific technologies support knowledge graphs?", "hybrid"),
        # Auto-classification (no mode specified)
        ("Tell me about retrieval systems", None),
    ]

    print("\n\n9. Testing retrieval with different modes:")
    print("-" * 80)

    for i, (query, expected_mode) in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        if expected_mode:
            print(f"Expected mode: {expected_mode}")
        else:
            print("Mode: auto-classify")
        print("=" * 80)

        try:
            # Retrieve with specified or auto mode
            result = await strategy.retrieve(query, top_k=5, mode=expected_mode)

            print(f"\nStrategy: {result.strategy}")
            print(f"Query mode: {result.metadata.get('query_mode')}")
            print(f"Retrieved: {len(result.chunks)} chunks")
            print(f"Entities: {len(result.entities)} entities")
            print(f"Relationships: {len(result.relationships)} relationships")
            print(f"Communities: {len(result.communities)} communities")
            print(f"Time: {result.retrieval_time_ms:.2f}ms")

            if result.mean_score > 0:
                print(f"Mean score: {result.mean_score:.3f}")

            # Show mode-specific metadata
            mode = result.metadata.get("query_mode")
            if mode == "local":
                print(
                    f"\nLocal mode metrics:"
                    f"\n  Seed entities: {result.metadata.get('seed_entities', 0)}"
                    f"\n  Expanded entities: {result.metadata.get('expanded_entities', 0)}"
                    f"\n  Max hops: {result.metadata.get('hops', 0)}"
                )
            elif mode == "global":
                print(
                    f"\nGlobal mode metrics:"
                    f"\n  Communities searched: {result.metadata.get('communities_searched', 0)}"
                    f"\n  Communities retrieved: {result.metadata.get('communities_retrieved', 0)}"
                )
            elif mode == "hybrid":
                print(
                    f"\nHybrid mode metrics:"
                    f"\n  Local weight: {result.metadata.get('local_weight', 0):.2f}"
                    f"\n  Local chunks: {result.metadata.get('local_chunks', 0)}"
                    f"\n  Global chunks: {result.metadata.get('global_chunks', 0)}"
                )

            # Show top entities
            if result.entities:
                print(f"\nTop entities:")
                for j, entity in enumerate(result.entities[:3], 1):
                    print(f"  [{j}] {entity.name} ({entity.type.value})")

            # Show top chunks
            if result.chunks:
                print(f"\nTop chunks:")
                for j, (chunk, score) in enumerate(
                    zip(result.chunks[:2], result.scores[:2]), 1
                ):
                    is_report = chunk.metadata.get("is_community_report", False)
                    chunk_type = "Community Report" if is_report else "Text Chunk"
                    print(f"  [{j}] {chunk_type} (Score: {score:.3f})")
                    print(f"      {chunk.text[:150]}...")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
