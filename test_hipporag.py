"""Test script for HippoRAG Personalized PageRank retrieval strategy."""

import asyncio
from pathlib import Path

from graphunified.config.settings import Settings
from graphunified.storage.graph_store import GraphStore
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore
from graphunified.strategies.hipporag import HippoRAGStrategy
from graphunified.utils.embedding_factory import create_embedding_client


async def main():
    """Test HippoRAG PPR-based retrieval."""

    # Load configuration
    config_path = Path("settings-local-embeddings.yaml")
    settings = Settings.load(config_path)

    print("=" * 80)
    print("Testing HippoRAG Personalized PageRank Strategy")
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

    print("5. Initializing HippoRAG strategy...")
    strategy = HippoRAGStrategy(
        config=settings.strategies.graphrag,
        vector_store=vector_store,
        graph_store=graph_store,
        parquet_store=parquet_store,
        embedding_client=embedding_client,
    )

    # Load data
    print("6. Loading entities and relationships...")
    entities = [e async for e in parquet_store.load_entities()]
    relationships = [r async for r in parquet_store.load_relationships()]
    chunks = [c async for c in parquet_store.load_chunks()]
    print(
        f"   Loaded {len(entities)} entities, {len(relationships)} relationships, "
        f"{len(chunks)} chunks"
    )

    # Index
    print("\n7. Building HippoRAG indexes...")
    await strategy.index(chunks, entities, relationships, [])

    # Validate
    print("8. Validating indexes...")
    is_valid = await strategy.validate_index()
    print(f"   Indexes valid: {is_valid}")

    if not is_valid:
        print("   ERROR: Index validation failed!")
        return

    # Test queries with different reasoning patterns
    test_cases = [
        # Single-hop queries
        ("What is GraphRAG?", "single-hop"),
        ("Define machine learning", "single-hop"),
        # Multi-hop queries (require graph traversal)
        (
            "How does entity extraction relate to knowledge graphs?",
            "multi-hop (2 hops)",
        ),
        ("What technologies enable semantic search?", "multi-hop (2-3 hops)"),
        # Associative queries (PPR excels here)
        (
            "What concepts are related to artificial intelligence?",
            "associative (broad)",
        ),
        ("How do embedding models support retrieval?", "associative (specific)"),
        # Comparative queries
        ("Compare GraphRAG with traditional RAG", "comparative"),
        # Temporal/causal queries
        ("How does indexing enable retrieval?", "causal"),
    ]

    print("\n\n9. Testing retrieval with PPR:")
    print("-" * 80)

    for i, (query, reasoning_type) in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Query {i}: {query}")
        print(f"Reasoning type: {reasoning_type}")
        print("=" * 80)

        try:
            result = await strategy.retrieve(query, top_k=5)

            print(f"\nStrategy: {result.strategy}")
            print(f"Retrieved: {len(result.chunks)} chunks")
            print(f"Activated entities: {len(result.entities)} entities")
            print(f"Relationships: {len(result.relationships)} relationships")
            print(f"Time: {result.retrieval_time_ms:.2f}ms")

            if result.mean_score > 0:
                print(f"Mean score: {result.mean_score:.3f}")

            # Show PPR metrics
            print(
                f"\nPPR Metrics:"
                f"\n  Seed entities: {result.metadata.get('seed_entities', 0)}"
                f"\n  Activated entities: {result.metadata.get('activated_entities', 0)}"
                f"\n  Damping factor: {result.metadata.get('damping_factor', 0):.2f}"
                f"\n  Mean PPR score: {result.metadata.get('mean_ppr_score', 0):.6f}"
            )

            # Show top activated entities
            if result.entities:
                print(f"\nTop activated entities (by PPR):")
                for j, entity in enumerate(result.entities[:5], 1):
                    print(f"  [{j}] {entity.name} ({entity.type.value})")
                    if entity.description:
                        print(f"      {entity.description[:100]}...")

            # Show relationships in activated subgraph
            if result.relationships:
                print(f"\nRelationships in activated subgraph:")
                for j, rel in enumerate(result.relationships[:3], 1):
                    # Find source and target entities
                    source = next(
                        (e for e in result.entities if e.id == rel.source_entity_id),
                        None,
                    )
                    target = next(
                        (e for e in result.entities if e.id == rel.target_entity_id),
                        None,
                    )
                    source_name = source.name if source else str(rel.source_entity_id)
                    target_name = target.name if target else str(rel.target_entity_id)
                    print(f"  [{j}] {source_name} --[{rel.type.value}]--> {target_name}")

            # Show top chunks
            if result.chunks:
                print(f"\nTop retrieved chunks:")
                for j, (chunk, score) in enumerate(
                    zip(result.chunks[:3], result.scores[:3]), 1
                ):
                    print(f"  [{j}] Score: {score:.3f}")
                    print(f"      Connected to {len(chunk.entity_ids)} entities")
                    print(f"      Text: {chunk.text[:120]}...")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

    # Print summary statistics
    print("\n\n" + "=" * 80)
    print("HippoRAG Summary")
    print("=" * 80)
    print(
        f"\nGraph Statistics:"
        f"\n  Nodes: {graph_store.graph.number_of_nodes()}"
        f"\n  Edges: {graph_store.graph.number_of_edges()}"
        f"\n  Avg Degree: {graph_store.get_stats()['avg_degree']:.2f}"
        f"\n  Density: {graph_store.get_stats()['density']:.4f}"
    )
    print(
        f"\nPPR Configuration:"
        f"\n  Seed entities: {strategy.num_seed_entities}"
        f"\n  Damping factor: {strategy.damping_factor}"
        f"\n  Max iterations: {strategy.ppr_max_iter}"
        f"\n  Tolerance: {strategy.ppr_tolerance}"
    )


if __name__ == "__main__":
    asyncio.run(main())
