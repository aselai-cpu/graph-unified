"""Test script for GraphRAG Global strategy."""

import asyncio
from pathlib import Path

from graphunified.config.settings import Settings
from graphunified.storage.graph_store import GraphStore
from graphunified.storage.parquet_store import ParquetStore
from graphunified.strategies.graphrag_global import GraphRAGGlobalStrategy
from graphunified.utils.embedding_factory import create_embedding_client
from graphunified.utils.llm import ClaudeClient


async def main():
    """Test GraphRAG Global retrieval."""

    # Load configuration
    config_path = Path("settings-local-embeddings.yaml")
    settings = Settings.load(config_path)

    print("=" * 80)
    print("Testing GraphRAG Global Strategy")
    print("=" * 80)

    # Initialize components
    print("\n1. Loading Parquet store...")
    parquet_store_path = Path("test-output-stage5")
    parquet_store = ParquetStore(parquet_store_path)

    print("2. Loading knowledge graph...")
    graph_store_path = parquet_store_path / "graph"
    graph_store = GraphStore(
        root_dir=graph_store_path,
        directed=True,
        graph_format="pickle",
    )

    # Load existing graph
    if (graph_store_path / "graph.pickle").exists():
        await graph_store.load()
        print(f"   Graph loaded: {graph_store.graph.number_of_nodes()} nodes, {graph_store.graph.number_of_edges()} edges")
    else:
        print("   ERROR: Graph not found! Run test_graphrag_local.py first to build the graph.")
        return

    print("3. Creating LLM and embedding clients...")
    llm_client = ClaudeClient.from_config(settings.llm)
    embedding_client = create_embedding_client(settings.embedding)

    print("4. Initializing GraphRAG Global strategy...")
    strategy = GraphRAGGlobalStrategy(
        config=settings.strategies.graphrag,
        graph_store=graph_store,
        parquet_store=parquet_store,
        llm_client=llm_client,
        embedding_client=embedding_client,
    )

    # Load data
    print("5. Loading entities, relationships, and chunks...")
    entities = [e async for e in parquet_store.load_entities()]
    relationships = [r async for r in parquet_store.load_relationships()]
    chunks = [c async for c in parquet_store.load_chunks()]
    print(f"   Loaded {len(entities)} entities, {len(relationships)} relationships, {len(chunks)} chunks")

    # Index (detect communities and generate reports)
    print("\n6. Detecting communities and generating reports...")
    print("   (This may take a few minutes due to LLM calls...)")
    await strategy.index(chunks, entities, relationships, [])

    print(f"   Detected {len(strategy._communities)} communities")
    print(f"   Generated {len(strategy._community_reports)} reports")

    # Validate indexes
    print("\n7. Validating indexes...")
    is_valid = await strategy.validate_index()
    print(f"   Indexes valid: {is_valid}")

    if not is_valid:
        print("   ERROR: Index validation failed!")
        return

    # Show community summaries
    print("\n8. Community Summaries:")
    print("-" * 80)
    for idx, community in enumerate(strategy._communities[:5]):  # Show first 5
        print(f"\nCommunity {idx}:")
        print(f"  Title: {community.title}")
        print(f"  Size: {len(community.entity_ids)} entities")
        print(f"  Summary: {community.summary[:200]}...")

    # Test queries
    queries = [
        "What are the main themes in this corpus?",
        "Summarize the key topics discussed",
        "What does the corpus say about technology?",
    ]

    print("\n\n9. Testing retrieval:")
    print("-" * 80)

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print('='*80)

        try:
            result = await strategy.retrieve(query, top_k=3)

            print(f"\nStrategy: {result.strategy}")
            print(f"Retrieved: {len(result.communities)} communities")
            print(f"Time: {result.retrieval_time_ms:.2f}ms")
            print(f"Total communities searched: {result.metadata['total_communities']}")

            if result.communities:
                print(f"\nTop communities:")
                for j, (community, score) in enumerate(zip(result.communities, result.scores), 1):
                    print(f"\n  [{j}] Score: {score:.3f}")
                    print(f"      Title: {community.title}")
                    print(f"      Size: {len(community.entity_ids)} entities")
                    print(f"      Summary: {community.summary[:150]}...")

            if result.chunks:
                print(f"\nCommunity reports (as chunks):")
                for j, chunk in enumerate(result.chunks[:2], 1):
                    print(f"\n  Report {j} (truncated):")
                    print(f"  {chunk.text[:300]}...")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
