"""Test script for GraphRAG Local strategy."""

import asyncio
from pathlib import Path

from graphunified.config.settings import Settings
from graphunified.storage.graph_store import GraphStore
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore
from graphunified.strategies.graphrag_local import GraphRAGLocalStrategy
from graphunified.utils.embedding_factory import create_embedding_client


async def main():
    """Test GraphRAG Local retrieval."""

    # Load configuration
    config_path = Path("settings-local-embeddings.yaml")
    settings = Settings.load(config_path)

    print("=" * 80)
    print("Testing GraphRAG Local Strategy")
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

    print("3. Loading/building knowledge graph...")
    graph_store_path = parquet_store_path / "graph"
    graph_store_path.mkdir(exist_ok=True)
    # Use pickle format instead of graphml (pickle supports dict metadata)
    graph_store = GraphStore(
        root_dir=graph_store_path,
        directed=True,
        graph_format="pickle",
    )

    # Try to load existing graph, or build new one
    graph_file = graph_store_path / "graph.pickle"
    if graph_file.exists():
        print("   Loading existing graph...")
        await graph_store.load()
    else:
        print("   Building new graph from entities and relationships...")
        entities = [e async for e in parquet_store.load_entities()]
        relationships = [r async for r in parquet_store.load_relationships()]
        print(f"   Loaded {len(entities)} entities, {len(relationships)} relationships")

        await graph_store.build_graph(entities, relationships)
        await graph_store.save()
        print(f"   Built graph: {graph_store.graph.number_of_nodes()} nodes, {graph_store.graph.number_of_edges()} edges")

    print(f"   Graph ready: {graph_store.graph.number_of_nodes()} nodes, {graph_store.graph.number_of_edges()} edges")

    print("4. Creating embedding client...")
    embedding_client = create_embedding_client(settings.embedding)

    print("5. Initializing GraphRAG Local strategy...")

    # Load all entities and relationships for caching
    print("   Loading entities and relationships for cache...")
    entities = [e async for e in parquet_store.load_entities()]
    relationships = [r async for r in parquet_store.load_relationships()]
    chunks = [c async for c in parquet_store.load_chunks()]

    strategy = GraphRAGLocalStrategy(
        config=settings.strategies.graphrag,
        vector_store=vector_store,
        graph_store=graph_store,
        parquet_store=parquet_store,
        embedding_client=embedding_client,
    )

    # Build caches
    await strategy._build_caches(entities, relationships)

    # Validate indexes
    print("\n6. Validating indexes...")
    is_valid = await strategy.validate_index()
    print(f"   Indexes valid: {is_valid}")

    if not is_valid:
        print("   ERROR: Index validation failed!")
        return

    # Test queries
    queries = [
        "What is GraphRAG?",
        "How are entities extracted?",
        "Tell me about knowledge graphs and retrieval",
    ]

    print("\n7. Testing retrieval:")
    print("-" * 80)

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}: {query}")
        print('='*80)

        try:
            result = await strategy.retrieve(query, top_k=5)

            print(f"\nStrategy: {result.strategy}")
            print(f"Retrieved: {len(result.chunks)} chunks")
            print(f"Entities: {len(result.entities)} entities in local graph")
            print(f"Relationships: {len(result.relationships)} relationships")
            print(f"Time: {result.retrieval_time_ms:.2f}ms")
            print(f"Mean chunk score: {result.mean_score:.3f}")

            print(f"\nMetadata:")
            for key, value in result.metadata.items():
                print(f"  {key}: {value}")

            if result.entities:
                print(f"\nTop entities:")
                for j, entity in enumerate(result.entities[:3], 1):
                    print(f"  [{j}] {entity.name} ({entity.type.value})")
                    if entity.description:
                        print(f"      {entity.description[:100]}...")

            if result.relationships:
                print(f"\nTop relationships:")
                for j, rel in enumerate(result.relationships[:3], 1):
                    # Find source and target entities
                    source = next((e for e in result.entities if e.id == rel.source_entity_id), None)
                    target = next((e for e in result.entities if e.id == rel.target_entity_id), None)
                    source_name = source.name if source else str(rel.source_entity_id)
                    target_name = target.name if target else str(rel.target_entity_id)
                    print(f"  [{j}] {source_name} --[{rel.type.value}]--> {target_name}")

            if result.chunks:
                print(f"\nTop chunks:")
                for j, (chunk, score) in enumerate(zip(result.chunks[:3], result.scores[:3]), 1):
                    print(f"  [{j}] Score: {score:.3f}")
                    print(f"      Text: {chunk.text[:120]}...")
                    print(f"      Connected to {len(chunk.entity_ids)} entities")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
