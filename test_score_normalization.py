"""Test score normalization across strategies."""

import asyncio
from pathlib import Path

from graphunified.config.settings import Settings
from graphunified.storage.graph_store import GraphStore
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore
from graphunified.strategies.naive import NaiveRAGStrategy
from graphunified.strategies.hybrid import HybridRAGStrategy
from graphunified.strategies.graphrag_local import GraphRAGLocalStrategy
from graphunified.utils.embedding_factory import create_embedding_client


async def main():
    """Test score normalization across all strategies."""

    # Load configuration
    config_path = Path("settings-local-embeddings.yaml")
    settings = Settings.load(config_path)

    print("=" * 80)
    print("Score Normalization Test")
    print("=" * 80)

    # Initialize components
    print("\nInitializing stores...")
    vector_store_path = Path("test-output-stage5/lancedb")
    vector_store = VectorStore.from_config(
        settings.storage.vector_db,
        vector_store_path,
        settings.embedding.dimension,
    )

    parquet_store_path = Path("test-output-stage5")
    parquet_store = ParquetStore(parquet_store_path)

    graph_store_path = parquet_store_path / "graph"
    graph_store = GraphStore(
        root_dir=graph_store_path,
        directed=True,
        graph_format="pickle",
    )
    await graph_store.load()

    embedding_client = create_embedding_client(settings.embedding)

    # Initialize strategies
    print("Initializing strategies...")
    naive = NaiveRAGStrategy(
        config=settings.strategies.naive,
        vector_store=vector_store,
        embedding_client=embedding_client,
    )

    # Load chunks for BM25 index
    from graphunified.index.stages.index import BM25Index
    chunks = [c async for c in parquet_store.load_chunks()]
    bm25_index = BM25Index()
    for chunk in chunks:
        bm25_index.add_document(str(chunk.id), chunk.text)

    hybrid = HybridRAGStrategy(
        config=settings.strategies.hybrid,
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_client=embedding_client,
    )

    entities = [e async for e in parquet_store.load_entities()]
    relationships = [r async for r in parquet_store.load_relationships()]

    graphrag_local = GraphRAGLocalStrategy(
        config=settings.strategies.graphrag,
        vector_store=vector_store,
        graph_store=graph_store,
        parquet_store=parquet_store,
        embedding_client=embedding_client,
    )
    await graphrag_local._build_caches(entities, relationships)

    # Test query
    query = "What is GraphRAG?"

    print(f"\n\nQuery: '{query}'\n")
    print("=" * 80)

    # Test each strategy
    strategies = [
        ("Naive RAG", naive),
        ("Hybrid RAG", hybrid),
        ("GraphRAG Local", graphrag_local),
    ]

    results = {}
    for name, strategy in strategies:
        print(f"\n{name}:")
        print("-" * 40)

        result = await strategy.retrieve(query, top_k=5)

        print(f"Retrieved: {len(result.chunks)} chunks")
        print(f"Raw scores: {[f'{s:.4f}' for s in result.scores[:5]]}")
        print(f"Score range: [{min(result.scores):.4f}, {max(result.scores):.4f}]")
        print(f"Mean score: {result.mean_score:.4f}")

        # Test normalization methods
        if result.scores:
            print(f"\nNormalization tests:")

            # Min-max normalization
            norm_minmax = strategy._normalize_scores(result.scores, method="minmax")
            print(
                f"  MinMax: {[f'{s:.4f}' for s in norm_minmax[:5]]} "
                f"(range: [{min(norm_minmax):.4f}, {max(norm_minmax):.4f}])"
            )

            # Rank normalization
            norm_rank = strategy._normalize_scores(result.scores, method="rank")
            print(
                f"  Rank:   {[f'{s:.4f}' for s in norm_rank[:5]]} "
                f"(range: [{min(norm_rank):.4f}, {max(norm_rank):.4f}])"
            )

            results[name] = {
                "raw": result.scores,
                "minmax": norm_minmax,
                "rank": norm_rank,
            }

    # Compare normalized scores across strategies
    print("\n\n" + "=" * 80)
    print("Cross-Strategy Comparison (Top Result)")
    print("=" * 80)

    print("\nRaw Scores (NOT comparable):")
    for name in results:
        raw_top = results[name]["raw"][0] if results[name]["raw"] else 0
        print(f"  {name:20s}: {raw_top:.6f}")

    print("\nMinMax Normalized (comparable):")
    for name in results:
        norm_top = results[name]["minmax"][0] if results[name]["minmax"] else 0
        print(f"  {name:20s}: {norm_top:.6f}")

    print("\nRank Normalized (comparable):")
    for name in results:
        rank_top = results[name]["rank"][0] if results[name]["rank"] else 0
        print(f"  {name:20s}: {rank_top:.6f}")

    print("\n\n" + "=" * 80)
    print("✅ Score Normalization Test Complete")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Raw scores vary widely across strategies (not comparable)")
    print("2. MinMax normalization brings all scores to [0, 1] range")
    print("3. Rank normalization makes all strategies directly comparable")
    print("4. Use normalized scores for query routing and strategy selection")


if __name__ == "__main__":
    asyncio.run(main())
