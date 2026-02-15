"""Test script for Hybrid RAG strategy."""

import asyncio
from pathlib import Path

from graphunified.config.settings import Settings
from graphunified.index.stages.index import BM25Index
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore
from graphunified.strategies.hybrid import HybridRAGStrategy
from graphunified.utils.embedding_factory import create_embedding_client


async def main():
    """Test Hybrid RAG retrieval."""

    # Load configuration
    config_path = Path("settings-local-embeddings.yaml")
    settings = Settings.load(config_path)

    print("=" * 80)
    print("Testing Hybrid RAG Strategy")
    print("=" * 80)

    # Initialize components
    print("\n1. Loading vector store...")
    vector_store_path = Path("test-output-stage5/lancedb")
    vector_store = VectorStore.from_config(
        settings.storage.vector_db,
        vector_store_path,
        settings.embedding.dimension,
    )

    print("2. Rebuilding BM25 index from chunks...")
    # Load chunks from Parquet
    parquet_store = ParquetStore(Path("test-output-stage5"))
    chunks = [chunk async for chunk in parquet_store.load_chunks()]
    print(f"   Loaded {len(chunks)} chunks")

    # Build BM25 index
    bm25_index = BM25Index(
        k1=settings.strategies.hybrid.bm25_k1,
        b=settings.strategies.hybrid.bm25_b,
    )
    for chunk in chunks:
        bm25_index.add_document(str(chunk.id), chunk.text)
    bm25_index.finalize()
    print(f"   Built BM25 index: {bm25_index.num_docs} docs, {len(bm25_index.inverted_index)} terms")

    print("3. Creating embedding client...")
    embedding_client = create_embedding_client(settings.embedding)

    print("4. Initializing Hybrid RAG strategy...")
    strategy = HybridRAGStrategy(
        config=settings.strategies.hybrid,
        vector_store=vector_store,
        bm25_index=bm25_index,
        embedding_client=embedding_client,
    )

    # Validate index
    print("\n5. Validating indexes...")
    is_valid = await strategy.validate_index()
    print(f"   Indexes valid: {is_valid}")

    if not is_valid:
        print("   ERROR: Index validation failed!")
        return

    # Test queries
    queries = [
        "What is GraphRAG?",
        "How does entity extraction work?",
        "Tell me about knowledge graphs",
    ]

    # Test with different alpha values
    alphas = [0.0, 0.5, 1.0]  # 0.0=BM25 only, 0.5=balanced, 1.0=vector only

    print("\n6. Testing retrieval with different alpha values:")
    print("-" * 80)

    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print('='*80)

        for alpha in alphas:
            print(f"\n  Alpha = {alpha} {'(BM25 only)' if alpha == 0.0 else '(Vector only)' if alpha == 1.0 else '(Balanced)'}")
            print("  " + "-" * 76)

            try:
                result = await strategy.retrieve(query, top_k=3, alpha=alpha)

                print(f"  Retrieved: {result.metadata['vector_candidates']} vector + {result.metadata['bm25_candidates']} BM25 candidates")
                print(f"  Fused to: {len(result.chunks)} chunks")
                print(f"  Time: {result.retrieval_time_ms:.2f}ms")
                print(f"  Mean score: {result.mean_score:.3f}")
                print(f"  Top score: {result.top_score:.3f}")

                if len(result.chunks) > 0:
                    top_chunk = result.chunks[0]
                    top_score = result.scores[0]
                    print(f"  Top result [{top_score:.3f}]: {top_chunk.text[:100]}...")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
