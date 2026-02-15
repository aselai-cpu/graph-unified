"""Test script for Naive RAG strategy."""

import asyncio
from pathlib import Path

from graphunified.config.settings import Settings
from graphunified.strategies.naive import NaiveRAGStrategy
from graphunified.storage.vector_store import VectorStore
from graphunified.utils.embedding_factory import create_embedding_client


async def main():
    """Test Naive RAG retrieval."""

    # Load configuration
    config_path = Path("settings-local-embeddings.yaml")
    settings = Settings.load(config_path)

    print("=" * 80)
    print("Testing Naive RAG Strategy")
    print("=" * 80)

    # Initialize components
    print("\n1. Loading vector store...")
    vector_store_path = Path("test-output-stage5/lancedb")
    vector_store = VectorStore.from_config(
        settings.storage.vector_db,
        vector_store_path,
        settings.embedding.dimension,
    )

    print("2. Creating embedding client...")
    embedding_client = create_embedding_client(settings.embedding)

    print("3. Initializing Naive RAG strategy...")
    strategy = NaiveRAGStrategy(
        config=settings.strategies.naive,
        vector_store=vector_store,
        embedding_client=embedding_client,
    )

    # Validate index
    print("\n4. Validating index...")
    is_valid = await strategy.validate_index()
    print(f"   Index valid: {is_valid}")

    if not is_valid:
        print("   ERROR: Index validation failed!")
        return

    # Test queries
    queries = [
        "What is GraphRAG?",
        "How does entity extraction work?",
        "Tell me about knowledge graphs",
    ]

    print("\n5. Testing retrieval:")
    print("-" * 80)

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)

        try:
            result = await strategy.retrieve(query, top_k=3)

            print(f"Strategy: {result.strategy}")
            print(f"Retrieved: {len(result.chunks)} chunks")
            print(f"Time: {result.retrieval_time_ms:.2f}ms")
            print(f"Mean score: {result.mean_score:.3f}")
            print(f"Top score: {result.top_score:.3f}")

            print("\nTop 3 chunks:")
            for j, (chunk, score) in enumerate(zip(result.chunks, result.scores), 1):
                print(f"\n  [{j}] Score: {score:.3f}")
                print(f"      Text: {chunk.text[:150]}...")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
