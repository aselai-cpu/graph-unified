#!/usr/bin/env python3
"""Example: Using entity-chunk reverse index in GraphRAG queries.

This demonstrates how to use the fast O(log n) entity-to-chunk lookup
in a typical GraphRAG local search scenario.
"""

import asyncio
from pathlib import Path
from typing import List

from graphunified.config.models import Chunk, Entity
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore


async def graphrag_local_search_example(
    query: str,
    vector_store: VectorStore,
    parquet_store: ParquetStore,
    top_k_entities: int = 5,
    top_k_chunks: int = 10,
):
    """Example GraphRAG local search using entity-chunk reverse index.

    This simulates a typical GraphRAG local search workflow:
    1. Embed the query
    2. Find similar entities (vector search)
    3. Fast lookup of chunks connected to those entities
    4. Generate answer from retrieved context

    Args:
        query: User query string
        vector_store: Vector store with entity-chunk index
        parquet_store: Parquet store for loading full data
        top_k_entities: Number of entities to retrieve
        top_k_chunks: Maximum chunks to return

    Returns:
        List of relevant chunks for answering the query
    """
    print(f"Query: {query}")
    print("-" * 60)

    # Step 1: Embed the query (simulated with dummy embedding)
    # In production, use: query_embedding = await embedding_client.embed([query])
    # For this example, we'll load entities and use the first one's embedding
    entities = []
    async for entity in parquet_store.load_entities():
        entities.append(entity)
        if len(entities) >= 10:  # Load first 10 for demo
            break

    if not entities or not entities[0].embedding:
        print("ERROR: No entities with embeddings found")
        return []

    query_embedding = entities[0].embedding  # Use first entity's embedding as proxy
    print(f"Step 1: Query embedded (dimension: {len(query_embedding)})")

    # Step 2: Find similar entities using vector search
    print(f"\nStep 2: Finding top {top_k_entities} similar entities...")
    entity_results = await vector_store.search_entities(
        query_vector=query_embedding,
        top_k=top_k_entities,
    )

    print(f"Found {len(entity_results)} entities:")
    for entity_id, distance, metadata in entity_results:
        print(f"  - {metadata['name']} ({metadata['type']}) [distance: {distance:.4f}]")

    # Extract entity IDs
    activated_entity_ids = [entity_id for entity_id, _, _ in entity_results]

    # Step 3: Fast lookup of chunks connected to activated entities
    # THIS IS THE KEY OPTIMIZATION - O(log n) instead of O(n) scan!
    print(f"\nStep 3: Fast lookup of chunks for {len(activated_entity_ids)} entities...")
    print("(Using entity-chunk reverse index)")

    relevant_chunks = await vector_store.get_chunks_by_entities(
        activated_entity_ids,
        parquet_store,
    )

    print(f"Found {len(relevant_chunks)} chunks connected to activated entities")

    # Limit to top_k_chunks
    relevant_chunks = relevant_chunks[:top_k_chunks]

    # Display chunk previews
    print(f"\nRetrieved {len(relevant_chunks)} chunks (limited to {top_k_chunks}):")
    for i, chunk in enumerate(relevant_chunks, 1):
        preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
        print(f"  {i}. Chunk {chunk.id}")
        print(f"     Preview: {preview}")
        print(f"     Entities: {len(chunk.entity_ids)}")
        print()

    # Step 4: Generate answer (not implemented in this example)
    print("Step 4: Generate answer from retrieved context")
    print("(Would pass chunks to LLM for answer generation)")

    return relevant_chunks


async def compare_with_naive_scan(
    activated_entity_ids: List[str],
    vector_store: VectorStore,
    parquet_store: ParquetStore,
):
    """Compare fast index lookup vs naive O(n) scan.

    This demonstrates the performance benefit of the reverse index.

    Args:
        activated_entity_ids: Entity IDs to search for
        vector_store: Vector store with reverse index
        parquet_store: Parquet store for loading chunks
    """
    import time
    from uuid import UUID

    print("\n" + "=" * 60)
    print("Performance Comparison: Index vs Naive Scan")
    print("=" * 60)

    # Method 1: Fast index lookup
    print("\nMethod 1: Entity-chunk reverse index (O(log n))")
    start_time = time.time()
    indexed_chunks = await vector_store.get_chunks_by_entities(
        activated_entity_ids,
        parquet_store,
    )
    index_duration = time.time() - start_time
    print(f"  Found {len(indexed_chunks)} chunks in {index_duration:.4f}s")

    # Method 2: Naive O(n) scan
    print("\nMethod 2: Naive Parquet scan (O(n))")
    start_time = time.time()
    scanned_chunks = []
    entity_id_uuids = {UUID(eid) for eid in activated_entity_ids}

    async for chunk in parquet_store.load_chunks():
        if any(eid in entity_id_uuids for eid in chunk.entity_ids):
            scanned_chunks.append(chunk)

    scan_duration = time.time() - start_time
    print(f"  Found {len(scanned_chunks)} chunks in {scan_duration:.4f}s")

    # Compare
    print("\nResults:")
    if len(indexed_chunks) == len(scanned_chunks):
        print(f"  Correctness: PASS (both found {len(indexed_chunks)} chunks)")
    else:
        print(f"  Correctness: FAIL (index: {len(indexed_chunks)}, scan: {len(scanned_chunks)})")

    if scan_duration > 0:
        speedup = scan_duration / index_duration if index_duration > 0 else float('inf')
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time saved: {(scan_duration - index_duration) * 1000:.2f}ms")
    else:
        print("  Speedup: N/A (dataset too small)")

    print("\nNote: Speedup increases with dataset size!")
    print("For 100K+ chunks, expect 10-100x speedup.")


async def main():
    """Main example runner."""
    print("=" * 60)
    print("Entity-Chunk Reverse Index Usage Example")
    print("=" * 60)
    print()

    # Setup paths
    test_output_dir = Path("/Users/aselaillayapparachchi/code/GraphRAG/Microsoft/graph-unified/test-output-stage5")

    if not test_output_dir.exists():
        print(f"ERROR: Test data directory not found: {test_output_dir}")
        print("Please run the indexing pipeline first to generate test data.")
        return

    # Initialize stores
    parquet_store = ParquetStore(test_output_dir)
    vector_store = VectorStore(test_output_dir / "lancedb", dimension=1024)

    try:
        # Example 1: GraphRAG local search
        print("EXAMPLE 1: GraphRAG Local Search Workflow")
        print("=" * 60)
        chunks = await graphrag_local_search_example(
            query="What is artificial intelligence?",
            vector_store=vector_store,
            parquet_store=parquet_store,
            top_k_entities=3,
            top_k_chunks=5,
        )

        if not chunks:
            print("No chunks found. Check that entity-chunk index is built.")
            return

        # Example 2: Performance comparison
        # Load some entities to compare
        entities = []
        async for entity in parquet_store.load_entities():
            entities.append(entity)
            if len(entities) >= 5:
                break

        if entities:
            entity_ids = [str(e.id) for e in entities[:3]]
            await compare_with_naive_scan(
                entity_ids,
                vector_store,
                parquet_store,
            )

        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: Example failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
