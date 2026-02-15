#!/usr/bin/env python3
"""Test script for entity-chunk reverse index functionality.

This script verifies the O(log n) entity-to-chunk lookup is working correctly.
"""

import asyncio
import time
from pathlib import Path
from typing import List
from uuid import UUID

from graphunified.config.models import Chunk
from graphunified.storage.parquet_store import ParquetStore
from graphunified.storage.vector_store import VectorStore


async def load_test_data(parquet_store: ParquetStore):
    """Load test data from parquet store.

    Returns:
        Tuple of (chunks, entities)
    """
    print("Loading chunks...")
    chunks = [chunk async for chunk in parquet_store.load_chunks()]
    print(f"  Loaded {len(chunks)} chunks")

    print("Loading entities...")
    entities = [entity async for entity in parquet_store.load_entities()]
    print(f"  Loaded {len(entities)} entities")

    # Check if chunks have entity_ids populated
    chunks_with_entities = [c for c in chunks if c.entity_ids]
    print(f"  {len(chunks_with_entities)} chunks have entity references")

    if chunks_with_entities:
        sample_chunk = chunks_with_entities[0]
        print(f"  Sample chunk {sample_chunk.id} has {len(sample_chunk.entity_ids)} entities")

    return chunks, entities


async def test_index_building(
    vector_store: VectorStore,
    chunks: List[Chunk],
):
    """Test building the entity-chunk reverse index."""
    print("\n=== Testing Index Building ===")

    # Build chunk embeddings dict
    chunk_embeddings = {str(c.id): c.embedding for c in chunks if c.embedding}
    print(f"Found embeddings for {len(chunk_embeddings)} chunks")

    # Build the index
    start_time = time.time()
    await vector_store.index_entity_chunk_mappings(chunks, chunk_embeddings)
    duration = time.time() - start_time

    print(f"Index built in {duration:.2f}s")

    # Verify table was created
    db = await vector_store._get_db()
    tables = await asyncio.to_thread(db.table_names)
    assert "entity_chunks" in tables, "entity_chunks table not created"
    print("Verified: entity_chunks table created successfully")

    # Check row count
    table = await vector_store._get_or_create_table("entity_chunks")
    count = await asyncio.to_thread(table.count_rows)
    print(f"Indexed {count} entity-chunk mappings")

    return count


async def test_entity_lookup(
    vector_store: VectorStore,
    parquet_store: ParquetStore,
    chunks: List[Chunk],
    entity_ids: List[str],
):
    """Test looking up chunks by entity IDs."""
    print(f"\n=== Testing Chunk Lookup for {len(entity_ids)} Entities ===")

    # Test the fast lookup
    start_time = time.time()
    found_chunks = await vector_store.get_chunks_by_entities(entity_ids, parquet_store)
    duration = time.time() - start_time

    print(f"Found {len(found_chunks)} chunks in {duration:.3f}s")

    # Verify correctness by comparing with O(n) scan
    print("\nVerifying correctness against O(n) scan...")
    start_time = time.time()
    expected_chunks = []
    entity_id_uuids = {UUID(eid) for eid in entity_ids}
    for chunk in chunks:
        if any(eid in entity_id_uuids for eid in chunk.entity_ids):
            expected_chunks.append(chunk)
    scan_duration = time.time() - start_time

    print(f"O(n) scan found {len(expected_chunks)} chunks in {scan_duration:.3f}s")

    # Compare results
    found_ids = {c.id for c in found_chunks}
    expected_ids = {c.id for c in expected_chunks}

    if found_ids == expected_ids:
        print("PASS: Results match O(n) scan exactly")
        speedup = scan_duration / duration if duration > 0 else float('inf')
        print(f"Speedup: {speedup:.2f}x faster")
    else:
        missing = expected_ids - found_ids
        extra = found_ids - expected_ids
        print(f"FAIL: Results don't match")
        if missing:
            print(f"  Missing {len(missing)} chunks: {list(missing)[:5]}")
        if extra:
            print(f"  Extra {len(extra)} chunks: {list(extra)[:5]}")

    return found_chunks


async def test_edge_cases(
    vector_store: VectorStore,
    parquet_store: ParquetStore,
):
    """Test edge cases for entity lookup."""
    print("\n=== Testing Edge Cases ===")

    # Test empty list
    result = await vector_store.get_chunks_by_entities([], parquet_store)
    assert result == [], "Empty entity list should return empty results"
    print("PASS: Empty entity list returns empty results")

    # Test non-existent entity
    fake_uuid = "00000000-0000-0000-0000-000000000000"
    result = await vector_store.get_chunks_by_entities([fake_uuid], parquet_store)
    assert result == [], "Non-existent entity should return empty results"
    print("PASS: Non-existent entity returns empty results")


async def main():
    """Main test runner."""
    print("=" * 60)
    print("Entity-Chunk Reverse Index Test")
    print("=" * 60)

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
        # Load test data
        chunks, entities = await load_test_data(parquet_store)

        if not chunks:
            print("ERROR: No chunks found in test data")
            return

        if not entities:
            print("ERROR: No entities found in test data")
            return

        # Build the index
        mapping_count = await test_index_building(vector_store, chunks)

        if mapping_count == 0:
            print("WARNING: No entity-chunk mappings created. Chunks may not have entity_ids.")
            return

        # Test with first 3 entities
        test_entity_ids = [str(e.id) for e in entities[:3]]
        print(f"\nTesting with entities: {[e.name for e in entities[:3]]}")

        await test_entity_lookup(
            vector_store,
            parquet_store,
            chunks,
            test_entity_ids,
        )

        # Test with more entities (10)
        if len(entities) >= 10:
            test_entity_ids = [str(e.id) for e in entities[:10]]
            print(f"\nTesting with 10 entities...")

            await test_entity_lookup(
                vector_store,
                parquet_store,
                chunks,
                test_entity_ids,
            )

        # Test edge cases
        await test_edge_cases(vector_store, parquet_store)

        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
