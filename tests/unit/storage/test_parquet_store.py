"""Tests for Parquet storage."""

import pytest

from graphunified.storage.parquet_store import ParquetStore


@pytest.mark.asyncio
class TestParquetStore:
    """Tests for ParquetStore."""

    async def test_save_and_load_documents(self, tmp_storage_dir, sample_document):
        """Test saving and loading documents."""
        store = ParquetStore(tmp_storage_dir)

        # Save document
        await store.save_documents([sample_document])
        await store.flush()

        # Load documents
        loaded_docs = [doc async for doc in store.load_documents()]

        assert len(loaded_docs) == 1
        loaded_doc = loaded_docs[0]
        assert loaded_doc.id == sample_document.id
        assert loaded_doc.filename == sample_document.filename
        assert loaded_doc.text == sample_document.text

    async def test_save_and_load_chunks(self, tmp_storage_dir, sample_chunks):
        """Test saving and loading chunks."""
        store = ParquetStore(tmp_storage_dir)

        # Save chunks
        await store.save_chunks(sample_chunks)
        await store.flush()

        # Load chunks
        loaded_chunks = [chunk async for chunk in store.load_chunks()]

        assert len(loaded_chunks) == len(sample_chunks)
        assert loaded_chunks[0].chunk_index == 0
        assert loaded_chunks[1].chunk_index == 1
        assert loaded_chunks[2].chunk_index == 2

    async def test_save_and_load_entities(self, tmp_storage_dir, sample_entities):
        """Test saving and loading entities."""
        store = ParquetStore(tmp_storage_dir)

        # Save entities
        await store.save_entities(sample_entities)
        await store.flush()

        # Load entities
        loaded_entities = [entity async for entity in store.load_entities()]

        assert len(loaded_entities) == len(sample_entities)
        assert loaded_entities[0].name == "climate change"
        assert loaded_entities[1].name == "global warming"

    async def test_save_and_load_relationships(self, tmp_storage_dir, sample_relationship):
        """Test saving and loading relationships."""
        store = ParquetStore(tmp_storage_dir)

        # Save relationship
        await store.save_relationships([sample_relationship])
        await store.flush()

        # Load relationships
        loaded_rels = [rel async for rel in store.load_relationships()]

        assert len(loaded_rels) == 1
        loaded_rel = loaded_rels[0]
        assert loaded_rel.id == sample_relationship.id
        assert loaded_rel.source_entity_id == sample_relationship.source_entity_id
        assert loaded_rel.target_entity_id == sample_relationship.target_entity_id

    async def test_batch_flush(self, tmp_storage_dir, sample_document):
        """Test that data is flushed when buffer reaches batch size."""
        store = ParquetStore(tmp_storage_dir, batch_size=2)

        # Add 3 documents (should auto-flush at 2)
        for i in range(3):
            doc = sample_document.model_copy()
            doc.filename = f"doc_{i}.txt"
            await store.save_documents([doc])

        # Final flush for remaining document
        await store.flush()

        # Load and verify all 3 documents were saved
        loaded_docs = [doc async for doc in store.load_documents()]
        assert len(loaded_docs) == 3

    async def test_load_from_empty_store(self, tmp_storage_dir):
        """Test loading from empty store returns no items."""
        store = ParquetStore(tmp_storage_dir)

        docs = [doc async for doc in store.load_documents()]
        assert len(docs) == 0

        chunks = [chunk async for chunk in store.load_chunks()]
        assert len(chunks) == 0
