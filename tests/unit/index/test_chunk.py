"""Unit tests for chunking stage."""

import pytest

from graphunified.config.models import Document
from graphunified.index.stages.chunk import ChunkStage


@pytest.mark.unit
class TestChunkStage:
    """Tests for ChunkStage."""

    @pytest.fixture
    def sample_document(self):
        """Create a sample document."""
        # Create text with known token count
        text = " ".join([f"word{i}" for i in range(100)])  # ~100 tokens
        return Document(
            filename="test.txt",
            text=text,
            char_count=len(text),
            token_count=100,
        )

    @pytest.mark.asyncio
    async def test_chunk_document(self, sample_document):
        """Test chunking a single document."""
        stage = ChunkStage(chunk_size=20, chunk_overlap=5)

        result = await stage.execute([sample_document])

        assert result.status.value == "completed"
        assert len(result.data) > 0

        chunks = result.data
        # With 100 tokens, chunk_size=20, overlap=5, stride=15
        # We expect around 7-8 chunks
        assert len(chunks) >= 6
        assert len(chunks) <= 10

        # Verify chunk properties
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.document_id == sample_document.id
            assert chunk.token_count <= 20
            assert chunk.start_char < chunk.end_char

    @pytest.mark.asyncio
    async def test_chunk_empty_documents(self):
        """Test chunking with empty document list."""
        stage = ChunkStage()

        result = await stage.execute([])

        assert result.status.value == "completed"
        assert len(result.data) == 0

    @pytest.mark.asyncio
    async def test_chunk_overlap_validation(self):
        """Test that overlap must be less than chunk size."""
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            ChunkStage(chunk_size=100, chunk_overlap=100)

    @pytest.mark.asyncio
    async def test_chunk_multiple_documents(self, sample_document):
        """Test chunking multiple documents."""
        doc2 = Document(
            filename="test2.txt",
            text="Short document.",
            char_count=15,
            token_count=3,
        )

        stage = ChunkStage(chunk_size=20, chunk_overlap=5)
        result = await stage.execute([sample_document, doc2])

        assert result.status.value == "completed"
        chunks = result.data

        # Verify chunks from both documents
        doc1_chunks = [c for c in chunks if c.document_id == sample_document.id]
        doc2_chunks = [c for c in chunks if c.document_id == doc2.id]

        assert len(doc1_chunks) > 0
        assert len(doc2_chunks) > 0

    @pytest.mark.asyncio
    async def test_chunk_preserves_character_positions(self, sample_document):
        """Test that character positions are preserved."""
        stage = ChunkStage(chunk_size=20, chunk_overlap=5)
        result = await stage.execute([sample_document])

        chunks = result.data

        # Verify positions are sequential and within document bounds
        for chunk in chunks:
            assert 0 <= chunk.start_char < len(sample_document.text)
            assert chunk.start_char < chunk.end_char <= len(sample_document.text)
