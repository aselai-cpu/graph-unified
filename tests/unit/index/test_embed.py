"""Unit tests for embedding stage."""

from uuid import uuid4

import pytest

from graphunified.config.models import Chunk, Entity, EntityType
from graphunified.index.stages.embed import EmbedStage


@pytest.mark.unit
class TestEmbedStage:
    """Tests for EmbedStage."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        doc_id = uuid4()
        return [
            Chunk(
                id=uuid4(),
                document_id=doc_id,
                chunk_index=0,
                text="This is a test chunk with some content.",
                start_char=0,
                end_char=40,
                token_count=10,
            ),
            Chunk(
                id=uuid4(),
                document_id=doc_id,
                chunk_index=1,
                text="Another test chunk with different content.",
                start_char=40,
                end_char=82,
                token_count=8,
            ),
        ]

    @pytest.fixture
    def sample_entities(self):
        """Create sample entities."""
        return [
            Entity(
                id=uuid4(),
                name="Test Entity",
                type=EntityType.CONCEPT,
                description="A test entity for embeddings",
                extraction_confidence=0.9,
            ),
            Entity(
                id=uuid4(),
                name="Another Entity",
                type=EntityType.ORGANIZATION,
                description="Another test entity",
                extraction_confidence=0.85,
            ),
        ]

    @pytest.fixture
    def mock_embedding_client(self, mocker):
        """Create a mock embedding client."""
        client = mocker.Mock()
        client.model = "test-model"

        # Mock embed method to return dummy embeddings
        async def mock_embed(texts):
            return [[0.1] * 1024 for _ in texts]

        client.embed = mocker.AsyncMock(side_effect=mock_embed)

        return client

    @pytest.mark.asyncio
    async def test_embed_chunks(self, mock_embedding_client, sample_chunks):
        """Test embedding chunks."""
        stage = EmbedStage(embedding_client=mock_embedding_client, embed_chunks=True, embed_entities=False)

        input_data = {"chunks": sample_chunks, "entities": [], "relationships": [], "entity_map": {}}

        result = await stage.execute(input_data)

        assert result.status.value == "completed"
        chunks = result.data["chunks"]

        # Verify all chunks have embeddings
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 1024
            assert chunk.embedding_model == "test-model"

        # Verify embed was called once with all chunk texts
        mock_embedding_client.embed.assert_called_once()
        call_args = mock_embedding_client.embed.call_args[0][0]
        assert len(call_args) == len(sample_chunks)

    @pytest.mark.asyncio
    async def test_embed_entities(self, mock_embedding_client, sample_entities):
        """Test embedding entities."""
        stage = EmbedStage(embedding_client=mock_embedding_client, embed_chunks=False, embed_entities=True)

        input_data = {"chunks": [], "entities": sample_entities, "relationships": [], "entity_map": {}}

        result = await stage.execute(input_data)

        assert result.status.value == "completed"
        entities = result.data["entities"]

        # Verify all entities have embeddings
        for entity in entities:
            assert entity.embedding is not None
            assert len(entity.embedding) == 1024
            assert entity.embedding_model == "test-model"

        # Verify embed was called
        mock_embedding_client.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_both_chunks_and_entities(
        self, mock_embedding_client, sample_chunks, sample_entities
    ):
        """Test embedding both chunks and entities."""
        stage = EmbedStage(embedding_client=mock_embedding_client, embed_chunks=True, embed_entities=True)

        input_data = {
            "chunks": sample_chunks,
            "entities": sample_entities,
            "relationships": [],
            "entity_map": {},
        }

        result = await stage.execute(input_data)

        assert result.status.value == "completed"

        chunks = result.data["chunks"]
        entities = result.data["entities"]

        # Verify chunks have embeddings
        assert all(c.embedding is not None for c in chunks)

        # Verify entities have embeddings
        assert all(e.embedding is not None for e in entities)

        # Verify embed was called twice (once for chunks, once for entities)
        assert mock_embedding_client.embed.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_empty_inputs(self, mock_embedding_client):
        """Test embedding with empty inputs."""
        stage = EmbedStage(embedding_client=mock_embedding_client)

        input_data = {"chunks": [], "entities": [], "relationships": [], "entity_map": {}}

        result = await stage.execute(input_data)

        assert result.status.value == "completed"
        assert result.data["chunks"] == []
        assert result.data["entities"] == []

    @pytest.mark.asyncio
    async def test_embed_preserves_metadata(self, mock_embedding_client, sample_chunks, sample_entities):
        """Test that embedding preserves all original metadata."""
        stage = EmbedStage(embedding_client=mock_embedding_client)

        input_data = {
            "chunks": sample_chunks,
            "entities": sample_entities,
            "relationships": [],
            "entity_map": {},
        }

        result = await stage.execute(input_data)

        chunks = result.data["chunks"]
        entities = result.data["entities"]

        # Verify chunk metadata preserved
        for original, embedded in zip(sample_chunks, chunks):
            assert embedded.id == original.id
            assert embedded.document_id == original.document_id
            assert embedded.text == original.text
            assert embedded.chunk_index == original.chunk_index

        # Verify entity metadata preserved
        for original, embedded in zip(sample_entities, entities):
            assert embedded.id == original.id
            assert embedded.name == original.name
            assert embedded.type == original.type
            assert embedded.description == original.description
