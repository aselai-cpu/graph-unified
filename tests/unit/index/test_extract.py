"""Unit tests for extraction stage."""

import json
from uuid import uuid4

import pytest

from graphunified.config.models import Chunk, Entity, EntityType, Relationship, RelationshipType
from graphunified.index.stages.extract import ExtractStage


@pytest.mark.unit
class TestExtractStage:
    """Tests for ExtractStage."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample chunks."""
        doc_id = uuid4()
        return [
            Chunk(
                id=uuid4(),
                document_id=doc_id,
                chunk_index=0,
                text="Dr. Jane Smith works for NASA in Washington DC. She leads the Mars exploration program.",
                start_char=0,
                end_char=100,
                token_count=20,
            ),
            Chunk(
                id=uuid4(),
                document_id=doc_id,
                chunk_index=1,
                text="NASA is part of the United States government. It was founded in 1958.",
                start_char=100,
                end_char=180,
                token_count=18,
            ),
        ]

    @pytest.fixture
    def mock_llm_client(self, mocker):
        """Create a mock LLM client."""
        client = mocker.Mock()

        # Mock entity extraction response
        entity_response = json.dumps(
            {
                "entities": [
                    {
                        "name": "Jane Smith",
                        "type": "PERSON",
                        "description": "A scientist at NASA",
                        "confidence": 0.95,
                    },
                    {
                        "name": "NASA",
                        "type": "ORGANIZATION",
                        "description": "National Aeronautics and Space Administration",
                        "confidence": 0.99,
                    },
                    {
                        "name": "Washington DC",
                        "type": "LOCATION",
                        "description": "Capital of the United States",
                        "confidence": 0.90,
                    },
                    {
                        "name": "United States government",
                        "type": "ORGANIZATION",
                        "description": "The federal government of the USA",
                        "confidence": 0.92,
                    },
                ]
            }
        )

        # Mock relationship extraction response
        relationship_response = json.dumps(
            {
                "relationships": [
                    {
                        "source": "Jane Smith",
                        "target": "NASA",
                        "type": "WORKS_FOR",
                        "description": "Jane Smith works at NASA",
                        "confidence": 0.95,
                    },
                    {
                        "source": "NASA",
                        "target": "Washington DC",
                        "type": "LOCATED_IN",
                        "description": "NASA is located in Washington DC",
                        "confidence": 0.85,
                    },
                    {
                        "source": "NASA",
                        "target": "United States government",
                        "type": "PART_OF",
                        "description": "NASA is part of the US government",
                        "confidence": 0.90,
                    },
                ]
            }
        )

        # Configure mock to return different responses for different calls
        client.generate = mocker.AsyncMock(
            side_effect=[
                (entity_response, 100, 200),  # First call: entities
                (relationship_response, 150, 180),  # Second call: relationships
            ]
        )

        return client

    @pytest.mark.asyncio
    async def test_extract_entities_and_relationships(self, mock_llm_client, sample_chunks):
        """Test extracting entities and relationships."""
        stage = ExtractStage(llm_client=mock_llm_client, batch_size=10)

        result = await stage.execute(sample_chunks)

        assert result.status.value == "completed"
        assert "entities" in result.data
        assert "relationships" in result.data

        entities = result.data["entities"]
        relationships = result.data["relationships"]

        assert len(entities) > 0
        assert len(relationships) > 0

        # Verify entity types
        for entity in entities:
            assert isinstance(entity, Entity)
            assert entity.name
            assert entity.type in EntityType

        # Verify relationship types
        for rel in relationships:
            assert isinstance(rel, Relationship)
            assert rel.type in RelationshipType

    @pytest.mark.asyncio
    async def test_extract_empty_chunks(self, mock_llm_client):
        """Test extraction with empty chunk list."""
        stage = ExtractStage(llm_client=mock_llm_client)

        result = await stage.execute([])

        assert result.status.value == "completed"
        assert result.data["entities"] == []
        assert result.data["relationships"] == []

    @pytest.mark.asyncio
    async def test_entity_deduplication(self, mock_llm_client):
        """Test entity deduplication."""
        # Create mock with duplicate entities
        duplicate_response = json.dumps(
            {
                "entities": [
                    {"name": "NASA", "type": "ORGANIZATION", "description": "Space agency", "confidence": 0.95},
                    {"name": "N.A.S.A.", "type": "ORGANIZATION", "description": "Space agency", "confidence": 0.90},
                    {
                        "name": "National Aeronautics and Space Administration",
                        "type": "ORGANIZATION",
                        "description": "Full name",
                        "confidence": 0.85,
                    },
                ]
            }
        )

        mock_llm_client.generate = pytest.Mock(
            return_value=(duplicate_response, 100, 200)
        )
        mock_llm_client.generate = pytest.Mock(
            side_effect=[
                (duplicate_response, 100, 200),
                (json.dumps({"relationships": []}), 100, 50),
            ]
        )

        stage = ExtractStage(llm_client=mock_llm_client, dedup_threshold=80)

        chunk = Chunk(
            id=uuid4(),
            document_id=uuid4(),
            chunk_index=0,
            text="NASA and N.A.S.A. are the same.",
            start_char=0,
            end_char=40,
            token_count=10,
        )

        result = await stage.execute([chunk])

        entities = result.data["entities"]

        # Should deduplicate similar entity names
        # With threshold 80, "NASA" and "N.A.S.A." should merge
        assert len(entities) < 3  # Fewer than original 3

    @pytest.mark.asyncio
    async def test_relationship_resolution(self, mock_llm_client, sample_chunks):
        """Test that relationships reference deduplicated entity IDs."""
        stage = ExtractStage(llm_client=mock_llm_client)

        result = await stage.execute(sample_chunks)

        entities = result.data["entities"]
        relationships = result.data["relationships"]
        entity_map = result.data["entity_map"]

        # Verify all relationship entity IDs exist in final entity list
        entity_ids = {e.id for e in entities}

        for rel in relationships:
            assert rel.source_entity_id in entity_ids, "Relationship source must reference existing entity"
            assert rel.target_entity_id in entity_ids, "Relationship target must reference existing entity"

    @pytest.mark.asyncio
    async def test_json_parsing_with_markdown(self, mock_llm_client, sample_chunks):
        """Test JSON parsing handles markdown code blocks."""
        # Mock response with markdown code blocks
        markdown_response = """```json
{
  "entities": [
    {
      "name": "Test Entity",
      "type": "CONCEPT",
      "description": "A test",
      "confidence": 0.9
    }
  ]
}
```"""

        mock_llm_client.generate = pytest.Mock(
            side_effect=[
                (markdown_response, 100, 200),
                (json.dumps({"relationships": []}), 100, 50),
            ]
        )

        stage = ExtractStage(llm_client=mock_llm_client)

        result = await stage.execute(sample_chunks)

        assert result.status.value == "completed"
        assert len(result.data["entities"]) > 0
