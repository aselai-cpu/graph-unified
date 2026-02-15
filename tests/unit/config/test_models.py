"""Tests for data models."""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from graphunified.config.models import (
    Chunk,
    Document,
    Entity,
    EntityType,
    Relationship,
    RelationshipType,
)


class TestDocument:
    """Tests for Document model."""

    def test_valid_document(self, sample_document):
        """Test creating a valid document."""
        assert sample_document.filename == "test_doc.txt"
        assert len(sample_document.text) > 0
        assert sample_document.char_count == 65
        assert sample_document.token_count == 15

    def test_empty_text_fails(self):
        """Test that empty text raises validation error."""
        with pytest.raises(ValidationError):
            Document(
                filename="test.txt",
                text="",
                metadata={},
            )

    def test_metadata_serializable(self):
        """Test that metadata must be JSON-serializable."""
        # Valid metadata
        doc = Document(
            filename="test.txt",
            text="Valid text",
            metadata={"key": "value", "number": 42, "list": [1, 2, 3]},
        )
        assert doc.metadata["key"] == "value"

        # Invalid metadata (contains non-serializable object)
        with pytest.raises(ValidationError):
            Document(
                filename="test.txt",
                text="Valid text",
                metadata={"function": lambda x: x},
            )


class TestChunk:
    """Tests for Chunk model."""

    def test_valid_chunk(self, sample_chunks):
        """Test creating valid chunks."""
        chunk = sample_chunks[0]
        assert chunk.chunk_index == 0
        assert chunk.text == "This is a test document"
        assert chunk.start_char < chunk.end_char
        assert chunk.token_count > 0

    def test_end_after_start_validation(self, sample_document):
        """Test that end_char must be greater than start_char."""
        with pytest.raises(ValidationError):
            Chunk(
                document_id=sample_document.id,
                chunk_index=0,
                text="Test",
                start_char=100,
                end_char=50,  # Invalid: end < start
                token_count=10,
            )


class TestEntity:
    """Tests for Entity model."""

    def test_valid_entity(self, sample_entities):
        """Test creating a valid entity."""
        entity = sample_entities[0]
        assert entity.name == "climate change"
        assert entity.type == EntityType.CONCEPT
        assert entity.extraction_confidence == 0.95

    def test_name_normalized(self):
        """Test that entity name is stripped."""
        entity = Entity(
            name="  Test Entity  ",
            type=EntityType.PERSON,
        )
        assert entity.name == "Test Entity"

    def test_unique_aliases(self):
        """Test that duplicate aliases are removed."""
        entity = Entity(
            name="Test",
            type=EntityType.PERSON,
            aliases=["Alias1", "Alias2", "Alias1", "Alias2"],
        )
        assert len(entity.aliases) == 2


class TestRelationship:
    """Tests for Relationship model."""

    def test_valid_relationship(self, sample_relationship):
        """Test creating a valid relationship."""
        assert sample_relationship.type == RelationshipType.RELATED_TO
        assert sample_relationship.extraction_confidence == 0.85
        assert sample_relationship.weight == 1.0

    def test_no_self_loops(self, sample_entities):
        """Test that self-loops are not allowed."""
        entity_id = sample_entities[0].id

        with pytest.raises(ValidationError):
            Relationship(
                source_entity_id=entity_id,
                target_entity_id=entity_id,  # Same as source
                type=RelationshipType.RELATED_TO,
            )
