"""Shared test fixtures for graph-unified tests."""

import tempfile
from pathlib import Path
from typing import Generator
from uuid import uuid4

import pytest

from graphunified.config.models import Chunk, Document, Entity, EntityType, Relationship, RelationshipType
from graphunified.config.settings import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    Settings,
    StorageConfig,
)


@pytest.fixture
def tmp_storage_dir() -> Generator[Path, None, None]:
    """Temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_document() -> Document:
    """Sample document for testing."""
    return Document(
        id=uuid4(),
        filename="test_doc.txt",
        text="This is a test document about climate change and global warming.",
        metadata={"source": "test", "category": "science"},
        char_count=65,
        token_count=15,
    )


@pytest.fixture
def sample_chunks(sample_document: Document) -> list[Chunk]:
    """Sample chunks for testing."""
    return [
        Chunk(
            id=uuid4(),
            document_id=sample_document.id,
            chunk_index=0,
            text="This is a test document",
            start_char=0,
            end_char=23,
            token_count=6,
        ),
        Chunk(
            id=uuid4(),
            document_id=sample_document.id,
            chunk_index=1,
            text="about climate change",
            start_char=24,
            end_char=44,
            token_count=4,
        ),
        Chunk(
            id=uuid4(),
            document_id=sample_document.id,
            chunk_index=2,
            text="and global warming.",
            start_char=45,
            end_char=65,
            token_count=5,
        ),
    ]


@pytest.fixture
def sample_entities(sample_chunks: list[Chunk]) -> list[Entity]:
    """Sample entities for testing."""
    return [
        Entity(
            id=uuid4(),
            name="climate change",
            type=EntityType.CONCEPT,
            description="Long-term shifts in temperatures and weather patterns",
            source_chunks=[sample_chunks[1].id],
            extraction_confidence=0.95,
        ),
        Entity(
            id=uuid4(),
            name="global warming",
            type=EntityType.CONCEPT,
            description="Increase in Earth's average surface temperature",
            source_chunks=[sample_chunks[2].id],
            extraction_confidence=0.90,
        ),
    ]


@pytest.fixture
def sample_relationship(sample_entities: list[Entity], sample_chunks: list[Chunk]) -> Relationship:
    """Sample relationship for testing."""
    return Relationship(
        id=uuid4(),
        source_entity_id=sample_entities[0].id,
        target_entity_id=sample_entities[1].id,
        type=RelationshipType.RELATED_TO,
        description="Climate change is closely related to global warming",
        source_chunks=[sample_chunks[1].id, sample_chunks[2].id],
        extraction_confidence=0.85,
    )


@pytest.fixture
def sample_llm_config() -> LLMConfig:
    """Sample LLM configuration for testing."""
    return LLMConfig(
        provider="anthropic",
        model="claude-3-haiku-20240307",
        api_key="test-key-12345",
        temperature=0.0,
        max_tokens=1000,
    )


@pytest.fixture
def sample_embedding_config() -> EmbeddingConfig:
    """Sample embedding configuration for testing."""
    return EmbeddingConfig(
        provider="voyage",
        model="voyage-3",
        api_key="test-key-67890",
        dimension=1024,
        batch_size=64,
    )


@pytest.fixture
def sample_settings(tmp_storage_dir: Path, sample_llm_config: LLMConfig, sample_embedding_config: EmbeddingConfig) -> Settings:
    """Sample settings for testing."""
    return Settings(
        version="1.0",
        llm=sample_llm_config,
        embedding=sample_embedding_config,
        storage=StorageConfig(root_dir=tmp_storage_dir),
    )


@pytest.fixture
def sample_config_yaml(tmp_storage_dir: Path) -> Path:
    """Sample configuration YAML file for testing."""
    config_file = tmp_storage_dir / "test-settings.yaml"
    config_content = """
version: "1.0"

llm:
  provider: "anthropic"
  model: "claude-3-haiku-20240307"
  api_key: "test-key-12345"
  temperature: 0.0
  max_tokens: 1000

embedding:
  provider: "voyage"
  model: "voyage-3"
  api_key: "test-key-67890"
  dimension: 1024
  batch_size: 64
"""
    config_file.write_text(config_content)
    return config_file
