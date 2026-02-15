"""Core data models for graph-unified using Pydantic v2."""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    """A source document in the corpus."""

    id: UUID = Field(default_factory=uuid4)
    filename: str = Field(..., min_length=1, max_length=512)
    text: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Computed fields
    char_count: int = Field(default=0)
    token_count: int = Field(default=0)

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Document text cannot be empty")
        return v

    @field_validator("metadata")
    @classmethod
    def metadata_serializable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata contains only JSON-serializable values."""
        try:
            json.dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON-serializable: {e}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "climate_report_2024.pdf",
                "text": "Global temperatures have risen...",
                "metadata": {"source": "IPCC", "year": 2024, "category": "climate"},
            }
        }
    }


class Chunk(BaseModel):
    """A text chunk with metadata and embedding."""

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID = Field(...)
    chunk_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=1)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)
    token_count: int = Field(..., ge=0)

    # Embedding (stored separately in vector DB, kept as list here for API)
    embedding: Optional[List[float]] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)

    # Graph connections (for bidirectional traversal in LightRAG, HippoRAG)
    entity_ids: List[UUID] = Field(default_factory=list)
    relationship_ids: List[UUID] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("end_char")
    @classmethod
    def end_after_start(cls, v: int, info: Any) -> int:
        if "start_char" in info.data and v <= info.data["start_char"]:
            raise ValueError("end_char must be greater than start_char")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "chunk_index": 0,
                "text": "Global temperatures have risen by 1.2°C...",
                "start_char": 0,
                "end_char": 512,
                "token_count": 128,
            }
        }
    }


class EntityType(str, Enum):
    """Standard entity types (extensible)."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    TECHNOLOGY = "TECHNOLOGY"
    DATE = "DATE"
    OTHER = "OTHER"


class Entity(BaseModel):
    """An extracted entity with metadata."""

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=256)
    type: EntityType = Field(...)
    description: Optional[str] = Field(default=None, max_length=2048)

    # Source tracking
    source_chunks: List[UUID] = Field(default_factory=list)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Embedding (for entity-centric retrieval)
    embedding: Optional[List[float]] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)

    # Metadata
    aliases: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Normalize entity name (strip, lowercase for matching)."""
        return v.strip()

    @field_validator("aliases")
    @classmethod
    def unique_aliases(cls, v: List[str]) -> List[str]:
        """Remove duplicate aliases."""
        return list(set(v))

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "IPCC",
                "type": "ORGANIZATION",
                "description": "Intergovernmental Panel on Climate Change",
                "source_chunks": ["chunk-uuid-1", "chunk-uuid-2"],
                "extraction_confidence": 0.95,
                "aliases": ["Intergovernmental Panel on Climate Change"],
            }
        }
    }


class RelationshipType(str, Enum):
    """Standard relationship types (extensible)."""

    RELATED_TO = "RELATED_TO"
    PART_OF = "PART_OF"
    LOCATED_IN = "LOCATED_IN"
    WORKS_FOR = "WORKS_FOR"
    CAUSES = "CAUSES"
    SIMILAR_TO = "SIMILAR_TO"
    OPPOSITE_OF = "OPPOSITE_OF"
    PRECEDES = "PRECEDES"
    OTHER = "OTHER"


class Relationship(BaseModel):
    """A directed relationship between entities."""

    id: UUID = Field(default_factory=uuid4)
    source_entity_id: UUID = Field(...)
    target_entity_id: UUID = Field(...)
    type: RelationshipType = Field(...)
    description: Optional[str] = Field(default=None, max_length=2048)

    # Source tracking
    source_chunks: List[UUID] = Field(default_factory=list)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Graph properties
    weight: float = Field(default=1.0, ge=0.0)

    # Embedding (for LightRAG global search - relationship-centric retrieval)
    embedding: Optional[List[float]] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_entity_id")
    @classmethod
    def no_self_loops(cls, v: UUID, info: Any) -> UUID:
        """Prevent self-referential relationships."""
        if "source_entity_id" in info.data and v == info.data["source_entity_id"]:
            raise ValueError("Self-loops not allowed (source == target)")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "source_entity_id": "entity-uuid-1",
                "target_entity_id": "entity-uuid-2",
                "type": "WORKS_FOR",
                "description": "IPCC scientists work for the organization",
                "source_chunks": ["chunk-uuid-1"],
                "extraction_confidence": 0.90,
                "weight": 1.0,
            }
        }
    }


class Community(BaseModel):
    """A community detected via Leiden algorithm."""

    id: UUID = Field(default_factory=uuid4)
    level: int = Field(..., ge=0)  # Hierarchical level
    entity_ids: List[UUID] = Field(..., min_length=1)

    # Hierarchical structure (for multi-level GraphRAG)
    parent_community_id: Optional[UUID] = Field(default=None)
    child_community_ids: List[UUID] = Field(default_factory=list)

    # Graph structure
    relationship_ids: List[UUID] = Field(default_factory=list)

    # Graph metrics
    size: int = Field(..., ge=1)
    density: float = Field(default=0.0, ge=0.0, le=1.0)

    # Summary (generated by LLM)
    title: Optional[str] = Field(default=None, max_length=256)
    summary: Optional[str] = Field(default=None)
    findings: List[str] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("size")
    @classmethod
    def size_matches_entities(cls, v: int, info: Any) -> int:
        if "entity_ids" in info.data and v != len(info.data["entity_ids"]):
            raise ValueError("size must equal len(entity_ids)")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "level": 0,
                "entity_ids": ["entity-1", "entity-2", "entity-3"],
                "size": 3,
                "density": 0.67,
                "title": "Climate Policy Organizations",
                "summary": "A cluster of organizations working on climate policy...",
            }
        }
    }


class CommunityReport(BaseModel):
    """Detailed LLM-generated report for a community."""

    id: UUID = Field(default_factory=uuid4)
    community_id: UUID = Field(...)

    # Report content
    title: str = Field(..., min_length=1, max_length=256)
    summary: str = Field(..., min_length=1)
    full_content: str = Field(...)
    findings: List[str] = Field(default_factory=list)

    # Report metadata
    token_count: int = Field(..., ge=0)
    rank: float = Field(default=0.0)  # Importance score for map-reduce

    # Embedding (for GraphRAG global search retrieval)
    embedding: Optional[List[float]] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "community_id": "community-uuid-1",
                "title": "Climate Policy Landscape",
                "summary": "Analysis of key climate policy organizations...",
                "full_content": "## Overview\n\nThe climate policy landscape...",
                "findings": [
                    "IPCC leads scientific assessment",
                    "Multiple NGOs coordinate advocacy",
                ],
                "token_count": 512,
                "rank": 8.5,
            }
        }
    }


class Fact(BaseModel):
    """A subject-predicate-object triple for HippoRAG fact retrieval."""

    id: UUID = Field(default_factory=uuid4)
    subject: str = Field(..., min_length=1, max_length=256)
    predicate: str = Field(..., min_length=1, max_length=256)
    object: str = Field(..., min_length=1, max_length=256)

    # Source tracking
    source_chunk: UUID = Field(...)
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Embedding (for fact-based retrieval in HippoRAG Stage 1)
    embedding: Optional[List[float]] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)

    # Entity links (for graph activation)
    entity_ids: List[UUID] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "json_schema_extra": {
            "example": {
                "subject": "IPCC",
                "predicate": "published",
                "object": "Assessment Report",
                "source_chunk": "chunk-uuid-1",
                "extraction_confidence": 0.92,
                "entity_ids": ["entity-uuid-1", "entity-uuid-2"],
            }
        }
    }


class EntityChunkEdge(BaseModel):
    """Weighted edge between entity and chunk nodes for HippoRAG bipartite graph."""

    entity_id: UUID = Field(...)
    chunk_id: UUID = Field(...)

    # Edge properties
    weight: float = Field(default=1.0, ge=0.0)
    mention_count: int = Field(default=1, ge=1)
    first_position: int = Field(default=0, ge=0)  # Character position of first mention

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("entity_id")
    @classmethod
    def validate_ids_different(cls, v: UUID, info: Any) -> UUID:
        """Ensure entity and chunk are different (no self-edges)."""
        if "chunk_id" in info.data and v == info.data["chunk_id"]:
            raise ValueError("Entity ID cannot equal Chunk ID")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "entity_id": "entity-uuid-1",
                "chunk_id": "chunk-uuid-1",
                "weight": 0.85,
                "mention_count": 3,
                "first_position": 42,
            }
        }
    }
