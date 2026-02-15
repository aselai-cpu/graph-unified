# Data Schemas & Models

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document defines all data models, schemas, and constraints for the Graph-Unified system. Every data structure used for storage, transmission, or processing is specified here with types, constraints, and validation rules.

## Core Data Models

### Document Model

Represents a source document before chunking.

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

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

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Document text cannot be empty")
        return v

    @field_validator('metadata')
    @classmethod
    def metadata_serializable(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata contains only JSON-serializable values."""
        import json
        try:
            json.dumps(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Metadata must be JSON-serializable: {e}")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "climate_report_2024.pdf",
                "text": "Global temperatures have risen...",
                "metadata": {
                    "source": "IPCC",
                    "year": 2024,
                    "category": "climate"
                }
            }
        }
```

**Parquet Schema:**
```python
import pyarrow as pa

DOCUMENT_SCHEMA = pa.schema([
    pa.field('id', pa.string()),  # UUID as string
    pa.field('filename', pa.string()),
    pa.field('text', pa.string()),
    pa.field('metadata', pa.string()),  # JSON string
    pa.field('created_at', pa.timestamp('us')),
    pa.field('updated_at', pa.timestamp('us')),
    pa.field('char_count', pa.int32()),
    pa.field('token_count', pa.int32()),
])
```

**Constraints:**
- `id`: Unique, non-null, UUID format
- `filename`: Non-empty, max 512 chars, should be unique but not enforced
- `text`: Non-empty, no max length
- `metadata`: JSON-serializable dict, stored as JSON string in Parquet
- `char_count`: `len(text)`
- `token_count`: Count via tiktoken with model's tokenizer

---

### Chunk Model

Represents a text chunk derived from a document.

```python
from typing import List, Optional
from uuid import UUID, uuid4

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

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('end_char')
    @classmethod
    def end_after_start(cls, v: int, info) -> int:
        if 'start_char' in info.data and v <= info.data['start_char']:
            raise ValueError("end_char must be greater than start_char")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "chunk_index": 0,
                "text": "Global temperatures have risen by 1.2°C...",
                "start_char": 0,
                "end_char": 512,
                "token_count": 128
            }
        }
```

**Parquet Schema:**
```python
CHUNK_SCHEMA = pa.schema([
    pa.field('id', pa.string()),
    pa.field('document_id', pa.string()),
    pa.field('chunk_index', pa.int32()),
    pa.field('text', pa.string()),
    pa.field('start_char', pa.int32()),
    pa.field('end_char', pa.int32()),
    pa.field('token_count', pa.int32()),
    pa.field('metadata', pa.string()),  # JSON string
    # Note: embeddings stored in vector DB, not Parquet
])
```

**Constraints:**
- `id`: Unique, non-null
- `document_id`: Foreign key to Document.id
- `chunk_index`: Sequential within document, 0-indexed
- `text`: Non-empty, typically 512-2048 tokens
- `start_char`, `end_char`: Character positions in original document
- `embedding`: 1536-dimensional float vector (voyage-3) or model-specific
- Composite uniqueness: `(document_id, chunk_index)` must be unique

---

### Entity Model

Represents an extracted named entity.

```python
from enum import Enum
from typing import List, Optional

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

    @field_validator('name')
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Normalize entity name (strip, lowercase for matching)."""
        return v.strip()

    @field_validator('aliases')
    @classmethod
    def unique_aliases(cls, v: List[str]) -> List[str]:
        """Remove duplicate aliases."""
        return list(set(v))

    class Config:
        json_schema_extra = {
            "example": {
                "name": "IPCC",
                "type": "ORGANIZATION",
                "description": "Intergovernmental Panel on Climate Change",
                "source_chunks": ["chunk-uuid-1", "chunk-uuid-2"],
                "extraction_confidence": 0.95,
                "aliases": ["Intergovernmental Panel on Climate Change"]
            }
        }
```

**Parquet Schema:**
```python
ENTITY_SCHEMA = pa.schema([
    pa.field('id', pa.string()),
    pa.field('name', pa.string()),
    pa.field('type', pa.string()),
    pa.field('description', pa.string()),
    pa.field('source_chunks', pa.list_(pa.string())),
    pa.field('extraction_confidence', pa.float32()),
    pa.field('aliases', pa.list_(pa.string())),
    pa.field('metadata', pa.string()),  # JSON string
])
```

**Constraints:**
- `id`: Unique, non-null
- `name`: Non-empty, max 256 chars, case-sensitive but normalized for matching
- `type`: Must be valid EntityType enum value
- `description`: Optional, max 2048 chars
- `source_chunks`: List of chunk IDs where entity was mentioned
- `extraction_confidence`: 0.0-1.0, default 1.0 for manual entries
- Uniqueness: `(name, type)` should be unique (fuzzy matching during extraction)

---

### Relationship Model

Represents a relationship between two entities.

```python
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

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('target_entity_id')
    @classmethod
    def no_self_loops(cls, v: UUID, info) -> UUID:
        """Prevent self-referential relationships."""
        if 'source_entity_id' in info.data and v == info.data['source_entity_id']:
            raise ValueError("Self-loops not allowed (source == target)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "source_entity_id": "entity-uuid-1",
                "target_entity_id": "entity-uuid-2",
                "type": "WORKS_FOR",
                "description": "IPCC scientists work for the organization",
                "source_chunks": ["chunk-uuid-1"],
                "extraction_confidence": 0.90,
                "weight": 1.0
            }
        }
```

**Parquet Schema:**
```python
RELATIONSHIP_SCHEMA = pa.schema([
    pa.field('id', pa.string()),
    pa.field('source_entity_id', pa.string()),
    pa.field('target_entity_id', pa.string()),
    pa.field('type', pa.string()),
    pa.field('description', pa.string()),
    pa.field('source_chunks', pa.list_(pa.string())),
    pa.field('extraction_confidence', pa.float32()),
    pa.field('weight', pa.float32()),
    pa.field('metadata', pa.string()),
])
```

**Constraints:**
- `id`: Unique, non-null
- `source_entity_id`: Foreign key to Entity.id
- `target_entity_id`: Foreign key to Entity.id
- `type`: Must be valid RelationshipType enum value
- `description`: Optional, max 2048 chars
- `source_chunks`: List of chunk IDs where relationship was mentioned
- `weight`: 0.0+, used for graph algorithms (default 1.0)
- No self-loops: `source_entity_id != target_entity_id`

---

### Community Model (GraphRAG-Specific)

Represents a detected community in the entity graph.

```python
class Community(BaseModel):
    """A community detected via Leiden algorithm."""

    id: UUID = Field(default_factory=uuid4)
    level: int = Field(..., ge=0)  # Hierarchical level
    entity_ids: List[UUID] = Field(..., min_length=1)

    # Graph metrics
    size: int = Field(..., ge=1)
    density: float = Field(default=0.0, ge=0.0, le=1.0)

    # Summary (generated by LLM)
    title: Optional[str] = Field(default=None, max_length=256)
    summary: Optional[str] = Field(default=None)
    findings: List[str] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('size')
    @classmethod
    def size_matches_entities(cls, v: int, info) -> int:
        if 'entity_ids' in info.data and v != len(info.data['entity_ids']):
            raise ValueError("size must equal len(entity_ids)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "level": 0,
                "entity_ids": ["entity-1", "entity-2", "entity-3"],
                "size": 3,
                "density": 0.67,
                "title": "Climate Policy Organizations",
                "summary": "A cluster of organizations working on climate policy..."
            }
        }
```

**Parquet Schema:**
```python
COMMUNITY_SCHEMA = pa.schema([
    pa.field('id', pa.string()),
    pa.field('level', pa.int32()),
    pa.field('entity_ids', pa.list_(pa.string())),
    pa.field('size', pa.int32()),
    pa.field('density', pa.float32()),
    pa.field('title', pa.string()),
    pa.field('summary', pa.string()),
    pa.field('findings', pa.list_(pa.string())),
    pa.field('metadata', pa.string()),
])
```

**Constraints:**
- `id`: Unique, non-null
- `level`: Hierarchical level (0 = finest, higher = more abstract)
- `entity_ids`: Non-empty list of entity IDs in community
- `size`: Must equal `len(entity_ids)`
- `density`: Edge density within community (0.0-1.0)
- `title`: Optional, max 256 chars
- `summary`: Optional, generated by LLM from entity descriptions

---

### CommunityReport Model

Represents a detailed summary of a community (GraphRAG global search).

```python
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
    rank: float = Field(default=0.0)  # Importance score

    class Config:
        json_schema_extra = {
            "example": {
                "community_id": "community-uuid-1",
                "title": "Climate Policy Landscape",
                "summary": "Analysis of key climate policy organizations...",
                "full_content": "## Overview\n\nThe climate policy landscape...",
                "findings": [
                    "IPCC leads scientific assessment",
                    "Multiple NGOs coordinate advocacy"
                ],
                "token_count": 512,
                "rank": 8.5
            }
        }
```

**Parquet Schema:**
```python
COMMUNITY_REPORT_SCHEMA = pa.schema([
    pa.field('id', pa.string()),
    pa.field('community_id', pa.string()),
    pa.field('title', pa.string()),
    pa.field('summary', pa.string()),
    pa.field('full_content', pa.string()),
    pa.field('findings', pa.list_(pa.string())),
    pa.field('token_count', pa.int32()),
    pa.field('rank', pa.float32()),
])
```

**Constraints:**
- `id`: Unique, non-null
- `community_id`: Foreign key to Community.id
- `title`: Non-empty, max 256 chars
- `summary`: Non-empty, typically 2-3 sentences
- `full_content`: Detailed markdown-formatted report
- `findings`: List of key insights (typically 3-7 items)
- `rank`: Importance score for map-reduce prioritization

---

## Configuration Schema (settings.yaml)

### Root Schema

```python
from typing import Literal, Union

class SettingsSchema(BaseModel):
    """Root configuration schema."""

    version: str = Field("1.0", pattern=r"^\d+\.\d+$")
    llm: LLMConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    extraction: ExtractionConfig
    strategies: StrategiesConfig
    storage: StorageConfig
    query: QueryConfig
    performance: PerformanceConfig
    logging: LoggingConfig
    prompts: Optional[PromptsConfig] = None
```

### LLM Configuration

```python
class RateLimitConfig(BaseModel):
    requests_per_minute: int = Field(50, ge=1, le=1000)
    tokens_per_minute: int = Field(40000, ge=1000)

class RetryConfig(BaseModel):
    max_attempts: int = Field(3, ge=1, le=10)
    backoff_factor: float = Field(2.0, ge=1.0, le=10.0)

class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai", "azure"] = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    api_key: str = Field(..., min_length=1)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=100, le=200000)
    timeout: int = Field(60, ge=10, le=600)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
```

### Embedding Configuration

```python
class EmbeddingConfig(BaseModel):
    provider: Literal["voyage", "openai", "cohere"] = "voyage"
    model: str = "voyage-3"
    api_key: str = Field(..., min_length=1)
    dimension: int = Field(1024, ge=384, le=4096)
    batch_size: int = Field(128, ge=1, le=512)
    normalize: bool = True
```

### Chunking Configuration

```python
class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "sentence", "paragraph", "semantic"] = "fixed"
    chunk_size: int = Field(512, ge=128, le=4096)
    chunk_overlap: int = Field(64, ge=0, le=512)
    respect_boundaries: bool = True  # Respect sentence/paragraph boundaries
    encoding_name: str = "cl100k_base"  # tiktoken encoding

    @field_validator('chunk_overlap')
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError("chunk_overlap must be < chunk_size")
        return v
```

### Extraction Configuration

```python
class ExtractionConfig(BaseModel):
    entity_types: List[str] = Field(
        default_factory=lambda: [
            "PERSON", "ORGANIZATION", "LOCATION",
            "EVENT", "CONCEPT", "TECHNOLOGY"
        ]
    )
    relationship_types: List[str] = Field(
        default_factory=lambda: [
            "RELATED_TO", "PART_OF", "LOCATED_IN",
            "WORKS_FOR", "CAUSES"
        ]
    )
    max_gleanings: int = Field(1, ge=0, le=3)  # Additional extraction passes
    min_confidence: float = Field(0.7, ge=0.0, le=1.0)
    enable_coreference: bool = False  # Coreference resolution
```

---

## JSON Schemas for API Communication

### Query Request

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["query", "strategy"],
  "properties": {
    "query": {
      "type": "string",
      "minLength": 1,
      "maxLength": 2000
    },
    "strategy": {
      "type": "string",
      "enum": ["naive", "hybrid", "graphrag_local", "graphrag_global", "lightrag", "hipporag"]
    },
    "top_k": {
      "type": "integer",
      "minimum": 1,
      "maximum": 100,
      "default": 10
    },
    "config": {
      "type": "object",
      "additionalProperties": true
    }
  }
}
```

### Query Response

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["response", "contexts", "metadata"],
  "properties": {
    "response": {
      "type": "string"
    },
    "contexts": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["text", "score"],
        "properties": {
          "text": {"type": "string"},
          "score": {"type": "number"},
          "source": {"type": "string"},
          "metadata": {"type": "object"}
        }
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "strategy": {"type": "string"},
        "latency_ms": {"type": "integer"},
        "tokens_used": {"type": "integer"},
        "cost_usd": {"type": "number"}
      }
    }
  }
}
```

### Entity Extraction Output (LLM Response)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["entities", "relationships"],
  "properties": {
    "entities": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
          "name": {"type": "string"},
          "type": {"type": "string"},
          "description": {"type": "string"}
        }
      }
    },
    "relationships": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["source", "target", "type"],
        "properties": {
          "source": {"type": "string"},
          "target": {"type": "string"},
          "type": {"type": "string"},
          "description": {"type": "string"}
        }
      }
    }
  }
}
```

---

## Validation Rules

### Cross-Model Constraints

1. **Referential Integrity:**
   - `Chunk.document_id` must reference existing `Document.id`
   - `Entity.source_chunks` must reference existing `Chunk.id` values
   - `Relationship.source_entity_id` and `target_entity_id` must reference existing `Entity.id`
   - `Community.entity_ids` must reference existing `Entity.id` values
   - `CommunityReport.community_id` must reference existing `Community.id`

2. **Uniqueness Constraints:**
   - `Document.id` unique across corpus
   - `Chunk.id` unique across corpus
   - `(Chunk.document_id, Chunk.chunk_index)` unique
   - `Entity.id` unique across corpus
   - `(Entity.name, Entity.type)` should be deduplicated during extraction
   - `Relationship.id` unique across corpus

3. **Cardinality Constraints:**
   - One document can have many chunks (1:N)
   - One chunk can mention many entities (M:N)
   - One entity can appear in many chunks (M:N)
   - Two entities can have at most one directed relationship per type (enforce or merge)
   - One community can have many entities (1:N)
   - One community has exactly one report (1:1)

### Data Quality Rules

1. **Text Quality:**
   - Documents must have non-empty text after stripping whitespace
   - Chunks must have `token_count > 0`
   - Entity descriptions should be complete sentences

2. **Confidence Thresholds:**
   - Entities with `extraction_confidence < min_confidence` should be filtered
   - Relationships with `extraction_confidence < min_confidence` should be filtered

3. **Graph Validity:**
   - No self-loops in relationships
   - Graph should be weakly connected (warn if disconnected components)
   - Community detection should produce non-empty communities

---

## Storage Migration Schema

For tracking schema versions and migrations.

```python
class SchemaVersion(BaseModel):
    version: str = Field(..., pattern=r"^\d+\.\d+$")
    applied_at: datetime = Field(default_factory=datetime.utcnow)
    migrations: List[str] = Field(default_factory=list)
```

**Migration tracking file:** `output/.schema_version.json`

---

## Summary

This specification defines:

- **6 core data models:** Document, Chunk, Entity, Relationship, Community, CommunityReport
- **Pydantic schemas** with validation for Python API
- **PyArrow schemas** for Parquet storage
- **JSON schemas** for API communication
- **Configuration schema** for settings.yaml
- **Validation rules** for data quality and integrity
- **Migration schema** for versioning

All models use:
- UUIDs for identifiers
- Pydantic for validation
- Parquet for storage
- JSON for serialization
- Type hints for clarity

**Next Steps:**
- Implement Pydantic models in `graphunified/config/models.py`
- Create Parquet schema constants in `graphunified/storage/schemas.py`
- Build validation utilities in `graphunified/utils/validation.py`
