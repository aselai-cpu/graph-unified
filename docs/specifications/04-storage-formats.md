# Storage Format Specifications

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document specifies all storage formats, file structures, schemas, and persistence strategies for Graph-Unified. It defines how data is stored on disk, versioned, and migrated.

## Storage Architecture

### Directory Structure

```
output/
├── .schema_version.json              # Schema version tracking
├── documents.parquet                 # Source documents
├── chunks.parquet                    # Text chunks
├── entities.parquet                  # Extracted entities
├── relationships.parquet             # Entity relationships
├── communities.parquet               # Graph communities (GraphRAG)
├── community_reports.parquet         # Community summaries (GraphRAG)
├── vectors/                          # Vector indexes
│   ├── chunks.lance/                 # LanceDB: chunk embeddings
│   └── entities.lance/               # LanceDB: entity embeddings
├── graphs/                           # Graph structures
│   ├── entity_graph.graphml          # NetworkX: entity graph
│   └── hipporag_graph.graphml        # HippoRAG: associative graph
├── indexes/                          # Auxiliary indexes
│   ├── bm25_index.pkl                # BM25 sparse index
│   └── lightrag_indexes.pkl          # LightRAG dual indexes
└── metadata/
    ├── indexing_stats.json           # Indexing statistics
    └── extraction_config.json        # Config used for extraction
```

---

## Parquet Schemas

### documents.parquet

**Purpose:** Store source documents before chunking.

**PyArrow Schema:**

```python
import pyarrow as pa

DOCUMENTS_SCHEMA = pa.schema([
    pa.field('id', pa.string(), nullable=False),
    pa.field('filename', pa.string(), nullable=False),
    pa.field('text', pa.string(), nullable=False),
    pa.field('metadata', pa.string(), nullable=True),  # JSON string
    pa.field('created_at', pa.timestamp('us'), nullable=False),
    pa.field('updated_at', pa.timestamp('us'), nullable=False),
    pa.field('char_count', pa.int32(), nullable=False),
    pa.field('token_count', pa.int32(), nullable=False),
])
```

**Example Row:**

```python
{
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "climate_report_2024.pdf",
    "text": "Global temperatures have risen by 1.2°C since pre-industrial times...",
    "metadata": '{"source": "IPCC", "year": 2024, "category": "climate"}',
    "created_at": "2026-02-15T10:30:00",
    "updated_at": "2026-02-15T10:30:00",
    "char_count": 45230,
    "token_count": 11200
}
```

**Constraints:**
- `id`: Unique, UUID format
- `filename`: Non-empty, typically unique
- `text`: Non-empty
- `metadata`: JSON-serialized dict
- `char_count`: Equals `len(text)`
- `token_count`: Token count using configured tokenizer

**Partitioning:** None (small corpus <10K docs)

**Compression:** Snappy (default)

---

### chunks.parquet

**Purpose:** Store text chunks with metadata.

**PyArrow Schema:**

```python
CHUNKS_SCHEMA = pa.schema([
    pa.field('id', pa.string(), nullable=False),
    pa.field('document_id', pa.string(), nullable=False),
    pa.field('chunk_index', pa.int32(), nullable=False),
    pa.field('text', pa.string(), nullable=False),
    pa.field('start_char', pa.int32(), nullable=False),
    pa.field('end_char', pa.int32(), nullable=False),
    pa.field('token_count', pa.int32(), nullable=False),
    pa.field('metadata', pa.string(), nullable=True),
])
```

**Example Row:**

```python
{
    "id": "chunk-uuid-1",
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "chunk_index": 0,
    "text": "Global temperatures have risen by 1.2°C since pre-industrial times. This warming is primarily driven by anthropogenic greenhouse gas emissions.",
    "start_char": 0,
    "end_char": 142,
    "token_count": 32,
    "metadata": '{"section": "introduction"}'
}
```

**Constraints:**
- `id`: Unique, UUID format
- `document_id`: Foreign key to documents.id
- `chunk_index`: Sequential within document, 0-indexed
- `text`: Non-empty
- `(document_id, chunk_index)`: Composite unique key

**Indexes:**
- Sorted by `(document_id, chunk_index)` for efficient document retrieval

**Note:** Embeddings stored separately in LanceDB, not in Parquet.

---

### entities.parquet

**Purpose:** Store extracted entities.

**PyArrow Schema:**

```python
ENTITIES_SCHEMA = pa.schema([
    pa.field('id', pa.string(), nullable=False),
    pa.field('name', pa.string(), nullable=False),
    pa.field('type', pa.string(), nullable=False),
    pa.field('description', pa.string(), nullable=True),
    pa.field('source_chunks', pa.list_(pa.string()), nullable=False),
    pa.field('extraction_confidence', pa.float32(), nullable=False),
    pa.field('aliases', pa.list_(pa.string()), nullable=True),
    pa.field('metadata', pa.string(), nullable=True),
])
```

**Example Row:**

```python
{
    "id": "entity-uuid-1",
    "name": "IPCC",
    "type": "ORGANIZATION",
    "description": "Intergovernmental Panel on Climate Change, UN body for assessing climate science",
    "source_chunks": ["chunk-uuid-1", "chunk-uuid-5", "chunk-uuid-12"],
    "extraction_confidence": 0.95,
    "aliases": ["Intergovernmental Panel on Climate Change"],
    "metadata": '{"founded": 1988, "headquarters": "Geneva"}'
}
```

**Constraints:**
- `id`: Unique, UUID format
- `name`: Non-empty, normalized (strip whitespace)
- `type`: Valid EntityType enum value
- `source_chunks`: List of chunk IDs (foreign keys)
- `extraction_confidence`: 0.0-1.0

**Indexes:**
- B-tree index on `name` for lookups
- Sorted by `type` for type-filtered queries

**Deduplication:**
- Fuzzy matching on `(name, type)` during extraction
- Merge entities with Levenshtein distance < 2 and same type

---

### relationships.parquet

**Purpose:** Store entity relationships.

**PyArrow Schema:**

```python
RELATIONSHIPS_SCHEMA = pa.schema([
    pa.field('id', pa.string(), nullable=False),
    pa.field('source_entity_id', pa.string(), nullable=False),
    pa.field('target_entity_id', pa.string(), nullable=False),
    pa.field('type', pa.string(), nullable=False),
    pa.field('description', pa.string(), nullable=True),
    pa.field('source_chunks', pa.list_(pa.string()), nullable=False),
    pa.field('extraction_confidence', pa.float32(), nullable=False),
    pa.field('weight', pa.float32(), nullable=False),
    pa.field('metadata', pa.string(), nullable=True),
])
```

**Example Row:**

```python
{
    "id": "rel-uuid-1",
    "source_entity_id": "entity-uuid-1",
    "target_entity_id": "entity-uuid-2",
    "type": "PUBLISHED",
    "description": "IPCC published the Climate Report 2024",
    "source_chunks": ["chunk-uuid-5"],
    "extraction_confidence": 0.90,
    "weight": 1.0,
    "metadata": '{"year": 2024}'
}
```

**Constraints:**
- `id`: Unique, UUID format
- `source_entity_id`, `target_entity_id`: Foreign keys to entities.id
- `type`: Valid RelationshipType enum value
- `source_entity_id != target_entity_id` (no self-loops)
- `weight`: 0.0+, default 1.0

**Indexes:**
- Sorted by `source_entity_id` for graph traversal
- Index on `type` for relationship filtering

---

### communities.parquet

**Purpose:** Store graph communities (GraphRAG).

**PyArrow Schema:**

```python
COMMUNITIES_SCHEMA = pa.schema([
    pa.field('id', pa.string(), nullable=False),
    pa.field('level', pa.int32(), nullable=False),
    pa.field('entity_ids', pa.list_(pa.string()), nullable=False),
    pa.field('size', pa.int32(), nullable=False),
    pa.field('density', pa.float32(), nullable=False),
    pa.field('title', pa.string(), nullable=True),
    pa.field('summary', pa.string(), nullable=True),
    pa.field('findings', pa.list_(pa.string()), nullable=True),
    pa.field('metadata', pa.string(), nullable=True),
])
```

**Example Row:**

```python
{
    "id": "community-uuid-1",
    "level": 0,
    "entity_ids": ["entity-1", "entity-2", "entity-3", "entity-4"],
    "size": 4,
    "density": 0.67,
    "title": "Climate Policy Organizations",
    "summary": "A cluster of international organizations focused on climate policy and science assessment.",
    "findings": [
        "IPCC leads scientific assessment",
        "Multiple NGOs coordinate advocacy",
        "Strong collaboration network"
    ],
    "metadata": '{"detection_algorithm": "leiden", "resolution": 1.0}'
}
```

**Constraints:**
- `id`: Unique, UUID format
- `level`: Hierarchical level (0 = finest granularity)
- `entity_ids`: Non-empty list
- `size`: Equals `len(entity_ids)`
- `density`: 0.0-1.0 (edge density within community)

**Notes:**
- Generated only for GraphRAG strategies
- Leiden algorithm used for detection

---

### community_reports.parquet

**Purpose:** Store detailed community summaries (GraphRAG global).

**PyArrow Schema:**

```python
COMMUNITY_REPORTS_SCHEMA = pa.schema([
    pa.field('id', pa.string(), nullable=False),
    pa.field('community_id', pa.string(), nullable=False),
    pa.field('title', pa.string(), nullable=False),
    pa.field('summary', pa.string(), nullable=False),
    pa.field('full_content', pa.string(), nullable=False),
    pa.field('findings', pa.list_(pa.string()), nullable=True),
    pa.field('token_count', pa.int32(), nullable=False),
    pa.field('rank', pa.float32(), nullable=False),
])
```

**Example Row:**

```python
{
    "id": "report-uuid-1",
    "community_id": "community-uuid-1",
    "title": "Climate Policy Ecosystem",
    "summary": "Analysis of key climate policy organizations and their collaborative networks.",
    "full_content": "## Overview\n\nThe climate policy landscape is dominated by...",
    "findings": [
        "IPCC serves as the primary scientific authority",
        "NGO networks facilitate knowledge transfer",
        "Policy advocacy is coordinated across regions"
    ],
    "token_count": 512,
    "rank": 8.5
}
```

**Constraints:**
- `id`: Unique, UUID format
- `community_id`: Foreign key to communities.id
- `title`: Non-empty, max 256 chars
- `full_content`: Markdown-formatted report
- `rank`: Importance score for map-reduce (0.0+)

---

## Vector Indexes

### LanceDB Format

**Technology:** LanceDB (columnar vector database)

**Storage:** `output/vectors/<table_name>.lance/`

**Tables:**

#### chunks.lance

**Purpose:** Chunk embeddings for vector search

**Schema:**

```python
{
    "id": "string",                    # Chunk UUID
    "embedding": "vector(1024)",       # Float vector (dimension from config)
    "document_id": "string",
    "chunk_index": "int",
    "metadata": "json"
}
```

**Index Type:** IVF_FLAT (Inverted File with flat quantization)

**Distance Metric:** Cosine similarity

**Example Query:**

```python
import lancedb

db = lancedb.connect("output/vectors")
table = db.open_table("chunks")

results = table.search(query_vector) \
    .metric("cosine") \
    .limit(10) \
    .to_list()
```

#### entities.lance

**Purpose:** Entity embeddings for entity-centric retrieval

**Schema:**

```python
{
    "id": "string",                    # Entity UUID
    "embedding": "vector(1024)",
    "name": "string",
    "type": "string",
    "metadata": "json"
}
```

**Used By:** GraphRAG Local, LightRAG, HippoRAG

---

## Graph Formats

### NetworkX GraphML

**Technology:** NetworkX with GraphML serialization

**Storage:** `output/graphs/<graph_name>.graphml`

**Formats:**

#### entity_graph.graphml

**Purpose:** Full entity relationship graph

**Node Attributes:**

```xml
<node id="entity-uuid-1">
  <data key="name">IPCC</data>
  <data key="type">ORGANIZATION</data>
  <data key="description">Intergovernmental Panel on Climate Change</data>
</node>
```

**Edge Attributes:**

```xml
<edge source="entity-uuid-1" target="entity-uuid-2">
  <data key="type">PUBLISHED</data>
  <data key="weight">1.0</data>
  <data key="description">IPCC published Climate Report 2024</data>
</edge>
```

**Used By:** GraphRAG Local, LightRAG

#### hipporag_graph.graphml

**Purpose:** HippoRAG-specific associative graph

**Node Attributes:**

```xml
<node id="entity-uuid-1">
  <data key="name">IPCC</data>
  <data key="activation">0.85</data>  <!-- Activation level -->
  <data key="last_accessed">2026-02-15T10:30:00</data>
</node>
```

**Edge Attributes:**

```xml
<edge source="entity-uuid-1" target="entity-uuid-2">
  <data key="type">ASSOCIATES_WITH</data>
  <data key="strength">0.75</data>  <!-- Association strength -->
  <data key="access_count">12</data>
</edge>
```

**Used By:** HippoRAG only

---

## Auxiliary Indexes

### BM25 Index

**Technology:** rank-bm25 library

**Storage:** `output/indexes/bm25_index.pkl` (pickle format)

**Structure:**

```python
{
    "corpus": List[List[str]],        # Tokenized documents
    "chunk_ids": List[str],           # Chunk IDs (parallel to corpus)
    "parameters": {
        "k1": 1.5,
        "b": 0.75,
        "epsilon": 0.25
    }
}
```

**Creation:**

```python
from rank_bm25 import BM25Okapi
import pickle

# Tokenize chunks
corpus = [chunk.text.lower().split() for chunk in chunks]

# Build BM25 index
bm25 = BM25Okapi(corpus)

# Save
with open("output/indexes/bm25_index.pkl", "wb") as f:
    pickle.dump({
        "bm25": bm25,
        "chunk_ids": [str(c.id) for c in chunks],
        "parameters": {"k1": 1.5, "b": 0.75, "epsilon": 0.25}
    }, f)
```

**Used By:** Hybrid RAG

---

### LightRAG Indexes

**Technology:** Custom dual indexes (pickle)

**Storage:** `output/indexes/lightrag_indexes.pkl`

**Structure:**

```python
{
    "entity_index": {
        "entity_id": {
            "embedding": np.ndarray,
            "related_entities": List[str],
            "chunks": List[str]
        }
    },
    "relationship_index": {
        "relationship_id": {
            "embedding": np.ndarray,  # Embedding of relationship description
            "source": str,
            "target": str,
            "type": str
        }
    }
}
```

**Used By:** LightRAG

---

## Metadata Files

### .schema_version.json

**Purpose:** Track schema version for migrations

**Format:**

```json
{
    "version": "1.0",
    "applied_at": "2026-02-15T10:30:00Z",
    "migrations": [
        "initial_schema_v1.0"
    ]
}
```

**Location:** `output/.schema_version.json`

---

### indexing_stats.json

**Purpose:** Store indexing run statistics

**Format:**

```json
{
    "run_id": "550e8400-e29b-41d4-a716-446655440000",
    "started_at": "2026-02-15T10:00:00Z",
    "completed_at": "2026-02-15T10:30:00Z",
    "duration_seconds": 1800,
    "status": "success",
    "stats": {
        "document_count": 1000,
        "chunk_count": 25000,
        "entity_count": 3500,
        "relationship_count": 8200,
        "community_count": 45
    },
    "costs": {
        "llm_calls": 12500,
        "llm_tokens": 1500000,
        "embedding_calls": 28500,
        "total_usd": 45.67
    },
    "config_hash": "abc123def456",  # Hash of extraction config
    "strategies_built": ["naive", "hybrid", "graphrag_local", "graphrag_global"]
}
```

**Location:** `output/metadata/indexing_stats.json`

---

### extraction_config.json

**Purpose:** Store extraction configuration used for indexing

**Format:**

```json
{
    "llm": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "temperature": 0.0
    },
    "extraction": {
        "entity_types": ["PERSON", "ORGANIZATION", "LOCATION"],
        "relationship_types": ["WORKS_FOR", "LOCATED_IN"],
        "max_gleanings": 1,
        "min_confidence": 0.7
    },
    "chunking": {
        "chunk_size": 512,
        "chunk_overlap": 64
    },
    "prompts": {
        "entity_extraction": "...",
        "relationship_extraction": "..."
    }
}
```

**Location:** `output/metadata/extraction_config.json`

**Purpose:** Enable validation that query-time config matches index-time config.

---

## File Formats Summary

| File | Format | Purpose | Size (10K docs) |
|------|--------|---------|-----------------|
| `documents.parquet` | Parquet | Source documents | ~50 MB |
| `chunks.parquet` | Parquet | Text chunks | ~125 MB |
| `entities.parquet` | Parquet | Extracted entities | ~5 MB |
| `relationships.parquet` | Parquet | Entity relationships | ~10 MB |
| `communities.parquet` | Parquet | Graph communities | ~1 MB |
| `community_reports.parquet` | Parquet | Community summaries | ~2 MB |
| `chunks.lance/` | LanceDB | Chunk embeddings | ~400 MB |
| `entities.lance/` | LanceDB | Entity embeddings | ~60 MB |
| `entity_graph.graphml` | GraphML | Entity graph | ~8 MB |
| `bm25_index.pkl` | Pickle | BM25 sparse index | ~30 MB |
| `lightrag_indexes.pkl` | Pickle | LightRAG indexes | ~15 MB |
| **Total** | | | **~706 MB** |

---

## Migration Strategy

### Schema Versioning

Use semantic versioning for schemas: `MAJOR.MINOR`

- **Major version change:** Breaking schema change (e.g., field removed)
- **Minor version change:** Backward-compatible change (e.g., field added)

### Migration Process

1. **Detect version mismatch:**

```python
def check_schema_version(output_dir: Path) -> str:
    version_file = output_dir / ".schema_version.json"
    if not version_file.exists():
        return "0.0"  # Legacy or missing

    with open(version_file) as f:
        data = json.load(f)
    return data["version"]
```

2. **Apply migrations:**

```python
def migrate(output_dir: Path, from_version: str, to_version: str):
    migrations = get_migrations(from_version, to_version)
    for migration in migrations:
        migration.apply(output_dir)

    # Update version file
    with open(output_dir / ".schema_version.json", "w") as f:
        json.dump({
            "version": to_version,
            "applied_at": datetime.utcnow().isoformat(),
            "migrations": [m.name for m in migrations]
        }, f)
```

3. **Example migration (1.0 → 1.1):**

```python
class AddAliasesFieldMigration:
    """Add 'aliases' field to entities.parquet."""

    name = "add_aliases_field_v1.1"

    def apply(self, output_dir: Path):
        # Load entities
        table = pq.read_table(output_dir / "entities.parquet")

        # Add empty aliases column
        aliases = pa.array([[]] * len(table), type=pa.list_(pa.string()))
        table = table.append_column("aliases", aliases)

        # Save
        pq.write_table(table, output_dir / "entities.parquet")
```

---

## Backup and Recovery

### Backup Strategy

1. **Snapshot backup:**

```bash
# Full backup
tar -czf backup_2026-02-15.tar.gz output/

# Incremental backup (only Parquet files)
tar -czf backup_parquet_2026-02-15.tar.gz output/*.parquet
```

2. **Parquet append-only:**

Parquet files support append operations, enabling incremental updates without rewriting.

### Recovery

1. **Restore from backup:**

```bash
tar -xzf backup_2026-02-15.tar.gz
```

2. **Rebuild indexes:**

If vector/graph indexes corrupted:

```bash
graph-unified rebuild-indexes --strategies naive,hybrid,graphrag_local
```

---

## Performance Considerations

### Parquet Optimization

- **Compression:** Snappy (fast, good ratio)
- **Row group size:** 100,000 rows
- **Column pruning:** Only read needed columns
- **Predicate pushdown:** Filter at Parquet level

### LanceDB Optimization

- **Index type:** IVF_FLAT for <1M vectors, IVF_PQ for larger
- **Partitioning:** Partition by document_id if corpus >100K documents
- **Caching:** LRU cache for frequently accessed vectors

### File System Layout

- **SSD recommended:** Random access for vector search
- **NFS compatible:** Works on network file systems
- **Concurrent reads:** Safe for multi-process reads
- **Single writer:** Only one process should write at a time

---

## Summary

This specification defines:

- **Directory structure** for organized storage
- **7 Parquet schemas** for structured data
- **2 LanceDB indexes** for vector search
- **2 GraphML files** for graph structures
- **3 auxiliary indexes** for strategy-specific needs
- **Metadata files** for versioning and stats
- **Migration strategy** for schema evolution
- **Backup procedures** for data safety

**Storage Characteristics:**
- ~706 MB total for 10K documents
- Parquet for structured data (columnar, compressed)
- LanceDB for vectors (columnar, indexed)
- GraphML for graphs (XML, portable)
- Pickle for Python-specific data (auxiliary indexes)

**Next Steps:**
- Implement Parquet I/O in `storage/parquet_store.py`
- Integrate LanceDB in `storage/vector_store.py`
- Build GraphML handling in `storage/graph_store.py`
- Create migration framework in `storage/migrations.py`
