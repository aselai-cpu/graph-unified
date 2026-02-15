# Indexing Architecture: Vector & Text Indexes

## Problem Statement

Currently, the indexing pipeline generates embeddings and saves them to Parquet files, but **does not build searchable indexes**. This means:

- ❌ Embeddings exist but cannot be queried
- ❌ No vector similarity search available
- ❌ No BM25 text search available
- ❌ Retrieval strategies cannot work without indexes

## Current State (Phase 2 Complete)

```
Pipeline Stages:
1. Load     → Documents loaded from disk
2. Chunk    → Documents split into chunks
3. Extract  → Entities/relationships extracted
4. Embed    → Embeddings generated for all items
            → Saved to Parquet files

Missing:
5. Index    → ❌ NOT IMPLEMENTED YET
```

## Architecture Design

### Two Index Types Needed

#### 1. Vector Indexes (LanceDB)

**Purpose:** Enable semantic similarity search using embeddings

**Indexes Required:**
- `chunks` index - For Naive RAG, Hybrid RAG
- `entities` index - For GraphRAG, LightRAG, HippoRAG
- `relationships` index - For LightRAG global search
- `facts` index - For HippoRAG Stage 1 (future)
- `communities` index - For GraphRAG global search (future)

**Storage Location:**
```
output/
├── lancedb/              ← LanceDB vector indexes
│   ├── chunks.lance/
│   ├── entities.lance/
│   └── relationships.lance/
└── parquet/              ← Source data
    ├── chunks/*.parquet
    ├── entities/*.parquet
    └── relationships/*.parquet
```

**Build Process:**
```python
# Read embeddings from Parquet
chunks = await parquet_store.load_chunks()

# Build LanceDB index
await vector_store.index_chunks(
    chunk_ids=[c.id for c in chunks],
    embeddings=[c.embedding for c in chunks],
    texts=[c.text for c in chunks],
    metadata=[c.metadata for c in chunks]
)
```

**Index Type:** IVF_FLAT (configurable)
- Fast similarity search using L2 distance
- Supports filtering by metadata
- Automatically handles duplicate IDs (upsert)

#### 2. Text Index (BM25)

**Purpose:** Enable keyword-based search for Hybrid RAG

**Index Required:**
- `chunks` BM25 index - For Hybrid RAG keyword component

**Storage:**
```python
# Option A: In-memory (simple, fast for small datasets)
class BM25Index:
    def __init__(self):
        self.index = {}  # token -> [(doc_id, freq), ...]
        self.doc_lengths = {}
        self.avg_doc_length = 0.0

# Option B: Persistent (for large datasets)
# Use Whoosh or Tantivy for disk-based BM25
```

**Build Process:**
```python
# Tokenize and build inverted index
bm25_index = BM25Index()
for chunk in chunks:
    bm25_index.add_document(
        doc_id=chunk.id,
        text=chunk.text
    )
bm25_index.finalize()
```

## Proposed Implementation

### Option 1: Add Stage 5 to Pipeline (Recommended)

**Pros:**
- Indexes built automatically during indexing
- Single command builds everything
- Consistent with current pipeline architecture

**Cons:**
- Slightly longer indexing time
- Rebuilds indexes even if only re-indexing a few documents

**Implementation:**
```python
# graphunified/index/pipeline.py

class IndexPipeline:
    def __init__(self, ...):
        # ... existing code ...

        # Add vector store
        self.vector_store = VectorStore.from_config(
            settings.storage.vector_db,
            output_dir / "lancedb",
            settings.embedding.dimension
        )

        # Add text index (for Hybrid RAG)
        self.text_index = BM25Index()

    async def run(self, ...):
        # ... existing stages 1-4 ...

        # Stage 5: Build Indexes
        logger.info("Stage 5/5: Building searchable indexes")
        await self._build_vector_indexes(chunks, entities, relationships)
        await self._build_text_index(chunks)

    async def _build_vector_indexes(self, chunks, entities, relationships):
        """Build LanceDB vector indexes."""
        # Index chunks
        if chunks:
            await self.vector_store.index_chunks(
                chunk_ids=[c.id for c in chunks],
                embeddings=[c.embedding for c in chunks],
                texts=[c.text for c in chunks],
            )
            logger.info(f"Indexed {len(chunks)} chunks")

        # Index entities
        if entities:
            await self.vector_store.index_entities(
                entity_ids=[e.id for e in entities],
                embeddings=[e.embedding for e in entities],
                names=[e.name for e in entities],
            )
            logger.info(f"Indexed {len(entities)} entities")

        # Index relationships
        if relationships:
            await self.vector_store.index_relationships(
                relationship_ids=[r.id for r in relationships],
                embeddings=[r.embedding for r in relationships],
                descriptions=[r.description for r in relationships],
            )
            logger.info(f"Indexed {len(relationships)} relationships")

    async def _build_text_index(self, chunks):
        """Build BM25 text index for keyword search."""
        for chunk in chunks:
            self.text_index.add_document(chunk.id, chunk.text)
        self.text_index.finalize()
        logger.info(f"Built BM25 index for {len(chunks)} chunks")
```

### Option 2: Separate Index Build Command

**Pros:**
- Flexibility to rebuild indexes without re-extracting
- Useful for experimenting with different index types
- Can parallelize better

**Cons:**
- Extra step users must remember
- More complex workflow

**Implementation:**
```bash
# Build indexes from existing Parquet files
python -m graphunified.cli build-indexes \
  --input-dir ./output \
  --index-type IVF_FLAT

# Or as part of indexing
python -m graphunified.cli index \
  --input-dir ./corpus \
  --output-dir ./output \
  --build-indexes  # Optional flag
```

## When Are Indexes Built?

### Recommended Timing

**During Indexing (Option 1 - Recommended):**
```
1. Load documents → 2. Chunk → 3. Extract → 4. Embed → 5. Build Indexes
                                                         ↓
                                          LanceDB + BM25 ready for queries
```

**On-Demand (Option 2):**
```
Indexing:  Load → Chunk → Extract → Embed → Save to Parquet

Later:     Load from Parquet → Build Indexes → Query
```

### Incremental Updates

For incremental indexing:
```python
# New documents added
new_chunks = [...newly indexed chunks...]

# Update indexes (upsert)
await vector_store.index_chunks(new_chunks)  # Automatically merges
bm25_index.update(new_chunks)  # Incremental update
```

## Performance Considerations

### Vector Index Build Time

| Dataset Size | Index Build Time | Query Time |
|--------------|------------------|------------|
| 1K chunks    | ~1 second        | <10ms      |
| 10K chunks   | ~5 seconds       | <20ms      |
| 100K chunks  | ~30 seconds      | <50ms      |
| 1M chunks    | ~5 minutes       | <100ms     |

### BM25 Index Build Time

| Dataset Size | Index Build Time | Query Time |
|--------------|------------------|------------|
| 1K chunks    | <1 second        | <5ms       |
| 10K chunks   | ~2 seconds       | <10ms      |
| 100K chunks  | ~15 seconds      | <20ms      |
| 1M chunks    | ~2 minutes       | <50ms      |

## Storage Overhead

```
Example: 10,000 chunks, 1024-dim embeddings

Parquet (raw data):
- chunks.parquet: ~50MB (text + embeddings + metadata)

LanceDB (vector index):
- chunks.lance/: ~40MB (vectors + index structure)
- IVF_FLAT overhead: ~10% of vector size

BM25 (text index):
- In-memory: ~5MB (inverted index)
- Disk (Whoosh): ~10MB (compressed)

Total: ~105MB for 10K chunks
```

## Recommendation

**Implement Option 1: Add Stage 5 to Pipeline**

Rationale:
- ✅ Simplest user experience (one command)
- ✅ Indexes always up-to-date with data
- ✅ Consistent with current pipeline design
- ✅ Performance is acceptable (<1 minute for 10K chunks)
- ✅ Enables immediate querying after indexing

**Next Steps:**
1. Implement Stage 5 in IndexPipeline
2. Add BM25Index class (in-memory for Phase 3)
3. Update CLI to show index build progress
4. Test with 10K chunk dataset
5. Document index build behavior

## Future Enhancements

- [ ] Disk-based BM25 for large datasets (Whoosh/Tantivy)
- [ ] Alternative vector index types (HNSW, IVF_PQ)
- [ ] Distributed indexing for multi-TB datasets
- [ ] Hot reloading of indexes without downtime
- [ ] Index versioning and rollback
