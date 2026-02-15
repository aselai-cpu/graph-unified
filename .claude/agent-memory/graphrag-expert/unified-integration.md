# Unified Multi-Strategy RAG Integration Patterns

## Architecture Philosophy

GraphRAG complements rather than replaces traditional RAG strategies. A unified system leverages:
- **Vector RAG**: Fast semantic similarity, good for factual retrieval
- **GraphRAG Local**: Entity-centric queries, relationship traversal, multi-hop reasoning
- **GraphRAG Global**: Thematic understanding, dataset-level insights

## Shared Chunking Strategy

### Why Share Chunking
- Consistency across retrieval strategies
- Reduced preprocessing overhead
- Easier cross-strategy result comparison
- Single source of truth for text segmentation

### Implementation Pattern
```python
# 1. Create shared text units (chunks)
from graphrag.index.operations.chunk_text import chunk_text

chunks = chunk_text(
    documents,
    chunk_size=800,
    chunk_overlap=100
)

# 2. Feed to multiple pipelines
# - GraphRAG indexer consumes chunks → builds KG
# - Vector RAG indexes same chunks → builds vector store
# - Both reference same document IDs and chunk IDs
```

### Text Unit Schema
GraphRAG creates `text_units.parquet`:
- id: Unique chunk identifier
- text: Chunk content
- document_ids: Source documents
- entity_ids: Extracted entities (links to entities.parquet)

Use this as canonical chunking output. Other strategies can:
1. Read text_units.parquet directly
2. Create vector embeddings from text_units.text
3. Maintain document_id references for traceability

## Modular Indexing Pipeline

### Separate Concerns
```
Input Documents
    ↓
Shared Chunking Layer (text_units)
    ↓
    ├─→ Entity Extraction → Knowledge Graph → Communities → GraphRAG Index
    ├─→ Dense Embeddings → Vector Store → Vector RAG Index
    └─→ BM25 Indexing → Lexical Search Index
```

### Programmatic GraphRAG Indexing
```python
from graphrag.index import create_pipeline_config, run_pipeline_with_config
from graphrag.config import GraphRagConfig

# Load or create config
config = GraphRagConfig.from_file("./settings.yaml")

# Customize for unified system
config.chunks.size = 800  # Match other strategies
config.chunks.overlap = 100
config.community_reports.max_length = 2000

# Run indexing
pipeline_config = create_pipeline_config(config)
await run_pipeline_with_config(pipeline_config)
```

### Storage Integration
GraphRAG outputs to `./output/<timestamp>/artifacts/`:
- `documents.parquet`
- `text_units.parquet`
- `entities.parquet`
- `relationships.parquet`
- `communities.parquet`
- `community_reports.parquet`

For unified systems:
1. Configure GraphRAG output to shared storage location
2. Other strategies read `text_units.parquet` for consistent chunking
3. Maintain separate vector stores per strategy
4. Use common document ID namespace

## Query Routing Strategy

### Intent-Based Router
```python
def route_query(query: str, intent: str) -> Strategy:
    if intent == "entity_specific":
        # "What are the key characteristics of [entity]?"
        return GraphRAGLocalSearch

    elif intent == "relationship":
        # "How are [entity1] and [entity2] connected?"
        return GraphRAGLocalSearch

    elif intent == "thematic_overview":
        # "What are the main themes in the dataset?"
        return GraphRAGGlobalSearch

    elif intent == "factual_lookup":
        # "What is the definition of [term]?"
        return VectorRAG

    elif intent == "multi_hop":
        # "What impact did [entity1] have on [entity2] through [entity3]?"
        return GraphRAGLocalSearch

    else:
        # Default: hybrid approach
        return HybridSearch
```

### Hybrid Retrieval Pattern
For complex queries, combine strategies:
```python
# 1. Vector RAG: Get relevant chunks
vector_results = vector_search(query, top_k=20)

# 2. Extract entities from results
entities = extract_entities_from_chunks(vector_results)

# 3. GraphRAG Local: Expand with entity neighborhoods
graph_results = local_search(entities, query)

# 4. Merge and rerank
final_results = rerank(vector_results + graph_results)
```

## Cross-Referencing Pattern

### Shared Identifiers
All strategies should maintain:
- **document_id**: Original document identifier
- **text_unit_id**: Chunk/text unit identifier
- **entity_id**: Entity identifier (from GraphRAG)

### Linking Example
```python
# Vector search returns text_unit_ids
vector_hits = ["text_unit_1", "text_unit_5", "text_unit_12"]

# Lookup entities in those units
entities_df = pd.read_parquet("entities.parquet")
relevant_entities = entities_df[
    entities_df['text_unit_ids'].apply(
        lambda x: any(unit in x for unit in vector_hits)
    )
]

# Now do graph traversal from those entities
graph_context = local_search_from_entities(relevant_entities['id'].tolist())
```

## Shared Entity Extraction Layer

### Efficiency Pattern
Entity extraction is expensive (LLM calls). Share extraction:

```python
# 1. GraphRAG extracts entities during indexing
# Output: entities.parquet with id, title, type, description

# 2. Other strategies can reuse extracted entities
entities_df = pd.read_parquet("./output/latest/artifacts/entities.parquet")

# 3. Enrich vector search with entity metadata
for chunk_id, chunk_text in text_units:
    # Find entities in this chunk
    chunk_entities = entities_df[
        entities_df['text_unit_ids'].apply(lambda x: chunk_id in x)
    ]

    # Add entity metadata to vector store
    vector_store.upsert(
        id=chunk_id,
        text=chunk_text,
        metadata={
            'entities': chunk_entities['title'].tolist(),
            'entity_types': chunk_entities['type'].tolist()
        }
    )
```

## Configuration Alignment

### Chunk Size Considerations
GraphRAG default: ~300 tokens
Vector RAG typical: 512-1024 tokens

For unified systems:
- Use 800 tokens as middle ground
- Ensures entities aren't fragmented across chunks
- Sufficient context for vector similarity
- Reasonable LLM context window usage

### Overlap Strategy
- Standard: 100 tokens (12.5% of 800-token chunks)
- Prevents entity boundary issues
- Maintains context continuity

### Example Unified Config
```yaml
# settings.yaml for unified system
chunks:
  size: 800
  overlap: 100
  group_by_columns: [document_id]  # Maintain document boundaries

entity_extraction:
  max_gleanings: 2
  entity_types: [organization, person, location, event, technology]

embeddings:
  model: text-embedding-3-small  # Share model across strategies
  vector_store: lance  # Or your unified vector store

community_reports:
  max_length: 2000
  max_cluster_size: 20

storage:
  base_dir: "./unified_output"  # Shared storage location
```

## Performance Considerations

### Indexing
- GraphRAG indexing is slowest (entity extraction + graph construction)
- Run GraphRAG indexing first, reuse outputs for other strategies
- Parallelize vector embedding generation separately

### Query Time
- Vector search: ~10-100ms
- GraphRAG Local: ~1-5s (depends on graph traversal depth)
- GraphRAG Global: ~5-30s (map-reduce over communities)
- Use caching for frequently accessed graph neighborhoods

### Cost Optimization
- Entity extraction: Most expensive (LLM calls per chunk)
- Community reports: Moderate (LLM calls per community)
- Embeddings: Low (embedding model calls)
- Reuse entity extraction across all strategies to minimize cost

## Example Unified System Architecture

```
┌─────────────────┐
│  Input Documents │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Chunker │ (text_units.parquet)
    └────┬────┘
         │
    ┌────▼──────────────────────────────┐
    │ Entity Extraction (GraphRAG)      │
    │ Output: entities.parquet          │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │ Graph Construction                │
    │ Output: relationships.parquet     │
    │         communities.parquet       │
    │         community_reports.parquet │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │ Vector Embeddings (Parallel)      │
    ├───────────────────────────────────┤
    │ 1. Entity descriptions → Index A  │
    │ 2. Community reports → Index B    │
    │ 3. Text units → Index C (Vector)  │
    └────┬──────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │ Unified Query Interface           │
    ├───────────────────────────────────┤
    │ - Intent Router                   │
    │ - Hybrid Search Orchestrator      │
    │ - Result Merger & Reranker        │
    └───────────────────────────────────┘
```

## Migration Path

### Adding GraphRAG to Existing Vector RAG
1. Keep existing vector store intact
2. Run GraphRAG indexing on same documents
3. Implement query router to selectively use GraphRAG
4. Gradually expand GraphRAG usage based on query patterns

### Shared Chunking Migration
1. Export existing chunks to text_units.parquet format
2. Configure GraphRAG to use existing chunks (skip chunking step)
3. Run entity extraction on existing chunks
4. Build graph from extracted entities

## Best Practices

1. **Start Simple**: Implement basic routing before complex hybrid strategies
2. **Monitor Performance**: Track query latency per strategy, optimize routing
3. **Cost Awareness**: Entity extraction is expensive, batch and cache where possible
4. **Validation**: Compare results across strategies to tune routing logic
5. **Observability**: Log which strategy served each query for analysis
6. **Incremental Updates**: Plan for incremental graph updates vs. full reindexing
