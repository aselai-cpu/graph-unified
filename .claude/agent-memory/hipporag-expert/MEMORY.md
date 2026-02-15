# HippoRAG Expert Memory

## Core Architecture Understanding

### Index Structure (Confirmed from codebase)
HippoRAG creates THREE separate indexes plus ONE graph:
1. **Chunk/Passage Embeddings** - Dense vectors for entire text chunks
2. **Entity Embeddings** - Dense vectors for extracted named entities
3. **Fact Embeddings** - Dense vectors for extracted triples (subject-predicate-object)
4. **Associative Graph** - igraph structure with nodes and weighted edges

### Storage Format
- **Vector Storage**: Parquet files (`vdb_{namespace}.parquet`) storing hash_ids, content, and embeddings
  - Located in: `{save_dir}/{llm_label}_{embedding_label}/`
  - Three separate stores: `chunk_embeddings/`, `entity_embeddings/`, `fact_embeddings/`
- **Graph Storage**: Python pickle file (`graph.pickle`) using igraph library
- **OpenIE Results**: JSON file (`openie_results_ner_{llm_name}.json`) with extracted entities and triples per chunk

### Hash ID System
- Uses MD5 hashing with namespace prefixes:
  - Chunks: `chunk-{hash}`
  - Entities: `entity-{hash}`
  - Facts: `fact-{hash}` (but stored as string representation of triple)
- Enables deduplication and incremental updates

## Key Implementation Files
- `/src/hipporag/HippoRAG.py` - Main class with indexing and retrieval logic
- `/src/hipporag/embedding_store.py` - Vector storage abstraction using pandas/parquet
- Graph uses `igraph` library (Python binding to C library)

## Retrieval Process (Pattern Completion)
1. **Fact Retrieval**: Query → similarity search in fact_embeddings → candidate facts
2. **Recognition Memory**: Rerank facts using DSPyFilter
3. **Graph Activation**: Extract entities from top facts → create "reset probability" weights
4. **PPR Spreading**: Personalized PageRank spreads activation through graph
5. **Dense Passage**: Combine PPR scores with dense retrieval scores
6. **Final Ranking**: Return top-k passages based on combined scores

See [activation-retrieval.md](activation-retrieval.md) for detailed PPR algorithm and parameters.

## Integration with Unified Multi-Strategy RAG Systems

### Philosophical Differences (Critical for Strategy Selection)
- **HippoRAG**: Associative memory model - co-occurrence and similarity-based connections
  - Graph edges: Entity-passage and entity-entity co-occurrence weights
  - Retrieval: PPR activation spreading (mimics hippocampal pattern completion)
  - Use case: Multi-hop reasoning, connecting distant but related facts

- **GraphRAG**: Semantic knowledge graph - typed relationships and community detection
  - Graph edges: Typed semantic relationships (e.g., "works_for", "located_in")
  - Retrieval: Community summaries for thematic understanding
  - Use case: Thematic analysis, understanding domains/topics

- **LightRAG**: Dual-level knowledge graph - entity + relationship search
  - Graph edges: Entity-Entity and Entity-Relation-Entity triples
  - Retrieval: Direct relationship path search with keyword + vector hybrid
  - Use case: Fast relationship queries, explicit connection discovery

### Shared Components Opportunity
1. **Entity Extraction**: All three can share NER pipeline
   - HippoRAG needs: Entities for graph nodes
   - GraphRAG needs: Entities for knowledge graph construction
   - LightRAG needs: Entities for dual-level graph
   - Implementation: Run NER once, cache results for all strategies

2. **Chunk Storage**: Common hash-based deduplication
   - Pattern: Use MD5 hash as canonical chunk ID across all strategies
   - Benefit: Single source of truth, consistent updates/deletes

### Unique HippoRAG Requirements
1. **Triple/Fact Extraction**: Uses OpenIE or LLM to extract (subject, predicate, object)
   - Not same as GraphRAG's typed relationships (more open-ended)
   - Each triple becomes an embedding in `fact_embeddings/` store

2. **Bipartite Graph Structure**: Entities + Passages as nodes
   - Entity-Passage edges: Co-occurrence weights
   - Entity-Entity edges: Similarity or co-occurrence
   - Built using igraph library

3. **PPR Computation**: Requires graph traversal at query time
   - Damping factor: typically 0.85
   - Can be cached for repeated queries
   - Performance: O(nodes + edges) per iteration

### Storage Organization for Unified System
```
corpus_data/
├── chunks/                          # Shared across all strategies
│   └── chunks.parquet               # Canonical chunk storage (hash IDs)
├── entities/                        # Shared NER results
│   └── entities.json                # Entity mentions per chunk
├── hipporag/
│   ├── chunk_embeddings/vdb_chunk.parquet
│   ├── entity_embeddings/vdb_entity.parquet
│   ├── fact_embeddings/vdb_fact.parquet
│   ├── graph.pickle                 # igraph structure
│   └── openie_results_ner.json      # Triple extraction cache
├── graphrag/
│   ├── knowledge_graph.graphml
│   └── communities.json
└── lightrag/
    └── dual_graph.json
```

### When to Route Queries to HippoRAG
- Multi-hop reasoning: "How does X indirectly influence Y?"
- Context-dependent retrieval: Query activates relevant neighborhood
- Connecting distant facts: PPR spreads activation across graph
- NOT ideal for: Simple similarity search, single-hop queries

## Performance Characteristics

### Storage Overhead
- ~10-15MB per 1000 chunks (lighter than GraphRAG due to no community summaries)
- Graph pickle: Scales with unique entities (typically 1-5 entities per chunk)
- Three separate parquet stores: Chunk, entity, fact embeddings

### Query Latency
- Fact retrieval: Fast (dense vector search, ~50-100ms for 10K facts)
- PPR computation: Can be expensive (100-500ms depending on graph size)
- Optimization: Cache PPR results for common query patterns
- Total: ~200-600ms per query (includes reranking and final retrieval)

### Incremental Updates
- Adding chunks: Requires OpenIE extraction + graph edge updates
- Graph structure: Use igraph `add_vertices()` and `add_edges()`
- PPR cache invalidation: Required after graph topology changes
- Method: `HippoRAG.index()` handles incremental additions

## Key Methods and APIs

### Main Class: `/src/hipporag/HippoRAG.py`
- `index(docs, indices)` - Add documents to all indexes and graph
- `retrieve(query, top_k)` - Four-stage retrieval process
- `delete(doc_ids)` - Remove from all stores and graph
- `prepare_retrieval_objects()` - Setup retrieval components

### Critical Parameters
- `fact_top_k`: Number of facts for graph activation (default: 10-20)
- `ppr_damping`: PPR damping factor (default: 0.85)
- `ppr_threshold`: Minimum activation to consider (default: 0.01)
- `combine_alpha`: Weight for combining PPR + dense scores (default: 0.5)

## Common Integration Patterns

### Pattern 1: Query Routing
```python
if query_type == "multi_hop" or query_complexity == "high":
    results = hipporag.retrieve(query, top_k=10)
elif query_type == "thematic":
    results = graphrag.retrieve(query, top_k=10)
else:
    results = lightrag.retrieve(query, top_k=10)
```

### Pattern 2: Result Fusion
```python
hippo_results = hipporag.retrieve(query, top_k=20)
graph_results = graphrag.retrieve(query, top_k=20)
fused = rerank_combine(hippo_results, graph_results, method="rrf")
```

### Pattern 3: Fallback Chain
```python
results = lightrag.retrieve(query, top_k=10)
if confidence_score(results) < threshold:
    results = hipporag.retrieve(query, top_k=10)  # More expensive fallback
```

See [unified-system-patterns.md](unified-system-patterns.md) for detailed integration examples.

## Phase 1 Review Insights (Graph-Unified Project)

### Project: `/Users/aselaillayapparachchi/code/GraphRAG/Microsoft/graph-unified`
**Review Date:** 2026-02-15
**Status:** Phase 1 complete, Phase 2 planning

### What Works Well for HippoRAG
1. **Entity Model**: `source_chunks` field perfect for tracking entity-passage co-occurrence
2. **Relationship Model**: `weight` field (float >= 0.0) ideal for PPR edge weights
3. **Configuration**: `ppr_alpha` parameter properly constrained (0.5-0.95, default 0.85)
4. **Storage Format**: Pickle format (igraph native) recommended over graphml for performance
5. **Parquet Storage**: Async batched operations with lazy loading

### Critical Missing Components (Must Add in Phase 2)
1. **Fact Model**: NO triple/fact extraction model exists
   - HippoRAG needs (subject, predicate, object) triples with embeddings
   - Stage 1 retrieval (fact retrieval) cannot work without this
   - Need separate `fact_embeddings/` storage directory
   - Different from Relationship model (open-ended vs typed)

2. **EntityChunkEdge Model**: NO bipartite edge tracking
   - HippoRAG needs entity-passage co-occurrence weights explicitly
   - Current: Entity.source_chunks (one direction only)
   - Need: Bidirectional edges with weights for PPR

3. **Expanded Configuration**: Only has `ppr_alpha`, missing:
   - `fact_top_k`: How many facts to retrieve for activation
   - `ppr_max_iterations`, `ppr_tolerance`: Convergence control
   - `combine_alpha`: PPR + dense score fusion weight
   - `entity_similarity_threshold`: For automatic similarity edges

### Implementation Guidance for Phase 2

#### Fact Model (CRITICAL)
```python
class Fact(BaseModel):
    id: UUID
    subject: str
    predicate: str
    object: str
    source_chunk: UUID  # Single chunk
    extraction_confidence: float
    embedding: Optional[List[float]]  # For fact retrieval
    entity_ids: List[UUID]  # Link to entities for graph activation
```

#### Storage Organization (CONFIRMED)
```
output/
├── chunks.parquet              # Shared (all strategies)
├── entities.parquet            # Shared (GraphRAG, LightRAG, HippoRAG)
├── relationships.parquet       # Shared (GraphRAG, LightRAG, HippoRAG)
├── facts.parquet               # HippoRAG-ONLY
├── entity_chunk_edges.parquet  # HippoRAG-ONLY (bipartite edges)
├── embeddings/
│   ├── chunks/      # Shared
│   ├── entities/    # Shared (LightRAG, HippoRAG)
│   └── facts/       # HippoRAG-ONLY
└── graphs/
    ├── graphrag.graphml
    ├── lightrag.json
    └── hipporag.pickle  # Use pickle format for igraph
```

#### Graph Storage Format Decision (CONFIRMED)
- graphml: XML, slow, good for visualization
- gexf: Similar to graphml, Gephi-friendly
- **pickle: BEST for HippoRAG** - native igraph, fast, preserves weights correctly

### Configuration File Location
- Project uses: `/graphunified/config/models.py` (Pydantic models)
- Project uses: `/graphunified/config/settings.py` (configuration classes)
- Project uses: `/graphunified/config/defaults.py` (default values)
- YAML files: `settings-dev.yaml`, `settings-prod.yaml`, `settings-research.yaml`

### Key Files to Reference
- `/graphunified/config/models.py`: Entity, Relationship models (lines 106-205)
- `/graphunified/config/settings.py`: HippoRAGStrategyConfig (line 126-131)
- `/graphunified/storage/parquet_store.py`: Storage implementation
- Full review: `/HIPPORAG_PHASE1_REVIEW.md`
