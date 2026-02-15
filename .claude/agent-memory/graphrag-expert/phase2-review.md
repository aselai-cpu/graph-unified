# Phase 2 Review: GraphRAG Integration Assessment

## Pipeline Architecture Observations

### Current Pipeline
```
Load → Chunk → Extract (Entities + Relationships) → Embed → Store
```

### Required GraphRAG Pipeline
```
Load → Chunk → Extract → BuildGraph → DetectCommunities → SummarizeCommunities → Embed → Store
```

**Gap**: Missing 3 critical stages for GraphRAG functionality.

## Entity/Relationship Extraction Quality

### Strengths
- **Good deduplication**: 90% fuzzy threshold with type-based clustering
- **Source tracking**: All entities/relationships track source_chunks for provenance
- **Confidence scores**: Extraction prompts request 0.0-1.0 confidence
- **Alias preservation**: Deduplicated entity names stored as aliases

### Issues Found
1. **No max gleanings**: Config defines it but extraction doesn't implement multi-pass refinement
2. **Relationship extraction scalability**: Passes all entity names into prompt (will exceed context for large docs)
3. **Fixed relationship weights**: Defaults to 1.0, should compute from co-occurrence + confidence
4. **Limited entity types**: 5 types vs Microsoft GraphRAG's typical 10-15

## Critical Missing Components

### 1. Graph Construction Stage
- No NetworkX/igraph builder
- No edge weight computation
- No graph validation (disconnected components, self-loops)

### 2. Leiden Community Detection
- No leidenalg integration
- No hierarchical partitioning
- No parent-child community linking

### 3. Community Summarization (Map-Reduce)
- No LLM-based report generation
- No token budget management
- No community ranking/importance scoring

## Data Model Assessment

### Well-Designed
- `Community` model has all necessary fields (level, entity_ids, parent_community_id, density)
- `CommunityReport` model ready for map-reduce (title, summary, findings, rank)
- Parquet schemas support hierarchical structures
- Storage design supports lazy loading of large graphs

### Needs Enhancement
```python
# Add to Community model:
- modularity: float  # Leiden quality metric
- internal_edges: int
- external_edges: int
- conductance: float

# Add to GraphRAGStrategyConfig:
- hierarchical_levels: int = 3
- min_community_size: int = 3
- community_report_max_tokens: int = 2000
- edge_weight_threshold: float = 0.1
```

## Phase 3 Implementation Checklist

### High Priority (Core Functionality)
- [ ] Implement BuildGraphStage (NetworkX from entities + relationships)
- [ ] Implement DetectCommunitiesStage (Leiden algorithm with hierarchy)
- [ ] Implement SummarizeCommunitiesStage (LLM-based reports)
- [ ] Split EmbedStage into pre/post community stages
- [ ] Add hierarchical community parent-child linking
- [ ] Update pipeline.py DAG with new stages

### Medium Priority (Quality Improvements)
- [ ] Implement max_gleanings in extraction
- [ ] Add relationship weight computation
- [ ] Chunk-level relationship extraction (avoid passing all entities)
- [ ] Graph validation checks
- [ ] Token budget management for summarization

### Low Priority (Nice to Have)
- [ ] Graph export to GraphML for visualization
- [ ] Support multiple community algorithms (Leiden/Louvain/LP)
- [ ] Expand entity types to 10-15 for richer domains
- [ ] Embedding-based entity deduplication (supplement fuzzy matching)

## Cost Estimation Notes

**Community Summarization Token Cost**:
- For 1000 communities × 2000 tokens/summary = 2M tokens
- At $3/M tokens (Claude Haiku): $6 per indexing run
- Recommend: Cache reports, use progressive bottom-up summarization

**Leiden Performance**:
- Scales to 100k+ nodes with igraph
- Expect 10-60 seconds for typical document corpus (1k-10k entities)

## Integration Patterns

### Query Router Logic
```python
if query_type == "entity_specific":
    strategy = "graphrag_local"  # Navigate entity neighborhoods
elif query_type == "thematic_broad":
    strategy = "graphrag_global"  # Map-reduce over community reports
```

### Shared Extraction Benefits
- Single extraction pipeline feeds all 6 strategies
- GraphRAG communities can enhance LightRAG entity retrieval
- HippoRAG can leverage GraphRAG's deduplication

## Known Issues from Microsoft GraphRAG

1. **Token costs**: Community summarization is most expensive indexing stage
2. **Latency**: Global search map-reduce adds 5-10s vs local search
3. **Quality sensitivity**: Poor entity extraction → poor communities → poor global search
4. **Tuning complexity**: Leiden resolution parameter requires domain-specific tuning

## References

- Microsoft GraphRAG architecture: Communities defined via Leiden algorithm with hierarchical partitioning
- Typical hierarchy: 2-4 levels (Level 0 = fine-grained, Level N = coarse)
- Community reports: 500-2000 tokens each, embeddings for retrieval
- Map-reduce: Retrieve top-K reports by embedding, synthesize with LLM
