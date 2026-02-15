# LightRAG Integration in Unified Multi-Strategy Systems

## Overview

This document captures patterns for integrating LightRAG into unified RAG systems that combine multiple retrieval strategies (GraphRAG, HippoRAG, VectorRAG, LightRAG).

## Architecture Pattern

### Multi-Strategy RAG System Structure:

```
┌─────────────────────────────────────────────────────────┐
│                    Query Interface                       │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   Query Router/Orchestrator              │
│  (Determines optimal strategy based on query type)       │
└─────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────┬───────┴────────┬──────────┐
        ↓          ↓                ↓          ↓
  ┌──────────┐ ┌─────────┐  ┌──────────┐ ┌──────────┐
  │ VectorRAG│ │ GraphRAG│  │ HippoRAG │ │ LightRAG │
  │          │ │         │  │          │ │          │
  │ Dense    │ │Community│  │ Episodic │ │Entity-   │
  │ Retrieval│ │ Reports │  │ Memory   │ │Relation  │
  └──────────┘ └─────────┘  └──────────┘ └──────────┘
        ↓          ↓                ↓          ↓
        └──────────┴────────────────┴──────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              Result Fusion & Reranking                   │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                LLM Answer Synthesis                      │
└─────────────────────────────────────────────────────────┘
```

## Component Sharing Strategy

### Shared Components (Single Processing):

1. **Document Chunking:**
   - Single chunking pass for all strategies
   - Store chunks with universal IDs
   - Configure chunk size based on most restrictive strategy needs
   - Location: `/chunks/` directory

2. **Entity Extraction:**
   - Single LLM call per chunk to extract entities
   - Output used by: GraphRAG, HippoRAG, LightRAG
   - Store entities with references to source chunks
   - Location: `/entities/` directory

3. **Text Embeddings:**
   - Cache embeddings for reuse across strategies
   - Different embedding models per strategy if needed
   - Location: `/embeddings/` directory

### Strategy-Specific Components:

**LightRAG Unique:**
- Relationship description generation (additional LLM call)
- Dual vector indexes (entities + relationships)
- Flat graph structure (GraphML file)
- Local/global/hybrid search logic

**GraphRAG Unique:**
- Community detection (Leiden algorithm)
- Hierarchical community structure
- Multi-level community report generation
- Community-based global search

**HippoRAG Unique:**
- Personalized PageRank scoring
- Episodic memory graph
- Synonym/paraphrase expansion
- PPR-based retrieval

## Storage Organization

### Directory Structure:

```
project_root/
├── config/
│   ├── lightrag_config.yaml
│   ├── graphrag_config.yaml
│   └── unified_config.yaml
│
├── data/
│   ├── raw/                    # Original documents
│   └── processed/
│       ├── chunks/             # Shared: Universal chunk storage
│       │   └── chunks.json
│       ├── entities/           # Shared: Entity extraction output
│       │   └── entities.json
│       └── embeddings/         # Shared: Cached embeddings
│           ├── chunk_embeddings.npy
│           └── entity_embeddings.npy
│
├── indexes/
│   ├── vector/                 # VectorRAG
│   │   └── faiss_index/
│   │
│   ├── graphrag/               # GraphRAG
│   │   ├── communities/
│   │   ├── reports/
│   │   └── indexes/
│   │
│   ├── hipporag/               # HippoRAG
│   │   ├── ppr_graph/
│   │   └── synonyms/
│   │
│   └── lightrag/               # LightRAG
│       ├── vdb_entities.json
│       ├── vdb_relationships.json
│       └── graph_chunk_entity_relation.graphml
│
└── router/
    ├── query_classifier.py
    ├── strategy_selector.py
    └── result_fusion.py
```

### Key Principles:

1. **Separation of Concerns:** Each strategy has isolated storage
2. **Shared Foundation:** Chunks, entities, embeddings centralized
3. **Cross-References:** Use universal IDs to link across strategies
4. **Incremental Updates:** Add to shared components once, update strategy indexes

## Query Routing Logic

### Query Classification Dimensions:

1. **Scope:** Entity-specific, Multi-entity, Thematic, Corpus-wide
2. **Complexity:** Single-hop, Multi-hop, Reasoning-required
3. **Temporal:** Recent events, Historical patterns, Timeless facts
4. **Granularity:** Precise facts, Conceptual understanding, High-level summary

### Routing Decision Tree:

```python
def route_query(query, context=None):
    """Route query to optimal strategy or combination of strategies."""

    query_features = analyze_query(query)

    # Entity-specific queries
    if query_features['has_explicit_entities']:
        if query_features['num_entities'] == 1:
            return ['lightrag_local', 'vectorrag']
        elif query_features['requires_multi_hop']:
            return ['lightrag_hybrid', 'hipporag']

    # Relationship-focused queries
    if query_features['asks_about_relationships']:
        return ['lightrag_global']

    # Thematic/conceptual queries
    if query_features['is_thematic']:
        if query_features['scope'] == 'corpus_wide':
            return ['graphrag_global', 'lightrag_global']
        else:
            return ['lightrag_global', 'vectorrag']

    # Summarization queries
    if query_features['requires_summary']:
        return ['graphrag_global']

    # Temporal/contextual queries
    if query_features['has_temporal_context']:
        return ['hipporag', 'lightrag_local']

    # Default: ensemble
    return ['vectorrag', 'lightrag_hybrid', 'graphrag_local']


def analyze_query(query):
    """Extract features from query to inform routing."""
    return {
        'has_explicit_entities': detect_entities(query),
        'num_entities': count_entities(query),
        'requires_multi_hop': detect_multi_hop_keywords(query),
        'asks_about_relationships': detect_relationship_keywords(query),
        'is_thematic': detect_thematic_keywords(query),
        'scope': estimate_scope(query),
        'requires_summary': detect_summary_keywords(query),
        'has_temporal_context': detect_temporal_keywords(query),
    }
```

### Keyword-Based Classification:

**LightRAG Local Indicators:**
- "what is [entity]"
- "how does [entity] work"
- "properties of [entity]"
- "definition of [entity]"

**LightRAG Global Indicators:**
- "how do X and Y relate"
- "what patterns exist"
- "relationship between"
- "connection between"
- "what themes"

**GraphRAG Global Indicators:**
- "summarize main points"
- "overall themes"
- "high-level overview"
- "key insights across"
- "what are the major"

**HippoRAG Indicators:**
- "recent discussions about"
- "when was [entity] mentioned"
- "context around [event]"
- "similar to [example]"

## Result Fusion Strategy

### Multi-Strategy Query Results:

When multiple strategies are invoked, results must be fused intelligently.

```python
def fuse_results(results_by_strategy, query):
    """Combine results from multiple strategies."""

    fused = {
        'chunks': [],
        'entities': [],
        'relationships': [],
        'communities': [],
        'scores': []
    }

    # Collect all chunks with strategy-specific scores
    for strategy, results in results_by_strategy.items():
        for chunk in results['chunks']:
            fused['chunks'].append({
                'chunk_id': chunk['id'],
                'text': chunk['text'],
                'source_strategy': strategy,
                'score': chunk['score']
            })

    # Deduplicate chunks (same chunk may appear from multiple strategies)
    fused['chunks'] = deduplicate_chunks(fused['chunks'])

    # Rerank using cross-strategy scoring
    fused['chunks'] = rerank_chunks(fused['chunks'], query)

    # Merge entity and relationship information
    fused['entities'] = merge_entities(results_by_strategy)
    fused['relationships'] = merge_relationships(results_by_strategy)

    return fused


def deduplicate_chunks(chunks):
    """Remove duplicate chunks, keeping highest scoring version."""
    chunk_map = {}

    for chunk in chunks:
        chunk_id = chunk['chunk_id']
        if chunk_id not in chunk_map or chunk['score'] > chunk_map[chunk_id]['score']:
            chunk_map[chunk_id] = chunk

    return list(chunk_map.values())


def rerank_chunks(chunks, query):
    """Rerank chunks using cross-encoder or LLM-based scoring."""
    # Strategy 1: Cross-encoder reranking
    query_chunk_pairs = [(query, c['text']) for c in chunks]
    rerank_scores = cross_encoder.predict(query_chunk_pairs)

    for chunk, score in zip(chunks, rerank_scores):
        chunk['rerank_score'] = score

    # Sort by rerank score
    chunks.sort(key=lambda x: x['rerank_score'], reverse=True)

    return chunks
```

### Fusion Strategies:

1. **Score-Based Weighting:**
   - Assign weights to strategies based on query type
   - Normalize scores across strategies
   - Combine weighted scores for final ranking

2. **Diversity-Aware Selection:**
   - Select top-k from each strategy
   - Ensure diversity across strategies
   - Avoid redundancy in final result set

3. **Confidence-Based Filtering:**
   - Filter results below confidence threshold per strategy
   - Use only high-confidence results from each
   - Fill remaining slots with next-best cross-strategy

## LightRAG-Specific Integration Points

### 1. Shared Entity Extraction with GraphRAG:

```python
# Single entity extraction pass
entities = await extract_entities(chunks, domain)

# Use for LightRAG entity index
lightrag.index_entities(entities)

# Use for GraphRAG community detection
graphrag.build_communities(entities)

# Use for HippoRAG graph construction
hipporag.build_graph(entities)
```

### 2. Relationship Generation (LightRAG-Only):

```python
# After shared entity extraction
entities_by_chunk = group_entities_by_chunk(entities)

# Generate relationship descriptions (LightRAG-specific)
relationships = await generate_relationship_descriptions(
    entities_by_chunk,
    chunks,
    domain
)

# Index relationships for LightRAG global search
lightrag.index_relationships(relationships)

# Note: GraphRAG doesn't need these (uses communities instead)
```

### 3. Query Processing Integration:

```python
async def process_query(query):
    """Process query using unified system."""

    # Route to strategies
    strategies = route_query(query)

    # Execute strategies in parallel
    tasks = []
    if 'lightrag_local' in strategies:
        tasks.append(lightrag.local_search(query))
    if 'lightrag_global' in strategies:
        tasks.append(lightrag.global_search(query))
    if 'graphrag_global' in strategies:
        tasks.append(graphrag.global_search(query))
    # ... other strategies

    results = await asyncio.gather(*tasks)

    # Fuse results
    fused_results = fuse_results(
        dict(zip(strategies, results)),
        query
    )

    # Generate final answer
    answer = await synthesize_answer(query, fused_results)

    return answer
```

## Performance Optimization

### 1. Incremental Indexing:

```python
async def add_documents(new_docs):
    """Add new documents to unified system."""

    # Step 1: Chunk (shared)
    new_chunks = chunk_documents(new_docs)
    save_chunks(new_chunks)

    # Step 2: Extract entities (shared)
    new_entities = await extract_entities(new_chunks)
    save_entities(new_entities)

    # Step 3: Update strategy indexes in parallel
    await asyncio.gather(
        lightrag.add_chunks(new_chunks, new_entities),
        graphrag.add_chunks(new_chunks, new_entities),
        hipporag.add_chunks(new_chunks, new_entities),
        vectorrag.add_chunks(new_chunks)
    )
```

### 2. Caching Strategy:

```python
class UnifiedRAGCache:
    """Cache to avoid redundant computation across strategies."""

    def __init__(self):
        self.chunk_embeddings = {}
        self.entity_embeddings = {}
        self.query_embeddings = {}
        self.query_results = {}  # LRU cache

    async def get_chunk_embedding(self, chunk_id, model):
        """Get cached or compute chunk embedding."""
        key = f"{chunk_id}_{model}"
        if key not in self.chunk_embeddings:
            chunk = load_chunk(chunk_id)
            self.chunk_embeddings[key] = await embed(chunk, model)
        return self.chunk_embeddings[key]

    def cache_query_result(self, query, strategy, result, ttl=3600):
        """Cache query results with TTL."""
        key = f"{hash(query)}_{strategy}"
        self.query_results[key] = (result, time.time() + ttl)
```

### 3. Parallel Processing:

```python
# Index multiple strategies simultaneously
async def build_unified_index(documents):
    """Build all strategy indexes in parallel."""

    # Shared preprocessing
    chunks = chunk_documents(documents)
    entities = await extract_entities(chunks)

    # Strategy-specific indexing in parallel
    await asyncio.gather(
        lightrag.build_index(chunks, entities),
        graphrag.build_index(chunks, entities),
        hipporag.build_index(chunks, entities),
        vectorrag.build_index(chunks)
    )
```

## Cost Management

### LightRAG-Specific Costs in Unified System:

**Indexing:**
- Entity extraction: Shared (amortized across strategies)
- Relationship generation: LightRAG-specific (additional cost)
- Embedding relationships: LightRAG-specific (additional cost)

**Strategy to Minimize Cost:**
1. Use cheaper model (GPT-4o-mini, Claude Haiku) for relationship generation
2. Generate relationships only for high-importance entity pairs
3. Batch relationship generation across chunks
4. Cache and reuse relationship patterns

**Cost Comparison (per 1000 documents):**
- VectorRAG only: $X (baseline)
- +LightRAG: $X + $Y (relationship generation)
- +GraphRAG: $X + $Z (community reports)
- Full unified: $X + $Y + $Z (shared entity extraction saves ~30%)

## Monitoring & Evaluation

### Key Metrics per Strategy:

**LightRAG:**
- Local search hit rate (queries served by local search)
- Global search relevance (relationship description quality)
- Graph density (entities per chunk, relationships per entity pair)
- Query latency (local vs global vs hybrid)

**Cross-Strategy:**
- Router accuracy (did we route to the right strategy?)
- Result overlap (how many chunks appear from multiple strategies?)
- Fusion effectiveness (does reranking improve results?)
- End-to-end latency (query → answer)

### Evaluation Framework:

```python
def evaluate_unified_system(test_queries):
    """Evaluate unified system with labeled test queries."""

    metrics = {
        'routing_accuracy': [],
        'strategy_hit_rate': defaultdict(list),
        'answer_quality': [],
        'latency': []
    }

    for query, expected_strategy, ground_truth in test_queries:
        start = time.time()

        # Route query
        selected_strategies = route_query(query)
        metrics['routing_accuracy'].append(
            expected_strategy in selected_strategies
        )

        # Execute
        answer = process_query(query)

        # Evaluate
        metrics['answer_quality'].append(
            evaluate_answer(answer, ground_truth)
        )
        metrics['latency'].append(time.time() - start)

        for strategy in selected_strategies:
            metrics['strategy_hit_rate'][strategy].append(1)

    return metrics
```

## Common Pitfalls & Solutions

### Pitfall 1: Storage Path Conflicts

**Problem:** LightRAG and GraphRAG writing to same file paths

**Solution:**
```python
# Configure separate storage paths
lightrag_config = {
    'working_dir': '/indexes/lightrag/',
    'entity_db': 'vdb_entities.json',
    'relationship_db': 'vdb_relationships.json'
}

graphrag_config = {
    'working_dir': '/indexes/graphrag/',
    'community_dir': 'communities/',
    'report_dir': 'reports/'
}
```

### Pitfall 2: Redundant Entity Extraction

**Problem:** Each strategy extracting entities separately

**Solution:**
```python
# Extract once, use everywhere
entities = await extract_entities(chunks)
save_entities(entities)  # Centralized storage

# Pass to strategies
lightrag.load_entities(entities)
graphrag.load_entities(entities)
hipporag.load_entities(entities)
```

### Pitfall 3: Poor Query Routing

**Problem:** Router sends queries to suboptimal strategies

**Solution:**
- Build labeled test set of queries with ideal strategies
- Train router on historical query-strategy-quality data
- A/B test different routing heuristics
- Allow fallback to ensemble when confidence is low

### Pitfall 4: Result Redundancy

**Problem:** Same chunks returned from multiple strategies without deduplication

**Solution:**
```python
# Implement chunk ID-based deduplication
def deduplicate_chunks(results_by_strategy):
    seen_chunks = {}
    for strategy, results in results_by_strategy.items():
        for chunk in results:
            if chunk['id'] not in seen_chunks:
                seen_chunks[chunk['id']] = chunk
            else:
                # Keep version with higher score
                if chunk['score'] > seen_chunks[chunk['id']]['score']:
                    seen_chunks[chunk['id']] = chunk
    return list(seen_chunks.values())
```

## Summary Checklist for Integration

- [ ] Separate storage directories for each strategy
- [ ] Shared chunking pipeline (single pass)
- [ ] Shared entity extraction (single LLM call per chunk)
- [ ] LightRAG relationship generation (additional LLM call)
- [ ] Query router with classification logic
- [ ] Result fusion with deduplication
- [ ] Cross-encoder reranking
- [ ] Caching for embeddings and results
- [ ] Incremental update support
- [ ] Monitoring and evaluation framework
- [ ] Cost tracking per strategy
- [ ] Fallback strategies for failures
