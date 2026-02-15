# Phase 3 Week 2 Review: GraphRAG Local & Global Implementation

## Implementation Status

### GraphRAG Local: 85% Complete ✅
- **Working**: Entity vector search → BFS expansion → chunk collection
- **Performance**: ~50ms, excellent
- **Missing**: Claims extraction (optional)
- **Verdict**: Production-ready

### GraphRAG Global: 60% Complete ⚠️
- **Working**: Louvain communities, LLM report generation
- **Critical Gaps**:
  - Keyword ranking (needs embeddings)
  - No map-reduce synthesis
  - Flat communities (needs Leiden hierarchy)
- **Verdict**: Needs significant work

## Critical Issues Identified

### Issue 1: Louvain vs Leiden
**Problem**: Using Louvain instead of Leiden loses hierarchical structure
- Leiden guarantees well-connected communities
- Hierarchy enables multi-level abstractions (2-4 levels typical)
- Microsoft GraphRAG's design assumes hierarchical communities

**Impact**: Can't do progressive refinement, missing 40% of Global's value

**Fix**: Add `python-igraph` + `leidenalg`, implement hierarchical detection

### Issue 2: Keyword vs Embedding Search
**Problem**: Global search uses keyword matching to rank communities
```python
# Current (wrong)
matches = sum(1 for token in query_tokens if token in report.lower())

# Should be (right)
similarity = cosine_similarity(query_embedding, community_embedding)
```

**Impact**: Misses semantically similar but lexically different communities

**Fix**: Embed community reports during indexing, vector search at query time

### Issue 3: Missing Map-Reduce
**Problem**: Global search returns raw community reports instead of synthesized answer

**Microsoft's approach**:
```
Query → Find Communities → MAP (extract insights) → REDUCE (synthesize) → Answer
```

**Your approach**:
```
Query → Find Communities → Return reports as chunks
```

**Impact**: Feels like vanilla RAG, not GraphRAG

**Fix**: Add map-reduce LLM pipeline (2 steps, 5-10 calls)

### Issue 4: Chunk Loading Scalability
**Problem**: Local search loads ALL chunks to filter by entities
```python
async for chunk in parquet_store.load_chunks():  # Loads everything
    if any(eid in entity_ids for eid in chunk.entity_ids):
```

**Impact**: Won't scale beyond 1K chunks

**Fix**: Build chunk_id → entity_ids index during initialization

## Community Detection Quality

**Test data**: 25 entities → 6 communities
- Size distribution: [7, 1, 7, 1, 4, 5]
- **Issue**: 2 singleton communities (over-fragmentation)
- **Verdict**: Acceptable for 25 entities, but Leiden would prevent singletons

## Community Report Quality

**Prompt structure**: Title / Summary / Key Themes / Importance

**Assessment**:
- ✅ Coherent, readable, thematically focused
- ⚠️ Missing FINDINGS (atomic facts for map-reduce)
- ⚠️ Missing RATING (importance score)
- ⚠️ Generic language (needs domain context in prompt)

**Comparison to Microsoft**: More narrative vs. fact-dense

**Recommendation**: Add FINDINGS section with bulleted facts

## Performance Analysis

### GraphRAG Local
- Current: 50ms (excellent)
- Bottleneck: Chunk loading at scale (10K+ entities)
- Fix: Index chunk→entity mapping

### GraphRAG Global
- Current: <1ms (deceptively fast)
- Expected with fixes: 2-8s (matches Microsoft)
  - Embedding search: +50-200ms
  - Map-reduce: +2-8s (5-10 LLM calls)
- This is acceptable for Global's use case (exploratory questions)

## Prioritized Recommendations

### Critical (10-12 hours)
1. **Embedding-based community search** (2-3h)
2. **Map-reduce synthesis** (4-6h)
3. **Fix chunk loading** (1-2h)

### Important (12-15 hours)
4. **Switch to Leiden** (4-6h)
5. **Update report structure** (2-3h)
6. **Community embeddings** (1-2h)

### Nice-to-have (16-20 hours)
7. Claims extraction
8. Bottom-up summarization
9. Graph quality metrics

## Key Comparisons

| Feature | Your Impl | Microsoft | Status |
|---------|-----------|-----------|--------|
| **Local: Entity search** | Vector | Vector | ✅ Match |
| **Local: Graph traversal** | BFS | Weighted BFS | ✅ Good |
| **Local: Claims** | None | Extracted | ⚠️ Missing |
| **Global: Communities** | Louvain | Leiden | ❌ Critical |
| **Global: Hierarchy** | Flat (1 level) | 2-4 levels | ❌ Critical |
| **Global: Search** | Keywords | Embeddings | ❌ Critical |
| **Global: Synthesis** | None | Map-reduce | ❌ Critical |

## Code Locations

- **Local strategy**: `/graphunified/strategies/graphrag_local.py` (480 lines)
- **Global strategy**: `/graphunified/strategies/graphrag_global.py` (552 lines)
- **Graph store**: `/graphunified/storage/graph_store.py` (detect_communities_louvain)
- **Test outputs**: `test_graphrag_local_output.txt`, `test_graphrag_global_output.txt`

## Architectural Integration

**Strengths**:
- Clean separation from other 5 strategies
- Proper base class usage
- Async throughout
- Good configuration pattern

**Concerns**:
- Global strategy doesn't leverage Local for hybrid queries
- No query routing between Local/Global based on question type
- Could share entity cache across strategies

## Production Readiness

### GraphRAG Local
- **Ready**: Yes, with minor fix for chunk loading
- **Use cases**: Entity-specific questions, multi-hop reasoning
- **Performance**: Excellent
- **Recommendation**: Ship it

### GraphRAG Global
- **Ready**: No, needs critical fixes
- **Use cases**: Currently limited (keyword matching too weak)
- **After fixes**: Broad thematic questions, corpus summarization
- **Recommendation**: Implement items 1-2-3 minimum before release

## Lessons Learned

1. **Leiden is not optional**: It's core to GraphRAG's philosophy
2. **Map-reduce is the innovation**: Without it, Global is just vector search
3. **Embeddings everywhere**: Keyword matching doesn't cut it in 2025
4. **Hierarchy matters**: Flat communities lose abstraction levels
5. **Scalability from day 1**: Chunk loading pattern won't work at scale

## References

- Microsoft GraphRAG uses Leiden algorithm with 2-4 hierarchical levels
- Community reports are 500-2000 tokens, embedded separately
- Global search is map-reduce (extract + synthesize), not retrieve
- Local search explores entity neighborhoods, typically 1-2 hops
- Typical performance: Local 50-200ms, Global 2-10s
