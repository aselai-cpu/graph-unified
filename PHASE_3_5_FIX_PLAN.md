# Phase 3.5: Critical Fixes & Enhancements

## Status Report

### ✅ Already Fixed
1. **Chunk.entity_ids population** - Working correctly! All chunks have entity_ids and relationship_ids.

### 🔧 To Fix

## Priority 1: Performance & Scalability (2-3 days)

### Fix 1: Replace O(n) Parquet Scan with Entity-to-Chunk Index
**Problem**: GraphRAG Local, LightRAG, HippoRAG all iterate through ALL chunks on every query
**Impact**: At 100K chunks, each query reads 50-200MB from disk
**Solution**: Build reverse index in LanceDB: entity_id → [chunk_ids]

**Files to modify:**
- `graphunified/storage/vector_store.py` - Add `get_chunks_by_entities()` method
- `graphunified/strategies/graphrag_local.py` - Use index instead of scan
- `graphunified/strategies/lightrag.py` - Use index instead of scan
- `graphunified/strategies/hipporag.py` - Use index instead of scan

**Implementation**: Create EntityChunkIndex table in LanceDB with schema:
```python
{
    "entity_id": str,
    "chunk_id": str,
    "chunk_embedding": List[float],  # cached for fast retrieval
    "chunk_text_preview": str,  # first 200 chars for display
}
```

---

## Priority 2: LightRAG Architecture Fix (2-3 days)

### Fix 2: Implement Relationship-Based Global Search
**Problem**: Using GraphRAG's community-based approach instead of LightRAG's relationship-based approach
**Impact**: Missing the key innovation of LightRAG - semantic relationship matching

**GitHub Reference**: https://github.com/HKUDS/LightRAG

**Solution**:
1. Build relationship vector index during `index()`
2. Global search: `query → relationship_embeddings → top relationships → extract entities → collect chunks`
3. Score by relationship relevance + entity connection strength

**Files to modify:**
- `graphunified/strategies/lightrag.py`:
  - `_global_retrieval()` - Rewrite to use relationship search
  - `index()` - Build relationship vector index
  - Add `_relationship_cache` for fast lookups

**New methods needed:**
```python
async def _search_relationships(self, query: str, top_k: int) -> List[Relationship]
async def _get_entities_from_relationships(self, rels: List[Relationship]) -> List[Entity]
```

---

## Priority 3: GraphRAG Global Enhancements (2-3 days)

### Fix 3a: Embedding-Based Community Search
**Problem**: Keyword matching is brittle and misses relevant communities
**Solution**: Use vector similarity on community summary embeddings

**Files to modify:**
- `graphunified/strategies/graphrag_global.py`:
  - `index()` - Embed community summaries
  - `_rank_communities()` - Use cosine similarity instead of keywords

### Fix 3b: Map-Reduce Answer Synthesis
**Problem**: Returning raw community reports instead of synthesized answers
**GitHub Reference**: https://github.com/microsoft/graphrag

**Solution**: After retrieving relevant communities, synthesize answer:
```
Query + Community Reports → LLM (Claude) → Synthesized Answer
```

**Files to modify:**
- `graphunified/strategies/graphrag_global.py`:
  - Add `_synthesize_answer()` method
  - Modify `retrieve()` to call synthesis
  - Return synthesized answer in chunk.text

### Fix 3c: Switch to Leiden Algorithm
**Problem**: Louvain gives flat communities, Leiden gives hierarchical
**Solution**: Install `igraph` and use Leiden algorithm

**Files to modify:**
- `requirements.txt` - Add `python-igraph`
- `graphunified/storage/graph_store.py` - Already has `detect_communities_leiden()` method
- `graphunified/strategies/graphrag_global.py` - Use Leiden instead of Louvain

---

## Priority 4: HippoRAG Enhancements (3-4 days)

### Fix 4a: Add Fact-Based Retrieval (Stage 1)
**Problem**: Skipping fact retrieval stage (pattern separation)
**GitHub Reference**: https://github.com/OSU-NLP-Group/HippoRAG

**Solution**: Implement 4-stage retrieval:
1. Stage 1: Query → Fact Retrieval (search triple embeddings)
2. Stage 2: Recognition Memory (rerank facts)
3. Stage 3: Graph Activation (extract entities from facts)
4. Stage 4: PPR Spreading

**Files to modify:**
- `graphunified/strategies/hipporag.py`:
  - Add `_retrieve_facts()` method
  - Add `_rerank_facts()` method
  - Modify `retrieve()` to use 4-stage pipeline
  - Update `_run_personalized_pagerank()` to use weighted resets

**Note**: `Fact` model and `VectorStore.search_facts()` already exist but unused

### Fix 4b: Add Dense Passage Retrieval
**Problem**: No direct query-chunk similarity scoring
**Solution**: Parallel dense retrieval + PPR, combine scores

**Files to modify:**
- `graphunified/strategies/hipporag.py`:
  - Add `_dense_passage_retrieval()` method
  - Add `_combine_retrieval_results()` method
  - Add `combine_alpha` parameter (default 0.5)

---

## Priority 5: Minor Fixes (1 day)

### Fix 5a: Score Normalization
**Problem**: Scores not comparable across strategies
**Solution**: Normalize all scores to [0, 1] range

**Files to modify:**
- `graphunified/strategies/base.py` - Add `_normalize_scores()` helper
- All strategy files - Call normalize before returning results

### Fix 5b: VectorStore Length Validation Bug
**Problem**: Chained comparison `a != b != c` logic error
**Solution**: Use `not (a == b == c)` pattern

**Files to modify:**
- `graphunified/storage/vector_store.py` - Lines 151, 207, 262

### Fix 5c: Persist Community Reports
**Problem**: Community reports only in memory, lost on restart
**Solution**: Save to Parquet using CommunityReport model

**Files to modify:**
- `graphunified/strategies/graphrag_global.py` - Save reports after generation
- `graphunified/storage/parquet_store.py` - Add `save_community_reports()` if missing

---

## Implementation Order

### Week 1 (Days 1-3): Performance Foundation
1. Day 1: Entity-to-chunk index (Fix 1)
2. Day 2: LightRAG relationship search (Fix 2)
3. Day 3: GraphRAG Global embeddings + map-reduce (Fix 3a, 3b)

### Week 2 (Days 4-6): Algorithm Enhancements
4. Day 4: Leiden algorithm (Fix 3c)
5. Day 5: HippoRAG fact retrieval (Fix 4a)
6. Day 6: HippoRAG dense retrieval (Fix 4b)

### Week 3 (Day 7): Polish
7. Day 7: Minor fixes (Fix 5a, 5b, 5c)

---

## Testing Strategy

After each fix:
1. Run existing test scripts to ensure no regressions
2. Add specific tests for new functionality
3. Measure performance improvement (latency, memory)
4. Update documentation

---

## Success Metrics

| Metric | Before | Target After Fixes |
|--------|--------|-------------------|
| **GraphRAG Local latency** | 50ms + O(n) scan | <100ms constant |
| **LightRAG global accuracy** | Low (keyword match) | High (semantic) |
| **GraphRAG Global quality** | Raw reports | Synthesized answers |
| **HippoRAG fidelity** | 4/10 vs paper | 8/10 vs paper |
| **All strategies chunk retrieval** | Sometimes empty | Always correct |
| **Score comparability** | Incomparable | Normalized [0,1] |

---

## GitHub References

- **LightRAG**: https://github.com/HKUDS/LightRAG
- **Microsoft GraphRAG**: https://github.com/microsoft/graphrag
- **HippoRAG**: https://github.com/OSU-NLP-Group/HippoRAG

---

## Next Steps

Ready to start implementation. Confirm approach or request modifications.
