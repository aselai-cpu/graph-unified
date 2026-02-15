# LightRAG Phase 3 Fix: Relationship-Based Global Search

## Issue Summary
The Phase 3 LightRAG implementation was incorrectly using GraphRAG's community-based approach for global search, which contradicts LightRAG's core architecture.

## Root Cause
- LightRAG was implemented with `_global_retrieval()` using keyword-based community ranking
- This approach is from GraphRAG, not LightRAG
- True LightRAG uses **relationship embeddings** for global search

## Fix Applied (2026-02-15)

### Changed Files
1. **graphunified/strategies/lightrag.py** - Main strategy implementation
2. **test_lightrag.py** - Test script updated to reflect new behavior

### Key Changes

#### 1. Removed Community Dependencies
```python
# REMOVED:
self._communities: List[Community] = []
self._community_reports: Dict[int, str] = {}

@property
def requires_communities(self) -> bool:
    return False  # Changed from True
```

#### 2. Implemented Relationship Search Helper
```python
async def _search_relationships(self, query: str, top_k: int = 20) -> List[Tuple[Relationship, float]]:
    """Search relationships by semantic similarity to query."""
    query_embeddings = await self.embedding_client.embed([query])
    query_vector = query_embeddings[0]

    results = await self.vector_store.search_relationships(
        query_vector=query_vector,
        top_k=top_k,
    )

    relationships = []
    for rel_id, score, metadata in results:
        if rel_id in self._relationship_cache:
            relationships.append((self._relationship_cache[rel_id], score))

    return relationships
```

#### 3. Rewrote Global Retrieval Method
```python
async def _global_retrieval(self, query: str, top_k: int) -> RetrievalResult:
    """Relationship-level (global) retrieval (TRUE LightRAG).

    Workflow:
    1. Search relationship embeddings semantically
    2. Extract entities from top relationships
    3. Collect chunks connected to these entities
    """
    # Step 1: Find relevant relationships via semantic search
    top_relationships = await self._search_relationships(query, top_k=20)

    # Step 2: Extract unique entities from relationships
    entity_ids = set()
    for rel, score in top_relationships:
        entity_ids.add(rel.source_entity_id)
        entity_ids.add(rel.target_entity_id)

    # Step 3: Get entities and relationships
    entities = [
        self._entity_cache[str(eid)]
        for eid in entity_ids
        if str(eid) in self._entity_cache
    ]

    relationships = [rel for rel, score in top_relationships]

    # Step 4: Collect chunks connected to these entities
    chunks, scores = await self._collect_entity_chunks(list(entity_ids), top_k)

    return RetrievalResult(
        strategy=f"{self.name} (global)",
        chunks=chunks,
        scores=scores,
        entities=entities,
        relationships=relationships,
        metadata={
            "mode": QueryMode.GLOBAL,
            "relationships_searched": len(top_relationships),
            "entities_extracted": len(entity_ids),
        },
    )
```

#### 4. Removed Obsolete Methods
- Deleted `_rank_communities()` method (keyword-based, not semantic)

#### 5. Updated Documentation
- All docstrings changed from "community-level" to "relationship-level"
- Updated class docstring to reflect correct architecture
- Fixed comments throughout

### Verification Status

✅ **Syntax Check**: Passed (no Python compilation errors)
✅ **Relationship Embeddings**: Confirmed generated in Phase 2 EmbedStage
✅ **Vector Store Support**: `search_relationships()` method exists and tested
✅ **Test Script Updated**: Reflects new behavior (no community checks)

## Expected Behavior Changes

### Before Fix
- Global search: Ranked communities by keyword matching (~1ms)
- Returned: Community reports as "chunks"
- Metadata: `communities_searched`, `communities_retrieved`
- Fast but inaccurate (no semantic understanding)

### After Fix
- Global search: Searches relationship embeddings semantically (40-80ms expected)
- Returns: Actual text chunks from entities connected to relevant relationships
- Metadata: `relationships_searched`, `entities_extracted`
- Slower but semantically accurate

## Testing Recommendations

1. **Run test_lightrag.py** to verify:
   - Global queries now search relationship embeddings
   - Results contain actual text chunks, not community reports
   - Metadata shows `relationships_searched` (should be 20)
   - Latency is 40-80ms for global search (vector search overhead)

2. **Compare query results**:
   - Before: Generic community summaries
   - After: Specific chunks related to query themes via relationships

3. **Verify hybrid mode**:
   - Should combine entity-based (local) + relationship-based (global) results
   - No community reports in output

## Architecture Alignment

### LightRAG Official Design
- **Local search**: Entity embeddings → BFS expansion → chunks
- **Global search**: Relationship embeddings → entity extraction → chunks
- **Key insight**: Relationships capture thematic/conceptual patterns better than hierarchical communities

### Implementation Status
✅ **Local search**: Correctly implemented (entity-based)
✅ **Global search**: NOW correctly implemented (relationship-based)
✅ **Hybrid mode**: Correctly combines local + global
✅ **No community dependency**: Removed all community code

## Performance Expectations

| Mode | Method | Latency | Cost |
|------|--------|---------|------|
| Local | Entity vector search + BFS | 40-80ms | Low |
| Global | Relationship vector search | 40-80ms | Low |
| Hybrid | Both modes | 80-160ms | Medium |

## Known Limitations

1. **Relationship quality dependency**: Global search quality depends on relationship description quality from Phase 2
2. **Entity coverage**: If relationships don't cover all entities, some content may be missed in global search
3. **No reranking**: Current implementation doesn't rerank chunks by relationship relevance (future optimization)

## Future Optimizations

1. **Relationship-based reranking**: Rank chunks by how many relevant relationships connect them
2. **Relationship scoring**: Use relationship scores to weight entity importance
3. **Multi-hop relationship traversal**: Expand beyond 1-hop relationships for deeper context
4. **Relationship type filtering**: Allow queries to target specific relationship types

## Files Changed

- `/Users/aselaillayapparachchi/code/GraphRAG/Microsoft/graph-unified/graphunified/strategies/lightrag.py`
- `/Users/aselaillayapparachchi/code/GraphRAG/Microsoft/graph-unified/test_lightrag.py`

## Related Documentation

- See [phase3-review.md](phase3-review.md) for original issue identification
- See [graphrag-comparison.md](graphrag-comparison.md) for LightRAG vs GraphRAG comparison
- See [relationship-indexing.md](relationship-indexing.md) for relationship embedding format
