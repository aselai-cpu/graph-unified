# GraphRAG Global Enhancements - Quick Reference

## What Changed?

### 1. Community Ranking: Keyword → Semantic Similarity

**Old Way**:
```python
# Count matching keywords
matches = sum(1 for token in query.split() if token in report)
score = matches / len(query_tokens)
```

**New Way**:
```python
# Semantic similarity with embeddings
query_embedding = await self.embedding_client.embed([query])
results = await self.vector_store.search_communities(query_embedding[0], top_k=5)
```

### 2. Answer Format: Raw Reports → Synthesized Answer

**Old Way**:
```python
# Return multiple community reports as separate chunks
chunks = []
for community, report in ranked:
    chunks.append(Chunk(text=report))  # Raw report
return chunks
```

**New Way**:
```python
# Synthesize single integrated answer
synthesized = await self._synthesize_answer(query, community_reports)
return [Chunk(text=synthesized, metadata={"is_synthesized": True})]
```

### 3. Community Detection: Louvain → Leiden

**Old Way**:
```python
# Only Louvain
node_to_community = await self.graph_store.detect_communities_louvain(resolution)
```

**New Way**:
```python
# Leiden with fallback
try:
    node_to_community = await self.graph_store.detect_communities_leiden(resolution)
except:
    node_to_community = await self.graph_store.detect_communities_louvain(resolution)
```

## Key Methods

### GraphRAGGlobalStrategy

```python
# New semantic ranking (replaces keyword matching)
async def _rank_communities_semantic(query: str, top_k: int) -> List[Tuple[Community, float]]

# Fallback keyword ranking
async def _rank_communities_keyword(query: str, top_k: int) -> List[Tuple[Community, float]]

# New synthesis method (core innovation)
async def _synthesize_answer(query: str, community_reports: List[...]) -> str
```

### VectorStore

```python
# Index community embeddings
async def index_communities(community_ids, embeddings, summaries) -> None

# Search communities by similarity
async def search_communities(query_vector, top_k) -> List[Tuple[str, float, Dict]]
```

## Usage Example

```python
from graphunified.strategies.graphrag_global import GraphRAGGlobalStrategy
from graphunified.storage.vector_store import VectorStore

# Initialize with vector store (new parameter)
strategy = GraphRAGGlobalStrategy(
    config=config,
    graph_store=graph_store,
    parquet_store=parquet_store,
    llm_client=llm_client,
    embedding_client=embedding_client,
    vector_store=vector_store,  # NEW: Required for semantic search
)

# Index (now includes embedding communities)
await strategy.index(chunks, entities, relationships, [])

# Retrieve (now returns synthesized answer)
result = await strategy.retrieve("What are the main themes?", top_k=3)

# Check result
print(f"Synthesized: {result.metadata['is_synthesized']}")  # True
print(f"Sources: {result.metadata['source_communities']}")  # [uuid1, uuid2, ...]
print(f"Answer: {result.chunks[0].text}")  # Integrated answer with citations
```

## Performance Expectations

| Operation | Latency | What Happens |
|-----------|---------|--------------|
| Semantic Ranking | ~50ms | Query embedding + vector search |
| Synthesis | ~100-200ms | LLM generates integrated answer |
| Total Retrieval | ~150-250ms | Ranking + synthesis |

**Note**: ~150-250ms is expected and correct (not a bug). The old <1ms implementation was incomplete.

## Verification Checklist

After running `python3 test_graphrag_global.py`:

- [ ] Logs show "Using Leiden algorithm" (or "falling back to Louvain")
- [ ] Retrieval time is ~150-250ms (not <1ms)
- [ ] Output shows "SYNTHESIZED ANSWER" section
- [ ] Answer cites multiple communities (e.g., "Community 1 indicates...")
- [ ] `result.metadata['is_synthesized']` is `True`
- [ ] `result.chunks` contains exactly 1 chunk (the synthesized answer)

## Common Issues

### Issue: "ImportError: No module named igraph"
**Solution**: Install python-igraph
```bash
pip install python-igraph>=0.11.0
```

### Issue: "VectorStore not found"
**Solution**: Make sure you pass `vector_store` parameter when creating strategy
```python
strategy = GraphRAGGlobalStrategy(..., vector_store=vector_store)
```

### Issue: Retrieval still <1ms
**Solution**: Check that semantic ranking is being called
```python
# Should see in logs:
"Ranking X communities for query..."
"Collecting X community reports..."
"Synthesizing answer from community reports..."
```

### Issue: Keyword fallback always triggered
**Solution**: Verify community embeddings are indexed
```python
# Check after indexing:
count = await vector_store.count("communities")
assert count > 0, "No community embeddings indexed"
```

## Migration Guide

### For Existing Code:

1. **Update initialization**:
```python
# Before
strategy = GraphRAGGlobalStrategy(
    config, graph_store, parquet_store, llm_client, embedding_client
)

# After
vector_store = VectorStore.from_config(...)
strategy = GraphRAGGlobalStrategy(
    config, graph_store, parquet_store, llm_client, embedding_client, vector_store
)
```

2. **Update result handling**:
```python
# Before
for chunk in result.chunks:
    print(chunk.text)  # Multiple raw reports

# After
synthesized = result.chunks[0].text  # Single synthesized answer
sources = result.metadata['source_communities']
print(f"Answer: {synthesized}")
print(f"Sources: {sources}")
```

3. **No config changes needed** - everything works with existing settings

## Testing Commands

```bash
# Verify code changes
python3 verify_enhancements.py

# Run integration test
python3 test_graphrag_global.py

# Check syntax
python3 -m py_compile graphunified/strategies/graphrag_global.py
python3 -m py_compile graphunified/storage/vector_store.py
```

## Files to Review

| File | Purpose | Changes |
|------|---------|---------|
| `graphunified/strategies/graphrag_global.py` | Main strategy | 3 new methods, semantic ranking, synthesis |
| `graphunified/storage/vector_store.py` | Vector storage | 2 new methods for communities |
| `test_graphrag_global.py` | Integration test | Updated for vector store |
| `ENHANCEMENTS_SUMMARY.md` | Full documentation | Complete details |
| `verify_enhancements.py` | Verification script | Automated checks |

## Key Takeaways

✅ **Semantic search** replaces keyword matching → Better community ranking
✅ **Synthesis** replaces raw reports → Integrated answers with citations
✅ **Leiden** replaces Louvain-only → Higher quality communities
✅ **~150-250ms latency** is expected → LLM synthesis takes time (worth it!)
✅ **Production-ready** → All error handling and fallbacks in place

## Questions?

Check these files:
- Full details: `ENHANCEMENTS_SUMMARY.md`
- Implementation notes: `.claude/agent-memory/graphrag-expert/phase3-week2-enhancements.md`
- Original assessment: `.claude/agent-memory/graphrag-expert/phase3-week2-review.md`
