# Phase 3 Detailed Review Notes

## Showstopper: Chunk.entity_ids Never Populated
- Phase 2 ExtractStage never writes entity_ids/relationship_ids to Chunk objects
- GraphRAG Local _collect_connected_chunks filters by `chunk.entity_ids` -- always empty
- LightRAG _collect_entity_chunks filters by `chunk.entity_ids` -- always empty
- HippoRAG _collect_activated_chunks filters by `chunk.entity_ids` -- always empty
- This means 4 of 6 strategies cannot retrieve chunks in graph-based mode

## O(n) Full Parquet Scan Per Query
- GraphRAG Local: `async for chunk in self.parquet_store.load_chunks()` line 384
- LightRAG: `async for chunk in self.parquet_store.load_chunks()` line 611
- HippoRAG: `async for chunk in self.parquet_store.load_chunks()` line 455
- Each query loads ALL chunks from disk, iterates, filters by entity_ids
- At scale this is a query-time disaster

## Duplicated Code Across Strategies
- Chunk reconstruction logic: identical in naive.py (lines 170-198) and hybrid.py (lines 352-388)
- Entity caching pattern: identical `{str(e.id): e for e in entities}` in graphrag_local, graphrag_global, lightrag, hipporag
- Community ranking by keyword match: duplicated in graphrag_global._rank_communities and lightrag._rank_communities
- Graph expansion BFS: duplicated in graphrag_local._expand_to_local_graph and lightrag._expand_local_graph

## Strategy Config Mismatch
- Strategy __init__ uses `getattr(config, 'top_k', 10)` because config classes lack these fields
- NaiveStrategyConfig has only `enabled: bool`
- HybridStrategyConfig has `enabled, alpha, bm25_k1, bm25_b`
- GraphRAGStrategyConfig has `enabled, leiden_resolution, max_community_size, generate_reports`
- Missing from configs: top_k, max_hops, damping_factor, num_seed_entities, local_weight, rrf_k

## GraphRAG Global Issues
- Community ranking uses naive keyword matching (`sum(1 for token in query_tokens if token in report_lower)`)
- Has embedding_client but never uses it for ranking (should use vector similarity on reports)
- Community reports stored in-memory only, not persisted
- Report generation has no caching -- regenerated every time
- `self._communities.index(community)` is O(n) linear scan per community

## LightRAG Issues
- Query classification is keyword-based with hardcoded word lists
- "what are" is in _global_keywords AND matches pattern for local queries containing "what"
- Overlap between keyword sets causes unpredictable routing
- Hybrid mode does not deduplicate chunks between local and global results
- `result.metadata["auto_classified"] = mode is None` -- always False (mode was just set above)

## HippoRAG Issues
- PPR uses `nx.pagerank()` which is global PageRank, not true Personalized PageRank
- Actually nx.pagerank with personalization IS PPR, so this is correct
- Damping factor convention: nx.pagerank alpha = probability of following edge (0.85 standard)
- Good: converts directed to undirected for PPR (correct for associative recall)
- But `self.graph_store.directed` attribute access -- not async safe if loaded lazily

## VectorStore Issues
- search_relationships and search_facts use different LanceDB API pattern than search_chunks/search_entities
- search_chunks/entities: `table.search(query).limit(top_k).to_pandas()` (correct async pattern)
- search_relationships: `table.search(query, limit=top_k)` (different API, not using limit() method)
- Inconsistent API usage may cause bugs with different LanceDB versions
- Length validation bug: `len(chunk_ids) != len(embeddings) != len(texts)` -- this is chained comparison, not pairwise!
  - `a != b != c` evaluates as `a != b and b != c`, so [1,2,3] would pass when all are different lengths

## Score Normalization Inconsistency
- Naive RAG: scores = 1/(1+distance), range [0, 1]
- Hybrid RAG: RRF scores = weighted sum of 1/(k+rank), range ~[0, 0.03]
- GraphRAG Local: scores = entity_overlap_ratio, range [0, 1]
- GraphRAG Global: scores = keyword_match_ratio, range [0, 1]
- LightRAG: varies by mode (entity ratio or keyword ratio)
- HippoRAG: 0.5*count_score + 0.5*ppr_score_sum, range varies wildly
- No normalization makes cross-strategy comparison impossible for a query router

## Missing from Phase 3
- No strategy tests at all
- No integration test for end-to-end query flow
- No query router (planned Phase 4)
- No response generation (retrieve only, no LLM answer synthesis)
- No embedding caching (still)
- CommunityReport model exists but unused (GraphRAG Global creates reports as strings only)
- Fact model exists but unused (HippoRAG should use fact-based retrieval per paper)
- EntityChunkEdge model exists but unused
