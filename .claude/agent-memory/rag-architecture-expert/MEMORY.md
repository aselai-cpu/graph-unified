# Graph-Unified RAG System - Architecture Knowledge

## Project Overview
- Multi-strategy RAG system: Naive, Hybrid, GraphRAG (Local/Global), LightRAG, HippoRAG
- Shared extraction pipeline concept across all strategies
- Python 3.10+, Pydantic v2, async throughout

## Key File Paths
- Data models: `graphunified/config/models.py`
- Settings: `graphunified/config/settings.py`
- Defaults: `graphunified/config/defaults.py`
- Vector store: `graphunified/storage/vector_store.py` (LanceDB, 5 indexes)
- Graph store: `graphunified/storage/graph_store.py` (NetworkX DiGraph, pickle/graphml)
- Parquet storage: `graphunified/storage/parquet_store.py`
- BM25 index: `graphunified/index/stages/index.py` (BM25Index class + IndexStage)
- Strategy base: `graphunified/strategies/base.py` (RetrievalStrategy ABC, RetrievalResult, QueryType)
- Strategies: `graphunified/strategies/{naive,hybrid,graphrag_local,graphrag_global,lightrag,hipporag}.py`
- Embedding factory: `graphunified/utils/embedding_factory.py`
- Local embeddings: `graphunified/utils/local_embedding.py` (sentence-transformers)
- Exceptions: `graphunified/exceptions.py`

## Phase 3 Architecture
- RetrievalStrategy ABC: index(), retrieve(), supports_query_type(), name, requires_*
- RetrievalResult: unified model (chunks, scores, entities, relationships, communities, metadata)
- QueryType enum: FACTOID, EXPLORATORY, RELATIONAL, THEMATIC, COMPARATIVE, TEMPORAL
- VectorStore (LanceDB): chunk/entity/relationship/fact/community indexes
- GraphStore (NetworkX): directed graph, Louvain/Leiden, BFS traversal
- BM25Index: in-memory inverted index
- 6 strategies, no query router yet (Phase 4)

## Phase 3 Critical Issues
- See `phase3-review.md` for full analysis
- GraphRAG Local/LightRAG/HippoRAG: load ALL chunks from Parquet per query (O(n) scan)
- Chunk.entity_ids never populated (Phase 2 bug), breaks all graph-based strategies
- GraphRAG Global: keyword-based community ranking, not vector-based
- LightRAG: keyword-based query classification (fragile)
- Code duplication: chunk reconstruction, entity caching across 4 strategies
- Strategy config classes lack top_k, max_hops, damping_factor fields (use getattr)
- No tests for any of the 6 strategies

## Technology Choices
- LLM: Claude (Anthropic SDK), default claude-3-5-sonnet-20241022
- Embeddings: Voyage AI or local sentence-transformers (BAAI/bge-large-en-v1.5)
- Storage: Parquet + LanceDB + NetworkX
- Tokenizer: tiktoken cl100k_base
- Community detection: Louvain (NetworkX), Leiden (optional igraph)

## Phase 2 Critical Findings (still relevant)
- See `phase2-review.md`
- Chunk.entity_ids/relationship_ids never populated during extraction
- Fact/EntityChunkEdge models exist but never produced
- Embeddings not persisted to Parquet
- Sequential LLM calls, no concurrency in extraction
