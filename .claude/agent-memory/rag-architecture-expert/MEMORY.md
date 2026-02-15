# Graph-Unified RAG System - Architecture Knowledge

## Project Overview
- Multi-strategy RAG system: Naive, Hybrid, GraphRAG (Local/Global), LightRAG, HippoRAG
- Shared extraction pipeline concept across all strategies
- Python 3.10+, Pydantic v2, async throughout

## Key File Paths
- Data models: `graphunified/config/models.py` (Document, Chunk, Entity, Relationship, Community, CommunityReport, Fact, EntityChunkEdge)
- Settings: `graphunified/config/settings.py` (Pydantic-based, YAML loading, env var substitution)
- Defaults: `graphunified/config/defaults.py`
- Storage interface: `graphunified/storage/base.py` (StorageBackend ABC)
- Parquet storage: `graphunified/storage/parquet_store.py` (partition-based writes, async via to_thread)
- Parquet schemas: `graphunified/storage/schemas.py` (PyArrow schemas)
- Pipeline orchestrator: `graphunified/index/pipeline.py`
- Pipeline stages: `graphunified/index/stages/{base,load,chunk,extract,embed}.py`
- Prompts: `graphunified/prompts/extraction.py`
- LLM client: `graphunified/utils/llm.py` (ClaudeClient with RateLimiter)
- Embedding client: `graphunified/utils/embedding.py` (Voyage AI, batching, normalization)
- Tokenizer: `graphunified/utils/tokenizer.py` (tiktoken, cl100k_base)
- Exceptions: `graphunified/exceptions.py` (hierarchy with base GraphUnifiedError)
- Config profiles: `settings-dev.yaml`, `settings-prod.yaml`, `settings-research.yaml`
- Tests: `tests/unit/index/test_extract.py`, `tests/integration/test_pipeline.py`

## Technology Choices
- LLM: Claude (Anthropic SDK), default claude-3-5-sonnet-20241022
- Embeddings: Voyage AI (voyage-3), 1024 dimensions, batch_size=128
- Storage: Parquet (PyArrow) partition-based for structured data, LanceDB planned for vectors
- Tokenizer: tiktoken cl100k_base
- Entity dedup: fuzzywuzzy library (ratio-based), 90% threshold
- Graph format: GraphML (configurable)
- Rate limiting: Custom sliding window (requests + tokens per minute)
- Retry: tenacity library

## Phase 1 Observations
- Storage had read-all-then-iterate antipattern (FIXED in Phase 2 with partitions)
- No delete/update operations in StorageBackend (still true)
- No embedding caching despite config flag existing (still true)
- LLM client tightly coupled to Anthropic (still true)
- Entity name normalization only strips whitespace (still true)

## Phase 2 Pipeline Architecture
- 4 stages: LoadStage -> ChunkStage -> ExtractStage -> EmbedStage
- Sequential execution (no parallelism despite docstring claiming "DAG execution")
- Extraction: entity extraction then relationship extraction (2 LLM calls per chunk batch)
- Fuzzy dedup at 90% threshold, grouped by entity type
- Separate embeddings for chunks (raw text) and entities (name + description)

## Phase 2 Review - Critical Findings
- See `phase2-review.md` for full analysis
- Key: sequential LLM calls, no concurrency in extraction batches
- O(n^2) entity dedup clustering algorithm
- Missing outputs: Fact, EntityChunkEdge, Community (needed by HippoRAG/GraphRAG)
- Chunk entity_ids/relationship_ids never populated
- Chunking ignores respect_boundaries config
- Checkpoint system is placeholder (no resume logic)
- Embeddings not stored in Parquet (chunk_to_dict omits them)
- No embedding caching
- Prompt mismatch: TECHNOLOGY/DATE types in enum but not in prompt
