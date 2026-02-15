# GraphRAG Expert Memory

## Phase 2 Integration Review

Phase 2 provides solid extraction foundation (60% ready) but missing critical GraphRAG stages:
- ✅ Entity/relationship extraction with deduplication
- ✅ Parquet storage with Community/CommunityReport schemas
- ❌ Graph construction stage (NetworkX/igraph builder)
- ❌ Leiden community detection (hierarchical partitioning)
- ❌ Community summarization (LLM-based map-reduce)

**Key insight**: Current pipeline ends at embedding. GraphRAG value comes from community detection + summarization stages that don't exist yet.

See detailed notes: `phase2-review.md`

## Core Architecture Patterns

### Indexing Pipeline Flow
Documents → Text Units (chunks) → Entities → Relationships → Knowledge Graph → Communities (Leiden) → Community Reports

Key outputs: 6 Parquet files (documents, text_units, entities, relationships, communities, community_reports)
Storage: `./output/<timestamp>/artifacts/*.parquet`

### Search Strategies
- **Local Search**: Entity-focused, explores entity neighborhoods in KG. Best for specific entity questions.
- **Global Search**: Map-reduce over community summaries. Best for thematic/broad questions.
- Both use vector embeddings: entity descriptions + community reports indexed separately

### Vector Indexes
Two distinct indexes created:
1. Entity description embeddings (for local search)
2. Community report embeddings (for global search)

## Unified RAG Integration

See detailed notes: `unified-integration.md`

### Modular Integration Pattern
GraphRAG works as both CLI and library. Key for unified systems:
- Use programmatic API: `from graphrag.index import create_pipeline_config, run_pipeline_with_config`
- Share chunking layer across strategies (one text_units output feeds multiple indexes)
- Maintain separate vector stores per strategy, shared entity extraction
- Cross-reference via document IDs and text unit IDs

### Query Router Pattern
Route based on intent:
- Entity-specific questions → GraphRAG Local Search
- Broad thematic questions → GraphRAG Global Search
- Simple factual → Traditional vector search
- Multi-hop reasoning → GraphRAG Local Search

## Configuration Essentials

See detailed notes: `configuration-patterns.md`

### Key Settings for Unified Systems
- `chunk_size`: 300-1200 tokens (align with other RAG strategies)
- `community_reports_max_length`: Controls global search context size
- `max_cluster_size`: 10-50 (affects community granularity)
- `max_gleanings`: 1-3 for entity extraction refinement

### Prompt Tuning Workflow
```bash
graphrag prompt-tune --root ./data --output ./prompts
```
Generates domain-specific prompts for: entity extraction, summarization, community reports

## CLI Quick Reference

### Indexing
```bash
graphrag index --root ./data --verbose
```

### Querying
```bash
graphrag query --method local --query "question" --model <model>
graphrag query --method global --query "question" --model <model>
```

### Configuration
Config file: `./settings.yaml` or `./settings.json`
Environment variables: GRAPHRAG_API_KEY, GRAPHRAG_LLM_MODEL

## Parquet Schema Overview

### entities.parquet
- id, title, type, description, text_unit_ids, document_ids
- description_embedding (vector for local search)

### relationships.parquet
- source, target, description, weight, text_unit_ids

### communities.parquet
- id, title, level (hierarchy), entity_ids

### community_reports.parquet
- community_id, title, summary, findings, rating
- summary_embedding (vector for global search)

## Key Code Modules

### Programmatic Usage
```python
from graphrag.index import create_pipeline_config, run_pipeline_with_config
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.structured_search.local_search import LocalSearch
from graphrag.query.structured_search.global_search import GlobalSearch
```

### Index Pipeline Location
`graphrag/index/` - pipeline configuration and execution
`graphrag/query/` - local and global search implementations
`graphrag/config/` - configuration models and defaults

## Common Issues & Solutions

### Entity Extraction Quality
- Tune prompts with `graphrag prompt-tune` for domain-specific terminology
- Increase `max_gleanings` (1-3) for better entity coverage
- Review entity types in prompts to match domain

### Performance & Cost
- Monitor token usage during indexing (entity extraction is most expensive)
- Reduce `chunk_size` or `max_gleanings` to control costs
- Use caching for repeated entity extractions

### Graph Quality
- Check entity/relationship counts in Parquet files
- Validate community detection: hierarchical levels should be 2-4
- Review community reports for coherence

## Version-Specific Notes

Current stable version patterns (as of Jan 2025):
- Leiden algorithm default for community detection
- OpenAI/Azure OpenAI primary LLM backends
- Parquet storage format standard (not CSV)
- Vector storage: FAISS, Lance, or external stores

## Phase 3 Week 2 Implementation Review

**GraphRAG Local**: 85% complete, production-ready ✅
- Entity search → BFS traversal → chunk collection
- Performance: ~50ms, excellent
- Minor issue: Chunk loading scalability (loads all chunks)

**GraphRAG Global**: 95% complete, production-ready ✅ (ENHANCED)
- ✅ Leiden algorithm with Louvain fallback
- ✅ Embedding-based community ranking (semantic similarity)
- ✅ Map-reduce answer synthesis with LLM
- Performance: ~150-250ms (expected for synthesis)
- Remaining: Multi-level hierarchical communities

See detailed notes:
- `phase3-week2-review.md` (original assessment)
- `phase3-week2-enhancements.md` (implementation details)

## Related Memory Files

- `phase2-review.md` - Phase 2 indexing pipeline assessment
- `phase3-week2-review.md` - Phase 3 GraphRAG Local/Global implementation review
- `phase3-week2-enhancements.md` - GraphRAG Global enhancements (Leiden, embeddings, synthesis)
- `unified-integration.md` - Detailed patterns for multi-strategy RAG systems
- `configuration-patterns.md` - Complete configuration reference and tuning guide
- `troubleshooting.md` - Common errors and debugging workflows
