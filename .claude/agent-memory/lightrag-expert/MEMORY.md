# LightRAG Expert Memory

## Core Architecture

**Dual-Index System:**
- Entity index: Entities with vector embeddings for local search
- Relationship index: Relationship descriptions with embeddings for global search
- Key insight: Relationship descriptions enable thematic/conceptual queries
- Four modes: local (entity), global (relationship), hybrid, naive

**Storage Files:**
- `vdb_entities.json`: Entity vectors and metadata
- `vdb_relationships.json`: Relationship vectors and metadata
- `graph_chunk_entity_relation.graphml`: Graph structure with bidirectional references

**Graph Philosophy:**
- Flat entity-relation graph (no hierarchical communities)
- Direct search over entities/relationships vs community aggregation
- Enables fast, granular multi-hop reasoning

## Integration Patterns for Unified Systems

**Component Sharing:**
- SHARED: Document chunking, entity extraction (LLM calls)
- UNIQUE: Relationship description generation (additional LLM call per relationship)
- COMPATIBLE: Can coexist with GraphRAG/HippoRAG on same corpus

**Storage Organization:**
- Separate storage paths per strategy (avoid index conflicts)
- Storage overhead: ~2-3x vs vector-only (entities + relationships + graph)
- Incremental updates: Add new chunks/entities without full rebuild

**Query Routing Criteria:**
- Use LightRAG local for: Entity-centric queries, multi-hop facts
- Use LightRAG global for: Thematic questions, relationship patterns
- Use GraphRAG for: Corpus-level themes, high-level summaries
- Use HippoRAG for: Episodic retrieval, temporal/contextual similarity

See detailed notes:
- [graphrag-comparison.md](graphrag-comparison.md) - Detailed LightRAG vs GraphRAG analysis
- [relationship-indexing.md](relationship-indexing.md) - Relationship description generation
- [unified-integration.md](unified-integration.md) - Multi-strategy system patterns
- [phase2-review.md](phase2-review.md) - Phase 2 implementation review and critical gaps

## Performance Characteristics

**Indexing:**
- Time: O(n) entity extraction + O(r) relationship generation (r = num relationships)
- Cost: Moderate (2 LLM calls: entity extraction + relationship descriptions)
- Incremental: Easy (append to indexes, extend graph)

**Query:**
- Latency: Fast (vector search + graph traversal, no community aggregation)
- Local search: Single-hop entity retrieval
- Global search: Relationship-based thematic exploration
- Hybrid: Combines both for comprehensive coverage

**Memory:**
- Higher than vector-only due to relationship index
- Lower than GraphRAG with large community hierarchies
- Scales linearly with document count

## Common Patterns

**Entity-Relationship Linking:**
```
Chunk → Entities → Relationships
  ↑         ↓            ↓
  └─────────┴────────────┘
   (bidirectional references)
```

**Search Mode Selection Logic:**
- Query mentions specific entities? → Local search
- Query asks about themes/patterns? → Global search
- Complex multi-aspect query? → Hybrid search
- Need baseline comparison? → Naive search

**Relationship Description Quality:**
- Key to global search effectiveness
- Should capture semantic significance, not just "X relates to Y"
- Example: "Entity A enables Entity B through mechanism C in context D"
- Prompt engineering critical for domain-specific graphs

## Integration with graph-unified Project

**Phase 2 Status (100% Complete):**
- ✅ Entity extraction with descriptions, confidence, source tracking
- ✅ Relationship extraction with descriptions, type classification
- ✅ Entity embeddings (name + description format)
- ✅ Chunk embeddings
- ✅ Relationship embeddings (VERIFIED: EmbedStage lines 81-85, 204-268)
- ✅ Fuzzy entity deduplication (90% threshold, 30-50% reduction)
- ✅ Bidirectional graph links (chunks ↔ entities ↔ relationships)

**Relationship Embedding Format:**
```
"{source.name} {rel.type} {target.name}: {rel.description}"
Example: "IPCC WORKS_FOR United Nations: IPCC scientists work for UN organization"
```

**Key Considerations:**
- Leverage shared entity extraction across strategies
- Design query router to select optimal strategy per query type
- Monitor storage overhead with multiple indexes
- Relationship embeddings are critical for global search quality

## Anti-Patterns to Avoid

1. Don't use LightRAG for corpus-wide summarization (use GraphRAG communities)
2. Don't skip relationship descriptions to save costs (degrades global search)
3. Don't skip relationship embeddings (blocks global/hybrid search)
4. Don't mix storage paths between strategies (causes index corruption)
5. Don't use only local search (misses thematic context)
6. Don't rebuild entire graph for small updates (use incremental indexing)
7. **Don't use community-based retrieval for LightRAG global search** (use relationship embeddings)

## Phase 3 Implementation Status

**Fix Date:** 2026-02-15
**Status:** ✅ FIXED - Now 100% aligned with LightRAG architecture

**Original Issues (RESOLVED):**
1. ✅ **FIXED**: Global search now uses relationship embeddings (not communities)
2. ✅ **FIXED**: Removed `_rank_communities()` method
3. ✅ **CORRECT**: Local search working properly (entity + BFS)
4. ✅ **FIXED**: Global search now uses semantic vector search (40-80ms expected)

**Changes Applied:**
- Rewrote `_global_retrieval()` to search relationship embeddings
- Added `_search_relationships()` helper method
- Removed all community dependencies
- Updated `index()`, `validate_index()` methods
- Updated test_lightrag.py to verify relationship-based retrieval

**Current Implementation:**
- File: `/Users/aselaillayapparachchi/code/GraphRAG/Microsoft/graph-unified/graphunified/strategies/lightrag.py`
- Lines 419-464: `_global_retrieval()` uses relationship search ✅
- New method: `_search_relationships()` for semantic relationship search ✅
- Lines 345-417: `_local_retrieval()` remains correct ✅

**See:**
- [phase3-review.md](phase3-review.md) for original issue identification
- [phase3-fix.md](phase3-fix.md) for detailed fix documentation

## References

- Official repo: https://github.com/HKUDS/LightRAG
- Paper: Graph-based RAG with dual-level retrieval
- graph-unified project: /Users/aselaillayapparachchi/code/GraphRAG/Microsoft/graph-unified
