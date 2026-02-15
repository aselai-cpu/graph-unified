# Phase 2 Detailed Review Notes

## Extraction Concurrency
- _extract_entities loops over batches sequentially (no asyncio.gather)
- _extract_relationships also sequential
- Rate limiter supports concurrent access but pipeline never exploits it
- max_concurrent config exists (default=10) but is never used in ExtractStage

## Entity Dedup Algorithm
- O(n^2) clustering: nested loop comparing all pairs within each entity type
- Iterative merge loop (while merged) adds another pass each time clusters merge
- fuzz.ratio is character-level edit distance, not semantic
- Will not catch "NASA" vs "National Aeronautics and Space Administration" (threshold=90)

## Prompt Issues
- Entity types in prompt: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT
- Entity types in enum: adds TECHNOLOGY, DATE, OTHER
- Relationship types in prompt: RELATED_TO, PART_OF, LOCATED_IN, WORKS_FOR, CAUSES
- Relationship types in enum: adds SIMILAR_TO, OPPOSITE_OF, PRECEDES, OTHER
- Extraction config has entity_types and relationship_types lists but prompts are hardcoded

## Missing Downstream Outputs
- Fact model exists but no FactExtractionStage (needed for HippoRAG)
- EntityChunkEdge model exists but never populated (needed for HippoRAG bipartite graph)
- Community model exists but no community detection stage (needed for GraphRAG)
- Chunk.entity_ids and Chunk.relationship_ids are never populated during extraction

## Storage Gaps
- chunk_to_dict omits embedding and embedding_model fields
- entity_to_dict omits embedding and embedding_model fields
- relationship_to_dict omits embedding and embedding_model fields
- Embeddings computed but never persisted to Parquet

## Chunking Limitations
- Token-based fixed windows only, despite config supporting "sentence", "paragraph", "semantic"
- respect_boundaries config exists but ChunkStage ignores it
- Encoding object created per-document (could cache)
- Character position calculation via decode_tokens is inefficient (O(n) per chunk)
