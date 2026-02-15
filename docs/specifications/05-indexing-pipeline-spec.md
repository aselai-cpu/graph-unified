# Indexing Pipeline Specification

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document specifies the complete indexing pipeline that transforms raw documents into searchable indexes across all six retrieval strategies. Every stage is defined with input/output contracts, algorithms, error handling, and performance characteristics.

## Pipeline Architecture

### Stage Dependency Graph

```
┌─────────┐
│  Load   │
└────┬────┘
     │
┌────▼────┐
│  Chunk  │
└────┬────┘
     │
     ├─────────────────────┬─────────────────────┐
     │                     │                     │
┌────▼────┐          ┌────▼────┐          ┌────▼────┐
│ Extract │          │  Embed  │          │   BM25  │
│         │          │ Chunks  │          │  Build  │
└────┬────┘          └────┬────┘          └────┬────┘
     │                     │                     │
     ├──────┬──────┬───────┴─────────┐           │
     │      │      │                 │           │
┌────▼────┐│┌────▼────┐         ┌──▼──────┐ ┌───▼────┐
│  Graph  │││  Embed  │         │  Naive  │ │ Hybrid │
│  Build  │││Entities │         │  Index  │ │ Index  │
└────┬────┘│└────┬────┘         └─────────┘ └────────┘
     │     │     │
┌────▼─────▼─────▼──────┐
│  Strategy Builders     │
│  - GraphRAG            │
│  - LightRAG            │
│  - HippoRAG            │
└────────────────────────┘
```

### Execution Model

- **Async DAG:** Stages execute asynchronously with dependency resolution
- **Parallel execution:** Independent stages run in parallel
- **Batching:** All LLM/embedding calls batched for efficiency
- **Error handling:** Stage failures logged, pipeline continues if possible
- **Progress tracking:** Callback-based progress reporting

---

## Stage 1: Load Documents

### Purpose

Load documents from input directory into memory structures.

### Input

- `input_path`: Path to directory containing documents
- `config`: Configuration dict with supported file types

### Processing

```python
from pathlib import Path
from typing import List, Dict, Any
from graphunified.config.models import Document
import hashlib

class LoadStage:
    """Load documents from disk."""

    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.html'}

    async def execute(
        self,
        input_data: Path,
        context: Dict[str, Any]
    ) -> StageResult:
        """Load documents from input_path."""
        documents = []

        # Recursively find documents
        for filepath in input_data.rglob('*'):
            if filepath.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                text = await self._read_file(filepath)
                doc = Document(
                    filename=str(filepath.relative_to(input_data)),
                    text=text,
                    metadata={
                        "file_type": filepath.suffix,
                        "file_size_bytes": filepath.stat().st_size,
                        "file_hash": self._hash_file(filepath)
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
                # Continue with other documents

        return StageResult(
            status="success",
            data=documents,
            metadata={
                "document_count": len(documents),
                "total_chars": sum(len(d.text) for d in documents)
            }
        )

    async def _read_file(self, filepath: Path) -> str:
        """Read file contents based on extension."""
        if filepath.suffix == '.txt' or filepath.suffix == '.md':
            return filepath.read_text(encoding='utf-8')
        elif filepath.suffix == '.pdf':
            return await self._read_pdf(filepath)
        elif filepath.suffix == '.docx':
            return await self._read_docx(filepath)
        elif filepath.suffix == '.html':
            return await self._read_html(filepath)
        else:
            raise ValueError(f"Unsupported file type: {filepath.suffix}")

    def _hash_file(self, filepath: Path) -> str:
        """Compute SHA256 hash of file."""
        return hashlib.sha256(filepath.read_bytes()).hexdigest()
```

### Output

```python
List[Document]  # List of loaded documents
```

### Error Handling

- **File read errors:** Log and skip file
- **Encoding errors:** Try alternate encodings (latin-1, cp1252)
- **Empty files:** Skip with warning
- **Corrupted files:** Skip with error log

### Performance

- **Parallelization:** Read files in parallel (asyncio)
- **Memory:** Stream large files (>10MB)
- **Typical duration:** 1-2 seconds for 1000 documents

---

## Stage 2: Chunk Documents

### Purpose

Split documents into overlapping chunks for embedding and retrieval.

### Input

```python
List[Document]  # From Load stage
```

### Configuration

```yaml
chunking:
  strategy: "fixed"          # fixed | sentence | paragraph | semantic
  chunk_size: 512            # Tokens
  chunk_overlap: 64          # Tokens
  respect_boundaries: true   # Respect sentence boundaries
  encoding_name: "cl100k_base"
```

### Processing

```python
import tiktoken
from typing import List
from graphunified.config.models import Document, Chunk

class ChunkStage:
    """Chunk documents using tiktoken."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.tokenizer = tiktoken.get_encoding(config.encoding_name)

    async def execute(
        self,
        input_data: List[Document],
        context: Dict[str, Any]
    ) -> StageResult:
        """Chunk all documents."""
        all_chunks = []

        for doc in input_data:
            doc_chunks = self._chunk_document(doc)
            all_chunks.extend(doc_chunks)

        return StageResult(
            status="success",
            data=all_chunks,
            metadata={
                "chunk_count": len(all_chunks),
                "avg_tokens_per_chunk": sum(c.token_count for c in all_chunks) / len(all_chunks)
            }
        )

    def _chunk_document(self, doc: Document) -> List[Chunk]:
        """Chunk a single document."""
        # Tokenize full text
        tokens = self.tokenizer.encode(doc.text)
        doc.token_count = len(tokens)

        chunks = []
        chunk_index = 0
        start_token = 0

        while start_token < len(tokens):
            # Extract chunk tokens
            end_token = min(start_token + self.config.chunk_size, len(tokens))
            chunk_tokens = tokens[start_token:end_token]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Find character positions in original text
            start_char, end_char = self._find_char_positions(
                doc.text,
                chunk_text,
                start_token
            )

            # Create chunk
            chunk = Chunk(
                document_id=doc.id,
                chunk_index=chunk_index,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                token_count=len(chunk_tokens),
                metadata={"document_filename": doc.filename}
            )
            chunks.append(chunk)

            # Advance with overlap
            start_token += (self.config.chunk_size - self.config.chunk_overlap)
            chunk_index += 1

            # Break if we've covered the full text
            if end_token == len(tokens):
                break

        return chunks

    def _find_char_positions(
        self,
        full_text: str,
        chunk_text: str,
        token_offset: int
    ) -> tuple[int, int]:
        """Find character positions of chunk in full text."""
        # Simple implementation: search for chunk_text
        # More robust: track cumulative char positions during tokenization
        start_char = full_text.find(chunk_text)
        if start_char == -1:
            # Fallback: estimate from token positions
            start_char = token_offset * 4  # Rough estimate: 1 token ≈ 4 chars
        end_char = start_char + len(chunk_text)
        return start_char, end_char
```

### Output

```python
List[Chunk]  # All chunks across all documents
```

### Algorithm Details

**Fixed Strategy (default):**

1. Tokenize full document text
2. Create overlapping windows of `chunk_size` tokens
3. Advance by `chunk_size - chunk_overlap` tokens
4. Decode tokens back to text
5. Track character positions

**Sentence Boundary Respect:**

If `respect_boundaries: true`:
- Use NLTK or spaCy for sentence segmentation
- Extend chunk to nearest sentence boundary
- Ensure minimum chunk size (allow +20% variation)

### Error Handling

- **Tokenization errors:** Log and skip document
- **Empty chunks:** Discard (should not occur)
- **Oversized chunks:** Warn if chunk exceeds 2x chunk_size

### Performance

- **Parallelization:** Chunk documents in parallel
- **Batching:** Process 100 documents per batch
- **Typical duration:** 10-30 seconds for 1000 documents

---

## Stage 3: Extract Entities & Relationships

### Purpose

Extract entities and relationships from chunks using Claude.

### Input

```python
List[Chunk]  # From Chunk stage
```

### Configuration

```yaml
extraction:
  entity_types: ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT"]
  relationship_types: ["RELATED_TO", "PART_OF", "LOCATED_IN", "WORKS_FOR"]
  max_gleanings: 1           # Additional extraction passes
  min_confidence: 0.7        # Filter low-confidence extractions
```

### Processing

```python
from graphunified.utils.llm import LLMClient
from graphunified.config.models import Entity, Relationship
import json

class ExtractStage:
    """Extract entities and relationships using Claude."""

    def __init__(self, llm_client: LLMClient, config: ExtractionConfig):
        self.llm = llm_client
        self.config = config
        self.prompt_template = self._load_prompt_template()

    async def execute(
        self,
        input_data: List[Chunk],
        context: Dict[str, Any]
    ) -> StageResult:
        """Extract from all chunks."""
        all_entities = []
        all_relationships = []

        # Batch process chunks
        batches = self._create_batches(input_data, batch_size=10)

        for batch in batches:
            entities, relationships = await self._extract_batch(batch)
            all_entities.extend(entities)
            all_relationships.extend(relationships)

        # Deduplicate entities
        deduped_entities = self._deduplicate_entities(all_entities)

        # Resolve relationship entity IDs
        resolved_relationships = self._resolve_relationships(
            all_relationships,
            deduped_entities
        )

        return StageResult(
            status="success",
            data=(deduped_entities, resolved_relationships),
            metadata={
                "entity_count": len(deduped_entities),
                "relationship_count": len(resolved_relationships),
                "llm_calls": len(batches)
            }
        )

    async def _extract_batch(
        self,
        chunks: List[Chunk]
    ) -> tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from chunk batch."""
        # Prepare prompt
        chunk_texts = "\n\n---\n\n".join([
            f"CHUNK {i+1}:\n{chunk.text}"
            for i, chunk in enumerate(chunks)
        ])

        prompt = self.prompt_template.format(
            chunk_texts=chunk_texts,
            entity_types=", ".join(self.config.entity_types),
            relationship_types=", ".join(self.config.relationship_types)
        )

        # Call LLM
        response = await self.llm.generate(
            prompt,
            temperature=0.0,  # Deterministic
            max_tokens=4096
        )

        # Parse JSON response
        try:
            data = json.loads(response)
            entities = self._parse_entities(data["entities"], chunks)
            relationships = self._parse_relationships(data["relationships"], chunks)
            return entities, relationships
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse extraction: {e}")
            return [], []

    def _parse_entities(
        self,
        entity_data: List[Dict],
        chunks: List[Chunk]
    ) -> List[Entity]:
        """Parse entity data from LLM response."""
        entities = []
        for item in entity_data:
            entity = Entity(
                name=item["name"],
                type=item["type"],
                description=item.get("description"),
                source_chunks=[chunk.id for chunk in chunks],
                extraction_confidence=item.get("confidence", 1.0)
            )
            # Filter by confidence threshold
            if entity.extraction_confidence >= self.config.min_confidence:
                entities.append(entity)
        return entities

    def _parse_relationships(
        self,
        rel_data: List[Dict],
        chunks: List[Chunk]
    ) -> List[Relationship]:
        """Parse relationship data from LLM response."""
        relationships = []
        for item in rel_data:
            # Note: entity IDs will be resolved after deduplication
            rel = Relationship(
                source_entity_name=item["source"],  # Temporary: name not ID
                target_entity_name=item["target"],
                type=item["type"],
                description=item.get("description"),
                source_chunks=[chunk.id for chunk in chunks],
                extraction_confidence=item.get("confidence", 1.0)
            )
            if rel.extraction_confidence >= self.config.min_confidence:
                relationships.append(rel)
        return relationships

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge duplicate entities using fuzzy matching."""
        from fuzzywuzzy import fuzz

        # Group by type
        by_type = {}
        for entity in entities:
            by_type.setdefault(entity.type, []).append(entity)

        deduped = []
        for entity_type, type_entities in by_type.items():
            # Cluster similar entities
            clusters = []
            for entity in type_entities:
                # Find matching cluster
                matched = False
                for cluster in clusters:
                    representative = cluster[0]
                    similarity = fuzz.ratio(
                        entity.name.lower(),
                        representative.name.lower()
                    )
                    if similarity >= 90:  # 90% similarity threshold
                        cluster.append(entity)
                        matched = True
                        break

                if not matched:
                    clusters.append([entity])

            # Merge each cluster
            for cluster in clusters:
                merged = self._merge_entity_cluster(cluster)
                deduped.append(merged)

        return deduped

    def _merge_entity_cluster(self, cluster: List[Entity]) -> Entity:
        """Merge a cluster of similar entities."""
        # Use most common name
        names = [e.name for e in cluster]
        canonical_name = max(set(names), key=names.count)

        # Merge descriptions (longest)
        descriptions = [e.description for e in cluster if e.description]
        canonical_description = max(descriptions, key=len) if descriptions else None

        # Union of source chunks
        all_chunks = set()
        for entity in cluster:
            all_chunks.update(entity.source_chunks)

        # Max confidence
        max_confidence = max(e.extraction_confidence for e in cluster)

        # Collect aliases
        aliases = [e.name for e in cluster if e.name != canonical_name]

        return Entity(
            name=canonical_name,
            type=cluster[0].type,
            description=canonical_description,
            source_chunks=list(all_chunks),
            extraction_confidence=max_confidence,
            aliases=aliases
        )

    def _resolve_relationships(
        self,
        relationships: List[Relationship],
        entities: List[Entity]
    ) -> List[Relationship]:
        """Resolve entity names to entity IDs in relationships."""
        # Build name → entity mapping
        entity_map = {}
        for entity in entities:
            entity_map[entity.name.lower()] = entity
            for alias in entity.aliases:
                entity_map[alias.lower()] = entity

        resolved = []
        for rel in relationships:
            source = entity_map.get(rel.source_entity_name.lower())
            target = entity_map.get(rel.target_entity_name.lower())

            if source and target:
                rel.source_entity_id = source.id
                rel.target_entity_id = target.id
                resolved.append(rel)
            else:
                logger.warning(
                    f"Could not resolve relationship: {rel.source_entity_name} "
                    f"→ {rel.target_entity_name}"
                )

        return resolved
```

### Output

```python
Tuple[List[Entity], List[Relationship]]
```

### Prompt Template

```python
EXTRACTION_PROMPT = """
You are an expert at extracting structured information from text.

Extract entities and relationships from the following text chunks.

ENTITY TYPES: {entity_types}
RELATIONSHIP TYPES: {relationship_types}

TEXT CHUNKS:
{chunk_texts}

Return a JSON object with this structure:
{{
  "entities": [
    {{"name": "entity name", "type": "ENTITY_TYPE", "description": "brief description", "confidence": 0.95}}
  ],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "type": "RELATIONSHIP_TYPE", "description": "brief description", "confidence": 0.90}}
  ]
}}

Guidelines:
- Extract only entities of the specified types
- Be specific: "United Nations" not "UN organization"
- Include confidence score (0.0-1.0)
- Relationships must connect entities you extracted
- Omit uncertain extractions (confidence < 0.7)
"""
```

### Error Handling

- **LLM errors:** Retry with exponential backoff (3 attempts)
- **JSON parse errors:** Log and skip batch
- **Rate limiting:** Respect rate limits (50 RPM)
- **Empty responses:** Log warning and continue

### Performance

- **Batching:** 10 chunks per LLM call
- **Parallelization:** 10 concurrent LLM calls
- **Typical duration:** 15-30 minutes for 10K chunks
- **Cost:** ~$30 for 10K chunks (1.5M tokens)

---

## Stage 4: Embed Chunks

### Purpose

Generate embeddings for chunks using Voyage AI.

### Input

```python
List[Chunk]  # From Chunk stage
```

### Configuration

```yaml
embedding:
  provider: "voyage"
  model: "voyage-3"
  dimension: 1024
  batch_size: 128
  normalize: true
```

### Processing

```python
from graphunified.utils.embedding import EmbeddingModel
import numpy as np

class EmbedChunksStage:
    """Generate embeddings for chunks."""

    def __init__(self, embedding_model: EmbeddingModel, config: EmbeddingConfig):
        self.embedding_model = embedding_model
        self.config = config

    async def execute(
        self,
        input_data: List[Chunk],
        context: Dict[str, Any]
    ) -> StageResult:
        """Embed all chunks."""
        # Extract texts
        texts = [chunk.text for chunk in input_data]

        # Batch embed
        embeddings = await self.embedding_model.embed_batch(
            texts,
            batch_size=self.config.batch_size,
            normalize=self.config.normalize
        )

        # Attach embeddings to chunks
        for chunk, embedding in zip(input_data, embeddings):
            chunk.embedding = embedding.tolist()
            chunk.embedding_model = self.config.model

        return StageResult(
            status="success",
            data=input_data,  # Chunks with embeddings
            metadata={
                "chunk_count": len(input_data),
                "embedding_dim": len(embeddings[0]),
                "api_calls": len(texts) // self.config.batch_size + 1
            }
        )
```

### Output

```python
List[Chunk]  # Chunks with .embedding populated
```

### Error Handling

- **API errors:** Retry with exponential backoff
- **Empty text:** Use zero vector
- **Dimension mismatch:** Raise error

### Performance

- **Batching:** 128 texts per API call
- **Parallelization:** Up to 10 concurrent API calls
- **Typical duration:** 2-5 minutes for 10K chunks
- **Cost:** ~$2.50 for 10K chunks (Voyage AI pricing)

---

## Stage 5: Build Strategy Indexes

### Purpose

Build retrieval indexes for each strategy.

### Strategies Built

1. **Naive:** Vector index only
2. **Hybrid:** Vector + BM25 index
3. **GraphRAG:** Graph + communities + summaries
4. **LightRAG:** Dual entity/relationship indexes
5. **HippoRAG:** Associative graph

### Sub-Stage: Naive Index

```python
from graphunified.storage.vector_store import VectorStore

class NaiveIndexBuilder:
    """Build naive vector index."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    async def build(self, chunks: List[Chunk]):
        """Build vector index from chunks."""
        ids = [str(chunk.id) for chunk in chunks]
        embeddings = np.array([chunk.embedding for chunk in chunks])
        metadata = [{"document_id": str(chunk.document_id)} for chunk in chunks]

        await self.vector_store.add(ids, embeddings, metadata)
```

### Sub-Stage: GraphRAG Index

```python
from graphunified.storage.graph_store import GraphStore
from graphunified.utils.graph_utils import detect_communities, summarize_community

class GraphRAGIndexBuilder:
    """Build GraphRAG indexes (communities + summaries)."""

    def __init__(
        self,
        graph_store: GraphStore,
        llm_client: LLMClient,
        config: GraphRAGConfig
    ):
        self.graph_store = graph_store
        self.llm = llm_client
        self.config = config

    async def build(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ):
        """Build graph, detect communities, generate summaries."""
        # Build graph
        await self.graph_store.add_entities(entities)
        await self.graph_store.add_relationships(relationships)

        # Detect communities using Leiden algorithm
        graph = await self.graph_store.get_graph()
        communities = detect_communities(
            graph,
            resolution=self.config.leiden_resolution
        )

        # Generate community summaries
        community_reports = []
        for community in communities:
            report = await self._generate_community_report(community, entities)
            community_reports.append(report)

        # Save communities and reports
        await self._save_communities(communities, community_reports)

    async def _generate_community_report(
        self,
        community: Community,
        entities: List[Entity]
    ) -> CommunityReport:
        """Generate LLM summary for community."""
        # Get entity descriptions
        community_entities = [
            e for e in entities if e.id in community.entity_ids
        ]
        entity_text = "\n\n".join([
            f"**{e.name}** ({e.type}): {e.description}"
            for e in community_entities
        ])

        # Prompt LLM
        prompt = COMMUNITY_REPORT_PROMPT.format(entity_text=entity_text)
        response = await self.llm.generate(prompt, temperature=0.1)

        # Parse response
        return CommunityReport(
            community_id=community.id,
            title=self._extract_title(response),
            summary=self._extract_summary(response),
            full_content=response,
            findings=self._extract_findings(response)
        )
```

**Community Report Prompt:**

```python
COMMUNITY_REPORT_PROMPT = """
Generate a comprehensive report about this group of related entities.

ENTITIES:
{entity_text}

Provide:
1. A concise title (max 10 words)
2. A 2-3 sentence summary
3. Key findings (3-7 bullet points)
4. Detailed analysis (2-3 paragraphs)

Format your response as:
# TITLE: [title]

## Summary
[summary]

## Key Findings
- [finding 1]
- [finding 2]
...

## Analysis
[detailed analysis]
"""
```

---

## Error Handling Strategy

### Failure Modes

1. **Stage failure:** Log error, mark stage as failed, continue pipeline if possible
2. **Data validation failure:** Skip invalid records, log warnings
3. **API errors:** Retry with exponential backoff (max 3 attempts)
4. **Out of memory:** Reduce batch sizes, process in smaller chunks
5. **Rate limiting:** Respect limits, queue requests

### Recovery

```python
class Pipeline:
    async def execute_with_recovery(self, input_data):
        """Execute pipeline with checkpoint/recovery."""
        checkpoint_file = Path("output/.checkpoint.json")

        # Load checkpoint if exists
        completed_stages = self._load_checkpoint(checkpoint_file)

        for stage in self.stages:
            if stage.get_name() in completed_stages:
                logger.info(f"Skipping completed stage: {stage.get_name()}")
                continue

            try:
                result = await stage.execute(input_data, self.context)
                completed_stages.add(stage.get_name())
                self._save_checkpoint(checkpoint_file, completed_stages)
            except Exception as e:
                logger.error(f"Stage {stage.get_name()} failed: {e}")
                if stage.is_critical():
                    raise
                # Continue with non-critical stages
```

---

## Progress Tracking

### Callback Interface

```python
from typing import Callable

ProgressCallback = Callable[[str, float], None]

# Example usage
def progress_callback(stage_name: str, progress_pct: float):
    print(f"{stage_name}: {progress_pct:.1f}%")

await indexer.index(
    input_path=Path("./docs"),
    progress_callback=progress_callback
)
```

### Progress Events

```python
# Stage start
callback("chunk", 0.0)

# Stage progress
callback("chunk", 25.0)
callback("chunk", 50.0)
callback("chunk", 75.0)

# Stage complete
callback("chunk", 100.0)
```

---

## Performance Benchmarks

### Expected Performance (10K Documents)

| Stage | Duration | Parallelization | Bottleneck |
|-------|----------|-----------------|------------|
| Load | 2 sec | 10 workers | Disk I/O |
| Chunk | 30 sec | 10 workers | CPU (tokenization) |
| Extract | 20 min | 10 concurrent LLM | API rate limits |
| Embed Chunks | 3 min | 10 concurrent API | API rate limits |
| Embed Entities | 30 sec | 10 concurrent API | API rate limits |
| Build Indexes | 2 min | Sequential | CPU (graph algorithms) |
| **Total** | **~26 min** | | |

### Optimization Strategies

1. **Batch sizing:** Tune batch sizes per stage
2. **Concurrency limits:** Respect API rate limits
3. **Caching:** Cache embeddings, skip re-extraction
4. **Incremental updates:** Only reprocess changed documents
5. **Distributed processing:** Run stages on multiple machines

---

## Summary

This specification defines:

- **6 pipeline stages** with input/output contracts
- **Detailed algorithms** for chunking, extraction, embedding
- **Prompt templates** for LLM-based extraction
- **Error handling** and recovery strategies
- **Progress tracking** interface
- **Performance benchmarks** and optimization strategies

**Key Insights:**
- Extraction is the bottleneck (~20 min for 10K docs)
- Batching is critical for API efficiency
- Entity deduplication is essential for graph quality
- Parallel execution reduces total time by 60%

**Next Steps:**
- Implement stage classes in `index/stages/`
- Build pipeline orchestrator in `index/pipeline.py`
- Create strategy builders in `index/strategies/`
- Add comprehensive error handling and logging
