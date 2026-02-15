# Graph-Unified: Technical Architecture & Design

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Design Document

## Executive Summary

Graph-Unified is a unified multi-strategy Retrieval-Augmented Generation (RAG) system that supports six different retrieval strategies through a shared chunking and entity extraction pipeline. By consolidating the extraction phase across all strategies, the system achieves 60-70% reduction in LLM costs while enabling fair comparison between retrieval approaches.

**Key Innovation:** Single extraction pass with strategy-specific post-processing, rather than running separate extraction pipelines for each strategy.

**Supported Strategies:**
1. **Naive RAG** - Direct vector similarity search on chunks
2. **Hybrid RAG** - Combined dense + sparse (BM25) retrieval
3. **GraphRAG Local** - Entity-centric neighborhood search (Microsoft)
4. **GraphRAG Global** - Community-based map-reduce (Microsoft)
5. **LightRAG** - Dual-index entity-relationship retrieval
6. **HippoRAG** - Hippocampus-inspired associative retrieval

**Target Use Cases:**
- Research teams comparing retrieval strategies
- Production systems needing query-type-specific retrieval
- Organizations optimizing RAG performance and cost
- Domain-specific knowledge bases (legal, medical, financial)

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Interface                            │
│              (graph-unified index | query)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        ┌───────▼────────┐            ┌────────▼─────────┐
        │  Index Pipeline │            │  Query Pipeline   │
        │                │            │                  │
        │  1. Chunk      │            │  1. Query Router │
        │  2. Extract    │            │  2. Retrievers   │
        │  3. Build      │            │  3. Generator    │
        └───────┬────────┘            └────────┬─────────┘
                │                               │
        ┌───────▼─────────────────────────────▼────────┐
        │           Storage Layer                       │
        │  - Parquet (canonical data)                   │
        │  - LanceDB/FAISS (vector indexes)            │
        │  - NetworkX/igraph (graph structures)        │
        └──────────────────────────────────────────────┘
```

### Core Design Principles

1. **Single Source of Truth**: All strategies share the same chunked documents and extracted entities
2. **Lazy Index Building**: Strategy-specific indexes built only when needed
3. **Cost Optimization**: One extraction pass replaces 6 separate extraction pipelines
4. **Fair Comparison**: Identical inputs enable meaningful strategy comparison
5. **Incremental Evolution**: Start with batch rebuild, evolve to incremental updates

### Module Breakdown

```
graphunified/
├── cli.py                      # Entry point (Click CLI)
├── config/
│   ├── settings.py            # Configuration schema (Pydantic)
│   ├── models.py              # Data models
│   └── defaults.py            # Default configurations
├── index/
│   ├── pipeline.py            # Orchestrator (async DAG)
│   ├── stages/
│   │   ├── chunk.py          # Document chunking
│   │   ├── extract.py        # Entity/relationship extraction
│   │   ├── embed.py          # Embedding generation
│   │   └── build.py          # Strategy-specific index builders
│   └── strategies/
│       ├── graphrag.py       # MS GraphRAG (communities + summaries)
│       ├── lightrag.py       # LightRAG (entity + relation indexes)
│       ├── hipporag.py       # HippoRAG (associative graph)
│       ├── hybrid.py         # Hybrid (dense + sparse)
│       └── naive.py          # Naive (vector only)
├── query/
│   ├── router.py             # Query routing logic
│   ├── retrievers/
│   │   ├── base.py           # Abstract retriever interface
│   │   ├── naive.py          # Naive retriever
│   │   ├── hybrid.py         # Hybrid retriever
│   │   ├── graphrag_local.py # GraphRAG local search
│   │   ├── graphrag_global.py# GraphRAG global search
│   │   ├── lightrag.py       # LightRAG retriever
│   │   └── hipporag.py       # HippoRAG retriever
│   └── generator.py          # Response generation (Claude)
├── storage/
│   ├── parquet_store.py      # Parquet operations
│   ├── vector_store.py       # LanceDB/FAISS wrapper
│   └── graph_store.py        # NetworkX/igraph wrapper
├── prompt_tune/
│   ├── tuner.py              # Domain-specific prompt tuning
│   └── templates.py          # Base prompt templates
└── utils/
    ├── llm.py                # Claude API client
    ├── embedding.py          # Embedding model wrapper
    └── metrics.py            # Evaluation metrics
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)

**Goal:** Establish core infrastructure

**Tasks:**
- Set up project structure and dependencies
- Implement `settings.yaml` schema with Pydantic validation
- Create Claude API client wrapper (`utils/llm.py`)
- Implement embedding model integration (voyage-3 or similar)
- Design Parquet storage schema for canonical data
- Create base data models (Document, Chunk, Entity, Relationship)

**Deliverables:**
- Working `settings.yaml` configuration
- Claude API integration with rate limiting
- Parquet storage operations (read/write/append)
- Unit tests for core utilities

**Success Criteria:**
- Can load configuration from YAML
- Can call Claude API with retry logic
- Can persist data to Parquet format

---

### Phase 2: Shared Pipeline (Weeks 3-4)

**Goal:** Build the unified chunking and extraction pipeline

**Tasks:**
- Implement document chunking with overlap (tiktoken-based)
- Design extraction prompt templates (zero-shot entity/relationship)
- Build extraction stage with Claude API
- Implement batch processing with rate limiting
- Create embedding generation stage
- Build pipeline orchestrator with async execution

**Deliverables:**
- `index/stages/chunk.py` - Document chunking
- `index/stages/extract.py` - Entity/relationship extraction
- `index/stages/embed.py` - Embedding generation
- `index/pipeline.py` - Async DAG orchestrator
- Integration tests for full pipeline

**Success Criteria:**
- Can chunk documents with configurable size/overlap
- Can extract entities and relationships using Claude
- Can generate embeddings for chunks and entities
- Pipeline completes on sample dataset (100 documents)

**Storage Schema After Phase 2:**
```
output/
├── documents.parquet          # id, filename, text, metadata
├── chunks.parquet             # chunk_id, doc_id, text, embedding
├── entities.parquet           # entity_id, name, type, description, chunks[]
└── relationships.parquet      # source_id, target_id, type, description, chunks[]
```

---

### Phase 3: Naive & Hybrid Strategies (Week 5)

**Goal:** Implement baseline retrieval strategies

**Tasks:**
- Build LanceDB vector store wrapper
- Implement naive vector search retriever
- Implement BM25 sparse index (rank-bm25 library)
- Build hybrid retriever with score fusion
- Create query generator using Claude
- Implement CLI for query execution

**Deliverables:**
- `storage/vector_store.py` - LanceDB wrapper
- `query/retrievers/naive.py` - Vector-only retrieval
- `query/retrievers/hybrid.py` - Dense + sparse fusion
- `query/generator.py` - Response generation
- `cli.py` - Query command with `--model` flag

**Success Criteria:**
- Naive retrieval returns top-k chunks by similarity
- Hybrid retrieval combines vector + BM25 scores
- Query command generates response with Claude
- Performance: <2s for query on 10K chunks

---

### Phase 4: GraphRAG Local/Global (Weeks 6-7)

**Goal:** Implement Microsoft GraphRAG strategies

**Tasks:**
- Implement Leiden community detection (graspologic)
- Build community summarization with Claude
- Create entity description embedding index
- Create community report embedding index
- Implement local search (entity neighborhood traversal)
- Implement global search (map-reduce over communities)
- Add strategy-specific CLI flags

**Deliverables:**
- `index/strategies/graphrag.py` - Community building
- `query/retrievers/graphrag_local.py` - Local search
- `query/retrievers/graphrag_global.py` - Global search
- Additional storage: communities, community_reports

**Success Criteria:**
- Leiden algorithm produces hierarchical communities
- Community reports capture thematic summaries
- Local search retrieves entity neighborhoods
- Global search performs map-reduce correctly
- Maintains compatibility with MS GraphRAG outputs

**Storage Additions:**
```
output/
├── communities.parquet        # community_id, level, entities[], relationships[]
└── community_reports.parquet  # community_id, title, summary, embedding
```

---

### Phase 5: LightRAG (Week 8)

**Goal:** Implement LightRAG's dual-index approach

**Tasks:**
- Create entity embedding index
- Create relationship description embedding index
- Implement four search modes:
  - Local: Entity-centric search
  - Global: Relationship-centric search
  - Hybrid: Combined entity + relationship
  - Naive: Direct chunk retrieval
- Build graph traversal for context expansion
- Optimize relationship description generation

**Deliverables:**
- `index/strategies/lightrag.py` - Dual index builder
- `query/retrievers/lightrag.py` - Four-mode retriever
- Relationship description embeddings in Parquet

**Success Criteria:**
- Entity index enables entity-centric queries
- Relationship index captures thematic connections
- Search modes return different relevant contexts
- Performance: <3s for hybrid search on 50K entities

**Design Note:**
LightRAG's relationship descriptions are higher-level than raw relationship records. They summarize thematic connections between entities, enabling global reasoning without hierarchical communities.

---

### Phase 6: HippoRAG (Week 9)

**Goal:** Implement hippocampus-inspired associative retrieval

**Tasks:**
- Build bipartite associative graph (entities ↔ passages)
- Create fact embedding index
- Implement Personalized PageRank (PPR) activation
- Build two-stage retrieval:
  - Pattern separation: Extract query entities/facts
  - Pattern completion: Activate passages via PPR
- Integrate with igraph for graph operations
- Optimize PPR parameters (damping, iterations)

**Deliverables:**
- `index/strategies/hipporag.py` - Associative graph builder
- `query/retrievers/hipporag.py` - PPR-based retriever
- Fact embeddings and graph structure storage

**Success Criteria:**
- Associative graph links entities to passages correctly
- PPR activation retrieves relevant passages
- Two-stage retrieval mimics pattern separation/completion
- Performance: <5s for query on 20K entities + 10K passages

**Storage Additions:**
```
output/
├── facts.parquet              # fact_id, text, embedding, chunk_ids[]
└── hippo_graph.pickle         # igraph bipartite graph
```

---

### Phase 7: Prompt Tuning & Evaluation (Week 10)

**Goal:** Enable domain adaptation and quality measurement

**Tasks:**
- Implement prompt tuning workflow (auto-generate extraction prompts)
- Create evaluation framework:
  - Retrieval metrics: MRR, Recall@k, NDCG
  - Generation metrics: ROUGE, BERTScore, human eval rubric
  - Cost tracking: Token usage per strategy
- Build comparison report generator
- Create benchmark dataset (queries + ground truth)
- Document tuning process in how-to guides

**Deliverables:**
- `prompt_tune/tuner.py` - Domain-specific prompt generation
- `utils/metrics.py` - Evaluation metrics
- Benchmark dataset (100 queries with labeled answers)
- Comparison report template

**Success Criteria:**
- Prompt tuning improves extraction quality (measured F1)
- Evaluation runs across all 6 strategies
- Report shows retrieval quality and cost per strategy
- Clear winner emerges for different query types

**Evaluation Queries to Test:**
- Factual lookup: "What is X?"
- Multi-hop reasoning: "How does X relate to Y?"
- Summarization: "What are the main themes of X?"
- Entity-centric: "Tell me about [entity]"
- Comparative: "Compare X and Y"

---

### Phase 8: Production Hardening (Weeks 11-12)

**Goal:** Prepare for production deployment

**Tasks:**
- Implement incremental indexing (detect changes, update only deltas)
- Add comprehensive logging and monitoring
- Optimize performance:
  - Batch embedding generation
  - Parallel index building
  - Query result caching
- Create Docker deployment configuration
- Write deployment guide (cloud + on-premise)
- Finalize API documentation
- Create migration guide from standalone tools

**Deliverables:**
- Incremental indexing support
- Structured logging (JSON format)
- Performance optimization (2x faster)
- Docker Compose setup
- Deployment documentation
- API reference docs

**Success Criteria:**
- Incremental indexing updates in <10% of full rebuild time
- Structured logs enable debugging and monitoring
- Docker deployment works out-of-box
- Documentation covers all deployment scenarios

---

## Technical Specifications

### Data Models

**Document:**
```python
@dataclass
class Document:
    id: str
    filename: str
    text: str
    metadata: dict[str, Any]
    created_at: datetime
```

**Chunk:**
```python
@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    embedding: list[float]
    token_count: int
```

**Entity:**
```python
@dataclass
class Entity:
    id: str
    name: str
    type: str  # PERSON, ORG, LOCATION, CONCEPT, EVENT
    description: str
    source_chunks: list[str]  # chunk IDs
    embedding: list[float]  # of description
```

**Relationship:**
```python
@dataclass
class Relationship:
    id: str
    source_entity_id: str
    target_entity_id: str
    type: str  # RELATED_TO, PART_OF, CAUSED_BY, etc.
    description: str
    source_chunks: list[str]
    weight: float  # co-occurrence strength
```

**Community (GraphRAG):**
```python
@dataclass
class Community:
    id: str
    level: int  # Hierarchical level (0 = base)
    entity_ids: list[str]
    relationship_ids: list[str]
    parent_community_id: str | None
```

**CommunityReport (GraphRAG):**
```python
@dataclass
class CommunityReport:
    id: str
    community_id: str
    title: str
    summary: str  # Generated by Claude
    findings: list[str]
    embedding: list[float]
```

---

### Configuration Schema (settings.yaml)

```yaml
# Graph-Unified Configuration
version: "1.0"

# LLM Configuration
llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"  # Sonnet for extraction
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.0  # Deterministic extraction
  max_tokens: 4096
  rate_limit:
    requests_per_minute: 50
    tokens_per_minute: 40000

# Embedding Configuration
embedding:
  provider: "voyageai"  # or "openai", "cohere"
  model: "voyage-3"
  dimension: 1024
  api_key: "${VOYAGE_API_KEY}"
  batch_size: 128

# Chunking Configuration
chunking:
  strategy: "token_overlap"  # or "sentence", "semantic"
  chunk_size: 512  # tokens
  overlap: 128  # tokens
  min_chunk_size: 100

# Extraction Configuration
extraction:
  entity_types:
    - PERSON
    - ORGANIZATION
    - LOCATION
    - CONCEPT
    - EVENT
    - PRODUCT
  relationship_types:
    - RELATED_TO
    - PART_OF
    - LOCATED_IN
    - WORKS_FOR
    - CAUSED_BY
    - SIMILAR_TO
  max_gleanings: 1  # Additional extraction passes
  tuple_delimiter: "<|>"
  record_delimiter: "##"

# Strategy-Specific Configurations
strategies:
  graphrag:
    community_detection:
      algorithm: "leiden"
      max_level: 3
      resolution: 1.0
    summarization:
      max_tokens: 2000
      temperature: 0.1
    local_search:
      max_hops: 2
      top_k_entities: 10
      top_k_relationships: 20
    global_search:
      map_max_tokens: 1000
      reduce_max_tokens: 2000
      top_k_communities: 10

  lightrag:
    relationship_description:
      max_length: 200  # tokens
      temperature: 0.2
    search_modes:
      local_top_k: 20
      global_top_k: 20
      hybrid_weights: [0.5, 0.5]  # [entity, relation]

  hipporag:
    fact_extraction:
      max_facts_per_chunk: 5
    ppr_config:
      damping: 0.85
      max_iterations: 100
      tolerance: 1e-6
    retrieval:
      top_k_entities: 10
      top_k_passages: 10

  hybrid:
    dense_weight: 0.7
    sparse_weight: 0.3
    bm25_config:
      k1: 1.5
      b: 0.75

# Storage Configuration
storage:
  root_dir: "./output"
  parquet:
    compression: "snappy"
    row_group_size: 10000
  vector_store:
    backend: "lancedb"  # or "faiss"
    path: "./output/lancedb"
    metric: "cosine"
  graph_store:
    backend: "networkx"  # or "igraph"
    path: "./output/graphs"

# Query Configuration
query:
  default_strategy: "hybrid"
  top_k: 10
  generation:
    model: "claude-3-5-sonnet-20241022"
    max_tokens: 2048
    temperature: 0.3
  context_window: 8000  # tokens for generation

# Performance Configuration
performance:
  indexing:
    max_workers: 4  # Parallel workers
    batch_size: 100  # Documents per batch
    checkpoint_interval: 1000  # Save every N docs
  query:
    cache_enabled: true
    cache_ttl: 3600  # seconds

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "json"
  output: "stdout"  # or file path
```

---

### Storage Format Specifications

**Documents Parquet Schema:**
```
id: string
filename: string
text: string
metadata: struct<key: string, value: string>
created_at: timestamp
indexed_at: timestamp
```

**Chunks Parquet Schema:**
```
id: string
document_id: string
text: string
start_char: int32
end_char: int32
token_count: int32
embedding: list<float32>
indexed_at: timestamp
```

**Entities Parquet Schema:**
```
id: string
name: string
type: string (enum)
description: string
source_chunks: list<string>
embedding: list<float32>
mention_count: int32
indexed_at: timestamp
```

**Relationships Parquet Schema:**
```
id: string
source_entity_id: string
target_entity_id: string
type: string (enum)
description: string
source_chunks: list<string>
weight: float
indexed_at: timestamp
```

**Communities Parquet Schema (GraphRAG):**
```
id: string
level: int32
entity_ids: list<string>
relationship_ids: list<string>
parent_community_id: string (nullable)
size: int32
indexed_at: timestamp
```

**Community Reports Parquet Schema (GraphRAG):**
```
id: string
community_id: string
title: string
summary: string
findings: list<string>
embedding: list<float32>
token_count: int32
indexed_at: timestamp
```

**Facts Parquet Schema (HippoRAG):**
```
id: string
text: string
chunk_ids: list<string>
embedding: list<float32>
indexed_at: timestamp
```

---

### API Interfaces

**Abstract Retriever Interface:**
```python
from abc import ABC, abstractmethod
from typing import Protocol

class Retriever(Protocol):
    """Base interface for all retrieval strategies."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> list[RetrievalResult]:
        """Retrieve relevant contexts for query.

        Args:
            query: User query string
            top_k: Number of results to return
            **kwargs: Strategy-specific parameters

        Returns:
            List of RetrievalResult sorted by relevance
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy identifier."""
        pass

@dataclass
class RetrievalResult:
    """Unified retrieval result format."""
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]  # Strategy-specific info
    entities: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)
```

**Pipeline Stage Interface:**
```python
class PipelineStage(Protocol):
    """Base interface for pipeline stages."""

    @abstractmethod
    async def process(
        self,
        input_data: Any,
        config: Config
    ) -> Any:
        """Process input data and return output."""
        pass

    @abstractmethod
    def get_stage_name(self) -> str:
        """Return stage identifier."""
        pass
```

---

## Query Router Logic

The query router selects the optimal retrieval strategy based on query characteristics:

**Decision Tree:**

```python
def route_query(query: str, user_strategy: str | None = None) -> str:
    """Route query to optimal strategy.

    Priority:
    1. User-specified strategy (--model flag)
    2. Query-type detection (NLP analysis)
    3. Default strategy (hybrid)
    """

    if user_strategy:
        return user_strategy

    # Detect query type
    query_type = analyze_query(query)

    match query_type:
        case "entity_centric":
            # "Tell me about X", "Who is Y?"
            return "graphrag_local"

        case "multi_hop":
            # "How does X relate to Y?", "What connects A and B?"
            return "lightrag"

        case "summarization":
            # "What are main themes?", "Summarize X"
            return "graphrag_global"

        case "associative":
            # "What else is related to X?", exploratory queries
            return "hipporag"

        case "factual":
            # "What is the value of X?", direct lookups
            return "hybrid"

        case _:
            return "hybrid"  # Safe default
```

**Query Analysis:**
- Entity mention detection (spaCy NER)
- Question type classification (what/who/why/how)
- Multi-hop indicator detection (connect, relate, path)
- Summarization intent detection (summarize, overview, main themes)

**User Override:**
Always respect explicit `--model` flag for:
- Strategy comparison experiments
- Known optimal strategy for domain
- Testing and debugging

---

## Retrieval Strategy Comparison

| Dimension | Naive | Hybrid | GraphRAG Local | GraphRAG Global | LightRAG | HippoRAG |
|-----------|-------|--------|---------------|----------------|----------|----------|
| **Query Type** | Simple factual | Keyword + semantic | Entity-centric | Summarization | Multi-hop reasoning | Associative exploration |
| **Indexing Cost** | Low | Low | High | Very High | Medium | High |
| **Query Latency** | <1s | <2s | <3s | <5s | <3s | <5s |
| **Accuracy (Factual)** | Medium | High | High | Low | Medium | Medium |
| **Accuracy (Multi-hop)** | Low | Low | Medium | High | Very High | High |
| **Graph Dependency** | None | None | Full | Full | Medium | Medium |
| **Memory Usage** | Low | Low | High | Very High | Medium | High |
| **Incremental Update** | Easy | Easy | Hard | Very Hard | Medium | Hard |
| **Best For** | Quick lookups | General purpose | Person/org queries | Domain overviews | Reasoning chains | Exploration |

**When to Use Each Strategy:**

- **Naive:** Prototyping, small datasets (<10K docs), simple factual queries
- **Hybrid:** Production default, balanced accuracy/speed, keyword + semantic needs
- **GraphRAG Local:** Entity-centric queries, "tell me about X", org charts, social networks
- **GraphRAG Global:** Domain summarization, thematic analysis, high-level overviews
- **LightRAG:** Multi-hop reasoning, "how does X affect Y", causal chains, research queries
- **HippoRAG:** Exploratory search, serendipitous discovery, associative brainstorming

---

## Testing Strategy

### Unit Tests

**Coverage Requirements:** >80% for all modules

**Test Structure:**
```
tests/
├── unit/
│   ├── test_config.py         # Configuration loading
│   ├── test_chunking.py        # Chunking logic
│   ├── test_extraction.py      # Entity/relation extraction
│   ├── test_embedding.py       # Embedding generation
│   ├── test_storage.py         # Parquet operations
│   └── test_retrievers.py      # Individual retriever logic
├── integration/
│   ├── test_pipeline.py        # End-to-end indexing
│   ├── test_query_flow.py      # End-to-end querying
│   └── test_strategies.py      # Strategy-specific flows
└── fixtures/
    ├── sample_docs.txt         # Test documents
    ├── expected_entities.json  # Ground truth
    └── test_queries.json       # Query test cases
```

**Key Test Cases:**

1. **Chunking Tests:**
   - Respects token limits
   - Maintains overlap correctly
   - Handles edge cases (short docs, exact boundaries)

2. **Extraction Tests:**
   - Extracts known entities from fixture
   - Formats output correctly (tuple/record delimiters)
   - Handles Claude API errors gracefully

3. **Retrieval Tests:**
   - Returns top-k results
   - Scores are in valid range [0, 1]
   - Results are sorted by relevance
   - Strategy-specific logic correct

### Integration Tests

**End-to-End Scenarios:**

1. **Full Indexing Flow:**
   - Load documents → chunk → extract → embed → build all indexes
   - Verify all Parquet files created
   - Verify vector stores populated
   - Verify graph structures correct

2. **Query Flow Per Strategy:**
   - Execute query on each strategy
   - Verify results returned
   - Verify generation includes retrieved context
   - Verify cost tracking accurate

3. **Incremental Update:**
   - Index baseline dataset
   - Add new documents
   - Verify only new docs processed
   - Verify indexes updated correctly

### Performance Tests

**Benchmarks:**

| Test | Dataset Size | Target Time | Target Cost |
|------|-------------|-------------|-------------|
| Index 1K docs | 1,000 docs (avg 500 tokens) | <10 min | <$5 |
| Index 10K docs | 10,000 docs | <90 min | <$50 |
| Query (naive) | 10K chunks | <1s | <$0.01 |
| Query (GraphRAG global) | 10K chunks, 500 communities | <5s | <$0.10 |
| Incremental update | +100 docs to 10K | <5 min | <$1 |

**Monitoring:**
- Token usage per stage
- API call counts
- Memory usage during indexing
- Query latency percentiles (p50, p95, p99)

### Evaluation Framework

**Retrieval Quality Metrics:**

1. **Mean Reciprocal Rank (MRR):**
   - Measures rank of first relevant result
   - Requires ground truth labeled queries

2. **Recall@K:**
   - Percentage of relevant docs in top-k
   - K ∈ {5, 10, 20}

3. **NDCG@K:**
   - Normalized discounted cumulative gain
   - Accounts for graded relevance

**Generation Quality Metrics:**

1. **ROUGE-L:**
   - Longest common subsequence with reference
   - Measures lexical overlap

2. **BERTScore:**
   - Semantic similarity using BERT embeddings
   - Captures meaning beyond lexical match

3. **Human Evaluation Rubric:**
   - Correctness (1-5): Answer accuracy
   - Completeness (1-5): Covers all aspects
   - Relevance (1-5): Addresses query intent
   - Fluency (1-5): Readability

**Cost Tracking:**
- Indexing cost per document
- Query cost per strategy
- Cost/accuracy tradeoff analysis

---

## Performance Considerations

### Indexing Optimization

**Bottlenecks:**
1. Claude API calls for extraction (slowest)
2. Embedding generation (second slowest)
3. Graph computation (GraphRAG/HippoRAG)

**Optimization Strategies:**

1. **Batching:**
   - Process 100 docs at a time
   - Batch embed 128 chunks per request
   - Parallel extraction with rate limiting

2. **Caching:**
   - Cache entity extractions (deduplication)
   - Reuse embeddings for identical text
   - Checkpoint intermediate results

3. **Parallelization:**
   - Async I/O for API calls
   - Parallel index building per strategy
   - Use asyncio.gather for concurrent operations

**Expected Performance:**
- 1,000 docs: ~10 minutes (single machine)
- 10,000 docs: ~90 minutes (single machine)
- 100,000 docs: ~15 hours (single machine) or ~2 hours (10 workers)

### Query Optimization

**Strategies:**

1. **Vector Search:**
   - Use approximate nearest neighbors (ANN)
   - LanceDB/FAISS provide ANN out-of-box
   - Trade-off: 98% recall for 10x speedup

2. **Graph Traversal:**
   - Limit hop depth (max 2-3 hops)
   - Prune low-weight edges
   - Use adjacency lists for fast neighbor lookup

3. **Result Caching:**
   - Cache query results (1 hour TTL)
   - Cache embeddings for common queries
   - Invalidate on index updates

**Expected Latency:**
- Naive: <1s (10K chunks)
- Hybrid: <2s (10K chunks)
- Graph-based: <5s (50K entities, 100K relationships)

### Memory Management

**Memory Usage Estimates:**

| Component | Size (10K docs) | Size (100K docs) |
|-----------|-----------------|------------------|
| Chunks (text) | ~50 MB | ~500 MB |
| Chunk embeddings | ~200 MB (1024-dim) | ~2 GB |
| Entity embeddings | ~50 MB (50K entities) | ~500 MB |
| Graph (NetworkX) | ~100 MB | ~1 GB |
| LanceDB index | ~300 MB | ~3 GB |
| **Total** | **~700 MB** | **~7 GB** |

**Memory Optimization:**
- Stream large Parquet files (don't load all in RAM)
- Use memory-mapped vector indexes
- Lazy-load graph structures
- Clear intermediate results after each stage

---

## Future Enhancements

### Near-Term (3-6 months)

1. **Incremental Indexing:**
   - Detect document changes (hash-based)
   - Update only affected entities/relationships
   - Recompute only affected communities

2. **Multi-Modal Support:**
   - Extract entities from images (via Claude vision)
   - Index PDFs with layout awareness
   - Table extraction and reasoning

3. **Advanced Query Routing:**
   - ML-based query type classification
   - User feedback for routing improvement
   - A/B testing framework for strategies

4. **Real-Time Indexing:**
   - Stream processing for incoming documents
   - Online graph updates
   - Hot-reload indexes without downtime

### Long-Term (6-12 months)

1. **Federated Search:**
   - Query multiple knowledge bases
   - Merge results from distributed indexes
   - Cross-domain entity linking

2. **Hybrid Index Structures:**
   - Combine GraphRAG communities with LightRAG relations
   - Best-of-both-worlds approach
   - Adaptive strategy selection

3. **Fine-Tuned Models:**
   - Fine-tune embedding model on domain data
   - Fine-tune LLM for extraction (distillation)
   - Learn optimal strategy routing

4. **Explainability:**
   - Visualize entity graphs
   - Show retrieval paths (why this chunk?)
   - Generate retrieval explanations

5. **Cost-Performance Tuning:**
   - Auto-tune extraction prompts for cost/quality
   - Adaptive context window sizing
   - Model selection per query complexity

---

## Dependencies

### Core Libraries

**LLM & Embeddings:**
- `anthropic>=0.18.0` - Claude API client
- `voyageai>=0.2.0` - Voyage embeddings (or `openai`, `cohere`)
- `tiktoken>=0.5.0` - Token counting

**Data Processing:**
- `pyarrow>=14.0.0` - Parquet read/write
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical operations

**Vector Search:**
- `lancedb>=0.5.0` - Vector database
- `faiss-cpu>=1.7.4` - Alternative vector search

**Graph Processing:**
- `networkx>=3.2` - Graph data structure
- `igraph>=0.11.0` - Fast graph algorithms (HippoRAG)
- `graspologic>=3.3.0` - Leiden algorithm (GraphRAG)

**Search & NLP:**
- `rank-bm25>=0.2.2` - BM25 sparse retrieval
- `spacy>=3.7.0` - NER for query analysis
- `sentence-transformers>=2.3.0` - Embedding utilities

**Infrastructure:**
- `click>=8.1.0` - CLI framework
- `pydantic>=2.5.0` - Configuration validation
- `pyyaml>=6.0` - YAML parsing
- `aiohttp>=3.9.0` - Async HTTP client
- `asyncio-throttle>=1.0.0` - Rate limiting

**Development:**
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `black>=23.0.0` - Code formatting
- `ruff>=0.1.0` - Linting
- `mypy>=1.7.0` - Type checking

### Dependency Management

**Installation:**
```bash
pip install -e ".[dev]"  # Install with dev dependencies
```

**pyproject.toml:**
```toml
[project]
name = "graph-unified"
version = "0.1.0"
dependencies = [
    "anthropic>=0.18.0",
    "voyageai>=0.2.0",
    # ... (all core libraries)
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    # ... (all dev libraries)
]
```

---

## Risk Mitigation

### Technical Risks

**Risk:** Claude API rate limits cause indexing failures
**Mitigation:**
- Implement exponential backoff retry logic
- Checkpoint progress every 1000 docs
- Resume from checkpoint on failure
- Provide batch mode for low-priority indexing

**Risk:** Memory overflow on large datasets
**Mitigation:**
- Stream Parquet files instead of loading all
- Process documents in batches
- Clear intermediate results
- Provide distributed mode for >100K docs

**Risk:** Graph algorithms don't scale to 100K+ entities
**Mitigation:**
- Use igraph instead of NetworkX (10x faster)
- Limit community depth to 3 levels
- Prune low-weight edges below threshold
- Provide sampling mode for exploration

**Risk:** Incompatible with MS GraphRAG outputs
**Mitigation:**
- Follow exact Parquet schema from graphrag
- Validate against reference outputs
- Provide migration tool from graphrag
- Document schema differences

### Operational Risks

**Risk:** High API costs surprise users
**Mitigation:**
- Display cost estimates before indexing
- Track and report costs per stage
- Provide cost optimization tips in docs
- Offer local model option (Ollama)

**Risk:** Complex configuration overwhelms users
**Mitigation:**
- Provide sensible defaults for all settings
- Offer presets (research, production, low-cost)
- Validate configuration on load
- Generate starter config with comments

**Risk:** Strategy comparison is apples-to-oranges
**Mitigation:**
- Use identical extraction and embeddings
- Document strategy-specific strengths
- Provide query type recommendations
- Show cost/quality tradeoffs clearly

---

## Success Metrics

**Technical Metrics:**
- Indexing speed: 100 docs/min on single machine
- Query latency: <3s average across all strategies
- Memory efficiency: <1 GB for 10K docs
- Accuracy: >0.8 MRR on benchmark queries

**User Metrics:**
- Time to first query: <30 minutes from install
- Configuration clarity: <5 minutes to customize
- Strategy selection: Users choose optimal strategy >70% of time
- Cost predictability: Actual cost within 20% of estimate

**Adoption Metrics:**
- GitHub stars: >500 in 3 months
- Production deployments: >10 organizations
- Community contributions: >20 PRs
- Documentation completeness: 100% API coverage

---

## Conclusion

Graph-Unified provides a unified framework for comparing and deploying multiple RAG strategies through a shared extraction pipeline. By consolidating entity extraction across all strategies, the system achieves significant cost savings (60-70% reduction) while enabling fair comparison between retrieval approaches.

The phased implementation roadmap progresses from naive baseline to sophisticated graph-based retrieval, allowing teams to adopt incrementally. The system supports both research (strategy comparison) and production (optimal strategy per query type) use cases.

**Key Differentiators:**
1. Single extraction pass with strategy-specific post-processing
2. Fair comparison enabled by identical inputs
3. Query-type-aware strategy selection
4. Cost-optimized Claude API usage
5. Compatible with MS GraphRAG outputs

**Next Steps:**
1. Review architecture with stakeholders
2. Set up development environment
3. Begin Phase 1 implementation
4. Create benchmark dataset for evaluation
5. Establish CI/CD pipeline for testing

---

**Document Metadata:**
- **Author:** Claude Code (Diataxis Documenter Agent)
- **Version:** 1.0
- **Status:** Ready for Implementation
- **Last Updated:** 2026-02-15
- **Review Cycle:** Monthly during active development

