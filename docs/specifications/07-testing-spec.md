# Testing Specifications

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document specifies comprehensive testing requirements for Graph-Unified, including unit tests, integration tests, end-to-end scenarios, performance benchmarks, and evaluation metrics.

## Testing Strategy

### Test Pyramid

```
              ┌────────────┐
              │ E2E Tests  │  10% - Full system workflows
              │  (~20)     │
              ├────────────┤
              │Integration │  30% - Component interactions
              │ Tests (~60)│
              ├────────────┤
              │ Unit Tests │  60% - Individual functions
              │  (~200)    │
              └────────────┘
```

### Coverage Targets

- **Overall coverage:** >80%
- **Core modules:** >90% (storage, index, query)
- **Utilities:** >85%
- **CLI:** >70%

---

## Unit Tests

### Storage Layer

#### test_parquet_store.py

```python
import pytest
from pathlib import Path
from graphunified.storage.parquet_store import ParquetStore
from graphunified.config.models import Chunk, Entity, Relationship

@pytest.fixture
def parquet_store(tmp_path):
    """Create temporary ParquetStore."""
    return ParquetStore(tmp_path)

@pytest.fixture
def sample_chunks():
    """Generate sample chunks."""
    return [
        Chunk(
            document_id="doc-1",
            chunk_index=0,
            text="Test chunk 1",
            start_char=0,
            end_char=12,
            token_count=3
        ),
        Chunk(
            document_id="doc-1",
            chunk_index=1,
            text="Test chunk 2",
            start_char=13,
            end_char=25,
            token_count=3
        )
    ]

@pytest.mark.asyncio
async def test_save_and_load_chunks(parquet_store, sample_chunks):
    """Test saving and loading chunks."""
    # Save
    await parquet_store.save_chunks(sample_chunks)

    # Load
    loaded = list(await parquet_store.load_chunks())

    # Verify
    assert len(loaded) == len(sample_chunks)
    assert loaded[0].text == sample_chunks[0].text
    assert loaded[0].document_id == sample_chunks[0].document_id

@pytest.mark.asyncio
async def test_load_chunks_by_document(parquet_store, sample_chunks):
    """Test loading chunks for specific document."""
    await parquet_store.save_chunks(sample_chunks)

    loaded = await parquet_store.get_by_document("doc-1")

    assert len(loaded) == 2
    assert all(c.document_id == "doc-1" for c in loaded)

@pytest.mark.asyncio
async def test_empty_store(parquet_store):
    """Test operations on empty store."""
    count = await parquet_store.count()
    assert count == 0

    loaded = list(await parquet_store.load_chunks())
    assert loaded == []
```

**Required Tests:**

- [x] Save and load chunks
- [x] Save and load entities
- [x] Save and load relationships
- [x] Query by document ID
- [x] Count operations
- [x] Empty store behavior
- [x] Schema validation
- [x] Concurrent reads
- [x] Append operations

---

#### test_vector_store.py

```python
import pytest
import numpy as np
from graphunified.storage.vector_store import VectorStore

@pytest.fixture
def vector_store(tmp_path):
    """Create temporary VectorStore."""
    return VectorStore(
        db_path=str(tmp_path / "vectors"),
        table_name="test_chunks",
        dimension=128
    )

@pytest.fixture
def sample_vectors():
    """Generate sample vectors."""
    np.random.seed(42)
    return np.random.rand(10, 128).astype(np.float32)

@pytest.mark.asyncio
async def test_add_vectors(vector_store, sample_vectors):
    """Test adding vectors."""
    ids = [f"vec-{i}" for i in range(10)]

    await vector_store.add(ids, sample_vectors)

    count = await vector_store.count()
    assert count == 10

@pytest.mark.asyncio
async def test_search(vector_store, sample_vectors):
    """Test vector search."""
    ids = [f"vec-{i}" for i in range(10)]
    await vector_store.add(ids, sample_vectors)

    # Search with first vector
    results = await vector_store.search(sample_vectors[0], top_k=3)

    # Should return 3 results
    assert len(results) == 3

    # First result should be the query vector itself
    assert results[0][0] == "vec-0"
    assert results[0][1] >= 0.99  # Cosine similarity ≈ 1.0

@pytest.mark.asyncio
async def test_delete_vectors(vector_store, sample_vectors):
    """Test deleting vectors."""
    ids = [f"vec-{i}" for i in range(10)]
    await vector_store.add(ids, sample_vectors)

    await vector_store.delete(["vec-0", "vec-1"])

    count = await vector_store.count()
    assert count == 8
```

**Required Tests:**

- [x] Add vectors
- [x] Search by similarity
- [x] Delete vectors
- [x] Metadata filtering
- [x] Dimension validation
- [x] Empty index behavior
- [x] Large batch operations (1K+ vectors)

---

### Indexing Pipeline

#### test_chunk_stage.py

```python
import pytest
from graphunified.index.stages.chunk import ChunkStage
from graphunified.config.models import Document
from graphunified.config.settings import ChunkingConfig

@pytest.fixture
def chunk_stage():
    config = ChunkingConfig(
        chunk_size=512,
        chunk_overlap=64,
        encoding_name="cl100k_base"
    )
    return ChunkStage(config)

@pytest.fixture
def sample_document():
    return Document(
        filename="test.txt",
        text="This is a test document. " * 100,  # 500 words
        metadata={}
    )

@pytest.mark.asyncio
async def test_chunk_document(chunk_stage, sample_document):
    """Test chunking a document."""
    result = await chunk_stage.execute([sample_document], {})

    chunks = result.data
    assert len(chunks) > 0
    assert all(c.document_id == sample_document.id for c in chunks)
    assert all(c.token_count > 0 for c in chunks)

@pytest.mark.asyncio
async def test_chunk_overlap(chunk_stage, sample_document):
    """Test that chunks have proper overlap."""
    result = await chunk_stage.execute([sample_document], {})
    chunks = result.data

    if len(chunks) >= 2:
        # Check overlap: end of chunk[i] overlaps with start of chunk[i+1]
        chunk_0_end = chunks[0].text[-20:]
        chunk_1_start = chunks[1].text[:20]
        # Should have some overlap
        assert any(word in chunk_1_start for word in chunk_0_end.split())

@pytest.mark.asyncio
async def test_empty_document(chunk_stage):
    """Test handling empty document."""
    doc = Document(filename="empty.txt", text="", metadata={})

    with pytest.raises(ValueError):
        await chunk_stage.execute([doc], {})
```

**Required Tests:**

- [x] Chunk single document
- [x] Chunk multiple documents
- [x] Verify overlap
- [x] Verify token counts
- [x] Handle empty document
- [x] Handle very long document (>1M chars)
- [x] Respect sentence boundaries

---

#### test_extract_stage.py

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from graphunified.index.stages.extract import ExtractStage
from graphunified.config.models import Chunk

@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value="""{
        "entities": [
            {"name": "IPCC", "type": "ORGANIZATION", "description": "Climate science body", "confidence": 0.95},
            {"name": "Climate Change", "type": "CONCEPT", "description": "Global warming phenomenon", "confidence": 0.90}
        ],
        "relationships": [
            {"source": "IPCC", "target": "Climate Change", "type": "STUDIES", "description": "IPCC studies climate change", "confidence": 0.85}
        ]
    }""")
    return client

@pytest.fixture
def extract_stage(mock_llm_client):
    config = ExtractionConfig(
        entity_types=["ORGANIZATION", "CONCEPT"],
        relationship_types=["STUDIES"],
        min_confidence=0.7
    )
    return ExtractStage(mock_llm_client, config)

@pytest.mark.asyncio
async def test_extract_entities(extract_stage):
    """Test entity extraction."""
    chunks = [
        Chunk(
            document_id="doc-1",
            chunk_index=0,
            text="The IPCC studies climate change.",
            start_char=0,
            end_char=33,
            token_count=7
        )
    ]

    result = await extract_stage.execute(chunks, {})
    entities, relationships = result.data

    assert len(entities) == 2
    assert entities[0].name == "IPCC"
    assert entities[0].type == "ORGANIZATION"
    assert len(relationships) == 1

@pytest.mark.asyncio
async def test_confidence_filtering(extract_stage):
    """Test that low-confidence extractions are filtered."""
    # Mock response with low-confidence entity
    extract_stage.llm.generate = AsyncMock(return_value="""{
        "entities": [
            {"name": "Test", "type": "CONCEPT", "description": "Test", "confidence": 0.5}
        ],
        "relationships": []
    }""")

    chunks = [Chunk(document_id="doc-1", chunk_index=0, text="Test", start_char=0, end_char=4, token_count=1)]

    result = await extract_stage.execute(chunks, {})
    entities, _ = result.data

    # Low confidence entity should be filtered
    assert len(entities) == 0
```

**Required Tests:**

- [x] Extract entities
- [x] Extract relationships
- [x] Confidence filtering
- [x] Entity deduplication
- [x] Relationship resolution
- [x] Handle malformed JSON
- [x] Handle empty response
- [x] Batch processing

---

### Query Layer

#### test_naive_retriever.py

```python
import pytest
from graphunified.query.retrievers.naive import NaiveRetriever

@pytest.fixture
async def naive_retriever(vector_store, chunk_store, embedding_model):
    """Create NaiveRetriever with test data."""
    # Add test chunks
    chunks = [...]
    await chunk_store.save_chunks(chunks)

    # Add embeddings
    embeddings = await embedding_model.embed([c.text for c in chunks])
    await vector_store.add([str(c.id) for c in chunks], embeddings)

    return NaiveRetriever(vector_store, chunk_store, embedding_model)

@pytest.mark.asyncio
async def test_retrieve(naive_retriever):
    """Test naive retrieval."""
    results = await naive_retriever.retrieve("climate change", top_k=5)

    assert len(results) == 5
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)
    assert all(0.0 <= r["score"] <= 1.0 for r in results)

@pytest.mark.asyncio
async def test_empty_query(naive_retriever):
    """Test handling empty query."""
    with pytest.raises(ValueError):
        await naive_retriever.retrieve("", top_k=5)
```

**Required Tests:**

- [x] Retrieve top-k
- [x] Score normalization
- [x] Empty query handling
- [x] Query with no results
- [x] Verify result format

---

## Integration Tests

### test_indexing_pipeline.py

```python
import pytest
from pathlib import Path
from graphunified.index.indexer import Indexer
from graphunified.config.settings import Settings

@pytest.fixture
def sample_corpus(tmp_path):
    """Create sample document corpus."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "doc1.txt").write_text("The IPCC studies climate change.")
    (docs_dir / "doc2.txt").write_text("Climate change impacts ecosystems.")

    return docs_dir

@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_indexing_pipeline(sample_corpus, tmp_path):
    """Test complete indexing pipeline."""
    # Load settings
    settings = Settings.load("test_settings.yaml")
    settings.storage.root_dir = tmp_path / "output"

    # Create indexer
    indexer = Indexer(settings)

    # Run indexing
    stats = await indexer.index(sample_corpus)

    # Verify outputs
    assert stats["document_count"] == 2
    assert stats["chunk_count"] > 0
    assert stats["entity_count"] > 0

    # Verify files created
    assert (tmp_path / "output" / "documents.parquet").exists()
    assert (tmp_path / "output" / "chunks.parquet").exists()
    assert (tmp_path / "output" / "entities.parquet").exists()

@pytest.mark.asyncio
@pytest.mark.integration
async def test_incremental_indexing(sample_corpus, tmp_path):
    """Test incremental indexing."""
    settings = Settings.load("test_settings.yaml")
    settings.storage.root_dir = tmp_path / "output"

    indexer = Indexer(settings)

    # Initial indexing
    stats1 = await indexer.index(sample_corpus)

    # Add new document
    (sample_corpus / "doc3.txt").write_text("New document about renewable energy.")

    # Incremental update
    stats2 = await indexer.update()

    # Should only process new document
    assert stats2["document_count"] == 1
```

**Required Tests:**

- [x] Full pipeline execution
- [x] Incremental indexing
- [x] Error recovery
- [x] Progress tracking
- [x] Output validation

---

### test_query_pipeline.py

```python
@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_query_pipeline(indexed_corpus):
    """Test complete query pipeline."""
    settings = Settings.load("test_settings.yaml")
    querier = Querier(settings)

    result = await querier.query("What is climate change?", strategy="naive")

    assert result.response
    assert len(result.contexts) > 0
    assert result.strategy == "naive"
    assert result.latency_ms > 0

@pytest.mark.asyncio
@pytest.mark.integration
async def test_strategy_comparison(indexed_corpus):
    """Test comparing multiple strategies."""
    querier = Querier(settings)

    results = await querier.compare_strategies(
        "What is climate change?",
        strategies=["naive", "hybrid", "graphrag_local"]
    )

    assert len(results) == 3
    assert "naive" in results
    assert "hybrid" in results
    assert "graphrag_local" in results
```

**Required Tests:**

- [x] Query execution
- [x] Strategy comparison
- [x] Query routing
- [x] Response generation
- [x] Error handling

---

## End-to-End Tests

### test_e2e_workflows.py

```python
@pytest.mark.e2e
def test_first_time_user_workflow(tmp_path):
    """Test: First-time user indexes corpus and queries."""
    # 1. Initialize config
    run_cli(["init", "--config", str(tmp_path / "settings.yaml")])

    # 2. Index documents
    run_cli(["index", "--input", "test_corpus", "--config", str(tmp_path / "settings.yaml")])

    # 3. Query with different strategies
    result1 = run_cli(["query", "What is climate change?", "--model", "naive"])
    result2 = run_cli(["query", "What is climate change?", "--model", "hybrid"])

    assert "climate change" in result1.lower()
    assert "climate change" in result2.lower()

@pytest.mark.e2e
def test_prompt_tuning_workflow(tmp_path):
    """Test: User tunes prompts for domain."""
    # 1. Prepare evaluation data
    eval_data = create_evaluation_dataset()

    # 2. Run baseline evaluation
    baseline = run_cli(["evaluate", "--eval-data", str(eval_data)])

    # 3. Tune prompts
    run_cli(["prompt-tune", "--sample-docs", "test_corpus/sample", "--iterations", "2"])

    # 4. Re-evaluate
    tuned = run_cli(["evaluate", "--eval-data", str(eval_data)])

    # Should show improvement
    assert tuned["f1"] > baseline["f1"]
```

**E2E Scenarios:**

1. **First-time setup:** Init → Index → Query
2. **Prompt tuning:** Baseline → Tune → Evaluate
3. **Strategy comparison:** Index → Query all strategies → Compare
4. **Incremental update:** Index → Update → Verify
5. **Production deployment:** Index → Query → Monitor
6. **Error recovery:** Index with errors → Resume
7. **Migration:** Import from standalone tools → Query
8. **Visualization:** Index → Visualize graph → Export
9. **Batch querying:** Index → Query 100+ queries
10. **Large corpus:** Index 10K documents → Query

---

## Performance Tests

### test_performance.py

```python
import pytest
import time

@pytest.mark.performance
def test_indexing_throughput():
    """Test indexing performance."""
    corpus_size = 1000  # documents

    start = time.time()
    stats = await indexer.index(corpus)
    duration = time.time() - start

    docs_per_sec = corpus_size / duration

    # Target: >30 docs/sec
    assert docs_per_sec > 30

@pytest.mark.performance
def test_query_latency():
    """Test query latency."""
    queries = [...]  # 100 test queries

    latencies = []
    for query in queries:
        start = time.time()
        await querier.query(query, strategy="naive")
        latency = time.time() - start
        latencies.append(latency)

    avg_latency = sum(latencies) / len(latencies)

    # Target: <200ms average
    assert avg_latency < 0.2

@pytest.mark.performance
def test_memory_usage():
    """Test memory usage during indexing."""
    import tracemalloc

    tracemalloc.start()

    await indexer.index(large_corpus)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Target: <4GB peak memory for 10K docs
    assert peak < 4 * 1024 * 1024 * 1024
```

**Performance Targets:**

| Metric | Target | Measurement |
|--------|--------|-------------|
| Indexing throughput | >30 docs/sec | 1000 docs test |
| Query latency (Naive) | <100ms | Average of 100 queries |
| Query latency (Hybrid) | <200ms | Average of 100 queries |
| Query latency (GraphRAG Local) | <500ms | Average of 100 queries |
| Memory usage (indexing) | <4GB | 10K documents |
| Memory usage (querying) | <1GB | Concurrent queries |

---

## Evaluation Metrics

### Retrieval Metrics

```python
def compute_retrieval_metrics(
    retrieved: List[str],
    relevant: List[str],
    k: int = 10
) -> Dict[str, float]:
    """Compute retrieval evaluation metrics."""
    retrieved_set = set(retrieved[:k])
    relevant_set = set(relevant)

    # Precision@K
    precision = len(retrieved_set & relevant_set) / k if k > 0 else 0.0

    # Recall@K
    recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0.0

    # F1@K
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, item in enumerate(retrieved[:k]):
        if item in relevant_set:
            mrr = 1.0 / (i + 1)
            break

    return {
        "precision@k": precision,
        "recall@k": recall,
        "f1@k": f1,
        "mrr": mrr
    }
```

### Answer Quality Metrics (LLM-as-Judge)

```python
JUDGE_PROMPT = """
Evaluate the quality of the answer to the query.

QUERY: {query}
GROUND TRUTH: {ground_truth}
ANSWER: {answer}

Rate the answer on:
1. Relevance (0-5): How well does it address the query?
2. Accuracy (0-5): Is the information correct?
3. Completeness (0-5): Does it cover key points?
4. Clarity (0-5): Is it easy to understand?

Return JSON:
{{
  "relevance": <score>,
  "accuracy": <score>,
  "completeness": <score>,
  "clarity": <score>,
  "overall": <average>
}}
"""

async def evaluate_answer_quality(
    query: str,
    answer: str,
    ground_truth: str,
    llm_client: LLMClient
) -> Dict[str, float]:
    """Evaluate answer quality using LLM."""
    prompt = JUDGE_PROMPT.format(
        query=query,
        answer=answer,
        ground_truth=ground_truth
    )

    response = await llm_client.generate(prompt, temperature=0.0)
    scores = json.loads(response)

    return scores
```

### Cost Metrics

```python
def compute_cost_metrics(stats: Dict[str, Any]) -> Dict[str, float]:
    """Compute cost metrics."""
    return {
        "total_cost_usd": stats["cost_usd"],
        "cost_per_document": stats["cost_usd"] / stats["document_count"],
        "cost_per_chunk": stats["cost_usd"] / stats["chunk_count"],
        "cost_per_entity": stats["cost_usd"] / stats["entity_count"],
    }
```

---

## Test Data

### Sample Corpus

**test_corpus/climate/**

- `ipcc_report.txt` (5000 words)
- `climate_policy.txt` (3000 words)
- `renewable_energy.txt` (2000 words)

**test_corpus/medical/**

- `diabetes_overview.txt` (4000 words)
- `treatment_guidelines.txt` (3500 words)

### Ground Truth Data

**eval_data/retrieval_gold.json**

```json
{
  "queries": [
    {
      "id": "q1",
      "query": "What causes climate change?",
      "relevant_chunks": ["chunk-id-1", "chunk-id-5", "chunk-id-12"],
      "relevant_entities": ["entity-id-1", "entity-id-3"]
    },
    ...
  ]
}
```

**eval_data/extraction_gold.json**

```json
{
  "documents": [
    {
      "filename": "test1.txt",
      "text": "The IPCC studies climate change.",
      "entities": [
        {"name": "IPCC", "type": "ORGANIZATION"},
        {"name": "climate change", "type": "CONCEPT"}
      ],
      "relationships": [
        {"source": "IPCC", "target": "climate change", "type": "STUDIES"}
      ]
    }
  ]
}
```

---

## CI/CD Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-asyncio pytest-cov

      - name: Run unit tests
        run: pytest tests/unit --cov=graphunified --cov-report=xml

      - name: Run integration tests
        run: pytest tests/integration -m integration

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## Summary

This specification defines:

- **200+ unit tests** covering all modules
- **60+ integration tests** for component interactions
- **20 E2E scenarios** for user workflows
- **Performance benchmarks** with targets
- **Evaluation metrics** for retrieval and generation quality
- **Test data** including sample corpus and ground truth
- **CI/CD integration** for automated testing

**Testing Priorities:**

1. **Critical path:** Storage, extraction, retrieval (>90% coverage)
2. **Error handling:** All failure modes tested
3. **Performance:** Benchmarks for latency and throughput
4. **Quality:** Retrieval accuracy and answer quality metrics

**Next Steps:**

- Implement unit tests in `tests/unit/`
- Create integration tests in `tests/integration/`
- Build E2E test suite in `tests/e2e/`
- Set up CI/CD pipeline
- Generate test data and ground truth
