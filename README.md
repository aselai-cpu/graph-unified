# Graph-Unified RAG

A unified implementation of multiple Retrieval-Augmented Generation (RAG) strategies with a shared extraction pipeline.

## Overview

Graph-Unified RAG provides a single codebase supporting six different RAG strategies:

- **Naive RAG**: Dense vector retrieval
- **Hybrid RAG**: Dense + sparse (BM25) retrieval
- **GraphRAG Local**: Entity-based local search
- **GraphRAG Global**: Community-based global search
- **LightRAG**: Dual-level retrieval with knowledge graphs
- **HippoRAG**: Neurologically-inspired retrieval with PPR

All strategies share a common extraction pipeline for efficiency and consistency.

## Project Status

### Phase 1: Foundation ✅ COMPLETE

**Status:** All deliverables implemented and tested (2026-02-15)

**Implemented:**
- ✅ Configuration system with environment variable substitution
- ✅ Pydantic data models (Document, Chunk, Entity, Relationship, Community)
- ✅ Claude API client with dual rate limiting
- ✅ Voyage AI embedding client with batching
- ✅ Parquet storage backend with async operations
- ✅ Tokenization utilities (tiktoken)
- ✅ Logging configuration
- ✅ Unit tests (39 tests, 83% coverage)

See [PHASE1_COMPLETE.md](PHASE1_COMPLETE.md) for detailed implementation notes.

### Phase 2: Shared Pipeline (Upcoming)

- Document loader (PDF, TXT, MD, HTML)
- Text chunker (fixed, sentence, semantic)
- Entity extractor (using Claude)
- Graph builder (NetworkX)
- Pipeline orchestrator

### Phase 3-7: Strategy Implementations (Upcoming)

## Installation

### Prerequisites

- Python 3.10 or higher
- API keys:
  - `ANTHROPIC_API_KEY` (for Claude)
  - `VOYAGE_API_KEY` (for embeddings)

### Setup

```bash
# Clone repository
cd graph-unified

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Configuration

Set your API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export VOYAGE_API_KEY="your-voyage-key"
```

Choose a configuration profile:

```bash
# Development (fast, cheap)
cp settings-dev.yaml settings.yaml

# Production (balanced)
cp settings-prod.yaml settings.yaml

# Research (high quality)
cp settings-research.yaml settings.yaml
```

## Quick Start

### Configuration Loading

```python
from pathlib import Path
from graphunified.config.settings import Settings

# Load configuration
settings = Settings.load(Path("settings.yaml"))

print(f"Using model: {settings.llm.model}")
print(f"Chunk size: {settings.chunking.chunk_size}")
```

### Working with Data Models

```python
from uuid import uuid4
from graphunified.config.models import Document, Chunk

# Create a document
doc = Document(
    id=uuid4(),
    filename="example.txt",
    text="Your document text here",
    metadata={"source": "example"},
    char_count=100,
    token_count=25,
)

# Create chunks
chunk = Chunk(
    id=uuid4(),
    document_id=doc.id,
    chunk_index=0,
    text="Chunk text",
    start_char=0,
    end_char=10,
    token_count=5,
)
```

### Storage Operations

```python
from pathlib import Path
from graphunified.storage.parquet_store import ParquetStore

# Initialize storage
store = ParquetStore(Path("./output"))

# Save documents
await store.save_documents([doc])
await store.save_chunks([chunk])
await store.flush()

# Load documents
async for document in store.load_documents():
    print(f"Loaded: {document.filename}")
```

### Token Counting

```python
from graphunified.utils.tokenizer import count_tokens

text = "Example text to tokenize"
token_count = count_tokens(text)
print(f"Tokens: {token_count}")
```

## Configuration Profiles

### Development Profile (`settings-dev.yaml`)
- **Model:** Claude 3 Haiku (fastest, cheapest)
- **Chunk size:** 256 tokens
- **Rate limit:** 10 requests/min
- **Use case:** Local development and testing

### Production Profile (`settings-prod.yaml`)
- **Model:** Claude 3.5 Sonnet (balanced)
- **Chunk size:** 512 tokens
- **Rate limit:** 50 requests/min
- **Use case:** Production deployments

### Research Profile (`settings-research.yaml`)
- **Model:** Claude Opus 4.6 (highest quality)
- **Chunk size:** 1024 tokens
- **Max gleanings:** 2 (multiple extraction passes)
- **Use case:** Research and evaluation

## Testing

### Run Unit Tests

```bash
# All tests
pytest tests/unit/ -v

# With coverage report
pytest tests/unit/ -v --cov=graphunified --cov-report=html

# Specific test module
pytest tests/unit/config/test_models.py -v
```

### Verification Script

```bash
python verify_phase1_no_api.py
```

Expected output:
```
✓ Test 1: Configuration Loading
✓ Test 2: Token Counting
✓ Test 3: Data Models
✓ Test 4: Storage Operations
✓ Test 5: Configuration Validation

Phase 1 Implementation Verification Complete!
```

## Architecture

### Project Structure

```
graphunified/
├── config/              # Configuration and data models
│   ├── defaults.py      # Default values
│   ├── models.py        # Pydantic data models
│   ├── settings.py      # Settings schema
│   └── validation.py    # Validation utilities
├── storage/             # Storage backends
│   ├── base.py          # Storage interfaces
│   ├── parquet_store.py # Parquet implementation
│   └── schemas.py       # PyArrow schemas
└── utils/               # Utility modules
    ├── embedding.py     # Embedding client
    ├── llm.py           # LLM client
    ├── logging.py       # Logging config
    └── tokenizer.py     # Tokenization
```

### Key Design Patterns

1. **Async Throughout**: All I/O operations use asyncio
2. **Dual Rate Limiting**: Track both requests/min and tokens/min
3. **Environment Variables**: No hardcoded secrets
4. **Batch Operations**: Buffer writes for efficiency
5. **Lazy Loading**: Use AsyncIterator for large datasets
6. **Strong Typing**: Pydantic models with validation

## API Clients

### Claude LLM Client

```python
from graphunified.utils.llm import ClaudeClient
from graphunified.config.settings import Settings

settings = Settings.load(Path("settings.yaml"))
client = ClaudeClient.from_config(settings.llm)

# Generate text
response, input_tokens, output_tokens = await client.generate(
    prompt="What is RAG?",
    temperature=0.0,
    max_tokens=500,
)
```

Features:
- Dual rate limiting (requests + tokens)
- Automatic retry with exponential backoff
- Token counting before API calls
- Sliding window rate limiter

### Voyage AI Embedding Client

```python
from graphunified.utils.embedding import EmbeddingClient

client = EmbeddingClient.from_config(settings.embedding)

# Generate embeddings (batch)
texts = ["Text 1", "Text 2", "Text 3"]
embeddings = await client.embed(texts)

# Generate query embedding
query_embedding = await client.embed_query("Search query")
```

Features:
- Automatic batching (default 128 texts)
- L2 normalization
- Dimension validation
- Separate query vs. document embeddings

## Storage

### Parquet Backend

```python
from graphunified.storage.parquet_store import ParquetStore

store = ParquetStore(
    root_dir=Path("./output"),
    batch_size=1000,
    compression="snappy",
)

# Save with automatic batching
await store.save_documents(documents)
await store.save_chunks(chunks)
await store.flush()

# Lazy loading
async for doc in store.load_documents():
    process(doc)
```

Features:
- Automatic batch flushing
- Snappy compression (fast, good ratio)
- Lazy loading with AsyncIterator
- PyArrow schema validation
- JSON serialization for nested fields

## Development

### Code Style

- **Formatter:** Black (line length: 100)
- **Linter:** Ruff
- **Type Checker:** mypy
- **Testing:** pytest with pytest-asyncio

### Running Linters

```bash
# Format code
black graphunified tests

# Lint code
ruff check graphunified tests

# Type check
mypy graphunified
```

### Adding New Tests

```python
import pytest
from graphunified.config.models import Document

@pytest.mark.asyncio
async def test_my_feature(sample_document):
    """Test description."""
    # Your test here
    assert sample_document.filename == "test_doc.txt"
```

## Contributing

This project follows a phased development approach:

1. **Phase 1** (✅ Complete): Foundation
2. **Phase 2** (Next): Shared Pipeline
3. **Phase 3**: Naive & Hybrid RAG
4. **Phase 4**: GraphRAG (Local & Global)
5. **Phase 5**: LightRAG
6. **Phase 6**: HippoRAG
7. **Phase 7**: Evaluation & Benchmarking

## License

MIT License - See LICENSE file for details

## Acknowledgments

This project implements techniques from:
- **GraphRAG**: Microsoft Research
- **LightRAG**: LightRAG paper
- **HippoRAG**: Neurologically-inspired RAG

## Support

For issues, questions, or contributions, please open a GitHub issue.

---

**Current Version:** 0.1.0
**Status:** Phase 1 Complete (Foundation)
**Last Updated:** 2026-02-15
