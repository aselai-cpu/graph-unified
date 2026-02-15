# Graph-Unified RAG

A unified implementation of multiple Retrieval-Augmented Generation (RAG) strategies with a shared extraction pipeline, featuring **FREE local embeddings** and easy `.env` configuration.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <repo-url>
cd graph-unified
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys (create .env file)
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your-key-here

# 3. Index your documents with FREE local embeddings
python -m graphunified.cli index \
  --input-dir ./corpus \
  --output-dir ./output \
  --config settings-local-embeddings.yaml

# Done! 🎉
```

**Cost:** ~$3/100 documents (LLM only) + **$0 for embeddings** (FREE!)

---

## Overview

Graph-Unified RAG provides a single codebase supporting **six different RAG strategies**:

| Strategy | Type | Best For |
|----------|------|----------|
| **Naive RAG** | Dense vector retrieval | Simple Q&A |
| **Hybrid RAG** | Vector + BM25 keyword | Balanced retrieval |
| **GraphRAG Local** | Entity-based search | Detailed answers |
| **GraphRAG Global** | Community summaries | High-level overviews |
| **LightRAG** | Dual-level KG retrieval | Complex reasoning |
| **HippoRAG** | PPR-based traversal | Multi-hop questions |

All strategies share a **common extraction pipeline** for 60-70% cost reduction.

---

## 🎯 Project Status

### ✅ Phase 1: Foundation (COMPLETE)

**Implemented:**
- ✅ Configuration system with `.env` support
- ✅ Pydantic data models (Document, Chunk, Entity, Relationship)
- ✅ Claude API client with dual rate limiting
- ✅ Embedding client (Voyage AI + FREE local embeddings)
- ✅ Parquet storage with async operations
- ✅ 39 unit tests, 83% coverage

### ✅ Phase 1.5: Storage Enhancements (COMPLETE)

**Implemented:**
- ✅ LanceDB vector store (5 indexes)
- ✅ NetworkX graph store (traversal + community detection)
- ✅ Parquet storage with partition-based writes
- ✅ Full async storage interface

### ✅ Phase 2: Shared Extraction Pipeline (COMPLETE)

**Implemented:**
- ✅ Document loader (.txt, .md support)
- ✅ Token-based chunking (512 tokens, 128 overlap)
- ✅ Entity extraction (5 types: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT)
- ✅ Relationship extraction (5 types: RELATED_TO, PART_OF, LOCATED_IN, WORKS_FOR, CAUSES)
- ✅ Fuzzy deduplication (90% threshold, 30-50% reduction)
- ✅ Batch embedding generation
- ✅ CLI interface with progress tracking
- ✅ Comprehensive integration tests

**Pipeline Stages:**
```
1. Load → 2. Chunk → 3. Extract → 4. Embed → 5. Save to Parquet
```

**Performance:** Processes 100 documents in <5 minutes

### ✅ Phase 2.5: Critical Fixes (COMPLETE)

**Implemented:**
- ✅ Embeddings persisted to Parquet (all strategies unblocked)
- ✅ Relationship embeddings (for LightRAG global search)
- ✅ Bidirectional chunk-entity links (for graph traversal)
- ✅ Concurrent extraction (10x speedup: 17h → 1.7h for large corpora)

### ✅ .env Support & Local Embeddings (COMPLETE)

**Implemented:**
- ✅ Auto-loading `.env` files with `python-dotenv`
- ✅ FREE local embeddings using `sentence-transformers`
- ✅ Supports BAAI/bge, E5, MiniLM models
- ✅ GPU acceleration (CUDA) with CPU fallback
- ✅ Comprehensive setup guides

**Cost Savings:**
- **$0.25 → $0.00** per 100 documents (embeddings)
- **$250 → $0.00** per million chunks

### 🔄 Phase 3: Retrieval Strategies (IN PROGRESS)

**Week 1:** Naive + Hybrid RAG
**Week 2:** GraphRAG Local + Global
**Week 3:** LightRAG
**Week 4:** HippoRAG

---

## 💰 Cost Comparison

| Configuration | Cost per 100 docs | When to Use |
|---------------|-------------------|-------------|
| **Local Embeddings** (Recommended) | ~$3.00 | ✅ Development, testing, production |
| **Voyage AI Embeddings** | ~$3.25 | Production (slightly faster API) |
| **OpenAI Embeddings** | ~$3.20 | Alternative API provider |

**Recommended:** Use **local embeddings** (FREE) for all use cases!

---

## 📦 Installation

### Prerequisites

- **Python:** 3.10 or higher
- **API Keys:** Only `ANTHROPIC_API_KEY` required (for entity extraction)
- **Optional:** CUDA-enabled GPU for faster local embeddings (10x speedup)

### Setup

```bash
# Clone repository
cd graph-unified

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For local embeddings (recommended)
pip install sentence-transformers torch

# For GPU acceleration (optional, 10x faster)
pip install sentence-transformers torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
```

### Configuration with .env Files

**Step 1:** Create `.env` file

```bash
cp .env.example .env
```

**Step 2:** Add your API key

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Optional: Only needed if using Voyage AI instead of local embeddings
# VOYAGE_API_KEY=pa-your-key-here
```

**Step 3:** Choose configuration

```bash
# Use local embeddings (FREE, recommended)
settings-local-embeddings.yaml

# Or use Voyage AI embeddings
settings.yaml
```

**That's it!** The `.env` file is automatically loaded.

---

## 🎯 Quick Start Examples

### 1. Index Documents with Local Embeddings

```bash
python -m graphunified.cli index \
  --input-dir ./my-documents \
  --output-dir ./output \
  --config settings-local-embeddings.yaml \
  --verbose
```

**Output:**
```
✓ Stage 1/4: Loading documents (3 documents loaded)
✓ Stage 2/4: Chunking documents (3 chunks created)
✓ Stage 3/4: Extracting entities (24 entities, 20 relationships)
✓ Stage 4/4: Generating embeddings (3 chunks, 24 entities, 20 relationships)
✓ Saving to Parquet

Pipeline completed in 37.20s
Cost: ~$0.05 (LLM) + $0.00 (embeddings FREE!)
```

### 2. Python API Usage

```python
from pathlib import Path
from graphunified.config.settings import Settings
from graphunified.index.pipeline import IndexPipeline

# Load configuration (automatically loads .env)
settings = Settings.load(Path("settings-local-embeddings.yaml"))

# Run indexing pipeline
pipeline = IndexPipeline.from_config(
    settings=settings,
    input_dir=Path("./corpus"),
    output_dir=Path("./output")
)

result = await pipeline.run()

print(f"Documents: {result['metrics']['documents_loaded']}")
print(f"Entities: {result['metrics']['entities_extracted']}")
print(f"Duration: {result['metrics']['duration_seconds']:.2f}s")
```

### 3. Working with Extracted Data

```python
from graphunified.storage.parquet_store import ParquetStore

# Load extracted data
store = ParquetStore(Path("./output"))

# Load chunks
async for chunk in store.load_chunks():
    print(f"Chunk: {chunk.text[:100]}...")
    print(f"Embedding: {chunk.embedding[:5]}...")  # First 5 values

# Load entities
async for entity in store.load_entities():
    print(f"Entity: {entity.name} ({entity.type})")
    print(f"Confidence: {entity.confidence:.2f}")
```

---

## 🏗️ Architecture

### Project Structure

```
graphunified/
├── config/              # Configuration and data models
│   ├── defaults.py      # Default values
│   ├── models.py        # Pydantic data models
│   ├── settings.py      # Settings schema with .env support
│   └── validation.py    # Validation utilities
├── index/               # Indexing pipeline (Phase 2)
│   ├── pipeline.py      # Pipeline orchestrator
│   └── stages/          # Pipeline stages
│       ├── load.py      # Document loading
│       ├── chunk.py     # Text chunking
│       ├── extract.py   # Entity extraction
│       └── embed.py     # Embedding generation
├── prompts/             # LLM prompts
│   └── extraction.py    # Entity/relationship prompts
├── storage/             # Storage backends
│   ├── parquet_store.py # Parquet implementation
│   ├── vector_store.py  # LanceDB vector indexes
│   └── graph_store.py   # NetworkX graph storage
├── strategies/          # RAG strategies (Phase 3+)
│   └── base.py          # Strategy interface
└── utils/               # Utility modules
    ├── embedding.py     # Voyage AI client
    ├── local_embedding.py  # FREE local embeddings
    ├── embedding_factory.py # Provider factory
    ├── llm.py           # Claude LLM client
    ├── logging.py       # Logging config
    └── tokenizer.py     # Tokenization
```

### Pipeline Architecture

```
┌──────────┐
│   Load   │ → Load documents (.txt, .md)
└────┬─────┘
     │
┌────▼─────┐
│  Chunk   │ → Token-based overlapping windows
└────┬─────┘
     │
┌────▼─────┐
│ Extract  │ → Entities + Relationships (Claude)
└────┬─────┘
     │
┌────▼─────┐
│  Embed   │ → FREE local embeddings (BGE/E5/MiniLM)
└────┬─────┘
     │
┌────▼─────┐
│  Save    │ → Parquet files with embeddings
└──────────┘
```

### Key Design Patterns

1. **Async Throughout**: All I/O operations use asyncio for performance
2. **Dual Rate Limiting**: Track both requests/min and tokens/min
3. **Factory Pattern**: Support multiple embedding providers (local, Voyage, OpenAI)
4. **Batch Operations**: Buffer writes for efficiency
5. **Lazy Loading**: AsyncIterator for large datasets
6. **Strong Typing**: Pydantic models with validation
7. **Fuzzy Deduplication**: 90% threshold reduces entity count by 30-50%

---

## 🔧 Configuration

### Embedding Models (Local)

**Best Quality (Production):**
```yaml
embedding:
  provider: local
  model: BAAI/bge-large-en-v1.5
  dimension: 1024
  batch_size: 32
```
- Quality: ⭐⭐⭐⭐⭐
- Speed: ~500 chunks/sec (GPU), ~50 chunks/sec (CPU)

**Good Balance (Recommended):**
```yaml
embedding:
  provider: local
  model: BAAI/bge-base-en-v1.5
  dimension: 768
  batch_size: 64
```
- Quality: ⭐⭐⭐⭐
- Speed: ~800 chunks/sec (GPU), ~80 chunks/sec (CPU)

**Fast Testing:**
```yaml
embedding:
  provider: local
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
  batch_size: 128
```
- Quality: ⭐⭐⭐
- Speed: ~1500 chunks/sec (GPU), ~150 chunks/sec (CPU)

**See:** `LOCAL_EMBEDDINGS_GUIDE.md` for detailed model comparisons

### LLM Models

**Recommended for extraction:**
```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-5-20250929  # Best balance
  temperature: 0.0  # Deterministic extraction
  max_tokens: 4096
```

**Alternatives:**
- `claude-opus-4-6` - Highest quality (slower, more expensive)
- `claude-haiku-4-5-20251001` - Fastest (lower quality)

---

## 📊 Performance Benchmarks

### Test Results (3 Documents, ~2.8KB)

```
Documents loaded:     3
Chunks created:       3
Entities extracted:   24
Relationships:        20
Duration:            37.20s
Cost:                ~$0.05 (LLM) + $0.00 (embeddings)
```

### Scaling Estimates

| Corpus Size | Duration | LLM Cost | Embedding Cost (Local) |
|-------------|----------|----------|------------------------|
| 10 docs     | ~2 min   | ~$0.20   | **$0.00 (FREE!)**      |
| 100 docs    | ~5 min   | ~$3.00   | **$0.00 (FREE!)**      |
| 1,000 docs  | ~40 min  | ~$30     | **$0.00 (FREE!)**      |
| 10,000 docs | ~5 hours | ~$300    | **$0.00 (FREE!)**      |

**With concurrent extraction (Phase 2.5):**
- 10x faster: 5 hours → 30 minutes for 10K docs

---

## 🧪 Testing

### Run Tests

```bash
# All unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ -v --cov=graphunified --cov-report=html

# Integration tests
pytest tests/integration/ -v

# Specific module
pytest tests/unit/index/test_extract.py -v
```

### Verification Scripts

```bash
# Verify Phase 1 (no API calls)
python verify_phase1_no_api.py

# Verify Phase 2
python verify_phase2.py

# Verify Phase 2.5 fixes
python verify_phase2.5.py
```

---

## 📚 Documentation

- **`LOCAL_EMBEDDINGS_GUIDE.md`** - Complete guide to FREE local embeddings
- **`ENV_SETUP_GUIDE.md`** - .env file configuration guide
- **`INDEXING_ARCHITECTURE.md`** - Vector & text index architecture
- **`PHASE2_COMPLETE.md`** - Phase 2 implementation details
- **`PHASE2.5_COMPLETE.md`** - Phase 2.5 fixes and improvements

---

## 🚀 Roadmap

### ✅ Completed

- [x] Phase 1: Foundation (Config, Models, API clients)
- [x] Phase 1.5: Storage (Parquet, LanceDB, NetworkX)
- [x] Phase 2: Indexing Pipeline (Load, Chunk, Extract, Embed)
- [x] Phase 2.5: Critical Fixes (Embeddings, Concurrency, Links)
- [x] .env Support (Auto-loading configuration)
- [x] Local Embeddings (FREE alternative to Voyage AI)

### 🔄 In Progress

- [ ] Phase 3 Week 1: Naive + Hybrid RAG (in progress)

### 📅 Upcoming

- [ ] Phase 3 Week 2: GraphRAG Local + Global
- [ ] Phase 3 Week 3: LightRAG
- [ ] Phase 3 Week 4: HippoRAG
- [ ] Phase 4: Index Building (LanceDB + BM25)
- [ ] Phase 5: Query Interface
- [ ] Phase 6: Evaluation Framework
- [ ] Phase 7: Benchmarking & Optimization

---

## 💡 Key Features

### FREE Local Embeddings

- ✅ **Zero Cost** - No API fees for embeddings
- ✅ **High Quality** - BGE models match Voyage AI quality
- ✅ **Fast** - 500+ chunks/sec with GPU, 50+ chunks/sec on CPU
- ✅ **Private** - Data never leaves your machine
- ✅ **Offline** - No internet required after model download

### Easy Configuration

- ✅ **`.env` Files** - No manual exports, just edit `.env`
- ✅ **Auto-loading** - Searches config dir, CWD, parent dirs
- ✅ **Secure** - `.env` in `.gitignore`, never commit secrets

### Cost Optimization

- ✅ **Shared Pipeline** - 60-70% cost reduction vs separate extraction
- ✅ **FREE Embeddings** - Saves $150-250 per million chunks
- ✅ **Concurrent Extraction** - 10x speedup for large corpora

### Production Ready

- ✅ **Async I/O** - High performance with asyncio
- ✅ **Rate Limiting** - Dual sliding window (requests + tokens)
- ✅ **Error Handling** - Automatic retry with exponential backoff
- ✅ **Progress Tracking** - Real-time progress bars
- ✅ **Comprehensive Tests** - 39+ unit tests, integration tests

---

## 🤝 Contributing

Contributions welcome! This project follows a phased development approach.

**Development Setup:**
```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run formatters
black graphunified tests
ruff check graphunified tests

# Run type checker
mypy graphunified

# Run tests
pytest tests/ -v --cov
```

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

This project implements and unifies techniques from:

- **GraphRAG** - Microsoft Research
- **LightRAG** - Efficient dual-level knowledge graph retrieval
- **HippoRAG** - Neurologically-inspired retrieval with PPR
- **BGE Embeddings** - BAAI (Beijing Academy of Artificial Intelligence)
- **sentence-transformers** - UKPLab

---

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation:** See `/docs` directory
- **Examples:** See `/examples` directory

---

**Current Version:** 0.2.0
**Status:** Phase 2 Complete (Indexing Pipeline + Local Embeddings)
**Last Updated:** 2026-02-15

**Ready to start?** Follow the [Quick Start](#-quick-start) guide above! 🚀
