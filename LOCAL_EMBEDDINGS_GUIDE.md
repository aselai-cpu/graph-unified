# Local Embeddings Guide (Free, No API Keys!)

## Overview

You can now use **free, local embeddings** with SentenceTransformers instead of paid API services like Voyage AI or OpenAI. This is perfect for:

- ✅ **Development & Testing** - No API costs
- ✅ **Privacy-Sensitive Data** - Embeddings stay local
- ✅ **Offline Usage** - No internet required (after model download)
- ✅ **High Volume** - Unlimited embedding generation
- ✅ **GPU Acceleration** - Fast with CUDA

---

## Quick Start

### 1. Install Dependencies

```bash
# Install sentence-transformers
pip install sentence-transformers torch

# Optional: Install with GPU support
pip install sentence-transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Update Configuration

Use `settings-local-embeddings.yaml` or update your config:

```yaml
embedding:
  provider: local  # Change from "voyage" to "local"
  model: BAAI/bge-large-en-v1.5  # Local model name
  api_key: ""  # Not needed!
  dimension: 1024
  batch_size: 32  # Adjust for your GPU memory
  normalize: true
```

### 3. Configure API Key

**Option A: Use .env file (Recommended)**
```bash
# Create .env file
cp .env.example .env

# Edit .env and add only LLM key
# ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
# No VOYAGE_API_KEY needed!
```

**Option B: Export directly**
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 4. Run Pipeline

```bash
# .env file is automatically loaded
python -m graphunified.cli index \
  --input-dir ./corpus \
  --output-dir ./output \
  --config settings-local-embeddings.yaml
```

**That's it!** Embeddings will be generated locally.

---

## Recommended Models

### Best Quality (Production)

**BAAI/bge-large-en-v1.5** (1024d)
```yaml
embedding:
  provider: local
  model: BAAI/bge-large-en-v1.5
  dimension: 1024
  batch_size: 32
```
- ✅ State-of-the-art quality
- ✅ Best for production use
- ⚠️ Requires ~2GB GPU memory
- Speed: ~500 chunks/sec (GPU), ~50 chunks/sec (CPU)

### Good Balance (Recommended for Most)

**BAAI/bge-base-en-v1.5** (768d)
```yaml
embedding:
  provider: local
  model: BAAI/bge-base-en-v1.5
  dimension: 768
  batch_size: 64
```
- ✅ Excellent quality
- ✅ Faster than large
- ⚠️ Requires ~1.5GB GPU memory
- Speed: ~800 chunks/sec (GPU), ~80 chunks/sec (CPU)

### Fast & Lightweight (Testing)

**sentence-transformers/all-MiniLM-L6-v2** (384d)
```yaml
embedding:
  provider: local
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
  batch_size: 128
```
- ✅ Very fast
- ✅ Works well on CPU
- ⚠️ Lower quality than BGE
- Speed: ~1500 chunks/sec (GPU), ~150 chunks/sec (CPU)

### Alternative: E5 Models

**intfloat/e5-large-v2** (1024d)
```yaml
embedding:
  provider: local
  model: intfloat/e5-large-v2
  dimension: 1024
  batch_size: 32
```
- ✅ Comparable to BGE-large
- ✅ Good alternative
- Note: Requires "query: " prefix for queries (handled automatically)

---

## Performance Comparison

### Speed (chunks per second)

| Model | Dimension | GPU (RTX 3090) | CPU (16-core) | GPU Memory |
|-------|-----------|----------------|---------------|------------|
| **bge-large** | 1024 | ~500 | ~50 | ~2GB |
| **bge-base** | 768 | ~800 | ~80 | ~1.5GB |
| **MiniLM-L6** | 384 | ~1500 | ~150 | ~0.5GB |
| **e5-large** | 1024 | ~450 | ~45 | ~2GB |

### Quality (MTEB Benchmark)

| Model | Retrieval Score | Similarity Score | Overall |
|-------|----------------|------------------|---------|
| **bge-large** | 54.3 | 75.7 | **63.2** |
| **bge-base** | 53.2 | 74.2 | **62.1** |
| **e5-large** | 52.8 | 73.9 | **61.8** |
| **MiniLM-L6** | 49.5 | 68.1 | **58.2** |

**Recommendation**: Use **bge-large** for production, **bge-base** for development.

---

## Cost Comparison

### Processing 1 Million Chunks (~500MB text)

| Provider | Cost | Speed | Notes |
|----------|------|-------|-------|
| **Voyage AI** | $250 | Fast (API) | Best for low volume |
| **OpenAI** | $200 | Fast (API) | Good quality |
| **Cohere** | $150 | Fast (API) | Cheapest API |
| **Local (bge-large)** | **$0** | Med-Fast (GPU) | **FREE!** |
| **Local (MiniLM)** | **$0** | Very Fast (CPU) | **FREE!** |

**Savings**: $150-250 per million chunks!

---

## Setup & Optimization

### GPU Setup (Recommended)

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

### Memory Optimization

**Batch Size Guidelines**:
```yaml
# For 8GB GPU (RTX 3070)
batch_size: 32  # bge-large
batch_size: 64  # bge-base

# For 12GB GPU (RTX 3080)
batch_size: 64  # bge-large
batch_size: 128  # bge-base

# For 24GB GPU (RTX 3090)
batch_size: 128  # bge-large
batch_size: 256  # bge-base
```

### CPU-Only Setup

```yaml
# Works fine on CPU, just slower
embedding:
  provider: local
  model: sentence-transformers/all-MiniLM-L6-v2  # Fastest on CPU
  dimension: 384
  batch_size: 64
```

**Expected Speed**: ~150 chunks/sec on modern CPU (16 cores)

---

## How It Works

### First Run (Downloads Model)
```bash
$ python -m graphunified.cli index -i ./corpus -o ./output

> Loading sentence-transformers model: BAAI/bge-large-en-v1.5
> Downloading model from HuggingFace... (~1.3GB)
> Model loaded on device: cuda
> Generating embeddings for 10000 chunks...
> [████████████████████] 10000/10000 (20 seconds)
```

**Model Cache**: `~/.cache/huggingface/hub/`

### Subsequent Runs (Uses Cached Model)
```bash
$ python -m graphunified.cli index -i ./corpus2 -o ./output2

> Loading sentence-transformers model: BAAI/bge-large-en-v1.5
> Model loaded on device: cuda (cached)
> Generating embeddings for 10000 chunks...
> [████████████████████] 10000/10000 (20 seconds)
```

---

## API vs Local: When to Use Each

### Use Local Embeddings When:
- ✅ High volume (>100K chunks)
- ✅ Privacy-sensitive data
- ✅ Development & testing
- ✅ Budget constraints
- ✅ Offline operation needed
- ✅ Have GPU available

### Use API Embeddings (Voyage/OpenAI) When:
- ✅ Low volume (<10K chunks)
- ✅ No GPU available
- ✅ Need absolute best quality
- ✅ Don't want to manage models
- ✅ Quick one-off tasks

---

## Troubleshooting

### Issue: "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Issue: "CUDA out of memory"
```yaml
# Reduce batch size
embedding:
  batch_size: 16  # Was 32
```

### Issue: Slow on CPU
```yaml
# Use faster model
embedding:
  model: sentence-transformers/all-MiniLM-L6-v2
  dimension: 384
```

### Issue: Model download fails
```bash
# Manual download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
```

---

## Example: Full Pipeline with Local Embeddings

```bash
# 1. Install dependencies
pip install sentence-transformers torch

# 2. Set only LLM API key (no embedding key needed)
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run indexing with local embeddings
python -m graphunified.cli index \
  --input-dir ./my-documents \
  --output-dir ./output \
  --config settings-local-embeddings.yaml \
  --verbose

# Output:
# > Loading sentence-transformers model: BAAI/bge-large-en-v1.5
# > Model loaded on device: cuda
# > Stage 1/4: Loading documents
# > Loaded 1000 documents
# > Stage 2/4: Chunking documents
# > Created 25000 chunks
# > Stage 3/4: Extracting entities and relationships
# > Extracted 5000 entities and 3000 relationships
# > Stage 4/4: Generating embeddings
# > Generating embeddings for 25000 chunks...
# > Embedded 25000 chunks (50 seconds)
# > Generating embeddings for 5000 entities...
# > Embedded 5000 entities (10 seconds)
# > Generating embeddings for 3000 relationships...
# > Embedded 3000 relationships (6 seconds)
# > Pipeline completed in 3.5 minutes
# > Cost: $3.00 (LLM only, embeddings FREE!)
```

---

## Migration from Voyage AI

### Before (Voyage AI)
```yaml
embedding:
  provider: voyage
  model: voyage-3
  api_key: ${VOYAGE_API_KEY}  # Required
  dimension: 1024
```

### After (Local)
```yaml
embedding:
  provider: local
  model: BAAI/bge-large-en-v1.5
  api_key: ""  # Not needed!
  dimension: 1024
```

**Same quality, zero cost!**

---

## Supported Models

All models from https://huggingface.co/models?library=sentence-transformers work:

**Popular Choices**:
- `BAAI/bge-large-en-v1.5` (English, 1024d)
- `BAAI/bge-base-en-v1.5` (English, 768d)
- `BAAI/bge-small-en-v1.5` (English, 384d)
- `intfloat/e5-large-v2` (English, 1024d)
- `intfloat/multilingual-e5-large` (100+ languages, 1024d)
- `sentence-transformers/all-MiniLM-L6-v2` (English, 384d)

**Multilingual**:
- `intfloat/multilingual-e5-large` (1024d, 100+ languages)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (768d)

---

## Summary

✅ **Zero Cost** - No API fees
✅ **High Quality** - BGE models match Voyage AI quality
✅ **Fast** - 500+ chunks/sec with GPU
✅ **Private** - Data stays local
✅ **Offline** - No internet needed

**Recommendation**: Use **BAAI/bge-large-en-v1.5** for production-quality local embeddings.

---

For questions or issues, see: `PHASE2.5_COMPLETE.md`
