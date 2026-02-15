# Configuration Specification

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document provides complete specifications for the `settings.yaml` configuration file, including field-by-field documentation, validation rules, default values, examples, and configuration profiles.

## Configuration File Structure

```yaml
version: "1.0"
llm: {...}
embedding: {...}
chunking: {...}
extraction: {...}
strategies: {...}
storage: {...}
query: {...}
performance: {...}
logging: {...}
prompts: {...}  # Optional custom prompts
```

---

## Field Specifications

### version

**Type:** String
**Required:** Yes
**Pattern:** `^\d+\.\d+$` (semantic version)
**Default:** "1.0"
**Description:** Configuration schema version for migration tracking.

**Example:**
```yaml
version: "1.0"
```

---

## LLM Configuration

### llm.provider

**Type:** String (enum)
**Required:** Yes
**Options:** `anthropic`, `openai`, `azure`
**Default:** `anthropic`
**Description:** LLM provider for extraction and generation.

### llm.model

**Type:** String
**Required:** Yes
**Default:** `claude-3-5-sonnet-20241022`
**Options:**
- `claude-3-5-sonnet-20241022` - Balanced (recommended)
- `claude-3-haiku-20240307` - Fast, economical
- `claude-opus-4-6` - Highest quality

**Description:** Model identifier for the LLM.

### llm.api_key

**Type:** String
**Required:** Yes
**Default:** `${ANTHROPIC_API_KEY}`
**Validation:** Non-empty string
**Description:** API key for LLM provider. Use environment variable syntax `${VAR_NAME}`.

**Security Note:** Never hardcode API keys. Always use environment variables.

### llm.temperature

**Type:** Float
**Required:** No
**Range:** 0.0 - 2.0
**Default:** 0.0
**Description:** Sampling temperature for LLM generation.

**Recommendations:**
- **Extraction:** 0.0 (deterministic)
- **Summarization:** 0.1 (slight creativity)
- **Generation:** 0.3 (balanced)

### llm.max_tokens

**Type:** Integer
**Required:** No
**Range:** 100 - 200000
**Default:** 4096
**Description:** Maximum tokens to generate per request.

**Recommendations:**
- **Extraction:** 1000-2000
- **Summarization:** 2000-4000
- **Generation:** 500-1000

### llm.timeout

**Type:** Integer
**Required:** No
**Range:** 10 - 600
**Default:** 60
**Unit:** Seconds
**Description:** Request timeout for LLM API calls.

### llm.rate_limit

Configuration for API rate limiting.

#### llm.rate_limit.requests_per_minute

**Type:** Integer
**Range:** 1 - 1000
**Default:** 50
**Description:** Maximum requests per minute to LLM API.

**Tier Recommendations:**
- Free tier: 5
- Tier 1: 50
- Tier 2: 100
- Tier 3: 500

#### llm.rate_limit.tokens_per_minute

**Type:** Integer
**Range:** 1000 - 5000000
**Default:** 40000
**Description:** Maximum tokens per minute (input + output).

### llm.retry

Configuration for retry logic.

#### llm.retry.max_attempts

**Type:** Integer
**Range:** 1 - 10
**Default:** 3
**Description:** Maximum retry attempts for failed requests.

#### llm.retry.backoff_factor

**Type:** Float
**Range:** 1.0 - 10.0
**Default:** 2.0
**Description:** Exponential backoff multiplier. Wait time = `backoff_factor ^ attempt`.

**Example:**
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.0
  max_tokens: 4096
  timeout: 60
  rate_limit:
    requests_per_minute: 50
    tokens_per_minute: 40000
  retry:
    max_attempts: 3
    backoff_factor: 2.0
```

---

## Embedding Configuration

### embedding.provider

**Type:** String (enum)
**Options:** `voyage`, `openai`, `cohere`
**Default:** `voyage`
**Description:** Embedding model provider.

### embedding.model

**Type:** String
**Default:** `voyage-3`
**Options:**
- `voyage-3` - 1024d, high quality (recommended)
- `voyage-2` - 1024d, lower cost
- `text-embedding-3-large` - OpenAI, 3072d
- `embed-english-v3.0` - Cohere, 1024d

**Description:** Embedding model identifier.

### embedding.api_key

**Type:** String
**Required:** Yes
**Default:** `${VOYAGE_API_KEY}`
**Description:** API key for embedding provider.

### embedding.dimension

**Type:** Integer
**Range:** 384 - 4096
**Default:** 1024
**Description:** Embedding vector dimension. Must match model output.

### embedding.batch_size

**Type:** Integer
**Range:** 1 - 512
**Default:** 128
**Description:** Batch size for embedding API calls.

**Recommendations:**
- Voyage AI: 128
- OpenAI: 100
- Cohere: 96

### embedding.normalize

**Type:** Boolean
**Default:** true
**Description:** Whether to L2-normalize embeddings.

**Example:**
```yaml
embedding:
  provider: "voyage"
  model: "voyage-3"
  api_key: "${VOYAGE_API_KEY}"
  dimension: 1024
  batch_size: 128
  normalize: true
```

---

## Chunking Configuration

### chunking.strategy

**Type:** String (enum)
**Options:** `fixed`, `sentence`, `paragraph`, `semantic`
**Default:** `fixed`
**Description:** Chunking strategy.

**Strategy Details:**
- **fixed:** Fixed token count (simple, fast)
- **sentence:** Sentence boundary-aware
- **paragraph:** Paragraph boundary-aware
- **semantic:** Semantic similarity-based (experimental)

### chunking.chunk_size

**Type:** Integer
**Range:** 128 - 4096
**Default:** 512
**Unit:** Tokens
**Description:** Target chunk size in tokens.

**Recommendations:**
- **Small chunks (256-512):** Better precision, more chunks
- **Medium chunks (512-1024):** Balanced (recommended)
- **Large chunks (1024-2048):** Better context, fewer chunks

### chunking.chunk_overlap

**Type:** Integer
**Range:** 0 - 512
**Default:** 64
**Unit:** Tokens
**Description:** Overlap between consecutive chunks.

**Recommendations:**
- **10-15% of chunk_size:** Typical
- **0:** No overlap (not recommended)

**Validation:** Must be less than `chunk_size`.

### chunking.respect_boundaries

**Type:** Boolean
**Default:** true
**Description:** Whether to respect sentence/paragraph boundaries.

**Effect:** May cause chunk sizes to vary ±20% from target.

### chunking.encoding_name

**Type:** String
**Default:** `cl100k_base`
**Options:**
- `cl100k_base` - GPT-3.5/4, Claude (recommended)
- `p50k_base` - GPT-3
- `r50k_base` - Legacy

**Description:** Tiktoken encoding for tokenization.

**Example:**
```yaml
chunking:
  strategy: "fixed"
  chunk_size: 512
  chunk_overlap: 64
  respect_boundaries: true
  encoding_name: "cl100k_base"
```

---

## Extraction Configuration

### extraction.entity_types

**Type:** List[String]
**Default:** `["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT", "TECHNOLOGY"]`
**Description:** Entity types to extract.

**Standard Types:**
- PERSON
- ORGANIZATION
- LOCATION
- EVENT
- DATE
- CONCEPT
- TECHNOLOGY
- PRODUCT
- OTHER

**Custom Types:** Define domain-specific types (e.g., "DRUG", "GENE", "LEGAL_CASE").

### extraction.relationship_types

**Type:** List[String]
**Default:** `["RELATED_TO", "PART_OF", "LOCATED_IN", "WORKS_FOR", "CAUSES"]`
**Description:** Relationship types to extract.

**Standard Types:**
- RELATED_TO
- PART_OF
- LOCATED_IN
- WORKS_FOR
- CAUSES
- SIMILAR_TO
- OPPOSITE_OF
- PRECEDES
- OTHER

### extraction.max_gleanings

**Type:** Integer
**Range:** 0 - 3
**Default:** 1
**Description:** Additional extraction passes to improve recall.

**Effect:**
- **0:** Single pass (faster, lower recall)
- **1:** Two passes (balanced, recommended)
- **2+:** Multiple passes (slower, higher recall)

**Cost Impact:** Each gleaning doubles extraction cost for that chunk.

### extraction.min_confidence

**Type:** Float
**Range:** 0.0 - 1.0
**Default:** 0.7
**Description:** Minimum confidence threshold for extracted entities/relationships.

**Recommendations:**
- **High precision:** 0.8-0.9
- **Balanced:** 0.7 (recommended)
- **High recall:** 0.5-0.6

### extraction.enable_coreference

**Type:** Boolean
**Default:** false
**Description:** Enable coreference resolution (experimental).

**Effect:** Resolves pronouns to entities ("he" → "John Smith").

**Example:**
```yaml
extraction:
  entity_types:
    - "PERSON"
    - "ORGANIZATION"
    - "LOCATION"
    - "CONCEPT"
  relationship_types:
    - "WORKS_FOR"
    - "LOCATED_IN"
    - "RELATED_TO"
  max_gleanings: 1
  min_confidence: 0.7
  enable_coreference: false
```

---

## Strategies Configuration

Configuration for each retrieval strategy.

### strategies.naive

#### strategies.naive.enabled

**Type:** Boolean
**Default:** true
**Description:** Enable Naive RAG strategy.

### strategies.hybrid

#### strategies.hybrid.enabled

**Type:** Boolean
**Default:** true

#### strategies.hybrid.alpha

**Type:** Float
**Range:** 0.0 - 1.0
**Default:** 0.5
**Description:** Weight for dense retrieval. Sparse weight = 1 - alpha.

#### strategies.hybrid.bm25_k1

**Type:** Float
**Range:** 0.5 - 3.0
**Default:** 1.5
**Description:** BM25 term frequency saturation parameter.

#### strategies.hybrid.bm25_b

**Type:** Float
**Range:** 0.0 - 1.0
**Default:** 0.75
**Description:** BM25 document length normalization parameter.

### strategies.graphrag

#### strategies.graphrag.enabled

**Type:** Boolean
**Default:** true

#### strategies.graphrag.leiden_resolution

**Type:** Float
**Range:** 0.1 - 2.0
**Default:** 1.0
**Description:** Leiden community detection resolution parameter.

**Effect:**
- **Low (0.5):** Fewer, larger communities
- **Medium (1.0):** Balanced (recommended)
- **High (1.5-2.0):** Many small communities

#### strategies.graphrag.max_community_size

**Type:** Integer
**Range:** 5 - 100
**Default:** 50
**Description:** Maximum entities per community.

#### strategies.graphrag.generate_reports

**Type:** Boolean
**Default:** true
**Description:** Generate LLM summaries for communities (for global search).

### strategies.lightrag

#### strategies.lightrag.enabled

**Type:** Boolean
**Default:** true

#### strategies.lightrag.entity_weight

**Type:** Float
**Range:** 0.0 - 1.0
**Default:** 0.6
**Description:** Weight for entity index in dual retrieval.

### strategies.hipporag

#### strategies.hipporag.enabled

**Type:** Boolean
**Default:** true

#### strategies.hipporag.ppr_alpha

**Type:** Float
**Range:** 0.5 - 0.95
**Default:** 0.85
**Description:** Personalized PageRank damping factor.

**Example:**
```yaml
strategies:
  naive:
    enabled: true
  hybrid:
    enabled: true
    alpha: 0.5
    bm25_k1: 1.5
    bm25_b: 0.75
  graphrag:
    enabled: true
    leiden_resolution: 1.0
    max_community_size: 50
    generate_reports: true
  lightrag:
    enabled: true
    entity_weight: 0.6
  hipporag:
    enabled: true
    ppr_alpha: 0.85
```

---

## Storage Configuration

### storage.root_dir

**Type:** String (Path)
**Default:** `./output`
**Description:** Root directory for all storage files.

### storage.parquet_compression

**Type:** String (enum)
**Options:** `snappy`, `gzip`, `brotli`, `none`
**Default:** `snappy`
**Description:** Compression codec for Parquet files.

**Recommendations:**
- **snappy:** Fast, good ratio (recommended)
- **gzip:** Better ratio, slower
- **brotli:** Best ratio, slowest

### storage.vector_db

#### storage.vector_db.backend

**Type:** String (enum)
**Options:** `lancedb`, `faiss`, `qdrant`
**Default:** `lancedb`
**Description:** Vector database backend.

#### storage.vector_db.index_type

**Type:** String (enum)
**Options:** `IVF_FLAT`, `IVF_PQ`, `HNSW`
**Default:** `IVF_FLAT`
**Description:** Vector index type.

**Recommendations:**
- **IVF_FLAT:** <1M vectors, best accuracy
- **IVF_PQ:** 1M-10M vectors, compressed
- **HNSW:** >10M vectors, fastest search

### storage.graph_format

**Type:** String (enum)
**Options:** `graphml`, `gexf`, `pickle`
**Default:** `graphml`
**Description:** Graph serialization format.

**Example:**
```yaml
storage:
  root_dir: "./output"
  parquet_compression: "snappy"
  vector_db:
    backend: "lancedb"
    index_type: "IVF_FLAT"
  graph_format: "graphml"
```

---

## Query Configuration

### query.default_strategy

**Type:** String (enum)
**Options:** `naive`, `hybrid`, `graphrag_local`, `graphrag_global`, `lightrag`, `hipporag`, `auto`
**Default:** `auto`
**Description:** Default retrieval strategy. `auto` uses query router.

### query.top_k

**Type:** Integer
**Range:** 1 - 100
**Default:** 10
**Description:** Default number of contexts to retrieve.

### query.generation

#### query.generation.enabled

**Type:** Boolean
**Default:** true
**Description:** Whether to generate responses (vs. retrieval only).

#### query.generation.temperature

**Type:** Float
**Range:** 0.0 - 1.0
**Default:** 0.3
**Description:** Temperature for response generation.

#### query.generation.max_tokens

**Type:** Integer
**Default:** 1000
**Description:** Maximum tokens for generated response.

### query.routing

#### query.routing.enabled

**Type:** Boolean
**Default:** true
**Description:** Enable automatic query routing.

#### query.routing.strategy

**Type:** String (enum)
**Options:** `rule_based`, `llm_based`
**Default:** `rule_based`
**Description:** Query routing strategy.

**Example:**
```yaml
query:
  default_strategy: "auto"
  top_k: 10
  generation:
    enabled: true
    temperature: 0.3
    max_tokens: 1000
  routing:
    enabled: true
    strategy: "rule_based"
```

---

## Performance Configuration

### performance.workers

**Type:** Integer
**Range:** 1 - 32
**Default:** 10
**Description:** Number of parallel workers for I/O tasks.

### performance.batch_size

**Type:** Integer
**Range:** 1 - 100
**Default:** 10
**Description:** Default batch size for processing.

### performance.cache_embeddings

**Type:** Boolean
**Default:** true
**Description:** Cache embeddings to avoid recomputation.

### performance.enable_profiling

**Type:** Boolean
**Default:** false
**Description:** Enable performance profiling (overhead ~5%).

**Example:**
```yaml
performance:
  workers: 10
  batch_size: 10
  cache_embeddings: true
  enable_profiling: false
```

---

## Logging Configuration

### logging.level

**Type:** String (enum)
**Options:** `DEBUG`, `INFO`, `WARNING`, `ERROR`
**Default:** `INFO`
**Description:** Logging level.

### logging.format

**Type:** String (enum)
**Options:** `text`, `json`
**Default:** `text`
**Description:** Log output format.

### logging.output

**Type:** String (enum)
**Options:** `stdout`, `file`, `both`
**Default:** `stdout`
**Description:** Where to write logs.

### logging.file_path

**Type:** String (Path)
**Default:** `./logs/graphunified.log`
**Description:** Log file path (if `output` includes `file`).

**Example:**
```yaml
logging:
  level: "INFO"
  format: "text"
  output: "stdout"
  file_path: "./logs/graphunified.log"
```

---

## Environment Variable Substitution

**Syntax:** `${VARIABLE_NAME}`

**Example:**
```yaml
llm:
  api_key: "${ANTHROPIC_API_KEY}"
embedding:
  api_key: "${VOYAGE_API_KEY}"
```

**Resolution:**
1. Check environment variable
2. If not found, raise `ConfigurationError`

---

## Configuration Validation

### Validation Rules

```python
from pydantic import field_validator

class ChunkingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

    @field_validator('chunk_overlap')
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError("chunk_overlap must be < chunk_size")
        return v
```

### Validation on Load

```python
try:
    settings = Settings.load("settings.yaml")
    settings.validate_completeness()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

---

## Configuration Profiles

### Development Profile

```yaml
version: "1.0"
llm:
  model: "claude-3-haiku-20240307"  # Fastest, cheapest
  temperature: 0.0
  rate_limit:
    requests_per_minute: 10
chunking:
  chunk_size: 256  # Smaller for fast testing
extraction:
  max_gleanings: 0  # Single pass
strategies:
  graphrag:
    generate_reports: false  # Skip expensive reports
logging:
  level: "DEBUG"
```

### Production Profile

```yaml
version: "1.0"
llm:
  model: "claude-3-5-sonnet-20241022"  # Balanced
  temperature: 0.0
  rate_limit:
    requests_per_minute: 50
chunking:
  chunk_size: 512
extraction:
  max_gleanings: 1
strategies:
  graphrag:
    generate_reports: true
logging:
  level: "INFO"
performance:
  enable_profiling: true
```

### Research Profile

```yaml
version: "1.0"
llm:
  model: "claude-opus-4-6"  # Highest quality
  temperature: 0.0
chunking:
  chunk_size: 1024  # Larger chunks
extraction:
  max_gleanings: 2  # Multiple passes
  min_confidence: 0.6  # High recall
strategies:
  graphrag:
    generate_reports: true
    leiden_resolution: 0.8  # Finer communities
logging:
  level: "DEBUG"
```

---

## Summary

This specification defines:

- **Complete field documentation** for all configuration options
- **Type specifications** with ranges and validation
- **Default values** with rationale
- **Configuration profiles** for different use cases
- **Environment variable** substitution syntax
- **Validation rules** for consistency

**Total Configuration Options:** 60+ fields

**Key Configuration Areas:**
- LLM settings (provider, model, rate limits)
- Embedding settings (provider, model, dimension)
- Chunking strategy (size, overlap, boundaries)
- Extraction settings (entity types, confidence threshold)
- Strategy-specific tuning
- Storage backends
- Query behavior
- Performance tuning
- Logging

**Next Steps:**
- Implement Settings schema in `config/settings.py`
- Add validation logic in `config/validation.py`
- Create profile templates in `config/profiles/`
- Build config generator for `init` command
