# Configuration Reference

Complete reference for `settings.yaml` configuration file.

## Overview

Graph-Unified uses a YAML configuration file (default: `settings.yaml`) to control all aspects of indexing and querying. This document describes every configuration option.

**Generate default configuration:**
```bash
graph-unified init
```

**Use custom configuration:**
```bash
graph-unified index --config custom_settings.yaml --input ./docs
```

---

## Configuration Schema

### Top-Level Structure

```yaml
version: "1.0"              # Configuration version (required)
llm: {...}                  # LLM provider configuration
embedding: {...}            # Embedding provider configuration
chunking: {...}             # Document chunking settings
extraction: {...}           # Entity/relationship extraction settings
strategies: {...}           # Strategy-specific configurations
storage: {...}              # Storage backend settings
query: {...}                # Query-time settings
performance: {...}          # Performance tuning
logging: {...}              # Logging configuration
prompts: {...}              # Custom prompt templates (optional)
```

---

## LLM Configuration

Controls the language model used for extraction and generation.

```yaml
llm:
  provider: "anthropic"                     # Provider: anthropic|openai|azure
  model: "claude-3-5-sonnet-20241022"      # Model identifier
  api_key: "${ANTHROPIC_API_KEY}"          # API key (use env var)
  temperature: 0.0                         # Sampling temperature (0.0-1.0)
  max_tokens: 4096                         # Maximum output tokens
  timeout: 60                              # Request timeout (seconds)
  rate_limit:
    requests_per_minute: 50                # Max requests per minute
    tokens_per_minute: 40000               # Max tokens per minute
  retry:
    max_attempts: 3                        # Max retry attempts
    backoff_factor: 2.0                    # Exponential backoff multiplier
```

### Field Descriptions

**`provider`** (required)
- Type: string
- Options: `anthropic`, `openai`, `azure`
- Default: `anthropic`
- Description: LLM provider. Currently only Anthropic (Claude) is fully supported.

**`model`** (required)
- Type: string
- Default: `claude-3-5-sonnet-20241022`
- Options:
  - `claude-3-5-sonnet-20241022` - Balanced quality and cost (recommended)
  - `claude-3-haiku-20240307` - Fastest, lowest cost, lower quality
  - `claude-opus-4-6` - Highest quality, highest cost
- Description: Claude model to use for extraction and generation.

**`api_key`** (required)
- Type: string
- Default: `${ANTHROPIC_API_KEY}`
- Description: API key for provider. Use environment variable format `${VAR_NAME}` to avoid hardcoding.

**`temperature`** (optional)
- Type: float
- Range: 0.0 - 1.0
- Default: 0.0
- Description: Sampling temperature. 0.0 = deterministic, 1.0 = maximum randomness. Use 0.0 for extraction (consistency).

**`max_tokens`** (optional)
- Type: integer
- Default: 4096
- Description: Maximum tokens to generate per request. Extraction typically needs 200-500, generation needs 500-2000.

**`timeout`** (optional)
- Type: integer
- Default: 60
- Description: Request timeout in seconds.

**`rate_limit`** (optional)
- Description: Rate limiting to avoid API errors.
- **`requests_per_minute`**: Max API calls per minute (default: 50)
- **`tokens_per_minute`**: Max tokens per minute (default: 40000)

**`retry`** (optional)
- Description: Retry logic for failed requests.
- **`max_attempts`**: Number of retries (default: 3)
- **`backoff_factor`**: Exponential backoff multiplier (default: 2.0)

---

## Embedding Configuration

Controls the embedding model used for vector search.

```yaml
embedding:
  provider: "voyageai"                     # Provider: voyageai|openai|cohere
  model: "voyage-3"                        # Model identifier
  dimension: 1024                          # Embedding dimension
  api_key: "${VOYAGE_API_KEY}"            # API key (use env var)
  batch_size: 128                          # Embeddings per batch
  normalize: true                          # L2 normalize embeddings
  timeout: 30                              # Request timeout (seconds)
```

### Field Descriptions

**`provider`** (required)
- Type: string
- Options: `voyageai`, `openai`, `cohere`
- Default: `voyageai`
- Description: Embedding provider. Voyage AI recommended for quality and cost.

**`model`** (required)
- Type: string
- Default: `voyage-3`
- Options (Voyage):
  - `voyage-3` - Latest, 1024-dim (recommended)
  - `voyage-large-2` - 1536-dim, higher quality
- Options (OpenAI):
  - `text-embedding-3-small` - 512/1536-dim, lower cost
  - `text-embedding-3-large` - 1024/3072-dim, higher quality
- Description: Embedding model to use.

**`dimension`** (required)
- Type: integer
- Default: 1024
- Description: Embedding dimension. Higher = more expressive, more storage. Use 1024 for most use cases.

**`api_key`** (required)
- Type: string
- Default: `${VOYAGE_API_KEY}`
- Description: API key for embedding provider.

**`batch_size`** (optional)
- Type: integer
- Default: 128
- Description: Number of texts to embed per API call. Larger = faster but more memory.

**`normalize`** (optional)
- Type: boolean
- Default: true
- Description: L2 normalize embeddings for cosine similarity. Keep true.

---

## Chunking Configuration

Controls how documents are split into chunks.

```yaml
chunking:
  strategy: "token_overlap"                # Strategy: token_overlap|sentence|semantic
  chunk_size: 512                          # Chunk size (tokens)
  overlap: 128                             # Overlap size (tokens)
  min_chunk_size: 100                      # Minimum chunk size (tokens)
  respect_boundaries: true                 # Respect sentence/paragraph boundaries
  separator: "\n\n"                        # Paragraph separator
```

### Field Descriptions

**`strategy`** (required)
- Type: string
- Options: `token_overlap`, `sentence`, `semantic`
- Default: `token_overlap`
- Description:
  - `token_overlap` - Fixed token windows with overlap (recommended)
  - `sentence` - Split on sentence boundaries
  - `semantic` - Semantic similarity-based splitting (experimental)

**`chunk_size`** (required)
- Type: integer
- Default: 512
- Range: 100 - 2048
- Description: Target chunk size in tokens. 512 = ~2 paragraphs, good balance.

**`overlap`** (required for token_overlap)
- Type: integer
- Default: 128
- Description: Overlap between adjacent chunks (tokens). Preserves context across boundaries.

**`min_chunk_size`** (optional)
- Type: integer
- Default: 100
- Description: Minimum chunk size. Smaller chunks discarded or merged.

**`respect_boundaries`** (optional)
- Type: boolean
- Default: true
- Description: Try to split on sentence/paragraph boundaries when possible (cleaner chunks).

---

## Extraction Configuration

Controls entity and relationship extraction.

```yaml
extraction:
  entity_types:                            # Entity types to extract
    - PERSON
    - ORGANIZATION
    - LOCATION
    - CONCEPT
    - EVENT
    - PRODUCT
  relationship_types:                      # Relationship types to extract
    - RELATED_TO
    - PART_OF
    - LOCATED_IN
    - WORKS_FOR
    - CAUSED_BY
    - CREATED_BY
  max_gleanings: 1                         # Additional extraction passes
  tuple_delimiter: "<|>"                   # Delimiter for output tuples
  record_delimiter: "##"                   # Delimiter between records
  extraction_prompt: null                  # Custom extraction prompt (path or inline)
  include_descriptions: true               # Extract entity descriptions
  min_confidence: 0.3                      # Minimum confidence score (0.0-1.0)
```

### Field Descriptions

**`entity_types`** (required)
- Type: list of strings
- Default: `[PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, PRODUCT]`
- Description: Entity types to extract. Customize for your domain (see [Entity Types Reference](entity-types.md)).

**`relationship_types`** (required)
- Type: list of strings
- Default: `[RELATED_TO, PART_OF, LOCATED_IN, WORKS_FOR, CAUSED_BY, CREATED_BY]`
- Description: Relationship types to extract. Customize for your domain.

**`max_gleanings`** (optional)
- Type: integer
- Default: 1
- Range: 0 - 3
- Description: Additional extraction passes to catch missed entities. 0 = single pass, 1 = two passes (recommended), 2+ = diminishing returns.

**`tuple_delimiter`** (optional)
- Type: string
- Default: `<|>`
- Description: Delimiter for output tuples (e.g., `entity<|>type<|>description`). Don't change unless needed.

**`record_delimiter`** (optional)
- Type: string
- Default: `##`
- Description: Delimiter between records. Don't change unless needed.

**`extraction_prompt`** (optional)
- Type: string (path) or null
- Default: null (use built-in prompt)
- Description: Path to custom extraction prompt YAML file. See [Prompt Tuning](../how-to/01-prompt-tuning.md).

**`include_descriptions`** (optional)
- Type: boolean
- Default: true
- Description: Extract descriptions for entities (required for most strategies).

**`min_confidence`** (optional)
- Type: float
- Range: 0.0 - 1.0
- Default: 0.3
- Description: Minimum confidence score for entities/relationships. Lower = more entities, more noise.

---

## Strategy-Specific Configuration

### GraphRAG Configuration

```yaml
strategies:
  graphrag:
    enabled: true                          # Enable GraphRAG strategies
    community_detection:
      algorithm: "leiden"                  # Algorithm: leiden|louvain
      max_level: 3                         # Maximum hierarchy levels
      resolution: 1.0                      # Resolution parameter
      seed: 42                             # Random seed (reproducibility)
    summarization:
      enabled: true                        # Generate community summaries
      max_tokens: 2000                     # Max tokens per summary
      temperature: 0.1                     # LLM temperature
      summary_prompt: null                 # Custom summary prompt
    local_search:
      max_hops: 2                          # Max graph traversal hops
      top_k_entities: 10                   # Top entities to retrieve
      top_k_relationships: 20              # Top relationships to retrieve
    global_search:
      map_max_tokens: 1000                 # Max tokens per map operation
      reduce_max_tokens: 2000              # Max tokens for reduce
      top_k_communities: 10                # Top communities to retrieve
```

### LightRAG Configuration

```yaml
strategies:
  lightrag:
    enabled: true                          # Enable LightRAG
    relationship_description:
      max_length: 200                      # Max tokens per description
      temperature: 0.2                     # LLM temperature
      description_prompt: null             # Custom description prompt
    search_modes:
      local_top_k: 20                      # Top-k for local (entity) search
      global_top_k: 20                     # Top-k for global (relation) search
      hybrid_weights: [0.5, 0.5]          # [entity_weight, relation_weight]
```

### HippoRAG Configuration

```yaml
strategies:
  hipporag:
    enabled: true                          # Enable HippoRAG
    fact_extraction:
      enabled: true                        # Extract facts (atomic statements)
      max_facts_per_chunk: 5              # Max facts per chunk
      fact_prompt: null                    # Custom fact extraction prompt
    ppr_config:
      damping: 0.85                        # PageRank damping factor
      max_iterations: 100                  # Max PPR iterations
      tolerance: 1e-6                      # Convergence tolerance
    retrieval:
      top_k_entities: 10                   # Top entities for activation
      top_k_passages: 10                   # Top passages to retrieve
      activation_threshold: 0.01           # Min activation score
```

### Hybrid RAG Configuration

```yaml
strategies:
  hybrid:
    enabled: true                          # Enable Hybrid RAG
    dense_weight: 0.7                      # Dense (vector) weight
    sparse_weight: 0.3                     # Sparse (BM25) weight
    bm25_config:
      k1: 1.5                              # BM25 term saturation
      b: 0.75                              # BM25 length normalization
```

### Naive RAG Configuration

```yaml
strategies:
  naive:
    enabled: true                          # Enable Naive RAG
    # No additional config (uses vector search only)
```

---

## Storage Configuration

Controls where and how data is stored.

```yaml
storage:
  root_dir: "./output"                     # Root output directory
  parquet:
    compression: "snappy"                  # Compression: snappy|gzip|none
    row_group_size: 10000                  # Rows per row group
  vector_store:
    backend: "lancedb"                     # Backend: lancedb|faiss
    path: "./output/lancedb"              # Vector store path
    metric: "cosine"                       # Distance metric: cosine|l2|ip
    index_type: "IVF_FLAT"                # Index type (FAISS only)
  graph_store:
    backend: "networkx"                    # Backend: networkx|igraph
    path: "./output/graphs"               # Graph store path
    compression: true                      # Compress graph files
```

### Field Descriptions

**`root_dir`** (required)
- Type: string
- Default: `./output`
- Description: Root directory for all output files.

**`parquet.compression`** (optional)
- Type: string
- Options: `snappy`, `gzip`, `none`
- Default: `snappy`
- Description: Parquet compression codec. Snappy = good balance (speed + size).

**`vector_store.backend`** (required)
- Type: string
- Options: `lancedb`, `faiss`
- Default: `lancedb`
- Description: Vector store backend. LanceDB recommended (disk-based, scalable).

**`vector_store.metric`** (optional)
- Type: string
- Options: `cosine`, `l2`, `ip` (inner product)
- Default: `cosine`
- Description: Distance metric for vector search. Use cosine with normalized embeddings.

**`graph_store.backend`** (required)
- Type: string
- Options: `networkx`, `igraph`
- Default: `networkx`
- Description: Graph library. NetworkX = pure Python, igraph = faster for large graphs.

---

## Query Configuration

Controls query-time behavior.

```yaml
query:
  default_strategy: "hybrid"               # Default retrieval strategy
  top_k: 10                                # Number of chunks to retrieve
  generation:
    model: "claude-3-5-sonnet-20241022"   # Model for generation (can differ from extraction)
    max_tokens: 2048                       # Max tokens for answer
    temperature: 0.3                       # Sampling temperature
  context_window: 8000                     # Max context tokens for generation
  router:
    enabled: false                         # Enable automatic query routing
    model_path: null                       # Path to routing model (if trained)
```

### Field Descriptions

**`default_strategy`** (required)
- Type: string
- Options: `naive`, `hybrid`, `graphrag-local`, `graphrag-global`, `lightrag`, `hipporag`
- Default: `hybrid`
- Description: Default strategy if not specified in query command.

**`top_k`** (optional)
- Type: integer
- Default: 10
- Description: Number of chunks to retrieve. More = better recall, higher cost.

**`generation.model`** (optional)
- Type: string
- Default: Same as `llm.model`
- Description: Model for answer generation. Can differ from extraction model (e.g., use Haiku for generation to save cost).

**`context_window`** (optional)
- Type: integer
- Default: 8000
- Description: Maximum context tokens to send to generator. Truncates if retrieved context exceeds.

**`router.enabled`** (optional)
- Type: boolean
- Default: false
- Description: Enable automatic query routing (select strategy based on query type). Experimental.

---

## Performance Configuration

Controls performance and resource usage.

```yaml
performance:
  indexing:
    max_workers: 4                         # Parallel workers for indexing
    batch_size: 100                        # Documents per batch
    checkpoint_interval: 1000              # Save checkpoint every N docs
  query:
    cache_enabled: true                    # Enable result caching
    cache_ttl: 3600                        # Cache TTL (seconds)
    cache_backend: "memory"                # Cache backend: memory|redis
```

### Field Descriptions

**`indexing.max_workers`** (optional)
- Type: integer
- Default: 4
- Description: Parallel workers for indexing. Higher = faster but more API rate limit risk. Set to 1 if hitting rate limits.

**`indexing.batch_size`** (optional)
- Type: integer
- Default: 100
- Description: Documents to process per batch. Larger = fewer checkpoints, more memory.

**`indexing.checkpoint_interval`** (optional)
- Type: integer
- Default: 1000
- Description: Save progress every N documents. Enables resuming interrupted indexing.

**`query.cache_enabled`** (optional)
- Type: boolean
- Default: true
- Description: Enable query result caching. Saves cost for repeated queries.

**`query.cache_ttl`** (optional)
- Type: integer
- Default: 3600 (1 hour)
- Description: Cache time-to-live in seconds.

---

## Logging Configuration

Controls logging behavior.

```yaml
logging:
  level: "INFO"                            # Log level: DEBUG|INFO|WARNING|ERROR
  format: "json"                           # Format: json|text
  output: "stdout"                         # Output: stdout|stderr|file_path
  file:
    path: "./logs/graph-unified.log"      # Log file path (if output=file_path)
    max_bytes: 10485760                    # Max log file size (10 MB)
    backup_count: 5                        # Number of backup files
```

### Field Descriptions

**`level`** (optional)
- Type: string
- Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- Default: `INFO`
- Description: Minimum log level. Use DEBUG for troubleshooting.

**`format`** (optional)
- Type: string
- Options: `json`, `text`
- Default: `json`
- Description: Log format. JSON = structured (easier parsing), text = human-readable.

---

## Prompts Configuration (Optional)

Custom prompt templates for extraction.

```yaml
prompts:
  extraction_template: "prompts/custom_extraction.yaml"
  summary_template: "prompts/custom_summary.yaml"
```

**See:** [Prompt Tuning Guide](../how-to/01-prompt-tuning.md)

---

## Example Configurations

### Minimal Configuration (Development)

```yaml
version: "1.0"

llm:
  provider: "anthropic"
  model: "claude-3-haiku-20240307"         # Cheapest model
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.0

embedding:
  provider: "voyageai"
  model: "voyage-3"
  dimension: 1024
  api_key: "${VOYAGE_API_KEY}"

chunking:
  strategy: "token_overlap"
  chunk_size: 512
  overlap: 128

extraction:
  entity_types: [PERSON, ORGANIZATION, CONCEPT]
  relationship_types: [RELATED_TO]

storage:
  root_dir: "./output"

query:
  default_strategy: "naive"
  top_k: 5
```

**Use case:** Quick prototyping, small corpora.

---

### Balanced Configuration (Production)

```yaml
version: "1.0"

llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"      # Balanced quality/cost
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.0
  rate_limit:
    requests_per_minute: 50

embedding:
  provider: "voyageai"
  model: "voyage-3"
  dimension: 1024
  api_key: "${VOYAGE_API_KEY}"
  batch_size: 128

chunking:
  strategy: "token_overlap"
  chunk_size: 512
  overlap: 128
  respect_boundaries: true

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
  max_gleanings: 1

strategies:
  graphrag:
    enabled: true
    summarization:
      enabled: true
      max_tokens: 1000
  lightrag:
    enabled: true
  hipporag:
    enabled: true
  hybrid:
    enabled: true

storage:
  root_dir: "./output"
  vector_store:
    backend: "lancedb"
  graph_store:
    backend: "networkx"

query:
  default_strategy: "hybrid"
  top_k: 10
  cache_enabled: true

performance:
  indexing:
    max_workers: 4
    checkpoint_interval: 1000

logging:
  level: "INFO"
  format: "json"
```

**Use case:** Production deployment, balanced quality and cost.

---

### High-Quality Configuration (Research)

```yaml
version: "1.0"

llm:
  provider: "anthropic"
  model: "claude-opus-4-6"                 # Highest quality
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.0

embedding:
  provider: "voyageai"
  model: "voyage-large-2"                  # Highest quality embeddings
  dimension: 1536
  api_key: "${VOYAGE_API_KEY}"

chunking:
  strategy: "semantic"                     # Semantic chunking
  chunk_size: 512
  overlap: 128

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
    - CREATED_BY
    - SIMILAR_TO
  max_gleanings: 2                         # Two additional passes

strategies:
  graphrag:
    enabled: true
    community_detection:
      max_level: 4                         # Deeper hierarchy
    summarization:
      max_tokens: 2000
      temperature: 0.1
  lightrag:
    enabled: true
  hipporag:
    enabled: true
  hybrid:
    enabled: true

query:
  default_strategy: "hybrid"
  top_k: 20
  generation:
    max_tokens: 4096
    temperature: 0.2

logging:
  level: "DEBUG"
```

**Use case:** Research, maximum quality, cost not a concern.

---

## Environment Variable Substitution

Configuration supports environment variable substitution:

```yaml
llm:
  api_key: "${ANTHROPIC_API_KEY}"          # Replaced at runtime

embedding:
  api_key: "${VOYAGE_API_KEY}"

storage:
  root_dir: "${OUTPUT_DIR:-./output}"      # Default: ./output
```

**Format:** `${VAR_NAME}` or `${VAR_NAME:-default_value}`

---

## Validation

Graph-Unified validates configuration on load:

```bash
graph-unified index --config invalid_settings.yaml --input ./docs
```

**Error output:**
```
Configuration Error: Invalid value for 'llm.temperature'
  Expected: float between 0.0 and 1.0
  Got: 2.5
  Location: llm.temperature
```

**Validate without running:**
```bash
graph-unified test-connection --config settings.yaml
```

---

## Domain Presets

Generate domain-specific configurations:

```bash
# Medical domain
graph-unified init --domain medical --output medical_settings.yaml

# Legal domain
graph-unified init --domain legal --output legal_settings.yaml

# Financial domain
graph-unified init --domain financial --output financial_settings.yaml
```

**Presets customize:**
- Entity types (domain-specific)
- Relationship types (domain-specific)
- Extraction prompts (domain-tuned)
- Default strategies (domain-optimal)

---

## See Also

- [CLI Reference](cli-reference.md) - Command-line interface
- [Prompt Tuning](../how-to/01-prompt-tuning.md) - Custom extraction prompts
- [Entity Types Reference](entity-types.md) - Entity type definitions
- [Performance Optimization](../how-to/05-performance-optimization.md) - Tuning for speed/cost

