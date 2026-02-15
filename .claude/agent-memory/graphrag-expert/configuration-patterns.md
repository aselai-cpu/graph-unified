# GraphRAG Configuration Patterns & Best Practices

## Configuration File Structure

GraphRAG uses YAML or JSON configuration files. Default location: `./settings.yaml`

### Minimal Configuration
```yaml
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-4-turbo-preview

embeddings:
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small

input:
  type: file
  base_dir: "./input"
  file_pattern: ".*\\.txt$"

storage:
  type: file
  base_dir: "./output"
```

## Key Configuration Sections

### 1. Chunking Configuration

```yaml
chunks:
  size: 800                # Tokens per chunk (300-1200 typical)
  overlap: 100             # Token overlap between chunks
  group_by_columns:        # Maintain boundaries
    - document_id
  encoding_model: cl100k_base  # tiktoken model for tokenization
```

**Tuning Guidelines:**
- **Small chunks (300-500)**: Better entity precision, more chunks to process
- **Medium chunks (800-1000)**: Balanced approach, recommended for most use cases
- **Large chunks (1200+)**: More context, risk of entity fragmentation

**Overlap:** 10-15% of chunk size prevents entity boundary issues

### 2. Entity Extraction Configuration

```yaml
entity_extraction:
  prompt: "./prompts/entity_extraction.txt"  # Custom prompt path
  max_gleanings: 2         # Refinement passes (0-3)
  entity_types:            # Domain-specific types
    - organization
    - person
    - location
    - event
    - technology
  strategy:
    type: graph_intelligence
```

**max_gleanings Tradeoff:**
- `0`: Fast, may miss entities
- `1`: Standard, good balance (default)
- `2`: Better recall, higher cost
- `3`: Maximum recall, expensive

**Entity Types:**
Customize for your domain. Examples:
- Legal: organization, person, statute, case, court
- Medical: disease, drug, symptom, procedure, patient
- Financial: company, security, transaction, regulation, executive

### 3. Graph Construction Configuration

```yaml
graph:
  max_cluster_size: 20     # Max entities per community
  use_leiden: true         # Leiden algorithm for community detection
  leiden_config:
    max_cluster_size: 20
    seed: 42               # Reproducibility
```

**max_cluster_size:**
- Small (5-10): Fine-grained communities, more reports to process
- Medium (10-30): Balanced granularity (recommended)
- Large (50+): Coarse communities, may lose specificity

### 4. Community Reports Configuration

```yaml
community_reports:
  prompt: "./prompts/community_report.txt"
  max_length: 2000         # Tokens per community summary
  max_input_length: 8000   # Context window for report generation
```

**max_length Impact:**
- Directly affects global search context size
- Larger = more detail but slower global search
- Recommended: 1500-3000 tokens

### 5. Embedding Configuration

```yaml
embeddings:
  llm:
    type: openai_embedding
    model: text-embedding-3-small  # or text-embedding-3-large
    api_base: https://api.openai.com/v1
  vector_store:
    type: lance              # Options: lance, lancedb
  strategy:
    type: openai
```

**Model Selection:**
- `text-embedding-3-small`: Fast, cost-effective, 1536 dimensions
- `text-embedding-3-large`: Higher quality, more expensive, 3072 dimensions
- `text-embedding-ada-002`: Legacy, 1536 dimensions

### 6. Storage Configuration

```yaml
storage:
  type: file
  base_dir: "./output"

# Or Azure Blob Storage
storage:
  type: blob
  connection_string: ${AZURE_STORAGE_CONNECTION_STRING}
  container_name: graphrag-output
  storage_account_name: myaccount
```

### 7. LLM Configuration

```yaml
llm:
  type: openai_chat
  model: gpt-4-turbo-preview
  api_key: ${GRAPHRAG_API_KEY}
  max_tokens: 4000
  temperature: 0.0         # Deterministic for extraction
  top_p: 1.0
  request_timeout: 60.0
  max_retries: 3
  retry_delay: 2.0

# Azure OpenAI
llm:
  type: azure_openai_chat
  api_base: https://YOUR_RESOURCE.openai.azure.com
  api_version: "2024-02-15-preview"
  deployment_name: gpt-4-turbo
  api_key: ${AZURE_OPENAI_API_KEY}
```

## Prompt Tuning Workflow

### Auto-generate Domain-Specific Prompts

```bash
graphrag prompt-tune \
  --root ./data \
  --output ./prompts \
  --n-subsample-max 20 \
  --model gpt-4-turbo-preview
```

**Parameters:**
- `--root`: Input data directory
- `--output`: Where to save generated prompts
- `--n-subsample-max`: Number of sample documents to analyze (5-50)
- `--language`: Target language for prompts
- `--model`: LLM to use for prompt generation

**Generated Prompts:**
- `entity_extraction.txt`: Customized entity types and examples
- `summarize_descriptions.txt`: Entity description summarization
- `community_report.txt`: Community summary generation

### Manual Prompt Customization

After auto-generation, refine prompts:

1. **Entity Extraction Prompt:**
   - Add domain-specific entity examples
   - Clarify entity type definitions
   - Add negative examples (what NOT to extract)

2. **Community Report Prompt:**
   - Adjust summary structure
   - Emphasize key aspects for your use case
   - Add report format guidelines

### Prompt Tuning Best Practices

1. Start with auto-generated prompts (use representative documents)
2. Run indexing on small dataset (~100 documents)
3. Review extracted entities and community reports
4. Refine prompts based on quality issues
5. Re-index and validate improvements
6. Iterate until quality is acceptable

## Cost Optimization Strategies

### 1. Control Token Usage

```yaml
# Reduce max_gleanings
entity_extraction:
  max_gleanings: 1          # Instead of 2-3

# Reduce community report length
community_reports:
  max_length: 1500          # Instead of 2000-3000

# Use smaller chunks
chunks:
  size: 500                 # Instead of 800-1200
```

### 2. Use Cost-Effective Models

```yaml
# For entity extraction (needs reasoning)
llm:
  model: gpt-4o-mini        # Much cheaper than gpt-4-turbo

# For embeddings
embeddings:
  llm:
    model: text-embedding-3-small  # vs text-embedding-3-large
```

### 3. Incremental Updates

For evolving datasets:
- Use GraphRAG's incremental indexing (when available)
- Process only new/changed documents
- Merge with existing graph

### 4. Batch Processing

```yaml
llm:
  concurrent_requests: 10   # Parallel LLM calls (balance speed vs. rate limits)
```

## Performance Tuning

### Indexing Performance

```yaml
# Increase parallelism
async:
  max_concurrency: 10       # Parallel operations
  thread_count: 4           # Worker threads

# Optimize storage I/O
storage:
  type: file
  base_dir: "/fast-ssd/output"  # Use fast storage
```

### Query Performance

**Local Search:**
```python
local_search = LocalSearch(
    context_builder=...,
    token_encoder=...,
    max_tokens=4000,          # Response context size
    conversation_history_max_turns=3,
    top_k_entities=20,        # Reduce for faster search
    top_k_relationships=10
)
```

**Global Search:**
```python
global_search = GlobalSearch(
    context_builder=...,
    token_encoder=...,
    max_tokens=8000,          # Larger for comprehensive summaries
    map_max_tokens=2000,      # Per-community context
    reduce_max_tokens=4000,   # Final synthesis context
    concurrency=4             # Parallel community processing
)
```

## Domain-Specific Configuration Examples

### Legal Documents
```yaml
chunks:
  size: 1000                # Longer for complex legal language
  overlap: 150

entity_extraction:
  max_gleanings: 2          # Higher accuracy needed
  entity_types:
    - case
    - statute
    - legal_principle
    - court
    - judge
    - party

community_reports:
  max_length: 2500          # Detailed legal summaries
```

### Medical Research
```yaml
chunks:
  size: 800
  overlap: 100

entity_extraction:
  max_gleanings: 2
  entity_types:
    - disease
    - drug
    - symptom
    - procedure
    - protein
    - gene
    - clinical_trial

llm:
  model: gpt-4-turbo-preview  # Better for technical/medical content
  temperature: 0.0            # Deterministic
```

### Business Intelligence
```yaml
chunks:
  size: 600                 # Shorter for metrics and figures
  overlap: 80

entity_extraction:
  max_gleanings: 1
  entity_types:
    - organization
    - person
    - product
    - technology
    - metric
    - event

community_reports:
  max_length: 1500          # Concise business summaries
```

## Environment Variables

```bash
# API Keys
export GRAPHRAG_API_KEY="your-openai-key"
export AZURE_OPENAI_API_KEY="your-azure-key"

# Storage
export AZURE_STORAGE_CONNECTION_STRING="your-connection-string"

# Override config values
export GRAPHRAG_LLM_MODEL="gpt-4o-mini"
export GRAPHRAG_EMBEDDING_MODEL="text-embedding-3-small"

# Logging
export GRAPHRAG_LOG_LEVEL="DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

## Configuration Validation

### Test Configuration
```bash
# Validate config without running full indexing
graphrag index --root ./data --config ./settings.yaml --dry-run
```

### Check Outputs
After indexing, validate:
```python
import pandas as pd

# Check entity counts
entities = pd.read_parquet("./output/latest/artifacts/entities.parquet")
print(f"Entities: {len(entities)}")
print(f"Entity types: {entities['type'].value_counts()}")

# Check community structure
communities = pd.read_parquet("./output/latest/artifacts/communities.parquet")
print(f"Communities: {len(communities)}")
print(f"Hierarchy levels: {communities['level'].value_counts()}")

# Check relationships
relationships = pd.read_parquet("./output/latest/artifacts/relationships.parquet")
print(f"Relationships: {len(relationships)}")
```

## Common Configuration Mistakes

1. **Chunk size too small:** Entities split across chunks, poor graph quality
2. **max_gleanings too high:** Excessive cost with diminishing returns
3. **Entity types too generic:** Extract too many low-value entities
4. **Community max_cluster_size wrong:** Too small = fragmented, too large = coarse
5. **Temperature > 0 for extraction:** Non-deterministic entity extraction
6. **Insufficient overlap:** Entities at chunk boundaries missed
7. **Wrong embedding model:** Dimension mismatch with vector store

## Version-Specific Notes

### GraphRAG 0.3.x+
- Parquet format standard (not CSV)
- Leiden algorithm default
- Improved prompt tuning with auto-detection
- Support for Azure Blob Storage

### Breaking Changes to Watch
- Configuration schema changes between versions
- LLM provider API updates (OpenAI, Azure)
- Storage format migrations
- Embedding model deprecations
