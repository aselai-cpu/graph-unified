# CLI Reference

Complete reference for Graph-Unified command-line interface.

## Installation

```bash
pip install graph-unified
```

## Global Options

Available for all commands:

```
--config PATH       Path to configuration file (default: ./settings.yaml)
--verbose          Enable verbose logging
--quiet            Suppress all output except errors
--version          Show version and exit
--help             Show help message and exit
```

---

## Commands

### `graph-unified init`

Generate default configuration file.

**Usage:**
```bash
graph-unified init [OPTIONS]
```

**Options:**
```
--output PATH      Output path for configuration (default: ./settings.yaml)
--domain TEXT      Domain preset (medical|legal|financial|generic)
--overwrite        Overwrite existing configuration
```

**Examples:**

Generate default configuration:
```bash
graph-unified init
```

Generate medical-domain configuration:
```bash
graph-unified init --domain medical --output medical_settings.yaml
```

**Output:**

Creates `settings.yaml` with default values:
```yaml
version: "1.0"
llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
...
```

---

### `graph-unified index`

Index documents to build retrieval indexes.

**Usage:**
```bash
graph-unified index [OPTIONS]
```

**Options:**
```
--input PATH            Input directory containing documents (required)
--output PATH           Output directory for indexes (default: ./output)
--config PATH           Configuration file path
--prompts PATH          Custom extraction prompts file
--batch-size INT        Documents per batch (default: 100)
--strategies LIST       Comma-separated strategies to build (default: all)
                       Options: naive,hybrid,graphrag-local,graphrag-global,lightrag,hipporag
--skip-extraction       Skip extraction, use existing data
--checkpoint-interval   Save checkpoint every N documents (default: 1000)
--resume               Resume from last checkpoint
```

**Examples:**

Index all documents:
```bash
graph-unified index --input ./documents --output ./my_index
```

Index with custom prompts:
```bash
graph-unified index \
  --input ./medical_docs \
  --prompts ./prompts/medical.yaml \
  --output ./medical_index
```

Build only specific strategies:
```bash
graph-unified index \
  --input ./docs \
  --strategies naive,hybrid,graphrag-local
```

Resume interrupted indexing:
```bash
graph-unified index --input ./docs --resume
```

**Output:**

Creates index directory structure:
```
output/
├── documents.parquet
├── chunks.parquet
├── entities.parquet
├── relationships.parquet
├── communities.parquet (if GraphRAG enabled)
├── community_reports.parquet (if GraphRAG Global enabled)
├── lancedb/
│   ├── chunks.lance
│   ├── entities.lance
│   └── relationships.lance (if LightRAG enabled)
└── graphs/
    ├── entity_graph.graphml
    └── hippo_graph.pickle (if HippoRAG enabled)
```

**Exit Codes:**
- `0`: Success
- `1`: General error
- `2`: Configuration error
- `3`: API error (rate limit, auth failure)
- `4`: Out of memory error

---

### `graph-unified query`

Query indexed corpus with retrieval strategies.

**Usage:**
```bash
graph-unified query [OPTIONS] QUERY_TEXT
```

**Options:**
```
--index PATH           Path to index directory (default: ./output)
--model TEXT           Retrieval strategy (required if not using default)
                      Options: naive|hybrid|graphrag-local|graphrag-global|lightrag|hipporag|all
--top-k INT           Number of chunks to retrieve (default: 10)
--output-format TEXT  Output format: text|json|markdown (default: text)
--show-context        Display retrieved contexts before answer
--show-costs          Display cost breakdown
--save PATH           Save results to file
```

**Examples:**

Query with default strategy (hybrid):
```bash
graph-unified query "What is Anthropic?"
```

Query with specific strategy:
```bash
graph-unified query --model graphrag-local "Tell me about Claude"
```

Query and show context:
```bash
graph-unified query \
  --model lightrag \
  --show-context \
  "How do transformers relate to attention?"
```

Compare all strategies:
```bash
graph-unified query --model all "What companies work on AI safety?"
```

Save results to file:
```bash
graph-unified query \
  --model hybrid \
  --output-format json \
  --save results.json \
  "What are the main themes?"
```

**Output (text format):**
```
Strategy: Hybrid RAG
Query: What is Anthropic?

Retrieved Contexts (Top 3):
1. [Score: 0.89] Anthropic is an AI safety company...
2. [Score: 0.76] The company focuses on developing...
3. [Score: 0.71] Anthropic's flagship product is Claude...

Generated Answer:
Anthropic is an AI safety company founded in 2021 by Dario
and Daniela Amodei. The company focuses on developing safe,
steerable AI systems and is known for creating Claude.

Time: 1.4s
Cost: $0.014
```

**Output (JSON format):**
```json
{
  "query": "What is Anthropic?",
  "strategy": "hybrid",
  "contexts": [
    {
      "chunk_id": "chunk_142",
      "text": "Anthropic is an AI safety company...",
      "score": 0.89,
      "entities": ["Anthropic"]
    }
  ],
  "answer": "Anthropic is an AI safety company...",
  "metadata": {
    "time_seconds": 1.4,
    "cost_usd": 0.014,
    "tokens_used": {
      "input": 2020,
      "output": 85
    }
  }
}
```

**Exit Codes:**
- `0`: Success
- `1`: General error
- `2`: Index not found
- `3`: API error
- `5`: Query parsing error

---

### `graph-unified prompt-tune`

Auto-generate domain-specific extraction prompts.

**Usage:**
```bash
graph-unified prompt-tune [OPTIONS]
```

**Options:**
```
--input PATH           Input corpus directory (required)
--domain TEXT          Domain name (e.g., "medical", "legal")
--output PATH          Output path for tuned prompts (required)
--sample-size INT      Number of documents to analyze (default: 100)
--entity-types LIST    Comma-separated entity types to extract
--relationship-types   Comma-separated relationship types
--num-examples INT     Examples per entity type (default: 3)
```

**Examples:**

Auto-tune for medical domain:
```bash
graph-unified prompt-tune \
  --input ./medical_docs \
  --domain medical \
  --output ./prompts/medical.yaml
```

Tune with custom entity types:
```bash
graph-unified prompt-tune \
  --input ./legal_docs \
  --domain legal \
  --entity-types "STATUTE,CASE,COURT,PERSON" \
  --output ./prompts/legal.yaml
```

**Output:**

Creates `medical.yaml`:
```yaml
# Auto-generated prompts for medical domain
entity_extraction_prompt: |
  You are an expert medical information extractor...

  # Entity Types
  - DRUG: Medications and pharmaceuticals
  - DISEASE: Medical conditions and disorders
  ...

relationship_extraction_prompt: |
  You are an expert medical relationship extractor...
```

**Exit Codes:**
- `0`: Success
- `1`: General error
- `2`: Insufficient corpus size
- `3`: API error

---

### `graph-unified evaluate`

Evaluate extraction or retrieval quality.

**Usage:**
```bash
graph-unified evaluate [OPTIONS]
```

**Options:**
```
--predictions PATH    Path to predictions (Parquet file or JSON)
--ground-truth PATH   Path to ground truth labels (JSON)
--output PATH         Output path for metrics (JSON)
--show-errors         Display detailed error analysis
--metrics LIST        Comma-separated metrics to compute
                     Options: precision,recall,f1,mrr,ndcg
```

**Examples:**

Evaluate entity extraction:
```bash
graph-unified evaluate \
  --predictions output/entities.parquet \
  --ground-truth eval/ground_truth_entities.json \
  --output metrics.json
```

Evaluate with error analysis:
```bash
graph-unified evaluate \
  --predictions output/entities.parquet \
  --ground-truth eval/ground_truth.json \
  --show-errors
```

**Output:**
```json
{
  "entity_metrics": {
    "precision": 0.84,
    "recall": 0.82,
    "f1": 0.83
  },
  "relationship_metrics": {
    "precision": 0.78,
    "recall": 0.72,
    "f1": 0.75
  },
  "errors": {
    "missed_entities": 18,
    "false_positive_entities": 12,
    "missed_relationships": 14
  }
}
```

**Exit Codes:**
- `0`: Success
- `1`: General error
- `2`: Ground truth file not found
- `6`: Prediction file format error

---

### `graph-unified inspect`

Inspect indexed data (entities, relationships, communities).

**Usage:**
```bash
graph-unified inspect [ENTITY_TYPE] [OPTIONS]
```

**Entity Types:**
```
entities              List extracted entities
relationships         List extracted relationships
communities           List GraphRAG communities (if available)
documents             List indexed documents
chunks                List document chunks
```

**Options:**
```
--index PATH          Path to index directory (default: ./output)
--limit INT           Maximum items to display (default: 20)
--filter TEXT         Filter by text (substring match)
--sort-by TEXT        Sort by field (name|mentions|connections)
--format TEXT         Output format: table|json|csv (default: table)
--output PATH         Save output to file
```

**Examples:**

List top entities:
```bash
graph-unified inspect entities --limit 10
```

Search for specific entity:
```bash
graph-unified inspect entities --filter "Anthropic"
```

List relationships sorted by strength:
```bash
graph-unified inspect relationships --sort-by weight --limit 20
```

Export entities to CSV:
```bash
graph-unified inspect entities \
  --format csv \
  --output entities.csv
```

**Output (table format):**
```
Extracted Entities
==================

┌────┬───────────────┬──────────────┬─────────┬─────────────┐
│ ID │ Name          │ Type         │ Mentions│ Connections │
├────┼───────────────┼──────────────┼─────────┼─────────────┤
│ 1  │ Anthropic     │ ORGANIZATION │ 8       │ 6           │
│ 2  │ Dario Amodei  │ PERSON       │ 3       │ 4           │
│ 3  │ Claude        │ PRODUCT      │ 5       │ 3           │
└────┴───────────────┴──────────────┴─────────┴─────────────┘
```

**Exit Codes:**
- `0`: Success
- `1`: General error
- `2`: Index not found
- `7`: Invalid entity type

---

### `graph-unified visualize`

Generate visualizations of entity graphs and communities.

**Usage:**
```bash
graph-unified visualize [OPTIONS]
```

**Options:**
```
--index PATH          Path to index directory (default: ./output)
--output PATH         Output file path (required, .png|.svg|.html)
--type TEXT           Visualization type (default: graph)
                     Options: graph|communities|relationships|heatmap
--max-entities INT    Maximum entities to include (default: 100)
--min-connections INT Minimum connections to include (default: 1)
--layout TEXT         Graph layout algorithm (default: force)
                     Options: force|circular|hierarchical|spring
--interactive         Generate interactive HTML visualization
```

**Examples:**

Generate entity graph:
```bash
graph-unified visualize --output entity_graph.png
```

Generate interactive community visualization:
```bash
graph-unified visualize \
  --type communities \
  --interactive \
  --output communities.html
```

Generate relationship heatmap:
```bash
graph-unified visualize \
  --type heatmap \
  --output relationship_heatmap.png
```

Filter to highly connected entities:
```bash
graph-unified visualize \
  --max-entities 50 \
  --min-connections 3 \
  --output core_graph.svg
```

**Exit Codes:**
- `0`: Success
- `1`: General error
- `2`: Index not found
- `8`: Visualization dependencies missing

---

### `graph-unified test-connection`

Test API connections (Claude, embedding provider).

**Usage:**
```bash
graph-unified test-connection [OPTIONS]
```

**Options:**
```
--config PATH         Configuration file path
--provider TEXT       Specific provider to test (anthropic|voyageai|openai)
```

**Examples:**

Test all connections:
```bash
graph-unified test-connection
```

Test specific provider:
```bash
graph-unified test-connection --provider anthropic
```

**Output:**
```
✓ Anthropic API: Connected (Claude Sonnet available)
✓ Voyage AI API: Connected (voyage-3 available)
All API connections successful.
```

**Exit Codes:**
- `0`: All connections successful
- `3`: API connection failed

---

### `graph-unified migrate`

Migrate from standalone tools (MS GraphRAG, LightRAG) to Graph-Unified.

**Usage:**
```bash
graph-unified migrate [OPTIONS]
```

**Options:**
```
--source TEXT         Source tool (graphrag|lightrag|hipporag)
--source-dir PATH     Source index directory (required)
--output PATH         Output directory for migrated index
--config PATH         Configuration file path
--verify              Verify migration integrity
```

**Examples:**

Migrate from MS GraphRAG:
```bash
graph-unified migrate \
  --source graphrag \
  --source-dir ./graphrag_output \
  --output ./unified_index
```

Migrate with verification:
```bash
graph-unified migrate \
  --source lightrag \
  --source-dir ./lightrag_index \
  --output ./unified_index \
  --verify
```

**Exit Codes:**
- `0`: Migration successful
- `1`: General error
- `2`: Source directory not found
- `9`: Migration verification failed

---

### `graph-unified update`

Incrementally update existing index with new/changed documents.

**Usage:**
```bash
graph-unified update [OPTIONS]
```

**Options:**
```
--input PATH          Input directory with new/changed documents
--index PATH          Existing index directory (default: ./output)
--config PATH         Configuration file path
--force-rebuild       Force full rebuild instead of incremental
--detect-changes      Auto-detect changed documents by hash
```

**Examples:**

Update with new documents:
```bash
graph-unified update --input ./new_docs --index ./output
```

Force full rebuild:
```bash
graph-unified update \
  --input ./all_docs \
  --index ./output \
  --force-rebuild
```

Auto-detect changes:
```bash
graph-unified update \
  --input ./docs \
  --detect-changes
```

**Exit Codes:**
- `0`: Update successful
- `1`: General error
- `2`: Index not found

---

## Environment Variables

Graph-Unified respects the following environment variables:

```bash
ANTHROPIC_API_KEY       # Anthropic API key (required)
VOYAGE_API_KEY          # Voyage AI API key (if using Voyage embeddings)
OPENAI_API_KEY          # OpenAI API key (if using OpenAI embeddings)
GRAPH_UNIFIED_CONFIG    # Default config file path
GRAPH_UNIFIED_LOG_LEVEL # Logging level (DEBUG|INFO|WARNING|ERROR)
```

---

## Configuration File

CLI commands use `settings.yaml` by default. Override with `--config`:

```bash
graph-unified index --config custom_settings.yaml --input ./docs
```

See [Configuration Reference](configuration-reference.md) for full schema.

---

## Exit Codes Summary

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | File/directory not found |
| 3 | API error (rate limit, authentication) |
| 4 | Out of memory |
| 5 | Query parsing error |
| 6 | Prediction file format error |
| 7 | Invalid entity type |
| 8 | Visualization dependencies missing |
| 9 | Migration verification failed |

---

## Shell Completion

Enable shell completion for bash/zsh:

**Bash:**
```bash
eval "$(_GRAPH_UNIFIED_COMPLETE=bash_source graph-unified)"
```

**Zsh:**
```zsh
eval "$(_GRAPH_UNIFIED_COMPLETE=zsh_source graph-unified)"
```

**Add to shell profile for persistence:**
```bash
echo 'eval "$(_GRAPH_UNIFIED_COMPLETE=bash_source graph-unified)"' >> ~/.bashrc
```

---

## Logging

Control logging verbosity:

```bash
# Verbose (DEBUG level)
graph-unified --verbose index --input ./docs

# Quiet (only errors)
graph-unified --quiet query "What is X?"

# Log to file
graph-unified index --input ./docs 2>&1 | tee indexing.log
```

---

## Common Workflows

### First-Time Setup

```bash
# 1. Initialize configuration
graph-unified init

# 2. Test API connections
graph-unified test-connection

# 3. Index documents
graph-unified index --input ./documents

# 4. Query
graph-unified query "What is X?"
```

### Production Deployment

```bash
# 1. Index with custom config
graph-unified index \
  --input ./production_docs \
  --config ./production_settings.yaml \
  --checkpoint-interval 500

# 2. Set up incremental updates (cron job)
0 2 * * * graph-unified update \
  --input /data/docs \
  --detect-changes

# 3. Query via API (programmatic)
# See API Reference for Python client usage
```

### Prompt Tuning Workflow

```bash
# 1. Auto-tune prompts
graph-unified prompt-tune \
  --input ./domain_docs \
  --domain medical \
  --output ./prompts/medical.yaml

# 2. Evaluate baseline
graph-unified index --input ./eval_docs --output ./baseline
graph-unified evaluate \
  --predictions ./baseline/entities.parquet \
  --ground-truth ./eval/ground_truth.json

# 3. Re-index with tuned prompts
graph-unified index \
  --input ./eval_docs \
  --prompts ./prompts/medical.yaml \
  --output ./tuned

# 4. Evaluate improvement
graph-unified evaluate \
  --predictions ./tuned/entities.parquet \
  --ground-truth ./eval/ground_truth.json
```

---

## Troubleshooting

### Command Not Found

**Error:** `graph-unified: command not found`

**Solution:**
```bash
# Ensure virtual environment activated
source .venv/bin/activate

# Reinstall in editable mode
pip install -e ".[dev]"
```

### API Authentication Errors

**Error:** `Anthropic API: Authentication failed`

**Solution:**
```bash
# Check API key set
echo $ANTHROPIC_API_KEY

# Set if missing
export ANTHROPIC_API_KEY="sk-ant-..."

# Verify connection
graph-unified test-connection
```

### Out of Memory

**Error:** `Out of memory during indexing`

**Solution:**
```bash
# Reduce batch size
graph-unified index --input ./docs --batch-size 50

# Or reduce chunk size in settings.yaml
chunking:
  chunk_size: 256  # Smaller chunks = less memory
```

### Rate Limiting

**Error:** `Rate limit exceeded`

**Solution:**
```bash
# Reduce parallel workers in settings.yaml
performance:
  indexing:
    max_workers: 1  # Sequential processing

# Or wait and resume
graph-unified index --input ./docs --resume
```

---

## See Also

- [Configuration Reference](configuration-reference.md) - Full settings.yaml schema
- [API Reference](api-reference.md) - Programmatic Python API
- [How-To Guides](../how-to/) - Task-specific guides
- [Tutorial](../tutorial/01-getting-started.md) - Step-by-step walkthrough

