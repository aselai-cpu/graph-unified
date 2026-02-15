# CLI Specification

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document specifies the complete command-line interface for Graph-Unified, including all commands, options, arguments, output formats, exit codes, and usage examples.

## Command Structure

```
graph-unified [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

## Global Options

Available for all commands:

```
--config PATH     Path to settings.yaml (default: ./settings.yaml)
--verbose, -v     Verbose output (DEBUG level logging)
--quiet, -q       Quiet mode (ERROR level only)
--version         Show version and exit
--help, -h        Show help message and exit
```

---

## Commands

### 1. init

**Purpose:** Generate default configuration file.

**Syntax:**
```bash
graph-unified init [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--output PATH` | Path | `./settings.yaml` | Output path for config file |
| `--profile PROFILE` | String | `balanced` | Configuration profile |
| `--overwrite` | Flag | False | Overwrite existing file |
| `--interactive` | Flag | False | Interactive configuration wizard |

**Profiles:**
- `minimal` - Minimal settings for testing
- `balanced` - Balanced for production (default)
- `research` - High-quality settings for research
- `dev` - Fast settings for development

**Examples:**

```bash
# Generate default config
graph-unified init

# Generate with specific profile
graph-unified init --profile research --output research_settings.yaml

# Interactive setup
graph-unified init --interactive

# Overwrite existing config
graph-unified init --overwrite
```

**Output:**

```
✓ Generated configuration: settings.yaml
  Profile: balanced
  Next steps:
    1. Set API keys in environment:
       export ANTHROPIC_API_KEY="your-key"
       export VOYAGE_API_KEY="your-key"
    2. Review and customize settings.yaml
    3. Run: graph-unified index --input ./documents
```

**Exit Codes:**
- `0` - Success
- `1` - File already exists (without --overwrite)
- `2` - Invalid profile name

---

### 2. index

**Purpose:** Index documents and build retrieval indexes.

**Syntax:**
```bash
graph-unified index [OPTIONS] --input PATH
```

**Options:**

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--input PATH` | Path | Yes | Input directory with documents |
| `--output PATH` | Path | No | Output directory (overrides config) |
| `--strategies LIST` | Comma-separated | No | Strategies to build (default: all enabled) |
| `--incremental` | Flag | No | Incremental indexing (skip unchanged) |
| `--resume` | Flag | No | Resume from checkpoint |
| `--dry-run` | Flag | No | Show plan without executing |
| `--progress` | Flag | No | Show progress bar (default: auto) |
| `--no-progress` | Flag | No | Disable progress output |

**Strategy Options:**

`--strategies` accepts comma-separated values:
- `naive` - Naive vector search
- `hybrid` - Hybrid dense+sparse
- `graphrag_local` - GraphRAG local search
- `graphrag_global` - GraphRAG global search
- `lightrag` - LightRAG dual-index
- `hipporag` - HippoRAG associative
- `all` - All strategies (default)

**Examples:**

```bash
# Index all documents with all strategies
graph-unified index --input ./documents

# Index with specific strategies
graph-unified index --input ./docs --strategies naive,hybrid

# Incremental indexing
graph-unified index --input ./docs --incremental

# Resume interrupted indexing
graph-unified index --input ./docs --resume

# Dry run to estimate cost
graph-unified index --input ./docs --dry-run
```

**Output:**

```
🔄 Indexing documents...

Stage: load          ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (1000/1000 docs)
Stage: chunk         ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (1000/1000 docs)
Stage: extract       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (250/250 batches)
Stage: embed_chunks  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (200/200 batches)
Stage: embed_entities━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (30/30 batches)
Stage: build_indexes ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% (6/6 strategies)

✓ Indexing complete!

  Documents:      1,000
  Chunks:         25,000
  Entities:       3,500
  Relationships:  8,200
  Communities:    45

  Duration:       26m 15s
  Cost:          $45.67

  Strategies built:
    ✓ naive
    ✓ hybrid
    ✓ graphrag_local
    ✓ graphrag_global
    ✓ lightrag
    ✓ hipporag

  Output: ./output
```

**Dry Run Output:**

```
📋 Indexing Plan (Dry Run)

Input:
  Path: ./documents
  Files: 1,000 documents
  Total size: 50 MB
  Estimated tokens: ~12.5M

Pipeline:
  1. Load documents (2s)
  2. Chunk (30s, ~25,000 chunks)
  3. Extract entities/relationships (20m, ~250 batches)
  4. Embed chunks (3m, ~200 batches)
  5. Embed entities (30s, ~30 batches)
  6. Build indexes (2m)

Estimated Cost:
  LLM (extraction):      $30.00 (1.5M tokens)
  LLM (summarization):   $10.00 (500K tokens)
  Embeddings:           $5.67 (28.5K calls)
  Total:                $45.67

Estimated Duration: ~26 minutes

Strategies:
  ✓ naive
  ✓ hybrid
  ✓ graphrag_local
  ✓ graphrag_global
  ✓ lightrag
  ✓ hipporag

To proceed: graph-unified index --input ./documents
```

**Exit Codes:**
- `0` - Success
- `1` - Input path not found
- `2` - Configuration error
- `3` - Indexing failed
- `4` - Interrupted (checkpoint saved)

---

### 3. query

**Purpose:** Query indexed knowledge base.

**Syntax:**
```bash
graph-unified query [OPTIONS] QUERY
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model STRATEGY` | String | `auto` | Retrieval strategy |
| `--top-k N` | Integer | `10` | Number of contexts to retrieve |
| `--no-generate` | Flag | False | Skip response generation (retrieval only) |
| `--explain` | Flag | False | Show retrieval details |
| `--output FORMAT` | String | `text` | Output format: text, json, markdown |

**Strategy Options:**
- `auto` - Auto-route (default)
- `naive` - Naive vector search
- `hybrid` - Hybrid retrieval
- `graphrag_local` - GraphRAG local
- `graphrag_global` - GraphRAG global
- `lightrag` - LightRAG
- `hipporag` - HippoRAG

**Examples:**

```bash
# Query with auto-routing
graph-unified query "What causes climate change?"

# Query with specific strategy
graph-unified query "What causes climate change?" --model hybrid

# Retrieval only (no generation)
graph-unified query "climate change" --no-generate --top-k 5

# Query with detailed explanation
graph-unified query "What causes climate change?" --explain

# JSON output
graph-unified query "climate change" --output json
```

**Output (Text Format):**

```
🔍 Query: What causes climate change?
📚 Strategy: hybrid (auto-selected)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Climate change is primarily caused by anthropogenic greenhouse gas
emissions, particularly CO₂ from fossil fuel combustion. Key factors
include:

1. Fossil fuel burning (coal, oil, natural gas)
2. Deforestation and land use changes
3. Industrial processes and agriculture
4. Feedback loops amplifying warming

The IPCC reports high confidence that human activities are the dominant
cause of observed warming since the mid-20th century.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Metadata:
  Strategy:      hybrid
  Contexts:      10
  Latency:       185ms
  Tokens:        450
  Cost:          $0.0023
```

**Output with --explain:**

```
🔍 Query: What causes climate change?
📚 Strategy: hybrid (auto-selected)

━━━━ Retrieval Details ━━━━

Retrieved Contexts (top 5):

[1] Score: 0.89 (Dense: 0.85, Sparse: 0.92)
    Source: chunk-abc123
    "Anthropogenic greenhouse gas emissions are the primary driver of
     climate change. CO₂ from fossil fuels accounts for..."

[2] Score: 0.87 (Dense: 0.90, Sparse: 0.84)
    Source: chunk-def456
    "The IPCC Sixth Assessment Report states with high confidence that
     human influence has warmed the atmosphere..."

[3] Score: 0.84 (Dense: 0.82, Sparse: 0.86)
    Source: chunk-ghi789
    "Deforestation contributes to climate change by reducing CO₂
     absorption capacity..."

━━━━ Response ━━━━

[Generated response here]

━━━━ Metadata ━━━━
  Strategy:      hybrid
  Alpha:         0.5 (dense weight)
  Contexts:      10 retrieved, 3 used for generation
  Latency:       185ms (retrieval: 85ms, generation: 100ms)
  Tokens:        450 (input: 350, output: 100)
  Cost:          $0.0023
```

**Output (JSON Format):**

```json
{
  "query": "What causes climate change?",
  "strategy": "hybrid",
  "response": "Climate change is primarily caused by...",
  "contexts": [
    {
      "text": "Anthropogenic greenhouse gas emissions...",
      "score": 0.89,
      "source": "chunk-abc123",
      "metadata": {"document_id": "doc-1", "chunk_index": 5}
    }
  ],
  "metadata": {
    "strategy": "hybrid",
    "latency_ms": 185,
    "tokens_used": 450,
    "cost_usd": 0.0023,
    "context_count": 10
  }
}
```

**Exit Codes:**
- `0` - Success
- `1` - Empty query
- `2` - Invalid strategy
- `3` - Index not found
- `4` - Query failed

---

### 4. prompt-tune

**Purpose:** Auto-generate domain-specific extraction prompts.

**Syntax:**
```bash
graph-unified prompt-tune [OPTIONS] --sample-docs PATH --eval-data PATH
```

**Options:**

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `--sample-docs PATH` | Path | Yes | Directory with sample documents |
| `--eval-data PATH` | Path | Yes | Ground truth JSON file |
| `--iterations N` | Integer | No | Tuning iterations (default: 3) |
| `--output PATH` | Path | No | Output path for tuned prompts |

**Examples:**

```bash
# Tune prompts with 3 iterations
graph-unified prompt-tune \
  --sample-docs ./corpus/sample \
  --eval-data ./eval/ground_truth.json \
  --iterations 3

# Tune and save to file
graph-unified prompt-tune \
  --sample-docs ./sample \
  --eval-data ./eval.json \
  --output ./prompts/tuned_extraction.yaml
```

**Output:**

```
🔧 Prompt Tuning

Sample documents: 50
Ground truth: 50 documents, 200 entities, 150 relationships

━━━━ Baseline Evaluation ━━━━
  Entity Precision:      0.75
  Entity Recall:         0.68
  Entity F1:             0.71
  Relationship F1:       0.65

━━━━ Iteration 1 ━━━━
  Analyzing errors...
  Generating improved prompt...
  Evaluating...
  Entity F1:             0.78 (+0.07)
  Relationship F1:       0.71 (+0.06)

━━━━ Iteration 2 ━━━━
  Analyzing errors...
  Generating improved prompt...
  Evaluating...
  Entity F1:             0.82 (+0.04)
  Relationship F1:       0.75 (+0.04)

━━━━ Iteration 3 ━━━━
  Analyzing errors...
  Generating improved prompt...
  Evaluating...
  Entity F1:             0.83 (+0.01)
  Relationship F1:       0.76 (+0.01)

✓ Prompt tuning complete!

  Improvement:
    Entity F1:           +0.12 (17% improvement)
    Relationship F1:     +0.11 (17% improvement)

  Tuned prompts saved to: ./prompts/tuned_extraction.yaml

  To use tuned prompts:
    1. Copy tuned prompts to settings.yaml:
       prompts:
         entity_extraction: [from tuned file]
    2. Re-index corpus:
       graph-unified index --input ./documents
```

**Exit Codes:**
- `0` - Success
- `1` - Sample docs not found
- `2` - Eval data invalid
- `3` - Tuning failed

---

### 5. evaluate

**Purpose:** Evaluate retrieval quality.

**Syntax:**
```bash
graph-unified evaluate [OPTIONS] --eval-data PATH
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--eval-data PATH` | Path | Required | Evaluation dataset JSON |
| `--strategies LIST` | Comma-separated | `all` | Strategies to evaluate |
| `--metrics LIST` | Comma-separated | `all` | Metrics to compute |
| `--output PATH` | Path | `./eval_results.json` | Output path |

**Metrics:**
- `precision@k` - Precision at k
- `recall@k` - Recall at k
- `f1@k` - F1 score at k
- `mrr` - Mean Reciprocal Rank
- `latency` - Query latency
- `cost` - Cost per query

**Examples:**

```bash
# Evaluate all strategies
graph-unified evaluate --eval-data ./eval/queries.json

# Evaluate specific strategies
graph-unified evaluate --eval-data ./eval/queries.json --strategies naive,hybrid

# Evaluate specific metrics
graph-unified evaluate --eval-data ./eval/queries.json --metrics precision@10,recall@10
```

**Output:**

```
📊 Evaluation Results

Dataset: 100 queries

━━━━ Naive ━━━━
  Precision@10:        0.72
  Recall@10:           0.65
  F1@10:               0.68
  MRR:                 0.58
  Avg Latency:         85ms
  Avg Cost:            $0.0015

━━━━ Hybrid ━━━━
  Precision@10:        0.78
  Recall@10:           0.70
  F1@10:               0.74
  MRR:                 0.62
  Avg Latency:         152ms
  Avg Cost:            $0.0020

━━━━ GraphRAG Local ━━━━
  Precision@10:        0.81
  Recall@10:           0.73
  F1@10:               0.77
  MRR:                 0.68
  Avg Latency:         320ms
  Avg Cost:            $0.0028

✓ Results saved to: ./eval_results.json
```

**Exit Codes:**
- `0` - Success
- `1` - Eval data not found
- `2` - Invalid format
- `3` - Evaluation failed

---

### 6. inspect

**Purpose:** Inspect indexed data.

**Syntax:**
```bash
graph-unified inspect [OPTIONS] COMMAND
```

**Subcommands:**

#### inspect stats

Show indexing statistics.

```bash
graph-unified inspect stats
```

**Output:**

```
📊 Index Statistics

Documents:       1,000
Chunks:          25,000
  Avg per doc:   25
  Avg tokens:    512

Entities:        3,500
  Top types:
    - ORGANIZATION:  1,200 (34%)
    - PERSON:         900 (26%)
    - CONCEPT:        700 (20%)
    - LOCATION:       500 (14%)

Relationships:   8,200
  Top types:
    - RELATED_TO:    3,500 (43%)
    - WORKS_FOR:     2,000 (24%)
    - LOCATED_IN:    1,500 (18%)

Communities:     45
  Avg size:        78 entities

Storage:
  Total size:      706 MB
  Parquet:         193 MB
  Vectors:         460 MB
  Graphs:          53 MB

Indexed:         2026-02-15 10:30:00
Duration:        26m 15s
Cost:            $45.67
```

#### inspect entities

List top entities.

```bash
graph-unified inspect entities [--top N] [--type TYPE]
```

**Output:**

```
🏢 Top Entities (by mention count)

1. IPCC (ORGANIZATION)
   Mentions: 45 chunks
   Description: Intergovernmental Panel on Climate Change

2. Climate Change (CONCEPT)
   Mentions: 38 chunks
   Description: Global warming phenomenon

3. Paris Agreement (EVENT)
   Mentions: 25 chunks
   Description: International climate accord
```

#### inspect graph

Show graph statistics.

```bash
graph-unified inspect graph
```

**Output:**

```
🕸️ Graph Statistics

Nodes:               3,500 entities
Edges:               8,200 relationships
Avg degree:          4.69
Density:             0.0013

Components:          5
  Largest:           3,450 nodes (98.6%)
  Isolated:          12 nodes (0.3%)

Communities:         45
  Avg size:          78 entities
  Largest:           250 entities
```

---

### 7. visualize

**Purpose:** Generate graph visualizations.

**Syntax:**
```bash
graph-unified visualize [OPTIONS] --output PATH
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--output PATH` | Path | Output HTML file |
| `--entity ENTITY` | String | Focus on entity neighborhood |
| `--max-nodes N` | Integer | Max nodes to display (default: 100) |
| `--layout LAYOUT` | String | Layout algorithm (default: force) |

**Examples:**

```bash
# Visualize full graph
graph-unified visualize --output graph.html

# Visualize entity neighborhood
graph-unified visualize --entity "IPCC" --output ipcc_network.html

# Large graph with specific layout
graph-unified visualize --max-nodes 500 --layout hierarchical --output large.html
```

**Exit Codes:**
- `0` - Success
- `1` - Index not found
- `3` - Visualization failed

---

### 8. delete

**Purpose:** Delete indexed data.

**Syntax:**
```bash
graph-unified delete [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--all` | Flag | Delete all indexed data |
| `--documents IDs` | Comma-separated | Delete specific documents |
| `--confirm` | Flag | Skip confirmation prompt |

**Examples:**

```bash
# Delete all data (with confirmation)
graph-unified delete --all

# Delete specific documents
graph-unified delete --documents doc-id-1,doc-id-2 --confirm
```

**Output:**

```
⚠️  WARNING: This will permanently delete indexed data.

  Documents to delete: ALL (1,000 documents)
  This cannot be undone.

  Proceed? [y/N]: y

🗑️  Deleting data...
✓ Deleted 1,000 documents
✓ Deleted all indexes
✓ Cleared storage

Output directory preserved: ./output (empty)
```

---

### 9. export

**Purpose:** Export indexed data.

**Syntax:**
```bash
graph-unified export [OPTIONS] --output PATH
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--format FORMAT` | String | Export format: json, csv, graphml |
| `--data-type TYPE` | String | Data to export: entities, relationships, all |
| `--output PATH` | Path | Output path |

**Examples:**

```bash
# Export entities to JSON
graph-unified export --data-type entities --format json --output entities.json

# Export graph to GraphML
graph-unified export --data-type graph --format graphml --output graph.graphml
```

---

### 10. migrate

**Purpose:** Migrate data from standalone tools.

**Syntax:**
```bash
graph-unified migrate [OPTIONS] --source PATH --tool TOOL
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--tool TOOL` | String | Source tool: graphrag, lightrag, hipporag |
| `--source PATH` | Path | Source data directory |

**Examples:**

```bash
# Migrate from MS GraphRAG
graph-unified migrate --tool graphrag --source ./graphrag_output

# Migrate from LightRAG
graph-unified migrate --tool lightrag --source ./lightrag_data
```

---

## Shell Completion

### Bash

```bash
# Install completion
graph-unified --install-completion bash

# Add to .bashrc
eval "$(graph-unified --show-completion bash)"
```

### Zsh

```bash
# Install completion
graph-unified --install-completion zsh

# Add to .zshrc
eval "$(graph-unified --show-completion zsh)"
```

---

## Exit Codes Summary

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (file not found, invalid argument) |
| 2 | Configuration error |
| 3 | Execution failed (indexing, query) |
| 4 | Interrupted (checkpoint saved) |
| 5 | Permission error |

---

## Summary

This specification defines:

- **10 CLI commands** with complete syntax
- **50+ command options** with types and defaults
- **Output formats** for all commands
- **Progress indicators** and status messages
- **Exit codes** for error handling
- **Shell completion** support

**Implementation:**
- Use Click framework for CLI
- Rich library for progress bars and formatting
- Typer for type hints and validation

**Next Steps:**
- Implement CLI in `cli.py` using Click
- Add progress indicators with Rich
- Create shell completion scripts
- Write CLI integration tests
