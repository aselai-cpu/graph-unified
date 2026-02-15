# Getting Started with Graph-Unified

## Overview

This tutorial walks you through your first Graph-Unified experience: indexing a small corpus and querying it with different retrieval strategies.

**What you'll learn:**
- Installing Graph-Unified
- Setting up configuration
- Indexing your first documents
- Querying with multiple strategies
- Comparing results

**Time required:** 30 minutes

**Prerequisites:**
- Python 3.10 or higher
- Anthropic API key (Claude access)
- Voyage AI API key (or alternative embedding provider)
- 2 GB free disk space

## Step 1: Installation (5 minutes)

### Install Python Package

```bash
# Clone the repository
git clone https://github.com/your-org/graph-unified.git
cd graph-unified

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

**Expected output:**
```
Successfully installed graph-unified-0.1.0
  anthropic-0.18.1
  voyageai-0.2.1
  lancedb-0.5.2
  ...
```

### Verify Installation

```bash
graph-unified --version
```

**Expected output:**
```
graph-unified version 0.1.0
```

**Troubleshooting:**
- **Command not found:** Ensure virtual environment is activated
- **Import errors:** Run `pip install -e ".[dev]"` again
- **Python version error:** Requires Python 3.10+

---

## Step 2: Configure API Keys (5 minutes)

### Get API Keys

1. **Anthropic API Key:**
   - Go to https://console.anthropic.com/
   - Sign up / log in
   - Navigate to API Keys
   - Create new key (copy immediately, shown once)

2. **Voyage AI API Key:**
   - Go to https://www.voyageai.com/
   - Sign up / log in
   - Get API key from dashboard

### Set Environment Variables

**On macOS/Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export VOYAGE_API_KEY="pa-..."
```

**On Windows:**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:VOYAGE_API_KEY="pa-..."
```

**Or create `.env` file:**
```bash
# In graph-unified directory
cat > .env << EOF
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
EOF
```

### Verify API Access

```bash
graph-unified test-connection
```

**Expected output:**
```
✓ Anthropic API: Connected (Claude Sonnet available)
✓ Voyage AI API: Connected (voyage-3 available)
All API connections successful.
```

**Troubleshooting:**
- **API key invalid:** Double-check key (no spaces, complete string)
- **Network error:** Check firewall/proxy settings
- **Rate limit error:** Wait 1 minute and retry

---

## Step 3: Prepare Sample Data (2 minutes)

### Download Sample Corpus

We'll use a small corpus about AI companies and research:

```bash
# Create input directory
mkdir -p input

# Download sample documents
curl -o input/sample_docs.zip \
  https://github.com/your-org/graph-unified/releases/download/v0.1.0/sample_docs.zip

# Extract
unzip input/sample_docs.zip -d input/
```

**Sample corpus contents:**
```
input/
├── anthropic.txt           # About Anthropic (200 words)
├── openai.txt              # About OpenAI (250 words)
├── transformers.txt        # Transformer architecture (300 words)
├── attention.txt           # Attention mechanism (200 words)
├── scaling_laws.txt        # Neural scaling laws (400 words)
```

**Or create your own:**

Create `input/sample.txt` with any text (minimum 100 words):

```bash
cat > input/sample.txt << 'EOF'
Anthropic is an AI safety company founded in 2021 by Dario Amodei and Daniela Amodei,
former members of OpenAI. The company focuses on developing safe, steerable AI systems.

Anthropic's flagship product is Claude, a large language model assistant designed to be
helpful, harmless, and honest. Claude uses Constitutional AI (CAI) training to align with
human values and reduce harmful outputs.

The company has raised significant funding from investors including Google, Spark Capital,
and Sound Ventures. Anthropic's research focuses on interpretability, scalability, and
AI safety.
EOF
```

---

## Step 4: Create Configuration (5 minutes)

### Generate Default Configuration

```bash
graph-unified init
```

**This creates `settings.yaml` with defaults:**

```yaml
version: "1.0"

# LLM Configuration
llm:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  api_key: "${ANTHROPIC_API_KEY}"
  temperature: 0.0
  max_tokens: 4096

# Embedding Configuration
embedding:
  provider: "voyageai"
  model: "voyage-3"
  dimension: 1024
  api_key: "${VOYAGE_API_KEY}"

# Chunking Configuration
chunking:
  strategy: "token_overlap"
  chunk_size: 512
  overlap: 128

# Extraction Configuration
extraction:
  entity_types:
    - PERSON
    - ORGANIZATION
    - LOCATION
    - CONCEPT
    - PRODUCT
  relationship_types:
    - RELATED_TO
    - PART_OF
    - FOUNDED_BY
    - WORKS_FOR
    - CREATED_BY

# Storage Configuration
storage:
  root_dir: "./output"

# Query Configuration
query:
  default_strategy: "hybrid"
  top_k: 10
```

**Key settings explained:**
- `llm.model`: Which Claude model to use (Sonnet = balanced)
- `chunking.chunk_size`: How large each chunk is (512 tokens = ~2 paragraphs)
- `extraction.entity_types`: What types of entities to extract
- `query.default_strategy`: Which retrieval strategy to use if not specified

**For this tutorial, defaults are fine.** Don't modify yet.

---

## Step 5: Index Documents (10 minutes)

### Run Indexing Pipeline

```bash
graph-unified index \
  --input ./input \
  --config ./settings.yaml
```

**What happens:**
1. **Load documents** from `./input` directory
2. **Chunk documents** into 512-token chunks with 128-token overlap
3. **Extract entities and relationships** using Claude
4. **Generate embeddings** using Voyage AI
5. **Build indexes** for all 6 strategies (parallel)

**Expected output:**
```
Graph-Unified Indexing Pipeline
================================

[1/6] Loading documents...
✓ Found 5 documents (1,350 words total)

[2/6] Chunking documents...
✓ Created 12 chunks (avg 450 tokens/chunk)

[3/6] Extracting entities and relationships...
⏳ Processing chunks [################] 12/12 (100%)
✓ Extracted 23 entities, 18 relationships

[4/6] Generating embeddings...
⏳ Embedding chunks [################] 12/12
⏳ Embedding entities [################] 23/23
✓ Generated 35 embeddings

[5/6] Building strategy-specific indexes...
  ✓ Naive index (vector only)
  ✓ Hybrid index (vector + BM25)
  ✓ GraphRAG Local index (entity neighborhoods)
  ⏳ GraphRAG Global index (computing communities...)
    - Detected 5 communities (level 0)
    - Detected 2 communities (level 1)
    ⏳ Generating community summaries [#######] 7/7
  ✓ GraphRAG Global index complete
  ✓ LightRAG index (entity + relationship)
  ✓ HippoRAG index (associative graph)

[6/6] Saving indexes...
✓ Saved to ./output

Indexing complete!
Time: 8m 32s
Cost: $1.47 (Claude: $1.44, Voyage: $0.03)
```

**What was saved:**
```
output/
├── documents.parquet          # Original documents
├── chunks.parquet             # Chunked text + embeddings
├── entities.parquet           # Extracted entities + embeddings
├── relationships.parquet      # Extracted relationships
├── communities.parquet        # GraphRAG communities
├── community_reports.parquet  # Community summaries
├── lancedb/                   # Vector indexes
│   ├── chunks.lance
│   ├── entities.lance
│   └── relationships.lance
└── graphs/                    # Graph structures
    ├── entity_graph.graphml
    └── hippo_graph.pickle
```

**Troubleshooting:**
- **API rate limit:** Reduce `performance.indexing.max_workers` in settings.yaml
- **Out of memory:** Reduce `chunking.chunk_size` or batch size
- **Extraction errors:** Check entity_types are valid (see reference docs)

---

## Step 6: Query with Different Strategies (5 minutes)

Now let's ask the same question using different retrieval strategies.

### Query 1: Factual Lookup (Naive Strategy)

```bash
graph-unified query \
  --model naive \
  "What is Anthropic?"
```

**Expected output:**
```
Strategy: Naive RAG
Query: What is Anthropic?

Retrieved Contexts (Top 3):
1. [Score: 0.89] Anthropic is an AI safety company founded in 2021...
2. [Score: 0.76] The company focuses on developing safe, steerable AI systems...
3. [Score: 0.71] Anthropic's flagship product is Claude, a large language model...

Generated Answer:
Anthropic is an AI safety company founded in 2021 by Dario Amodei and Daniela Amodei.
The company focuses on developing safe, steerable AI systems and is known for creating
Claude, a large language model assistant.

Time: 1.2s
Cost: $0.01
```

**Why naive works here:** Simple factual question, answer in single chunk.

---

### Query 2: Entity-Centric (GraphRAG Local)

```bash
graph-unified query \
  --model graphrag-local \
  "Tell me about Dario Amodei."
```

**Expected output:**
```
Strategy: GraphRAG Local
Query: Tell me about Dario Amodei.

Entity Neighborhood:
- Entity: Dario Amodei (PERSON)
  Description: Co-founder of Anthropic, former member of OpenAI
- Connected entities:
  - Anthropic (ORGANIZATION) via FOUNDED
  - Daniela Amodei (PERSON) via CO_FOUNDER_WITH
  - OpenAI (ORGANIZATION) via PREVIOUSLY_WORKED_AT
- Relationships:
  - Dario Amodei → FOUNDED → Anthropic (2021)
  - Dario Amodei → PREVIOUSLY_WORKED_AT → OpenAI

Generated Answer:
Dario Amodei is the co-founder of Anthropic, an AI safety company founded in 2021.
He previously worked at OpenAI before founding Anthropic with his sister, Daniela Amodei.
At Anthropic, he leads the development of Claude and research into AI safety and alignment.

Time: 2.1s
Cost: $0.02
```

**Why GraphRAG Local excels:** Entity-centric query benefits from explicit entity graph traversal.

---

### Query 3: Multi-Hop Reasoning (LightRAG)

```bash
graph-unified query \
  --model lightrag \
  "How do transformers relate to attention mechanisms?"
```

**Expected output:**
```
Strategy: LightRAG (Hybrid mode)
Query: How do transformers relate to attention mechanisms?

Retrieved Contexts:
- Entity: Transformers (CONCEPT)
  "Neural network architecture for sequence processing..."
- Entity: Attention Mechanism (CONCEPT)
  "Mechanism allowing models to focus on relevant parts..."
- Relationship: Transformers → USES → Attention Mechanism
  Description: "Transformers rely on self-attention mechanisms to process input
  sequences in parallel, replacing recurrent connections with attention-based
  context aggregation..."

Generated Answer:
Transformers fundamentally rely on attention mechanisms as their core architectural
component. Unlike RNNs, transformers use self-attention to process all input tokens
in parallel, allowing each token to attend to all other tokens. This attention-based
approach enables transformers to capture long-range dependencies efficiently.

Time: 2.5s
Cost: $0.02
```

**Why LightRAG excels:** Multi-hop query benefits from relationship-centric retrieval.

---

### Query 4: Summarization (GraphRAG Global)

```bash
graph-unified query \
  --model graphrag-global \
  "What are the main topics in this corpus?"
```

**Expected output:**
```
Strategy: GraphRAG Global
Query: What are the main topics in this corpus?

Retrieved Community Summaries (Top 2):
1. Community 1 (Level 1): AI Companies and Products
   Entities: Anthropic, OpenAI, Claude, GPT
   Summary: This community represents major AI companies and their products, focusing
   on the development of large language models and AI safety research...

2. Community 2 (Level 1): Machine Learning Architectures
   Entities: Transformers, Attention, Neural Networks, Scaling Laws
   Summary: This community covers foundational ML concepts, particularly transformer
   architectures and their components. Key themes include attention mechanisms and
   scaling behavior...

Generated Answer:
The corpus covers two main topics:

1. **AI Companies and Products**: Focus on Anthropic and OpenAI, their founding teams,
   and products like Claude. Emphasizes AI safety and alignment research.

2. **Machine Learning Architectures**: Focus on transformer architecture, attention
   mechanisms, and scaling laws. Covers technical foundations of modern language models.

These topics are interconnected, as the companies discussed develop systems based on
the architectures described.

Time: 3.8s
Cost: $0.08
```

**Why GraphRAG Global excels:** High-level summarization benefits from pre-computed community summaries.

---

## Step 7: Compare Strategies (3 minutes)

Let's compare how different strategies handle the same query:

```bash
graph-unified query \
  --model all \
  "What companies work on AI safety?"
```

**This runs the query across all strategies and shows comparison:**

```
Strategy Comparison
===================
Query: What companies work on AI safety?

┌──────────────────┬──────────┬──────────┬──────────────────────────────────┐
│ Strategy         │ Time (s) │ Cost ($) │ Answer Preview                   │
├──────────────────┼──────────┼──────────┼──────────────────────────────────┤
│ Naive            │ 1.1      │ 0.01     │ Anthropic is an AI safety co...  │
│ Hybrid           │ 1.4      │ 0.01     │ Anthropic and OpenAI both wo...  │
│ GraphRAG Local   │ 2.2      │ 0.02     │ Two companies mentioned: Ant...  │
│ GraphRAG Global  │ 3.9      │ 0.08     │ The corpus mentions Anthropi...  │
│ LightRAG         │ 2.3      │ 0.02     │ Anthropic focuses explicitly...  │
│ HippoRAG         │ 2.8      │ 0.02     │ Anthropic is the primary com...  │
└──────────────────┴──────────┴──────────┴──────────────────────────────────┘

Recommendation: GraphRAG Local or Hybrid
Reason: Entity-centric query (companies) benefits from entity-aware retrieval.
```

**Key observations:**
- **Naive is fastest** but misses "OpenAI" (only finds Anthropic)
- **Hybrid finds both** companies (BM25 helps with keyword "companies")
- **GraphRAG Local is most complete** (traverses entity graph to find all orgs)
- **GraphRAG Global is overkill** (slow, expensive, same answer)

---

## Step 8: Understanding Your Results (5 minutes)

### View Indexed Data

**See extracted entities:**
```bash
graph-unified inspect entities --limit 10
```

**Output:**
```
Extracted Entities
==================

1. Anthropic (ORGANIZATION)
   Description: AI safety company founded in 2021
   Mentions: 8 chunks
   Connections: 6 relationships

2. Dario Amodei (PERSON)
   Description: Co-founder of Anthropic
   Mentions: 3 chunks
   Connections: 4 relationships

3. Claude (PRODUCT)
   Description: Large language model assistant by Anthropic
   Mentions: 5 chunks
   Connections: 3 relationships

...
```

**See extracted relationships:**
```bash
graph-unified inspect relationships --limit 5
```

**Output:**
```
Extracted Relationships
=======================

1. Dario Amodei → FOUNDED → Anthropic
   Description: Co-founded Anthropic in 2021
   Source chunks: [3, 7]

2. Anthropic → CREATES → Claude
   Description: Developed Claude as flagship product
   Source chunks: [4, 8]

3. Transformers → USES → Attention Mechanism
   Description: Core architectural component
   Source chunks: [11, 12]

...
```

**Visualize entity graph:**
```bash
graph-unified visualize \
  --output entity_graph.png
```

This creates a visualization showing entities (nodes) and relationships (edges).

---

## What You've Learned

**Congratulations!** You've:
- ✓ Installed Graph-Unified
- ✓ Configured API access
- ✓ Indexed your first corpus
- ✓ Queried with 6 different strategies
- ✓ Compared strategy performance
- ✓ Inspected extracted data

**Key takeaways:**
1. **Different strategies excel at different queries:**
   - Naive/Hybrid: Fast, good for factual lookups
   - GraphRAG Local: Best for entity-centric queries
   - GraphRAG Global: Best for summarization
   - LightRAG: Best for multi-hop reasoning
   - HippoRAG: Best for exploratory queries

2. **Unified extraction saves cost:**
   - One extraction pass builds all 6 indexes
   - Fair comparison (identical inputs)

3. **Strategy selection matters:**
   - Right strategy can be 10x faster and 5x cheaper
   - Query routing helps automatically

---

## Next Steps

**Learn More:**
- [Understanding Retrieval Strategies](../explanation/02-understanding-retrieval-strategies.md) - Deep dive into when each strategy works
- [Indexing Your Own Data](02-indexing-your-data.md) - Tutorial for custom corpora
- [Prompt Tuning](../how-to/01-prompt-tuning.md) - Optimize extraction for your domain

**Try Different Queries:**
- Factual: "When was X founded?"
- Entity-centric: "Tell me about Y"
- Multi-hop: "How does A relate to B?"
- Summarization: "What are the main themes?"
- Exploratory: "What else is related to Z?"

**Customize Configuration:**
- Adjust chunk size for your documents
- Add domain-specific entity types
- Tune extraction prompts

**Build an Application:**
- Integrate Graph-Unified into your app
- See [API Reference](../reference/api-reference.md)

---

## Troubleshooting

### "Command not found: graph-unified"

**Cause:** Virtual environment not activated or installation failed.

**Fix:**
```bash
source .venv/bin/activate  # Activate venv
pip install -e ".[dev]"    # Reinstall
```

---

### "API key invalid" error

**Cause:** Environment variable not set or incorrect.

**Fix:**
```bash
# Check if set
echo $ANTHROPIC_API_KEY

# Set if missing
export ANTHROPIC_API_KEY="sk-ant-..."

# Verify
graph-unified test-connection
```

---

### Indexing takes too long

**Cause:** Many documents or large chunks.

**Fix:**
- Reduce `chunking.chunk_size` (e.g., 256 tokens)
- Reduce `performance.indexing.max_workers` (avoid rate limits)
- Process subset first (test with 10 docs)

---

### Query returns no results

**Cause:** Query doesn't match indexed content.

**Fix:**
- Check what entities were extracted: `graph-unified inspect entities`
- Try different strategy: `--model hybrid` (better keyword matching)
- Expand query: "Tell me about Anthropic's products and research"

---

### Out of memory during indexing

**Cause:** Too many embeddings in memory.

**Fix:**
- Reduce `embedding.batch_size` in settings.yaml
- Index smaller batches: `--batch-size 50`
- Ensure sufficient RAM (2 GB minimum)

---

## Summary

Graph-Unified makes it easy to:
1. **Index once** with shared extraction (saves 60-70% cost)
2. **Query many ways** with 6 retrieval strategies
3. **Compare fairly** with identical inputs

Start with defaults, then customize for your domain. The tutorial used a tiny corpus (5 docs), but the same workflow scales to 100,000+ documents.

**Happy querying!**

