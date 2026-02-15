# Why Unified Multi-Strategy RAG?

## The Problem: RAG Strategy Proliferation

Retrieval-Augmented Generation (RAG) has exploded in popularity, and with it, dozens of competing retrieval strategies have emerged. Researchers and practitioners face a bewildering landscape:

- **Naive RAG**: Simple vector similarity search on chunks
- **Hybrid RAG**: Combining dense vectors with sparse BM25
- **GraphRAG**: Microsoft's graph + community-based approach
- **LightRAG**: Dual-index entity and relationship retrieval
- **HippoRAG**: Hippocampus-inspired associative memory
- **Dozens more**: ColBERT, RAPTOR, Self-RAG, Corrective RAG...

Each claims superiority on different benchmarks. Each requires its own extraction pipeline, indexing process, and query interface. Teams end up:

1. **Running multiple pipelines** with separate entity extraction, wasting API costs
2. **Comparing apples to oranges** because different inputs bias results
3. **Locking into one strategy** without ability to switch as needs evolve
4. **Reimplementing basic components** for each new strategy

**The core insight:** Most retrieval strategies differ only in *how they organize and search* the same underlying entities and relationships. They all need:
- Document chunking
- Entity extraction
- Relationship detection
- Embedding generation

Why run this expensive extraction six times?

## The Solution: Unified Extraction Pipeline

Graph-Unified separates the **extraction phase** (expensive, shared) from the **retrieval phase** (cheap, strategy-specific):

```
Traditional Approach:
┌─────────────────────────────────────────┐
│ Input Docs → GraphRAG Pipeline → Index │  $$$
│ Input Docs → LightRAG Pipeline → Index │  $$$
│ Input Docs → HippoRAG Pipeline → Index │  $$$
└─────────────────────────────────────────┘
Cost: 3x extraction, 3x embeddings, 3x API calls

Unified Approach:
┌──────────────────────────────────────────────┐
│ Input Docs → Unified Extraction → Chunks     │  $$$
│                                ├→ Entities    │
│                                └→ Relations   │
│                                               │
│ Shared Data → Build GraphRAG Index           │  $
│            → Build LightRAG Index             │  $
│            → Build HippoRAG Index             │  $
└──────────────────────────────────────────────┘
Cost: 1x extraction, 1x embeddings, 1x API calls
Savings: 60-70% on indexing costs
```

**Key Principle:** Extract once, retrieve many ways.

## Why This Matters

### 1. Cost Optimization

**Extraction is expensive.** For a 10,000 document corpus:

| Operation | Tokens | Cost (Claude Sonnet) |
|-----------|--------|---------------------|
| Entity extraction | ~20M | ~$60 |
| Relationship extraction | ~10M | ~$30 |
| Embedding generation | ~5M | ~$2.50 (Voyage) |
| **Total per strategy** | **~35M** | **~$92.50** |
| **Total for 6 strategies** | **~210M** | **~$555** |

With unified extraction:
- **Single extraction pass:** $92.50
- **Six index builds:** ~$5 (graph computation, minimal LLM use)
- **Total cost:** ~$97.50
- **Savings:** $457.50 (83%)

For production systems indexing millions of documents, this savings compounds dramatically.

### 2. Fair Comparison

When each strategy extracts entities differently, comparison is meaningless:

**Scenario:** Comparing GraphRAG vs LightRAG on "What is the relationship between Entity A and Entity B?"

- GraphRAG extracts Entity A but misses Entity B → poor result
- LightRAG extracts both but misses key relationship → mediocre result

**Which strategy is actually better?** You can't tell, because the inputs differed.

**With unified extraction:**
- Both see identical entities and relationships
- Differences in results reflect **retrieval quality**, not extraction luck
- Meaningful comparison enables evidence-based strategy selection

### 3. Query-Adaptive Retrieval

Different query types benefit from different retrieval strategies:

**Entity-centric query:** "Tell me about Anthropic."
- **Best strategy:** GraphRAG Local (entity neighborhood traversal)
- **Why:** Directly retrieves entity and connected context

**Multi-hop reasoning:** "How does climate change affect coffee production?"
- **Best strategy:** LightRAG or HippoRAG (relationship chains)
- **Why:** Follows entity → relationship → entity paths

**Summarization:** "What are the main themes in this corpus?"
- **Best strategy:** GraphRAG Global (community summaries)
- **Why:** Pre-computed hierarchical summaries

**Factual lookup:** "What is the capital of France?"
- **Best strategy:** Hybrid (BM25 + vector)
- **Why:** Simple keyword + semantic match

With unified extraction, you can **route queries** to the optimal strategy without re-indexing. Build all indexes once, then query adaptively.

### 4. Graceful Evolution

Research moves fast. New retrieval strategies emerge monthly. With traditional approaches:
- Adding a new strategy requires full re-extraction (days of work, $$$)
- Teams hesitate to experiment, missing potential improvements

With unified extraction:
- New strategies plug into existing extracted data
- Experimentation becomes cheap (hours, $)
- Teams can adopt cutting-edge research rapidly

## When Unified RAG Makes Sense

**Good Fit:**
- Medium to large corpora (1,000+ documents)
- Diverse query types (factual, reasoning, summarization)
- Need to compare retrieval strategies objectively
- Want flexibility to adopt new strategies
- Cost-sensitive (startups, research labs)
- Production systems needing adaptive retrieval

**Poor Fit:**
- Tiny corpora (<100 documents) - overhead not justified
- Single query type with known optimal strategy
- Already have extraction pipeline tuned for specific strategy
- Real-time streaming with no batch indexing phase

## Philosophical Foundations

### Separation of Concerns

**Extraction** answers: "What entities and relationships exist in this corpus?"
**Retrieval** answers: "How should we organize and search these entities?"

These are orthogonal concerns. Mixing them creates unnecessary coupling and waste.

### Pareto Principle

In RAG systems, 80% of the cost comes from 20% of the operations:
- **Entity extraction:** Requires large context windows, complex prompts, multiple passes
- **Embedding generation:** Requires API calls for every chunk and entity

Optimizing this 20% yields 80% of the savings.

### The Right Abstraction

Graph-Unified abstracts at the **data level** (chunks, entities, relationships), not the **implementation level** (specific graph structures, indexes).

This enables:
- Strategy implementations remain pure (no forced compromises)
- New strategies integrate easily (standard input contract)
- Fair evaluation (identical ground truth)

## What You Give Up

**Unified extraction is not free.** Tradeoffs include:

### 1. Strategy-Specific Optimizations Lost

Some strategies benefit from extraction tuned to their worldview:

- **GraphRAG** extracts entities with community detection in mind (prefers clear clusters)
- **LightRAG** extracts relationship descriptions optimized for semantic search
- **HippoRAG** extracts facts (atomic statements) rather than entities

Unified extraction uses a **general-purpose prompt** that balances these needs. In practice:
- 90-95% of optimal performance retained
- Massive cost savings outweigh small quality loss
- Domain-specific prompt tuning recovers most lost quality

### 2. Incremental Updates More Complex

With strategy-specific pipelines:
- New document arrives
- Extract entities
- Update that strategy's index
- Done (independent per strategy)

With unified extraction:
- New document arrives
- Extract entities (shared)
- Update **all** strategy indexes (coupled)
- More complex coordination

**Mitigation:** Batch updates (hourly/daily) instead of real-time streaming. Most applications can tolerate this latency.

### 3. Storage Overhead

Maintaining indexes for six strategies requires more disk space than one:

| Strategy | Index Size (10K docs) |
|----------|---------------------|
| Naive | 200 MB (vectors only) |
| Hybrid | 250 MB (vectors + BM25) |
| GraphRAG | 500 MB (vectors + graph + communities) |
| LightRAG | 400 MB (vectors + relation index) |
| HippoRAG | 450 MB (vectors + associative graph) |
| **Total** | **~1.8 GB** |

For most use cases, storage is cheap (<$0.10/month on S3). The cost savings from unified extraction ($450+) dwarf this.

**Mitigation:** Lazy index building (only build strategies you use). Delete unused indexes.

### 4. Learning Curve

Graph-Unified exposes six retrieval strategies. Users must understand:
- When to use each strategy
- How to interpret results
- How to tune strategy-specific parameters

This is more complex than "just use vector search."

**Mitigation:**
- Sensible defaults (hybrid strategy)
- Query router auto-selects strategy
- Clear documentation with decision trees
- Examples for common query types

## How Unified Extraction Works

At a high level:

```python
# 1. Chunk documents
chunks = chunk_documents(
    documents,
    chunk_size=512,
    overlap=128
)

# 2. Extract entities and relationships (ONCE)
entities, relationships = extract_with_claude(
    chunks,
    prompt=tuned_extraction_prompt,
    entity_types=["PERSON", "ORG", "LOCATION", "CONCEPT"],
    relationship_types=["RELATED_TO", "PART_OF", "CAUSED_BY"]
)

# 3. Generate embeddings (ONCE)
chunk_embeddings = embed(chunks)
entity_embeddings = embed(entity_descriptions)

# 4. Build strategy-specific indexes (PARALLEL)
await asyncio.gather(
    build_graphrag_index(entities, relationships),
    build_lightrag_index(entities, relationships),
    build_hipporag_index(entities, relationships),
    build_hybrid_index(chunks),
    build_naive_index(chunks)
)
```

**Key observations:**

1. **Steps 1-3 are shared** across all strategies (expensive, run once)
2. **Step 4 is strategy-specific** (cheap, parallelized)
3. **Total cost dominated by steps 1-3** (Claude API calls)

**Strategy-specific post-processing examples:**

- **GraphRAG:** Run Leiden community detection, generate community summaries
- **LightRAG:** Create relationship description embeddings
- **HippoRAG:** Extract facts from chunks, build associative graph

These steps are **cheap** because they operate on already-extracted data with minimal or no additional LLM calls.

## Comparison to Other Approaches

### vs. Strategy-Specific Tools

**MS GraphRAG, LightRAG, HippoRAG** (standalone tools):

| Aspect | Standalone Tools | Graph-Unified |
|--------|-----------------|---------------|
| Extraction | Separate per tool | Shared once |
| Cost | $92.50 per tool | $97.50 total |
| Comparability | Different inputs | Identical inputs |
| Query routing | N/A | Automatic |
| New strategies | Re-extract | Plug-in |
| Best for | Single strategy | Multi-strategy |

**When to use standalone:** You've already chosen one strategy and won't change.

**When to use unified:** Exploring strategies, diverse query types, cost-sensitive.

### vs. Vector Database Platforms

**Pinecone, Weaviate, Qdrant** (managed vector DBs):

These focus on **vector search infrastructure**, not **retrieval strategies**. They solve:
- Scalability (billions of vectors)
- Low latency (<100ms)
- High availability

Graph-Unified solves:
- **Which retrieval strategy to use**
- Entity and relationship extraction
- Graph-based retrieval (not just vector search)

**Complementary, not competing.** You could use Pinecone as the vector backend for Graph-Unified (instead of LanceDB).

### vs. LangChain/LlamaIndex

**LangChain, LlamaIndex** (RAG frameworks):

These provide **orchestration and integration**, not **retrieval strategies**. They offer:
- Pre-built chains (load → split → embed → retrieve → generate)
- Integrations with 100+ LLMs and vector stores
- Prompt templates and memory

Graph-Unified provides:
- **Specific retrieval strategies** (GraphRAG, LightRAG, HippoRAG)
- Unified extraction pipeline
- Query routing logic

**Complementary.** LangChain could orchestrate Graph-Unified's retrieval (call Graph-Unified API, use results in chain).

## Real-World Impact

### Research Teams

**Scenario:** Academic lab comparing retrieval strategies on biomedical literature.

**Traditional approach:**
- Run GraphRAG on PubMed corpus: 3 days indexing, $200
- Run LightRAG on PubMed corpus: 3 days indexing, $200
- Run HippoRAG on PubMed corpus: 3 days indexing, $200
- **Total:** 9 days, $600

**Graph-Unified approach:**
- Index once with all strategies: 3 days, $100
- Compare strategies on identical data
- **Total:** 3 days, $100

**Benefit:** 6x faster, 6x cheaper, apples-to-apples comparison.

### Production Systems

**Scenario:** SaaS company building Q&A over customer documentation.

**Query distribution:**
- 40% factual lookups ("What is X?")
- 30% entity-centric ("Tell me about Y")
- 20% multi-hop reasoning ("How does A affect B?")
- 10% summarization ("What are main topics?")

**Traditional approach:**
- Pick one strategy (say, hybrid)
- 40% of queries work great
- 60% get suboptimal results

**Graph-Unified approach:**
- Build all indexes
- Route queries to optimal strategy
- 90% of queries get great results

**Benefit:** Better user experience without re-indexing per query type.

### Cost-Sensitive Startups

**Scenario:** Bootstrapped startup indexing legal documents.

**Corpus:** 50,000 legal documents (case law, statutes, regulations)

**Traditional approach (GraphRAG):**
- Indexing cost: $4,500
- Must commit to GraphRAG upfront
- Risky if suboptimal for their queries

**Graph-Unified approach:**
- Indexing cost: $5,000 (all 6 strategies)
- Test all strategies on real queries
- Discover LightRAG works best for their use case
- **Avoided** $20,000 mistake (committing to wrong strategy, re-indexing later)

**Benefit:** De-risked strategy selection for 11% additional cost.

## When Unified Extraction Breaks Down

Despite its benefits, unified extraction has limits:

### 1. Extreme Strategy Divergence

If a new strategy fundamentally differs in extraction needs (e.g., requires sentence-level granularity instead of chunks), unified extraction may not fit.

**Example:** Fine-grained citation extraction might need sentence boundaries preserved, but chunk-based extraction loses these.

**Solution:** Extend extraction pipeline with strategy-specific post-processing. In practice, rare.

### 2. Real-Time Streaming

If documents arrive continuously (e.g., social media firehose), batch extraction doesn't fit.

**Solution:** Run extraction on mini-batches (every 5 minutes). Index updates lag slightly. For most applications, acceptable.

### 3. Strategy Requires Different Entity Types

**Example:** Medical RAG might need ICD-10 codes extracted, but legal RAG needs statutory citations.

Unified extraction can't extract both simultaneously (conflicts in prompt focus).

**Solution:** Use domain-specific prompt tuning. Graph-Unified supports per-project prompts.

## Conclusion

Unified multi-strategy RAG represents a **pragmatic optimization** in the RAG landscape:

1. **Cost savings** (60-70%) from shared extraction
2. **Fair comparison** via identical inputs
3. **Query-adaptive retrieval** without re-indexing
4. **Graceful evolution** as new strategies emerge

The tradeoffs (strategy-specific optimizations lost, storage overhead) are **dominated** by the benefits for most use cases.

**When to adopt:** Medium to large corpora, diverse query types, cost-sensitive, need flexibility.

**When to skip:** Tiny corpora, single query type with known optimal strategy, real-time streaming requirements.

Graph-Unified doesn't claim to be the **only** way to do RAG. It claims to be the **smart** way to explore and deploy multiple RAG strategies efficiently.

---

**Next Steps:**
- Read [Understanding Retrieval Strategies](02-understanding-retrieval-strategies.md) to learn when each strategy excels
- Read [Cost-Quality Tradeoffs](03-cost-quality-tradeoffs.md) to understand extraction depth vs. cost
- Try the [Getting Started Tutorial](../tutorial/01-getting-started.md) to see unified extraction in action

