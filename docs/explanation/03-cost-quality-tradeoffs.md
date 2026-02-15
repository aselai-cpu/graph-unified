# Cost-Quality Tradeoffs in RAG

## Introduction

Every RAG system faces a fundamental tension: **better quality costs more money**. This guide explains the cost drivers in Graph-Unified, where to spend your budget for maximum impact, and how to optimize the cost-quality frontier.

**Core principle:** Spend on extraction (high impact), save on retrieval (low impact).

## Cost Breakdown

For a typical 10,000 document corpus (average 500 tokens/doc):

| Phase | Operation | Tokens | Cost (Claude Sonnet) | % of Total |
|-------|-----------|--------|---------------------|-----------|
| **Indexing** | Document loading | 0 | $0 | 0% |
| | Chunking | 0 | $0 | 0% |
| | Entity extraction | ~20M | $60 | 60% |
| | Relationship extraction | ~10M | $30 | 30% |
| | Embedding (Voyage) | ~5M | $2.50 | 2.5% |
| | Community detection | 0 | $0 | 0% |
| | Community summaries | ~2M | $6 | 6% |
| | Index building | 0 | $0 | 0% |
| **Query** | Query embedding | ~20 | <$0.01 | <0.1% |
| | Retrieval | 0 | $0 | 0% |
| | Generation | ~2,000 | $0.02 | <0.1% |

**Key insight:** Extraction dominates cost (90%). Optimization must focus here.

## Cost Drivers in Detail

### 1. Entity Extraction (Largest Cost)

**What it does:** Uses Claude to identify entities (PERSON, ORG, LOCATION, CONCEPT) in each chunk.

**Why it's expensive:**
- Requires full chunk context (512 tokens per chunk)
- Complex prompt (examples, instructions, formatting rules)
- Multiple extraction passes (optional "gleanings")

**Cost calculation:**
```
Chunks = Documents × (Avg tokens / chunk_size)
       = 10,000 × (500 / 512) ≈ 9,800 chunks

Tokens per extraction = chunk (512) + prompt (300) + output (200)
                      = 1,012 tokens

Total tokens = 9,800 × 1,012 ≈ 10M tokens
Cost = 10M × $3/M (Sonnet input) + 2M × $15/M (Sonnet output)
     = $30 + $30 = $60
```

**Optimization levers:**

1. **Chunk size:** Larger chunks = fewer API calls but more tokens per call. Sweet spot: 512 tokens.
2. **Extraction prompt:** Shorter prompt saves tokens. Zero-shot vs. few-shot (zero-shot saves ~200 tokens/call).
3. **Gleanings:** Additional extraction passes for missed entities. Cost: +100% per gleaning. Benefit: +10-20% recall.
4. **Model choice:** Claude Haiku ($0.25/$1.25 per M) vs. Sonnet ($3/$15 per M). Haiku: 12x cheaper, ~10% lower quality.

**Cost-quality tradeoff:**

| Configuration | Cost (10K docs) | Entity Recall | Entity Precision |
|---------------|----------------|---------------|-----------------|
| Haiku, zero-shot, chunk=512 | $5 | 0.75 | 0.85 |
| Sonnet, zero-shot, chunk=512 | $60 | 0.85 | 0.92 |
| Sonnet, 1 gleaning, chunk=512 | $120 | 0.92 | 0.93 |
| Sonnet, few-shot, chunk=512 | $80 | 0.90 | 0.95 |

**Recommendation:** Start with Sonnet, zero-shot. Add gleanings only if evaluation shows low recall (<0.80).

---

### 2. Relationship Extraction (Second Largest Cost)

**What it does:** Identifies relationships between entities (RELATED_TO, PART_OF, CAUSED_BY, etc.).

**Why it's expensive:**
- Requires same context as entity extraction
- Often combined with entity extraction (single prompt)
- Relationship descriptions add output tokens

**Cost calculation:**
```
Combined extraction (entities + relationships):
Total tokens ≈ 20M (input) + 4M (output)
Cost = 20M × $3/M + 4M × $15/M
     = $60 + $60 = $120

With unified extraction:
Cost = $90 (optimized prompt, shared across both)
```

**Optimization levers:**

1. **Combined extraction:** Extract entities and relationships in one pass (saves 50% vs. separate).
2. **Relationship types:** Limit to high-value types (CAUSED_BY, PART_OF) vs. generic (RELATED_TO).
3. **Relationship filtering:** Post-process to remove low-confidence relationships (weight < threshold).
4. **Tuple format:** Use compact format (`source<|>relation<|>target`) instead of verbose JSON.

**Cost-quality tradeoff:**

| Configuration | Cost (10K docs) | Relationship Recall | Relationship Precision |
|---------------|----------------|---------------------|----------------------|
| Combined extraction, 5 types | $90 | 0.70 | 0.80 |
| Combined extraction, 10 types | $110 | 0.80 | 0.75 |
| Separate extraction, 5 types | $150 | 0.75 | 0.85 |

**Recommendation:** Combined extraction with 5-7 high-value relationship types. Filter post-extraction (weight > 0.3).

---

### 3. Community Summaries (GraphRAG Only)

**What it does:** Generates natural language summaries of entity communities using Claude.

**Why it's expensive:**
- Each community requires LLM call
- Community context can be large (100+ entities)
- Summaries are long (500-1000 tokens output)

**Cost calculation:**
```
Communities (Leiden, 10K docs) ≈ 500 (level 0) + 50 (level 1) + 5 (level 2)
                                = 555 communities

Tokens per summary = context (1,000) + prompt (200) + output (800)
                   = 2,000 tokens

Total tokens = 555 × 2,000 ≈ 1.1M tokens
Cost = 1.1M × $3/M + 0.4M × $15/M
     = $3.30 + $6 = $9.30
```

**Optimization levers:**

1. **Hierarchical level:** Summarize only level 1+ (skip base communities). Saves 90% cost, loses 10% detail.
2. **Summary length:** Max tokens (500 vs. 1000). Shorter = cheaper but less informative.
3. **Template summaries:** For small communities (<5 entities), use template instead of LLM. Saves ~50% cost.
4. **Lazy summarization:** Generate summaries on-demand at query time (only for retrieved communities). Moves cost from indexing to query.

**Cost-quality tradeoff:**

| Configuration | Cost (10K docs) | Summary Quality | Global Search Accuracy |
|---------------|----------------|----------------|----------------------|
| All levels, 1000 tokens | $30 | High | 0.88 |
| Level 1+, 1000 tokens | $10 | Medium-High | 0.85 |
| Level 1+, 500 tokens | $5 | Medium | 0.82 |
| Template only | $1 | Low | 0.70 |

**Recommendation:** Level 1+, 800 tokens. GraphRAG Global only (skip if not using).

---

### 4. Embeddings (Small but Necessary Cost)

**What it does:** Converts chunks, entities, and relationships to vectors for similarity search.

**Why it's cheap:**
- Embedding models are cheap ($0.025/M tokens for Voyage)
- No generation (output) tokens
- Highly parallelizable

**Cost calculation:**
```
Items to embed:
- Chunks: 10,000
- Entities: 50,000
- Relationships: 100,000
- Community summaries: 555

Avg tokens per item ≈ 100

Total tokens = 160,555 × 100 ≈ 16M tokens
Cost = 16M × $0.025/M = $0.40
```

**Optimization levers:**

1. **Embedding model:** Voyage-3 ($0.025/M) vs. text-embedding-3-large ($0.13/M). Voyage: 5x cheaper, similar quality.
2. **Dimension:** 1024-dim vs. 512-dim. Lower dimension: 50% storage, ~2% accuracy loss.
3. **Selective embedding:** Embed only chunks + entities (skip relationships). Saves 60% embedding cost, breaks LightRAG.

**Cost-quality tradeoff:**

| Configuration | Cost (10K docs) | Retrieval Accuracy (Naive) |
|---------------|----------------|---------------------------|
| Voyage-3, 1024-dim | $0.40 | 0.85 |
| Voyage-3, 512-dim | $0.40 | 0.83 |
| OpenAI, 1024-dim | $2.00 | 0.86 |

**Recommendation:** Voyage-3, 1024-dim. Embedding cost is negligible; don't optimize here.

---

### 5. Query-Time Costs (Negligible)

**What it does:** Retrieval + generation for each query.

**Why it's cheap:**
- Retrieval is algorithmic (no API calls)
- Generation uses small context window (2K tokens)
- Pay only for queries executed (not stored data)

**Cost calculation:**
```
Query: "What is X?"

Retrieval: $0 (local computation)
Generation: query (20) + context (2,000) + output (500)
          = 2,520 tokens
Cost = 2,020 × $3/M + 500 × $15/M
     = $0.006 + $0.0075 = $0.014 per query

1,000 queries/day × $0.014 = $14/day = $420/month
```

**Optimization levers:**

1. **Context window:** Top-k chunks (5 vs. 10 vs. 20). Larger k: better accuracy, higher cost.
2. **Retrieval strategy:** GraphRAG Global most expensive ($0.10/query), Naive cheapest ($0.01/query).
3. **Result caching:** Cache query results (1 hour TTL). ~50% queries are duplicates/similar. Saves 50% cost.
4. **Model choice:** Haiku for simple queries, Sonnet for complex. Reduces avg cost 3x with smart routing.

**Cost-quality tradeoff (1,000 queries):**

| Configuration | Cost | Accuracy (Factual) | Accuracy (Multi-hop) |
|---------------|------|-------------------|---------------------|
| Naive, Haiku, top-5 | $3 | 0.75 | 0.50 |
| Hybrid, Sonnet, top-10 | $14 | 0.85 | 0.65 |
| GraphRAG Global, Sonnet, top-20 | $100 | 0.70 | 0.90 |

**Recommendation:** Hybrid, Sonnet, top-10 for general use. Route complex queries to specialized strategies.

---

## Total Cost of Ownership

### Indexing Costs

| Corpus Size | Extraction Cost | Embedding Cost | Community Summaries | Total Indexing |
|-------------|----------------|----------------|---------------------|---------------|
| 1K docs | $9 | $0.04 | $1 | $10 |
| 10K docs | $90 | $0.40 | $10 | $100 |
| 100K docs | $900 | $4 | $100 | $1,004 |
| 1M docs | $9,000 | $40 | $1,000 | $10,040 |

**Amortization:** Indexing is one-time cost. For long-lived corpora, cost per query approaches zero.

**Example:** 100K docs, 10K queries/month, 12 months:
- Indexing: $1,004 (year 1)
- Querying: $140/month × 12 = $1,680
- Total: $2,684
- **Cost per query:** $2,684 / 120,000 = $0.022

### Query Costs

| Query Volume | Strategy | Cost/Query | Monthly Cost |
|--------------|----------|-----------|-------------|
| 1K/month | Hybrid | $0.014 | $14 |
| 10K/month | Hybrid | $0.014 | $140 |
| 100K/month | Hybrid | $0.014 | $1,400 |
| 10K/month | Mixed (auto-routing) | $0.025 | $250 |

**Caching benefit:** With 50% cache hit rate, costs reduce by 50%.

### Comparison to Alternatives

**vs. Running separate tools:**

| Approach | Indexing Cost (10K docs) | Notes |
|----------|------------------------|-------|
| GraphRAG only | $100 | Single strategy |
| LightRAG only | $100 | Single strategy |
| HippoRAG only | $100 | Single strategy |
| All three separately | $300 | 3x extraction |
| Graph-Unified (all) | $100 | Shared extraction |
| **Savings** | **$200** | **67% reduction** |

**vs. Managed services (Pinecone, OpenAI):**

Managed services charge for storage + queries, not extraction. Different cost model:

| Service | Indexing | Storage (10K docs) | Queries (10K/mo) | Total (Year 1) |
|---------|----------|-------------------|-----------------|---------------|
| Graph-Unified | $100 | $0 (local) | $140 | $240 |
| Pinecone | $0 | $70/mo × 12 = $840 | $0 (included) | $840 |
| OpenAI Assistants | $0 | $20/mo × 12 = $240 | $200 | $440 |

Graph-Unified wins for large corpora (storage cost dominates). Managed services win for small corpora (simplicity).

---

## Optimization Strategies

### 1. Spend on Extraction Quality

**Rationale:** Extraction is one-time cost with compounding benefits.

**High-ROI investments:**
- Use Sonnet (not Haiku) for extraction: +10% quality for +12x cost, but amortized over all queries
- Add one gleaning pass: +10% recall for +100% extraction cost
- Domain-specific prompt tuning: +15% F1 for 2 hours of work

**Low-ROI investments:**
- Extracting 20 entity types (vs. 5): +5% recall, +50% extraction cost, +100% noise
- Separate entity and relationship extraction: +5% precision, +50% cost

### 2. Save on Query Costs

**Rationale:** Query costs recur per query. Small optimizations compound.

**High-ROI optimizations:**
- Result caching (1 hour TTL): -50% query cost for ~zero effort
- Query routing (Haiku for simple, Sonnet for complex): -30% avg cost for 1 day implementation
- Top-k tuning (10 vs. 20): -40% context cost, -5% accuracy

**Low-ROI optimizations:**
- Smaller embedding dimension (512 vs. 1024): -2% accuracy for negligible savings
- Skipping retrieval (use cached responses): Breaks personalization, saves $0.01/query

### 3. Defer Expensive Operations

**Rationale:** Don't pay for features you don't use.

**Lazy operations:**
- Community summaries: Generate on-demand at query time (GraphRAG Global only)
- HippoRAG graph: Build only if HippoRAG queries occur
- Relationship embeddings: Build only for LightRAG

**Savings:** 30-50% indexing cost if not all strategies used.

### 4. Incremental Updates

**Rationale:** Don't re-extract unchanged documents.

**Approach:**
- Hash documents (SHA-256)
- Detect changes on re-index
- Extract only new/changed documents
- Update only affected indexes

**Savings:** 90% cost reduction for small updates (10% corpus change).

---

## Cost Optimization Checklist

**Before indexing:**
- [ ] Do I need all 6 strategies? (If no, skip expensive ones like GraphRAG Global)
- [ ] Have I tuned the extraction prompt for my domain? (Prompt tuning takes 2 hours, improves quality 15%)
- [ ] Am I using the right model tier? (Sonnet for quality, Haiku for cost)
- [ ] Have I set chunk_size optimally? (512 tokens is sweet spot)
- [ ] Do I need gleanings? (Check extraction recall on sample)

**After indexing:**
- [ ] Have I enabled result caching? (50% cost savings)
- [ ] Am I using query routing? (Right strategy for each query type)
- [ ] Have I tuned top-k? (10 is usually enough)
- [ ] Am I monitoring actual costs? (Track tokens used per query)
- [ ] Can I delete unused indexes? (Free up storage)

**For production:**
- [ ] Have I set up incremental indexing? (90% savings on updates)
- [ ] Am I batching queries? (Reduce API overhead)
- [ ] Have I profiled slow queries? (Optimize expensive strategies)
- [ ] Do I have cost alerts? (Prevent surprise bills)

---

## Cost Scenarios

### Scenario 1: Research Lab (Budget-Constrained)

**Profile:**
- Corpus: 10,000 papers
- Queries: 100/month
- Goal: Compare retrieval strategies

**Optimizations:**
- Use Haiku for extraction: $7 (vs. $90 Sonnet)
- Skip community summaries: -$10
- Cache aggressively: -50% query cost
- **Total Year 1:** $7 + $1.40 = $8.40

**Tradeoff:** Lower extraction quality (-10% F1), but acceptable for exploration.

---

### Scenario 2: Production SaaS (Quality-Focused)

**Profile:**
- Corpus: 100,000 documents
- Queries: 10,000/month
- Goal: High accuracy, user-facing

**Optimizations:**
- Use Sonnet + 1 gleaning: $2,000
- Full community summaries: $100
- Query routing (smart strategy selection): -30% avg query cost
- **Total Year 1:** $2,100 + $1,400 = $3,500

**Tradeoff:** Higher cost, but maximizes quality for paying users.

---

### Scenario 3: Enterprise (Scale-Focused)

**Profile:**
- Corpus: 1,000,000 documents
- Queries: 100,000/month
- Goal: Minimize TCO

**Optimizations:**
- Sonnet extraction (quality necessary at scale): $10,000
- Incremental indexing (daily updates): -90% update cost
- Multi-tier query routing (Haiku/Sonnet): -40% avg query cost
- **Total Year 1:** $10,000 + $10,000 = $20,000

**Tradeoff:** High upfront cost, but low per-query cost ($0.017/query).

---

## Conclusion

**Key principles:**

1. **Extraction dominates cost** (90%). Optimize here first.
2. **Quality compounds.** Spend on extraction quality; save on query costs.
3. **Right-size.** Don't build indexes you won't use.
4. **Cache aggressively.** 50% of queries are duplicates/similar.
5. **Route queries.** Use cheap strategies for simple queries.

**Cost-quality frontier:**
- **Minimum viable:** Haiku extraction, naive retrieval, no gleanings → $10 / 10K docs, 0.75 accuracy
- **Balanced:** Sonnet extraction, hybrid retrieval, no gleanings → $100 / 10K docs, 0.85 accuracy
- **Maximum quality:** Sonnet extraction + gleanings, multi-strategy routing → $200 / 10K docs, 0.92 accuracy

**Recommendation for most users:** Balanced configuration. Sonnet quality is worth the cost, gleanings rarely are.

---

**Next Steps:**
- Read [Prompt Tuning Guide](../how-to/01-prompt-tuning.md) to optimize extraction quality
- Read [Performance Optimization](../how-to/05-performance-optimization.md) to speed up indexing
- Use [Cost Estimation Tool](../reference/cost-calculator.md) to estimate your costs

