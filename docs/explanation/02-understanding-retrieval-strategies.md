# Understanding Retrieval Strategies

## Introduction

Not all retrieval strategies are created equal. Each excels at different query types, reflecting different philosophies about how knowledge should be organized and accessed.

This guide explains the **conceptual foundations** of each strategy, **when to use them**, and **why they work** (or don't) for specific query types.

## The Retrieval Strategy Spectrum

Retrieval strategies exist on a spectrum from **simple** to **sophisticated**:

```
Simple                                                    Sophisticated
│                                                                      │
Naive ──── Hybrid ──── LightRAG ──── HippoRAG ──── GraphRAG Local ──── GraphRAG Global
│          │           │             │             │                   │
Fast       Balanced    Multi-hop     Associative   Entity-centric     Summarization
Low cost   General     Reasoning     Exploration   Neighborhoods      Thematic
```

**Key insight:** More sophisticated doesn't mean universally better. Simple strategies often win for simple queries.

## Strategy Deep-Dives

### 1. Naive RAG: Direct Vector Search

**Philosophy:** Relevant chunks are semantically similar to the query.

**How it works:**
1. Embed the query (convert to vector)
2. Find k-nearest chunks by cosine similarity
3. Return those chunks as context

**Strengths:**
- Fast (<1 second)
- Low memory footprint
- Easy to understand and debug
- Works well for factual lookups

**Weaknesses:**
- Misses multi-hop reasoning (requires connecting entities across chunks)
- Vulnerable to semantic ambiguity (query similar to wrong chunks)
- No entity awareness (treats all text equally)

**When to use:**
- Simple factual queries: "What is X?", "When did Y happen?"
- Small corpora (<10K documents)
- Prototyping (baseline before adding complexity)
- Real-time constraints (need <1s latency)

**Example:**

**Query:** "What is the capital of France?"

**Naive RAG:**
1. Embed query: [0.23, -0.45, 0.67, ...]
2. Find similar chunks:
   - Chunk 142: "Paris is the capital and largest city of France." (similarity: 0.92)
   - Chunk 289: "France is a country in Western Europe." (similarity: 0.81)
3. Return chunks → Claude generates: "The capital of France is Paris."

**Result:** Perfect for this simple factual query.

---

**Query:** "How did the French Revolution influence European politics?"

**Naive RAG:**
1. Finds chunks about "French Revolution" and "European politics"
2. Misses causal connections spanning multiple chunks
3. Returns disconnected facts

**Result:** Mediocre answer (facts present, but missing reasoning).

---

### 2. Hybrid RAG: Dense + Sparse Fusion

**Philosophy:** Combine semantic similarity (dense vectors) with keyword matching (sparse BM25) for best of both worlds.

**How it works:**
1. Dense search: Embed query, find k-nearest chunks (like naive)
2. Sparse search: BM25 keyword scoring on query terms
3. Fusion: Combine scores (e.g., 0.7 * dense + 0.3 * sparse)
4. Return top-k fused results

**Why fusion helps:**
- Dense vectors capture meaning ("car" ≈ "automobile")
- BM25 captures specificity (rare terms boost relevance)
- Fusion mitigates weaknesses of each

**Strengths:**
- Better than naive on keyword-heavy queries
- Handles technical jargon (BM25 boosts exact matches)
- Still fast (<2 seconds)
- General-purpose (good default)

**Weaknesses:**
- Still no multi-hop reasoning
- No entity awareness
- Tuning fusion weights can be tricky

**When to use:**
- Production default (balanced performance)
- Queries with specific terminology
- When naive RAG misses important keywords
- General-purpose Q&A systems

**Example:**

**Query:** "What are the side effects of metformin?"

**Naive RAG (dense only):**
- Finds chunks semantically similar to "side effects" and "metformin"
- Might return general drug side effect info (not metformin-specific)

**Hybrid RAG:**
- Dense: Semantic similarity to "side effects"
- Sparse: Exact match on "metformin" (rare term, high BM25 score)
- Fusion: Strongly prefers chunks with "metformin" + side effect semantics
- Returns metformin-specific side effects

**Result:** More precise than naive due to keyword specificity.

---

### 3. GraphRAG Local: Entity Neighborhood Search

**Philosophy:** Queries are about entities and their immediate context (relationships, attributes, connected entities).

**How it works:**
1. Extract entities from query ("Anthropic", "Claude")
2. Find those entities in the knowledge graph
3. Traverse graph to retrieve:
   - Entity descriptions
   - Directly connected entities (1-2 hops)
   - Relationships between them
4. Return entity neighborhood as context

**Why graphs help:**
- Entities are first-class citizens (not buried in chunks)
- Relationships provide explicit connections
- Neighborhood captures relevant context without false positives

**Strengths:**
- Excellent for entity-centric queries
- High precision (returns exactly the entity's context)
- Structured context (entities + relationships, not just text)
- Handles ambiguity (different entities with same name)

**Weaknesses:**
- Requires entity extraction (upfront cost)
- Poor for non-entity queries ("What is the meaning of life?")
- Hop depth tuning needed (too few = incomplete, too many = noise)

**When to use:**
- Entity-centric queries: "Tell me about X", "Who is Y?"
- Knowledge base Q&A (organizations, people, products)
- Disambiguation needed ("Which Apple?" → company vs. fruit)
- Relationship queries: "What is the relationship between X and Y?"

**Example:**

**Query:** "Tell me about Anthropic."

**Naive/Hybrid RAG:**
- Finds chunks mentioning "Anthropic"
- Returns scattered mentions across documents
- May miss key relationships (e.g., "Anthropic → founded by → Dario Amodei")

**GraphRAG Local:**
1. Identify entity: "Anthropic" (ORG)
2. Retrieve entity node:
   - Description: "Anthropic is an AI safety company..."
   - Type: ORGANIZATION
3. Retrieve 1-hop neighbors:
   - Connected entities: "Dario Amodei" (PERSON), "Claude" (PRODUCT)
   - Relationships: "FOUNDED_BY", "DEVELOPS"
4. Retrieve 2-hop neighbors (optional):
   - "Dario Amodei" → "PREVIOUSLY_WORKED_AT" → "OpenAI"
5. Return structured context:
   - Anthropic description
   - Founders (Dario Amodei, Daniela Amodei)
   - Products (Claude)
   - History (founded after OpenAI)

**Result:** Comprehensive entity profile, better than scattered chunks.

---

### 4. GraphRAG Global: Community-Based Summarization

**Philosophy:** Some queries require understanding the **big picture** (themes, high-level patterns) rather than specific facts.

**How it works:**
1. Build hierarchical communities of entities (Leiden algorithm)
   - Level 0: Dense clusters of tightly connected entities
   - Level 1: Meta-clusters of Level 0 communities
   - Level 2: Meta-meta-clusters...
2. Generate community summaries using Claude:
   - "This community represents X entities connected by Y relationships. Key themes: ..."
3. Embed community summaries
4. At query time:
   - Find k most relevant community summaries (vector search)
   - Map: Ask Claude to extract query-relevant info from each summary
   - Reduce: Combine map outputs into final answer

**Why communities help:**
- Pre-computed summaries are hierarchical abstractions
- Map-reduce enables reasoning over entire corpus (not just top-k chunks)
- Themes and patterns emerge from community structure

**Strengths:**
- Excellent for summarization and thematic queries
- Scales to large corpora (communities compress information)
- Reveals hidden structure (communities = implicit topics)
- Answers "big picture" questions naive RAG can't

**Weaknesses:**
- Expensive (community detection + summary generation)
- Slow at query time (map-reduce overhead)
- Poor for specific factual queries (summaries lose detail)
- Hierarchical depth tuning needed

**When to use:**
- Summarization: "What are the main themes in this corpus?"
- Thematic queries: "What does this corpus say about climate change?"
- High-level overviews: "Give me an overview of X domain"
- Corpus exploration: "What topics are covered?"

**Example:**

**Query:** "What are the main research areas in AI safety?"

**Naive/Hybrid RAG:**
- Finds chunks mentioning "AI safety" and "research"
- Returns scattered mentions of specific techniques
- Misses thematic organization (which areas are central?)

**GraphRAG Global:**
1. Find relevant communities (vector search on summaries):
   - Community 42 (Level 1): "Alignment and value learning" (10 entities, 23 relationships)
     - Summary: "This community focuses on aligning AI systems with human values..."
   - Community 57 (Level 1): "Robustness and adversarial examples" (8 entities, 15 relationships)
     - Summary: "This community covers techniques for making AI robust to attacks..."
   - Community 89 (Level 1): "Interpretability and explainability" (12 entities, 28 relationships)
     - Summary: "This community addresses understanding AI decision-making..."
2. Map: Extract key research areas from each summary
3. Reduce: Combine into structured answer:
   - Main areas: Alignment, Robustness, Interpretability
   - Sub-areas within each
   - Key researchers and papers

**Result:** High-level thematic answer impossible with chunk-based retrieval.

---

### 5. LightRAG: Dual-Index Entity-Relationship Retrieval

**Philosophy:** Knowledge exists at two levels: **entities** (concepts) and **relationships** (how concepts connect). Different queries target different levels.

**How it works:**
1. Build two indexes:
   - **Entity index:** Embeddings of entity descriptions
   - **Relationship index:** Embeddings of relationship descriptions
2. Four search modes:
   - **Local:** Search entity index (entity-centric, like GraphRAG Local)
   - **Global:** Search relationship index (relationship-centric)
   - **Hybrid:** Combine entity + relationship search
   - **Naive:** Fall back to chunk search

**Key innovation:** Relationship descriptions are **thematic summaries** of connections:

**Raw relationship:**
- Source: "Climate change"
- Target: "Coffee production"
- Type: AFFECTS
- Description: "Rising temperatures and altered rainfall..."

**Relationship description (for indexing):**
- "Climate change affects coffee production through temperature increases, rainfall pattern changes, and increased pest prevalence. This impacts crop yields and coffee quality."

Embedding this description enables semantic search for causal chains.

**Strengths:**
- Excellent for multi-hop reasoning
- Relationship descriptions capture thematic connections
- Four modes provide flexibility
- Simpler than GraphRAG (no community detection)

**Weaknesses:**
- Relationship description quality varies
- Global mode may miss specific facts (trades detail for theme)
- Tuning hybrid weights tricky

**When to use:**
- Multi-hop reasoning: "How does X affect Y?"
- Causal queries: "What causes Z?"
- Exploratory questions: "What connects A and B?"
- When GraphRAG is overkill (no need for communities)

**Example:**

**Query:** "How does climate change affect coffee production?"

**Entity-centric approach (GraphRAG Local):**
- Finds "Climate change" entity and "Coffee production" entity
- Retrieves neighborhoods
- May miss explicit causal connection if not direct edge

**Relationship-centric approach (LightRAG Global):**
1. Search relationship index for query
2. Find relationship description: "Climate change affects coffee production through..."
3. Retrieve that relationship's context (source chunks, connected entities)
4. Return rich causal explanation

**Result:** Directly retrieves the causal chain, better than entity-only search.

---

**Query:** "What are the properties of graphene?"

**Relationship-centric approach:**
- Poor fit (query is about entity properties, not relationships)

**Entity-centric approach (LightRAG Local):**
1. Search entity index for "graphene"
2. Find entity description: "Graphene is a single layer of carbon atoms... Properties: high electrical conductivity, mechanical strength..."
3. Return entity description

**Result:** Entity mode works better for this query type.

---

### 6. HippoRAG: Hippocampus-Inspired Associative Retrieval

**Philosophy:** Human memory retrieves information through **association** and **activation spreading**, not just similarity. Mimic the hippocampus's pattern separation and pattern completion.

**How it works:**
1. Build **associative graph** (bipartite: entities ↔ passages)
   - Edges connect entities to passages mentioning them
   - Edge weights reflect co-occurrence strength
2. Extract **facts** (atomic statements) from chunks
3. At query time:
   - **Pattern separation:** Extract entities/facts from query
   - **Pattern completion:** Use Personalized PageRank (PPR) to activate passages
     - Start from query entities
     - Spread activation through graph
     - High-activation passages are relevant

**Why associative retrieval helps:**
- Retrieves passages **connected through entities**, not just similar
- Activation spreading finds indirect connections
- Mimics human recall ("X reminds me of Y, which reminds me of Z")

**Strengths:**
- Excellent for exploratory queries
- Finds serendipitous connections (not obvious from query)
- Good for "what else is related to X?"
- Handles sparse queries (few keywords)

**Weaknesses:**
- Complex (requires graph + PPR tuning)
- Slower than vector search (graph algorithms)
- May over-activate (retrieve loosely related passages)
- Tuning PPR parameters (damping, iterations) is non-trivial

**When to use:**
- Exploratory search: "What else is related to X?"
- Sparse queries: "Tell me about neural networks" (many possible aspects)
- Association-based retrieval: "I'm interested in X, what should I read?"
- When naive search returns too-narrow results

**Example:**

**Query:** "What else is related to transformers?" (machine learning)

**Naive RAG:**
- Returns chunks directly mentioning "transformers"
- Misses related but not explicitly linked concepts (e.g., "attention mechanisms", "BERT", "language models")

**HippoRAG:**
1. Extract entities from query: "Transformers" (CONCEPT)
2. Find "Transformers" entity in graph
3. PPR activation starting from "Transformers":
   - Direct connections: "Attention mechanism" (high activation)
   - 1-hop: "BERT", "GPT", "Vision transformers" (medium activation)
   - 2-hop: "Self-supervision", "Masked language modeling" (low activation)
4. Retrieve passages with high activation scores
5. Return diverse context covering transformers and related concepts

**Result:** Broad exploratory answer, discovers connections naive search misses.

---

## Comparison Matrix

| Strategy | Query Type | Latency | Accuracy (Factual) | Accuracy (Multi-hop) | Accuracy (Summary) | Cost | Complexity |
|----------|-----------|---------|-------------------|---------------------|-------------------|------|-----------|
| **Naive** | Factual | <1s | Medium | Low | Low | $ | Low |
| **Hybrid** | General | <2s | High | Low | Low | $ | Low |
| **GraphRAG Local** | Entity-centric | <3s | High | Medium | Low | $$$ | High |
| **GraphRAG Global** | Summarization | <5s | Low | High | Very High | $$$$ | Very High |
| **LightRAG** | Multi-hop | <3s | Medium | Very High | Medium | $$ | Medium |
| **HippoRAG** | Exploratory | <5s | Medium | High | Medium | $$$ | High |

**Cost breakdown:**
- $: Vector index only
- $$: Vector + relationship index
- $$$: Vector + graph + communities (no summaries)
- $$$$: Vector + graph + communities + summaries

---

## Decision Tree: Choosing a Strategy

```
Start: What type of query?

┌─ Factual lookup ("What is X?")
│  └─ Contains specific terminology?
│     ├─ Yes → Hybrid RAG
│     └─ No → Naive RAG
│
┌─ Entity-centric ("Tell me about Y")
│  └─ Need relationships/context?
│     ├─ Yes → GraphRAG Local
│     └─ No → Hybrid RAG
│
┌─ Multi-hop reasoning ("How does A affect B?")
│  └─ LightRAG (relationship-centric)
│
┌─ Summarization ("What are main themes?")
│  └─ GraphRAG Global (community summaries)
│
┌─ Exploratory ("What else is related?")
│  └─ Diverse results needed?
│     ├─ Yes → HippoRAG (associative)
│     └─ No → LightRAG Hybrid
│
└─ Unknown/Mixed
   └─ Hybrid RAG (safe default)
```

---

## Strategy Combinations

Some queries benefit from **combining strategies**:

### Sequential Combination

**Query:** "Who are the key researchers in AI safety, and what are their main contributions?"

**Approach:**
1. Use **GraphRAG Global** to identify key researchers (high-level summary)
2. For each researcher, use **GraphRAG Local** to get detailed contributions

**Why:** Global finds the "who", Local fills in the "what."

### Parallel Combination

**Query:** "What is the relationship between climate change and agriculture?"

**Approach:**
1. Run **LightRAG Global** (relationship-centric)
2. Run **GraphRAG Local** (entity neighborhoods)
3. Merge results (de-duplicate, rank by combined score)

**Why:** LightRAG finds causal chains, GraphRAG Local finds detailed entity context. Together, they provide comprehensive coverage.

### Fallback Combination

**Query:** Unknown/ambiguous query

**Approach:**
1. Try **Hybrid RAG** (fast, general-purpose)
2. If results score low (confidence < 0.6), try **HippoRAG** (exploratory)

**Why:** Hybrid handles most queries well. HippoRAG catches edge cases.

---

## Common Misconceptions

### "GraphRAG is always better because it uses graphs"

**False.** Graphs add value only when:
- Entities and relationships matter to the query
- Query requires multi-hop reasoning or thematic understanding

For simple factual lookups, graph overhead provides no benefit. Hybrid RAG wins on speed and simplicity.

### "More sophisticated = higher accuracy"

**False.** Accuracy depends on **query-strategy match**:
- GraphRAG Global is terrible for "What is X?" (too high-level)
- Naive RAG is terrible for "What are main themes?" (too granular)

**Right tool for right job.**

### "I can just always use the best strategy"

**False.** Strategies have different costs:
- GraphRAG Global: $0.10 per query (map-reduce over communities)
- Naive RAG: $0.01 per query (vector search only)

For high-volume production systems, using GraphRAG Global for all queries is prohibitively expensive. Query routing matters.

### "Unified extraction means all strategies are identical"

**False.** Extraction is shared, but **retrieval differs fundamentally**:
- Naive/Hybrid: Chunk-based
- GraphRAG Local: Entity-neighborhood
- GraphRAG Global: Community-summary
- LightRAG: Dual-index (entity + relationship)
- HippoRAG: Associative activation

Shared extraction enables fair comparison, but strategies remain distinct in retrieval philosophy.

---

## Query Type Examples

### Factual Lookups
- "What is the capital of France?"
- "When was Anthropic founded?"
- "What is the speed of light?"

**Best strategy:** Hybrid RAG (keyword specificity)

### Entity-Centric Queries
- "Tell me about Claude."
- "Who is Dario Amodei?"
- "What is Anthropic's mission?"

**Best strategy:** GraphRAG Local (entity neighborhood)

### Multi-Hop Reasoning
- "How does climate change affect coffee production?"
- "What is the relationship between X and Y?"
- "How did event A lead to event B?"

**Best strategy:** LightRAG (relationship chains)

### Summarization
- "What are the main themes in this corpus?"
- "Give me an overview of AI safety research."
- "What topics are covered in these documents?"

**Best strategy:** GraphRAG Global (community summaries)

### Exploratory
- "What else is related to transformers?"
- "I'm interested in quantum computing, what should I read?"
- "What unexpected connections exist in this corpus?"

**Best strategy:** HippoRAG (associative activation)

### Comparative
- "Compare approach A and approach B."
- "What are the differences between X and Y?"
- "Pros and cons of Z?"

**Best strategy:** Hybrid RAG or GraphRAG Local (retrieve both entities, let LLM compare)

---

## Conclusion

Each retrieval strategy embodies a different **theory of relevance**:

- **Naive/Hybrid:** Relevant = semantically/lexically similar
- **GraphRAG Local:** Relevant = in entity's neighborhood
- **GraphRAG Global:** Relevant = in same thematic community
- **LightRAG:** Relevant = connected through entity-relationship chains
- **HippoRAG:** Relevant = associatively activated

No single theory dominates. Query-strategy alignment determines success.

**Key takeaway:** Build all indexes once (unified extraction), then route queries to the optimal strategy. This maximizes accuracy while minimizing cost.

---

**Next Steps:**
- Read [Cost-Quality Tradeoffs](03-cost-quality-tradeoffs.md) to understand how extraction depth affects results
- Read [Query Routing Logic](04-query-routing.md) to see how automatic strategy selection works
- Try [Comparing Retrieval Strategies](../how-to/03-compare-strategies.md) to run experiments on your data

