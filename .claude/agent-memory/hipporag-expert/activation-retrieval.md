# HippoRAG Activation-Based Retrieval

## Neurological Inspiration

HippoRAG models the hippocampus's two-stage memory process:
1. **Pattern Separation** (DG/CA3): Separates similar experiences into distinct representations
2. **Pattern Completion** (CA3/CA1): Recovers full memories from partial cues via spreading activation

This differs fundamentally from traditional RAG's single-stage similarity search.

## Four-Stage Retrieval Algorithm

### Stage 1: Fact Retrieval (Pattern Separation)
```
Query → Embed → Similarity Search in fact_embeddings → Top-K facts
```
- Searches the **fact_embeddings** store (triples: subject-predicate-object)
- Returns candidate facts that are semantically similar to query
- Typically retrieve 50-100 candidate facts
- Fast: Standard dense retrieval (~50-100ms for 10K facts)

**Purpose**: Initial filtering to identify relevant atomic facts

### Stage 2: Recognition Memory (Reranking)
```
Top-K facts → DSPy Filter/Reranker → Filtered facts with scores
```
- Uses DSPy-based filter to rerank facts by relevance
- Removes spurious matches from Stage 1
- Reduces to top 10-20 most relevant facts
- Adds ~50-100ms to query latency

**Purpose**: Precision filtering before expensive graph activation

### Stage 3: Graph Activation (Creating Reset Probabilities)
```
Filtered facts → Extract entities → Create reset probability vector
```
- Extracts all entities mentioned in top facts
- Builds "reset probability" vector for PPR:
  - Entities in relevant facts get non-zero weights
  - Weights proportional to fact relevance scores
  - All other nodes get zero weight
- Creates activation "seed" for spreading

**Purpose**: Identify starting nodes for activation spreading

### Stage 4: Personalized PageRank Spreading
```
Reset probabilities → PPR on graph → Node activation scores
```
- Runs PPR with reset probabilities from Stage 3
- Activation spreads through graph edges:
  - Entity → Entity (similarity/co-occurrence)
  - Entity → Passage (co-occurrence)
- Passages connected to activated entities receive scores
- Combines PPR scores with dense retrieval scores

**Purpose**: Discover contextually relevant passages via associative connections

## PPR Algorithm Details

### Standard PPR Formula
```
s_t+1 = (1 - α) * A * s_t + α * r
```
- `s_t`: Activation scores at iteration t
- `A`: Column-normalized adjacency matrix
- `α`: Damping/teleport probability (typically 0.85)
- `r`: Reset probability vector (from Stage 3)

### Convergence
- Iterates until `||s_t+1 - s_t|| < ε` (typically ε = 1e-6)
- Usually converges in 10-30 iterations
- Time: O((nodes + edges) * iterations)

### HippoRAG Implementation Notes
- Uses igraph's `personalized_pagerank()` method
- Can cache results for repeated queries with same entities
- Threshold parameter: Ignore nodes with activation < threshold

## Score Combination

### Final Passage Ranking
```
final_score = α * ppr_score + (1 - α) * dense_score
```
- `ppr_score`: Normalized PPR activation for passage node
- `dense_score`: Direct similarity between query and passage embedding
- `α`: Combination weight (default: 0.5)

**Rationale**: Balances graph-based discovery with direct semantic similarity

## Performance Optimization Strategies

### 1. PPR Caching
- Cache PPR results keyed by activated entity set
- Invalidate cache when graph topology changes
- Reduces repeated computation for similar queries

### 2. Sparse Graph Representation
- Use sparse adjacency matrix for large graphs
- Prune low-weight edges (< threshold)
- Reduces PPR computation time

### 3. Approximate PPR
- Monte Carlo approximation for very large graphs
- Random walk sampling instead of power iteration
- Trade accuracy for speed (useful for >100K nodes)

### 4. Parallel Fact Retrieval
- Stage 1 and dense passage retrieval can run in parallel
- Both are independent vector searches
- Combine results in Stage 4

## Key Parameters and Tuning

### Fact Retrieval (Stage 1)
- `fact_top_k`: 50-100 for initial candidates
- Higher values: Better recall, slower Stage 2

### Recognition Memory (Stage 2)
- `filtered_fact_count`: 10-20 for graph activation
- Higher values: More activation seeds, potentially noisier

### PPR (Stage 4)
- `damping`: 0.85 standard, higher = more spreading
- `threshold`: 0.01 typical, filters weak activations
- `max_iterations`: 30 typical, increase for large graphs

### Score Combination
- `combine_alpha`: 0.5 balanced, higher = favor PPR
- Tune based on query type:
  - Multi-hop queries: Higher alpha (0.6-0.8)
  - Direct queries: Lower alpha (0.3-0.5)

## Comparison with Other Retrieval Methods

### vs. Traditional RAG (Dense Retrieval Only)
- Traditional: Query → Embed → Top-K by similarity
- HippoRAG: Query → Facts → Graph activation → Contextual passages
- Benefit: Discovers passages that aren't directly similar to query

### vs. GraphRAG (Community-Based)
- GraphRAG: Query → Community summaries → Relevant communities
- HippoRAG: Query → Facts → PPR spreading → Activated passages
- Difference: HippoRAG is query-dependent, GraphRAG is query-independent communities

### vs. LightRAG (Relationship Search)
- LightRAG: Query → Entity/Relation match → Direct paths
- HippoRAG: Query → Facts → Associative spreading → Neighborhoods
- Difference: HippoRAG finds implicit connections, LightRAG finds explicit relationships

## When Activation-Based Retrieval Excels

### Strong Use Cases
1. **Multi-hop reasoning**: "How does A influence C through B?"
   - Direct similarity may miss intermediate B
   - PPR spreads activation A → B → C

2. **Context-dependent retrieval**: Same entity, different contexts
   - Reset probabilities create query-specific neighborhoods
   - Different queries activate different subgraphs

3. **Connecting distant concepts**: Low direct similarity, high associative relevance
   - Graph structure captures co-occurrence patterns
   - PPR discovers non-obvious connections

### Weak Use Cases
1. **Simple factoid queries**: "What is the capital of France?"
   - Direct similarity sufficient
   - Graph spreading adds latency without benefit

2. **Keyword matching**: "Find documents mentioning 'quantum computing'"
   - Lexical search or entity search more efficient
   - PPR overhead not justified

3. **Very small corpora**: < 100 documents
   - Graph too sparse for meaningful spreading
   - Dense retrieval alone performs similarly

## Implementation Gotchas

### Graph Construction
- Ensure entity normalization (e.g., "OpenAI" vs "openai" vs "Open AI")
- Edge weights matter: Co-occurrence frequency or similarity
- Bipartite structure: Entities AND passages as nodes

### PPR Computation
- Check convergence: Monitor iteration count
- Handle disconnected components: Some nodes may be unreachable
- Numerical stability: Normalize adjacency matrix properly

### Score Interpretation
- PPR scores are relative, not absolute
- Normalize before combining with dense scores
- Different α values change score distributions

## Debugging Tips

### Low Recall
- Check Stage 1 fact_top_k: May be too low
- Verify fact extraction quality: Are important triples captured?
- Examine graph connectivity: Are relevant entities connected?

### High Latency
- Profile PPR computation: Is graph too dense?
- Consider approximate PPR for large graphs
- Cache PPR results for common query patterns

### Poor Relevance
- Tune combine_alpha: Balance PPR vs dense retrieval
- Check entity extraction: Are query entities recognized?
- Verify edge weights: Do they reflect true co-occurrence?
