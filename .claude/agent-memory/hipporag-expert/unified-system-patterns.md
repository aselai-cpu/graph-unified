# Unified Multi-Strategy RAG System Patterns

## Strategy Selection Decision Tree

```
Query Analysis
├─ Query Type
│  ├─ Multi-hop reasoning → HippoRAG
│  ├─ Thematic/domain overview → GraphRAG
│  ├─ Direct relationship query → LightRAG
│  └─ Simple similarity → Standard RAG
├─ Query Complexity
│  ├─ High (requires inference) → HippoRAG
│  ├─ Medium (entity-centric) → LightRAG
│  └─ Low (keyword match) → Standard RAG
└─ Expected Result Type
   ├─ Connections/paths → HippoRAG or LightRAG
   ├─ Summaries/themes → GraphRAG
   └─ Specific facts → Standard RAG
```

## Pattern 1: Router-Based Architecture

### Implementation
```python
class UnifiedRAGRouter:
    def __init__(self):
        self.hipporag = HippoRAG(config)
        self.graphrag = GraphRAG(config)
        self.lightrag = LightRAG(config)
        self.standard_rag = StandardRAG(config)

    def route_query(self, query: str) -> str:
        """Analyze query and route to appropriate strategy"""
        features = self.analyze_query(query)

        # Multi-hop or inference queries
        if features.requires_reasoning or features.hop_count > 1:
            strategy = "hipporag"

        # Thematic or domain overview
        elif features.is_thematic or "overview" in query.lower():
            strategy = "graphrag"

        # Direct entity/relationship queries
        elif features.has_explicit_entities and features.query_relationship:
            strategy = "lightrag"

        # Simple similarity
        else:
            strategy = "standard"

        return self._execute_strategy(strategy, query)

    def analyze_query(self, query: str) -> QueryFeatures:
        """Extract features for routing decision"""
        return QueryFeatures(
            requires_reasoning=self._check_reasoning_keywords(query),
            hop_count=self._estimate_hops(query),
            is_thematic=self._check_thematic_keywords(query),
            has_explicit_entities=self._extract_entities(query),
            query_relationship=self._check_relationship_keywords(query)
        )
```

### Routing Keywords
```python
REASONING_KEYWORDS = [
    "how does", "why does", "explain how", "connect",
    "relationship between", "influence", "impact of"
]

THEMATIC_KEYWORDS = [
    "overview of", "summarize", "main themes", "key topics",
    "domain overview", "landscape of"
]

RELATIONSHIP_KEYWORDS = [
    "between", "connects", "related to", "associated with",
    "works with", "located in"
]
```

## Pattern 2: Result Fusion (Ensemble)

### Parallel Retrieval + Reranking
```python
async def ensemble_retrieve(query: str, top_k: int = 10):
    """Run multiple strategies in parallel and fuse results"""

    # Execute strategies in parallel
    results = await asyncio.gather(
        hipporag.retrieve(query, top_k=20),
        graphrag.retrieve(query, top_k=20),
        lightrag.retrieve(query, top_k=20)
    )

    # Normalize scores (each strategy has different scale)
    normalized_results = [
        normalize_scores(r, method="min_max") for r in results
    ]

    # Fuse using Reciprocal Rank Fusion (RRF)
    fused = reciprocal_rank_fusion(normalized_results, k=60)

    # Optional: Rerank with cross-encoder
    reranked = rerank_with_cross_encoder(query, fused)

    return reranked[:top_k]
```

### Reciprocal Rank Fusion
```python
def reciprocal_rank_fusion(result_lists, k=60):
    """Combine rankings from multiple systems"""
    scores = defaultdict(float)
    doc_content = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            scores[doc.id] += 1.0 / (k + rank)
            doc_content[doc.id] = doc

    # Sort by fused score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_content[doc_id] for doc_id, _ in ranked]
```

### Score Normalization
```python
def normalize_scores(results, method="min_max"):
    """Normalize scores to [0, 1] range"""
    scores = [r.score for r in results]

    if method == "min_max":
        min_s, max_s = min(scores), max(scores)
        for r in results:
            r.score = (r.score - min_s) / (max_s - min_s + 1e-10)

    elif method == "z_score":
        mean_s = sum(scores) / len(scores)
        std_s = (sum((s - mean_s)**2 for s in scores) / len(scores))**0.5
        for r in results:
            r.score = (r.score - mean_s) / (std_s + 1e-10)

    return results
```

## Pattern 3: Fallback Chain

### Confidence-Based Fallback
```python
def fallback_retrieve(query: str, top_k: int = 10):
    """Try fast strategy first, fallback to expensive if needed"""

    # Try LightRAG first (fastest)
    results = lightrag.retrieve(query, top_k=top_k)
    confidence = compute_confidence(results)

    if confidence > 0.8:
        return results, "lightrag"

    # Fallback to HippoRAG (more expensive but better coverage)
    results = hipporag.retrieve(query, top_k=top_k)
    confidence = compute_confidence(results)

    if confidence > 0.6:
        return results, "hipporag"

    # Final fallback to GraphRAG (thematic understanding)
    results = graphrag.retrieve(query, top_k=top_k)
    return results, "graphrag"

def compute_confidence(results):
    """Estimate confidence in retrieval results"""
    if not results:
        return 0.0

    # Score gap between top results
    score_gap = results[0].score - results[1].score if len(results) > 1 else 0

    # Average score of top-3
    avg_top3 = sum(r.score for r in results[:3]) / min(3, len(results))

    # Combine signals
    confidence = 0.5 * avg_top3 + 0.5 * min(score_gap, 1.0)
    return confidence
```

## Pattern 4: Shared Component Architecture

### Shared Entity Extraction
```python
class SharedEntityExtractor:
    """Single NER pipeline shared across all strategies"""

    def __init__(self):
        self.ner_model = load_ner_model()
        self.cache = {}  # Cache entity mentions per chunk

    def extract_entities(self, chunk_id: str, text: str):
        """Extract and cache entities for all strategies"""
        if chunk_id in self.cache:
            return self.cache[chunk_id]

        entities = self.ner_model(text)
        self.cache[chunk_id] = entities

        return entities

class UnifiedIndexer:
    """Index content across all strategies with shared components"""

    def __init__(self):
        self.entity_extractor = SharedEntityExtractor()
        self.hipporag = HippoRAG(config)
        self.graphrag = GraphRAG(config)
        self.lightrag = LightRAG(config)

    def index_document(self, doc: Document):
        """Index once, populate all strategies"""
        # Chunk document (shared)
        chunks = self.chunk_document(doc)

        # Extract entities once (shared)
        for chunk in chunks:
            chunk.entities = self.entity_extractor.extract_entities(
                chunk.id, chunk.text
            )

        # HippoRAG: Extract triples + build associative graph
        triples = self.extract_triples(chunks)
        self.hipporag.index(chunks, triples)

        # GraphRAG: Build knowledge graph + communities
        kg = self.build_knowledge_graph(chunks)
        self.graphrag.index(chunks, kg)

        # LightRAG: Build dual-level graph
        dual_graph = self.build_dual_graph(chunks)
        self.lightrag.index(chunks, dual_graph)
```

### Shared Chunk Storage
```python
class CanonicalChunkStore:
    """Single source of truth for document chunks"""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.chunks_file = f"{storage_path}/chunks.parquet"

    def add_chunks(self, chunks: List[Chunk]):
        """Add chunks with hash-based deduplication"""
        # Load existing
        existing = pd.read_parquet(self.chunks_file) if os.path.exists(self.chunks_file) else pd.DataFrame()

        # Create hash IDs
        new_data = []
        for chunk in chunks:
            chunk_hash = hashlib.md5(chunk.text.encode()).hexdigest()
            chunk.id = f"chunk-{chunk_hash}"
            new_data.append({
                "id": chunk.id,
                "text": chunk.text,
                "metadata": json.dumps(chunk.metadata)
            })

        # Deduplicate and save
        df = pd.concat([existing, pd.DataFrame(new_data)]).drop_duplicates(subset=["id"])
        df.to_parquet(self.chunks_file)

        return [c.id for c in chunks]

    def get_chunks(self, chunk_ids: List[str]):
        """Retrieve chunks by ID"""
        df = pd.read_parquet(self.chunks_file)
        return df[df["id"].isin(chunk_ids)]
```

## Pattern 5: Query Analysis Pipeline

### Multi-Stage Query Processing
```python
class QueryAnalyzer:
    """Analyze query to determine optimal strategy"""

    def analyze(self, query: str) -> QueryPlan:
        """Create execution plan for query"""

        # Stage 1: Intent classification
        intent = self.classify_intent(query)

        # Stage 2: Entity detection
        entities = self.extract_query_entities(query)

        # Stage 3: Complexity estimation
        complexity = self.estimate_complexity(query, entities)

        # Stage 4: Strategy selection
        strategies = self.select_strategies(intent, complexity, entities)

        return QueryPlan(
            intent=intent,
            entities=entities,
            complexity=complexity,
            strategies=strategies
        )

    def classify_intent(self, query: str) -> str:
        """Classify query intent"""
        # Use small classifier or LLM
        return classifier.predict(query)  # "factoid", "reasoning", "thematic", etc.

    def estimate_complexity(self, query: str, entities: List[str]) -> int:
        """Estimate query complexity (0-10)"""
        score = 0

        # Multiple entities = potential multi-hop
        score += min(len(entities) - 1, 3)

        # Reasoning keywords
        if any(kw in query.lower() for kw in ["how", "why", "explain"]):
            score += 3

        # Relationship keywords
        if any(kw in query.lower() for kw in ["between", "connect", "relate"]):
            score += 2

        # Question structure
        if query.count("?") > 1:
            score += 2

        return min(score, 10)

    def select_strategies(self, intent: str, complexity: int, entities: List[str]) -> List[str]:
        """Select retrieval strategies based on analysis"""
        strategies = []

        if complexity >= 7:
            strategies.append("hipporag")

        if intent == "thematic":
            strategies.append("graphrag")

        if len(entities) >= 2 and complexity < 7:
            strategies.append("lightrag")

        # Always include standard as baseline
        strategies.append("standard")

        return strategies
```

## Pattern 6: Adaptive Weighting

### Dynamic Strategy Weights
```python
class AdaptiveEnsemble:
    """Learn optimal strategy weights over time"""

    def __init__(self):
        self.strategy_weights = {
            "hipporag": 0.25,
            "graphrag": 0.25,
            "lightrag": 0.25,
            "standard": 0.25
        }
        self.feedback_history = []

    def retrieve_with_feedback(self, query: str, top_k: int = 10):
        """Retrieve and collect feedback for weight adaptation"""

        # Get results from all strategies
        all_results = self.get_all_results(query, top_k)

        # Weighted combination
        combined = self.weighted_combine(all_results)

        # Return results and callback for feedback
        return combined, lambda feedback: self.update_weights(query, feedback)

    def weighted_combine(self, all_results: Dict[str, List[Result]]) -> List[Result]:
        """Combine results using learned weights"""
        scores = defaultdict(float)
        docs = {}

        for strategy, results in all_results.items():
            weight = self.strategy_weights[strategy]
            for result in results:
                scores[result.id] += weight * result.score
                docs[result.id] = result

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [docs[doc_id] for doc_id, _ in ranked]

    def update_weights(self, query: str, feedback: Dict[str, float]):
        """Update strategy weights based on feedback"""
        # Feedback: {"hipporag": 0.9, "graphrag": 0.3, ...}

        # Store feedback
        self.feedback_history.append({
            "query": query,
            "feedback": feedback
        })

        # Adaptive update (exponential moving average)
        alpha = 0.1
        for strategy, score in feedback.items():
            self.strategy_weights[strategy] = (
                (1 - alpha) * self.strategy_weights[strategy] +
                alpha * score
            )

        # Normalize weights
        total = sum(self.strategy_weights.values())
        self.strategy_weights = {
            k: v / total for k, v in self.strategy_weights.items()
        }
```

## Implementation Considerations

### Latency Management
- **Parallel execution**: Run independent strategies concurrently
- **Early termination**: Stop if confidence threshold met
- **Caching**: Cache frequently accessed results
- **Timeouts**: Set per-strategy timeouts to prevent blocking

### Cost Optimization
- **Strategy tiers**: Free (standard) → Low cost (LightRAG) → Medium (HippoRAG) → High (GraphRAG)
- **Progressive retrieval**: Start with cheap, upgrade if needed
- **Batch processing**: Amortize expensive operations across queries

### Consistency
- **Shared chunk IDs**: Use hash-based canonical IDs
- **Synchronized updates**: Update all strategies together
- **Version tracking**: Track index versions per strategy

### Monitoring
- **Strategy usage**: Track which strategies are used most
- **Performance metrics**: Latency, accuracy per strategy
- **Cost tracking**: Monitor API calls, compute time per strategy
- **Error rates**: Track failures per strategy for debugging
