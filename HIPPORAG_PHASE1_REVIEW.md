# HippoRAG Phase 1 Review: Neurologically-Inspired Retrieval Foundations

**Date:** 2026-02-15
**Reviewer:** HippoRAG Expert (Claude Sonnet 4.5)
**Status:** ✅ Strong foundation with specific recommendations

---

## Executive Summary

The Phase 1 implementation provides a **solid foundation** for HippoRAG's neurologically-inspired retrieval patterns. The data models and configuration system are well-designed and **80% ready** for HippoRAG integration. However, there are **critical gaps** that must be addressed in Phase 2 to properly support HippoRAG's unique architectural requirements.

**Overall Grade: B+** (Strong foundation, but missing HippoRAG-specific components)

---

## Detailed Analysis

### ✅ What Works Well for HippoRAG

#### 1. Entity Model (models.py lines 106-149)

**Strengths:**
- `source_chunks` field enables tracking entity-passage co-occurrence (critical for HippoRAG's bipartite graph)
- `extraction_confidence` supports filtering low-quality entities
- `embedding` field supports entity-centric retrieval
- `aliases` enables entity resolution and deduplication

**HippoRAG Usage:**
```python
# This model directly supports HippoRAG's entity nodes
# Each entity becomes a node in the bipartite graph
entity = Entity(
    name="IPCC",
    type=EntityType.ORGANIZATION,
    source_chunks=[chunk1_id, chunk2_id],  # Co-occurrence weights
    extraction_confidence=0.95,
    embedding=[...]  # For entity similarity edges
)
```

**Impact:** ✅ Excellent support for pattern separation (entity extraction)

---

#### 2. Relationship Model (models.py lines 166-205)

**Strengths:**
- `weight` field (line 180) is perfect for PPR edge weights
- `source_chunks` enables provenance tracking
- `extraction_confidence` supports quality filtering
- Prevents self-loops (line 189)

**HippoRAG Usage:**
```python
# Weight field will be used for PPR graph traversal
relationship = Relationship(
    source_entity_id=entity1_id,
    target_entity_id=entity2_id,
    type=RelationshipType.RELATED_TO,
    weight=0.75,  # Co-occurrence or similarity weight
    extraction_confidence=0.90
)
```

**Impact:** ✅ Strong support for associative connections

---

#### 3. Configuration: PPR Alpha (settings.py line 130)

**Current Implementation:**
```python
class HippoRAGStrategyConfig(BaseModel):
    enabled: bool = True
    ppr_alpha: float = Field(defaults.DEFAULT_HIPPORAG_PPR_ALPHA, ge=0.5, le=0.95)
```

**Strengths:**
- Properly constrained (0.5-0.95 is appropriate for PPR damping)
- Default of 0.85 (defaults.py line 60) aligns with HippoRAG paper

**Limitations:**
- ⚠️ Only captures damping factor, missing other critical PPR parameters

**What's Missing:**
```python
# HippoRAG needs additional configuration
class HippoRAGStrategyConfig(BaseModel):
    enabled: bool = True

    # PPR parameters
    ppr_alpha: float = Field(0.85, ge=0.5, le=0.95)  # Damping factor
    ppr_max_iterations: int = Field(100, ge=10, le=500)  # Convergence
    ppr_tolerance: float = Field(1e-6, ge=1e-8, le=1e-4)  # Convergence threshold

    # Retrieval parameters
    fact_top_k: int = Field(20, ge=5, le=100)  # Facts for graph activation
    entity_top_k: int = Field(50, ge=10, le=200)  # Activated entities
    combine_alpha: float = Field(0.5, ge=0.0, le=1.0)  # PPR + dense fusion

    # Graph construction
    entity_similarity_threshold: float = Field(0.7, ge=0.5, le=0.95)
    max_edges_per_node: int = Field(50, ge=10, le=200)  # Pruning
```

**Impact:** ⚠️ Partial support - needs expansion for full HippoRAG control

---

#### 4. Graph Storage Format (settings.py line 158)

**Current Implementation:**
```python
graph_format: Literal["graphml", "gexf", "pickle"] = defaults.DEFAULT_GRAPH_FORMAT
```

**Analysis:**
- **graphml** (default): Good for visualization, human-readable XML
- **gexf**: Better for Gephi visualization
- **pickle**: ✅ **BEST for HippoRAG** - preserves igraph objects natively

**Recommendation:**
HippoRAG should use **pickle** format because:
1. Preserves graph structure without serialization overhead
2. Maintains edge weights as native floats (critical for PPR)
3. Supports efficient graph operations (PPR requires iterative computation)
4. igraph's native format for Python

**HippoRAG Configuration Override:**
```yaml
# settings-hipporag.yaml
storage:
  graph_format: "pickle"  # Override default for HippoRAG
```

**Impact:** ✅ Supports HippoRAG but needs format guidance

---

### ⚠️ Critical Gaps for HippoRAG

#### 1. Missing: Triple/Fact Extraction Model

**Problem:**
HippoRAG requires **fact embeddings** (subject-predicate-object triples) in addition to entities and relationships. The current models only support typed relationships, not open-ended fact extraction.

**What HippoRAG Needs:**
```python
class Fact(BaseModel):
    """A subject-predicate-object triple for HippoRAG fact retrieval."""

    id: UUID = Field(default_factory=uuid4)
    subject: str = Field(..., min_length=1)
    predicate: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)

    # Source tracking
    source_chunk: UUID = Field(...)  # Single chunk per fact
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    # Embedding for fact retrieval (first stage of HippoRAG)
    embedding: Optional[List[float]] = Field(default=None)
    embedding_model: Optional[str] = Field(default=None)

    # Linked entities (for graph activation)
    entity_ids: List[UUID] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

**Storage Schema:**
```python
FACT_SCHEMA = pa.schema([
    pa.field('id', pa.string()),
    pa.field('subject', pa.string()),
    pa.field('predicate', pa.string()),
    pa.field('object', pa.string()),
    pa.field('source_chunk', pa.string()),
    pa.field('extraction_confidence', pa.float32()),
    pa.field('entity_ids', pa.list_(pa.string())),
    pa.field('metadata', pa.string()),
])
```

**Why This Matters:**
HippoRAG's retrieval pipeline has 4 stages:
1. **Fact Retrieval:** Query → similar facts (needs `Fact` model with embeddings)
2. **Recognition Memory:** Rerank facts (uses DSPy filter)
3. **Graph Activation:** Extract entities from top facts → reset probabilities
4. **PPR Spreading:** Compute activation scores → retrieve passages

Without a `Fact` model, stage 1 cannot be implemented.

**Impact:** 🚨 **CRITICAL** - Must add before Phase 2 entity extraction

---

#### 2. Missing: Chunk-Entity Co-occurrence Tracking

**Problem:**
HippoRAG needs **bipartite graph** structure with:
- Entity nodes
- Passage nodes
- Entity-passage edges (co-occurrence weights)
- Entity-entity edges (similarity weights)

Current models track entity → chunks (`Entity.source_chunks`) but not the reverse.

**What's Needed:**
```python
class Chunk(BaseModel):
    # ... existing fields ...

    # HippoRAG-specific: track entities mentioned in this chunk
    entity_ids: List[UUID] = Field(default_factory=list)
    entity_co_occurrence_weights: Dict[UUID, float] = Field(default_factory=dict)
```

**Alternative (Better for storage):**
Create a separate co-occurrence table:
```python
class EntityChunkEdge(BaseModel):
    """Edge in HippoRAG's bipartite graph."""

    entity_id: UUID = Field(...)
    chunk_id: UUID = Field(...)
    weight: float = Field(default=1.0, ge=0.0)  # TF-IDF or raw count

    # Metadata
    first_position: int = Field(...)  # Character position in chunk
    mention_count: int = Field(default=1, ge=1)
```

**Storage:**
```python
ENTITY_CHUNK_EDGE_SCHEMA = pa.schema([
    pa.field('entity_id', pa.string()),
    pa.field('chunk_id', pa.string()),
    pa.field('weight', pa.float32()),
    pa.field('first_position', pa.int32()),
    pa.field('mention_count', pa.int32()),
])
```

**Impact:** 🚨 **CRITICAL** - Required for bipartite graph construction

---

#### 3. Missing: Entity-Entity Similarity Edges

**Problem:**
HippoRAG's graph includes entity-entity edges based on:
- Embedding similarity (cosine distance)
- Co-occurrence in same passages
- Relationship extraction (already supported)

Current `Relationship` model assumes typed relationships (WORKS_FOR, LOCATED_IN). HippoRAG also needs **similarity edges** that are automatically computed, not extracted.

**What's Needed:**
Either:
1. Use `RelationshipType.SIMILAR_TO` and compute automatically, OR
2. Create separate `EntitySimilarityEdge` model

**Option 1 (Preferred - uses existing model):**
```python
# During graph construction
for entity_a, entity_b in high_similarity_pairs:
    rel = Relationship(
        source_entity_id=entity_a.id,
        target_entity_id=entity_b.id,
        type=RelationshipType.SIMILAR_TO,
        weight=cosine_similarity(entity_a.embedding, entity_b.embedding),
        extraction_confidence=1.0,  # Computed, not extracted
        metadata={"source": "embedding_similarity"}
    )
```

**Configuration Needed:**
```python
class HippoRAGStrategyConfig(BaseModel):
    # ... existing fields ...

    # Entity similarity
    compute_entity_similarity: bool = True
    entity_similarity_threshold: float = Field(0.7, ge=0.5, le=0.95)
    entity_similarity_top_k: int = Field(10, ge=5, le=50)  # Max edges per entity
```

**Impact:** ⚠️ **IMPORTANT** - Can use existing model but needs configuration

---

#### 4. Missing: PPR Computation Parameters

**Problem:**
The configuration only has `ppr_alpha` (damping). HippoRAG needs:
- Max iterations for convergence
- Tolerance threshold
- Restart node weights (from query)
- Edge pruning strategy

**What's Needed (from Section 3 above):**
See full configuration expansion in "Configuration: PPR Alpha" section.

**Impact:** ⚠️ **IMPORTANT** - Needed for reproducible PPR behavior

---

### 🔍 HippoRAG-Specific Recommendations for Phase 2

#### Recommendation 1: Add Fact Extraction Pipeline

**Priority:** 🚨 CRITICAL
**Phase:** Phase 2 (Knowledge Graph Construction)

**Implementation:**
```python
# graphunified/pipelines/fact_extractor.py

class FactExtractor:
    def __init__(self, llm_client: LLMClient, config: ExtractionConfig):
        self.llm = llm_client
        self.config = config

    async def extract_facts(self, chunk: Chunk) -> List[Fact]:
        """Extract subject-predicate-object triples from chunk text."""

        prompt = f"""Extract factual triples from this text.

Text: {chunk.text}

Return JSON with format:
{{
  "facts": [
    {{"subject": "...", "predicate": "...", "object": "..."}},
    ...
  ]
}}
"""

        response = await self.llm.generate(prompt, max_tokens=1000)
        facts_json = json.loads(response)

        facts = []
        for fact_dict in facts_json["facts"]:
            fact = Fact(
                subject=fact_dict["subject"],
                predicate=fact_dict["predicate"],
                object=fact_dict["object"],
                source_chunk=chunk.id,
                extraction_confidence=0.9  # Could use LLM confidence
            )
            facts.append(fact)

        return facts
```

**Storage Integration:**
Add to `ParquetStore`:
```python
async def save_facts(self, facts: List[Fact]) -> None:
    """Save facts to buffer, flushing if full."""
    self.fact_buffer.extend(facts)
    if len(self.fact_buffer) >= self.batch_size:
        await self._flush_facts()
```

---

#### Recommendation 2: Build Bipartite Graph Structure

**Priority:** 🚨 CRITICAL
**Phase:** Phase 2 (Knowledge Graph Construction)

**Implementation:**
```python
# graphunified/graph/hipporag_builder.py

import igraph as ig
from collections import defaultdict

class HippoRAGGraphBuilder:
    def __init__(self, config: HippoRAGStrategyConfig):
        self.config = config
        self.graph = ig.Graph(directed=False)  # Undirected for PPR

    def build_graph(
        self,
        entities: List[Entity],
        chunks: List[Chunk],
        relationships: List[Relationship],
        entity_chunk_edges: List[EntityChunkEdge]
    ) -> ig.Graph:
        """Build HippoRAG's bipartite graph with entities and passages."""

        # Add entity nodes
        entity_ids = {e.id: idx for idx, e in enumerate(entities)}
        self.graph.add_vertices(len(entities))
        self.graph.vs["type"] = ["entity"] * len(entities)
        self.graph.vs["name"] = [e.name for e in entities]

        # Add passage nodes
        chunk_offset = len(entities)
        chunk_ids = {c.id: idx + chunk_offset for idx, c in enumerate(chunks)}
        self.graph.add_vertices(len(chunks))
        self.graph.vs[chunk_offset:]["type"] = ["passage"] * len(chunks)

        # Add entity-passage edges (co-occurrence)
        edges = []
        weights = []
        for edge in entity_chunk_edges:
            entity_idx = entity_ids[edge.entity_id]
            chunk_idx = chunk_ids[edge.chunk_id]
            edges.append((entity_idx, chunk_idx))
            weights.append(edge.weight)

        self.graph.add_edges(edges)
        self.graph.es["weight"] = weights

        # Add entity-entity edges (relationships)
        for rel in relationships:
            src_idx = entity_ids[rel.source_entity_id]
            tgt_idx = entity_ids[rel.target_entity_id]
            self.graph.add_edge(src_idx, tgt_idx, weight=rel.weight)

        # Add entity-entity similarity edges
        if self.config.compute_entity_similarity:
            self._add_similarity_edges(entities, entity_ids)

        return self.graph

    def _add_similarity_edges(
        self,
        entities: List[Entity],
        entity_ids: Dict[UUID, int]
    ):
        """Add edges between similar entities based on embeddings."""
        # Compute pairwise similarities
        # Add edges above threshold
        # Prune to top-k per node
        pass
```

---

#### Recommendation 3: Implement PPR Retrieval

**Priority:** 🚨 CRITICAL
**Phase:** Phase 3 (Strategy Implementation)

**Implementation:**
```python
# graphunified/strategies/hipporag.py

from igraph import Graph

class HippoRAGStrategy:
    def __init__(
        self,
        graph: Graph,
        fact_embeddings: EmbeddingStore,
        chunk_embeddings: EmbeddingStore,
        config: HippoRAGStrategyConfig
    ):
        self.graph = graph
        self.fact_embeddings = fact_embeddings
        self.chunk_embeddings = chunk_embeddings
        self.config = config

    async def retrieve(self, query: str, top_k: int) -> List[Chunk]:
        """HippoRAG's 4-stage retrieval pipeline."""

        # Stage 1: Fact retrieval
        query_embedding = await self.embed_query(query)
        candidate_facts = await self.fact_embeddings.similarity_search(
            query_embedding,
            top_k=self.config.fact_top_k
        )

        # Stage 2: Recognition memory (reranking)
        top_facts = await self.rerank_facts(candidate_facts, query)

        # Stage 3: Graph activation (extract entities)
        activated_entities = self.extract_entities_from_facts(top_facts)
        reset_probs = self.compute_reset_probabilities(activated_entities)

        # Stage 4: PPR spreading
        ppr_scores = self.graph.personalized_pagerank(
            reset=reset_probs,
            damping=self.config.ppr_alpha,
            directed=False
        )

        # Get passage nodes
        passage_nodes = [v for v in self.graph.vs if v["type"] == "passage"]
        passage_scores = {v["chunk_id"]: ppr_scores[v.index] for v in passage_nodes}

        # Combine with dense retrieval
        dense_scores = await self.dense_retrieve(query, top_k=top_k*2)
        combined_scores = self.combine_scores(
            passage_scores,
            dense_scores,
            alpha=self.config.combine_alpha
        )

        # Return top-k passages
        top_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [chunk_id for chunk_id, score in top_chunks]
```

---

#### Recommendation 4: Optimize Storage for HippoRAG

**Priority:** ⚠️ IMPORTANT
**Phase:** Phase 2 (Shared Pipeline)

**Rationale:**
HippoRAG needs THREE separate embedding stores:
1. Chunk embeddings (shared with all strategies)
2. Entity embeddings (shared with LightRAG)
3. **Fact embeddings (HippoRAG-specific)**

**Storage Organization:**
```
output/
├── chunks.parquet              # Shared
├── entities.parquet            # Shared
├── relationships.parquet       # Shared
├── facts.parquet               # HippoRAG-specific
├── entity_chunk_edges.parquet  # HippoRAG-specific
├── embeddings/
│   ├── chunks/                 # Shared (Naive, Hybrid, HippoRAG)
│   ├── entities/               # Shared (LightRAG, HippoRAG)
│   └── facts/                  # HippoRAG-specific
└── graphs/
    ├── graphrag.graphml        # GraphRAG knowledge graph
    ├── lightrag.json           # LightRAG dual-level graph
    └── hipporag.pickle         # HippoRAG bipartite graph (igraph)
```

**Configuration:**
```yaml
# settings.yaml - add HippoRAG-specific paths
strategies:
  hipporag:
    enabled: true
    ppr_alpha: 0.85
    fact_top_k: 20
    storage:
      fact_embeddings_dir: "embeddings/facts"
      graph_file: "graphs/hipporag.pickle"
```

---

### 📊 Comparison: HippoRAG vs Other Strategies

| Component | Naive | Hybrid | GraphRAG | LightRAG | HippoRAG |
|-----------|-------|--------|----------|----------|----------|
| **Chunks** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Entities** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Relationships** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Facts/Triples** | ❌ | ❌ | ❌ | ❌ | ✅ Required |
| **Entity Embeddings** | ❌ | ❌ | ❌ | ✅ | ✅ |
| **Fact Embeddings** | ❌ | ❌ | ❌ | ❌ | ✅ Required |
| **Graph Structure** | ❌ | ❌ | Knowledge Graph | Dual-Level | Bipartite |
| **Graph Algorithm** | ❌ | ❌ | Leiden Communities | Keyword Search | PPR |
| **Retrieval Method** | Vector | Vector+BM25 | Community Summaries | Entity+Relationship | PPR+Dense |

**Key Insight:**
HippoRAG is the **most complex** strategy in terms of data requirements. It needs:
- All shared components (chunks, entities, relationships)
- **Additional fact extraction** (unique to HippoRAG)
- **Three separate embedding stores** (chunks, entities, facts)
- **Bipartite graph structure** (not just entity graph)
- **PPR computation** (iterative algorithm)

---

## Answers to Your Questions

### Q1: Do the Entity and Relationship models support HippoRAG's neurologically-inspired retrieval patterns?

**Answer: ✅ Yes, with caveats**

**What Works:**
- Entity model has `source_chunks` for co-occurrence tracking ✅
- Relationship model has `weight` for PPR edge weights ✅
- Both have `extraction_confidence` for quality filtering ✅
- Entity embeddings support similarity edges ✅

**What's Missing:**
- ❌ No `Fact` model for triple extraction (stage 1 of retrieval)
- ❌ No explicit bipartite edge tracking (entity ↔ chunk edges)
- ⚠️ Relationship model assumes typed relationships, but HippoRAG also needs similarity edges (can work around)

**Verdict:** Models support pattern separation (entity extraction) and pattern completion (graph traversal) philosophically, but need additional models for full implementation.

---

### Q2: Is the ppr_alpha parameter sufficient for controlling Personalized PageRank damping?

**Answer: ⚠️ No, insufficient**

**What's Present:**
- `ppr_alpha` (damping factor): ✅ Correctly constrained (0.5-0.95)
- Default value of 0.85: ✅ Matches HippoRAG paper

**What's Missing:**
- ❌ `ppr_max_iterations`: How long to run PPR (convergence)
- ❌ `ppr_tolerance`: When to stop iterating (convergence threshold)
- ❌ `fact_top_k`: How many facts to retrieve for activation
- ❌ `entity_top_k`: How many entities to activate
- ❌ `combine_alpha`: Weight for combining PPR + dense scores

**Recommendation:**
Expand `HippoRAGStrategyConfig` to include full PPR control (see Section 3 above).

---

### Q3: Will the graph storage format options work well for HippoRAG's graph-based retrieval?

**Answer: ✅ Yes, but use pickle format**

**Analysis:**
- **graphml** (current default): XML format, good for visualization, slow for large graphs
- **gexf**: Similar to graphml, Gephi-friendly
- **pickle** ✅ **RECOMMENDED for HippoRAG**: Native igraph serialization, preserves all metadata

**Why Pickle:**
1. Preserves graph structure exactly (no serialization loss)
2. Fast load/save (binary format)
3. Maintains edge weights as native floats (critical for PPR)
4. Supports all igraph attributes

**Recommendation:**
Override default in HippoRAG configuration:
```yaml
strategies:
  hipporag:
    storage:
      graph_format: "pickle"  # Override global default
```

---

### Q4: Are relationship weights properly modeled for PPR calculations?

**Answer: ✅ Yes, excellent design**

**Analysis:**
- `Relationship.weight` field (line 180) is perfect for PPR edge weights
- Type: `float`, constrained `>= 0.0`
- Default: `1.0` (uniform weights)

**How HippoRAG Will Use:**
```python
# During graph construction
for relationship in relationships:
    graph.add_edge(
        source=entity_ids[relationship.source_entity_id],
        target=entity_ids[relationship.target_entity_id],
        weight=relationship.weight  # Used directly in PPR
    )

# During PPR
ppr_scores = graph.personalized_pagerank(
    reset=reset_probs,
    damping=0.85,
    weights="weight"  # Uses relationship.weight
)
```

**Additional Consideration:**
HippoRAG needs weights for:
1. Entity-entity edges (relationships): ✅ Supported
2. Entity-passage edges (co-occurrence): ⚠️ Need `EntityChunkEdge` model
3. Entity-entity similarity: ✅ Can use `Relationship.weight`

**Verdict:** Weight modeling is correct, but need to extend to entity-passage edges.

---

### Q5: What HippoRAG-specific considerations should Phase 2 address for knowledge graph construction?

**Answer: 4 Critical Additions**

#### 1. 🚨 CRITICAL: Fact Extraction Pipeline

**What:** Extract subject-predicate-object triples from chunks
**Why:** HippoRAG stage 1 (fact retrieval) requires fact embeddings
**How:** Use OpenIE or LLM-based extraction (Claude can do this)

**Implementation:**
- Add `Fact` model (see Recommendation 1)
- Add `FactExtractor` class
- Add fact embeddings to storage
- Integrate into indexing pipeline

---

#### 2. 🚨 CRITICAL: Bipartite Graph Construction

**What:** Build graph with entity nodes + passage nodes + edges
**Why:** HippoRAG's PPR requires bipartite structure
**How:** Track entity-chunk co-occurrence explicitly

**Implementation:**
- Add `EntityChunkEdge` model (see Recommendation 2)
- Modify graph builder to create bipartite structure
- Store graph as pickle format (igraph native)
- Compute co-occurrence weights (TF-IDF or raw counts)

---

#### 3. ⚠️ IMPORTANT: Entity Similarity Edges

**What:** Automatically compute entity-entity edges based on embedding similarity
**Why:** Improves PPR spreading to related entities
**How:** Cosine similarity on entity embeddings, threshold + prune

**Implementation:**
- Add configuration for similarity threshold
- Compute pairwise similarities (efficient for <10k entities)
- Prune to top-k edges per entity (avoid dense graph)
- Store as `Relationship` with `type=SIMILAR_TO`

---

#### 4. ⚠️ IMPORTANT: Incremental Graph Updates

**What:** Support adding new documents without full reindex
**Why:** Production systems need incremental updates
**How:** Use igraph's `add_vertices()` and `add_edges()` methods

**Implementation:**
- Cache existing graph structure
- Extract entities/facts from new chunks
- Merge with existing entities (deduplication)
- Add new nodes and edges
- Recompute entity similarities only for new entities

---

## Summary: HippoRAG Readiness

### ✅ Ready (No Changes Needed)
1. Entity model with co-occurrence tracking
2. Relationship model with edge weights
3. Configuration system for PPR parameters
4. Graph storage format options (use pickle)
5. Parquet storage for entities/relationships
6. Embedding client (Voyage AI)

### ⚠️ Needs Extension (Minor Changes)
1. Expand `HippoRAGStrategyConfig` with full PPR parameters
2. Add entity similarity computation to graph builder
3. Override graph format to pickle for HippoRAG
4. Add incremental update support

### 🚨 Missing (Critical Additions)
1. `Fact` model for triple extraction
2. `FactExtractor` pipeline component
3. Fact embedding storage (third embedding store)
4. `EntityChunkEdge` model for bipartite edges
5. `HippoRAGGraphBuilder` for bipartite construction
6. `HippoRAGStrategy` retrieval implementation

---

## Recommended Phase 2 Priorities for HippoRAG

### Priority 1: Add Fact Model & Extractor (Week 1)
- Define `Fact` Pydantic model
- Implement `FactExtractor` using Claude
- Add fact storage to `ParquetStore`
- Add fact embeddings directory

### Priority 2: Build Bipartite Graph (Week 2)
- Define `EntityChunkEdge` model
- Implement `HippoRAGGraphBuilder`
- Use igraph library
- Store as pickle format

### Priority 3: Extend Configuration (Week 2)
- Add full PPR parameters
- Add fact retrieval parameters
- Add entity similarity parameters
- Add score fusion parameters

### Priority 4: Test Integration (Week 3)
- Verify fact extraction quality
- Verify graph structure (bipartite)
- Verify edge weights
- Benchmark PPR performance

---

## Conclusion

Phase 1 provides a **strong foundation** for HippoRAG, with excellent design choices for entities, relationships, and configuration. However, **critical components are missing**:

1. **Fact extraction** (unique to HippoRAG)
2. **Bipartite graph structure** (entity + passage nodes)
3. **Fact embeddings** (third embedding store)

These must be added in Phase 2 before HippoRAG strategy implementation in Phase 3.

**Overall Assessment:**
- **Data Models:** B+ (good, but missing Fact model)
- **Configuration:** B (good PPR alpha, but needs expansion)
- **Storage:** A (excellent Parquet + pickle support)
- **Graph Support:** B (relationship weights perfect, but needs bipartite structure)

**Recommendation:** Proceed with Phase 2 knowledge graph construction, adding HippoRAG-specific models and builders alongside shared components.

---

**Reviewed By:** HippoRAG Expert (Claude Sonnet 4.5)
**Date:** 2026-02-15
**Next Review:** After Phase 2 completion
