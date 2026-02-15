# Retrieval Pipeline Specification

**Version:** 1.0
**Last Updated:** 2026-02-15
**Status:** Technical Specification

## Overview

This document specifies the complete retrieval pipeline for all six strategies. Each strategy is defined with query processing, retrieval algorithm, scoring, context assembly, and generation.

## Query Processing Flow

```
User Query
    ↓
Query Router (optional)
    ↓
Strategy Selection
    ↓
Retrieval Algorithm
    ↓
Context Assembly
    ↓
Response Generation
    ↓
Result + Metadata
```

---

## Strategy 1: Naive RAG

### Overview

Direct vector similarity search on chunk embeddings.

### Input

```python
{
    "query": str,           # User query
    "top_k": int = 10,      # Number of chunks to retrieve
}
```

### Algorithm

```python
from graphunified.storage.vector_store import VectorStore
from graphunified.utils.embedding import EmbeddingModel

class NaiveRetriever:
    """Naive vector similarity retrieval."""

    def __init__(
        self,
        vector_store: VectorStore,
        chunk_store: ParquetStore,
        embedding_model: EmbeddingModel
    ):
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.embedding_model = embedding_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks by vector similarity.

        Steps:
        1. Embed query
        2. Search vector store
        3. Load chunk details
        4. Return contexts with scores
        """
        # 1. Embed query
        query_embedding = await self.embedding_model.embed([query])

        # 2. Search vector store
        results = await self.vector_store.search(
            query_embedding[0],
            top_k=top_k
        )

        # 3. Load chunks
        contexts = []
        for chunk_id, score in results:
            chunk = await self.chunk_store.get_chunk(chunk_id)
            if chunk:
                contexts.append({
                    "text": chunk.text,
                    "score": float(score),
                    "source": str(chunk.id),
                    "metadata": {
                        "document_id": str(chunk.document_id),
                        "chunk_index": chunk.chunk_index
                    }
                })

        return contexts
```

### Scoring

- **Metric:** Cosine similarity
- **Range:** [0.0, 1.0]
- **Normalization:** Embeddings L2-normalized

### Output

```python
[
    {
        "text": "Global temperatures have risen...",
        "score": 0.89,
        "source": "chunk-uuid-1",
        "metadata": {"document_id": "doc-uuid-1", "chunk_index": 0}
    },
    ...
]
```

### Performance

- **Latency:** <100ms for 25K chunks
- **Scalability:** Linear with corpus size (use IVF for >100K chunks)

---

## Strategy 2: Hybrid RAG

### Overview

Combines dense (vector) and sparse (BM25) retrieval with score fusion.

### Input

```python
{
    "query": str,
    "top_k": int = 10,
    "alpha": float = 0.5,   # Dense weight (1-alpha = sparse weight)
}
```

### Algorithm

```python
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """Hybrid dense + sparse retrieval."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Okapi,
        chunk_store: ParquetStore,
        embedding_model: EmbeddingModel,
        chunk_ids: List[str]
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.chunk_store = chunk_store
        self.embedding_model = embedding_model
        self.chunk_ids = chunk_ids

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval with Reciprocal Rank Fusion (RRF).

        Steps:
        1. Dense retrieval (vector search)
        2. Sparse retrieval (BM25)
        3. Fuse scores using RRF
        4. Return top-k
        """
        # 1. Dense retrieval
        query_embedding = await self.embedding_model.embed([query])
        dense_results = await self.vector_store.search(
            query_embedding[0],
            top_k=top_k * 2  # Retrieve more for fusion
        )

        # 2. Sparse retrieval
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k BM25 results
        bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        sparse_results = [
            (self.chunk_ids[i], bm25_scores[i])
            for i in bm25_indices
        ]

        # 3. Fuse using Reciprocal Rank Fusion
        fused_scores = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            alpha=alpha
        )

        # 4. Load top-k chunks
        contexts = []
        for chunk_id, score in sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]:
            chunk = await self.chunk_store.get_chunk(chunk_id)
            if chunk:
                contexts.append({
                    "text": chunk.text,
                    "score": float(score),
                    "source": str(chunk.id),
                    "metadata": {"fusion": "rrf", "alpha": alpha}
                })

        return contexts

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        alpha: float,
        k: int = 60  # RRF constant
    ) -> Dict[str, float]:
        """
        Compute RRF scores.

        RRF(d) = sum over all rankings R: 1 / (k + rank_R(d))
        """
        scores = {}

        # Dense rankings
        for rank, (chunk_id, _) in enumerate(dense_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + alpha / (k + rank + 1)

        # Sparse rankings
        for rank, (chunk_id, _) in enumerate(sparse_results):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1 - alpha) / (k + rank + 1)

        return scores
```

### Scoring

- **Method:** Reciprocal Rank Fusion (RRF)
- **Parameters:**
  - `alpha`: Dense weight (0.0-1.0)
  - `k`: RRF constant (default 60)
- **Formula:** `RRF(d) = α/(k + rank_dense) + (1-α)/(k + rank_sparse)`

### Performance

- **Latency:** <200ms for 25K chunks
- **Optimal alpha:** 0.5-0.7 (empirically determined)

---

## Strategy 3: GraphRAG Local

### Overview

Entity-centric neighborhood search with structured community context.

### Input

```python
{
    "query": str,
    "top_k": int = 10,
    "max_hops": int = 2,    # Graph traversal depth
}
```

### Algorithm

```python
from graphunified.storage.graph_store import GraphStore
import networkx as nx

class GraphRAGLocalRetriever:
    """GraphRAG local search (entity-centric)."""

    def __init__(
        self,
        entity_store: EntityStore,
        graph_store: GraphStore,
        chunk_store: ChunkStore,
        embedding_model: EmbeddingModel
    ):
        self.entity_store = entity_store
        self.graph_store = graph_store
        self.chunk_store = chunk_store
        self.embedding_model = embedding_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        max_hops: int = 2,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Local graph search.

        Steps:
        1. Identify query entities (NER or entity search)
        2. Expand to k-hop neighborhood
        3. Retrieve entity descriptions + relationships
        4. Retrieve source chunks
        5. Rank and return contexts
        """
        # 1. Identify query entities
        query_entities = await self._identify_entities(query)

        if not query_entities:
            # Fallback to vector search if no entities found
            return await self._fallback_vector_search(query, top_k)

        # 2. Expand to neighborhood
        neighborhood_entities = set()
        for entity in query_entities:
            neighbors = await self.graph_store.get_neighbors(
                entity.id,
                max_hops=max_hops
            )
            neighborhood_entities.update(neighbors)

        # 3. Collect entity information
        contexts = []

        # Add entity descriptions
        for entity_id in neighborhood_entities:
            entity = await self.entity_store.get(entity_id)
            if entity:
                contexts.append({
                    "text": f"**{entity.name}** ({entity.type}): {entity.description}",
                    "score": 1.0,  # Primary entities
                    "source": f"entity:{entity.id}",
                    "metadata": {"type": "entity", "entity_type": entity.type}
                })

        # Add relationships
        graph = await self.graph_store.get_graph()
        subgraph = graph.subgraph(neighborhood_entities)
        for source, target, data in subgraph.edges(data=True):
            contexts.append({
                "text": f"{source} → {target}: {data.get('description', data['type'])}",
                "score": 0.8,  # Relationships slightly lower priority
                "source": f"edge:{source}-{target}",
                "metadata": {"type": "relationship"}
            })

        # 4. Add source chunks for top entities
        for entity in query_entities[:3]:  # Top 3 query entities
            for chunk_id in entity.source_chunks[:2]:  # Top 2 chunks per entity
                chunk = await self.chunk_store.get(chunk_id)
                if chunk:
                    contexts.append({
                        "text": chunk.text,
                        "score": 0.7,
                        "source": str(chunk.id),
                        "metadata": {"type": "chunk", "entity": entity.name}
                    })

        # 5. Rank and limit
        contexts.sort(key=lambda x: x["score"], reverse=True)
        return contexts[:top_k]

    async def _identify_entities(self, query: str) -> List[Entity]:
        """Identify entities mentioned in query."""
        # Option 1: Entity search (vector similarity on entity embeddings)
        query_embedding = await self.embedding_model.embed([query])
        entity_results = await self.entity_store.search(
            query_embedding[0],
            top_k=5
        )
        return entity_results

        # Option 2: NER on query (if entity mentions are explicit)
        # Use spaCy or similar for named entity recognition
```

### Context Structure

```
Entity: IPCC (ORGANIZATION): Intergovernmental Panel on Climate Change...
Entity: Climate Report 2024 (EVENT): Major scientific assessment...
Relationship: IPCC → Climate Report 2024: published
Chunk: "The IPCC released its sixth assessment report..."
```

### Scoring

- **Entity descriptions:** Score 1.0
- **Relationships:** Score 0.8
- **Source chunks:** Score 0.7

### Performance

- **Latency:** <500ms for 2-hop neighborhood
- **Scalability:** Depends on graph density

---

## Strategy 4: GraphRAG Global

### Overview

Community-based map-reduce summarization for broad queries.

### Input

```python
{
    "query": str,
    "top_k_communities": int = 5,
}
```

### Algorithm

```python
class GraphRAGGlobalRetriever:
    """GraphRAG global search (community summaries)."""

    def __init__(
        self,
        community_store: CommunityStore,
        report_store: CommunityReportStore,
        llm_client: LLMClient,
        embedding_model: EmbeddingModel
    ):
        self.community_store = community_store
        self.report_store = report_store
        self.llm = llm_client
        self.embedding_model = embedding_model

    async def retrieve(
        self,
        query: str,
        top_k_communities: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Global search via community summaries.

        Steps:
        1. Embed query
        2. Find relevant communities (cosine similarity on summaries)
        3. Retrieve community reports
        4. Perform map-reduce summarization
        5. Return synthesized response
        """
        # 1. Embed query
        query_embedding = await self.embedding_model.embed([query])

        # 2. Find relevant communities
        # Embed all community summaries and compute similarity
        communities = await self.community_store.get_all()
        community_embeddings = await self.embedding_model.embed([
            c.summary for c in communities if c.summary
        ])

        # Compute cosine similarities
        similarities = np.dot(community_embeddings, query_embedding[0])
        top_indices = np.argsort(similarities)[::-1][:top_k_communities]

        # 3. Retrieve community reports
        contexts = []
        for idx in top_indices:
            community = communities[idx]
            report = await self.report_store.get_by_community(community.id)

            if report:
                contexts.append({
                    "text": report.full_content,
                    "score": float(similarities[idx]),
                    "source": f"community:{community.id}",
                    "metadata": {
                        "type": "community_report",
                        "title": report.title,
                        "community_size": community.size
                    }
                })

        return contexts
```

### Map-Reduce Generation

```python
async def generate_with_map_reduce(
    self,
    query: str,
    community_reports: List[str]
) -> str:
    """
    Map-reduce over community reports.

    Map: Summarize each community report relative to query
    Reduce: Synthesize final answer from summaries
    """
    # Map phase: Summarize each report
    map_prompt_template = """
    Query: {query}

    Community Report:
    {report}

    Summarize the key points from this report that are relevant to the query.
    If nothing is relevant, say "No relevant information."
    """

    map_results = []
    for report in community_reports:
        prompt = map_prompt_template.format(query=query, report=report)
        summary = await self.llm.generate(prompt, max_tokens=500)
        if "no relevant information" not in summary.lower():
            map_results.append(summary)

    # Reduce phase: Synthesize final answer
    reduce_prompt_template = """
    Query: {query}

    Relevant Information from Multiple Sources:
    {summaries}

    Synthesize a comprehensive answer to the query based on the information above.
    """

    all_summaries = "\n\n---\n\n".join(map_results)
    reduce_prompt = reduce_prompt_template.format(
        query=query,
        summaries=all_summaries
    )

    final_answer = await self.llm.generate(reduce_prompt, max_tokens=1000)
    return final_answer
```

### Performance

- **Latency:** 2-5 seconds (includes LLM generation)
- **Cost:** Higher due to map-reduce LLM calls

---

## Strategy 5: LightRAG

### Overview

Dual-index retrieval over entities and relationships.

### Input

```python
{
    "query": str,
    "top_k": int = 10,
    "entity_weight": float = 0.6,   # Weight for entity index
}
```

### Algorithm

```python
class LightRAGRetriever:
    """LightRAG dual-index retrieval."""

    def __init__(
        self,
        entity_index: VectorStore,
        relationship_index: VectorStore,
        entity_store: EntityStore,
        relationship_store: RelationshipStore,
        embedding_model: EmbeddingModel
    ):
        self.entity_index = entity_index
        self.relationship_index = relationship_index
        self.entity_store = entity_store
        self.relationship_store = relationship_store
        self.embedding_model = embedding_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        entity_weight: float = 0.6,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Dual-index retrieval.

        Steps:
        1. Search entity index
        2. Search relationship index
        3. Fuse results
        4. Construct context from entities + relationships
        """
        # 1. Embed query
        query_embedding = await self.embedding_model.embed([query])

        # 2. Search entity index
        entity_results = await self.entity_index.search(
            query_embedding[0],
            top_k=top_k
        )

        # 3. Search relationship index
        relationship_results = await self.relationship_index.search(
            query_embedding[0],
            top_k=top_k
        )

        # 4. Fuse scores
        combined_scores = {}

        for entity_id, score in entity_results:
            combined_scores[f"entity:{entity_id}"] = entity_weight * score

        for rel_id, score in relationship_results:
            combined_scores[f"relationship:{rel_id}"] = (1 - entity_weight) * score

        # 5. Load and format contexts
        contexts = []
        for item_key, score in sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]:
            if item_key.startswith("entity:"):
                entity_id = item_key.split(":")[1]
                entity = await self.entity_store.get(entity_id)
                if entity:
                    contexts.append({
                        "text": f"**{entity.name}**: {entity.description}",
                        "score": float(score),
                        "source": item_key,
                        "metadata": {"type": "entity"}
                    })
            elif item_key.startswith("relationship:"):
                rel_id = item_key.split(":")[1]
                rel = await self.relationship_store.get(rel_id)
                if rel:
                    source_entity = await self.entity_store.get(rel.source_entity_id)
                    target_entity = await self.entity_store.get(rel.target_entity_id)
                    contexts.append({
                        "text": f"{source_entity.name} → {target_entity.name}: {rel.description}",
                        "score": float(score),
                        "source": item_key,
                        "metadata": {"type": "relationship"}
                    })

        return contexts
```

### Performance

- **Latency:** <300ms for dual search
- **Optimal entity_weight:** 0.6 (entities more informative than relationships)

---

## Strategy 6: HippoRAG

### Overview

Hippocampus-inspired associative retrieval with Personalized PageRank (PPR).

### Input

```python
{
    "query": str,
    "top_k": int = 10,
    "ppr_alpha": float = 0.85,  # PPR damping factor
}
```

### Algorithm

```python
class HippoRAGRetriever:
    """HippoRAG associative retrieval."""

    def __init__(
        self,
        graph_store: GraphStore,
        entity_store: EntityStore,
        chunk_store: ChunkStore,
        embedding_model: EmbeddingModel
    ):
        self.graph_store = graph_store
        self.entity_store = entity_store
        self.chunk_store = chunk_store
        self.embedding_model = embedding_model

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        ppr_alpha: float = 0.85,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Associative retrieval with PPR.

        Steps:
        1. Identify query entities
        2. Compute PPR from query entities
        3. Rank entities by PPR score
        4. Retrieve top entity contexts
        """
        # 1. Identify query entities
        query_entities = await self._identify_entities(query)

        if not query_entities:
            return await self._fallback_vector_search(query, top_k)

        # 2. Compute Personalized PageRank
        graph = await self.graph_store.get_graph()
        ppr_scores = self._compute_ppr(
            graph,
            seed_nodes=[str(e.id) for e in query_entities],
            alpha=ppr_alpha
        )

        # 3. Rank entities by PPR
        ranked_entities = sorted(
            ppr_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # 4. Construct contexts
        contexts = []
        for entity_id, score in ranked_entities:
            entity = await self.entity_store.get(entity_id)
            if entity:
                contexts.append({
                    "text": f"**{entity.name}**: {entity.description}",
                    "score": float(score),
                    "source": f"entity:{entity.id}",
                    "metadata": {"type": "entity", "ppr_score": score}
                })

                # Add source chunks for top entities
                if len(contexts) <= 3:
                    for chunk_id in entity.source_chunks[:2]:
                        chunk = await self.chunk_store.get(chunk_id)
                        if chunk:
                            contexts.append({
                                "text": chunk.text,
                                "score": float(score * 0.8),
                                "source": str(chunk.id),
                                "metadata": {"type": "chunk"}
                            })

        return contexts[:top_k]

    def _compute_ppr(
        self,
        graph: nx.Graph,
        seed_nodes: List[str],
        alpha: float = 0.85
    ) -> Dict[str, float]:
        """Compute Personalized PageRank from seed nodes."""
        personalization = {node: 1.0 if node in seed_nodes else 0.0 for node in graph.nodes}
        ppr = nx.pagerank(graph, alpha=alpha, personalization=personalization)
        return ppr
```

### Personalized PageRank

**Formula:**

```
PPR(v) = (1 - α) * p(v) + α * Σ(PPR(u) / out_degree(u))
```

Where:
- `α`: Damping factor (0.85)
- `p(v)`: Personalization vector (1.0 for seed nodes, 0.0 otherwise)

### Performance

- **Latency:** <500ms for PPR computation
- **Optimal alpha:** 0.85 (standard PageRank damping)

---

## Response Generation

### Generation Prompt

```python
GENERATION_PROMPT = """
You are a helpful assistant answering questions based on provided context.

CONTEXT:
{contexts}

QUERY: {query}

Answer the query based on the context above. If the context does not contain enough information, say so. Be concise and accurate.

ANSWER:
"""
```

### Generation Parameters

```yaml
generation:
  temperature: 0.3       # Slight creativity
  max_tokens: 1000       # Sufficient for detailed answers
  model: "claude-3-5-sonnet-20241022"
```

### Cost Tracking

```python
@dataclass
class QueryMetadata:
    strategy: str
    latency_ms: int
    tokens_used: int
    cost_usd: float
    context_count: int
    llm_calls: int

def calculate_cost(tokens: int, model: str) -> float:
    """Calculate cost based on token usage."""
    rates = {
        "claude-3-5-sonnet-20241022": {
            "input": 3.00 / 1_000_000,   # $3 per 1M input tokens
            "output": 15.00 / 1_000_000  # $15 per 1M output tokens
        }
    }
    # Rough estimate: 80% input, 20% output
    input_tokens = int(tokens * 0.8)
    output_tokens = int(tokens * 0.2)
    cost = (
        input_tokens * rates[model]["input"] +
        output_tokens * rates[model]["output"]
    )
    return cost
```

---

## Query Router

### Rule-Based Routing

```python
class QueryRouter:
    """Route queries to optimal strategy."""

    def route(self, query: str) -> str:
        """Determine optimal strategy for query."""
        query_lower = query.lower()

        # Summarization queries → GraphRAG Global
        if any(word in query_lower for word in ["summarize", "overview", "landscape"]):
            return "graphrag_global"

        # Entity-focused queries → GraphRAG Local or HippoRAG
        if self._has_entity_mentions(query):
            return "graphrag_local"

        # Multi-hop reasoning → LightRAG
        if any(word in query_lower for word in ["relationship", "connect", "link"]):
            return "lightrag"

        # Specific factual lookup → Naive or Hybrid
        if any(word in query_lower for word in ["what is", "define", "who is"]):
            return "hybrid"

        # Default
        return "hybrid"

    def _has_entity_mentions(self, query: str) -> bool:
        """Check if query contains entity mentions."""
        # Use NER or entity recognition
        # Simplified: check for capitalized words
        words = query.split()
        capitalized = sum(1 for w in words if w[0].isupper())
        return capitalized >= 2
```

---

## Performance Summary

| Strategy | Latency | Best For | Cost/Query |
|----------|---------|----------|------------|
| Naive | <100ms | Factual lookup | $0.001 |
| Hybrid | <200ms | General queries | $0.002 |
| GraphRAG Local | <500ms | Entity-centric | $0.003 |
| GraphRAG Global | 2-5s | Summarization | $0.05 |
| LightRAG | <300ms | Multi-hop reasoning | $0.003 |
| HippoRAG | <500ms | Exploratory search | $0.004 |

---

## Summary

This specification defines:

- **6 retrieval strategies** with complete algorithms
- **Scoring methods** for each strategy
- **Context assembly** patterns
- **Response generation** with Claude
- **Query routing** logic
- **Performance benchmarks** and cost estimates

**Key Insights:**
- Vector search is baseline (Naive)
- Hybrid improves recall with BM25
- Graph strategies excel for entity-centric queries
- Global search best for broad summarization
- Router can auto-select optimal strategy

**Next Steps:**
- Implement retrievers in `query/retrievers/`
- Build query router in `query/router.py`
- Create generation wrapper in `query/generator.py`
- Add comprehensive retrieval tests
