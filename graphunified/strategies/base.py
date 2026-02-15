"""Base classes and interfaces for retrieval strategies."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from graphunified.config.models import Chunk, Community, Entity, Relationship


class QueryType(str, Enum):
    """Types of queries that strategies can handle."""

    FACTOID = "factoid"  # Specific fact-based questions
    EXPLORATORY = "exploratory"  # Broad, open-ended questions
    RELATIONAL = "relational"  # Questions about relationships between entities
    THEMATIC = "thematic"  # Questions about themes, trends, patterns
    COMPARATIVE = "comparative"  # Comparison questions
    TEMPORAL = "temporal"  # Time-based questions


class RetrievalResult(BaseModel):
    """Standardized retrieval result across all strategies."""

    # Retrieved chunks
    chunks: List[Chunk] = Field(default_factory=list)

    # Relevance scores (parallel to chunks)
    scores: List[float] = Field(default_factory=list)

    # Strategy that produced this result
    strategy: str = Field(...)

    # Optional context from graph-based strategies
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    communities: List[Community] = Field(default_factory=list)

    # Strategy-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Performance metrics
    retrieval_time_ms: float = Field(default=0.0)
    total_chunks_searched: int = Field(default=0)

    def __len__(self) -> int:
        """Number of retrieved chunks."""
        return len(self.chunks)

    @property
    def top_score(self) -> float:
        """Highest relevance score."""
        return max(self.scores) if self.scores else 0.0

    @property
    def mean_score(self) -> float:
        """Average relevance score."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    model_config = {
        "json_schema_extra": {
            "example": {
                "strategy": "hybrid",
                "chunks": [],
                "scores": [0.92, 0.87, 0.81],
                "retrieval_time_ms": 145.3,
                "total_chunks_searched": 1000,
                "metadata": {"alpha": 0.5, "bm25_k1": 1.5},
            }
        }
    }


class RetrievalStrategy(ABC):
    """Abstract base class for all retrieval strategies.

    All strategies (Naive, Hybrid, GraphRAG Local/Global, LightRAG, HippoRAG)
    must implement this interface for consistent query routing and evaluation.
    """

    def __init__(self, config: Any):
        """Initialize strategy with configuration.

        Args:
            config: Strategy-specific configuration object
        """
        self.config = config

    @abstractmethod
    async def index(
        self,
        chunks: List[Chunk],
        entities: List[Entity],
        relationships: List[Relationship],
        communities: List[Community],
    ) -> None:
        """Build strategy-specific indexes from shared extracted data.

        This is called after the shared extraction pipeline completes.
        Each strategy builds its own retrieval indexes from the shared data.

        Args:
            chunks: All text chunks with embeddings
            entities: All extracted entities with embeddings
            relationships: All extracted relationships
            communities: All detected communities (for GraphRAG)

        Raises:
            IndexingError: If indexing fails
        """
        pass

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant context for a query.

        Args:
            query: User query string
            top_k: Number of results to retrieve
            **kwargs: Strategy-specific parameters

        Returns:
            RetrievalResult with chunks, scores, and metadata

        Raises:
            RetrievalError: If retrieval fails
        """
        pass

    @abstractmethod
    def supports_query_type(self, query_type: QueryType) -> bool:
        """Check if this strategy handles a given query type.

        Used by the query router to select appropriate strategies.

        Args:
            query_type: Type of query

        Returns:
            True if strategy can handle this query type

        Example:
            Naive RAG supports: FACTOID
            GraphRAG Global supports: EXPLORATORY, THEMATIC
            LightRAG supports: RELATIONAL, EXPLORATORY
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging and routing.

        Returns:
            Human-readable strategy name (e.g., "Naive RAG", "GraphRAG Local")
        """
        pass

    @property
    def requires_entities(self) -> bool:
        """Whether this strategy requires entity extraction.

        Returns:
            True if strategy needs entities (default: False)
        """
        return False

    @property
    def requires_relationships(self) -> bool:
        """Whether this strategy requires relationship extraction.

        Returns:
            True if strategy needs relationships (default: False)
        """
        return False

    @property
    def requires_communities(self) -> bool:
        """Whether this strategy requires community detection.

        Returns:
            True if strategy needs communities (default: False)
        """
        return False

    async def validate_index(self) -> bool:
        """Validate that indexes are properly built.

        Returns:
            True if indexes are valid and ready for retrieval

        Raises:
            IndexValidationError: If validation fails
        """
        return True

    def _normalize_scores(
        self, scores: List[float], method: str = "minmax"
    ) -> List[float]:
        """Normalize scores to [0, 1] range for cross-strategy comparison.

        Args:
            scores: Raw scores from retrieval
            method: Normalization method ('minmax', 'rank', 'softmax', 'sigmoid')

        Returns:
            Normalized scores in [0, 1] range
        """
        from graphunified.strategies.utils import normalize_scores, rank_normalize_scores

        if method == "rank":
            return rank_normalize_scores(scores)
        else:
            return normalize_scores(scores, method=method)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
