"""Strategy selection module.

Maps QueryType to optimal retrieval strategies based on support matrix.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from graphunified.strategies.base import QueryType

logger = logging.getLogger(__name__)


@dataclass
class StrategySelection:
    """Result of strategy selection.

    Attributes:
        strategies: List of strategy names in priority order
        weights: Normalized weights for each strategy (parallel to strategies)
        reasoning: Explanation of strategy selection
    """

    strategies: List[str]
    weights: List[float]
    reasoning: str


class StrategySelector:
    """Selects optimal retrieval strategies based on query type.

    Uses a support matrix derived from Phase 3 evaluation results to map
    query types to the most effective retrieval strategies.
    """

    # Strategy support matrix: QueryType -> [(strategy_name, weight), ...]
    # Weights represent relative effectiveness (1.0 = best, lower = less effective)
    # Based on Phase 3 evaluation results
    STRATEGY_SUPPORT: Dict[QueryType, List[Tuple[str, float]]] = {
        QueryType.FACTOID: [
            ("Naive RAG", 1.0),  # Excellent for direct facts
            ("Hybrid", 0.9),  # Good keyword matching
            ("LightRAG", 0.7),  # Can do local queries
            ("HippoRAG", 0.6),  # If personalized memory needed
        ],
        QueryType.EXPLORATORY: [
            ("GraphRAG Global", 1.0),  # Designed for broad queries
            ("LightRAG", 0.9),  # Good global mode
            ("Hybrid", 0.7),  # Can surface diverse results
            ("Naive RAG", 0.5),  # May miss broader context
        ],
        QueryType.RELATIONAL: [
            ("LightRAG", 1.0),  # Excellent at relationships
            ("GraphRAG Local", 0.9),  # Good at entity connections
            ("HippoRAG", 0.8),  # Knowledge graph integration
            ("Hybrid", 0.6),  # Limited graph awareness
        ],
        QueryType.THEMATIC: [
            ("GraphRAG Global", 1.0),  # Community summaries capture themes
            ("LightRAG", 0.8),  # Can identify patterns
            ("Hybrid", 0.6),  # May surface thematic keywords
            ("Naive RAG", 0.4),  # Limited thematic understanding
        ],
        QueryType.COMPARATIVE: [
            ("Hybrid", 1.0),  # Good at diverse retrieval
            ("LightRAG", 0.9),  # Can compare via graph
            ("GraphRAG Local", 0.8),  # Entity comparison
            ("Naive RAG", 0.7),  # Basic comparison possible
        ],
        QueryType.TEMPORAL: [
            ("Naive RAG", 1.0),  # If metadata has timestamps
            ("Hybrid", 0.9),  # Can match temporal keywords
            ("LightRAG", 0.7),  # Graph can encode sequences
            ("GraphRAG Local", 0.6),  # Entity-level temporal
        ],
    }

    def __init__(self, available_strategies: List[str], max_strategies: int = 3):
        """Initialize strategy selector.

        Args:
            available_strategies: List of strategy names available for selection
            max_strategies: Maximum number of strategies to select per query (default: 3)
        """
        self.available_strategies = set(available_strategies)
        self.max_strategies = max_strategies

        logger.info(
            f"Initialized StrategySelector with {len(self.available_strategies)} "
            f"strategies, max_strategies={max_strategies}"
        )

    def select(self, query_type: QueryType, top_k: int = None) -> StrategySelection:
        """Select optimal strategies for a query type.

        Args:
            query_type: Classified query type
            top_k: Number of strategies to select (default: self.max_strategies)

        Returns:
            StrategySelection with strategies, weights, and reasoning
        """
        if top_k is None:
            top_k = self.max_strategies

        # Get strategy support for this query type
        supported = self.STRATEGY_SUPPORT.get(query_type, [])

        # Filter to available strategies only
        available_supported = [
            (name, weight)
            for name, weight in supported
            if name in self.available_strategies
        ]

        if not available_supported:
            # Fallback: use all available strategies with equal weight
            logger.warning(
                f"No supported strategies available for {query_type.value}, "
                f"using all available strategies"
            )
            strategies = list(self.available_strategies)[: top_k]
            weights = [1.0 / len(strategies)] * len(strategies)
            reasoning = f"No specific strategy support defined, using available strategies"

            return StrategySelection(
                strategies=strategies, weights=weights, reasoning=reasoning
            )

        # Select top-k strategies
        selected = available_supported[:top_k]
        strategies = [name for name, _ in selected]
        raw_weights = [weight for _, weight in selected]

        # Normalize weights to sum to 1.0
        total_weight = sum(raw_weights)
        normalized_weights = [w / total_weight for w in raw_weights]

        # Generate reasoning
        strategy_list = ", ".join(
            [f"{name} ({w:.2f})" for name, w in zip(strategies, normalized_weights)]
        )
        reasoning = f"Selected for {query_type.value} queries: {strategy_list}"

        logger.debug(f"Selected {len(strategies)} strategies for {query_type.value}: {strategies}")

        return StrategySelection(
            strategies=strategies, weights=normalized_weights, reasoning=reasoning
        )

    def get_single_best(self, query_type: QueryType) -> str:
        """Get the single best strategy for a query type.

        Args:
            query_type: Classified query type

        Returns:
            Name of the best strategy for this query type
        """
        selection = self.select(query_type, top_k=1)
        return selection.strategies[0] if selection.strategies else list(self.available_strategies)[0]
