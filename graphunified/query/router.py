"""Main query router orchestrator.

Coordinates classification, strategy selection, execution, fusion, and synthesis.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from graphunified.strategies.base import QueryType, RetrievalResult, RetrievalStrategy, Chunk
from graphunified.query.classifier import QueryClassifier, create_classifier, ClassificationResult
from graphunified.query.selector import StrategySelector, StrategySelection
from graphunified.query.executor import MultiStrategyExecutor, ExecutionResult
from graphunified.query.fusion import ResultFusion, FusedResult
from graphunified.query.generator import ResponseGenerator, GeneratedResponse
from graphunified.config.settings import QueryRouterConfig, LLMConfig, EmbeddingConfig

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    """Complete query routing result.

    Attributes:
        answer: Synthesized answer or raw chunks (depending on synthesis_enabled)
        query: Original query text
        query_type: Classified query type
        classification_confidence: Confidence of classification
        strategies_used: List of strategy names used (in execution order)
        strategy_weights: Weights used for fusion (parallel to strategies_used)
        chunks: Retrieved chunks (fused if multi-strategy)
        scores: Relevance scores (parallel to chunks)
        retrieval_results: Raw results from each strategy
        total_time_ms: Total routing time in milliseconds
        classification_time_ms: Time spent on classification
        selection_time_ms: Time spent on strategy selection
        retrieval_time_ms: Time spent on retrieval
        synthesis_time_ms: Time spent on synthesis (0 if disabled)
        total_llm_tokens: Total LLM tokens used (classification + synthesis)
        metadata: Additional metadata
    """

    answer: str
    query: str
    query_type: QueryType
    classification_confidence: float
    strategies_used: List[str]
    strategy_weights: List[float]
    chunks: List[Chunk] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    retrieval_results: Dict[str, RetrievalResult] = field(default_factory=dict)
    total_time_ms: float = 0.0
    classification_time_ms: float = 0.0
    selection_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    synthesis_time_ms: float = 0.0
    total_llm_tokens: int = 0
    metadata: Dict = field(default_factory=dict)


class QueryRouter:
    """Main query router orchestrator.

    Coordinates the complete query routing pipeline:
    1. Classification: Determine query type
    2. Selection: Choose optimal strategy(ies)
    3. Execution: Execute selected strategy (Phase 1: single strategy only)
    4. Fusion: Merge results (Phase 2+)
    5. Synthesis: Generate answer (Phase 2+)
    """

    def __init__(
        self,
        config: QueryRouterConfig,
        strategies: Dict[str, RetrievalStrategy],
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
    ):
        """Initialize query router.

        Args:
            config: Query router configuration
            strategies: Dictionary of initialized retrieval strategies
            llm_config: LLM configuration
            embedding_config: Embedding configuration
        """
        self.config = config
        self.strategies = strategies
        self.llm_config = llm_config
        self.embedding_config = embedding_config

        # Initialize classifier
        self.classifier = create_classifier(
            mode=config.classifier.mode,
            llm_config=llm_config if config.classifier.mode != "rule_based" else None,
            confidence_threshold=config.classifier.confidence_threshold,
        )

        # Initialize strategy selector
        available_strategies = list(strategies.keys())
        max_strategies = 1 if not config.multi_strategy_enabled else config.max_strategies_per_query
        self.selector = StrategySelector(
            available_strategies=available_strategies,
            max_strategies=max_strategies,
        )

        # Initialize executor (Phase 2)
        self.executor = MultiStrategyExecutor(timeout=30.0)

        # Initialize fusion (Phase 2)
        self.fusion = ResultFusion(
            method=config.fusion.method, rrf_k=config.fusion.rrf_k
        )

        # Initialize generator (Phase 2)
        if config.response_synthesis_enabled:
            self.generator = ResponseGenerator(
                llm_config=llm_config, temperature=config.synthesis_temperature
            )
        else:
            self.generator = None

        logger.info(
            f"Initialized QueryRouter with {len(strategies)} strategies, "
            f"classifier={config.classifier.mode}, "
            f"multi_strategy={config.multi_strategy_enabled}, "
            f"synthesis={config.response_synthesis_enabled}"
        )

    @classmethod
    async def from_config(
        cls,
        config: QueryRouterConfig,
        strategies: Dict[str, RetrievalStrategy],
        llm_config: LLMConfig,
        embedding_config: EmbeddingConfig,
    ) -> "QueryRouter":
        """Create router from configuration.

        Args:
            config: Query router configuration
            strategies: Dictionary of initialized retrieval strategies
            llm_config: LLM configuration
            embedding_config: Embedding configuration

        Returns:
            Initialized QueryRouter
        """
        return cls(
            config=config,
            strategies=strategies,
            llm_config=llm_config,
            embedding_config=embedding_config,
        )

    async def route(
        self,
        query: str,
        top_k: int = 10,
        force_query_type: Optional[QueryType] = None,
        force_strategy: Optional[str] = None,
        _retry_with_fallback: bool = True,  # Internal flag for recursion control
    ) -> RouterResult:
        """Route a query through the complete pipeline.

        Supports both single-strategy and multi-strategy execution with fusion,
        plus confidence-based fallback chains (Phase 3).

        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            force_query_type: Override classification (for testing)
            force_strategy: Override strategy selection (for testing)
            _retry_with_fallback: Internal flag to prevent infinite recursion

        Returns:
            Complete RouterResult with answer and metadata
        """
        start_time = time.time()
        total_tokens = 0

        # Step 1: Classify query
        classify_start = time.time()
        if force_query_type:
            classification = ClassificationResult(
                query_type=force_query_type,
                confidence=1.0,
                reasoning="Forced by user override",
                method="override",
            )
        else:
            classification = await self.classifier.classify(query)

        classification_time_ms = (time.time() - classify_start) * 1000
        logger.info(
            f"Classified query as {classification.query_type.value} "
            f"(confidence: {classification.confidence:.2f}, method: {classification.method})"
        )

        # Step 2: Select strategy(ies)
        select_start = time.time()
        if force_strategy:
            if force_strategy not in self.strategies:
                raise ValueError(f"Strategy {force_strategy} not available")
            selection = StrategySelection(
                strategies=[force_strategy],
                weights=[1.0],
                reasoning="Forced by user override",
            )
        else:
            # Select based on config: single or multi-strategy
            num_strategies = 1 if not self.config.multi_strategy_enabled else None
            selection = self.selector.select(classification.query_type, top_k=num_strategies)

        selection_time_ms = (time.time() - select_start) * 1000
        logger.info(f"Selected {len(selection.strategies)} strategies: {selection.strategies}")

        # Step 3: Execute strategy(ies)
        retrieve_start = time.time()

        if len(selection.strategies) == 1:
            # Single-strategy execution
            execution_result = await self._execute_single(
                query, selection.strategies[0], top_k
            )
            final_chunks = execution_result.results[selection.strategies[0]].chunks if execution_result.results else []
            final_scores = execution_result.results[selection.strategies[0]].scores if execution_result.results else []

        else:
            # Multi-strategy execution with fusion
            execution_result = await self.executor.execute(
                query, self.strategies, selection.strategies, top_k
            )

            # Fuse results
            if execution_result.results:
                fused = self.fusion.fuse(
                    execution_result.results, selection.weights, top_k
                )
                final_chunks = fused.chunks
                final_scores = fused.scores
            else:
                final_chunks = []
                final_scores = []

        retrieval_time_ms = (time.time() - retrieve_start) * 1000

        logger.info(
            f"Retrieved {len(final_chunks)} chunks in {retrieval_time_ms:.0f}ms "
            f"({len(selection.strategies)} strategies, "
            f"{len(execution_result.results)} successful, {len(execution_result.errors)} failed)"
        )

        # Step 4: Generate answer
        synthesis_start = time.time()
        if self.config.response_synthesis_enabled and self.generator and final_chunks:
            generated = await self.generator.generate(
                query, final_chunks, classification.query_type
            )
            answer = generated.answer
            total_tokens += generated.input_tokens + generated.output_tokens
        else:
            answer = self._format_raw_chunks(final_chunks)

        synthesis_time_ms = (time.time() - synthesis_start) * 1000

        # Calculate total time
        total_time_ms = (time.time() - start_time) * 1000

        # Build result
        router_result = RouterResult(
            answer=answer,
            query=query,
            query_type=classification.query_type,
            classification_confidence=classification.confidence,
            strategies_used=selection.strategies,
            strategy_weights=selection.weights,
            chunks=final_chunks,
            scores=final_scores,
            retrieval_results=execution_result.results,
            total_time_ms=total_time_ms,
            classification_time_ms=classification_time_ms,
            selection_time_ms=selection_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            synthesis_time_ms=synthesis_time_ms,
            total_llm_tokens=total_tokens,
            metadata={
                "classification_method": classification.method,
                "classification_reasoning": classification.reasoning,
                "selection_reasoning": selection.reasoning,
                "num_strategies": len(selection.strategies),
                "execution_errors": {k: str(v) for k, v in execution_result.errors.items()},
            },
        )

        logger.info(
            f"Query routing completed in {total_time_ms:.0f}ms "
            f"({len(final_chunks)} chunks, {total_tokens} tokens)"
        )

        # Step 5: Fallback logic (Phase 3)
        # If no results and low confidence, retry with EXPLORATORY fallback
        if (
            self.config.fallback_enabled
            and _retry_with_fallback
            and not final_chunks
            and classification.confidence < self.config.fallback_confidence_threshold
            and classification.query_type != QueryType.EXPLORATORY
        ):
            logger.warning(
                f"No results with low confidence ({classification.confidence:.2f} "
                f"< {self.config.fallback_confidence_threshold}), "
                f"retrying with EXPLORATORY fallback"
            )

            # Retry with EXPLORATORY query type (no further recursion)
            fallback_result = await self.route(
                query,
                top_k=top_k,
                force_query_type=QueryType.EXPLORATORY,
                force_strategy=force_strategy,
                _retry_with_fallback=False,  # Prevent infinite recursion
            )

            # Add fallback metadata
            fallback_result.metadata["fallback_triggered"] = True
            fallback_result.metadata["original_query_type"] = classification.query_type.value
            fallback_result.metadata["original_confidence"] = classification.confidence

            return fallback_result

        return router_result

    async def _execute_single(
        self, query: str, strategy_name: str, top_k: int
    ) -> ExecutionResult:
        """Execute a single strategy (helper for route()).

        Args:
            query: Query text
            strategy_name: Strategy to execute
            top_k: Number of chunks

        Returns:
            ExecutionResult with single strategy result
        """
        strategy = self.strategies[strategy_name]

        try:
            result = await strategy.retrieve(query, top_k=top_k)
            return ExecutionResult(results={strategy_name: result})

        except Exception as e:
            logger.error(f"Strategy {strategy_name} failed: {e}")
            return ExecutionResult(errors={strategy_name: e})

    def _format_raw_chunks(self, chunks: List[Chunk]) -> str:
        """Format retrieved chunks as raw text (no LLM synthesis).

        Args:
            chunks: Retrieved chunks

        Returns:
            Formatted string with all chunk text
        """
        if not chunks:
            return "No relevant information found."

        # Format chunks with source numbers
        formatted_parts = []
        for i, chunk in enumerate(chunks, start=1):
            formatted_parts.append(f"[Source {i}]:\n{chunk.text}\n")

        return "\n".join(formatted_parts)
