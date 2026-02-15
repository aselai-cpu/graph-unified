"""Multi-strategy parallel execution module.

Executes multiple retrieval strategies concurrently with timeout handling.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List

from graphunified.strategies.base import RetrievalStrategy, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of multi-strategy execution.

    Attributes:
        results: Successfully completed results {strategy_name: RetrievalResult}
        errors: Failed strategies {strategy_name: Exception}
        execution_time_ms: Total execution time in milliseconds
    """

    results: Dict[str, RetrievalResult] = field(default_factory=dict)
    errors: Dict[str, Exception] = field(default_factory=dict)
    execution_time_ms: float = 0.0


class MultiStrategyExecutor:
    """Executes multiple retrieval strategies in parallel.

    Uses asyncio to run strategies concurrently, with timeout handling
    for each individual strategy execution.
    """

    def __init__(self, timeout: float = 30.0):
        """Initialize multi-strategy executor.

        Args:
            timeout: Maximum time (in seconds) to wait for each strategy (default: 30.0)
        """
        self.timeout = timeout

    async def execute(
        self,
        query: str,
        strategies: Dict[str, RetrievalStrategy],
        strategy_names: List[str],
        top_k: int,
    ) -> ExecutionResult:
        """Execute multiple strategies in parallel.

        Args:
            query: Query text
            strategies: Dictionary of all available strategies
            strategy_names: Names of strategies to execute
            top_k: Number of chunks to retrieve per strategy

        Returns:
            ExecutionResult with successful results and errors
        """
        start_time = time.time()

        # Filter to requested strategies
        selected_strategies = {
            name: strategies[name] for name in strategy_names if name in strategies
        }

        if not selected_strategies:
            logger.warning(f"No valid strategies to execute: {strategy_names}")
            return ExecutionResult(execution_time_ms=(time.time() - start_time) * 1000)

        logger.info(f"Executing {len(selected_strategies)} strategies in parallel")

        # Create tasks for each strategy
        tasks = {
            name: asyncio.create_task(self._execute_single(name, strategy, query, top_k))
            for name, strategy in selected_strategies.items()
        }

        # Wait for all tasks to complete
        results = {}
        errors = {}

        for name, task in tasks.items():
            try:
                result = await asyncio.wait_for(task, timeout=self.timeout)
                results[name] = result
                logger.debug(
                    f"Strategy {name} completed: {len(result.chunks)} chunks "
                    f"in {result.retrieval_time_ms:.0f}ms"
                )

            except asyncio.TimeoutError:
                error = TimeoutError(f"Strategy {name} timed out after {self.timeout}s")
                errors[name] = error
                logger.warning(str(error))

            except Exception as e:
                errors[name] = e
                logger.error(f"Strategy {name} failed: {e}")

        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Multi-strategy execution completed in {execution_time_ms:.0f}ms: "
            f"{len(results)} successful, {len(errors)} failed"
        )

        return ExecutionResult(
            results=results, errors=errors, execution_time_ms=execution_time_ms
        )

    async def _execute_single(
        self,
        name: str,
        strategy: RetrievalStrategy,
        query: str,
        top_k: int,
    ) -> RetrievalResult:
        """Execute a single strategy with error handling.

        Args:
            name: Strategy name (for logging)
            strategy: Strategy instance
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            RetrievalResult from strategy

        Raises:
            Exception: Any exception from strategy execution
        """
        try:
            result = await strategy.retrieve(query, top_k=top_k)
            return result

        except Exception as e:
            logger.error(f"Error executing strategy {name}: {e}")
            raise
