"""Query routing and response synthesis package.

This package provides intelligent query routing that automatically:
1. Classifies queries into types (FACTOID, EXPLORATORY, RELATIONAL, etc.)
2. Selects optimal retrieval strategy(ies) for each query type
3. Executes multiple strategies in parallel when beneficial
4. Fuses results with deduplication and score normalization
5. Synthesizes coherent answers using LLM

Usage:
    from graphunified.query import QueryRouter, RouterResult

    router = await QueryRouter.from_config(config, strategies, llm_config, embedding_config)
    result = await router.route("What is GraphRAG?", top_k=10)
    print(result.answer)
"""

from graphunified.query.router import QueryRouter, RouterResult

__all__ = ["QueryRouter", "RouterResult"]
