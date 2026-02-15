"""Example usage of the Query Router.

This script demonstrates how to use the QueryRouter with different configurations.
"""

import asyncio
from pathlib import Path

from graphunified.config.settings import Settings
from graphunified.query import QueryRouter
from graphunified.strategies.base import QueryType


async def example_basic_routing():
    """Example: Basic single-strategy routing."""
    print("=" * 80)
    print("Example 1: Basic Single-Strategy Routing")
    print("=" * 80)

    # Load configuration
    config_path = Path("config.yaml")
    settings = Settings.load(config_path)

    # Initialize strategies (simplified - in practice, initialize all strategies)
    # strategies = {
    #     "Naive RAG": await NaiveRAGStrategy.from_config(...),
    #     "Hybrid": await HybridStrategy.from_config(...),
    #     # ... etc
    # }
    strategies = {}  # Placeholder

    # Create router with rule-based classifier, single-strategy mode
    router_config = settings.query.router
    router_config.classifier.mode = "rule_based"
    router_config.multi_strategy_enabled = False

    router = await QueryRouter.from_config(
        config=router_config,
        strategies=strategies,
        llm_config=settings.llm,
        embedding_config=settings.embedding,
    )

    # Route a factoid query
    query = "What is GraphRAG?"
    result = await router.route(query, top_k=10)

    # Print results
    print(f"\nQuery: {query}")
    print(f"Classified as: {result.query_type.value}")
    print(f"Confidence: {result.classification_confidence:.2f}")
    print(f"Strategy used: {result.strategies_used[0]}")
    print(f"Chunks retrieved: {len(result.chunks)}")
    print(f"Total time: {result.total_time_ms:.0f}ms")
    print(f"  - Classification: {result.classification_time_ms:.0f}ms")
    print(f"  - Retrieval: {result.retrieval_time_ms:.0f}ms")
    print(f"LLM tokens used: {result.total_llm_tokens}")
    print(f"\nAnswer:\n{result.answer[:500]}...")


async def example_multi_strategy_routing():
    """Example: Multi-strategy routing with fusion."""
    print("\n" + "=" * 80)
    print("Example 2: Multi-Strategy Routing with Fusion")
    print("=" * 80)

    # Load configuration
    config_path = Path("config.yaml")
    settings = Settings.load(config_path)

    # Initialize strategies (placeholder)
    strategies = {}

    # Create router with multi-strategy enabled
    router_config = settings.query.router
    router_config.classifier.mode = "rule_based"
    router_config.multi_strategy_enabled = True
    router_config.max_strategies_per_query = 3
    router_config.fusion.method = "rrf"

    router = await QueryRouter.from_config(
        config=router_config,
        strategies=strategies,
        llm_config=settings.llm,
        embedding_config=settings.embedding,
    )

    # Route a relational query
    query = "How does GraphRAG relate to LightRAG?"
    result = await router.route(query, top_k=10)

    # Print results
    print(f"\nQuery: {query}")
    print(f"Classified as: {result.query_type.value}")
    print(f"Strategies used: {', '.join(result.strategies_used)}")
    print(f"Strategy weights: {result.strategy_weights}")
    print(f"Chunks retrieved: {len(result.chunks)}")
    print(f"Total time: {result.total_time_ms:.0f}ms")
    print(f"  - Retrieval: {result.retrieval_time_ms:.0f}ms")
    print(f"  - Fusion: ~{result.retrieval_time_ms * 0.02:.0f}ms")
    print(f"  - Synthesis: {result.synthesis_time_ms:.0f}ms")
    print(f"\nFusion metadata:")
    print(f"  Method: {result.metadata.get('fusion_method', 'N/A')}")
    print(f"  Execution errors: {result.metadata.get('execution_errors', {})}")


async def example_hybrid_classifier():
    """Example: Hybrid classifier with LLM fallback."""
    print("\n" + "=" * 80)
    print("Example 3: Hybrid Classifier (Rule-based → LLM Fallback)")
    print("=" * 80)

    # Load configuration
    config_path = Path("config.yaml")
    settings = Settings.load(config_path)

    # Initialize strategies (placeholder)
    strategies = {}

    # Create router with hybrid classifier
    router_config = settings.query.router
    router_config.classifier.mode = "hybrid"
    router_config.classifier.confidence_threshold = 0.7
    router_config.multi_strategy_enabled = False

    router = await QueryRouter.from_config(
        config=router_config,
        strategies=strategies,
        llm_config=settings.llm,
        embedding_config=settings.embedding,
    )

    # Test with an ambiguous query
    query = "Tell me about the system"
    result = await router.route(query, top_k=10)

    # Print results
    print(f"\nQuery: {query}")
    print(f"Classification method: {result.metadata['classification_method']}")
    print(f"Classified as: {result.query_type.value}")
    print(f"Confidence: {result.classification_confidence:.2f}")
    print(f"Reasoning: {result.metadata['classification_reasoning']}")
    print(f"LLM tokens (classification): {result.total_llm_tokens}")


async def example_fallback_chain():
    """Example: Confidence-based fallback chain."""
    print("\n" + "=" * 80)
    print("Example 4: Confidence-Based Fallback Chain")
    print("=" * 80)

    # Load configuration
    config_path = Path("config.yaml")
    settings = Settings.load(config_path)

    # Initialize strategies (placeholder)
    strategies = {}

    # Create router with fallback enabled
    router_config = settings.query.router
    router_config.classifier.mode = "rule_based"
    router_config.fallback_enabled = True
    router_config.fallback_confidence_threshold = 0.5

    router = await QueryRouter.from_config(
        config=router_config,
        strategies=strategies,
        llm_config=settings.llm,
        embedding_config=settings.embedding,
    )

    # Test with a vague query (no clear keywords)
    query = "stuff about things"
    result = await router.route(query, top_k=10)

    # Print results
    print(f"\nQuery: {query}")
    print(f"Fallback triggered: {result.metadata.get('fallback_triggered', False)}")
    if result.metadata.get('fallback_triggered'):
        print(f"Original query type: {result.metadata['original_query_type']}")
        print(f"Original confidence: {result.metadata['original_confidence']:.2f}")
        print(f"Fallback query type: {result.query_type.value}")


async def example_testing_overrides():
    """Example: Testing with overrides."""
    print("\n" + "=" * 80)
    print("Example 5: Testing with Overrides")
    print("=" * 80)

    # Load configuration
    config_path = Path("config.yaml")
    settings = Settings.load(config_path)

    # Initialize strategies (placeholder)
    strategies = {}

    router = await QueryRouter.from_config(
        config=settings.query.router,
        strategies=strategies,
        llm_config=settings.llm,
        embedding_config=settings.embedding,
    )

    query = "What is machine learning?"

    # Test 1: Force query type
    print("\n--- Test 1: Force Query Type ---")
    result = await router.route(query, force_query_type=QueryType.COMPARATIVE)
    print(f"Forced type: COMPARATIVE")
    print(f"Classification reasoning: {result.metadata['classification_reasoning']}")

    # Test 2: Force strategy
    print("\n--- Test 2: Force Strategy ---")
    result = await router.route(query, force_strategy="Hybrid")
    print(f"Forced strategy: Hybrid")
    print(f"Strategies used: {result.strategies_used}")

    # Test 3: Disable synthesis
    print("\n--- Test 3: Disable Synthesis ---")
    router.config.response_synthesis_enabled = False
    result = await router.route(query, top_k=3)
    print(f"Synthesis disabled")
    print(f"Synthesis time: {result.synthesis_time_ms:.0f}ms (should be 0)")
    print(f"Answer format: {'Raw chunks' if '[Source' in result.answer else 'Synthesized'}")


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Query Router Examples" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\nNOTE: These examples require initialized strategies.")
    print("      They are provided for demonstration purposes.\n")

    # Uncomment to run examples (requires configured system)
    # await example_basic_routing()
    # await example_multi_strategy_routing()
    # await example_hybrid_classifier()
    # await example_fallback_chain()
    # await example_testing_overrides()

    print("\nTo run these examples:")
    print("1. Ensure your config.yaml is properly configured")
    print("2. Initialize all required strategies")
    print("3. Uncomment the example functions in main()")
    print("4. Run: python examples/query_router_example.py\n")


if __name__ == "__main__":
    asyncio.run(main())
