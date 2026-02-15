"""Unit test for score normalization utilities."""

from graphunified.strategies.utils import normalize_scores, rank_normalize_scores


def test_normalize_scores():
    """Test score normalization methods."""

    print("=" * 80)
    print("Score Normalization Unit Tests")
    print("=" * 80)

    # Test Case 1: Different score ranges (simulating different strategies)
    print("\n\nTest Case 1: Cross-Strategy Score Comparison")
    print("-" * 80)

    # Simulated strategy scores (different ranges)
    naive_scores = [0.85, 0.82, 0.78, 0.75, 0.70]  # Range: [0.70, 0.85]
    hybrid_scores = [0.025, 0.020, 0.018, 0.015, 0.012]  # Range: [0.012, 0.025]
    graphrag_scores = [0.60, 0.55, 0.50, 0.45, 0.40]  # Range: [0.40, 0.60]

    print("Raw Scores (NOT comparable across strategies):")
    print(f"  Naive RAG:      {naive_scores}")
    print(f"  Hybrid RAG:     {hybrid_scores}")
    print(f"  GraphRAG Local: {graphrag_scores}")

    # Normalize with MinMax
    print("\nMinMax Normalized ([0, 1] range):")
    naive_norm = normalize_scores(naive_scores, method="minmax")
    hybrid_norm = normalize_scores(hybrid_scores, method="minmax")
    graphrag_norm = normalize_scores(graphrag_scores, method="minmax")

    print(f"  Naive RAG:      {[f'{s:.3f}' for s in naive_norm]}")
    print(f"  Hybrid RAG:     {[f'{s:.3f}' for s in hybrid_norm]}")
    print(f"  GraphRAG Local: {[f'{s:.3f}' for s in graphrag_norm]}")

    # Normalize with Rank
    print("\nRank Normalized (position-based):")
    naive_rank = rank_normalize_scores(naive_scores)
    hybrid_rank = rank_normalize_scores(hybrid_scores)
    graphrag_rank = rank_normalize_scores(graphrag_scores)

    print(f"  Naive RAG:      {[f'{s:.3f}' for s in naive_rank]}")
    print(f"  Hybrid RAG:     {[f'{s:.3f}' for s in hybrid_rank]}")
    print(f"  GraphRAG Local: {[f'{s:.3f}' for s in graphrag_rank]}")

    # Test Case 2: Edge Cases
    print("\n\nTest Case 2: Edge Cases")
    print("-" * 80)

    # Empty list
    empty = normalize_scores([], method="minmax")
    print(f"Empty list: {empty}")
    assert empty == [], "Empty list should return empty"

    # Single score
    single = normalize_scores([0.5], method="minmax")
    print(f"Single score: {single}")
    assert single == [1.0], "Single score should normalize to 1.0"

    # All same scores
    same = normalize_scores([0.5, 0.5, 0.5], method="minmax")
    print(f"All same: {same}")
    assert same == [1.0, 1.0, 1.0], "Same scores should all be 1.0"

    # Test Case 3: Validation
    print("\n\nTest Case 3: Validation Checks")
    print("-" * 80)

    test_scores = [0.9, 0.8, 0.7, 0.6, 0.5]

    # MinMax should always produce [0, 1] range
    norm = normalize_scores(test_scores, method="minmax")
    assert min(norm) == 0.0, f"Min should be 0.0, got {min(norm)}"
    assert max(norm) == 1.0, f"Max should be 1.0, got {max(norm)}"
    print(f"✓ MinMax produces [0, 1] range: min={min(norm):.3f}, max={max(norm):.3f}")

    # Rank should preserve order
    rank = rank_normalize_scores(test_scores)
    for i in range(len(rank) - 1):
        assert rank[i] >= rank[i + 1], f"Rank order violated at {i}"
    print(f"✓ Rank normalization preserves order")

    # Rank should produce [0, 1] range
    assert 0.0 <= min(rank) <= 1.0, f"Rank min out of range: {min(rank)}"
    assert 0.0 <= max(rank) <= 1.0, f"Rank max out of range: {max(rank)}"
    print(f"✓ Rank produces [0, 1] range: min={min(rank):.3f}, max={max(rank):.3f}")

    print("\n\n" + "=" * 80)
    print("✅ All Tests Passed!")
    print("=" * 80)

    print("\nKey Takeaways:")
    print("1. MinMax normalization brings all scores to [0, 1] with same relative order")
    print("2. Rank normalization makes top result always 1.0 across strategies")
    print("3. Use MinMax for preserving score magnitudes, Rank for pure ordering")
    print("4. Normalized scores enable fair cross-strategy comparison")


if __name__ == "__main__":
    test_normalize_scores()
