"""Utility functions for retrieval strategies."""

from typing import List


def normalize_scores(scores: List[float], method: str = "minmax") -> List[float]:
    """Normalize scores to [0, 1] range for cross-strategy comparison.

    Args:
        scores: Raw scores from a strategy
        method: Normalization method ('minmax', 'softmax', or 'sigmoid')

    Returns:
        Normalized scores in [0, 1] range
    """
    if not scores:
        return []

    if method == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same - return uniform
            return [1.0] * len(scores)

        return [(s - min_score) / (max_score - min_score) for s in scores]

    elif method == "softmax":
        # Softmax: exp(x) / sum(exp(x))
        import math
        exp_scores = [math.exp(min(s, 700)) for s in scores]  # Cap at 700 to avoid overflow
        sum_exp = sum(exp_scores)
        return [e / sum_exp for e in exp_scores]

    elif method == "sigmoid":
        # Sigmoid: 1 / (1 + exp(-x))
        import math
        return [1 / (1 + math.exp(-min(max(s, -700), 700))) for s in scores]

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def rank_normalize_scores(scores: List[float]) -> List[float]:
    """Rank-based normalization: score = 1 - (rank / n).

    Useful when absolute score magnitudes don't matter, only relative ranking.

    Args:
        scores: Raw scores from a strategy

    Returns:
        Normalized scores based on rank
    """
    if not scores:
        return []

    n = len(scores)

    # Create (score, original_index) pairs
    scored_pairs = [(s, i) for i, s in enumerate(scores)]

    # Sort by score (descending)
    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    # Assign rank-based scores
    normalized = [0.0] * n
    for rank, (_, orig_idx) in enumerate(scored_pairs):
        normalized[orig_idx] = 1.0 - (rank / n)

    return normalized
