"""Result fusion module.

Fuses retrieval results from multiple strategies using RRF, weighted, or rank fusion.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple
from uuid import UUID

from graphunified.strategies.base import RetrievalResult, Chunk
from graphunified.strategies.utils import normalize_scores, rank_normalize_scores

logger = logging.getLogger(__name__)


@dataclass
class FusedResult:
    """Result of fusing multiple strategy results.

    Attributes:
        chunks: Fused and deduplicated chunks (sorted by score)
        scores: Fused scores (parallel to chunks)
        source_strategies: Strategy names contributing to each chunk
        metadata: Fusion metadata (method, original_count, etc.)
    """

    chunks: List[Chunk] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    source_strategies: List[List[str]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class ResultFusion:
    """Fuses retrieval results from multiple strategies.

    Supports three fusion algorithms:
    1. RRF (Reciprocal Rank Fusion) - Standard multi-source fusion
    2. Weighted Fusion - Weighted average of normalized scores
    3. Rank Fusion - Position-based fusion (Borda count variant)
    """

    def __init__(
        self,
        method: Literal["rrf", "weighted", "rank"] = "rrf",
        rrf_k: int = 60,
    ):
        """Initialize result fusion.

        Args:
            method: Fusion method (rrf, weighted, rank)
            rrf_k: Constant for RRF formula (default: 60)
        """
        self.method = method
        self.rrf_k = rrf_k

    def fuse(
        self,
        results: Dict[str, RetrievalResult],
        weights: List[float],
        top_k: int,
    ) -> FusedResult:
        """Fuse results from multiple strategies.

        Args:
            results: Retrieved results {strategy_name: RetrievalResult}
            weights: Strategy weights (parallel to strategy names)
            top_k: Number of chunks to return

        Returns:
            FusedResult with deduplicated chunks and scores
        """
        if not results:
            logger.warning("No results to fuse")
            return FusedResult()

        strategy_names = list(results.keys())

        # Build weight map
        weight_map = {name: w for name, w in zip(strategy_names, weights)}

        logger.info(
            f"Fusing {len(results)} strategy results using {self.method} "
            f"(weights: {weight_map})"
        )

        # Apply fusion algorithm
        if self.method == "rrf":
            fused = self._rrf_fusion(results, weight_map)
        elif self.method == "weighted":
            fused = self._weighted_fusion(results, weight_map)
        elif self.method == "rank":
            fused = self._rank_fusion(results, weight_map)
        else:
            raise ValueError(f"Unknown fusion method: {self.method}")

        # Sort by fused score (descending) and take top_k
        scored_chunks = list(zip(fused["chunks"], fused["scores"], fused["sources"]))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        scored_chunks = scored_chunks[:top_k]

        # Unzip
        chunks, scores, sources = zip(*scored_chunks) if scored_chunks else ([], [], [])

        logger.info(
            f"Fusion complete: {len(chunks)} chunks from {len(results)} strategies "
            f"(original total: {sum(len(r.chunks) for r in results.values())})"
        )

        return FusedResult(
            chunks=list(chunks),
            scores=list(scores),
            source_strategies=list(sources),
            metadata={
                "method": self.method,
                "original_count": sum(len(r.chunks) for r in results.values()),
                "deduplicated_count": len(fused["chunks"]),
                "strategy_weights": weight_map,
            },
        )

    def _rrf_fusion(
        self, results: Dict[str, RetrievalResult], weights: Dict[str, float]
    ) -> Dict:
        """Reciprocal Rank Fusion (RRF).

        Formula: score = sum(weight / (k + rank) for each strategy)
        where k = self.rrf_k (typically 60)

        Args:
            results: Strategy results
            weights: Strategy weights

        Returns:
            Dict with chunks, scores, and source lists
        """
        # Build chunk -> (strategies, ranks) map
        chunk_data: Dict[UUID, Tuple[Chunk, List[str], List[int]]] = {}

        for strategy_name, result in results.items():
            for rank, chunk in enumerate(result.chunks):
                if chunk.id not in chunk_data:
                    chunk_data[chunk.id] = (chunk, [], [])

                _, strategies, ranks = chunk_data[chunk.id]
                strategies.append(strategy_name)
                ranks.append(rank)

        # Calculate RRF scores
        fused_chunks = []
        fused_scores = []
        fused_sources = []

        for chunk_id, (chunk, strategies, ranks) in chunk_data.items():
            # RRF formula: sum(weight / (k + rank))
            rrf_score = sum(
                weights[strategy] / (self.rrf_k + rank)
                for strategy, rank in zip(strategies, ranks)
            )

            fused_chunks.append(chunk)
            fused_scores.append(rrf_score)
            fused_sources.append(strategies)

        return {
            "chunks": fused_chunks,
            "scores": fused_scores,
            "sources": fused_sources,
        }

    def _weighted_fusion(
        self, results: Dict[str, RetrievalResult], weights: Dict[str, float]
    ) -> Dict:
        """Weighted fusion using normalized scores.

        Formula: score = sum(normalized_score * weight for each strategy)

        Args:
            results: Strategy results
            weights: Strategy weights

        Returns:
            Dict with chunks, scores, and source lists
        """
        # Normalize scores for each strategy
        normalized_results = {}
        for strategy_name, result in results.items():
            norm_scores = normalize_scores(result.scores, method="minmax")
            normalized_results[strategy_name] = (result, norm_scores)

        # Build chunk -> (strategies, normalized_scores) map
        chunk_data: Dict[UUID, Tuple[Chunk, List[str], List[float]]] = {}

        for strategy_name, (result, norm_scores) in normalized_results.items():
            for chunk, norm_score in zip(result.chunks, norm_scores):
                if chunk.id not in chunk_data:
                    chunk_data[chunk.id] = (chunk, [], [])

                _, strategies, scores = chunk_data[chunk.id]
                strategies.append(strategy_name)
                scores.append(norm_score)

        # Calculate weighted scores
        fused_chunks = []
        fused_scores = []
        fused_sources = []

        for chunk_id, (chunk, strategies, scores) in chunk_data.items():
            # Weighted average
            weighted_score = sum(
                score * weights[strategy] for strategy, score in zip(strategies, scores)
            )

            fused_chunks.append(chunk)
            fused_scores.append(weighted_score)
            fused_sources.append(strategies)

        return {
            "chunks": fused_chunks,
            "scores": fused_scores,
            "sources": fused_sources,
        }

    def _rank_fusion(
        self, results: Dict[str, RetrievalResult], weights: Dict[str, float]
    ) -> Dict:
        """Rank-based fusion (Borda count variant).

        Formula: score = sum(rank_normalized_score * weight for each strategy)

        Args:
            results: Strategy results
            weights: Strategy weights

        Returns:
            Dict with chunks, scores, and source lists
        """
        # Rank-normalize scores for each strategy
        rank_normalized_results = {}
        for strategy_name, result in results.items():
            rank_scores = rank_normalize_scores(result.scores)
            rank_normalized_results[strategy_name] = (result, rank_scores)

        # Build chunk -> (strategies, rank_scores) map
        chunk_data: Dict[UUID, Tuple[Chunk, List[str], List[float]]] = {}

        for strategy_name, (result, rank_scores) in rank_normalized_results.items():
            for chunk, rank_score in zip(result.chunks, rank_scores):
                if chunk.id not in chunk_data:
                    chunk_data[chunk.id] = (chunk, [], [])

                _, strategies, scores = chunk_data[chunk.id]
                strategies.append(strategy_name)
                scores.append(rank_score)

        # Calculate rank-weighted scores
        fused_chunks = []
        fused_scores = []
        fused_sources = []

        for chunk_id, (chunk, strategies, scores) in chunk_data.items():
            # Weighted sum of rank scores
            rank_weighted_score = sum(
                score * weights[strategy] for strategy, score in zip(strategies, scores)
            )

            fused_chunks.append(chunk)
            fused_scores.append(rank_weighted_score)
            fused_sources.append(strategies)

        return {
            "chunks": fused_chunks,
            "scores": fused_scores,
            "sources": fused_sources,
        }
