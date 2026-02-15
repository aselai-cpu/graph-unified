"""Query classification module.

Classifies queries into types: FACTOID, EXPLORATORY, RELATIONAL, THEMATIC, COMPARATIVE, TEMPORAL.
Supports rule-based, LLM-based, and hybrid classification modes.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set

from graphunified.strategies.base import QueryType
from graphunified.config.settings import LLMConfig
from graphunified.utils.llm import ClaudeClient
from graphunified.utils.json_utils import parse_json_response
from graphunified.prompts.query import QUERY_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of query classification.

    Attributes:
        query_type: Classified query type
        confidence: Confidence score (0.0-1.0)
        reasoning: Explanation of classification decision
        method: Classification method used (rule_based, llm_based, hybrid)
    """

    query_type: QueryType
    confidence: float
    reasoning: str
    method: Literal["rule_based", "llm_based", "hybrid"]


class QueryClassifier(ABC):
    """Abstract base class for query classifiers."""

    @abstractmethod
    async def classify(self, query: str) -> ClassificationResult:
        """Classify a query into a QueryType.

        Args:
            query: Query text to classify

        Returns:
            Classification result with query_type, confidence, and reasoning
        """
        pass


class RuleBasedClassifier(QueryClassifier):
    """Rule-based classifier using keyword pattern matching.

    Inspired by LightRAG's query classification approach, this classifier
    uses weighted keyword matching to classify queries into types.
    """

    def __init__(self):
        """Initialize rule-based classifier with keyword patterns."""
        # Keyword patterns for each query type
        # Weights: position matters - earlier matches weighted higher
        self._patterns: Dict[QueryType, List[str]] = {
            QueryType.FACTOID: [
                "what is",
                "what are",
                "define",
                "definition of",
                "who is",
                "who are",
                "when did",
                "when was",
                "where is",
                "where are",
                "which",
                "how many",
                "how much",
                "name",
                "list",
            ],
            QueryType.EXPLORATORY: [
                "summarize",
                "summary of",
                "overview",
                "tell me about",
                "explain",
                "describe",
                "discuss",
                "what do you know",
                "general",
                "broad",
                "main",
                "key concepts",
                "introduction to",
            ],
            QueryType.RELATIONAL: [
                "how does",
                "how do",
                "relationship between",
                "relation between",
                "connection between",
                "linked to",
                "associated with",
                "related to",
                "interact",
                "interaction",
                "dependency",
                "depend on",
                "affects",
                "influences",
            ],
            QueryType.THEMATIC: [
                "themes",
                "patterns",
                "trends",
                "emerging",
                "common",
                "recurring",
                "overall",
                "across",
                "throughout",
                "in general",
                "typically",
                "usually",
            ],
            QueryType.COMPARATIVE: [
                "compare",
                "comparison",
                "difference between",
                "distinguish",
                "versus",
                "vs",
                "similar",
                "similarity",
                "contrast",
                "like",
                "unlike",
                "differ",
                "better",
                "worse",
            ],
            QueryType.TEMPORAL: [
                "timeline",
                "sequence",
                "history",
                "historical",
                "chronology",
                "before",
                "after",
                "then",
                "next",
                "previous",
                "followed by",
                "evolution",
                "development over",
                "progression",
            ],
        }

    async def classify(self, query: str) -> ClassificationResult:
        """Classify query using keyword pattern matching.

        Args:
            query: Query text to classify

        Returns:
            Classification result with query_type and confidence
        """
        query_lower = query.lower()

        # Score each query type based on keyword matches
        scores: Dict[QueryType, float] = {}
        for query_type, keywords in self._patterns.items():
            score = 0.0
            for idx, keyword in enumerate(keywords):
                if keyword in query_lower:
                    # Position-based weighting: earlier keywords weighted higher
                    position_weight = 1.0 / (1 + idx * 0.1)

                    # Start position weighting: keywords at start of query weighted higher
                    start_pos = query_lower.find(keyword)
                    start_weight = 1.0 / (1 + start_pos * 0.01)

                    score += position_weight * start_weight

            scores[query_type] = score

        # Find best match
        if not scores or max(scores.values()) == 0:
            # Default to EXPLORATORY for unclear queries
            return ClassificationResult(
                query_type=QueryType.EXPLORATORY,
                confidence=0.3,
                reasoning="No clear keyword matches found, defaulting to exploratory",
                method="rule_based",
            )

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        total_score = sum(scores.values())

        # Calculate confidence as normalized score
        confidence = min(best_score / total_score if total_score > 0 else 0.3, 0.95)

        # Generate reasoning
        matched_keywords = [
            kw for kw in self._patterns[best_type] if kw in query_lower
        ]
        reasoning = f"Matched keywords: {', '.join(matched_keywords[:3])}"

        logger.debug(f"Classified query as {best_type.value} with confidence {confidence:.2f}")

        return ClassificationResult(
            query_type=best_type,
            confidence=confidence,
            reasoning=reasoning,
            method="rule_based",
        )


class LLMBasedClassifier(QueryClassifier):
    """LLM-based classifier using Claude for query classification."""

    def __init__(self, llm_config: LLMConfig):
        """Initialize LLM-based classifier.

        Args:
            llm_config: LLM configuration
        """
        self.llm_client = ClaudeClient(config=llm_config)

    async def classify(self, query: str) -> ClassificationResult:
        """Classify query using LLM.

        Args:
            query: Query text to classify

        Returns:
            Classification result from LLM
        """
        # Generate classification prompt
        prompt = QUERY_CLASSIFICATION_PROMPT.format(query=query)

        # Call LLM
        response_text, input_tokens, output_tokens = await self.llm_client.generate(
            prompt=prompt, temperature=0.0, max_tokens=200
        )

        # Parse JSON response
        try:
            result = parse_json_response(response_text)

            query_type_str = result["query_type"].lower()
            query_type = QueryType(query_type_str)
            confidence = float(result["confidence"])
            reasoning = result["reasoning"]

            logger.debug(
                f"LLM classified query as {query_type.value} "
                f"with confidence {confidence:.2f} "
                f"(tokens: {input_tokens}+{output_tokens})"
            )

            return ClassificationResult(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                method="llm_based",
            )

        except Exception as e:
            logger.error(f"Failed to parse LLM classification response: {e}")
            # Fall back to exploratory
            return ClassificationResult(
                query_type=QueryType.EXPLORATORY,
                confidence=0.3,
                reasoning=f"LLM classification failed: {str(e)}",
                method="llm_based",
            )


class HybridClassifier(QueryClassifier):
    """Hybrid classifier: rules first, LLM fallback for low confidence."""

    def __init__(self, llm_config: LLMConfig, confidence_threshold: float = 0.7):
        """Initialize hybrid classifier.

        Args:
            llm_config: LLM configuration
            confidence_threshold: If rule-based confidence < threshold, use LLM
        """
        self.rule_classifier = RuleBasedClassifier()
        self.llm_classifier = LLMBasedClassifier(llm_config=llm_config)
        self.confidence_threshold = confidence_threshold

    async def classify(self, query: str) -> ClassificationResult:
        """Classify using rules first, LLM if confidence is low.

        Args:
            query: Query text to classify

        Returns:
            Classification result (from rules or LLM)
        """
        # Try rule-based first
        rule_result = await self.rule_classifier.classify(query)

        # If confidence is high enough, use rule-based result
        if rule_result.confidence >= self.confidence_threshold:
            logger.debug(
                f"Hybrid classifier: using rule-based result "
                f"(confidence {rule_result.confidence:.2f} >= {self.confidence_threshold})"
            )
            rule_result.method = "hybrid"
            return rule_result

        # Otherwise, fall back to LLM
        logger.debug(
            f"Hybrid classifier: rule-based confidence {rule_result.confidence:.2f} "
            f"< {self.confidence_threshold}, falling back to LLM"
        )
        llm_result = await self.llm_classifier.classify(query)
        llm_result.method = "hybrid"
        llm_result.reasoning = f"Rule-based low confidence, LLM says: {llm_result.reasoning}"

        return llm_result


def create_classifier(
    mode: Literal["rule_based", "llm_based", "hybrid"],
    llm_config: Optional[LLMConfig] = None,
    confidence_threshold: float = 0.7,
) -> QueryClassifier:
    """Factory function to create a query classifier.

    Args:
        mode: Classification mode (rule_based, llm_based, hybrid)
        llm_config: LLM configuration (required for llm_based and hybrid)
        confidence_threshold: Threshold for hybrid mode (default: 0.7)

    Returns:
        Configured QueryClassifier instance

    Raises:
        ValueError: If llm_config is missing for llm_based or hybrid mode
    """
    if mode == "rule_based":
        return RuleBasedClassifier()

    elif mode == "llm_based":
        if llm_config is None:
            raise ValueError("llm_config required for llm_based classifier")
        return LLMBasedClassifier(llm_config=llm_config)

    elif mode == "hybrid":
        if llm_config is None:
            raise ValueError("llm_config required for hybrid classifier")
        return HybridClassifier(
            llm_config=llm_config, confidence_threshold=confidence_threshold
        )

    else:
        raise ValueError(f"Unknown classifier mode: {mode}")
