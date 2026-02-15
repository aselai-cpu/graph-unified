"""Response generation module.

Synthesizes coherent answers from retrieved chunks using LLM.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Set

from graphunified.strategies.base import QueryType, Chunk
from graphunified.config.settings import LLMConfig
from graphunified.utils.llm import ClaudeClient
from graphunified.prompts.query import RESPONSE_SYNTHESIS_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class GeneratedResponse:
    """Result of LLM response generation.

    Attributes:
        answer: Synthesized answer text
        chunks_used: Chunks that were included in context
        citations: List of source numbers cited in answer (e.g., [1, 3, 5])
        input_tokens: LLM input tokens used
        output_tokens: LLM output tokens used
    """

    answer: str
    chunks_used: List[Chunk] = field(default_factory=list)
    citations: Set[int] = field(default_factory=set)
    input_tokens: int = 0
    output_tokens: int = 0


class ResponseGenerator:
    """Generates synthesized responses using LLM.

    Takes retrieved chunks and generates a coherent answer that:
    - Integrates information from multiple sources
    - Cites sources using [Source N] notation
    - Adapts style based on query type
    """

    def __init__(self, llm_config: LLMConfig, temperature: float = 0.3, max_context_chunks: int = 10):
        """Initialize response generator.

        Args:
            llm_config: LLM configuration
            temperature: Generation temperature (default: 0.3)
            max_context_chunks: Maximum chunks to include in context (default: 10)
        """
        self.llm_client = ClaudeClient(config=llm_config)
        self.temperature = temperature
        self.max_context_chunks = max_context_chunks

    async def generate(
        self,
        query: str,
        chunks: List[Chunk],
        query_type: QueryType,
    ) -> GeneratedResponse:
        """Generate synthesized answer from chunks.

        Args:
            query: Original query text
            chunks: Retrieved chunks (sorted by relevance)
            query_type: Classified query type

        Returns:
            GeneratedResponse with answer and metadata
        """
        if not chunks:
            logger.warning("No chunks provided for synthesis, returning empty response")
            return GeneratedResponse(
                answer="I don't have enough information to answer this question.",
                chunks_used=[],
            )

        # Limit chunks to max_context_chunks
        chunks_to_use = chunks[: self.max_context_chunks]

        # Build context string
        context = self._build_context(chunks_to_use)

        # Generate prompt
        prompt = RESPONSE_SYNTHESIS_PROMPT.format(
            query=query, context=context, query_type=query_type.value
        )

        # Call LLM
        logger.debug(
            f"Generating response for {query_type.value} query "
            f"({len(chunks_to_use)} chunks, ~{len(context)} chars)"
        )

        try:
            response_text, input_tokens, output_tokens = await self.llm_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=1500,  # Allow longer answers
            )

            # Extract citations
            citations = self._extract_citations(response_text)

            logger.info(
                f"Generated response: {output_tokens} output tokens, "
                f"{len(citations)} sources cited"
            )

            return GeneratedResponse(
                answer=response_text.strip(),
                chunks_used=chunks_to_use,
                citations=citations,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fall back to raw chunks on error
            return GeneratedResponse(
                answer=f"Error generating response: {str(e)}\n\n"
                + self._format_fallback(chunks_to_use),
                chunks_used=chunks_to_use,
            )

    def _build_context(self, chunks: List[Chunk]) -> str:
        """Build context string from chunks.

        Args:
            chunks: Chunks to include in context

        Returns:
            Formatted context string with source numbers
        """
        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(f"[Source {i}]:\n{chunk.text}")

        return "\n\n".join(context_parts)

    def _extract_citations(self, text: str) -> Set[int]:
        """Extract source citations from generated text.

        Looks for [Source N] patterns in the text.

        Args:
            text: Generated response text

        Returns:
            Set of cited source numbers
        """
        # Pattern: [Source N] or [source N]
        pattern = r"\[[Ss]ource\s+(\d+)\]"
        matches = re.findall(pattern, text)

        citations = {int(m) for m in matches}
        return citations

    def _format_fallback(self, chunks: List[Chunk]) -> str:
        """Format chunks as fallback when LLM generation fails.

        Args:
            chunks: Chunks to format

        Returns:
            Formatted string
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(f"[Source {i}]:\n{chunk.text}")

        return "\n\n".join(parts)
