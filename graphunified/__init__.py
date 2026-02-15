"""Graph-Unified RAG: A unified multi-strategy RAG system.

This package provides a unified implementation of multiple RAG strategies:
- Naive RAG (dense retrieval)
- Hybrid RAG (dense + sparse retrieval)
- GraphRAG Local (entity-based local search)
- GraphRAG Global (community-based global search)
- LightRAG (dual-level retrieval with knowledge graphs)
- HippoRAG (neurologically-inspired retrieval)

All strategies share a common extraction pipeline for efficiency.
"""

from graphunified.__version__ import __version__

__all__ = ["__version__"]
