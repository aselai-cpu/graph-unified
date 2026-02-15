"""Base storage interfaces."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator, List

from graphunified.config.models import (
    Chunk,
    Community,
    CommunityReport,
    Document,
    Entity,
    Relationship,
)


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def save_documents(self, documents: List[Document]) -> None:
        """Save documents to storage."""
        pass

    @abstractmethod
    async def load_documents(self) -> AsyncIterator[Document]:
        """Load documents from storage."""
        pass

    @abstractmethod
    async def save_chunks(self, chunks: List[Chunk]) -> None:
        """Save chunks to storage."""
        pass

    @abstractmethod
    async def load_chunks(self) -> AsyncIterator[Chunk]:
        """Load chunks from storage."""
        pass

    @abstractmethod
    async def save_entities(self, entities: List[Entity]) -> None:
        """Save entities to storage."""
        pass

    @abstractmethod
    async def load_entities(self) -> AsyncIterator[Entity]:
        """Load entities from storage."""
        pass

    @abstractmethod
    async def save_relationships(self, relationships: List[Relationship]) -> None:
        """Save relationships to storage."""
        pass

    @abstractmethod
    async def load_relationships(self) -> AsyncIterator[Relationship]:
        """Load relationships from storage."""
        pass

    @abstractmethod
    async def save_communities(self, communities: List[Community]) -> None:
        """Save communities to storage."""
        pass

    @abstractmethod
    async def load_communities(self) -> AsyncIterator[Community]:
        """Load communities from storage."""
        pass

    @abstractmethod
    async def save_community_reports(self, reports: List[CommunityReport]) -> None:
        """Save community reports to storage."""
        pass

    @abstractmethod
    async def load_community_reports(self) -> AsyncIterator[CommunityReport]:
        """Load community reports from storage."""
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Flush any buffered data to storage."""
        pass
