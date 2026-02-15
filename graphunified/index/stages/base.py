"""Base classes for pipeline stages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class StageStatus(str, Enum):
    """Status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a pipeline stage execution."""

    status: StageStatus
    data: Any
    metadata: Dict[str, Any]
    duration: float  # seconds


# Type alias for progress callbacks
ProgressCallback = Callable[[str, float], None]


class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: str, progress_callback: Optional[ProgressCallback] = None):
        """Initialize pipeline stage.

        Args:
            name: Stage name
            progress_callback: Optional callback for progress updates
        """
        self.name = name
        self.progress_callback = progress_callback

    @abstractmethod
    async def execute(self, input_data: Any) -> StageResult:
        """Execute the pipeline stage.

        Args:
            input_data: Input data from previous stage

        Returns:
            StageResult with output data and metadata
        """
        pass

    def _report_progress(self, progress: float) -> None:
        """Report progress if callback is available.

        Args:
            progress: Progress value between 0.0 and 1.0
        """
        if self.progress_callback:
            self.progress_callback(self.name, progress)
