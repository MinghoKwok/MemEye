"""Configuration for ClusterManager."""

from dataclasses import dataclass


@dataclass
class ClusterManagerConfig:
    """Configuration for benchmark-scoped MemScene clustering."""

    similarity_threshold: float = 0.65
    max_time_gap_days: float = 7.0

    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0.0, 1.0], got {self.similarity_threshold}"
            )

        if self.max_time_gap_days < 0:
            raise ValueError(
                f"max_time_gap_days must be >= 0, got {self.max_time_gap_days}"
            )

    @property
    def max_time_gap_seconds(self) -> float:
        """Get max time gap in seconds."""
        return self.max_time_gap_days * 24 * 60 * 60
