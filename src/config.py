"""Shared project paths and logging configuration."""

import logging
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Paths and defaults used by the e-commerce data pipeline."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    log_level: int = logging.INFO

    @property
    def raw_data_dir(self) -> Path:
        """Return the directory for generated input data."""
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """Return the directory for analytics output."""
        return self.project_root / "data" / "processed"


def get_config() -> ProjectConfig:
    """Return the project configuration."""
    return ProjectConfig()


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent console logging for command-line runs."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
