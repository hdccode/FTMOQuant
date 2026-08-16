"""Fail-closed DEVELOPMENT access and explicit walk-forward windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class ResearchAccessError(ValueError):
    """Raised when research requests sealed or out-of-range data."""


class ResearchPartition(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    FINAL_HOLDOUT = "final_holdout"


@dataclass(frozen=True, slots=True)
class DevelopmentAccessPolicy:
    """Configurable DEVELOPMENT interval with no validation/holdout accessor."""

    start_utc: datetime
    end_exclusive_utc: datetime

    def __post_init__(self) -> None:
        start = _utc(self.start_utc, "start_utc")
        end = _utc(self.end_exclusive_utc, "end_exclusive_utc")
        if start >= end:
            raise ResearchAccessError("DEVELOPMENT interval must be non-empty")

    def require_interval(
        self,
        start_utc: datetime,
        end_exclusive_utc: datetime,
        *,
        partition: ResearchPartition = ResearchPartition.DEVELOPMENT,
    ) -> tuple[datetime, datetime]:
        if partition is not ResearchPartition.DEVELOPMENT:
            raise ResearchAccessError(f"{partition.value} is sealed and inaccessible")
        start = _utc(start_utc, "start_utc")
        end = _utc(end_exclusive_utc, "end_exclusive_utc")
        if start >= end:
            raise ResearchAccessError("requested interval must be non-empty")
        if start < self.start_utc or end > self.end_exclusive_utc:
            raise ResearchAccessError("requested interval leaves DEVELOPMENT")
        return start, end

    def require_development_path(self, path: Path) -> Path:
        forbidden = {"validation", "holdout", "final_holdout", "final-holdout"}
        components = tuple(part.casefold() for part in path.parts)
        if any(marker in component for marker in forbidden for component in components):
            raise ResearchAccessError("sealed partition path is inaccessible")
        return path


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One explicit rolling train/evaluation window, wholly in DEVELOPMENT."""

    fold_id: str
    train_start_utc: datetime
    train_end_exclusive_utc: datetime
    evaluate_start_utc: datetime
    evaluate_end_exclusive_utc: datetime

    def validate(self, policy: DevelopmentAccessPolicy) -> None:
        if not self.fold_id.strip():
            raise ResearchAccessError("fold_id must not be empty")
        train_start, train_end = policy.require_interval(
            self.train_start_utc, self.train_end_exclusive_utc
        )
        evaluate_start, evaluate_end = policy.require_interval(
            self.evaluate_start_utc, self.evaluate_end_exclusive_utc
        )
        if train_end > evaluate_start:
            raise ResearchAccessError(
                "walk-forward training may not overlap evaluation"
            )
        if evaluate_start >= evaluate_end or train_start >= train_end:
            raise ResearchAccessError("walk-forward windows must be non-empty")


@dataclass(frozen=True, slots=True)
class DevelopmentSearchContext:
    """The complete data-boundary surface exposed to a generic search."""

    access_policy: DevelopmentAccessPolicy
    windows: tuple[WalkForwardWindow, ...]

    def __post_init__(self) -> None:
        if not self.windows:
            raise ResearchAccessError("at least one DEVELOPMENT fold is required")
        ids = tuple(window.fold_id for window in self.windows)
        if len(set(ids)) != len(ids):
            raise ResearchAccessError("fold IDs must be unique")
        for window in self.windows:
            window.validate(self.access_policy)


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ResearchAccessError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)
