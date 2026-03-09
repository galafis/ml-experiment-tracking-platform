# -*- coding: utf-8 -*-
"""
Core experiment domain models.

Provides immutable dataclasses representing the fundamental entities
in the experiment tracking domain: experiments, runs, metrics,
parameters, and artifacts — with full serialization support.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class RunStatus(str, Enum):
    """Lifecycle states for an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactType(str, Enum):
    """Classification of stored artifacts."""

    MODEL = "model"
    DATASET = "dataset"
    PLOT = "plot"
    LOG = "log"
    CONFIG = "config"
    OTHER = "other"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Metric:
    """A single metric measurement recorded during a run.

    Attributes
    ----------
    key : str
        Metric name (e.g. ``accuracy``, ``loss``).
    value : float
        Numeric value of the metric.
    step : int
        Training step or epoch at which the metric was recorded.
    timestamp : datetime
        When the metric was logged.
    """

    key: str
    value: float
    step: int = 0
    timestamp: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metric:
        """Deserialize from dictionary."""
        data = dict(data)
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class Parameter:
    """A hyperparameter or configuration value for a run.

    Attributes
    ----------
    key : str
        Parameter name (e.g. ``learning_rate``).
    value : str
        Parameter value stored as string for uniformity.
    """

    key: str
    value: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Parameter:
        return cls(**data)


@dataclass
class Artifact:
    """Reference to a stored artifact (model checkpoint, dataset, plot, etc.).

    Attributes
    ----------
    id : str
        Unique artifact identifier.
    name : str
        Human-readable artifact name.
    artifact_type : ArtifactType
        Classification of the artifact.
    uri : str
        Storage URI pointing to the artifact blob.
    size_bytes : int
        File size in bytes.
    checksum : str
        SHA-256 checksum for integrity verification.
    metadata : dict
        Arbitrary key-value metadata.
    created_at : datetime
        Creation timestamp.
    """

    name: str
    artifact_type: ArtifactType
    uri: str
    size_bytes: int = 0
    checksum: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["artifact_type"] = self.artifact_type.value
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        data = dict(data)
        if isinstance(data.get("artifact_type"), str):
            data["artifact_type"] = ArtifactType(data["artifact_type"])
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class Run:
    """A single execution of an experiment.

    Attributes
    ----------
    id : str
        Unique run identifier.
    experiment_id : str
        Parent experiment identifier.
    name : str
        Human-readable run name.
    status : RunStatus
        Current lifecycle status.
    metrics : list[Metric]
        Recorded metrics during this run.
    parameters : list[Parameter]
        Hyperparameters used in this run.
    artifacts : list[Artifact]
        Artifacts produced by this run.
    tags : dict[str, str]
        Arbitrary tags for categorisation.
    start_time : datetime
        Run start timestamp.
    end_time : datetime | None
        Run completion timestamp (``None`` while running).
    created_at : datetime
        Record creation timestamp.
    """

    experiment_id: str
    name: str = ""
    status: RunStatus = RunStatus.PENDING
    metrics: list[Metric] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Transition run to RUNNING state."""
        self.status = RunStatus.RUNNING
        self.start_time = _utcnow()

    def complete(self) -> None:
        """Transition run to COMPLETED state."""
        self.status = RunStatus.COMPLETED
        self.end_time = _utcnow()

    def fail(self) -> None:
        """Transition run to FAILED state."""
        self.status = RunStatus.FAILED
        self.end_time = _utcnow()

    def cancel(self) -> None:
        """Transition run to CANCELLED state."""
        self.status = RunStatus.CANCELLED
        self.end_time = _utcnow()

    # ------------------------------------------------------------------
    # Metric / parameter helpers
    # ------------------------------------------------------------------

    def log_metric(self, key: str, value: float, step: int = 0) -> Metric:
        """Record a metric and return it."""
        metric = Metric(key=key, value=value, step=step)
        self.metrics.append(metric)
        return metric

    def log_parameter(self, key: str, value: str) -> Parameter:
        """Record a parameter and return it."""
        param = Parameter(key=key, value=value)
        self.parameters.append(param)
        return param

    def add_artifact(self, artifact: Artifact) -> None:
        """Attach an artifact to this run."""
        self.artifacts.append(artifact)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Elapsed time in seconds, or ``None`` if incomplete."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def get_metric(self, key: str) -> list[Metric]:
        """Return all metric entries matching *key*."""
        return [m for m in self.metrics if m.key == key]

    def latest_metric(self, key: str) -> Optional[Metric]:
        """Return the most recent metric entry for *key*."""
        entries = self.get_metric(key)
        return entries[-1] if entries else None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "name": self.name,
            "status": self.status.value,
            "metrics": [m.to_dict() for m in self.metrics],
            "parameters": [p.to_dict() for p in self.parameters],
            "artifacts": [a.to_dict() for a in self.artifacts],
            "tags": dict(self.tags),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "created_at": self.created_at.isoformat(),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        data = dict(data)
        data["status"] = RunStatus(data.get("status", "pending"))
        data["metrics"] = [Metric.from_dict(m) for m in data.get("metrics", [])]
        data["parameters"] = [Parameter.from_dict(p) for p in data.get("parameters", [])]
        data["artifacts"] = [Artifact.from_dict(a) for a in data.get("artifacts", [])]
        for dt_field in ("start_time", "end_time", "created_at"):
            val = data.get(dt_field)
            if isinstance(val, str):
                data[dt_field] = datetime.fromisoformat(val)
        return cls(**data)


@dataclass
class Experiment:
    """Top-level experiment grouping multiple runs.

    Attributes
    ----------
    id : str
        Unique experiment identifier.
    project_id : str
        Parent project identifier.
    name : str
        Experiment name.
    description : str
        Human-readable description.
    tags : dict[str, str]
        Arbitrary tags for categorisation.
    runs : list[Run]
        Child runs belonging to this experiment.
    created_at : datetime
        Creation timestamp.
    updated_at : datetime
        Last modification timestamp.
    """

    project_id: str
    name: str
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    runs: list[Run] = field(default_factory=list)
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def add_run(self, run: Run) -> None:
        """Attach a run to this experiment."""
        self.runs.append(run)
        self.updated_at = _utcnow()

    def get_run(self, run_id: str) -> Optional[Run]:
        """Retrieve a run by its ID."""
        for run in self.runs:
            if run.id == run_id:
                return run
        return None

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    @property
    def completed_runs(self) -> list[Run]:
        return [r for r in self.runs if r.status == RunStatus.COMPLETED]

    def best_run(self, metric_key: str, maximize: bool = True) -> Optional[Run]:
        """Return the run with the best final value for *metric_key*."""
        candidates: list[tuple[float, Run]] = []
        for run in self.completed_runs:
            latest = run.latest_metric(metric_key)
            if latest is not None:
                candidates.append((latest.value, run))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0], reverse=maximize)
        return candidates[0][1]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "tags": dict(self.tags),
            "runs": [r.to_dict() for r in self.runs],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        data = dict(data)
        data["runs"] = [Run.from_dict(r) for r in data.get("runs", [])]
        for dt_field in ("created_at", "updated_at"):
            val = data.get(dt_field)
            if isinstance(val, str):
                data[dt_field] = datetime.fromisoformat(val)
        return cls(**data)
