# -*- coding: utf-8 -*-
"""
Run management layer.

Provides the RunManager class for starting, ending, and logging
parameters, metrics, and artifacts within experiment runs.
"""

from __future__ import annotations

from typing import Any, Optional

from src.models.experiment import (
    Artifact,
    ArtifactType,
    Metric,
    Parameter,
    Run,
    RunStatus,
    _new_id,
    _utcnow,
)
from src.storage.sqlite_store import SQLiteStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RunManager:
    """Manage the lifecycle of experiment runs.

    Parameters
    ----------
    store : SQLiteStore
        Persistent storage backend.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def start_run(
        self,
        experiment_id: str,
        run_name: str = "",
        tags: Optional[dict[str, str]] = None,
    ) -> Run:
        """Start a new run for the given experiment.

        Parameters
        ----------
        experiment_id : str
            Parent experiment identifier.
        run_name : str
            Optional human-readable name for the run.
        tags : dict[str, str] | None
            Arbitrary key-value tags.

        Returns
        -------
        Run
            The newly created and started run.
        """
        run = Run(
            id=_new_id(),
            experiment_id=experiment_id,
            name=run_name,
            status=RunStatus.RUNNING,
            tags=tags or {},
            start_time=_utcnow(),
            created_at=_utcnow(),
        )
        self._store.save_run(run)
        logger.info("Started run '%s' (id=%s) for experiment %s", run_name, run.id, experiment_id)
        return run

    def end_run(self, run_id: str, status: RunStatus = RunStatus.COMPLETED) -> Optional[Run]:
        """End a run by setting its status and end time.

        Parameters
        ----------
        run_id : str
            Run identifier.
        status : RunStatus
            Final status (COMPLETED, FAILED, or CANCELLED).

        Returns
        -------
        Run | None
            Updated run, or ``None`` if not found.
        """
        run = self._store.get_run(run_id)
        if run is None:
            return None

        run.status = status
        run.end_time = _utcnow()
        self._store.save_run(run)
        logger.info("Ended run id=%s with status=%s", run_id, status.value)
        return run

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        """Log a single parameter to a run.

        Parameters
        ----------
        run_id : str
            Target run identifier.
        key : str
            Parameter name.
        value : Any
            Parameter value (will be converted to string).
        """
        self._store.save_param(run_id, key, str(value))
        logger.debug("Logged param %s=%s for run %s", key, value, run_id)

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        """Log multiple parameters at once."""
        for key, value in params.items():
            self.log_param(run_id, key, value)

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        """Log a single metric to a run.

        Parameters
        ----------
        run_id : str
            Target run identifier.
        key : str
            Metric name.
        value : float
            Metric value.
        step : int
            Training step or epoch.
        """
        self._store.save_metric(run_id, key, value, step)
        logger.debug("Logged metric %s=%.6f (step=%d) for run %s", key, value, step, run_id)

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int = 0) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(run_id, key, value, step)

    def log_artifact(
        self,
        run_id: str,
        name: str,
        artifact_type: ArtifactType,
        uri: str,
        size_bytes: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Log an artifact reference to a run.

        Parameters
        ----------
        run_id : str
            Target run identifier.
        name : str
            Artifact display name.
        artifact_type : ArtifactType
            Classification of the artifact.
        uri : str
            Storage URI pointing to the artifact blob.
        size_bytes : int
            File size in bytes.
        metadata : dict | None
            Extra key-value metadata.

        Returns
        -------
        str
            Artifact identifier.
        """
        artifact_id = _new_id()
        self._store.save_artifact(
            run_id=run_id,
            artifact_id=artifact_id,
            name=name,
            artifact_type=artifact_type.value,
            uri=uri,
            size_bytes=size_bytes,
            metadata=metadata or {},
        )
        logger.info("Logged artifact '%s' (type=%s) for run %s", name, artifact_type.value, run_id)
        return artifact_id

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set a tag on a run."""
        self._store.save_tag(run_id, key, value)
        logger.debug("Set tag %s=%s for run %s", key, value, run_id)

    def get_run(self, run_id: str) -> Optional[Run]:
        """Retrieve a run by its identifier."""
        return self._store.get_run(run_id)

    def get_runs_by_experiment(self, experiment_id: str) -> list[Run]:
        """Retrieve all runs belonging to an experiment."""
        return self._store.get_runs_by_experiment(experiment_id)
