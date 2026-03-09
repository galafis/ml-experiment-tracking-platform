# -*- coding: utf-8 -*-
"""
High-level tracking client.

Provides a unified ``TrackingClient`` API for experiment tracking, with
context-manager support for runs and optional autologging for
scikit-learn models.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from src.models.experiment import ArtifactType, Experiment, Run, RunStatus
from src.storage.file_store import FileArtifactStore
from src.storage.sqlite_store import SQLiteStore
from src.tracking.experiment import ExperimentManager
from src.tracking.run import RunManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class _RunContext:
    """Lightweight handle returned by ``TrackingClient.start_run``.

    Enables convenient in-run logging via attribute access while the
    context manager keeps track of lifecycle transitions.
    """

    def __init__(self, run: Run, client: TrackingClient) -> None:
        self.run = run
        self._client = client

    @property
    def run_id(self) -> str:
        return self.run.id

    def log_param(self, key: str, value: Any) -> None:
        self._client.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        self._client.log_params(params)

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self._client.log_metric(key, value, step)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        self._client.log_metrics(metrics, step)

    def log_artifact(self, local_path: str, artifact_name: Optional[str] = None) -> str:
        return self._client.log_artifact(local_path, artifact_name)

    def set_tag(self, key: str, value: str) -> None:
        self._client.set_tag(key, value)


class TrackingClient:
    """Unified experiment-tracking facade.

    Manages experiment lifecycle, run lifecycle, and all logging
    operations through a single entry point. Supports context-manager
    syntax for automatic run completion on exit.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
    artifact_root : str
        Root directory for artifact storage.

    Examples
    --------
    >>> client = TrackingClient("./tracking.db", "./artifacts")
    >>> exp = client.create_experiment("my_experiment")
    >>> with client.start_run(exp.id, "run_1") as ctx:
    ...     ctx.log_param("lr", 0.01)
    ...     ctx.log_metric("accuracy", 0.95)
    """

    def __init__(
        self,
        db_path: str = "./tracking.db",
        artifact_root: str = "./artifacts",
    ) -> None:
        self._store = SQLiteStore(db_path)
        self._artifact_store = FileArtifactStore(artifact_root)
        self._experiment_mgr = ExperimentManager(self._store)
        self._run_mgr = RunManager(self._store)
        self._active_run: Optional[Run] = None

    # ------------------------------------------------------------------
    # Experiment API
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[dict[str, str]] = None,
    ) -> Experiment:
        """Create a new experiment."""
        return self._experiment_mgr.create_experiment(
            name=name, description=description, tags=tags
        )

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self._experiment_mgr.get_experiment(experiment_id)

    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name."""
        return self._experiment_mgr.get_experiment_by_name(name)

    def list_experiments(self) -> list[Experiment]:
        """List all experiments."""
        return self._experiment_mgr.list_experiments()

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment."""
        return self._experiment_mgr.delete_experiment(experiment_id)

    # ------------------------------------------------------------------
    # Run API
    # ------------------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        experiment_id: str,
        run_name: str = "",
        tags: Optional[dict[str, str]] = None,
    ) -> Generator[_RunContext, None, None]:
        """Start a new run as a context manager.

        On normal exit the run is marked COMPLETED; on exception it is
        marked FAILED.

        Parameters
        ----------
        experiment_id : str
            Parent experiment identifier.
        run_name : str
            Optional human-readable run name.
        tags : dict[str, str] | None
            Optional tags for the run.

        Yields
        ------
        _RunContext
            Handle for in-run logging.
        """
        run = self._run_mgr.start_run(experiment_id, run_name, tags)
        self._active_run = run
        ctx = _RunContext(run, self)
        try:
            yield ctx
            self._run_mgr.end_run(run.id, RunStatus.COMPLETED)
        except Exception:
            self._run_mgr.end_run(run.id, RunStatus.FAILED)
            raise
        finally:
            self._active_run = None

    def start_run_simple(
        self,
        experiment_id: str,
        run_name: str = "",
        tags: Optional[dict[str, str]] = None,
    ) -> Run:
        """Start a run without context-manager semantics.

        Callers must explicitly call ``end_run`` to finalise.
        """
        run = self._run_mgr.start_run(experiment_id, run_name, tags)
        self._active_run = run
        return run

    def end_run(
        self,
        run_id: Optional[str] = None,
        status: RunStatus = RunStatus.COMPLETED,
    ) -> Optional[Run]:
        """End the active run (or a specific run by ID)."""
        target = run_id or (self._active_run.id if self._active_run else None)
        if target is None:
            logger.warning("No active run to end")
            return None
        result = self._run_mgr.end_run(target, status)
        if self._active_run and self._active_run.id == target:
            self._active_run = None
        return result

    def get_run(self, run_id: str) -> Optional[Run]:
        """Retrieve a run by ID."""
        return self._run_mgr.get_run(run_id)

    def get_runs(self, experiment_id: str) -> list[Run]:
        """List all runs for an experiment."""
        return self._run_mgr.get_runs_by_experiment(experiment_id)

    # ------------------------------------------------------------------
    # Logging API (uses active run)
    # ------------------------------------------------------------------

    def _require_active_run(self) -> str:
        if self._active_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        return self._active_run.id

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to the active run."""
        run_id = self._require_active_run()
        self._run_mgr.log_param(run_id, key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters to the active run."""
        run_id = self._require_active_run()
        self._run_mgr.log_params(run_id, params)

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        """Log a metric to the active run."""
        run_id = self._require_active_run()
        self._run_mgr.log_metric(run_id, key, value, step)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log multiple metrics to the active run."""
        run_id = self._require_active_run()
        self._run_mgr.log_metrics(run_id, metrics, step)

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the active run."""
        run_id = self._require_active_run()
        self._run_mgr.set_tag(run_id, key, value)

    def log_artifact(
        self,
        local_path: str,
        artifact_name: Optional[str] = None,
    ) -> str:
        """Save a local file as an artifact for the active run.

        Parameters
        ----------
        local_path : str
            Path to the local file.
        artifact_name : str | None
            Display name (defaults to filename).

        Returns
        -------
        str
            Artifact identifier.
        """
        run_id = self._require_active_run()
        path = Path(local_path)
        name = artifact_name or path.name

        uri = self._artifact_store.save_artifact(run_id, path)
        size = path.stat().st_size if path.exists() else 0

        ext = path.suffix.lower()
        type_map = {
            ".pkl": ArtifactType.MODEL,
            ".joblib": ArtifactType.MODEL,
            ".h5": ArtifactType.MODEL,
            ".csv": ArtifactType.DATASET,
            ".parquet": ArtifactType.DATASET,
            ".png": ArtifactType.PLOT,
            ".jpg": ArtifactType.PLOT,
            ".svg": ArtifactType.PLOT,
            ".log": ArtifactType.LOG,
            ".yaml": ArtifactType.CONFIG,
            ".yml": ArtifactType.CONFIG,
            ".json": ArtifactType.CONFIG,
        }
        artifact_type = type_map.get(ext, ArtifactType.OTHER)

        return self._run_mgr.log_artifact(
            run_id=run_id,
            name=name,
            artifact_type=artifact_type,
            uri=uri,
            size_bytes=size,
        )

    def log_model(self, model: Any, model_name: str = "model") -> str:
        """Serialize and log a model artifact.

        The model is pickled and stored through the artifact subsystem.

        Parameters
        ----------
        model : Any
            Fitted model object (must be picklable).
        model_name : str
            Base name for the artifact file.

        Returns
        -------
        str
            Artifact identifier.
        """
        run_id = self._require_active_run()
        filename = f"{model_name}.pkl"
        tmp_path = Path(self._artifact_store.root_dir) / "_tmp" / filename
        tmp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp_path, "wb") as f:
            pickle.dump(model, f)

        uri = self._artifact_store.save_artifact(run_id, tmp_path)
        size = tmp_path.stat().st_size

        artifact_id = self._run_mgr.log_artifact(
            run_id=run_id,
            name=filename,
            artifact_type=ArtifactType.MODEL,
            uri=uri,
            size_bytes=size,
        )

        try:
            tmp_path.unlink()
        except OSError:
            pass

        return artifact_id

    # ------------------------------------------------------------------
    # Autolog
    # ------------------------------------------------------------------

    def autolog_sklearn(
        self,
        model: Any,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
    ) -> dict[str, float]:
        """Automatically log parameters and evaluation metrics for a
        fitted scikit-learn estimator.

        Extracts hyperparameters via ``get_params()`` and computes
        regression or classification metrics on the test set.

        Parameters
        ----------
        model : sklearn estimator
            A fitted scikit-learn model.
        X_train, y_train : array-like
            Training data (used only for informational parameters).
        X_test, y_test : array-like
            Test data for metric computation.

        Returns
        -------
        dict[str, float]
            Dictionary of computed evaluation metrics.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
        import numpy as np

        # Log model parameters
        params = model.get_params()
        for k, v in params.items():
            self.log_param(k, v)

        # Log dataset size
        self.log_param("train_samples", len(X_train))
        self.log_param("test_samples", len(X_test))
        self.log_param("model_class", type(model).__name__)

        # Compute and log metrics
        predictions = model.predict(X_test)
        metrics: dict[str, float] = {}

        is_classifier = hasattr(model, "predict_proba") or hasattr(model, "classes_")

        if is_classifier:
            metrics["accuracy"] = float(accuracy_score(y_test, predictions))
            try:
                metrics["f1_score"] = float(
                    f1_score(y_test, predictions, average="weighted")
                )
            except Exception:
                pass
        else:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, predictions)))
            metrics["mae"] = float(mean_absolute_error(y_test, predictions))
            metrics["r2"] = float(r2_score(y_test, predictions))

        self.log_metrics(metrics)

        # Log model artifact
        self.log_model(model, type(model).__name__)

        return metrics

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def active_run(self) -> Optional[Run]:
        """Return the currently active run, if any."""
        return self._active_run

    @property
    def store(self) -> SQLiteStore:
        """Direct access to the underlying store (for advanced queries)."""
        return self._store

    def close(self) -> None:
        """Close the storage backend connection."""
        self._store.close()
