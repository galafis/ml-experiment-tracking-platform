# -*- coding: utf-8 -*-
"""
Experiment management layer.

Provides the ExperimentManager class for creating, retrieving, listing,
and deleting experiments using a SQLite-backed storage backend. Each
experiment groups related runs for a particular modelling objective.
"""

from __future__ import annotations

from typing import Any, Optional

from src.models.experiment import Experiment, _new_id, _utcnow
from src.storage.sqlite_store import SQLiteStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentManager:
    """High-level manager for experiment lifecycle operations.

    Parameters
    ----------
    store : SQLiteStore
        Persistent storage backend.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: Optional[dict[str, str]] = None,
        project_id: str = "default",
    ) -> Experiment:
        """Create a new experiment and persist it.

        Parameters
        ----------
        name : str
            Human-readable experiment name (must be unique within a project).
        description : str
            Short description of the experiment's objective.
        tags : dict[str, str] | None
            Arbitrary key-value tags for filtering.
        project_id : str
            Parent project identifier (defaults to ``"default"``).

        Returns
        -------
        Experiment
            The newly created experiment.
        """
        experiment = Experiment(
            id=_new_id(),
            project_id=project_id,
            name=name,
            description=description,
            tags=tags or {},
            created_at=_utcnow(),
            updated_at=_utcnow(),
        )
        self._store.save_experiment(experiment)
        logger.info("Created experiment '%s' (id=%s)", name, experiment.id)
        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve an experiment by its unique identifier."""
        return self._store.get_experiment(experiment_id)

    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Retrieve an experiment by its name."""
        experiments = self._store.list_experiments()
        for exp in experiments:
            if exp.name == name:
                return exp
        return None

    def list_experiments(
        self,
        project_id: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> list[Experiment]:
        """List experiments with optional filtering.

        Parameters
        ----------
        project_id : str | None
            Filter by parent project.
        tags : dict[str, str] | None
            Filter by matching tags.

        Returns
        -------
        list[Experiment]
            Matching experiments ordered by creation time descending.
        """
        experiments = self._store.list_experiments()

        if project_id:
            experiments = [e for e in experiments if e.project_id == project_id]

        if tags:
            experiments = [
                e
                for e in experiments
                if all(e.tags.get(k) == v for k, v in tags.items())
            ]

        return experiments

    def update_experiment(
        self,
        experiment_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Optional[Experiment]:
        """Update experiment metadata.

        Only non-None fields are updated.

        Returns
        -------
        Experiment | None
            Updated experiment, or ``None`` if not found.
        """
        experiment = self._store.get_experiment(experiment_id)
        if experiment is None:
            return None

        if name is not None:
            experiment.name = name
        if description is not None:
            experiment.description = description
        if tags is not None:
            experiment.tags = tags
        experiment.updated_at = _utcnow()

        self._store.save_experiment(experiment)
        logger.info("Updated experiment '%s' (id=%s)", experiment.name, experiment_id)
        return experiment

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its child runs.

        Returns
        -------
        bool
            ``True`` if the experiment was deleted, ``False`` if not found.
        """
        deleted = self._store.delete_experiment(experiment_id)
        if deleted:
            logger.info("Deleted experiment id=%s", experiment_id)
        return deleted
