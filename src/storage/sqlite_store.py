# -*- coding: utf-8 -*-
"""
SQLite storage backend.

Provides a lightweight, self-contained relational store for experiments,
runs, parameters, metrics, artifacts, and tags using Python's built-in
``sqlite3`` module. Ideal for local development, single-machine
deployments, and automated testing.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.models.experiment import (
    Artifact,
    ArtifactType,
    Experiment,
    Metric,
    Parameter,
    Run,
    RunStatus,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SQLiteStore:
    """Persistent SQLite storage for the experiment tracking platform.

    On instantiation the store creates all necessary tables if they do
    not already exist.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.  Use ``":memory:"`` for an
        ephemeral in-memory database (useful for tests).
    """

    def __init__(self, db_path: str = "./tracking.db") -> None:
        self._db_path = db_path

        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create all tables if they do not exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS experiments (
            id              TEXT PRIMARY KEY,
            project_id      TEXT NOT NULL DEFAULT 'default',
            name            TEXT NOT NULL,
            description     TEXT DEFAULT '',
            tags_json       TEXT DEFAULT '{}',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
            id              TEXT PRIMARY KEY,
            experiment_id   TEXT NOT NULL,
            name            TEXT DEFAULT '',
            status          TEXT DEFAULT 'pending',
            tags_json       TEXT DEFAULT '{}',
            start_time      TEXT,
            end_time        TEXT,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS params (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            key         TEXT NOT NULL,
            value       TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            key         TEXT NOT NULL,
            value       REAL NOT NULL,
            step        INTEGER DEFAULT 0,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            id              TEXT PRIMARY KEY,
            run_id          TEXT NOT NULL,
            name            TEXT NOT NULL,
            artifact_type   TEXT DEFAULT 'other',
            uri             TEXT DEFAULT '',
            size_bytes      INTEGER DEFAULT 0,
            checksum        TEXT DEFAULT '',
            metadata_json   TEXT DEFAULT '{}',
            created_at      TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tags (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id  TEXT NOT NULL,
            key     TEXT NOT NULL,
            value   TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_params_run ON params(run_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
        CREATE INDEX IF NOT EXISTS idx_metrics_key ON metrics(key);
        CREATE INDEX IF NOT EXISTS idx_artifacts_run ON artifacts(run_id);
        CREATE INDEX IF NOT EXISTS idx_tags_run ON tags(run_id);
        """
        self._conn.executescript(ddl)
        self._conn.commit()
        logger.info("SQLite tables initialised at %s", self._db_path)

    # ------------------------------------------------------------------
    # Experiment CRUD
    # ------------------------------------------------------------------

    def save_experiment(self, experiment: Experiment) -> None:
        """Insert or replace an experiment record."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO experiments
                (id, project_id, name, description, tags_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment.id,
                experiment.project_id,
                experiment.name,
                experiment.description,
                json.dumps(experiment.tags),
                experiment.created_at.isoformat(),
                experiment.updated_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Fetch an experiment by its ID, including runs."""
        row = self._conn.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        if row is None:
            return None
        experiment = self._row_to_experiment(row)
        experiment.runs = self.get_runs_by_experiment(experiment_id)
        return experiment

    def list_experiments(self) -> list[Experiment]:
        """Return all experiments ordered by creation time descending."""
        rows = self._conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and all its child records."""
        runs = self.get_runs_by_experiment(experiment_id)
        for run in runs:
            self.delete_run(run.id)

        cursor = self._conn.execute(
            "DELETE FROM experiments WHERE id = ?", (experiment_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Run CRUD
    # ------------------------------------------------------------------

    def save_run(self, run: Run) -> None:
        """Insert or replace a run record."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (id, experiment_id, name, status, tags_json, start_time, end_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.id,
                run.experiment_id,
                run.name,
                run.status.value,
                json.dumps(run.tags),
                run.start_time.isoformat() if run.start_time else None,
                run.end_time.isoformat() if run.end_time else None,
                run.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Optional[Run]:
        """Fetch a run with all its child entities."""
        row = self._conn.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None

        run = self._row_to_run(row)
        run.parameters = self._get_params(run_id)
        run.metrics = self._get_metrics(run_id)
        run.artifacts = self._get_artifacts(run_id)
        run.tags.update(self._get_tags(run_id))
        return run

    def get_runs_by_experiment(self, experiment_id: str) -> list[Run]:
        """Fetch all runs for an experiment."""
        rows = self._conn.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY created_at DESC",
            (experiment_id,),
        ).fetchall()
        runs = []
        for row in rows:
            run = self._row_to_run(row)
            run.parameters = self._get_params(run.id)
            run.metrics = self._get_metrics(run.id)
            run.artifacts = self._get_artifacts(run.id)
            run.tags.update(self._get_tags(run.id))
            runs.append(run)
        return runs

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all child records."""
        for table in ("params", "metrics", "artifacts", "tags"):
            self._conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        cursor = self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Parameter operations
    # ------------------------------------------------------------------

    def save_param(self, run_id: str, key: str, value: str) -> None:
        """Insert a parameter for a run (replaces if same key exists)."""
        self._conn.execute(
            "DELETE FROM params WHERE run_id = ? AND key = ?", (run_id, key)
        )
        self._conn.execute(
            "INSERT INTO params (run_id, key, value) VALUES (?, ?, ?)",
            (run_id, key, value),
        )
        self._conn.commit()

    def _get_params(self, run_id: str) -> list[Parameter]:
        rows = self._conn.execute(
            "SELECT key, value FROM params WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [Parameter(key=r["key"], value=r["value"]) for r in rows]

    # ------------------------------------------------------------------
    # Metric operations
    # ------------------------------------------------------------------

    def save_metric(
        self, run_id: str, key: str, value: float, step: int = 0
    ) -> None:
        """Append a metric measurement for a run."""
        self._conn.execute(
            "INSERT INTO metrics (run_id, key, value, step, timestamp) VALUES (?, ?, ?, ?, ?)",
            (run_id, key, value, step, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def _get_metrics(self, run_id: str) -> list[Metric]:
        rows = self._conn.execute(
            "SELECT key, value, step, timestamp FROM metrics WHERE run_id = ? ORDER BY step",
            (run_id,),
        ).fetchall()
        return [
            Metric(
                key=r["key"],
                value=r["value"],
                step=r["step"],
                timestamp=datetime.fromisoformat(r["timestamp"]),
            )
            for r in rows
        ]

    def search_runs_by_metric(
        self,
        metric_key: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        experiment_id: Optional[str] = None,
    ) -> list[Run]:
        """Search runs by metric value range.

        Parameters
        ----------
        metric_key : str
            Metric name to filter by.
        min_value : float | None
            Minimum metric value (inclusive).
        max_value : float | None
            Maximum metric value (inclusive).
        experiment_id : str | None
            Optionally restrict to a single experiment.

        Returns
        -------
        list[Run]
            Runs matching the filter criteria.
        """
        query = """
            SELECT DISTINCT r.id AS run_id
            FROM runs r
            JOIN metrics m ON r.id = m.run_id
            WHERE m.key = ?
        """
        params: list[Any] = [metric_key]

        if min_value is not None:
            query += " AND m.value >= ?"
            params.append(min_value)

        if max_value is not None:
            query += " AND m.value <= ?"
            params.append(max_value)

        if experiment_id is not None:
            query += " AND r.experiment_id = ?"
            params.append(experiment_id)

        rows = self._conn.execute(query, params).fetchall()
        return [self.get_run(r["run_id"]) for r in rows if self.get_run(r["run_id"])]

    # ------------------------------------------------------------------
    # Artifact operations
    # ------------------------------------------------------------------

    def save_artifact(
        self,
        run_id: str,
        artifact_id: str,
        name: str,
        artifact_type: str,
        uri: str,
        size_bytes: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Insert an artifact record."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO artifacts
                (id, run_id, name, artifact_type, uri, size_bytes, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                run_id,
                name,
                artifact_type,
                uri,
                size_bytes,
                json.dumps(metadata or {}),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def _get_artifacts(self, run_id: str) -> list[Artifact]:
        rows = self._conn.execute(
            "SELECT * FROM artifacts WHERE run_id = ?", (run_id,)
        ).fetchall()
        return [
            Artifact(
                id=r["id"],
                name=r["name"],
                artifact_type=ArtifactType(r["artifact_type"]),
                uri=r["uri"],
                size_bytes=r["size_bytes"],
                metadata=json.loads(r["metadata_json"] or "{}"),
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Tag operations
    # ------------------------------------------------------------------

    def save_tag(self, run_id: str, key: str, value: str) -> None:
        """Insert or replace a tag on a run."""
        self._conn.execute(
            "DELETE FROM tags WHERE run_id = ? AND key = ?", (run_id, key)
        )
        self._conn.execute(
            "INSERT INTO tags (run_id, key, value) VALUES (?, ?, ?)",
            (run_id, key, value),
        )
        self._conn.commit()

    def _get_tags(self, run_id: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM tags WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Row mappers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
        return Experiment(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            description=row["description"] or "",
            tags=json.loads(row["tags_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> Run:
        return Run(
            id=row["id"],
            experiment_id=row["experiment_id"],
            name=row["name"] or "",
            status=RunStatus(row["status"]),
            tags=json.loads(row["tags_json"] or "{}"),
            start_time=(
                datetime.fromisoformat(row["start_time"])
                if row["start_time"]
                else None
            ),
            end_time=(
                datetime.fromisoformat(row["end_time"]) if row["end_time"] else None
            ),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.info("SQLite connection closed")
