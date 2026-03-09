# -*- coding: utf-8 -*-
"""
Model registry with versioning and stage management.

Provides the ``ModelRegistry`` class for registering trained models,
managing versions, and transitioning models through deployment stages
(None -> Staging -> Production -> Archived).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelStage(str, Enum):
    """Deployment stage for a registered model version."""

    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class RegisteredModel:
    """A named model in the registry (can have multiple versions)."""

    name: str
    description: str = ""
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelVersion:
    """A specific version of a registered model."""

    name: str
    version: int
    source_run_id: str
    artifact_uri: str
    stage: ModelStage = ModelStage.NONE
    description: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "source_run_id": self.source_run_id,
            "artifact_uri": self.artifact_uri,
            "stage": self.stage.value,
            "description": self.description,
            "metrics": dict(self.metrics),
            "tags": dict(self.tags),
            "created_at": self.created_at.isoformat(),
        }


class ModelRegistry:
    """SQLite-backed model registry with versioning and stage management.

    Manages the lifecycle of trained models from registration through
    deployment stages. Each named model can have multiple versions, and
    each version tracks its source run, metrics, and current stage.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (shared with SQLiteStore or
        a separate file).
    """

    def __init__(self, db_path: str = "./tracking.db") -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS registered_models (
            name            TEXT PRIMARY KEY,
            description     TEXT DEFAULT '',
            tags_json       TEXT DEFAULT '{}',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            name            TEXT NOT NULL,
            version         INTEGER NOT NULL,
            source_run_id   TEXT NOT NULL,
            artifact_uri    TEXT DEFAULT '',
            stage           TEXT DEFAULT 'none',
            description     TEXT DEFAULT '',
            metrics_json    TEXT DEFAULT '{}',
            tags_json       TEXT DEFAULT '{}',
            created_at      TEXT NOT NULL,
            PRIMARY KEY (name, version),
            FOREIGN KEY (name) REFERENCES registered_models(name) ON DELETE CASCADE
        );
        """
        self._conn.executescript(ddl)
        self._conn.commit()

    def register_model(
        self,
        name: str,
        source_run_id: str,
        artifact_uri: str = "",
        description: str = "",
        metrics: Optional[dict[str, float]] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> ModelVersion:
        """Register a new model version.

        If the model name does not exist, a new ``RegisteredModel``
        entry is created. The version number is auto-incremented.

        Parameters
        ----------
        name : str
            Model name (e.g. ``"house_price_rf"``).
        source_run_id : str
            ID of the run that produced this model.
        artifact_uri : str
            URI pointing to the serialised model artifact.
        description : str
            Human-readable description of this version.
        metrics : dict[str, float] | None
            Key performance metrics to store with the version.
        tags : dict[str, str] | None
            Arbitrary tags.

        Returns
        -------
        ModelVersion
            The newly registered model version.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Ensure registered_models entry exists
        existing = self._conn.execute(
            "SELECT name FROM registered_models WHERE name = ?", (name,)
        ).fetchone()

        if existing is None:
            self._conn.execute(
                """
                INSERT INTO registered_models
                    (name, description, tags_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, description, json.dumps(tags or {}), now, now),
            )

        # Determine next version number
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS max_v FROM model_versions WHERE name = ?",
            (name,),
        ).fetchone()
        next_version = row["max_v"] + 1

        # Insert version
        self._conn.execute(
            """
            INSERT INTO model_versions
                (name, version, source_run_id, artifact_uri, stage,
                 description, metrics_json, tags_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                next_version,
                source_run_id,
                artifact_uri,
                ModelStage.NONE.value,
                description,
                json.dumps(metrics or {}),
                json.dumps(tags or {}),
                now,
            ),
        )

        # Update the registered model timestamp
        self._conn.execute(
            "UPDATE registered_models SET updated_at = ? WHERE name = ?",
            (now, name),
        )
        self._conn.commit()

        version = ModelVersion(
            name=name,
            version=next_version,
            source_run_id=source_run_id,
            artifact_uri=artifact_uri,
            stage=ModelStage.NONE,
            description=description,
            metrics=metrics or {},
            tags=tags or {},
            created_at=datetime.fromisoformat(now),
        )
        logger.info("Registered model '%s' version %d from run %s", name, next_version, source_run_id)
        return version

    def get_model(self, name: str, version: Optional[int] = None) -> Optional[ModelVersion]:
        """Retrieve a model version.

        Parameters
        ----------
        name : str
            Model name.
        version : int | None
            Specific version number. If ``None``, returns the latest.

        Returns
        -------
        ModelVersion | None
            The requested model version, or ``None`` if not found.
        """
        if version is not None:
            row = self._conn.execute(
                "SELECT * FROM model_versions WHERE name = ? AND version = ?",
                (name, version),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT * FROM model_versions WHERE name = ? ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_version(row)

    def get_model_by_stage(self, name: str, stage: ModelStage) -> Optional[ModelVersion]:
        """Retrieve the model version at a specific stage.

        Returns
        -------
        ModelVersion | None
            Latest version at the requested stage, or ``None``.
        """
        row = self._conn.execute(
            """
            SELECT * FROM model_versions
            WHERE name = ? AND stage = ?
            ORDER BY version DESC LIMIT 1
            """,
            (name, stage.value),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_version(row)

    def list_versions(self, name: str) -> list[ModelVersion]:
        """List all versions of a named model.

        Returns
        -------
        list[ModelVersion]
            All versions ordered by version number ascending.
        """
        rows = self._conn.execute(
            "SELECT * FROM model_versions WHERE name = ? ORDER BY version ASC",
            (name,),
        ).fetchall()
        return [self._row_to_version(r) for r in rows]

    def list_models(self) -> list[RegisteredModel]:
        """List all registered model names."""
        rows = self._conn.execute(
            "SELECT * FROM registered_models ORDER BY updated_at DESC"
        ).fetchall()
        return [
            RegisteredModel(
                name=r["name"],
                description=r["description"] or "",
                tags=json.loads(r["tags_json"] or "{}"),
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
            for r in rows
        ]

    def transition_stage(
        self,
        name: str,
        version: int,
        stage: ModelStage,
    ) -> Optional[ModelVersion]:
        """Transition a model version to a new deployment stage.

        If the target stage is PRODUCTION, any other version currently
        in Production for the same model is moved to Archived.

        Parameters
        ----------
        name : str
            Model name.
        version : int
            Version number to transition.
        stage : ModelStage
            Target stage.

        Returns
        -------
        ModelVersion | None
            Updated model version, or ``None`` if not found.
        """
        existing = self.get_model(name, version)
        if existing is None:
            return None

        # If transitioning to Production, archive current production version
        if stage == ModelStage.PRODUCTION:
            self._conn.execute(
                """
                UPDATE model_versions SET stage = ?
                WHERE name = ? AND stage = ?
                """,
                (ModelStage.ARCHIVED.value, name, ModelStage.PRODUCTION.value),
            )

        self._conn.execute(
            "UPDATE model_versions SET stage = ? WHERE name = ? AND version = ?",
            (stage.value, name, version),
        )
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE registered_models SET updated_at = ? WHERE name = ?",
            (now, name),
        )
        self._conn.commit()

        logger.info(
            "Transitioned model '%s' v%d to stage '%s'", name, version, stage.value
        )
        return self.get_model(name, version)

    def delete_model(self, name: str) -> bool:
        """Delete a model and all its versions."""
        self._conn.execute(
            "DELETE FROM model_versions WHERE name = ?", (name,)
        )
        cursor = self._conn.execute(
            "DELETE FROM registered_models WHERE name = ?", (name,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> ModelVersion:
        return ModelVersion(
            name=row["name"],
            version=row["version"],
            source_run_id=row["source_run_id"],
            artifact_uri=row["artifact_uri"] or "",
            stage=ModelStage(row["stage"]),
            description=row["description"] or "",
            metrics=json.loads(row["metrics_json"] or "{}"),
            tags=json.loads(row["tags_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
