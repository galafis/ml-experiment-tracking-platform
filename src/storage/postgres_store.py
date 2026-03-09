# -*- coding: utf-8 -*-
"""
PostgreSQL storage backend using SQLAlchemy async engine.

Handles persistence for experiments, runs, metrics, parameters,
and projects using relational tables with full CRUD, querying,
and filtering capabilities.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config.settings import DatabaseConfig
from src.models.experiment import (
    Artifact,
    ArtifactType,
    Experiment,
    Metric,
    Parameter,
    Run,
    RunStatus,
)
from src.models.project import Project
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Table definitions
# ---------------------------------------------------------------------------

metadata_obj = sa.MetaData()

projects_table = sa.Table(
    "projects",
    metadata_obj,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("name", sa.String(256), nullable=False, index=True),
    sa.Column("description", sa.Text, default=""),
    sa.Column("team_json", sa.Text, default="null"),
    sa.Column("experiment_ids_json", sa.Text, default="[]"),
    sa.Column("tags_json", sa.Text, default="{}"),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

experiments_table = sa.Table(
    "experiments",
    metadata_obj,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("project_id", sa.String(64), sa.ForeignKey("projects.id"), index=True),
    sa.Column("name", sa.String(256), nullable=False, index=True),
    sa.Column("description", sa.Text, default=""),
    sa.Column("tags_json", sa.Text, default="{}"),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
)

runs_table = sa.Table(
    "runs",
    metadata_obj,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column(
        "experiment_id",
        sa.String(64),
        sa.ForeignKey("experiments.id"),
        index=True,
    ),
    sa.Column("name", sa.String(256), default=""),
    sa.Column("status", sa.String(32), default="pending"),
    sa.Column("tags_json", sa.Text, default="{}"),
    sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
    sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)

metrics_table = sa.Table(
    "metrics",
    metadata_obj,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.id"), index=True),
    sa.Column("key", sa.String(256), nullable=False, index=True),
    sa.Column("value", sa.Float, nullable=False),
    sa.Column("step", sa.Integer, default=0),
    sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
)

parameters_table = sa.Table(
    "parameters",
    metadata_obj,
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.id"), index=True),
    sa.Column("key", sa.String(256), nullable=False),
    sa.Column("value", sa.Text, nullable=False),
)

artifacts_table = sa.Table(
    "artifacts",
    metadata_obj,
    sa.Column("id", sa.String(64), primary_key=True),
    sa.Column("run_id", sa.String(64), sa.ForeignKey("runs.id"), index=True),
    sa.Column("name", sa.String(512), nullable=False),
    sa.Column("artifact_type", sa.String(32), default="other"),
    sa.Column("uri", sa.Text, default=""),
    sa.Column("size_bytes", sa.BigInteger, default=0),
    sa.Column("checksum", sa.String(128), default=""),
    sa.Column("metadata_json", sa.Text, default="{}"),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
)


class PostgresStore:
    """Async PostgreSQL storage for structured experiment data.

    Parameters
    ----------
    config : DatabaseConfig
        PostgreSQL connection configuration.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._engine = create_async_engine(
            config.dsn,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            echo=config.echo_sql,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def initialize(self) -> None:
        """Create tables if they do not exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(metadata_obj.create_all)
        logger.info("PostgreSQL tables initialised")

    async def close(self) -> None:
        """Dispose of the connection pool."""
        await self._engine.dispose()
        logger.info("PostgreSQL connection pool closed")

    # ------------------------------------------------------------------
    # Project CRUD
    # ------------------------------------------------------------------

    async def create_project(self, project: Project) -> Project:
        """Insert a new project."""
        async with self._session_factory() as session:
            await session.execute(
                projects_table.insert().values(
                    id=project.id,
                    name=project.name,
                    description=project.description,
                    team_json=json.dumps(
                        project.team.to_dict() if project.team else None
                    ),
                    experiment_ids_json=json.dumps(project.experiment_ids),
                    tags_json=json.dumps(project.tags),
                    created_at=project.created_at,
                    updated_at=project.updated_at,
                )
            )
            await session.commit()
        logger.info("Created project %s (%s)", project.name, project.id)
        return project

    async def get_project(self, project_id: str) -> Optional[Project]:
        """Fetch a project by ID."""
        async with self._session_factory() as session:
            result = await session.execute(
                projects_table.select().where(projects_table.c.id == project_id)
            )
            row = result.mappings().first()
        if not row:
            return None
        return self._row_to_project(row)

    async def list_projects(
        self, offset: int = 0, limit: int = 50
    ) -> list[Project]:
        """List projects with pagination."""
        async with self._session_factory() as session:
            result = await session.execute(
                projects_table.select()
                .order_by(projects_table.c.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            rows = result.mappings().all()
        return [self._row_to_project(r) for r in rows]

    async def update_project(self, project: Project) -> Project:
        """Update an existing project."""
        async with self._session_factory() as session:
            await session.execute(
                projects_table.update()
                .where(projects_table.c.id == project.id)
                .values(
                    name=project.name,
                    description=project.description,
                    team_json=json.dumps(
                        project.team.to_dict() if project.team else None
                    ),
                    experiment_ids_json=json.dumps(project.experiment_ids),
                    tags_json=json.dumps(project.tags),
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        return project

    async def delete_project(self, project_id: str) -> bool:
        """Delete a project by ID. Returns True if deleted."""
        async with self._session_factory() as session:
            result = await session.execute(
                projects_table.delete().where(projects_table.c.id == project_id)
            )
            await session.commit()
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Experiment CRUD
    # ------------------------------------------------------------------

    async def create_experiment(self, experiment: Experiment) -> Experiment:
        """Insert a new experiment."""
        async with self._session_factory() as session:
            await session.execute(
                experiments_table.insert().values(
                    id=experiment.id,
                    project_id=experiment.project_id,
                    name=experiment.name,
                    description=experiment.description,
                    tags_json=json.dumps(experiment.tags),
                    created_at=experiment.created_at,
                    updated_at=experiment.updated_at,
                )
            )
            await session.commit()
        logger.info("Created experiment %s (%s)", experiment.name, experiment.id)
        return experiment

    async def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Fetch an experiment with all its runs."""
        async with self._session_factory() as session:
            result = await session.execute(
                experiments_table.select().where(
                    experiments_table.c.id == experiment_id
                )
            )
            row = result.mappings().first()
        if not row:
            return None
        experiment = self._row_to_experiment(row)
        experiment.runs = await self.list_runs(experiment_id)
        return experiment

    async def list_experiments(
        self,
        project_id: Optional[str] = None,
        name_contains: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[Experiment]:
        """List experiments with optional filtering."""
        query = experiments_table.select()
        if project_id:
            query = query.where(experiments_table.c.project_id == project_id)
        if name_contains:
            query = query.where(
                experiments_table.c.name.ilike(f"%{name_contains}%")
            )
        query = (
            query.order_by(experiments_table.c.created_at.desc())
            .offset(offset)
            .limit(limit)
        )

        async with self._session_factory() as session:
            result = await session.execute(query)
            rows = result.mappings().all()

        experiments = [self._row_to_experiment(r) for r in rows]

        if tags:
            experiments = [
                e
                for e in experiments
                if all(e.tags.get(k) == v for k, v in tags.items())
            ]
        return experiments

    async def update_experiment(self, experiment: Experiment) -> Experiment:
        """Update experiment metadata."""
        async with self._session_factory() as session:
            await session.execute(
                experiments_table.update()
                .where(experiments_table.c.id == experiment.id)
                .values(
                    name=experiment.name,
                    description=experiment.description,
                    tags_json=json.dumps(experiment.tags),
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await session.commit()
        return experiment

    async def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment and its child records."""
        runs = await self.list_runs(experiment_id)
        for run in runs:
            await self.delete_run(run.id)

        async with self._session_factory() as session:
            result = await session.execute(
                experiments_table.delete().where(
                    experiments_table.c.id == experiment_id
                )
            )
            await session.commit()
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Run CRUD
    # ------------------------------------------------------------------

    async def create_run(self, run: Run) -> Run:
        """Insert a new run."""
        async with self._session_factory() as session:
            await session.execute(
                runs_table.insert().values(
                    id=run.id,
                    experiment_id=run.experiment_id,
                    name=run.name,
                    status=run.status.value,
                    tags_json=json.dumps(run.tags),
                    start_time=run.start_time,
                    end_time=run.end_time,
                    created_at=run.created_at,
                )
            )
            # Persist metrics, parameters, artifacts
            for metric in run.metrics:
                await self._insert_metric(session, run.id, metric)
            for param in run.parameters:
                await self._insert_parameter(session, run.id, param)
            for artifact in run.artifacts:
                await self._insert_artifact(session, run.id, artifact)
            await session.commit()
        logger.info("Created run %s for experiment %s", run.id, run.experiment_id)
        return run

    async def get_run(self, run_id: str) -> Optional[Run]:
        """Fetch a run with its metrics, parameters, and artifacts."""
        async with self._session_factory() as session:
            result = await session.execute(
                runs_table.select().where(runs_table.c.id == run_id)
            )
            row = result.mappings().first()
            if not row:
                return None

            run = self._row_to_run(row)

            # Load metrics
            m_result = await session.execute(
                metrics_table.select()
                .where(metrics_table.c.run_id == run_id)
                .order_by(metrics_table.c.step)
            )
            run.metrics = [self._row_to_metric(r) for r in m_result.mappings().all()]

            # Load parameters
            p_result = await session.execute(
                parameters_table.select().where(parameters_table.c.run_id == run_id)
            )
            run.parameters = [
                Parameter(key=r["key"], value=r["value"])
                for r in p_result.mappings().all()
            ]

            # Load artifacts
            a_result = await session.execute(
                artifacts_table.select().where(artifacts_table.c.run_id == run_id)
            )
            run.artifacts = [
                self._row_to_artifact(r) for r in a_result.mappings().all()
            ]

        return run

    async def list_runs(
        self,
        experiment_id: str,
        status: Optional[RunStatus] = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[Run]:
        """List runs for an experiment."""
        query = runs_table.select().where(
            runs_table.c.experiment_id == experiment_id
        )
        if status:
            query = query.where(runs_table.c.status == status.value)
        query = (
            query.order_by(runs_table.c.created_at.desc()).offset(offset).limit(limit)
        )

        async with self._session_factory() as session:
            result = await session.execute(query)
            rows = result.mappings().all()

        runs: list[Run] = []
        for row in rows:
            run = self._row_to_run(row)
            # Eagerly load child entities
            full_run = await self.get_run(run.id)
            if full_run:
                runs.append(full_run)
        return runs

    async def update_run(self, run: Run) -> Run:
        """Update run status and timestamps."""
        async with self._session_factory() as session:
            await session.execute(
                runs_table.update()
                .where(runs_table.c.id == run.id)
                .values(
                    name=run.name,
                    status=run.status.value,
                    tags_json=json.dumps(run.tags),
                    start_time=run.start_time,
                    end_time=run.end_time,
                )
            )
            await session.commit()
        return run

    async def delete_run(self, run_id: str) -> bool:
        """Delete a run and all child records."""
        async with self._session_factory() as session:
            await session.execute(
                metrics_table.delete().where(metrics_table.c.run_id == run_id)
            )
            await session.execute(
                parameters_table.delete().where(parameters_table.c.run_id == run_id)
            )
            await session.execute(
                artifacts_table.delete().where(artifacts_table.c.run_id == run_id)
            )
            result = await session.execute(
                runs_table.delete().where(runs_table.c.id == run_id)
            )
            await session.commit()
        return result.rowcount > 0

    # ------------------------------------------------------------------
    # Metric operations
    # ------------------------------------------------------------------

    async def log_metric(self, run_id: str, metric: Metric) -> None:
        """Append a metric to an existing run."""
        async with self._session_factory() as session:
            await self._insert_metric(session, run_id, metric)
            await session.commit()

    async def log_metrics(self, run_id: str, metrics: list[Metric]) -> None:
        """Batch-log multiple metrics."""
        async with self._session_factory() as session:
            for m in metrics:
                await self._insert_metric(session, run_id, m)
            await session.commit()

    async def get_metrics(
        self,
        run_id: str,
        key: Optional[str] = None,
    ) -> list[Metric]:
        """Retrieve metrics for a run, optionally filtered by key."""
        query = metrics_table.select().where(metrics_table.c.run_id == run_id)
        if key:
            query = query.where(metrics_table.c.key == key)
        query = query.order_by(metrics_table.c.step)

        async with self._session_factory() as session:
            result = await session.execute(query)
            rows = result.mappings().all()
        return [self._row_to_metric(r) for r in rows]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search_experiments(
        self, query_text: str, limit: int = 20
    ) -> list[Experiment]:
        """Full-text search across experiment names and descriptions."""
        pattern = f"%{query_text}%"
        stmt = (
            experiments_table.select()
            .where(
                sa.or_(
                    experiments_table.c.name.ilike(pattern),
                    experiments_table.c.description.ilike(pattern),
                )
            )
            .limit(limit)
        )
        async with self._session_factory() as session:
            result = await session.execute(stmt)
            rows = result.mappings().all()
        return [self._row_to_experiment(r) for r in rows]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _insert_metric(
        session: AsyncSession, run_id: str, metric: Metric
    ) -> None:
        await session.execute(
            metrics_table.insert().values(
                run_id=run_id,
                key=metric.key,
                value=metric.value,
                step=metric.step,
                timestamp=metric.timestamp,
            )
        )

    @staticmethod
    async def _insert_parameter(
        session: AsyncSession, run_id: str, param: Parameter
    ) -> None:
        await session.execute(
            parameters_table.insert().values(
                run_id=run_id, key=param.key, value=param.value
            )
        )

    @staticmethod
    async def _insert_artifact(
        session: AsyncSession, run_id: str, artifact: Artifact
    ) -> None:
        await session.execute(
            artifacts_table.insert().values(
                id=artifact.id,
                run_id=run_id,
                name=artifact.name,
                artifact_type=artifact.artifact_type.value,
                uri=artifact.uri,
                size_bytes=artifact.size_bytes,
                checksum=artifact.checksum,
                metadata_json=json.dumps(artifact.metadata),
                created_at=artifact.created_at,
            )
        )

    @staticmethod
    def _row_to_project(row: Any) -> Project:
        team_data = json.loads(row["team_json"]) if row["team_json"] else None
        from src.models.project import Team

        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            team=Team.from_dict(team_data) if team_data else None,
            experiment_ids=json.loads(row["experiment_ids_json"] or "[]"),
            tags=json.loads(row["tags_json"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_experiment(row: Any) -> Experiment:
        return Experiment(
            id=row["id"],
            project_id=row["project_id"],
            name=row["name"],
            description=row["description"] or "",
            tags=json.loads(row["tags_json"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_run(row: Any) -> Run:
        return Run(
            id=row["id"],
            experiment_id=row["experiment_id"],
            name=row["name"] or "",
            status=RunStatus(row["status"]),
            tags=json.loads(row["tags_json"] or "{}"),
            start_time=row["start_time"],
            end_time=row["end_time"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_metric(row: Any) -> Metric:
        return Metric(
            key=row["key"],
            value=row["value"],
            step=row["step"],
            timestamp=row["timestamp"],
        )

    @staticmethod
    def _row_to_artifact(row: Any) -> Artifact:
        return Artifact(
            id=row["id"],
            name=row["name"],
            artifact_type=ArtifactType(row["artifact_type"]),
            uri=row["uri"] or "",
            size_bytes=row["size_bytes"] or 0,
            checksum=row["checksum"] or "",
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"],
        )
