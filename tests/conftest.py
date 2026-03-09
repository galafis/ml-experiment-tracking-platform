# -*- coding: utf-8 -*-
"""Shared test fixtures for the ML Experiment Tracking Platform."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.comparison.comparator import ExperimentComparator
from src.registry.model_registry import ModelRegistry
from src.storage.file_store import FileArtifactStore
from src.storage.sqlite_store import SQLiteStore
from src.tracking.client import TrackingClient
from src.tracking.experiment import ExperimentManager
from src.tracking.run import RunManager


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provide a clean temporary directory."""
    return tmp_path


@pytest.fixture
def sqlite_store() -> SQLiteStore:
    """Create an in-memory SQLite store."""
    store = SQLiteStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def file_store(tmp_path: Path) -> FileArtifactStore:
    """Create a temporary file artifact store."""
    return FileArtifactStore(str(tmp_path / "artifacts"))


@pytest.fixture
def experiment_manager(sqlite_store: SQLiteStore) -> ExperimentManager:
    """Create an experiment manager with in-memory storage."""
    return ExperimentManager(sqlite_store)


@pytest.fixture
def run_manager(sqlite_store: SQLiteStore) -> RunManager:
    """Create a run manager with in-memory storage."""
    return RunManager(sqlite_store)


@pytest.fixture
def comparator(sqlite_store: SQLiteStore) -> ExperimentComparator:
    """Create a comparator with in-memory storage."""
    return ExperimentComparator(sqlite_store)


@pytest.fixture
def tracking_client(tmp_path: Path) -> TrackingClient:
    """Create a tracking client with temporary storage."""
    db_path = str(tmp_path / "test_tracking.db")
    artifact_root = str(tmp_path / "test_artifacts")
    client = TrackingClient(db_path=db_path, artifact_root=artifact_root)
    yield client
    client.close()


@pytest.fixture
def model_registry(tmp_path: Path) -> ModelRegistry:
    """Create a model registry with temporary storage."""
    db_path = str(tmp_path / "test_registry.db")
    reg = ModelRegistry(db_path=db_path)
    yield reg
    reg.close()
