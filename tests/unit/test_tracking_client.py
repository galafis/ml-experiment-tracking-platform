# -*- coding: utf-8 -*-
"""Unit tests for TrackingClient."""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import pytest

from src.models.experiment import RunStatus
from src.tracking.client import TrackingClient


class TestTrackingClient:
    """Tests for the high-level tracking client."""

    def test_create_experiment(self, tracking_client: TrackingClient):
        """Creating an experiment should store it in the backend."""
        exp = tracking_client.create_experiment("test_exp", description="desc")
        assert exp.name == "test_exp"

        retrieved = tracking_client.get_experiment(exp.id)
        assert retrieved is not None
        assert retrieved.name == "test_exp"

    def test_context_manager_completed(self, tracking_client: TrackingClient):
        """Normal exit from the context manager should mark run COMPLETED."""
        exp = tracking_client.create_experiment("ctx_test")

        with tracking_client.start_run(exp.id, "good_run") as ctx:
            ctx.log_param("key", "value")
            ctx.log_metric("score", 0.99)
            run_id = ctx.run_id

        run = tracking_client.get_run(run_id)
        assert run.status == RunStatus.COMPLETED
        assert tracking_client.active_run is None

    def test_context_manager_failed(self, tracking_client: TrackingClient):
        """Exception inside the context manager should mark run FAILED."""
        exp = tracking_client.create_experiment("fail_test")

        with pytest.raises(ValueError):
            with tracking_client.start_run(exp.id, "bad_run") as ctx:
                run_id = ctx.run_id
                raise ValueError("simulated error")

        run = tracking_client.get_run(run_id)
        assert run.status == RunStatus.FAILED

    def test_log_param_requires_active_run(self, tracking_client: TrackingClient):
        """Logging without an active run should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="No active run"):
            tracking_client.log_param("key", "value")

    def test_log_metrics(self, tracking_client: TrackingClient):
        """Logging multiple metrics should persist all values."""
        exp = tracking_client.create_experiment("metrics_test")
        with tracking_client.start_run(exp.id, "m_run") as ctx:
            ctx.log_metrics({"a": 1.0, "b": 2.0, "c": 3.0})
            run_id = ctx.run_id

        run = tracking_client.get_run(run_id)
        keys = {m.key for m in run.metrics}
        assert keys == {"a", "b", "c"}

    def test_log_model(self, tracking_client: TrackingClient):
        """log_model should serialize and store a model artifact."""
        exp = tracking_client.create_experiment("model_test")

        class FakeModel:
            value = 42

        with tracking_client.start_run(exp.id, "model_run") as ctx:
            ctx.log_param("type", "fake")
            artifact_id = tracking_client.log_model(FakeModel(), "my_model")
            run_id = ctx.run_id

        run = tracking_client.get_run(run_id)
        assert len(run.artifacts) == 1
        assert run.artifacts[0].name == "my_model.pkl"

    def test_list_experiments(self, tracking_client: TrackingClient):
        """list_experiments should return all created experiments."""
        tracking_client.create_experiment("e1")
        tracking_client.create_experiment("e2")

        experiments = tracking_client.list_experiments()
        names = {e.name for e in experiments}
        assert "e1" in names
        assert "e2" in names
