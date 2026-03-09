# -*- coding: utf-8 -*-
"""Unit tests for the FastAPI REST API."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api_client():
    """Create a test client with isolated database."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_api.db")
    artifact_root = os.path.join(tmp_dir, "artifacts")

    os.environ["MLTRACK_DB_PATH"] = db_path
    os.environ["MLTRACK_ARTIFACT_ROOT"] = artifact_root

    # Force reimport to pick up env vars
    import importlib
    import src.api.server as server_mod

    importlib.reload(server_mod)

    client = TestClient(server_mod.app)
    yield client

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


class TestExperimentEndpoints:
    """Tests for experiment CRUD endpoints."""

    def test_create_and_get_experiment(self, api_client):
        resp = api_client.post(
            "/experiments",
            json={"name": "api_test", "description": "API test experiment"},
        )
        assert resp.status_code == 200
        data = resp.json()
        exp_id = data["id"]
        assert data["name"] == "api_test"

        resp = api_client.get(f"/experiments/{exp_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "api_test"

    def test_list_experiments(self, api_client):
        resp = api_client.get("/experiments")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_nonexistent_experiment(self, api_client):
        resp = api_client.get("/experiments/nonexistent_id")
        assert resp.status_code == 404


class TestRunEndpoints:
    """Tests for run lifecycle endpoints."""

    def test_create_and_end_run(self, api_client):
        # Create experiment first
        exp_resp = api_client.post(
            "/experiments", json={"name": "run_api_test"}
        )
        exp_id = exp_resp.json()["id"]

        # Start run
        run_resp = api_client.post(
            "/runs",
            json={"experiment_id": exp_id, "run_name": "api_run"},
        )
        assert run_resp.status_code == 200
        run_id = run_resp.json()["id"]

        # Log param
        param_resp = api_client.post(
            f"/runs/{run_id}/params",
            json={"key": "lr", "value": "0.01"},
        )
        assert param_resp.status_code == 200

        # Log metric
        metric_resp = api_client.post(
            f"/runs/{run_id}/metrics",
            json={"key": "accuracy", "value": 0.95, "step": 0},
        )
        assert metric_resp.status_code == 200

        # End run
        end_resp = api_client.put(
            f"/runs/{run_id}/end",
            json={"status": "completed"},
        )
        assert end_resp.status_code == 200
        assert end_resp.json()["status"] == "completed"

    def test_get_experiment_runs(self, api_client):
        exp_resp = api_client.post(
            "/experiments", json={"name": "runs_list_test"}
        )
        exp_id = exp_resp.json()["id"]
        api_client.post("/runs", json={"experiment_id": exp_id, "run_name": "r1"})
        api_client.post("/runs", json={"experiment_id": exp_id, "run_name": "r2"})

        resp = api_client.get(f"/experiments/{exp_id}/runs")
        assert resp.status_code == 200
        assert len(resp.json()) == 2
