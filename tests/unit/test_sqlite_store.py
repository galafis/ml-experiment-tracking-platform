# -*- coding: utf-8 -*-
"""Unit tests for SQLiteStore."""

from __future__ import annotations

import pytest

from src.models.experiment import RunStatus
from src.storage.sqlite_store import SQLiteStore


class TestSQLiteStoreExperiments:
    """Tests for experiment CRUD in SQLiteStore."""

    def test_save_and_get_experiment(self, experiment_manager):
        """Saving an experiment should make it retrievable by ID."""
        exp = experiment_manager.create_experiment(
            name="test_exp", description="A test experiment"
        )
        retrieved = experiment_manager.get_experiment(exp.id)
        assert retrieved is not None
        assert retrieved.name == "test_exp"
        assert retrieved.description == "A test experiment"

    def test_list_experiments(self, experiment_manager):
        """Listing experiments should return all created experiments."""
        experiment_manager.create_experiment(name="exp_1")
        experiment_manager.create_experiment(name="exp_2")
        experiments = experiment_manager.list_experiments()
        assert len(experiments) == 2
        names = {e.name for e in experiments}
        assert names == {"exp_1", "exp_2"}

    def test_delete_experiment(self, experiment_manager):
        """Deleting an experiment should remove it from the store."""
        exp = experiment_manager.create_experiment(name="doomed")
        assert experiment_manager.delete_experiment(exp.id) is True
        assert experiment_manager.get_experiment(exp.id) is None

    def test_delete_nonexistent_experiment(self, experiment_manager):
        """Deleting a nonexistent experiment should return False."""
        assert experiment_manager.delete_experiment("fake_id") is False

    def test_get_experiment_by_name(self, experiment_manager):
        """Should be able to find an experiment by its name."""
        experiment_manager.create_experiment(name="unique_name")
        found = experiment_manager.get_experiment_by_name("unique_name")
        assert found is not None
        assert found.name == "unique_name"

    def test_get_experiment_by_name_not_found(self, experiment_manager):
        """Looking up a nonexistent name should return None."""
        assert experiment_manager.get_experiment_by_name("nope") is None


class TestSQLiteStoreRuns:
    """Tests for run CRUD in SQLiteStore."""

    def test_start_and_get_run(self, experiment_manager, run_manager):
        """Starting a run should make it retrievable by ID."""
        exp = experiment_manager.create_experiment(name="run_test")
        run = run_manager.start_run(exp.id, "my_run")
        assert run.status == RunStatus.RUNNING
        assert run.name == "my_run"

        retrieved = run_manager.get_run(run.id)
        assert retrieved is not None
        assert retrieved.id == run.id
        assert retrieved.status == RunStatus.RUNNING

    def test_end_run(self, experiment_manager, run_manager):
        """Ending a run should update its status and end_time."""
        exp = experiment_manager.create_experiment(name="end_test")
        run = run_manager.start_run(exp.id, "to_end")
        run_manager.end_run(run.id, RunStatus.COMPLETED)

        retrieved = run_manager.get_run(run.id)
        assert retrieved.status == RunStatus.COMPLETED
        assert retrieved.end_time is not None

    def test_get_runs_by_experiment(self, experiment_manager, run_manager):
        """Should return all runs belonging to a specific experiment."""
        exp = experiment_manager.create_experiment(name="multi_run")
        run_manager.start_run(exp.id, "run_a")
        run_manager.start_run(exp.id, "run_b")

        runs = run_manager.get_runs_by_experiment(exp.id)
        assert len(runs) == 2


class TestSQLiteStoreParams:
    """Tests for parameter logging."""

    def test_log_and_retrieve_params(self, experiment_manager, run_manager):
        """Logged parameters should be retrievable from the run."""
        exp = experiment_manager.create_experiment(name="param_test")
        run = run_manager.start_run(exp.id, "param_run")
        run_manager.log_param(run.id, "lr", "0.01")
        run_manager.log_param(run.id, "epochs", "100")

        retrieved = run_manager.get_run(run.id)
        param_dict = {p.key: p.value for p in retrieved.parameters}
        assert param_dict["lr"] == "0.01"
        assert param_dict["epochs"] == "100"

    def test_param_overwrite(self, experiment_manager, run_manager):
        """Logging the same param key twice should overwrite the value."""
        exp = experiment_manager.create_experiment(name="overwrite_test")
        run = run_manager.start_run(exp.id, "ow_run")
        run_manager.log_param(run.id, "lr", "0.01")
        run_manager.log_param(run.id, "lr", "0.001")

        retrieved = run_manager.get_run(run.id)
        param_dict = {p.key: p.value for p in retrieved.parameters}
        assert param_dict["lr"] == "0.001"


class TestSQLiteStoreMetrics:
    """Tests for metric logging."""

    def test_log_and_retrieve_metrics(self, experiment_manager, run_manager):
        """Logged metrics should be retrievable from the run."""
        exp = experiment_manager.create_experiment(name="metric_test")
        run = run_manager.start_run(exp.id, "metric_run")
        run_manager.log_metric(run.id, "loss", 0.5, step=1)
        run_manager.log_metric(run.id, "loss", 0.3, step=2)
        run_manager.log_metric(run.id, "accuracy", 0.85, step=1)

        retrieved = run_manager.get_run(run.id)
        assert len(retrieved.metrics) == 3

        loss_metrics = retrieved.get_metric("loss")
        assert len(loss_metrics) == 2
        assert loss_metrics[0].value == 0.5
        assert loss_metrics[1].value == 0.3

    def test_search_runs_by_metric(self, sqlite_store, experiment_manager, run_manager):
        """Should find runs whose metric values fall within the range."""
        exp = experiment_manager.create_experiment(name="search_test")

        run1 = run_manager.start_run(exp.id, "good_run")
        run_manager.log_metric(run1.id, "rmse", 0.15)
        run_manager.end_run(run1.id)

        run2 = run_manager.start_run(exp.id, "bad_run")
        run_manager.log_metric(run2.id, "rmse", 0.95)
        run_manager.end_run(run2.id)

        results = sqlite_store.search_runs_by_metric(
            "rmse", max_value=0.5, experiment_id=exp.id
        )
        assert len(results) == 1
        assert results[0].id == run1.id
