# -*- coding: utf-8 -*-
"""Unit tests for ExperimentComparator."""

from __future__ import annotations

import pytest

from src.comparison.comparator import ExperimentComparator
from src.models.experiment import RunStatus
from src.tracking.experiment import ExperimentManager
from src.tracking.run import RunManager


class TestExperimentComparator:
    """Tests for run comparison and ranking."""

    def _create_runs(self, experiment_manager, run_manager):
        """Helper: create an experiment with three runs of varying quality."""
        exp = experiment_manager.create_experiment(name="comp_test")

        run_a = run_manager.start_run(exp.id, "model_a")
        run_manager.log_params(run_a.id, {"lr": "0.01"})
        run_manager.log_metric(run_a.id, "rmse", 0.50)
        run_manager.log_metric(run_a.id, "r2", 0.80)
        run_manager.end_run(run_a.id, RunStatus.COMPLETED)

        run_b = run_manager.start_run(exp.id, "model_b")
        run_manager.log_params(run_b.id, {"lr": "0.001"})
        run_manager.log_metric(run_b.id, "rmse", 0.30)
        run_manager.log_metric(run_b.id, "r2", 0.92)
        run_manager.end_run(run_b.id, RunStatus.COMPLETED)

        run_c = run_manager.start_run(exp.id, "model_c")
        run_manager.log_params(run_c.id, {"lr": "0.1"})
        run_manager.log_metric(run_c.id, "rmse", 0.70)
        run_manager.log_metric(run_c.id, "r2", 0.65)
        run_manager.end_run(run_c.id, RunStatus.COMPLETED)

        return exp, [run_a, run_b, run_c]

    def test_compare_runs(self, experiment_manager, run_manager, comparator):
        """Comparing runs should return all metrics for each run."""
        exp, runs = self._create_runs(experiment_manager, run_manager)
        run_ids = [r.id for r in runs]

        comparison = comparator.compare_runs(run_ids)
        assert len(comparison) == 3

        for entry in comparison:
            assert "run_id" in entry
            assert "metrics" in entry
            assert "rmse" in entry["metrics"]
            assert "r2" in entry["metrics"]

    def test_best_run_minimize(self, experiment_manager, run_manager, comparator):
        """best_run with maximize=False should return the lowest metric."""
        exp, runs = self._create_runs(experiment_manager, run_manager)

        best = comparator.best_run(exp.id, "rmse", maximize=False)
        assert best is not None
        assert best["run_name"] == "model_b"
        assert best["metric_value"] == 0.30

    def test_best_run_maximize(self, experiment_manager, run_manager, comparator):
        """best_run with maximize=True should return the highest metric."""
        exp, runs = self._create_runs(experiment_manager, run_manager)

        best = comparator.best_run(exp.id, "r2", maximize=True)
        assert best is not None
        assert best["run_name"] == "model_b"
        assert best["metric_value"] == 0.92

    def test_rank_runs(self, experiment_manager, run_manager, comparator):
        """Ranking runs should return them in sorted order."""
        exp, runs = self._create_runs(experiment_manager, run_manager)

        ranking = comparator.rank_runs(exp.id, "rmse", maximize=False, top_n=10)
        assert len(ranking) == 3
        assert ranking[0]["rank"] == 1
        assert ranking[0]["run_name"] == "model_b"
        assert ranking[2]["run_name"] == "model_c"

    def test_comparison_table_format(self, experiment_manager, run_manager, comparator):
        """The comparison table should be a non-empty formatted string."""
        exp, runs = self._create_runs(experiment_manager, run_manager)
        run_ids = [r.id for r in runs]

        table = comparator.generate_comparison_table(run_ids, metric_keys=["rmse"])
        assert isinstance(table, str)
        assert "rmse" in table
        assert "model_a" in table
        assert "+" in table  # table border character
