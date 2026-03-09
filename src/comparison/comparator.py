# -*- coding: utf-8 -*-
"""
Experiment comparison and analysis.

Provides the ``ExperimentComparator`` class for side-by-side metric
comparison, ranking, and tabular reporting across multiple experiment
runs. Useful for selecting the best model or understanding how
hyperparameter changes affect performance.
"""

from __future__ import annotations

from typing import Any, Optional

from src.models.experiment import Run, RunStatus
from src.storage.sqlite_store import SQLiteStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentComparator:
    """Compare runs across or within experiments.

    Parameters
    ----------
    store : SQLiteStore
        Storage backend to read run data from.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def compare_runs(
        self,
        run_ids: list[str],
        metric_keys: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Compare multiple runs side by side.

        For each run, extracts its parameters and the latest value of
        each requested metric.

        Parameters
        ----------
        run_ids : list[str]
            List of run identifiers to compare.
        metric_keys : list[str] | None
            Metric names to include. If ``None``, all metrics found
            across the runs are included.

        Returns
        -------
        list[dict[str, Any]]
            One dictionary per run containing ``run_id``, ``run_name``,
            ``status``, ``params``, and ``metrics``.
        """
        runs: list[Run] = []
        for rid in run_ids:
            run = self._store.get_run(rid)
            if run:
                runs.append(run)

        if metric_keys is None:
            keys_set: set[str] = set()
            for run in runs:
                for m in run.metrics:
                    keys_set.add(m.key)
            metric_keys = sorted(keys_set)

        results = []
        for run in runs:
            param_dict = {p.key: p.value for p in run.parameters}
            metric_dict: dict[str, Optional[float]] = {}
            for key in metric_keys:
                latest = run.latest_metric(key)
                metric_dict[key] = latest.value if latest else None

            results.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "status": run.status.value,
                    "params": param_dict,
                    "metrics": metric_dict,
                }
            )
        return results

    def best_run(
        self,
        experiment_id: str,
        metric_key: str,
        maximize: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Find the best run for a given metric within an experiment.

        Parameters
        ----------
        experiment_id : str
            Experiment to search in.
        metric_key : str
            Metric name to optimise.
        maximize : bool
            If ``True``, the run with the highest metric value wins;
            otherwise the lowest value wins.

        Returns
        -------
        dict[str, Any] | None
            Dictionary with ``run_id``, ``run_name``, ``metric_value``,
            and ``params``; or ``None`` if no qualifying runs exist.
        """
        runs = self._store.get_runs_by_experiment(experiment_id)
        candidates: list[tuple[float, Run]] = []

        for run in runs:
            if run.status != RunStatus.COMPLETED:
                continue
            latest = run.latest_metric(metric_key)
            if latest is not None:
                candidates.append((latest.value, run))

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0], reverse=maximize)
        best_val, best_run = candidates[0]
        param_dict = {p.key: p.value for p in best_run.parameters}

        return {
            "run_id": best_run.id,
            "run_name": best_run.name,
            "metric_key": metric_key,
            "metric_value": best_val,
            "params": param_dict,
        }

    def generate_comparison_table(
        self,
        run_ids: list[str],
        metric_keys: Optional[list[str]] = None,
        param_keys: Optional[list[str]] = None,
        sort_by: Optional[str] = None,
        ascending: bool = True,
    ) -> str:
        """Generate a formatted text comparison table.

        Parameters
        ----------
        run_ids : list[str]
            Runs to include.
        metric_keys : list[str] | None
            Metrics to show. ``None`` means all.
        param_keys : list[str] | None
            Parameters to show. ``None`` means all.
        sort_by : str | None
            Metric key to sort the table by.
        ascending : bool
            Sort direction.

        Returns
        -------
        str
            Formatted table suitable for console output.
        """
        comparison = self.compare_runs(run_ids, metric_keys)
        if not comparison:
            return "No runs to compare."

        if metric_keys is None:
            mk: set[str] = set()
            for entry in comparison:
                mk.update(entry["metrics"].keys())
            metric_keys = sorted(mk)

        if param_keys is None:
            pk: set[str] = set()
            for entry in comparison:
                pk.update(entry["params"].keys())
            param_keys = sorted(pk)

        # Sort if requested
        if sort_by and sort_by in metric_keys:
            comparison.sort(
                key=lambda e: e["metrics"].get(sort_by) or float("inf")
                if ascending
                else -(e["metrics"].get(sort_by) or float("inf")),
            )

        # Build header
        columns = ["Run Name", "Status"] + param_keys + metric_keys
        col_widths = [len(c) for c in columns]

        # Build rows
        rows: list[list[str]] = []
        for entry in comparison:
            row_cells = [
                entry["run_name"] or entry["run_id"][:12],
                entry["status"],
            ]
            for pk in param_keys:
                row_cells.append(str(entry["params"].get(pk, "-")))
            for mk in metric_keys:
                val = entry["metrics"].get(mk)
                row_cells.append(f"{val:.4f}" if val is not None else "-")
            rows.append(row_cells)

            for i, cell in enumerate(row_cells):
                col_widths[i] = max(col_widths[i], len(cell))

        # Format
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        header_line = (
            "|"
            + "|".join(f" {columns[i]:<{col_widths[i]}} " for i in range(len(columns)))
            + "|"
        )
        lines = [sep, header_line, sep]
        for row_cells in rows:
            line = (
                "|"
                + "|".join(
                    f" {row_cells[i]:<{col_widths[i]}} " for i in range(len(row_cells))
                )
                + "|"
            )
            lines.append(line)
        lines.append(sep)
        return "\n".join(lines)

    def rank_runs(
        self,
        experiment_id: str,
        metric_key: str,
        maximize: bool = True,
        top_n: int = 10,
    ) -> list[dict[str, Any]]:
        """Rank runs by a metric and return the top N.

        Parameters
        ----------
        experiment_id : str
            Target experiment.
        metric_key : str
            Metric to rank by.
        maximize : bool
            Whether higher values are better.
        top_n : int
            Maximum number of results.

        Returns
        -------
        list[dict[str, Any]]
            Ranked list of run summaries with ``rank``, ``run_id``,
            ``run_name``, ``metric_value``, and ``params``.
        """
        runs = self._store.get_runs_by_experiment(experiment_id)
        candidates: list[tuple[float, Run]] = []

        for run in runs:
            if run.status != RunStatus.COMPLETED:
                continue
            latest = run.latest_metric(metric_key)
            if latest is not None:
                candidates.append((latest.value, run))

        candidates.sort(key=lambda t: t[0], reverse=maximize)

        ranked = []
        for rank, (val, run) in enumerate(candidates[:top_n], start=1):
            ranked.append(
                {
                    "rank": rank,
                    "run_id": run.id,
                    "run_name": run.name,
                    "metric_key": metric_key,
                    "metric_value": val,
                    "params": {p.key: p.value for p in run.parameters},
                }
            )
        return ranked
