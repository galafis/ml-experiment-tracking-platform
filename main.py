# -*- coding: utf-8 -*-
"""
ML Experiment Tracking Platform — Complete Demo.

Demonstrates the full workflow of the platform:
  1. Creating an experiment for house price prediction.
  2. Running three different models (LinearRegression, RandomForest,
     GradientBoosting) and logging their parameters and metrics.
  3. Comparing all runs side by side.
  4. Identifying and registering the best model in the registry.
  5. Promoting the best model to Production stage.

This script uses synthetic housing data generated via sklearn's
``make_regression`` to keep things self-contained.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.comparison.comparator import ExperimentComparator
from src.registry.model_registry import ModelRegistry, ModelStage
from src.tracking.client import TrackingClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run the complete experiment tracking demo."""

    print("=" * 70)
    print("  ML Experiment Tracking Platform — Demo")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    db_path = "./demo_tracking.db"
    artifact_root = "./demo_artifacts"

    client = TrackingClient(db_path=db_path, artifact_root=artifact_root)
    registry = ModelRegistry(db_path=db_path)
    comparator = ExperimentComparator(client.store)

    # ------------------------------------------------------------------
    # 1. Create experiment
    # ------------------------------------------------------------------

    print("[1/6] Creating experiment 'house_price_prediction'...")
    experiment = client.create_experiment(
        name="house_price_prediction",
        description="Compare regression models for predicting house prices using synthetic data.",
        tags={"domain": "real_estate", "task": "regression"},
    )
    print(f"      Experiment created: id={experiment.id}")
    print()

    # ------------------------------------------------------------------
    # 2. Generate synthetic data
    # ------------------------------------------------------------------

    print("[2/6] Generating synthetic housing dataset...")
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        noise=15.0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"      Training set: {X_train.shape[0]} samples")
    print(f"      Test set:     {X_test.shape[0]} samples")
    print()

    # ------------------------------------------------------------------
    # 3. Train and track three models
    # ------------------------------------------------------------------

    models_config = [
        {
            "name": "LinearRegression",
            "model": LinearRegression(),
            "params": {"fit_intercept": True, "model_type": "linear"},
        },
        {
            "name": "RandomForest",
            "model": RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            "params": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "model_type": "ensemble",
            },
        },
        {
            "name": "GradientBoosting",
            "model": GradientBoostingRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            ),
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": 42,
                "model_type": "boosting",
            },
        },
    ]

    run_ids: list[str] = []

    print("[3/6] Training and tracking models...")
    print()
    for cfg in models_config:
        model_name = cfg["name"]
        model = cfg["model"]
        params = cfg["params"]

        print(f"      --- {model_name} ---")

        with client.start_run(experiment.id, model_name) as ctx:
            # Log parameters
            ctx.log_params(params)

            # Train
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # Compute metrics
            rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
            mae = float(mean_absolute_error(y_test, predictions))
            r2 = float(r2_score(y_test, predictions))

            # Log metrics
            ctx.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

            # Log model artifact
            ctx.set_tag("framework", "scikit-learn")
            client.log_model(model, model_name)

            run_ids.append(ctx.run_id)

            print(f"        RMSE : {rmse:.4f}")
            print(f"        MAE  : {mae:.4f}")
            print(f"        R2   : {r2:.4f}")
            print()

    # ------------------------------------------------------------------
    # 4. Compare all runs
    # ------------------------------------------------------------------

    print("[4/6] Comparing runs...")
    print()

    table = comparator.generate_comparison_table(
        run_ids=run_ids,
        metric_keys=["rmse", "mae", "r2"],
        param_keys=["model_type", "n_estimators", "max_depth", "learning_rate"],
    )
    print(table)
    print()

    # ------------------------------------------------------------------
    # 5. Find and register best model
    # ------------------------------------------------------------------

    print("[5/6] Finding best model (lowest RMSE)...")
    best = comparator.best_run(experiment.id, "rmse", maximize=False)

    if best:
        print(f"      Best run : {best['run_name']}")
        print(f"      RMSE     : {best['metric_value']:.4f}")
        print(f"      Run ID   : {best['run_id']}")
        print()

        print("      Registering best model in the registry...")
        version = registry.register_model(
            name="house_price_predictor",
            source_run_id=best["run_id"],
            description=f"Best model: {best['run_name']} (RMSE={best['metric_value']:.4f})",
            metrics={"rmse": best["metric_value"]},
            tags={"task": "regression", "algorithm": best["run_name"]},
        )
        print(f"      Registered: {version.name} v{version.version}")

        # Promote to staging, then production
        registry.transition_stage(version.name, version.version, ModelStage.STAGING)
        print(f"      Stage: {ModelStage.STAGING.value}")

        registry.transition_stage(version.name, version.version, ModelStage.PRODUCTION)
        print(f"      Stage: {ModelStage.PRODUCTION.value}")

    print()

    # ------------------------------------------------------------------
    # 6. Show ranking
    # ------------------------------------------------------------------

    print("[6/6] Model ranking by R2 (higher is better)...")
    ranking = comparator.rank_runs(experiment.id, "r2", maximize=True)
    for entry in ranking:
        print(
            f"      #{entry['rank']} {entry['run_name']:<25s} "
            f"R2 = {entry['metric_value']:.4f}"
        )

    print()
    print("=" * 70)
    print("  Demo completed successfully!")
    print(f"  Database : {db_path}")
    print(f"  Artifacts: {artifact_root}/")
    print("=" * 70)

    client.close()
    registry.close()


if __name__ == "__main__":
    main()
