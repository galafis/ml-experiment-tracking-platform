# -*- coding: utf-8 -*-
"""
FastAPI REST API server for the ML Experiment Tracking Platform.

Exposes HTTP endpoints for experiment management, run lifecycle,
parameter and metric logging, and model registry operations. Designed
for remote experiment tracking from training scripts running on
different machines.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.comparison.comparator import ExperimentComparator
from src.registry.model_registry import ModelRegistry, ModelStage
from src.storage.sqlite_store import SQLiteStore
from src.tracking.experiment import ExperimentManager
from src.tracking.run import RunManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("MLTRACK_DB_PATH", "./tracking.db")
ARTIFACT_ROOT = os.environ.get("MLTRACK_ARTIFACT_ROOT", "./artifacts")

# ---------------------------------------------------------------------------
# Shared instances
# ---------------------------------------------------------------------------

store = SQLiteStore(DB_PATH)
experiment_mgr = ExperimentManager(store)
run_mgr = RunManager(store)
comparator = ExperimentComparator(store)
registry = ModelRegistry(DB_PATH)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ML Experiment Tracking Platform",
    description=(
        "Track, compare, and manage machine learning experiments "
        "with full parameter/metric logging and model versioning."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CreateExperimentRequest(BaseModel):
    name: str
    description: str = ""
    tags: dict[str, str] = Field(default_factory=dict)
    project_id: str = "default"


class ExperimentResponse(BaseModel):
    id: str
    project_id: str
    name: str
    description: str
    tags: dict[str, str]
    created_at: str
    updated_at: str


class CreateRunRequest(BaseModel):
    experiment_id: str
    run_name: str = ""
    tags: dict[str, str] = Field(default_factory=dict)


class EndRunRequest(BaseModel):
    status: str = "completed"


class LogParamRequest(BaseModel):
    key: str
    value: str


class LogMetricRequest(BaseModel):
    key: str
    value: float
    step: int = 0


class LogParamsRequest(BaseModel):
    params: dict[str, str]


class LogMetricsRequest(BaseModel):
    metrics: dict[str, float]
    step: int = 0


class RegisterModelRequest(BaseModel):
    name: str
    source_run_id: str
    artifact_uri: str = ""
    description: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class TransitionStageRequest(BaseModel):
    stage: str


class CompareRunsRequest(BaseModel):
    run_ids: list[str]
    metric_keys: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Service health check."""
    return {"status": "healthy", "service": "ml-experiment-tracking-platform"}


# ---------------------------------------------------------------------------
# Experiment endpoints
# ---------------------------------------------------------------------------

@app.post("/experiments", tags=["Experiments"], response_model=ExperimentResponse)
def create_experiment(req: CreateExperimentRequest):
    """Create a new experiment."""
    exp = experiment_mgr.create_experiment(
        name=req.name,
        description=req.description,
        tags=req.tags,
        project_id=req.project_id,
    )
    return _experiment_to_response(exp)


@app.get("/experiments", tags=["Experiments"])
def list_experiments():
    """List all experiments."""
    experiments = experiment_mgr.list_experiments()
    return [_experiment_to_response(e) for e in experiments]


@app.get("/experiments/{experiment_id}", tags=["Experiments"])
def get_experiment(experiment_id: str):
    """Get an experiment by ID."""
    exp = experiment_mgr.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    return _experiment_to_response(exp)


@app.delete("/experiments/{experiment_id}", tags=["Experiments"])
def delete_experiment(experiment_id: str):
    """Delete an experiment and all its runs."""
    deleted = experiment_mgr.delete_experiment(experiment_id)
    if not deleted:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    return {"deleted": True}


# ---------------------------------------------------------------------------
# Run endpoints
# ---------------------------------------------------------------------------

@app.post("/runs", tags=["Runs"])
def create_run(req: CreateRunRequest):
    """Start a new run."""
    run = run_mgr.start_run(req.experiment_id, req.run_name, req.tags)
    return run.to_dict()


@app.get("/runs/{run_id}", tags=["Runs"])
def get_run(run_id: str):
    """Get a run by ID."""
    run = run_mgr.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run.to_dict()


@app.put("/runs/{run_id}/end", tags=["Runs"])
def end_run(run_id: str, req: EndRunRequest):
    """End a run with a terminal status."""
    from src.models.experiment import RunStatus

    try:
        status = RunStatus(req.status)
    except ValueError:
        raise HTTPException(400, f"Invalid status: {req.status}")

    run = run_mgr.end_run(run_id, status)
    if run is None:
        raise HTTPException(404, f"Run '{run_id}' not found")
    return run.to_dict()


@app.get("/experiments/{experiment_id}/runs", tags=["Runs"])
def get_experiment_runs(experiment_id: str):
    """Get all runs for an experiment."""
    runs = run_mgr.get_runs_by_experiment(experiment_id)
    return [r.to_dict() for r in runs]


# ---------------------------------------------------------------------------
# Parameter / Metric endpoints
# ---------------------------------------------------------------------------

@app.post("/runs/{run_id}/params", tags=["Logging"])
def log_param(run_id: str, req: LogParamRequest):
    """Log a single parameter."""
    run_mgr.log_param(run_id, req.key, req.value)
    return {"logged": True}


@app.post("/runs/{run_id}/params/batch", tags=["Logging"])
def log_params_batch(run_id: str, req: LogParamsRequest):
    """Log multiple parameters."""
    run_mgr.log_params(run_id, req.params)
    return {"logged": len(req.params)}


@app.post("/runs/{run_id}/metrics", tags=["Logging"])
def log_metric(run_id: str, req: LogMetricRequest):
    """Log a single metric."""
    run_mgr.log_metric(run_id, req.key, req.value, req.step)
    return {"logged": True}


@app.post("/runs/{run_id}/metrics/batch", tags=["Logging"])
def log_metrics_batch(run_id: str, req: LogMetricsRequest):
    """Log multiple metrics."""
    run_mgr.log_metrics(run_id, req.metrics, req.step)
    return {"logged": len(req.metrics)}


# ---------------------------------------------------------------------------
# Model Registry endpoints
# ---------------------------------------------------------------------------

@app.post("/models", tags=["Model Registry"])
def register_model(req: RegisterModelRequest):
    """Register a new model version."""
    version = registry.register_model(
        name=req.name,
        source_run_id=req.source_run_id,
        artifact_uri=req.artifact_uri,
        description=req.description,
        metrics=req.metrics,
        tags=req.tags,
    )
    return version.to_dict()


@app.get("/models", tags=["Model Registry"])
def list_models():
    """List all registered models."""
    models = registry.list_models()
    return [
        {
            "name": m.name,
            "description": m.description,
            "tags": m.tags,
            "created_at": m.created_at.isoformat(),
            "updated_at": m.updated_at.isoformat(),
        }
        for m in models
    ]


@app.get("/models/{name}", tags=["Model Registry"])
def get_model(name: str, version: Optional[int] = Query(None)):
    """Get a model version (latest if version is omitted)."""
    mv = registry.get_model(name, version)
    if mv is None:
        raise HTTPException(404, f"Model '{name}' not found")
    return mv.to_dict()


@app.get("/models/{name}/versions", tags=["Model Registry"])
def list_model_versions(name: str):
    """List all versions of a model."""
    versions = registry.list_versions(name)
    return [v.to_dict() for v in versions]


@app.put("/models/{name}/versions/{version}/stage", tags=["Model Registry"])
def transition_model_stage(name: str, version: int, req: TransitionStageRequest):
    """Transition a model version to a new stage."""
    try:
        stage = ModelStage(req.stage)
    except ValueError:
        raise HTTPException(400, f"Invalid stage: {req.stage}")

    mv = registry.transition_stage(name, version, stage)
    if mv is None:
        raise HTTPException(404, f"Model '{name}' version {version} not found")
    return mv.to_dict()


# ---------------------------------------------------------------------------
# Comparison endpoints
# ---------------------------------------------------------------------------

@app.post("/compare", tags=["Comparison"])
def compare_runs(req: CompareRunsRequest):
    """Compare multiple runs side by side."""
    results = comparator.compare_runs(req.run_ids, req.metric_keys)
    return {"comparisons": results}


@app.get("/experiments/{experiment_id}/best", tags=["Comparison"])
def get_best_run(
    experiment_id: str,
    metric: str = Query(..., description="Metric key to optimise"),
    maximize: bool = Query(True, description="True for higher is better"),
):
    """Find the best run for a given metric."""
    result = comparator.best_run(experiment_id, metric, maximize)
    if result is None:
        raise HTTPException(404, "No qualifying runs found")
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _experiment_to_response(exp) -> dict[str, Any]:
    return {
        "id": exp.id,
        "project_id": exp.project_id,
        "name": exp.name,
        "description": exp.description,
        "tags": exp.tags,
        "created_at": exp.created_at.isoformat(),
        "updated_at": exp.updated_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the API server via uvicorn."""
    import uvicorn

    logger.info("Starting API server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
