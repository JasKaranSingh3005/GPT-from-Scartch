from __future__ import annotations

from fastapi import APIRouter, Query, Response

from backend.dependencies import JobsDep, SettingsDep
from backend.schemas.experiments import (
    BatchDetail, BatchList, BatchSummary, CompareResponse, ExperimentList, ExperimentRunRequest,
)
from backend.schemas.training import TrainingJob
from backend.services import experiments as svc

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("", response_model=ExperimentList)
def list_experiments() -> ExperimentList:
    """The experiment definitions from experiments/definitions.py."""
    return ExperimentList(**svc.definitions())


@router.post("/run", response_model=TrainingJob, status_code=201)
def run_experiments(body: ExperimentRunRequest, jobs: JobsDep) -> TrainingJob:
    """Queue a batch through the training worker. Finished variants are skipped unless
    `force` is set. Track it with the /training/jobs endpoints."""
    job_id = jobs.submit_experiment(body.model_dump())
    return TrainingJob(**jobs.view(job_id))


@router.get("/runs", response_model=BatchList)
def list_runs(settings: SettingsDep) -> BatchList:
    return BatchList(runs=[BatchSummary(**b) for b in svc.list_batches(settings.experiments_dir)])


@router.get("/runs/{run_id}", response_model=BatchDetail)
def get_run(run_id: str, settings: SettingsDep) -> BatchDetail:
    return BatchDetail(**svc.batch_detail(settings.experiments_dir, run_id))


@router.get("/compare", response_model=CompareResponse)
def compare_runs(
    settings: SettingsDep, run_ids: list[str] = Query(min_length=2, max_length=4)
) -> CompareResponse:
    return CompareResponse(**svc.compare(settings.experiments_dir, run_ids))


@router.get("/export/{run_id}")
def export_run(run_id: str, settings: SettingsDep) -> Response:
    text = svc.export_csv(settings.experiments_dir, run_id)
    return Response(
        content=text, media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'},
    )