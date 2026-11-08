from __future__ import annotations

from fastapi import APIRouter, Query

from backend.dependencies import JobsDep
from backend.schemas.training import (
    JobStatus,
    LogsResponse,
    MetricRow,
    MetricsResponse,
    ResumeRequest,
    TrainingJob,
    TrainingJobList,
    TrainingJobRequest,
)

router = APIRouter(prefix="/training", tags=["training"])


@router.post("/jobs", response_model=TrainingJob, status_code=201)
def create_job(body: TrainingJobRequest, jobs: JobsDep) -> TrainingJob:
    """Queue a training job. The worker runs it in a separate process, one at a time."""
    return TrainingJob(**jobs.view(jobs.submit(body.model_dump())))


@router.get("/jobs", response_model=TrainingJobList)
def list_jobs(
    jobs: JobsDep, status: JobStatus | None = None, limit: int = Query(default=50, ge=1, le=200)
) -> TrainingJobList:
    return TrainingJobList(jobs=[TrainingJob(**j) for j in jobs.list_jobs(status, limit)])


@router.get("/jobs/{job_id}", response_model=TrainingJob)
def get_job(job_id: str, jobs: JobsDep) -> TrainingJob:
    return TrainingJob(**jobs.view(job_id))


@router.post("/jobs/{job_id}/stop", response_model=TrainingJob)
def stop_job(job_id: str, jobs: JobsDep) -> TrainingJob:
    """Queued jobs stop immediately. Running jobs stop within one training step and save a
    resumable checkpoint; the status becomes STOPPED when the worker has exited."""
    return TrainingJob(**jobs.stop(job_id))


@router.post("/jobs/{job_id}/resume", response_model=TrainingJob, status_code=201)
def resume_job(job_id: str, jobs: JobsDep, body: ResumeRequest | None = None) -> TrainingJob:
    """Start a NEW job that continues this job's run from its last checkpoint."""
    new_id = jobs.resume(job_id, body.max_iters if body else None)
    return TrainingJob(**jobs.view(new_id))


@router.get("/jobs/{job_id}/metrics", response_model=MetricsResponse)
def job_metrics(
    job_id: str, jobs: JobsDep, after_iter: int | None = Query(default=None, ge=0)
) -> MetricsResponse:
    rows = jobs.metrics(job_id, after_iter)
    return MetricsResponse(
        job_id=job_id, status=jobs.view(job_id)["status"], rows=[MetricRow(**r) for r in rows]
    )


@router.get("/jobs/{job_id}/logs", response_model=LogsResponse)
def job_logs(
    job_id: str, jobs: JobsDep, tail: int = Query(default=200, ge=1, le=1000)
) -> LogsResponse:
    return LogsResponse(
        job_id=job_id, status=jobs.view(job_id)["status"], lines=jobs.logs(job_id, tail)
    )