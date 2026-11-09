import json
from dataclasses import replace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from experiments.definitions import EXPERIMENTS
from tests.backend.test_training_jobs import FakeLauncher, wait_for, write_config

API = "/api/v1/experiments"


@pytest.fixture
def job_settings(settings, tmp_path):
    return replace(settings, config_path=write_config(tmp_path), max_queued_jobs=2,
                   max_train_iters=1000)


@pytest.fixture
def launcher():
    return FakeLauncher()


@pytest.fixture
def fclient(job_settings, launcher):
    with TestClient(create_app(job_settings, job_launcher=launcher)) as c:
        yield c


def write_run(root, batch, name, best_val, params=100):
    run = root / batch / name
    run.mkdir(parents=True)
    pd.DataFrame({"iter": [0, 10], "train_loss": [4.0, best_val - 0.1],
                  "val_loss": [4.1, best_val], "elapsed_s": [0.1, 1.0]}).to_csv(
        run / "history.csv", index=False)
    (run / "summary.json").write_text(json.dumps({
        "parameters": params, "training_time_s": 1.5, "final_train_loss": best_val - 0.1,
        "final_val_loss": best_val, "best_val_loss": best_val, "best_iter": 10}))


def test_definitions_come_from_the_experiment_module(fclient):
    body = fclient.get(API).json()
    assert [e["key"] for e in body["experiments"]] == list(EXPERIMENTS)
    heads = next(e for e in body["experiments"] if e["key"] == "heads")
    one = next(v for v in heads["variants"] if v["name"] == "heads_1")
    assert one["model"]["n_head"] == 1 and one["model"]["n_embd"] == body["base_model"]["n_embd"]
    bigram = next(v for e in body["experiments"] for v in e["variants"] if v["name"] == "bigram")
    assert bigram["model"] == {} and bigram["model_type"] == "bigram"


def test_run_queues_a_batch_job_for_the_worker(fclient, launcher):
    r = fclient.post(f"{API}/run", json={"experiments": ["heads"], "max_iters": 100, "seed": 3})
    assert r.status_code == 201
    body = r.json()
    assert body["kind"] == "experiment" and body["run_name"] == "exp-it100_s3"
    wait_for(lambda: len(launcher.launched) == 1)
    spec = json.loads((launcher.launched[0][0] / "spec.json").read_text())
    assert spec["batch_id"] == "it100_s3" and spec["experiments"] == ["heads"]
    assert spec["out_root"].endswith("experiments") and spec["force"] is False


@pytest.mark.parametrize("patch,code", [
    ({"experiments": ["nonsense"]}, 422), ({"max_iters": 5000}, 422),
    ({"max_iters": 0}, 422), ({"seed": -1}, 422),
])
def test_run_validation(fclient, launcher, patch, code):
    assert fclient.post(f"{API}/run", json=patch).status_code == code
    assert launcher.launched == []


def test_same_batch_cannot_be_queued_twice_and_experiment_jobs_cannot_resume(fclient, launcher):
    body = {"experiments": ["heads"], "max_iters": 100, "seed": 3}
    job_id = fclient.post(f"{API}/run", json=body).json()["job_id"]
    assert fclient.post(f"{API}/run", json=body).status_code == 409
    wait_for(lambda: len(launcher.launched) == 1)
    launcher.finish(0, result={"status": "completed"})
    wait_for(lambda: fclient.get(f"/api/v1/training/jobs/{job_id}").json()["status"] == "COMPLETED")
    r = fclient.post(f"/api/v1/training/jobs/{job_id}/resume")
    assert r.status_code == 422 and "submitting the same experiment" in r.json()["detail"]


def test_experiment_progress_comes_from_the_workers_progress_file(fclient, launcher):
    job_id = fclient.post(f"{API}/run", json={"max_iters": 100}).json()["job_id"]
    wait_for(lambda: len(launcher.launched) == 1)
    (launcher.launched[0][0] / "progress.json").write_text(
        json.dumps({"current": "heads_2", "done": 3, "total": 12}))
    job = fclient.get(f"/api/v1/training/jobs/{job_id}").json()
    assert job["progress"] == 0.25 and job["detail"]["current"] == "heads_2"


def test_results_listing_detail_compare_and_export(fclient, job_settings):
    root = job_settings.experiments_dir
    write_run(root, "it100_s1", "base", 2.0)
    write_run(root, "it100_s1", "bigram", 2.5, params=10)
    write_run(root, "it100_s2", "base", 2.1)
    (root / "not_a_batch").mkdir()

    runs = fclient.get(f"{API}/runs").json()["runs"]
    assert {r["batch_id"] for r in runs} == {"it100_s1", "it100_s2"}
    one = next(r for r in runs if r["batch_id"] == "it100_s1")
    assert one["variants_finished"] == 2 and one["max_iters"] == 100 and one["seed"] == 1

    detail = fclient.get(f"{API}/runs/it100_s1").json()
    rows = {r["run"]: r for r in detail["rows"]}
    assert rows["base"]["best_val_loss"] == 2.0 and rows["bigram"]["parameters"] == 10
    assert [p["iter"] for p in detail["curves"]["base"]] == [0, 10] and detail["batch"] is None

    cmp = fclient.get(f"{API}/compare", params=[("run_ids", "it100_s1"), ("run_ids", "it100_s2")]).json()
    by = {v["variant"]: v for v in cmp["variants"]}
    assert by["base"]["best_val_loss"] == {"it100_s1": 2.0, "it100_s2": 2.1}
    assert by["bigram"]["best_val_loss"]["it100_s2"] is None

    csv = fclient.get(f"{API}/export/it100_s1")
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    assert "attachment" in csv.headers["content-disposition"]
    lines = csv.text.strip().splitlines()
    assert lines[0].startswith("run,label,parameters") and len(lines) == 3


@pytest.mark.parametrize("path", ["/runs/it1_s1", "/runs/bad", "/export/it1_s1", "/export/bad"])
def test_unknown_or_malformed_batches_are_404(fclient, path):
    assert fclient.get(API + path).status_code == 404


def test_compare_needs_two_to_four_ids(fclient):
    assert fclient.get(f"{API}/compare", params=[("run_ids", "it1_s1")]).status_code == 422
    assert fclient.get(f"{API}/compare", params=[("run_ids", "it1_s1"), ("run_ids", "it2_s1")]).status_code == 404


def test_empty_results_directory_lists_nothing(fclient):
    assert fclient.get(f"{API}/runs").json() == {"runs": []}