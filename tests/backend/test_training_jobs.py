import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.services.jobs import JobManager

API = "/api/v1/training/jobs"
BODY = dict(name="job1", model_type="gpt", n_embd=16, n_head=2, n_layer=1, block_size=8,
            dropout=0.0, batch_size=4, learning_rate=0.01, max_iters=20, eval_interval=10,
            eval_iters=2, seed=0)


def write_config(tmp_path: Path) -> Path:
    text = tmp_path / "tiny.txt"
    text.write_text("to be or not to be, that is the question. " * 120, encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("\n".join([
        "data:", "  url: unused", f'  path: "{text.as_posix()}"', "  train_fraction: 0.9",
        "model:", "  n_embd: 16", "  n_head: 2", "  n_layer: 1", "  block_size: 8", "  dropout: 0.0",
        "training:", "  batch_size: 4", "  learning_rate: 0.01", "  max_iters: 20",
        "  eval_interval: 10", "  eval_iters: 2", "  seed: 0", "",
    ]), encoding="utf-8")
    return cfg


class FakeProcess:
    pid = 4242

    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class FakeLauncher:
    def __init__(self):
        self.launched: list[tuple[Path, FakeProcess]] = []

    def __call__(self, job_dir, log_file):
        proc = FakeProcess()
        self.launched.append((Path(job_dir), proc))
        return proc

    def finish(self, i, result=None, error=None, code=0):
        job_dir, proc = self.launched[i]
        if result is not None:
            (job_dir / "result.json").write_text(json.dumps(result))
        if error is not None:
            (job_dir / "error.json").write_text(json.dumps({"error": error}))
        proc.returncode = code


def wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out waiting for condition")


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


def submit(client, **over):
    return client.post(API, json={**BODY, **over})


def status(client, job_id):
    return client.get(f"{API}/{job_id}").json()["status"]


def test_create_job_returns_a_queued_job_and_launches_it(fclient, launcher):
    r = submit(fclient)
    assert r.status_code == 201
    body = r.json()
    assert len(body["job_id"]) == 12 and body["model_id"] == "job1:best"
    assert body["config"]["training"]["max_iters"] == 20 and body["config"]["model"]["n_head"] == 2
    wait_for(lambda: len(launcher.launched) == 1)
    assert wait_for(lambda: status(fclient, body["job_id"]) == "RUNNING")
    assert fclient.get(API).json()["jobs"][0]["job_id"] == body["job_id"]


@pytest.mark.parametrize("patch,code", [
    ({"n_head": 3}, 422), ({"max_iters": 5000}, 422), ({"name": "../x"}, 422),
    ({"name": ""}, 422), ({"batch_size": 0}, 422), ({"model_type": "rnn"}, 422),
    ({"learning_rate": 0}, 422), ({"name": "tiny"}, 409),  # runs/tiny already exists
])
def test_invalid_or_conflicting_requests(fclient, launcher, patch, code):
    r = submit(fclient, **patch)
    assert r.status_code == code and "detail" in r.json()
    assert launcher.launched == []


def test_duplicate_active_name_is_409(fclient):
    assert submit(fclient).status_code == 201
    assert submit(fclient).status_code == 409


def test_only_one_job_runs_at_a_time(fclient, launcher):
    a = submit(fclient, name="a").json()["job_id"]
    wait_for(lambda: len(launcher.launched) == 1)
    b = submit(fclient, name="b").json()["job_id"]
    time.sleep(0.6)
    assert len(launcher.launched) == 1 and status(fclient, b) == "QUEUED"
    launcher.finish(0, result={"status": "completed", "best_val_loss": 1.5})
    wait_for(lambda: len(launcher.launched) == 2)
    assert status(fclient, a) == "COMPLETED"
    done = fclient.get(f"{API}/{a}").json()
    assert done["progress"] == 1.0 and done["result"]["best_val_loss"] == 1.5
    assert "run_dir" not in done["result"]


def test_queue_is_bounded(fclient, launcher):
    submit(fclient, name="a")
    wait_for(lambda: len(launcher.launched) == 1)
    assert submit(fclient, name="b").status_code == 201
    assert submit(fclient, name="c").status_code == 201
    assert submit(fclient, name="d").status_code == 429


def test_stopping_a_queued_job_never_launches_it(fclient, launcher):
    submit(fclient, name="a")
    wait_for(lambda: len(launcher.launched) == 1)
    b = submit(fclient, name="b").json()["job_id"]
    r = fclient.post(f"{API}/{b}/stop")
    assert r.status_code == 200 and r.json()["status"] == "STOPPED"
    launcher.finish(0, result={"status": "completed"})
    time.sleep(0.8)
    assert len(launcher.launched) == 1


def test_stopping_a_running_job_signals_the_worker(fclient, launcher):
    job_id = submit(fclient).json()["job_id"]
    wait_for(lambda: len(launcher.launched) == 1)
    wait_for(lambda: status(fclient, job_id) == "RUNNING")
    body = fclient.post(f"{API}/{job_id}/stop").json()
    assert body["stop_requested"] is True and body["status"] == "RUNNING"
    assert (launcher.launched[0][0] / "stop.flag").exists()
    launcher.finish(0, result={"status": "stopped", "iteration": 10})
    wait_for(lambda: status(fclient, job_id) == "STOPPED")
    assert fclient.post(f"{API}/{job_id}/stop").status_code == 409  # already finished


def test_failed_worker_reports_its_error_or_exit_code(fclient, launcher):
    a = submit(fclient, name="a").json()["job_id"]
    wait_for(lambda: len(launcher.launched) == 1)
    launcher.finish(0, error="RuntimeError: out of memory", code=1)
    wait_for(lambda: status(fclient, a) == "FAILED")
    assert "out of memory" in fclient.get(f"{API}/{a}").json()["error"]

    b = submit(fclient, name="b").json()["job_id"]
    wait_for(lambda: len(launcher.launched) == 2)
    launcher.finish(1, code=3)
    wait_for(lambda: status(fclient, b) == "FAILED")
    assert "code 3" in fclient.get(f"{API}/{b}").json()["error"]


def test_resume_validation(fclient, launcher, job_settings):
    a = submit(fclient, name="a").json()["job_id"]
    wait_for(lambda: len(launcher.launched) == 1)
    assert fclient.post(f"{API}/{a}/resume").status_code == 409  # still running
    launcher.finish(0, result={"status": "stopped", "iteration": 0})
    wait_for(lambda: status(fclient, a) == "STOPPED")
    assert fclient.post(f"{API}/{a}/resume").status_code == 409  # no checkpoint was written
    assert fclient.post(f"{API}/{'0' * 12}/resume").status_code == 404


def test_metrics_and_logs_endpoints_tolerate_torn_lines(fclient, launcher):
    job_id = submit(fclient).json()["job_id"]
    wait_for(lambda: len(launcher.launched) == 1)
    job_dir = launcher.launched[0][0]
    rows = [{"iter": i, "train_loss": 3.0 - i / 20, "val_loss": 3.1 - i / 20, "elapsed_s": i}
            for i in (0, 10)]
    (job_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + '\n{"iter": 20, "tr')
    (job_dir / "log.txt").write_text("\n".join(f"line {i}" for i in range(10)))
    body = fclient.get(f"{API}/{job_id}/metrics").json()
    assert [r["iter"] for r in body["rows"]] == [0, 10]
    assert [r["iter"] for r in fclient.get(f"{API}/{job_id}/metrics?after_iter=0").json()["rows"]] == [10]
    job = fclient.get(f"{API}/{job_id}").json()
    assert job["latest"]["iter"] == 10 and job["progress"] == 0.5
    assert fclient.get(f"{API}/{job_id}/logs?tail=3").json()["lines"] == ["line 7", "line 8", "line 9"]


@pytest.mark.parametrize("bad", ["nope", "../x", "0" * 12, "ABCDEF123456"])
def test_unknown_job_ids_are_404(fclient, bad):
    for suffix in ("", "/metrics", "/logs"):
        assert fclient.get(f"{API}/{bad}{suffix}").status_code == 404
    assert fclient.post(f"{API}/{bad}/stop").status_code == 404


def test_training_can_be_disabled(job_settings, launcher):
    with TestClient(create_app(replace(job_settings, training_enabled=False),
                               job_launcher=launcher)) as c:
        r = submit(c)
        assert r.status_code == 403 and launcher.launched == []
        assert c.get(API).json() == {"jobs": []}  # reading is still allowed


def test_a_running_job_is_marked_failed_after_a_server_restart(job_settings, launcher):
    first = JobManager(job_settings, launcher=launcher, poll_seconds=0.05)
    first.start()
    job_id = first.submit(BODY)
    wait_for(lambda: first.view(job_id)["status"] == "RUNNING")
    first._shutdown.set()  # simulate the server dying: no clean shutdown of the job
    first._wake.set()
    first._thread.join()

    second = JobManager(job_settings, launcher=FakeLauncher(), poll_seconds=0.05)
    second.start()
    try:
        view = second.view(job_id)
        assert view["status"] == "FAILED" and "restarted" in view["error"]
    finally:
        second.shutdown()


def test_unresponsive_worker_is_terminated_after_the_grace_period(job_settings, launcher):
    mgr = JobManager(job_settings, launcher=launcher, grace_seconds=0.3, poll_seconds=0.05)
    mgr.start()
    try:
        job_id = mgr.submit(BODY)
        wait_for(lambda: len(launcher.launched) == 1)
        wait_for(lambda: mgr.view(job_id)["status"] == "RUNNING")
        mgr.stop(job_id)
        wait_for(lambda: mgr.view(job_id)["status"] == "STOPPED")
        assert launcher.launched[0][1].terminated
        assert "terminated" in mgr.view(job_id)["error"]
    finally:
        mgr.shutdown()


def test_real_worker_trains_then_resumes_through_the_api(settings, tmp_path):
    s = replace(settings, config_path=write_config(tmp_path), max_train_iters=1000)
    with TestClient(create_app(s)) as client:

        def finished(job_id):
            body = client.get(f"{API}/{job_id}").json()
            return body if body["status"] in ("COMPLETED", "FAILED", "STOPPED") else None

        r = client.post(API, json={**BODY, "name": "e2e"})
        assert r.status_code == 201
        first = wait_for(lambda: finished(r.json()["job_id"]), timeout=240)
        assert first["status"] == "COMPLETED", first["error"]
        assert first["result"]["parameters"] > 0 and first["progress"] == 1.0
        assert "e2e:best" in [m["model_id"] for m in client.get("/api/v1/models").json()["models"]]
        rows = client.get(f"{API}/{first['job_id']}/metrics").json()["rows"]
        assert [x["iter"] for x in rows] == [0, 10, 20]
        assert any("iter" in line for line in
                   client.get(f"{API}/{first['job_id']}/logs").json()["lines"])

        r2 = client.post(f"{API}/{first['job_id']}/resume", json={"max_iters": 40})
        assert r2.status_code == 201 and r2.json()["resumed_from"] == first["job_id"]
        second = wait_for(lambda: finished(r2.json()["job_id"]), timeout=240)
        assert second["status"] == "COMPLETED", second["error"]
        assert second["result"]["resumed_from_iteration"] == 20
        rows = client.get(f"{API}/{second['job_id']}/metrics").json()["rows"]
        assert [x["iter"] for x in rows] == [0, 10, 20, 30, 40]