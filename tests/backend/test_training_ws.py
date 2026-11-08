import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.main import create_app
from backend.websocket.training import TrainingStream
from tests.backend.test_training_jobs import (
    API, BODY, FakeLauncher, wait_for, write_config,
)


@pytest.fixture
def job_settings(settings, tmp_path):
    return replace(settings, config_path=write_config(tmp_path), max_queued_jobs=2,
                   max_train_iters=1000)


def row(i, val):
    return {"iter": i, "train_loss": val + 0.05, "val_loss": val, "elapsed_s": float(i)}


def test_stream_emits_started_metrics_checkpoint_and_terminal_events(job_settings):
    launcher = FakeLauncher()
    with TestClient(create_app(job_settings, job_launcher=launcher)) as client:
        job_id = client.post(API, json=BODY).json()["job_id"]
        wait_for(lambda: len(launcher.launched) == 1)
        wait_for(lambda: client.get(f"{API}/{job_id}").json()["status"] == "RUNNING")
        job_dir = launcher.launched[0][0]
        stream = TrainingStream(client.app.state.jobs, job_id)

        events, done = stream.poll()
        assert [e["event"] for e in events] == ["training.started"] and not done

        lines = [row(0, 4.1), row(10, 3.0)]
        (job_dir / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
        events, done = stream.poll()
        assert [e["event"] for e in events] == [
            "training.metrics", "training.checkpoint_saved",
            "training.metrics", "training.checkpoint_saved"]
        assert events[0]["data"]["iter"] == 0 and events[2]["data"]["val_loss"] == 3.0

        with (job_dir / "metrics.jsonl").open("a") as f:  # worse than best: no checkpoint event
            f.write(json.dumps(row(20, 3.5)) + "\n")
        events, _ = stream.poll()
        assert [e["event"] for e in events] == ["training.metrics"]

        FakeLauncher.finish(launcher, 0, result={"status": "completed"})
        wait_for(lambda: client.get(f"{API}/{job_id}").json()["status"] == "COMPLETED")
        events, done = stream.poll()
        assert done and events[-1]["event"] == "training.completed"


def test_websocket_replays_history_and_closes_after_the_terminal_event(job_settings):
    launcher = FakeLauncher()
    with TestClient(create_app(job_settings, job_launcher=launcher)) as client:
        job_id = client.post(API, json=BODY).json()["job_id"]
        wait_for(lambda: len(launcher.launched) == 1)
        job_dir = launcher.launched[0][0]
        (job_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(r) for r in [row(0, 4.1), row(10, 3.0)]) + "\n")
        launcher.finish(0, result={"status": "completed"})
        wait_for(lambda: client.get(f"{API}/{job_id}").json()["status"] == "COMPLETED")

        with client.websocket_connect(f"/ws/training/{job_id}") as ws:
            names = []
            while True:
                try:
                    names.append(ws.receive_json()["event"])
                except WebSocketDisconnect:
                    break
        assert names[0] == "training.started" and names[-1] == "training.completed"
        assert names.count("training.metrics") == 2


def test_unknown_job_gets_an_error_event_and_code_4404(job_settings):
    with TestClient(create_app(job_settings, job_launcher=FakeLauncher())) as client:
        with client.websocket_connect("/ws/training/" + "0" * 12) as ws:
            assert ws.receive_json()["event"] == "error"
            with pytest.raises(WebSocketDisconnect) as info:
                ws.receive_json()
            assert info.value.code == 4404


def test_foreign_origin_is_rejected(job_settings):
    with TestClient(create_app(job_settings, job_launcher=FakeLauncher())) as client:
        with pytest.raises(WebSocketDisconnect) as info:
            with client.websocket_connect("/ws/training/" + "0" * 12,
                                          headers={"origin": "http://evil.example"}):
                pass
        assert info.value.code == 4403