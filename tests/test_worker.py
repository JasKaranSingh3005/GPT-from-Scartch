import json
from dataclasses import asdict

import torch

from training.worker import config_from_dict, main
from utils.config import Config, DataConfig, ModelConfig, TrainingConfig


def make_cfg(text_path, max_iters=20) -> Config:
    return Config(
        data=DataConfig(url="unused", path=str(text_path), train_fraction=0.9),
        model=ModelConfig(n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.0),
        training=TrainingConfig(batch_size=4, learning_rate=1e-2, max_iters=max_iters,
                                eval_interval=10, eval_iters=2, seed=0),
    )


def write_spec(tmp_path, cfg, resume=False):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "spec.json").write_text(json.dumps({
        "job_id": "abc123abc123", "config": asdict(cfg), "model_type": "gpt", "run_name": "w",
        "out_root": str(tmp_path / "runs"), "device": "cpu", "resume": resume,
    }))
    return job_dir


def text_file(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("to be or not to be, that is the question. " * 120, encoding="utf-8")
    return p


def test_config_roundtrip():
    cfg = make_cfg("x.txt")
    assert config_from_dict(asdict(cfg)) == cfg


def test_worker_completes_and_reports_through_files(tmp_path):
    job_dir = write_spec(tmp_path, make_cfg(text_file(tmp_path)))
    assert main([str(job_dir)]) == 0
    result = json.loads((job_dir / "result.json").read_text())
    assert result["status"] == "completed" and result["parameters"] > 0
    rows = [json.loads(line) for line in (job_dir / "metrics.jsonl").read_text().splitlines()]
    assert [r["iter"] for r in rows] == [0, 10, 20] and rows[0]["job_id"] == "abc123abc123"
    assert (tmp_path / "runs" / "w" / "best.pt").is_file()


def test_stop_flag_stops_cleanly_and_leaves_a_resumable_run(tmp_path):
    cfg = make_cfg(text_file(tmp_path))
    job_dir = write_spec(tmp_path, cfg)
    (job_dir / "stop.flag").touch()
    assert main([str(job_dir)]) == 0
    assert json.loads((job_dir / "result.json").read_text())["status"] == "stopped"
    assert not (tmp_path / "runs" / "w" / "summary.json").exists()
    assert (tmp_path / "runs" / "w" / "last.pt").is_file()


def test_failure_writes_error_json_and_returns_1(tmp_path):
    cfg = make_cfg(tmp_path / "missing.txt")
    job_dir = write_spec(tmp_path, cfg)
    assert main([str(job_dir)]) == 1
    assert json.loads((job_dir / "error.json").read_text())["error"]
    assert not (job_dir / "result.json").exists()


def test_usage_error():
    assert main([]) == 2