import json
from dataclasses import asdict

from experiments.run_experiments import run_all
from training.worker import main
from utils.config import Config, DataConfig, ModelConfig, TrainingConfig


def base_cfg(text_path) -> Config:
    return Config(
        data=DataConfig(url="unused", path=str(text_path), train_fraction=0.9),
        model=ModelConfig(),
        training=TrainingConfig(batch_size=4, learning_rate=1e-3, max_iters=2, eval_interval=1,
                                eval_iters=2, seed=0),
    )


def write_spec(tmp_path, text_path, experiments=("bigram_vs_gpt",)):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "spec.json").write_text(json.dumps({
        "job_id": "abc123abc123", "kind": "experiment", "config": asdict(base_cfg(text_path)),
        "model_type": "experiment", "run_name": "exp-it2_s0", "device": "cpu", "resume": False,
        "out_root": str(tmp_path / "experiments"), "batch_id": "it2_s0",
        "experiments": list(experiments), "max_iters": 2, "seed": 0, "force": False,
    }))
    return job_dir


def text_file(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("to be or not to be, that is the question. " * 120, encoding="utf-8")
    return p


def test_worker_runs_a_batch_with_the_existing_runner(tmp_path):
    job_dir = write_spec(tmp_path, text_file(tmp_path))
    assert main([str(job_dir)]) == 0
    result = json.loads((job_dir / "result.json").read_text())
    assert result["status"] == "completed" and sorted(result["variants_run"]) == ["base", "bigram"]
    out = tmp_path / "experiments" / "it2_s0"
    assert (out / "bigram" / "summary.json").is_file() and (out / "base" / "summary.json").is_file()
    batch = json.loads((out / "batch.json").read_text())
    assert batch["seed"] == 0 and batch["variants"] == ["bigram", "base"]
    assert json.loads((job_dir / "progress.json").read_text())["total"] == 2


def test_rerunning_a_batch_skips_finished_variants(tmp_path):
    text = text_file(tmp_path)
    assert main([str(write_spec(tmp_path, text))]) == 0
    second = tmp_path / "job2"
    second.mkdir()
    (second / "spec.json").write_text((tmp_path / "job" / "spec.json").read_text())
    assert main([str(second)]) == 0
    assert json.loads((second / "result.json").read_text())["variants_run"] == []


def test_stop_flag_stops_before_running_anything(tmp_path):
    job_dir = write_spec(tmp_path, text_file(tmp_path))
    (job_dir / "stop.flag").touch()
    assert main([str(job_dir)]) == 0
    assert json.loads((job_dir / "result.json").read_text())["status"] == "stopped"
    assert not (tmp_path / "experiments" / "it2_s0" / "bigram" / "summary.json").exists()


def test_failed_variants_fail_the_job_and_are_named(tmp_path):
    job_dir = write_spec(tmp_path, tmp_path / "missing.txt")
    assert main([str(job_dir)]) == 1
    err = json.loads((job_dir / "error.json").read_text())["error"]
    assert "bigram" in err and "base" in err
    assert not (job_dir / "result.json").exists()


def test_run_all_hooks_stop_the_batch_and_report_progress(tmp_path):
    started: list[str] = []
    seen: list[tuple[str, int, int]] = []

    def run_fn(cfg, model_type, name, out_root, use_tensorboard, should_stop=None):
        started.append(name)
        (out_root / name).mkdir(parents=True, exist_ok=True)
        (out_root / name / "summary.json").write_text("{}")
        return {"status": "completed"}

    cfg = base_cfg(tmp_path / "x.txt")
    run_all(cfg, ["bigram", "base", "heads_1"], tmp_path, 10, run_fn=run_fn,
            should_stop=lambda: len(started) >= 1, on_variant=lambda *a: seen.append(a))
    assert started == ["bigram"] and seen == [("bigram", 1, 3)]