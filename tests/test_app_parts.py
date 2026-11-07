from pathlib import Path

import pandas as pd
import pytest
import torch
from streamlit.testing.v1 import AppTest

from app.components.architecture_page import layernorm_demo, scaling_demo
from app.components.config_page import build_train_command
from app.components.runs import (
    list_runs,
    load_history,
    read_summary,
    summary_row,
    validate_run_name,
)
from models.gpt import GPT
from training.train import run_experiment
from utils.config import Config, DataConfig, ModelConfig, TrainingConfig
from utils.estimate import (
    gpt_parameter_breakdown,
    gpt_parameter_count,
    training_memory_estimate_mb,
    valid_head_counts,
)

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
RUNS = ROOT / "outputs" / "runs"


@pytest.mark.parametrize("v,c,h,l,bs", [(11, 16, 4, 2, 8), (65, 64, 4, 4, 64), (65, 48, 3, 1, 32)])
def test_parameter_estimate_matches_real_model(v, c, h, l, bs):
    real = GPT(vocab_size=v, n_embd=c, n_head=h, n_layer=l, block_size=bs).num_parameters()
    assert gpt_parameter_count(v, c, l, bs) == real
    assert sum(gpt_parameter_breakdown(v, c, l, bs).values()) == real


def test_head_count_does_not_change_parameters():
    a = GPT(vocab_size=11, n_embd=16, n_head=1, n_layer=2, block_size=8).num_parameters()
    b = GPT(vocab_size=11, n_embd=16, n_head=8, n_layer=2, block_size=8).num_parameters()
    assert a == b


def test_valid_head_counts_and_memory():
    assert valid_head_counts(64, [1, 2, 3, 4, 6, 8]) == [1, 2, 4, 8]
    mem = training_memory_estimate_mb(1_000_000, 32, 64, 4, 4)
    assert mem["Gradients"] == mem["Weights"] and mem["AdamW state"] == 2 * mem["Weights"]


def test_build_train_command_contains_every_setting():
    cmd = build_train_command(
        "x", ModelConfig(n_embd=128, n_head=4, n_layer=3, block_size=32, dropout=0.2),
        TrainingConfig(batch_size=16, learning_rate=3e-4, max_iters=500),
    )
    for part in ("--name x", "--n-embd 128", "--n-head 4", "--n-layer 3", "--block-size 32",
                 "--dropout 0.2", "--batch-size 16", "--lr 0.0003", "--max-iters 500"):
        assert part in cmd


def test_scaling_demo_shows_why_we_scale():
    r = scaling_demo(256)
    assert r["std_raw"] > 10 and abs(r["std_scaled"] - 1.0) < 0.15
    assert r["max_weight_raw"] > r["max_weight_scaled"]


def test_layernorm_demo_normalizes():
    before, after = layernorm_demo(shift=5.0, scale=3.0)
    assert abs(after.mean().item()) < 1e-4
    assert abs(after.std(unbiased=False).item() - 1.0) < 1e-2


def test_run_helpers(tmp_path):
    run = tmp_path / "r1"
    run.mkdir()
    pd.DataFrame({"iter": [0, 10], "train_loss": [4.0, 2.0], "val_loss": [4.1, 2.5],
                  "elapsed_s": [0.1, 1.0]}).to_csv(run / "history.csv", index=False)
    (tmp_path / "empty").mkdir()
    assert list_runs(tmp_path) == ["r1"]
    hist = load_history("r1", tmp_path)
    row = summary_row("r1", hist, None)
    assert row["best iter"] == 10 and row["parameters"] is None
    assert row["val - train"] == pytest.approx(0.5)
    assert read_summary(run) is None
    (run / "summary.json").write_text("{broken")
    assert read_summary(run) is None
    (run / "summary.json").write_text('{"parameters": 5, "model_type": "gpt"}')
    assert summary_row("r1", hist, read_summary(run))["parameters"] == 5


def test_validate_run_name(tmp_path):
    (tmp_path / "taken").mkdir()
    assert validate_run_name("fresh_1", tmp_path) is None
    assert validate_run_name("taken", tmp_path) is not None
    assert validate_run_name("../evil", tmp_path) is not None
    assert validate_run_name("", tmp_path) is not None


def test_run_experiment_reports_progress(tmp_path):
    text_file = tmp_path / "t.txt"
    text_file.write_text("to be or not to be, that is the question. " * 120, encoding="utf-8")
    cfg = Config(
        data=DataConfig(url="unused", path=str(text_file), train_fraction=0.9),
        model=ModelConfig(n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.0),
        training=TrainingConfig(batch_size=4, learning_rate=1e-2, max_iters=20,
                                eval_interval=10, eval_iters=2, seed=0),
    )
    seen: list[float] = []
    run_experiment(cfg, "gpt", "p", out_root=tmp_path / "runs", use_tensorboard=False,
                   on_eval=lambda row: seen.append(row["iter"]))
    assert seen == [0, 10, 20]


def test_pages_that_need_no_checkpoints_render():
    at = AppTest.from_file(APP, default_timeout=120).run()
    for page in ("Model Configuration", "Training Dashboard", "Architecture Explorer"):
        at.radio(key="nav_page").set_value(page).run()
        assert not at.exception, page
    at.radio(key="nav_page").set_value("Model Configuration").run()
    at.button(key="cfg_verify").click().run()
    assert not at.exception and not at.error


@pytest.mark.skipif(
    not ((RUNS / "gpt_small" / "best.pt").is_file()
         and (RUNS / "bigram_baseline" / "best.pt").is_file()),
    reason="needs gpt_small and bigram_baseline runs",
)
def test_comparison_page_generates_for_both_models():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="nav_page").set_value("Model Comparison").run()
    assert not at.exception
    at.slider(key="cmp_len").set_value(50)
    at.button(key="cmp_button").click().run()
    assert not at.exception
    result = at.session_state["cmp_result"]
    assert len(result["bigram"]) >= 50 and len(result["gpt"]) >= 50