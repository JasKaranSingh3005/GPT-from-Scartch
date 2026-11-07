import csv
import json

import pytest
import torch

from models.gpt import GPT
from training.checkpoint import load_model
from training.evaluate import estimate_loss
from training.train import run_experiment, train_model
from utils.config import Config, DataConfig, ModelConfig, TrainingConfig


def tiny_splits() -> dict[str, torch.Tensor]:
    pattern = torch.tensor([0, 1, 2, 3] * 200)  # fully predictable sequence
    return {"train": pattern, "val": pattern}


def tiny_gpt() -> GPT:
    return GPT(vocab_size=4, n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.0)


def test_train_model_records_history_and_learns():
    torch.manual_seed(0)
    history = train_model(
        tiny_gpt(), tiny_splits(), block_size=8, batch_size=8, learning_rate=1e-2,
        max_iters=60, eval_interval=20, eval_iters=5, device="cpu",
    )
    assert [r["iter"] for r in history] == [0, 20, 40, 60]
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["val_loss"] < history[0]["val_loss"]


def test_on_best_is_called_at_first_evaluation():
    calls: list[int] = []
    train_model(
        tiny_gpt(), tiny_splits(), block_size=8, batch_size=4, learning_rate=1e-2,
        max_iters=20, eval_interval=10, eval_iters=2, device="cpu", on_best=calls.append,
    )
    assert calls and calls[0] == 0


def test_invalid_intervals_raise():
    with pytest.raises(ValueError):
        train_model(
            tiny_gpt(), tiny_splits(), block_size=8, batch_size=4, learning_rate=1e-2,
            max_iters=10, eval_interval=0, eval_iters=2, device="cpu",
        )


def test_estimate_loss_restores_train_eval_mode():
    model = tiny_gpt()
    args = (tiny_splits(), 8, 4, 2, "cpu")
    model.train()
    estimate_loss(model, *args)
    assert model.training
    model.eval()
    estimate_loss(model, *args)
    assert not model.training


def test_run_experiment_writes_all_outputs(tmp_path):
    text_file = tmp_path / "tiny.txt"
    text_file.write_text("to be or not to be, that is the question. " * 120, encoding="utf-8")
    cfg = Config(
        data=DataConfig(url="unused", path=str(text_file), train_fraction=0.9),
        model=ModelConfig(n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.0),
        training=TrainingConfig(
            batch_size=4, learning_rate=1e-2, max_iters=20, eval_interval=10,
            eval_iters=2, seed=0,
        ),
    )
    summary = run_experiment(cfg, "gpt", "t", out_root=tmp_path / "runs", use_tensorboard=False)

    run_dir = tmp_path / "runs" / "t"
    for name in ("history.csv", "summary.json", "best.pt", "last.pt"):
        assert (run_dir / name).is_file()
    with (run_dir / "history.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 3  # iters 0, 10, 20
    assert json.loads((run_dir / "summary.json").read_text())["parameters"] == summary["parameters"]
    model, tokenizer, _ = load_model(run_dir / "last.pt")
    assert model.num_parameters() == summary["parameters"]
    assert tokenizer.vocab_size == summary["vocab_size"]