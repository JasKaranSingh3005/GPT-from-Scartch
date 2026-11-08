import warnings

import pytest
import torch

from models.gpt import GPT
from training.checkpoint import (
    capture_rng_state, load_checkpoint, load_model, restore_rng_state, save_checkpoint,
)
from training.train import run_experiment, train_model
from utils.config import Config, DataConfig, ModelConfig, TrainingConfig


def tiny_gpt() -> GPT:
    torch.manual_seed(0)
    return GPT(vocab_size=4, n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.0)


def splits() -> dict[str, torch.Tensor]:
    pattern = torch.tensor([0, 1, 2, 3] * 200)
    return {"train": pattern, "val": pattern}


KW = dict(block_size=8, batch_size=4, learning_rate=1e-2, eval_interval=10, eval_iters=2,
          device="cpu")


def test_should_stop_ends_training_and_reports_the_iteration():
    stopped: list[int] = []
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 15  # false for iterations 0..14, true at iteration 15

    history = train_model(tiny_gpt(), splits(), max_iters=100, should_stop=stop,
                          on_stop=stopped.append, **KW)
    assert stopped == [15]
    assert [r["iter"] for r in history] == [0, 10]  # no evaluation after the stop


def test_stop_checked_before_the_final_iteration_is_not_needed():
    stopped: list[int] = []
    history = train_model(tiny_gpt(), splits(), max_iters=20, should_stop=lambda: False,
                          on_stop=stopped.append, **KW)
    assert stopped == [] and history[-1]["iter"] == 20


def test_resuming_does_not_duplicate_evaluation_rows():
    model = tiny_gpt()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = train_model(model, splits(), max_iters=20, optimizer=opt,
                        should_stop=lambda: True, **KW)  # stops at 0 after evaluating it
    assert [r["iter"] for r in first] == [0]
    full = train_model(model, splits(), max_iters=20, optimizer=opt, start_iter=0,
                       history=first, **KW)
    assert [r["iter"] for r in full] == [0, 10, 20]


def test_invalid_start_iter_raises():
    with pytest.raises(ValueError):
        train_model(tiny_gpt(), splits(), max_iters=10, start_iter=11, **KW)


def test_rng_state_roundtrip():
    state = capture_rng_state()
    a = torch.rand(5)
    restore_rng_state(state)
    assert torch.equal(a, torch.rand(5))


def make_cfg(tmp_path, max_iters: int) -> Config:
    text_file = tmp_path / "t.txt"
    if not text_file.exists():
        text_file.write_text("to be or not to be, that is the question. " * 120, encoding="utf-8")
    return Config(
        data=DataConfig(url="unused", path=str(text_file), train_fraction=0.9),
        model=ModelConfig(n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.1),
        training=TrainingConfig(batch_size=4, learning_rate=1e-2, max_iters=max_iters,
                                eval_interval=10, eval_iters=2, seed=0),
    )


def run(cfg, name, tmp_path, **kw):
    return run_experiment(cfg, "gpt", name, out_root=tmp_path / "runs",
                          use_tensorboard=False, device="cpu", **kw)


def stop_after(n: int):
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > n

    return stop


def test_stopped_run_is_resumable_and_matches_an_uninterrupted_run(tmp_path):
    cfg = make_cfg(tmp_path, 20)
    run(cfg, "full", tmp_path, resumable=True)

    stopped = run(cfg, "split", tmp_path, resumable=True, should_stop=stop_after(10))
    run_dir = tmp_path / "runs" / "split"
    assert stopped["status"] == "stopped" and stopped["iteration"] == 10
    assert not (run_dir / "summary.json").exists()  # never mistaken for a finished run
    assert (run_dir / "history.csv").is_file()
    ckpt = load_checkpoint(run_dir / "last.pt")  # loads with weights_only=True
    assert ckpt["iteration"] == 10 and ckpt["training_state"]["iteration"] == 10
    assert "optimizer" in ckpt["training_state"] and "rng" in ckpt["training_state"]

    done = run(cfg, "split", tmp_path, resume=True)
    assert done["status"] == "completed" and done["resumed_from_iteration"] == 10
    assert (run_dir / "summary.json").is_file()

    a = load_checkpoint(tmp_path / "runs" / "full" / "last.pt")
    b = load_checkpoint(run_dir / "last.pt")
    assert a["iteration"] == b["iteration"] == 20
    for name, tensor in a["state_dict"].items():
        assert torch.equal(tensor, b["state_dict"][name]), name  # exact continuation
    assert [r["iter"] for r in b["history"]] == [0, 10, 20]


def test_resume_errors(tmp_path):
    cfg = make_cfg(tmp_path, 20)
    with pytest.raises(FileNotFoundError):
        run(cfg, "ghost", tmp_path, resume=True)

    run(cfg, "plain", tmp_path)  # not resumable: no training state saved
    assert "training_state" not in load_checkpoint(tmp_path / "runs" / "plain" / "last.pt")
    with pytest.raises(ValueError, match="no training state"):
        run(cfg, "plain", tmp_path, resume=True)

    run(cfg, "rs", tmp_path, resumable=True, should_stop=stop_after(10))
    with pytest.raises(ValueError, match="already at iteration"):
        run(make_cfg(tmp_path, 10), "rs", tmp_path, resume=True)
    with pytest.raises(ValueError, match="Run is a"):
        run_experiment(cfg, "bigram", "rs", out_root=tmp_path / "runs",
                       use_tensorboard=False, device="cpu", resume=True)


def test_old_format_checkpoints_still_load(tmp_path):
    model = tiny_gpt()
    kwargs = dict(vocab_size=4, n_embd=16, n_head=2, n_layer=1, block_size=8, dropout=0.0)
    path = save_checkpoint(tmp_path / "old.pt", model, model_type="gpt", model_kwargs=kwargs,
                           chars=list("abcd"), iteration=3)
    loaded, tokenizer, ckpt = load_model(path)
    assert "training_state" not in ckpt and tokenizer.vocab_size == 4
    idx = torch.randint(4, (1, 5))
    assert torch.allclose(model.eval()(idx)[0], loaded(idx)[0], atol=1e-6)


def test_cuda_rng_mismatch_warns_instead_of_failing():
    state = capture_rng_state()
    state["cuda"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        restore_rng_state(state)  # no warning and no error when there is no CUDA state