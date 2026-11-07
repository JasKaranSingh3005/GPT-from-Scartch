import json

import pandas as pd
import pytest

from experiments.definitions import (
    EXPERIMENTS, VARIANTS, build_config, selected_variants,
)
from experiments.plot_results import collect_table, plot_experiment
from experiments.run_experiments import run_all
from utils.config import Config, DataConfig, ModelConfig, TrainingConfig
from utils.estimate import gpt_parameter_count


def base_config() -> Config:
    return Config(
        data=DataConfig(url="unused", path="unused.txt"),
        model=ModelConfig(),
        training=TrainingConfig(learning_rate=1e-3, seed=5),
    )


def test_every_experiment_references_defined_variants():
    for exp in EXPERIMENTS.values():
        assert exp.variants and all(v in VARIANTS for v in exp.variants)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_every_variant_builds_a_valid_config(name):
    cfg = build_config(base_config(), VARIANTS[name], max_iters=200)
    assert cfg.model.n_embd % cfg.model.n_head == 0
    assert cfg.training.max_iters == 200 and cfg.training.eval_interval == 20
    assert cfg.training.seed == 5  # seed comes from the base config


def test_each_gpt_variant_changes_exactly_one_thing():
    base = build_config(base_config(), VARIANTS["base"], 100).model
    for name, v in VARIANTS.items():
        if v.model_type != "gpt" or name == "base":
            continue
        assert len(v.overrides) == 1, name
        (field_name,) = v.overrides
        assert getattr(build_config(base_config(), v, 100).model, field_name) != getattr(base, field_name)


def test_bigram_gets_higher_learning_rate_than_gpt():
    assert build_config(base_config(), VARIANTS["bigram"], 100).training.learning_rate == 1e-2
    assert build_config(base_config(), VARIANTS["base"], 100).training.learning_rate == 1e-3


def test_selected_variants_dedupes_and_validates():
    names = selected_variants(["heads", "dropout"])
    assert names.count("base") == 1 and names[0] == "heads_1"
    assert set(selected_variants()) == set(VARIANTS)  # every variant is used somewhere
    with pytest.raises(ValueError):
        selected_variants(["nonsense"])


def test_head_count_does_not_change_parameters_but_layers_and_width_do():
    def params(name):
        m = build_config(base_config(), VARIANTS[name], 100).model
        return gpt_parameter_count(65, m.n_embd, m.n_layer, m.block_size)

    assert params("heads_1") == params("base") == params("heads_8")
    assert params("layers_1") < params("layers_2") < params("base")
    assert params("embd_32") < params("base") < params("embd_128")


def fake_run(cfg, model_type, name, out_root, use_tensorboard):
    run = out_root / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "summary.json").write_text("{}")
    return {"name": name}


def test_run_all_skips_finished_runs_and_force_reruns(tmp_path):
    done, failed = run_all(base_config(), ["base", "heads_1"], tmp_path, 100, run_fn=fake_run)
    assert list(done) == ["base", "heads_1"] and not failed
    done, _ = run_all(base_config(), ["base", "heads_1"], tmp_path, 100, run_fn=fake_run)
    assert done == {}
    done, _ = run_all(base_config(), ["base"], tmp_path, 100, force=True, run_fn=fake_run)
    assert list(done) == ["base"]


def test_run_all_continues_after_a_failure(tmp_path):
    def flaky(cfg, model_type, name, out_root, use_tensorboard):
        if name == "heads_1":
            raise RuntimeError("boom")
        return fake_run(cfg, model_type, name, out_root, use_tensorboard)

    done, failed = run_all(base_config(), ["heads_1", "base"], tmp_path, 100, run_fn=flaky)
    assert list(done) == ["base"] and failed == {"heads_1": "boom"}


def write_fake_run(out_dir, name, final_val, best_val):
    run = out_dir / name
    run.mkdir(parents=True)
    pd.DataFrame({"iter": [0, 10], "train_loss": [4.0, 2.0], "val_loss": [4.1, final_val],
                  "elapsed_s": [0.1, 1.0]}).to_csv(run / "history.csv", index=False)
    (run / "summary.json").write_text(json.dumps({
        "parameters": 123, "training_time_s": 1.234, "final_train_loss": 2.0,
        "final_val_loss": final_val, "best_val_loss": best_val, "best_iter": 10,
    }))


def test_collect_table_and_plot(tmp_path):
    write_fake_run(tmp_path, "base", 2.5, 2.4)
    write_fake_run(tmp_path, "heads_1", 2.6, 2.6)
    table = collect_table(tmp_path, list(VARIANTS))  # runs that don't exist are skipped
    assert set(table["run"]) == {"base", "heads_1"}
    row = table[table["run"] == "base"].iloc[0]
    assert row["best_val_loss"] == 2.4 and row["val_minus_train"] == pytest.approx(0.5)

    path = plot_experiment("heads", tmp_path, tmp_path / "plots")
    assert path is not None and path.is_file() and path.stat().st_size > 0
    assert plot_experiment("dropout", tmp_path / "empty", tmp_path / "plots") is None