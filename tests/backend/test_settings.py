import pytest
import torch

from backend.core.config import PROJECT_ROOT, Settings, resolve_device
from utils.device import get_device


def test_defaults():
    s = Settings.from_env({})
    assert s.data_dir == PROJECT_ROOT / "outputs"
    assert s.runs_dir == PROJECT_ROOT / "outputs" / "runs"
    assert s.device == "auto" and s.max_new_tokens == 1000 and s.max_loaded_models == 3


def test_overrides(tmp_path):
    s = Settings.from_env({
        "GPTLAB_DATA_DIR": str(tmp_path), "GPTLAB_DEVICE": "CPU",
        "GPTLAB_MAX_NEW_TOKENS": "50", "GPTLAB_MAX_LOADED": "1",
        "GPTLAB_CORS_ORIGINS": "http://a.test, http://b.test",
    })
    assert s.data_dir == tmp_path and s.device == "cpu"
    assert s.max_new_tokens == 50 and s.max_loaded_models == 1
    assert s.cors_origins == ("http://a.test", "http://b.test")


def test_relative_data_dir_is_resolved_against_project_root():
    assert Settings.from_env({"GPTLAB_DATA_DIR": "custom"}).data_dir == PROJECT_ROOT / "custom"


@pytest.mark.parametrize("env", [
    {"GPTLAB_MAX_NEW_TOKENS": "abc"}, {"GPTLAB_MAX_NEW_TOKENS": "0"},
    {"GPTLAB_MAX_LOADED": "-1"}, {"GPTLAB_DEVICE": "tpu"},
])
def test_invalid_environment_is_rejected(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto") == get_device()


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available here")
def test_cuda_request_fails_when_cuda_is_unavailable():
    with pytest.raises(ValueError):
        resolve_device("cuda")


def test_training_settings_defaults_and_overrides(tmp_path):
    s = Settings.from_env({})
    assert s.training_enabled is True and s.max_queued_jobs == 4 and s.max_train_iters == 20000
    assert s.jobs_dir == s.data_dir / "jobs" and s.db_path == s.data_dir / "jobs.sqlite3"
    s = Settings.from_env({"GPTLAB_TRAINING_ENABLED": "off", "GPTLAB_MAX_QUEUED": "2",
                           "GPTLAB_MAX_TRAIN_ITERS": "500", "GPTLAB_CONFIG": str(tmp_path / "c.yaml")})
    assert s.training_enabled is False and s.max_queued_jobs == 2 and s.max_train_iters == 500
    assert s.config_path == tmp_path / "c.yaml"


@pytest.mark.parametrize("env", [{"GPTLAB_TRAINING_ENABLED": "maybe"}, {"GPTLAB_MAX_QUEUED": "0"}])
def test_invalid_training_environment_is_rejected(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)