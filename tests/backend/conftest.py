import pytest
import torch
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.main import create_app
from models.bigram import BigramLanguageModel
from models.gpt import GPT
from tests.backend.helpers import CHARS, GPT_KWARGS
from training.checkpoint import save_checkpoint


@pytest.fixture
def gpt_model() -> GPT:
    torch.manual_seed(0)
    return GPT(**GPT_KWARGS).eval()


@pytest.fixture
def settings(tmp_path, gpt_model) -> Settings:
    """Three real checkpoints on disk: tiny:best, tiny:last (GPT) and bg:best (bigram)."""
    runs = tmp_path / "runs"
    for kind, iteration in (("best", 10), ("last", 20)):
        save_checkpoint(
            runs / "tiny" / f"{kind}.pt", gpt_model, model_type="gpt",
            model_kwargs=GPT_KWARGS, chars=CHARS, iteration=iteration,
        )
    save_checkpoint(
        runs / "bg" / "best.pt", BigramLanguageModel(7), model_type="bigram",
        model_kwargs={"vocab_size": 7}, chars=CHARS, iteration=5,
    )
    return Settings(data_dir=tmp_path, device="cpu", max_new_tokens=100, max_loaded_models=2)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c