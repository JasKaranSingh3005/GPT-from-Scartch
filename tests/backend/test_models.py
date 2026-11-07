import json
import os

import pytest
import torch

from models.tokenizer import CharTokenizer
from tests.backend.helpers import CHARS, GPT_KWARGS
from training.checkpoint import save_checkpoint

API = "/api/v1/models"


def bump_mtime(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))


def test_list_models(client):
    models = client.get(API).json()["models"]
    assert [m["model_id"] for m in models] == ["bg:best", "tiny:best", "tiny:last"]
    assert all(m["loaded"] is False and m["size_bytes"] > 0 for m in models)


def test_list_is_empty_when_there_are_no_runs(tmp_path):
    from fastapi.testclient import TestClient
    from backend.core.config import Settings
    from backend.main import create_app

    with TestClient(create_app(Settings(data_dir=tmp_path / "nothing", device="cpu"))) as c:
        assert c.get(API).json() == {"models": []}


def test_detail_comes_from_the_checkpoint(client, gpt_model):
    body = client.get(f"{API}/tiny:best").json()
    assert body["parameters"] == gpt_model.num_parameters()
    assert body["iteration"] == 10 and body["vocab_size"] == 7
    assert body["model_type"] == "gpt" and body["loaded"] is False
    assert body["run_best_val_loss"] is None  # no summary.json yet, so nothing is invented


def test_detail_includes_run_summary_when_present(client, settings):
    (settings.runs_dir / "tiny" / "summary.json").write_text(
        json.dumps({"best_val_loss": 1.5, "final_val_loss": 1.7, "training_time_s": 12.0})
    )
    body = client.get(f"{API}/tiny:best").json()
    assert body["run_best_val_loss"] == 1.5 and body["run_final_val_loss"] == 1.7
    assert body["run_training_time_s"] == 12.0


def test_config_endpoint(client):
    body = client.get(f"{API}/tiny:best/config").json()
    assert body["model_type"] == "gpt"
    assert body["model_kwargs"]["n_embd"] == 16
    assert body["head_size"] == 8 and body["context_length"] == 8
    bigram = client.get(f"{API}/bg:best/config").json()
    assert bigram["head_size"] is None and bigram["context_length"] is None


def test_parameters_endpoint_is_consistent_with_the_real_model(client, gpt_model):
    body = client.get(f"{API}/tiny:best/parameters").json()
    assert body["total"] == gpt_model.num_parameters()
    assert body["breakdown_matches_actual"] is True
    assert sum(body["breakdown"].values()) == body["total"]
    assert sum(t["numel"] for t in body["tensors"]) == body["total"]
    assert "token_embedding.weight" in {t["name"] for t in body["tensors"]}


def test_bigram_parameters_have_no_gpt_breakdown(client):
    body = client.get(f"{API}/bg:best/parameters").json()
    assert body["total"] == 49  # one 7 x 7 embedding table
    assert body["breakdown"] is None and body["breakdown_matches_actual"] is None


@pytest.mark.parametrize("suffix", ["", "/config", "/parameters"])
@pytest.mark.parametrize("bad_id", ["nope:best", "tiny:other", "tiny", "..:best", "tiny:best:x"])
def test_unknown_model_ids_are_404(client, bad_id, suffix):
    r = client.get(f"{API}/{bad_id}{suffix}")
    assert r.status_code == 404 and "detail" in r.json()


def test_load_and_unload_cycle(client):
    registry = client.app.state.registry
    r = client.post(f"{API}/load", json={"model_id": "tiny:best"})
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is True and body["device"] == "cpu" and body["parameters"] > 0
    assert client.get(f"{API}/tiny:best").json()["loaded"] is True

    first = registry.get_loaded("tiny:best")
    client.post(f"{API}/load", json={"model_id": "tiny:best"})
    assert registry.get_loaded("tiny:best") is first  # idempotent, not reloaded

    assert client.post(f"{API}/unload", json={"model_id": "tiny:best"}).json()["unloaded"] is True
    assert client.post(f"{API}/unload", json={"model_id": "tiny:best"}).json()["unloaded"] is False
    assert registry.get_loaded("tiny:best") is None


def test_loaded_model_reproduces_the_original_outputs(client, gpt_model):
    client.post(f"{API}/load", json={"model_id": "tiny:best"})
    loaded = client.app.state.registry.get_loaded("tiny:best")
    assert not loaded.model.training
    assert loaded.tokenizer.chars == CharTokenizer.from_text("abcdef\n").chars
    idx = torch.randint(7, (2, 5))
    assert torch.allclose(loaded.model(idx)[0], gpt_model(idx)[0], atol=1e-6)


def test_loaded_bigram_works(client):
    r = client.post(f"{API}/load", json={"model_id": "bg:best"})
    assert r.status_code == 200 and r.json()["parameters"] == 49


def test_load_limit_returns_409(client):
    assert client.post(f"{API}/load", json={"model_id": "tiny:best"}).status_code == 200
    assert client.post(f"{API}/load", json={"model_id": "tiny:last"}).status_code == 200
    assert client.post(f"{API}/load", json={"model_id": "bg:best"}).status_code == 409
    # an already-loaded model can still be (re)requested at the limit
    assert client.post(f"{API}/load", json={"model_id": "tiny:best"}).status_code == 200
    client.post(f"{API}/unload", json={"model_id": "tiny:last"})
    assert client.post(f"{API}/load", json={"model_id": "bg:best"}).status_code == 200


def test_corrupt_checkpoint_is_422_and_never_loaded(client, settings):
    bad = settings.runs_dir / "bad"
    bad.mkdir()
    (bad / "best.pt").write_bytes(b"not a checkpoint")
    assert "bad:best" in [m["model_id"] for m in client.get(API).json()["models"]]
    assert client.get(f"{API}/bad:best").status_code == 422
    r = client.post(f"{API}/load", json={"model_id": "bad:best"})
    assert r.status_code == 422 and "detail" in r.json()
    assert client.app.state.registry.loaded_ids() == []


@pytest.mark.parametrize("body", [{}, {"model_id": ""}, {"model_id": "x" * 200}])
def test_load_request_validation(client, body):
    assert client.post(f"{API}/load", json=body).status_code == 422


def test_unknown_model_load_is_404(client):
    assert client.post(f"{API}/load", json={"model_id": "ghost:best"}).status_code == 404


def test_metadata_refreshes_when_the_checkpoint_changes(client, settings, gpt_model):
    assert client.get(f"{API}/tiny:best").json()["iteration"] == 10
    path = settings.runs_dir / "tiny" / "best.pt"
    save_checkpoint(path, gpt_model, model_type="gpt", model_kwargs=GPT_KWARGS,
                    chars=CHARS, iteration=99)
    bump_mtime(path)
    assert client.get(f"{API}/tiny:best").json()["iteration"] == 99


def test_loaded_model_is_replaced_when_the_checkpoint_changes(client, settings, gpt_model):
    registry = client.app.state.registry
    client.post(f"{API}/load", json={"model_id": "tiny:best"})
    first = registry.get_loaded("tiny:best")
    path = settings.runs_dir / "tiny" / "best.pt"
    save_checkpoint(path, gpt_model, model_type="gpt", model_kwargs=GPT_KWARGS,
                    chars=CHARS, iteration=99)
    bump_mtime(path)
    client.post(f"{API}/load", json={"model_id": "tiny:best"})
    second = registry.get_loaded("tiny:best")
    assert second is not first and second.meta.iteration == 99