import pytest
import torch

from models.sampling import next_token_probs
from models.tokenizer import CharTokenizer
from tests.backend.helpers import CHARS

API = "/api/v1/generate/distribution"
BODY = {"model_id": "tiny:best", "prompt": "abc", "temperature": 1.0, "top_n": 7}


def test_distribution_matches_the_models_own_probabilities(client, gpt_model):
    body = client.post(API, json={**BODY, "temperature": 0.8}).json()
    ids = CharTokenizer(CHARS).encode("abc")
    with torch.no_grad():
        logits, _ = gpt_model(torch.tensor([ids]))
    probs = next_token_probs(logits[:, -1, :], 0.8, None)[0]
    top = body["entries"][0]
    assert top["id"] == int(probs.argmax())
    assert top["probability"] == pytest.approx(float(probs.max()), abs=1e-5)
    values = [e["probability"] for e in body["entries"]]
    assert values == sorted(values, reverse=True)
    assert sum(values) == pytest.approx(1.0, abs=1e-4)
    assert body["shown_mass"] == pytest.approx(1.0, abs=1e-4)


def test_top_k_leaves_exactly_k_candidates_with_probability(client):
    entries = client.post(API, json={**BODY, "top_k": 2}).json()["entries"]
    assert sum(1 for e in entries if e["probability"] > 0) == 2


def test_lower_temperature_concentrates_probability(client):
    cold = client.post(API, json={**BODY, "temperature": 0.5}).json()["entries"][0]
    hot = client.post(API, json={**BODY, "temperature": 2.0}).json()["entries"][0]
    assert cold["probability"] > hot["probability"]


def test_defaults_prompt_cropping_and_bigram(client):
    body = client.post(API, json={"model_id": "tiny:best", "prompt": "abc"}).json()
    assert len(body["entries"]) == 7 and body["prompt_tokens"] == 3 and body["cropped"] is False
    long = client.post(API, json={**BODY, "prompt": "abcdef" * 3}).json()  # 18 > block_size 8
    assert long["cropped"] is True and long["prompt_tokens"] == 8
    assert client.post(API, json={**BODY, "prompt": ""}).json()["prompt_tokens"] == 1
    bigram = client.post(API, json={**BODY, "model_id": "bg:best"})
    assert bigram.status_code == 200 and bigram.json()["cropped"] is False


@pytest.mark.parametrize("patch,status", [
    ({"temperature": 0}, 422), ({"temperature": 6}, 422), ({"top_k": 0}, 422),
    ({"top_n": 0}, 422), ({"top_n": 101}, 422), ({"prompt": "xyz"}, 422),
    ({"prompt": "a" * 4001}, 422), ({"model_id": "ghost:best"}, 404),
])
def test_invalid_requests(client, patch, status):
    r = client.post(API, json={**BODY, **patch})
    assert r.status_code == status and "detail" in r.json()