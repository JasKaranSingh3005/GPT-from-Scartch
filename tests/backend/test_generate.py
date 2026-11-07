import json
from backend.services.inference import GenerationParams, stream_tokens
import pytest

API = "/api/v1/generate"
BODY = {"model_id": "tiny:best", "prompt": "ab", "max_new_tokens": 12,
        "temperature": 0.8, "top_k": 4, "seed": 7}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_generate_returns_real_text_and_metadata(client):
    r = client.post(API, json=BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["tokens_generated"] == 12 and body["prompt_tokens"] == 2
    assert body["text"] == "ab" + body["completion"] and len(body["completion"]) == 12
    assert set(body["completion"]) <= set("abcdef\n")
    assert body["device"] == "cpu" and body["seed"] == 7 and body["seed_note"]
    assert body["seconds"] > 0 and body["tokens_per_second"] > 0


def test_generate_loads_the_model_on_demand(client):
    assert client.get("/api/v1/models/tiny:best").json()["loaded"] is False
    client.post(API, json=BODY)
    assert client.get("/api/v1/models/tiny:best").json()["loaded"] is True


def test_same_seed_gives_same_text_and_different_seed_differs(client):
    a = client.post(API, json=BODY).json()["completion"]
    b = client.post(API, json=BODY).json()["completion"]
    c = client.post(API, json={**BODY, "seed": 8, "max_new_tokens": 60}).json()["completion"]
    d = client.post(API, json={**BODY, "seed": 7, "max_new_tokens": 60}).json()["completion"]
    assert a == b and c != d


def test_unseeded_generation_has_no_seed_note(client):
    body = client.post(API, json={**BODY, "seed": None}).json()
    assert body["seed"] is None and body["seed_note"] is None


def test_generation_works_for_the_bigram(client):
    r = client.post(API, json={**BODY, "model_id": "bg:best"})
    assert r.status_code == 200 and r.json()["tokens_generated"] == 12


def test_empty_prompt_is_allowed(client):
    r = client.post(API, json={**BODY, "prompt": ""})
    assert r.status_code == 200 and r.json()["prompt_tokens"] == 1


@pytest.mark.parametrize("patch,status", [
    ({"max_new_tokens": 101}, 422),        # server limit in test settings is 100
    ({"max_new_tokens": 0}, 422),
    ({"temperature": 0}, 422),
    ({"temperature": 99}, 422),
    ({"top_k": 0}, 422),
    ({"seed": -1}, 422),
    ({"prompt": "xyz"}, 422),              # characters outside the vocabulary
    ({"prompt": "a" * 4001}, 422),
    ({"model_id": "ghost:best"}, 404),
])
def test_invalid_requests_are_rejected_cleanly(client, patch, status):
    r = client.post(API, json={**BODY, **patch})
    assert r.status_code == status and "detail" in r.json()


def test_unknown_characters_are_named_in_the_error(client):
    detail = client.post(API, json={**BODY, "prompt": "axz"}).json()["detail"]
    assert "x" in detail and "z" in detail


def test_stream_emits_start_tokens_done_and_matches_the_non_stream_text(client):
    r = client.post(f"{API}/stream", json=BODY)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [e for e, _ in events]
    assert names[0] == "start" and names[-1] == "done"
    assert names.count("token") == 12
    streamed = "".join(d["text"] for e, d in events if e == "token")
    assert streamed == client.post(API, json=BODY).json()["completion"]
    assert events[-1][1]["tokens_generated"] == 12
    assert [d["index"] for e, d in events if e == "token"] == list(range(1, 13))


def test_stream_validation_errors_are_normal_http_errors(client):
    assert client.post(f"{API}/stream", json={**BODY, "prompt": "xyz"}).status_code == 422
    assert client.post(f"{API}/stream", json={**BODY, "max_new_tokens": 500}).status_code == 422
    assert client.post(f"{API}/stream", json={**BODY, "model_id": "ghost:best"}).status_code == 404


def test_compare_uses_one_seed_for_every_variant(client):
    body = {"model_id": "tiny:best", "prompt": "a", "max_new_tokens": 30, "seed": 3,
            "variants": [{"label": "same-1"}, {"label": "same-2"},
                         {"label": "cold", "temperature": 0.1}]}
    r = client.post(f"{API}/compare", json=body)
    assert r.status_code == 200
    out = {x["label"]: x["result"] for x in r.json()["results"]}
    assert r.json()["seed"] == 3
    assert out["same-1"]["completion"] == out["same-2"]["completion"]  # controlled comparison
    assert out["cold"]["seed"] == 3


def test_compare_can_mix_models(client):
    body = {"model_id": "tiny:best", "prompt": "a", "max_new_tokens": 10,
            "variants": [{"label": "gpt"}, {"label": "bigram", "model_id": "bg:best"}]}
    results = client.post(f"{API}/compare", json=body).json()["results"]
    assert [x["result"]["model_id"] for x in results] == ["tiny:best", "bg:best"]


@pytest.mark.parametrize("variants", [
    [{"label": "only"}],
    [{"label": "x"}, {"label": "x"}],
])
def test_compare_validation(client, variants):
    body = {"model_id": "tiny:best", "variants": variants}
    assert client.post(f"{API}/compare", json=body).status_code == 422
    
def test_abandoned_stream_does_not_hold_the_inference_lock(client):
    registry = client.app.state.registry
    loaded = registry.load("tiny:best")
    stream = stream_tokens(registry, loaded, GenerationParams("ab", 50, 0.8, None, 1))
    next(stream)
    next(stream)
    # The stream is deliberately NOT closed, simulating a client that vanished.
    assert registry.inference_lock.acquire(timeout=1), "lock still held by abandoned stream"
    registry.inference_lock.release()
    assert stream is not None  # keep the reference alive until here