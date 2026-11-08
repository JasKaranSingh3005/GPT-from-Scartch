import pytest
import torch

API = "/api/v1/visualize"
BODY = {"model_id": "tiny:best", "prompt": "abcab\nf", "layer": 0}


def test_attention_returns_the_models_real_weights(client, gpt_model):
    r = client.post(f"{API}/attention", json=BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["n_layers"] == 2 and body["n_heads"] == 2 and body["block_size"] == 8
    assert [t["char"] for t in body["tokens"]] == list("abcab\nf")
    w = torch.tensor(body["weights"])
    assert w.shape == (2, 7, 7)
    assert torch.allclose(w.sum(-1), torch.ones(2, 7), atol=1e-3)
    assert torch.all(torch.triu(w, diagonal=1) == 0)  # causal mask is visible
    ids = torch.tensor([[t["id"] for t in body["tokens"]]])
    _, _, direct = gpt_model(ids, return_weights=True)
    assert torch.allclose(w, direct[0][0], atol=1e-4)  # same numbers as a direct model call


def test_second_layer_differs_from_first(client):
    a = client.post(f"{API}/attention", json=BODY).json()["weights"]
    b = client.post(f"{API}/attention", json={**BODY, "layer": 1}).json()["weights"]
    assert a != b


def test_long_prompt_is_cropped_to_block_size_and_flagged(client):
    r = client.post(f"{API}/attention", json={**BODY, "prompt": "abcdef" * 4}).json()
    assert r["cropped"] is True and len(r["tokens"]) == 8


def test_max_tokens_crops_further(client):
    r = client.post(f"{API}/attention", json={**BODY, "max_tokens": 3}).json()
    assert r["cropped"] is True
    assert "".join(t["char"] for t in r["tokens"]) == "b\nf"  # last 3 of "abcab\nf"


@pytest.mark.parametrize("patch,status", [
    ({"layer": 2}, 422),             # model has layers 0 and 1
    ({"prompt": ""}, 422),
    ({"prompt": "xyz"}, 422),
    ({"max_tokens": 0}, 422),
    ({"max_tokens": 500}, 422),
    ({"layer": -1}, 422),
    ({"model_id": "bg:best"}, 422),  # bigram has no attention
    ({"model_id": "ghost:best"}, 404),
])
def test_attention_rejects_bad_requests(client, patch, status):
    r = client.post(f"{API}/attention", json={**BODY, **patch})
    assert r.status_code == status and "detail" in r.json()


def test_forward_pass_reports_real_shapes(client):
    body = {"model_id": "tiny:best", "prompt": "abcab"}
    r = client.post(f"{API}/forward-pass", json=body)
    assert r.status_code == 200
    stages = {s["name"]: s for s in r.json()["stages"]}
    assert stages["token_embedding"]["shape"] == [1, 5, 16]
    assert stages["position_embedding"]["shape"] == [5, 16]
    assert stages["blocks.0"]["shape"] == [1, 5, 16] and stages["blocks.1"]["shape"] == [1, 5, 16]
    assert stages["ln_f"]["shape"] == [1, 5, 16]
    assert stages["lm_head"]["shape"] == [1, 5, 7]
    assert all(s["std"] >= 0 for s in stages.values())


def test_forward_pass_hooks_are_removed(client):
    body = {"model_id": "tiny:best", "prompt": "abc"}
    client.post(f"{API}/forward-pass", json=body)
    model = client.app.state.registry.get_loaded("tiny:best").model
    assert all(len(m._forward_hooks) == 0 for m in model.modules())
    assert len(client.post(f"{API}/forward-pass", json=body).json()["stages"]) == 6


def test_architecture_comes_from_the_checkpoint(client, gpt_model):
    r = client.get(f"{API}/architecture", params={"model_id": "tiny:best"})
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["n_embd"] == 16 and body["config"]["block_size"] == 8
    stages = {s["key"]: s for s in body["stages"]}
    assert stages["token_embedding"]["output_shape"] == "(B, T, 16)"
    assert stages["lm_head"]["output_shape"] == "(B, T, 7)"
    assert stages["blocks"]["details"]["head_size"] == 8
    assert body["parameters_total"] == gpt_model.num_parameters()
    assert body["stage_sum_matches_total"] is True
    assert sum(s["parameters"] for s in body["stages"]) == body["parameters_total"]


def test_architecture_errors(client):
    assert client.get(f"{API}/architecture", params={"model_id": "bg:best"}).status_code == 422
    assert client.get(f"{API}/architecture", params={"model_id": "ghost:best"}).status_code == 404