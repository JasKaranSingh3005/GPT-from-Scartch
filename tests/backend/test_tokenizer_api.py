API = "/api/v1/tokenizer"


def test_encode_returns_ids_tokens_and_context_info(client):
    r = client.post(f"{API}/encode", json={"model_id": "tiny:best", "text": "ab\nf"})
    assert r.status_code == 200
    body = r.json()
    assert body["ids"] == [1, 2, 0, 6]  # sorted vocabulary: "\n"=0 a=1 b=2 c=3 d=4 e=5 f=6
    assert body["length"] == 4 and body["vocab_size"] == 7 and body["block_size"] == 8
    assert [t["char"] for t in body["tokens"]] == ["a", "b", "\n", "f"]


def test_bigram_has_no_block_size(client):
    body = client.post(f"{API}/encode", json={"model_id": "bg:best", "text": "a"}).json()
    assert body["block_size"] is None


def test_encode_unknown_characters_is_422_and_names_them(client):
    r = client.post(f"{API}/encode", json={"model_id": "tiny:best", "text": "axz"})
    assert r.status_code == 422 and "x" in r.json()["detail"]


def test_decode_roundtrip_and_errors(client):
    ids = client.post(f"{API}/encode", json={"model_id": "tiny:best", "text": "fade"}).json()["ids"]
    r = client.post(f"{API}/decode", json={"model_id": "tiny:best", "ids": ids})
    assert r.json()["text"] == "fade"
    assert client.post(f"{API}/decode", json={"model_id": "tiny:best", "ids": [0, 99]}).status_code == 422


def test_vocabulary_matches_the_checkpoint(client):
    r = client.get(f"{API}/vocabulary", params={"model_id": "tiny:best"})
    body = r.json()
    assert body["vocab_size"] == 7 and [e["id"] for e in body["entries"]] == list(range(7))
    assert body["entries"][0] == {"id": 0, "char": "\n", "codepoint": "U+000A"}


def test_tokenizer_works_whether_or_not_the_model_is_loaded(client):
    cold = client.get(f"{API}/vocabulary", params={"model_id": "tiny:best"}).json()
    client.post("/api/v1/models/load", json={"model_id": "tiny:best"})
    warm = client.get(f"{API}/vocabulary", params={"model_id": "tiny:best"}).json()
    assert cold == warm


def test_unknown_model_is_404(client):
    assert client.get(f"{API}/vocabulary", params={"model_id": "ghost:best"}).status_code == 404
    assert client.post(f"{API}/encode", json={"model_id": "ghost:best", "text": "a"}).status_code == 404