import torch

from backend import __version__


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["version"] == __version__
    assert body["uptime_seconds"] >= 0


def test_system_info_reports_the_real_environment(client):
    body = client.get("/api/v1/system/info").json()
    assert body["torch_version"] == torch.__version__
    assert body["cuda_available"] == torch.cuda.is_available()
    assert body["device"] == "cpu"  # the test settings force cpu
    expected_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    assert len(body["gpus"]) == expected_gpus
    assert body["checkpoints_found"] == 3 and body["models_loaded"] == 0
    assert body["runs_dir_exists"] is True
    assert body["limits"] == {"max_new_tokens": 100, "max_loaded_models": 2}


def test_cors_preflight_allows_the_dev_frontend(client):
    r = client.options("/api/v1/health", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET",
    })
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_openapi_document_lists_the_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/v1/health", "/api/v1/system/info", "/api/v1/models", "/api/v1/models/load"):
        assert p in paths