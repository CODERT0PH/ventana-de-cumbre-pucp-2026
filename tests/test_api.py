from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_and_routes():
    assert client.get("/health").json()["ok"] is True
    rutas = client.get("/rutas").json()
    assert {r["id"] for r in rutas} >= {"laguna69", "pisco"}


def test_plan_unknown_route():
    assert client.get("/plan", params={"ruta_id": "nope"}).status_code == 404


def test_plan_without_model_says_what_is_missing(monkeypatch, tmp_path):
    import api.main as m
    monkeypatch.setattr(m, "WEIGHTS", tmp_path / "no.pt")
    m._state.clear()
    r = client.get("/plan", params={"ruta_id": "pisco"})
    assert r.status_code == 503 and "Faltan archivos del modelo" in r.json()["detail"]
