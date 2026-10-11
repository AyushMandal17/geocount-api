"""End-to-end API test with a fake model (no weights or torch needed)."""
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app import main


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Obb:
    xywhr = _T([[100.0, 100.0, 40.0, 20.0, 0.2]])
    conf = _T([0.9])
    cls = _T([1])

    def __len__(self):
        return 1


class _Res:
    obb = _Obb()


class FakeModel:
    names = {0: "plane", 1: "ship"}

    def predict(self, imgs, **kw):
        return [_Res() for _ in (imgs if isinstance(imgs, list) else [imgs])]


def _png(w=800, h=600):
    ok, buf = cv2.imencode(".png", np.zeros((h, w, 3), np.uint8))
    return buf.tobytes()


def client(monkeypatch):
    monkeypatch.setattr(main, "get_model", lambda: FakeModel())
    return TestClient(main.app)


def test_health_and_classes(monkeypatch):
    c = client(monkeypatch)
    assert c.get("/api/health").json()["status"] == "ok"
    assert "ship" in c.get("/api/classes").json()["classes"]


def test_count_with_question_and_gsd(monkeypatch):
    c = client(monkeypatch)
    r = c.post("/api/count", files={"image": ("a.png", _png(), "image/png")},
               data={"question": "How many ships?", "gsd": "0.5"})
    assert r.status_code == 200
    j = r.json()
    assert j["answer"] == "There is 1 ship."
    assert j["counts"] == [{"class": "ship", "count": 1}]
    assert j["measurements"][0]["mean_length_m"] == 20.0  # 40px * 0.5 m/px
    assert j["annotated_image"].startswith("data:image/jpeg;base64,")


def test_tiled_large_image_counts_each_tile(monkeypatch):
    c = client(monkeypatch)
    r = c.post("/api/count", files={"image": ("a.png", _png(2000, 2000), "image/png")})
    assert r.json()["total"] > 1  # one fake box per tile, offsets make them distinct


def test_bad_inputs(monkeypatch):
    c = client(monkeypatch)
    r = c.post("/api/count", files={"image": ("a.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
    r = c.post("/api/count", files={"image": ("a.png", _png(), "image/png")}, data={"conf": "0.01"})
    assert r.status_code == 422

def test_imgsz_option(monkeypatch):
    c = client(monkeypatch)
    r = c.post("/api/count", files={"image": ("a.png", _png(), "image/png")}, data={"imgsz": "480"})
    assert r.status_code == 200 and r.json()["settings"]["imgsz"] == 480
    r = c.post("/api/count", files={"image": ("a.png", _png(), "image/png")}, data={"imgsz": "100"})
    assert r.status_code == 422