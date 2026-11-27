"""Response compression on the Plumber app (audit A3-05): big JSON is gzipped on request, small bodies are not."""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fluxpype.plumber.server import create_app  # noqa: E402

TOKEN = "t0k3n-for-tests"
H = {"X-Flux-Token": TOKEN}


@pytest.fixture
def client(make_scheduler):
    sched = make_scheduler()
    app = create_app(sched, TOKEN, demo=True, tick_interval=0.1)
    with TestClient(app, base_url="http://localhost") as c:
        c.sched = sched
        yield c


def test_large_json_is_gzipped_only_when_asked(client):
    raw = client.get("/api/profiles", headers={**H, "Accept-Encoding": "identity"})
    assert raw.status_code == 200
    assert "content-encoding" not in raw.headers
    assert len(raw.content) > 2048  # above the middleware minimum_size, so only the header decides
    zipped = client.get("/api/profiles", headers={**H, "Accept-Encoding": "gzip"})
    assert zipped.headers["content-encoding"] == "gzip"
    assert zipped.json() == raw.json()  # httpx decodes it: same payload
    assert int(zipped.headers["content-length"]) < len(raw.content)


def test_small_bodies_are_left_alone(client):
    r = client.get("/api/status", headers={**H, "Accept-Encoding": "gzip"})
    assert r.status_code == 200 and "content-encoding" not in r.headers


def test_gzip_q0_gets_identity(client):
    r = client.get("/api/profiles", headers={**H, "Accept-Encoding": "gzip;q=0, identity"})
    assert r.status_code == 200 and "content-encoding" not in r.headers
    r = client.get("/api/profiles", headers={**H, "Accept-Encoding": "gzip;q=0.5"})
    assert r.headers["content-encoding"] == "gzip"


def _data_file(client, name, payload):
    data = client.sched.data_dir()
    os.makedirs(data, exist_ok=True)
    path = os.path.join(data, name)
    with open(path, "wb") as f:
        f.write(payload)
    return path


def test_images_video_and_fits_are_never_gzipped(client):
    body = b"\x00\x01" * 5000  # far above minimum_size
    for name in ("movie.mp4", "map.fits", "blob.bin"):  # all served as application/octet-stream by raw_reply
        path = _data_file(client, name, body)
        r = client.get(
            "/api/cache/file", params={"path": os.path.basename(path)}, headers={**H, "Accept-Encoding": "gzip"}
        )
        assert r.status_code == 200, name
        assert "gzip" not in r.headers.get("content-encoding", ""), name
        assert r.content == body, name
    png = _data_file(client, "big.png", b"\x89PNG\r\n\x1a\n" + body)
    r = client.get("/api/artifacts", params={"path": png}, headers={**H, "Accept-Encoding": "gzip"})
    assert r.status_code == 200 and "gzip" not in r.headers.get("content-encoding", "")


def test_range_requests_are_served_as_bytes(client):
    body = bytes(range(256)) * 40
    path = _data_file(client, "movie.mp4", body)
    r = client.get(
        "/api/cache/file",
        params={"path": os.path.basename(path)},
        headers={**H, "Accept-Encoding": "gzip", "Range": "bytes=10-99"},
    )
    assert r.status_code == 206
    assert "gzip" not in r.headers.get("content-encoding", "")
    assert r.content == body[10:100]


def test_partial_json_response_is_not_gzipped():
    """A 206 is skipped even without a Range header on the request (response-level rule)."""
    from fastapi import FastAPI
    from fastapi.responses import Response

    from fluxpype.plumber.server import SelectiveGZip

    app = FastAPI()
    app.add_middleware(SelectiveGZip)

    @app.get("/p")
    def partial():
        return Response(b"x" * 5000, status_code=206, media_type="text/plain")

    @app.get("/t")
    def text():
        return Response(b"x" * 5000, media_type="text/plain")

    c = TestClient(app)
    assert "gzip" not in c.get("/p", headers={"Accept-Encoding": "gzip"}).headers.get("content-encoding", "")
    assert c.get("/t", headers={"Accept-Encoding": "gzip"}).headers["content-encoding"] == "gzip"
