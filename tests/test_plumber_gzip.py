"""Response compression on the Plumber app (audit A3-05): big JSON is gzipped on request, small bodies are not."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fluxpype.plumber.server import create_app  # noqa: E402

TOKEN = "t0k3n-for-tests"
H = {"X-Flux-Token": TOKEN}


@pytest.fixture
def client(make_scheduler):
    app = create_app(make_scheduler(), TOKEN, demo=True, tick_interval=0.1)
    with TestClient(app, base_url="http://localhost") as c:
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


def test_streams_and_images_are_excluded_from_compression():
    from starlette.middleware.gzip import GZipMiddleware

    excluded = getattr(GZipMiddleware(None, minimum_size=2048), "exclude_content_types", None)
    if excluded is None:
        pytest.skip("this Starlette has no exclusion list")
    assert "text/event-stream" in excluded and "image/png" in excluded
