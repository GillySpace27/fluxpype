"""FastAPI app via TestClient: auth, token cookie, job lifecycle on the fake backend, path safety."""

import os
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

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


def wait_job(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}", headers=H).json()
        if job["status"] in ("ok", "failed", "cancelled", "partial"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


# ------------------------------------------------------------------ auth


def test_api_requires_the_token(client):
    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"X-Flux-Token": "nope"}).status_code == 401
    assert client.get("/api/status", headers=H).status_code == 200


def test_token_in_the_url_becomes_a_strict_httponly_cookie(client):
    r = client.get("/?token=" + TOKEN, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"  # the token is stripped from the URL
    cookie = r.headers["set-cookie"]
    assert "flux_plumber=" in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
    # the cookie alone now authenticates API calls
    assert client.get("/api/status").status_code == 200
    assert client.get("/?token=wrong", follow_redirects=False).status_code == 403


def test_unknown_hosts_are_refused(client):
    r = client.get("/api/status", headers={**H, "Host": "evil.example.com"})
    assert r.status_code == 403
    r = client.get("/api/status", headers={**H, "Host": "localhost:8765"})
    assert r.status_code == 200


def test_cross_origin_writes_are_refused(client):
    r = client.patch("/api/queue", json={"paused": True}, headers={**H, "Origin": "http://evil.example.com"})
    assert r.status_code == 403
    r = client.patch("/api/queue", json={"paused": False}, headers={**H, "Origin": "http://localhost:8765"})
    assert r.status_code == 200


def test_websocket_needs_the_cookie_and_a_sane_origin(client):
    host = {"Host": "localhost:8765"}  # the WebSocket test client ignores base_url
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws", headers=host):
            pass
    client.cookies.set("flux_plumber", TOKEN)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws", headers={**host, "Origin": "http://evil.example.com"}):
            pass
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/ws", headers={"Host": "evil.example.com"}):
            pass
    with client.websocket_connect("/api/ws", headers={**host, "Origin": "http://localhost:8765"}) as ws:
        assert ws.receive_json()["type"] == "hello"


# ------------------------------------------------------------------ read endpoints


def test_status_and_profiles(client):
    st = client.get("/api/status", headers=H).json()
    assert st["demo"] and [b["name"] for b in st["backends"]][:1] == ["fake"]
    assert st["active_backend"] == "fake" and st["sandboxed"] is True
    assert len(st["stages"]) == 10 and "run_script" in st["forbidden_keys"]
    profs = {p["name"]: p for p in client.get("/api/profiles", headers=H).json()["profiles"]}
    assert profs["t_two"]["task_count"] == 4 and profs["Real_Run"]["errors"]


def test_profile_detail_shows_sources_and_cache_hits(client):
    p = client.get("/api/profiles/t_one", headers=H).json()
    by_key = {e["key"]: e for e in p["entries"]}
    assert by_key["batch_name"]["source"] == "profile" and by_key["relax_threshold"]["source"] == "DEFAULT"
    assert by_key["relax_threshold"]["comment"] and by_key["run_script"]["editable"] is False
    assert by_key["fl_mhdlib"]["resolved"] and "${" not in by_key["fl_mhdlib"]["resolved"]
    assert "auto" not in by_key["fl_mhdlib"]["resolved"]  # base_dir = auto is shown as the real folder
    assert by_key["data_dir"]["resolved"] is None  # no placeholder to resolve in the test config
    assert p["task_count"] == 1 and p["tasks"][0]["cache"] == {
        "magnetogram": False,
        "footpoints": False,
        "pfss": False,
        "world": False,
        "relaxed_world": False,
    }
    assert client.get("/api/profiles/missing", headers=H).status_code == 404


def test_unknown_things_are_404(client):
    for url in ("/api/jobs/nope", "/api/tasks/nope", "/api/tasks/nope/log", "/api/tasks/nope/relax"):
        assert client.get(url, headers=H).status_code == 404, url
    assert client.get("/api/nonsense", headers=H).status_code == 404


# ------------------------------------------------------------------ jobs


def test_job_lifecycle(client):
    r = client.post(
        "/api/jobs", json={"profile": "t_one", "backend": "fake", "scenario": "ok", "speed": 300}, headers=H
    )
    assert r.status_code == 201, r.text
    job = r.json()
    done = wait_job(client, job["id"])
    assert done["status"] == "ok"
    tid = done["tasks"][0]["id"]
    full = client.get(f"/api/jobs/{job['id']}?states=1", headers=H).json()
    assert full["states"][tid]["status"] == "ok"
    t = client.get(f"/api/tasks/{tid}", headers=H).json()
    assert t["mode"] == "events" and t["stages"][4]["id"] == "relax"
    # log: tail, then from an offset
    log = client.get(f"/api/tasks/{tid}/log", headers=H).json()
    assert "Program Complete for CR2225" in log["text"] and log["file"] == "run.log"
    again = client.get(f"/api/tasks/{tid}/log?offset={log['offset']}", headers=H).json()
    assert again["text"] == "" and again["offset"] == log["offset"]
    head = client.get(f"/api/tasks/{tid}/log?offset=0&limit=2048", headers=H).json()
    assert head["start"] == 0 and head["text"].startswith("Log file:")
    # relax series
    series = client.get(f"/api/tasks/{tid}/relax", headers=H).json()
    assert series["n"] > 10 and series["cycles"] and series["points"][0]["stiffness"] > 50
    later = client.get(f"/api/tasks/{tid}/relax?after={series['n'] - 2}", headers=H).json()
    assert len(later["points"]) == 2
    # overview and jobs list
    ov = client.get("/api/overview", headers=H).json()
    assert ov["recent"][0]["id"] == tid and ov["recent"][0]["thumb"].endswith(".png")
    assert client.get("/api/jobs", headers=H).json()["jobs"][0]["id"] == job["id"]
    # the thumbnail is servable
    img = client.get("/api/artifacts", params={"path": ov["recent"][0]["thumb"]}, headers=H)
    assert img.status_code == 200 and img.headers["content-type"] == "image/png" and img.content[:4] == b"\x89PNG"
    assert client.delete(f"/api/jobs/{job['id']}", headers=H).status_code == 200


def test_bad_job_requests(client):
    post = lambda body: client.post("/api/jobs", json=body, headers=H)  # noqa: E731
    assert post({"profile": "nope"}).status_code == 400
    assert post({"profile": "t_one", "overrides": {"run_script": "x"}}).status_code == 400
    assert post({"profile": "t_one", "overrides": {"max_cycles": "__import__('os')"}}).status_code == 400
    assert post({"profile": "t_one", "scenario": "bogus"}).status_code == 400
    assert post({"profile": "t_one", "speed": 0}).status_code == 400
    assert post({"profile": "Real_Run"}).status_code == 400
    assert client.sched.store.list_jobs() == []


def test_cancel_and_retry_via_api(client):
    client.patch("/api/queue", json={"paused": True}, headers=H)
    job = client.post("/api/jobs", json={"profile": "t_one", "backend": "fake", "speed": 300}, headers=H).json()
    q = client.get("/api/queue", headers=H).json()
    assert q["paused"] and q["pending_tasks"][0]["profile"] == "t_one"
    assert client.delete(f"/api/jobs/{job['id']}", headers=H).status_code == 409  # still queued
    assert client.post(f"/api/jobs/{job['id']}/cancel", headers=H).json()["status"] == "cancelled"
    assert client.post(f"/api/jobs/{job['id']}/retry", headers=H).status_code == 200
    assert client.patch("/api/queue", json={"max_parallel": 0}, headers=H).status_code == 400
    client.patch("/api/queue", json={"paused": False}, headers=H)
    assert wait_job(client, job["id"])["status"] == "ok"
    assert client.post(f"/api/jobs/{job['id']}/retry", headers=H).status_code == 409  # nothing failed


def test_websocket_pushes_job_state(client):
    client.cookies.set("flux_plumber", TOKEN)
    with client.websocket_connect("/api/ws", headers={"Host": "localhost:8765"}) as ws:
        assert ws.receive_json()["type"] == "hello"
        client.post("/api/jobs", json={"profile": "t_one", "backend": "fake", "speed": 300}, headers=H)
        kinds = set()
        end = time.time() + 30
        while time.time() < end and not {"job_state", "task_state"} <= kinds:
            kinds.add(ws.receive_json()["type"])
        assert {"job_state", "task_state"} <= kinds


def test_agent_requests_are_recorded_and_pushed(client):
    assert client.get("/api/status", headers=H).json()["agent"] is None
    client.cookies.set("flux_plumber", TOKEN)
    with client.websocket_connect("/api/ws", headers={"Host": "localhost:8765"}) as ws:
        assert ws.receive_json()["type"] == "hello"
        client.get("/api/queue", headers={**H, "X-Flux-Client": "mcp", "X-Flux-Tool": "get_job"})
        msg = ws.receive_json()
        assert msg["type"] == "agent_activity" and msg["data"] == {"tool": "get_job", "count": 1}
    agent = client.get("/api/status", headers=H).json()["agent"]
    assert agent["tool"] == "get_job" and agent["count"] == 1 and 0 <= agent["age_s"] < 30


def test_agent_heartbeat_marks_alive_without_a_tool_call(client):
    link = client.get("/api/status", headers=H).json()["agent_link"]
    assert link == {"alive": None, "seen": None}
    client.get("/api/queue", headers={**H, "X-Flux-Client": "mcp-heartbeat"})
    st = client.get("/api/status", headers=H).json()
    assert st["agent"] is None  # a beat is not a tool call
    assert 0 <= st["agent_link"]["alive"] < 30 and 0 <= st["agent_link"]["seen"] < 30


# ------------------------------------------------------------------ artifacts


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "/etc/passwd.png", "../../../etc/passwd.png", "~/.ssh/id_rsa.png", "/proc/self/environ.png"],
)
def test_artifact_paths_are_confined(client, path):
    r = client.get("/api/artifacts", params={"path": path}, headers=H)
    assert r.status_code in (403, 404, 415), (path, r.status_code)
    assert r.content[:4] != b"\x89PNG"


def test_artifact_serves_only_images_inside_the_data_dir(client, tmp_path):
    data = client.sched.data_dir()
    os.makedirs(data, exist_ok=True)
    png = os.path.join(data, "ok.png")
    with open(png, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\nfake")
    txt = os.path.join(data, "secret.txt")
    open(txt, "w").write("x")
    svg = os.path.join(data, "x.svg")
    open(svg, "w").write("<svg/>")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"\x89PNG")
    os.symlink(outside, os.path.join(data, "link.png"))
    assert client.get("/api/artifacts", params={"path": png}, headers=H).status_code == 200
    assert client.get("/api/artifacts", params={"path": txt}, headers=H).status_code == 415
    assert client.get("/api/artifacts", params={"path": svg}, headers=H).status_code == 415  # scripts could ride in SVG
    assert client.get("/api/artifacts", params={"path": str(outside)}, headers=H).status_code == 403
    assert client.get("/api/artifacts", params={"path": os.path.join(data, "link.png")}, headers=H).status_code == 403
    assert (
        client.get("/api/artifacts", params={"path": os.path.join(data, "..", "outside.png")}, headers=H).status_code
        == 403
    )


# ------------------------------------------------------------------ cache


def test_cache_endpoints(client):
    data = client.sched.data_dir()
    for rel in ("magnetograms/CR2225_r4_hmi.fits", "pfss/pfss_cr2225_r4_hmi.pkl"):
        p = os.path.join(data, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(b"x" * 100)
    c = client.get("/api/cache?rescan=true", headers=H).json()
    assert c["total_count"] == 2 and {t["kind"]: t["count"] for t in c["tiles"]}["magnetogram"] == 1
    r = client.post("/api/cache/trash", json={"paths": ["pfss/pfss_cr2225_r4_hmi.pkl"]}, headers=H)
    assert r.status_code == 200 and r.json()["size"] == 100
    trash = client.get("/api/cache/trash", headers=H).json()
    assert trash["count"] == 1
    assert (
        client.post("/api/cache/trash/restore", json={"ids": [trash["items"][0]["id"]]}, headers=H).status_code == 200
    )
    assert os.path.exists(os.path.join(data, "pfss/pfss_cr2225_r4_hmi.pkl"))
    client.post("/api/cache/trash", json={"paths": ["pfss/pfss_cr2225_r4_hmi.pkl"]}, headers=H)
    assert client.delete("/api/cache/trash", headers=H).json()["removed"] == 1


def test_cache_refuses_traversal(client, tmp_path):
    (tmp_path / "keep.txt").write_text("keep")
    for bad in ("../keep.txt", "/etc/hostname", "a/../../keep.txt"):
        r = client.post("/api/cache/trash", json={"paths": [bad]}, headers=H)
        assert r.status_code == 400, bad
    assert (tmp_path / "keep.txt").read_text() == "keep"


def test_flux_endpoints_parse_and_stay_confined(client, tmp_path):
    FIXTURE = (
        "NEW\t100\t0.6\t0\t0.8\t-1\t(null)\t0\nNEW\t200\t0\t0.6\t0.8\t1\t(null)\t0\n"
        "LINE\t-10\t-3\t-4\t-10\t-11\t0  0 0 0  0 0 0\n"
        "LINE\t101\t-302\t-301\t200\t100\t1  0 0.6 0.8  0.6 0 0.8\n\tVERTEX\t101\t102\t1\t0.4\t0.4\t1.5\n"
    )
    data = client.sched.data_dir()
    os.makedirs(os.path.join(data, "world"), exist_ok=True)
    with open(os.path.join(data, "world", "w.flux"), "w") as f:
        f.write(FIXTURE)
    (tmp_path / "out.flux").write_text(FIXTURE)
    job = client.post("/api/jobs", json={"profile": "t_one", "backend": "fake", "speed": 300}, headers=H).json()
    for url in ("/api/cache/flux", f"/api/jobs/{job['id']}/data/flux"):
        r = client.get(url, params={"path": "world/w.flux"}, headers=H)
        assert r.status_code == 200, url
        w = r.json()
        assert w["counts"]["fluxons"] == 2 and w["fluxons"]["id"] == [101] and w["decimated"] is None
        assert client.get(url, params={"path": "world/none.flux"}, headers=H).status_code == 404
        for bad in ("../out.flux", "/etc/hostname", str(tmp_path / "out.flux")):
            assert client.get(url, params={"path": bad}, headers=H).status_code in (400, 403, 404), (url, bad)


# ------------------------------------------------------------------ frontend


def test_spa_fallback_and_static_traversal(client):
    r = client.get("/jobs/abc", headers=H)
    assert r.status_code in (200, 503)  # index.html (or the build-missing page before the frontend exists)
    assert client.get("/..%2f..%2fetc%2fpasswd", headers=H).status_code in (200, 404, 503)
    r = client.get("/..%2f..%2fetc%2fpasswd", headers=H)
    assert "root:" not in r.text


# ------------------------------------------------------------------ log grep


def _log_task(client, lines):
    job = client.post("/api/jobs", json={"profile": "t_one", "backend": "fake", "speed": 300}, headers=H).json()
    done = wait_job(client, job["id"])
    task = client.sched.store.get_task(done["tasks"][0]["id"])
    with open(os.path.join(task["run_dir"], "run.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return task["id"]


def test_log_grep_returns_numbered_windows(client):
    lines = [f"line {i}" for i in range(1, 31)]
    lines[9] = "the Needle is here"
    lines[11] = "another NEEDLE"
    lines[24] = "needle again"
    tid = _log_task(client, lines)
    r = client.get(f"/api/tasks/{tid}/log", params={"grep": "needle", "context": 1}, headers=H).json()
    assert r["matches"] == 3 and not r["truncated"]
    assert r["text"].split("\n") == [
        "9-line 9",
        "10:the Needle is here",
        "11-line 11",
        "12:another NEEDLE",
        "13-line 13",
        "--",
        "24-line 24",
        "25:needle again",
        "26-line 26",
    ]
    # a regex is not interpreted, and no match is not an error
    none = client.get(f"/api/tasks/{tid}/log", params={"grep": "n.edle"}, headers=H).json()
    assert none["matches"] == 0 and none["text"] == ""
    # default context is 3, and it is capped at 20
    d = client.get(f"/api/tasks/{tid}/log", params={"grep": "needle again"}, headers=H).json()
    assert d["lines"] == 7
    big = client.get(f"/api/tasks/{tid}/log", params={"grep": "needle again", "context": 500}, headers=H).json()
    assert big["context"] == 20


def test_log_grep_is_capped_at_400_lines(client):
    tid = _log_task(client, [f"hit {i}" for i in range(1000)])
    r = client.get(f"/api/tasks/{tid}/log", params={"grep": "hit", "context": 0}, headers=H).json()
    assert r["truncated"] and r["lines"] == 400 and len(r["text"].split("\n")) == 400
    # without grep the log endpoint is unchanged
    plain = client.get(f"/api/tasks/{tid}/log", headers=H).json()
    assert "offset" in plain and "matches" not in plain


# ------------------------------------------------------------------ sign-in, address, installable app


def test_pages_without_a_token_get_the_sign_in_page(client):
    r = client.get("/jobs")
    assert r.status_code == 401 and "Sign in" in r.text and 'action="/login"' in r.text
    assert client.get("/login").status_code == 200


def test_login_form_sets_the_cookie(client):
    bad = client.post("/login", data={"token": "nope"}, follow_redirects=False)
    assert bad.status_code == 403 and "not this Plumber" in bad.text
    ok = client.post("/login", data={"token": TOKEN}, follow_redirects=False)
    assert ok.status_code == 303 and "flux_plumber=" in ok.headers["set-cookie"]
    assert client.get("/api/status").status_code == 200


def test_login_form_refuses_cross_origin_posts(client):
    r = client.post("/login", data={"token": TOKEN}, headers={"Origin": "http://evil.example"}, follow_redirects=False)
    assert r.status_code == 403


def test_signed_in_page_loads_refresh_the_cookie(client):
    client.cookies.set("flux_plumber", TOKEN)
    r = client.get("/jobs")
    assert r.status_code in (200, 503) and "Max-Age=2592000" in r.headers.get("set-cookie", "")


def test_plumber_localhost_names_are_allowed_and_others_are_not(client):
    for host in ("plumber.localhost:8765", "demo.plumber.localhost:8775"):
        assert client.get("/api/status", headers={**H, "Host": host}).status_code == 200
    for host in ("evil.example", "plumber.localhost.evil.example", "xplumber.localhost"):
        assert client.get("/api/status", headers={**H, "Host": host}).status_code == 403


def test_manifest_and_icons_are_public(client):
    m = client.get("/manifest.webmanifest")
    assert m.status_code == 200 and m.headers["content-type"].startswith("application/manifest+json")
    body = m.json()
    assert body["display"] == "standalone" and body["short_name"] == "Plumber (demo)"
    assert {i["sizes"] for i in body["icons"]} == {"192x192", "512x512"}


def test_custom_cookie_name_keeps_instances_apart(make_scheduler):
    app = create_app(make_scheduler(), TOKEN, demo=False, run_loop=False, cookie_name="flux_plumber_abc")
    with TestClient(app, base_url="http://localhost") as c:
        r = c.get(f"/?token={TOKEN}", follow_redirects=False)
        assert "flux_plumber_abc=" in r.headers["set-cookie"]
        c.cookies.set("flux_plumber", TOKEN)  # another instance's cookie does not count here
        c.cookies.delete("flux_plumber_abc")
        assert c.get("/api/status").status_code == 401
