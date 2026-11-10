"""FastAPI app for FLUX Studio: REST, one WebSocket, token auth, and the prebuilt frontend."""

import asyncio
import hmac
import json
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from . import config_io, paths
from .backends.fake import SCENARIOS
from .cache import Cache, CacheError
from .monitor import STAGES, first_match_for_context
from .scheduler import Scheduler, SchedulerError
from .store import FINISHED

COOKIE = "flux_studio"
IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
WEB_DIR = Path(__file__).resolve().parent / "web" / "dist"


def clean(obj: Any) -> Any:
    """JSON-safe copy: NaN and infinities become null (browsers reject them)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def reply(obj: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(clean(obj), status_code=status)


class JobIn(BaseModel):
    profile: str
    overrides: Dict[str, str] = Field(default_factory=dict)
    backend: Optional[str] = None
    tasks: Optional[List[List[str]]] = None  # [cr, nflux, adapt, method]
    scenario: Optional[str] = None  # fake backend only
    speed: Optional[float] = None  # fake backend only


class QueueIn(BaseModel):
    paused: Optional[bool] = None
    max_parallel: Optional[int] = None
    cpu_budget: Optional[int] = None


class TrashIn(BaseModel):
    paths: List[str]


class RestoreIn(BaseModel):
    ids: List[str]


class Hub:
    """Fan-out of server pushes to WebSocket clients (thread-safe publish)."""

    def __init__(self):
        self.clients: set = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def publish(self, kind: str, data: Any) -> None:
        if not self.clients or self.loop is None:
            return
        message = json.dumps(clean({"type": kind, "data": data}))
        try:
            self.loop.call_soon_threadsafe(self._push, message)
        except RuntimeError:
            pass

    def _push(self, message: str) -> None:
        for q in list(self.clients):
            if q.qsize() < 500:
                q.put_nowait(message)


def host_of(value: str) -> str:
    """Hostname part of a Host header or Origin value (no port, brackets kept off IPv6)."""
    if "://" in value:
        value = urlsplit(value).netloc
    if value.startswith("["):
        return value[1 : value.index("]")] if "]" in value else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def create_app(
    scheduler: Scheduler,
    token: str,
    allowed_hosts=("localhost", "127.0.0.1", "::1"),
    demo: bool = False,
    tick_interval: float = 0.5,
    version: str = "dev",
    run_loop: bool = True,
) -> FastAPI:
    hub = Hub()

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        hub.loop = asyncio.get_running_loop()
        task = asyncio.create_task(loop()) if run_loop else None
        try:
            yield
        finally:
            if task:
                task.cancel()

    app = FastAPI(title="FLUX Studio", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    allowed = {h.lower() for h in allowed_hosts}
    sched = scheduler
    caches: Dict[str, Cache] = {}
    log_offsets: Dict[str, int] = {}
    last_state: Dict[str, str] = {}
    app.state.hub, app.state.scheduler = hub, sched
    sched.on_change = hub.publish

    def authed(cookies, headers) -> bool:
        given = cookies.get(COOKIE) or headers.get("x-flux-token") or ""
        return hmac.compare_digest(given.encode(), token.encode())

    def origin_ok(headers) -> bool:
        origin = headers.get("origin")
        if not origin:
            return True
        return host_of(origin).lower() in allowed or host_of(origin) == host_of(headers.get("host", ""))

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = host_of(request.headers.get("host", "")).lower()
        if host not in allowed:
            return Response("Host not allowed", status_code=403)
        path = request.url.path
        supplied = request.query_params.get("token")
        if supplied is not None and not path.startswith("/api/"):
            if not hmac.compare_digest(supplied.encode(), token.encode()):
                return Response("Bad token", status_code=403)
            resp = RedirectResponse(path or "/", status_code=303)
            resp.set_cookie(COOKIE, token, httponly=True, samesite="strict", path="/", max_age=60 * 60 * 24 * 30)
            return resp
        if path.startswith("/api/"):
            if not authed(request.cookies, request.headers):
                return JSONResponse({"detail": "missing or wrong token; open the URL printed by flux-studio"}, status_code=401)
            if request.method not in ("GET", "HEAD", "OPTIONS") and not origin_ok(request.headers):
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)

    # ------------------------------------------------------------------ helpers
    def cache_for(profile: Optional[str] = None) -> Cache:
        data_dir = sched.data_dir(profile)
        if data_dir not in caches:
            caches[data_dir] = Cache(data_dir)
        return caches[data_dir]

    def fail(e: Exception, code: int = 400):
        raise HTTPException(status_code=code, detail=str(e))

    def spark(task_id: str, n: int = 60) -> List[float]:
        t = sched.store.get_task(task_id)
        if not t:
            return []
        pts = sched.monitor_for(t).relax_points()
        return [p["stiffness"] for p in pts[-n:] if p.get("stiffness") is not None]

    # ------------------------------------------------------------------ REST
    @app.get("/api/status")
    def status():
        return reply(
            {
                "version": version,
                "demo": demo,
                "studio_home": str(sched.home),
                "config_path": sched.config_path,
                "data_dir": sched.data_dir(),
                "backends": [
                    {"name": n, **b.capabilities().__dict__, "active": True} for n, b in sched.backends.items()
                ]
                + [
                    {"name": "ssh", "kind": "ssh", "slots": 0, "shared_fs": False, "description": "Remote host over SSH", "active": False},
                    {"name": "slurm", "kind": "slurm", "slots": 0, "shared_fs": True, "description": "Slurm cluster", "active": False},
                ],
                "default_backend": sched.default_backend,
                "scenarios": list(SCENARIOS) if "fake" in sched.backends else [],
                "stages": [{"id": s, "label": l} for s, l in STAGES],
                "forbidden_keys": sorted(config_io.FORBIDDEN_KEYS),
            }
        )

    @app.get("/api/profiles")
    def profiles():
        cf = sched.parsed()
        rows = []
        for name in cf.profiles:
            eff = cf.effective(name)
            plan = config_io.expand_tasks(eff)
            own = cf.sections[name]
            rows.append(
                {
                    "name": name,
                    "batch_name": config_io.resolve(eff, "batch_name"),
                    "task_count": len(plan.tasks),
                    "axes": plan.axes,
                    "errors": plan.errors,
                    "warnings": plan.warnings,
                    "own_keys": len(own),
                    "is_default_profile": cf.defaults.get("config_name") is not None and cf.defaults["config_name"].value == name,
                }
            )
        return reply({"profiles": rows})

    @app.get("/api/profiles/{name}")
    def profile(name: str):
        cf = sched.parsed()
        if name not in cf.sections:
            raise HTTPException(404, "unknown profile")
        eff = cf.effective(name)
        plan = config_io.expand_tasks(eff)
        base = eff["base_dir"].value if "base_dir" in eff else "auto"
        shown_base = config_io.auto_base_dir(str(paths.checkout_root())) if base.strip() in ("", "auto") else base
        entries = []
        for key, e in eff.items():
            entries.append(
                {
                    "key": key,
                    "value": e.value,
                    "resolved": config_io.resolve(eff, key, {"base_dir": shown_base}) if "${" in e.value else None,
                    "source": "DEFAULT" if e.section == "DEFAULT" else "profile",
                    "overrides_default": e.overrides_default,
                    "comment": e.comment,
                    "editable": key not in config_io.FORBIDDEN_KEYS,
                }
            )
        data_dir = config_io.resolved_path(eff, "data_dir", str(paths.checkout_root()))
        tasks = []
        for adapt, cr, nflux, method in plan.tasks[:200]:
            ctx = sched.probe_context_for(name, cr, nflux, adapt)
            tasks.append(
                {
                    "cr": cr,
                    "nflux": nflux,
                    "adapt": adapt,
                    "method": method,
                    "inst": config_io.task_instrument(adapt),
                    "cache": first_match_for_context(ctx) if ctx else {},
                }
            )
        return reply(
            {
                "name": name,
                "entries": entries,
                "tasks": tasks,
                "task_count": len(plan.tasks),
                "truncated": len(plan.tasks) > 200,
                "errors": plan.errors,
                "warnings": plan.warnings,
                "axes": plan.axes,
                "data_dir": data_dir,
                "slots": sched._slots(eff),
            }
        )

    @app.post("/api/jobs")
    def create_job(body: JobIn):
        extra: Dict[str, str] = {}
        backend = body.backend or sched.default_backend
        if body.scenario is not None or body.speed is not None:
            if backend != "fake":
                raise HTTPException(400, "scenario and speed only apply to the fake backend")
            if body.scenario is not None:
                if body.scenario not in SCENARIOS:
                    raise HTTPException(400, f"unknown scenario {body.scenario!r}")
                extra["scenario"] = body.scenario
            if body.speed is not None:
                if not 0.1 <= body.speed <= 1000:
                    raise HTTPException(400, "speed must be between 0.1 and 1000")
                extra["speed"] = str(body.speed)
        try:
            return reply(sched.create_job(body.profile, body.overrides, backend, body.tasks, extra), 201)
        except SchedulerError as e:
            fail(e)

    @app.get("/api/jobs")
    def jobs(limit: int = 100):
        out = []
        for j in sched.store.list_jobs(min(max(limit, 1), 500)):
            out.append(sched.job_view(j["id"]))
        return reply({"jobs": out})

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, states: bool = False):
        view = sched.job_view(job_id)
        if not view:
            raise HTTPException(404, "unknown job")
        if states:
            view["states"] = {t["id"]: sched.task_state(t["id"]) for t in view["tasks"]}
        return reply(view)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if not sched.store.get_job(job_id):
            raise HTTPException(404, "unknown job")
        sched.cancel_job(job_id)
        return reply(sched.job_view(job_id))

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        try:
            return reply(sched.retry_job(job_id))
        except SchedulerError as e:
            fail(e, 409)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        try:
            sched.delete_job(job_id)
        except SchedulerError as e:
            fail(e, 409)
        return reply({"deleted": job_id})

    @app.get("/api/tasks/{task_id}")
    def task(task_id: str):
        try:
            return reply(sched.task_state(task_id))
        except SchedulerError as e:
            fail(e, 404)

    @app.post("/api/tasks/{task_id}/cancel")
    def cancel_task(task_id: str):
        try:
            sched.cancel_task(task_id)
            return reply(sched.task_state(task_id))
        except SchedulerError as e:
            fail(e, 404)

    @app.get("/api/tasks/{task_id}/relax")
    def task_relax(task_id: str, after: int = 0):
        try:
            return reply(sched.relax_series(task_id, max(after, 0)))
        except SchedulerError as e:
            fail(e, 404)

    def log_file(t) -> Optional[str]:
        for name in ("run.log", "stdout.log"):
            p = os.path.join(t["run_dir"], name)
            if os.path.exists(p):
                return p
        return None

    @app.get("/api/tasks/{task_id}/log")
    def task_log(task_id: str, offset: int = -1, limit: int = 262144):
        """Log text from byte `offset` (default: the last `limit` bytes). Returns the next offset."""
        t = sched.store.get_task(task_id)
        if not t:
            raise HTTPException(404, "unknown task")
        path = log_file(t)
        if path is None:
            return reply({"offset": 0, "size": 0, "text": "", "file": None})
        size = os.path.getsize(path)
        limit = min(max(limit, 1024), 2 * 1024 * 1024)
        start = max(size - limit, 0) if offset < 0 else min(offset, size)
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(limit)
        # keep whole lines: drop a partial first line when tailing, never split a UTF-8 sequence
        if offset < 0 and start > 0 and b"\n" in data:
            cut = data.index(b"\n") + 1
            start, data = start + cut, data[cut:]
        end = start + len(data)
        return reply({"offset": end, "start": start, "size": size, "text": data.decode("utf-8", "replace"), "file": os.path.basename(path)})

    @app.get("/api/queue")
    def queue():
        pending = []
        for t in sched.store.tasks_with_status("pending"):
            job = sched.store.get_job(t["job_id"])
            pending.append({**sched.task_row_view(t), "profile": job["profile"]})
        return reply({**sched.queue_state(), "pending_tasks": pending})

    @app.patch("/api/queue")
    def patch_queue(body: QueueIn):
        try:
            return reply(sched.set_queue(body.paused, body.max_parallel, body.cpu_budget))
        except SchedulerError as e:
            fail(e)

    @app.get("/api/overview")
    def overview():
        running = []
        for t in sched.store.tasks_with_status("running"):
            state = sched.task_state(t["id"])
            state["spark"] = spark(t["id"])
            running.append(state)
        pending = []
        for t in sched.store.tasks_with_status("pending"):
            job = sched.store.get_job(t["job_id"])
            pending.append({**sched.task_row_view(t), "profile": job["profile"]})
        recent = []
        for t in sched.store.recent_finished(12):
            job = sched.store.get_job(t["job_id"])
            state = sched.task_state(t["id"])
            arts = [a for a in state.get("artifacts", []) if a.get("path")]
            ok_stages = sum(1 for s in state.get("stages", []) if s["status"] == "ok")
            recent.append(
                {
                    **sched.task_row_view(t),
                    "profile": job["profile"],
                    "thumb": arts[-1]["path"] if arts else None,
                    "stages_ok": ok_stages,
                    "stages": [{"id": s["id"], "status": s["status"], "source": s["source"]} for s in state.get("stages", [])],
                    "failure": next((s["detail"] for s in state.get("stages", []) if s["status"] == "failed" and s.get("detail")), None),
                }
            )
        return reply({"queue": sched.queue_state(), "running": running, "pending": pending, "recent": recent})

    # ------------------------------------------------------------------ cache
    @app.get("/api/cache")
    def cache_get(rescan: bool = False, profile: Optional[str] = None):
        c = cache_for(profile)
        summary = c.summary(rescan)
        records = c.records()[:20000]
        return reply({**summary, "records": records, "truncated": summary["total_count"] > len(records)})

    @app.post("/api/cache/trash")
    def cache_trash(body: TrashIn):
        try:
            result = cache_for().move_to_trash(body.paths)
        except CacheError as e:
            fail(e)
        hub.publish("cache_changed", {})
        return reply(result)

    @app.get("/api/cache/trash")
    def cache_trash_list():
        c = cache_for()
        return reply({"items": c.list_trash(), **c.trash_summary()})

    @app.post("/api/cache/trash/restore")
    def cache_restore(body: RestoreIn):
        try:
            restored = cache_for().restore(body.ids)
        except CacheError as e:
            fail(e)
        hub.publish("cache_changed", {})
        return reply({"restored": restored})

    @app.delete("/api/cache/trash")
    def cache_empty():
        result = cache_for().empty_trash()
        hub.publish("cache_changed", {})
        return reply(result)

    # ------------------------------------------------------------------ artifacts
    @app.get("/api/artifacts")
    def artifact(path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in IMAGE_TYPES:
            raise HTTPException(415, "only images are served")
        real = os.path.realpath(os.path.expanduser(path))
        roots = [os.path.realpath(sched.data_dir()), os.path.realpath(str(paths.runs_dir(sched.home)))]
        for prof in sched.parsed().profiles:
            roots.append(os.path.realpath(sched.data_dir(prof)))
        if not any(real == r or real.startswith(r + os.sep) for r in set(roots)):
            raise HTTPException(403, "path is outside the data and run directories")
        if f"{os.sep}{'.flux_trash'}{os.sep}" in real:
            raise HTTPException(403, "that file is in the trash")
        if not os.path.isfile(real):
            raise HTTPException(404, "no such file")
        return FileResponse(real, media_type=IMAGE_TYPES[ext], headers={"Cache-Control": "private, max-age=30"})

    # ------------------------------------------------------------------ websocket
    @app.websocket("/api/ws")
    async def ws(sock: WebSocket):
        host = host_of(sock.headers.get("host", "")).lower()
        if host not in allowed or not authed(sock.cookies, sock.headers) or not origin_ok(sock.headers):
            await sock.close(code=4401)
            return
        await sock.accept()
        q: asyncio.Queue = asyncio.Queue()
        hub.clients.add(q)
        sender = None
        try:
            await sock.send_text(json.dumps({"type": "hello", "data": {"version": version, "demo": demo}}))
            sender = asyncio.create_task(_pump(sock, q))
            while True:
                await sock.receive_text()  # ignore client messages; detects disconnect
        except WebSocketDisconnect:
            pass
        finally:
            hub.clients.discard(q)
            if sender:
                sender.cancel()

    async def _pump(sock: WebSocket, q: asyncio.Queue):
        while True:
            await sock.send_text(await q.get())

    # ------------------------------------------------------------------ background loop
    def push_states() -> None:
        for t in sched.store.tasks_with_status("running"):
            state = sched.task_state(t["id"])
            # observed_at changes every poll; compare without it
            key = json.dumps({k: v for k, v in clean(state).items() if k != "observed_at"}, sort_keys=True, default=str)
            if last_state.get(t["id"]) != key:
                last_state[t["id"]] = key
                hub.publish("task_state", state)
            path = log_file(t)
            if path:
                size = os.path.getsize(path)
                pos = log_offsets.get(t["id"], max(size - 8192, 0))
                if size > pos:
                    with open(path, "rb") as f:
                        f.seek(pos)
                        data = f.read(65536)
                    cut = data.rfind(b"\n") + 1
                    if cut:
                        log_offsets[t["id"]] = pos + cut
                        hub.publish("log_append", {"task_id": t["id"], "start": pos, "offset": pos + cut, "text": data[:cut].decode("utf-8", "replace")})
                else:
                    log_offsets.setdefault(t["id"], pos)

    async def loop():
        hub.loop = asyncio.get_running_loop()
        while True:
            try:
                await asyncio.to_thread(sched.tick)
                if hub.clients:
                    await asyncio.to_thread(push_states)
            except Exception as e:  # noqa: BLE001 - the loop must survive anything
                print(f"[studio] tick failed: {e!r}")
            await asyncio.sleep(tick_interval)

    # ------------------------------------------------------------------ frontend
    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "not found")
        if not WEB_DIR.is_dir():
            return HTMLResponse(
                "<h1>FLUX Studio</h1><p>The frontend build is missing (fluxpype/studio/web/dist). "
                "Run <code>npm install &amp;&amp; npm run build</code> in fluxpype/studio/web.</p>",
                status_code=503,
            )
        target = (WEB_DIR / full_path).resolve()
        if full_path and WEB_DIR.resolve() in target.parents and target.is_file():
            headers = {"Cache-Control": "public, max-age=31536000, immutable"} if "/assets/" in f"/{full_path}" else {}
            return FileResponse(target, headers=headers)
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app
