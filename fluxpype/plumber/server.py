"""FastAPI app for Fluxpype Plumber: REST, one WebSocket, token auth, and the prebuilt frontend."""

import asyncio
import hmac
import json
import math
import os
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, quote, urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, Field

from . import config_io, paths
from .backends.fake import SCENARIOS
from .cache import Cache, CacheError
from .health import Health
from .presets import PresetError
from .monitor import STAGES, first_match_for_context
from .scheduler import Scheduler, SchedulerError

COOKIE = "flux_plumber"
COOKIE_MAX_AGE = 60 * 60 * 24 * 30
APP_DOMAIN = "plumber.localhost"  # *.localhost always resolves to loopback (RFC 6761)
PUBLIC_FILES = {"/manifest.webmanifest", "/sw.js", "/login", "/favicon.ico"}
PUBLIC_PREFIXES = ("/assets/", "/icons/")
# results product id -> pipeline stage id (monitor.STAGES)
PRODUCT_STAGE = {"relaxed_world": "relax"}

IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
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


class BackendIn(BaseModel):
    name: str


class DataDirIn(BaseModel):
    path: Optional[str] = None


class PresetIn(BaseModel):
    name: str
    base: Optional[str] = None
    values: Dict[str, str] = Field(default_factory=dict)


class PresetSaveIn(BaseModel):
    values: Dict[str, str] = Field(default_factory=dict)
    remove: List[str] = Field(default_factory=list)


class RenameIn(BaseModel):
    name: str


class HiddenIn(BaseModel):
    hidden: bool


class ImportIn(BaseModel):
    text: str


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


def login_html(error: str, token_file: str, demo: bool) -> str:
    """A small sign-in page for a browser that has no (or a stale) token cookie."""
    from html import escape

    err = f'<p class="err">{escape(error)}</p>' if error else ""
    title = "Plumber (demo)" if demo else "Plumber"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Sign in: {title}</title>
<link rel="manifest" href="/manifest.webmanifest"><link rel="icon" href="/icons/icon-192.png">
<meta name="theme-color" content="#0b0d12">
<style>
:root{{color-scheme:dark light;--bg:#0b0d12;--card:#141821;--ink:#e8ebf2;
--muted:#9aa3b5;--line:#262c3a;--accent:#ff9f43}}
@media (prefers-color-scheme: light){{:root{{--bg:#f6f7fb;--card:#fff;--ink:#141821;--muted:#5b6475;--line:#dde1ea}}}}
body{{margin:0;min-height:100vh;display:grid;place-items:center;background:var(--bg);color:var(--ink);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Inter",system-ui,sans-serif;padding:16px}}
.card{{width:100%;max-width:420px;background:var(--card);border:1px solid var(--line);border-radius:18px;padding:28px}}
h1{{font-size:18px;margin:0 0 4px}}p{{color:var(--muted);margin:0 0 16px}}
.err{{color:#ff6b6b}}
code{{white-space:nowrap;font:12px ui-monospace,Menlo,monospace;background:var(--bg);padding:2px 6px;border-radius:6px}}
input{{width:100%;box-sizing:border-box;padding:10px 12px;border-radius:10px;border:1px solid var(--line);
background:var(--bg);color:var(--ink);font:13px ui-monospace,Menlo,monospace;margin-bottom:12px}}
button{{width:100%;padding:10px;border:0;border-radius:10px;background:var(--accent);color:#1b1206;font-weight:600;cursor:pointer}}
</style></head><body><form class="card" method="post" action="/login">
<h1>Sign in to {title}</h1>
<p>This browser has no valid token for this Plumber. Paste the token from
<code>{escape(token_file)}</code>, or run <code>flux-plumb --open</code> to sign in automatically.</p>{err}
<input name="token" type="password" placeholder="Token" autocomplete="current-password" autofocus required>
<button type="submit">Sign in</button></form></body></html>"""


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
    cookie_name: str = COOKIE,
    token_hint: str = "the URL printed by flux-plumb",
    token_file: str = "~/.fluxpype/plumber/token",
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

    app = FastAPI(title="Fluxpype Plumber", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    allowed = {h.lower() for h in allowed_hosts}
    sched = scheduler
    caches: Dict[str, Cache] = {}
    log_offsets: Dict[str, int] = {}
    last_state: Dict[str, str] = {}
    app.state.hub, app.state.scheduler = hub, sched
    sched.on_change = hub.publish

    def authed(cookies, headers) -> bool:
        given = cookies.get(cookie_name) or headers.get("x-flux-token") or ""
        return hmac.compare_digest(given.encode(), token.encode())

    def host_allowed(host: str) -> bool:
        return host in allowed or host == APP_DOMAIN or host.endswith("." + APP_DOMAIN)

    def with_cookie(resp: Response) -> Response:
        resp.set_cookie(cookie_name, token, httponly=True, samesite="strict", path="/", max_age=COOKIE_MAX_AGE)
        return resp

    def login_page(error: str = "", status: int = 401) -> HTMLResponse:
        return HTMLResponse(
            login_html(error, token_file, demo), status_code=status, headers={"Cache-Control": "no-store"}
        )

    def origin_ok(headers) -> bool:
        origin = headers.get("origin")
        if not origin:
            return True
        return host_of(origin).lower() in allowed or host_of(origin) == host_of(headers.get("host", ""))

    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = host_of(request.headers.get("host", "")).lower()
        if not host_allowed(host):
            return Response("Host not allowed", status_code=403)
        path = request.url.path
        supplied = request.query_params.get("token")
        if supplied is not None and not path.startswith("/api/"):
            if not hmac.compare_digest(supplied.encode(), token.encode()):
                return login_page("That token is not this Plumber's current token.", 403)
            return with_cookie(RedirectResponse(path or "/", status_code=303))
        if path == "/login" and request.method == "POST":
            if not origin_ok(request.headers):
                return Response("cross-origin request refused", status_code=403)
            form = parse_qs((await request.body()).decode("utf-8", "replace"))
            given = (form.get("token") or [""])[0].strip()
            if not hmac.compare_digest(given.encode(), token.encode()):
                return login_page("That token is not this Plumber's current token.", 403)
            return with_cookie(RedirectResponse("/", status_code=303))
        if path.startswith("/api/"):
            if not authed(request.cookies, request.headers):
                return JSONResponse({"detail": f"missing or wrong token; open {token_hint}"}, status_code=401)
            if request.method not in ("GET", "HEAD", "OPTIONS") and not origin_ok(request.headers):
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
            return await call_next(request)
        if path in PUBLIC_FILES or path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)
        if not authed(request.cookies, request.headers):
            return login_page()
        # a page load by a signed-in browser: slide the cookie's 30-day window forward
        return with_cookie(await call_next(request))

    # ------------------------------------------------------------------ helpers
    health = Health(sched, version)
    preset_cache: Dict[str, Any] = {}

    def cache_for(profile: Optional[str] = None, backend: Optional[str] = None) -> Cache:
        data_dir = sched.data_dir(profile, backend)
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
                "plumber_home": str(sched.home),
                "config_path": sched.config_path,
                "data_dir": sched.data_dir(),
                "backends": [
                    {"name": n, **b.capabilities().__dict__, "active": n == sched.active_backend}
                    for n, b in sched.backends.items()
                ],
                "default_backend": sched.default_backend,
                "active_backend": sched.active_backend,
                "sandboxed": sched.is_sandboxed(),
                "scenarios": list(SCENARIOS) if "fake" in sched.backends else [],
                "stages": [{"id": s, "label": l} for s, l in STAGES],
                "forbidden_keys": sorted(config_io.FORBIDDEN_KEYS),
            }
        )

    @app.get("/api/health")
    def health_(refresh: bool = False):
        return reply(health.report(refresh))

    @app.put("/api/backend")
    def set_backend(body: BackendIn):
        try:
            state = sched.set_active_backend(body.name)
        except SchedulerError as e:
            fail(e)
        preset_cache.clear()
        hub.publish("backend_changed", {"active_backend": sched.active_backend})
        return reply({"active_backend": sched.active_backend, "queue": state})

    @app.get("/api/data-dir")
    def data_dir_get():
        return reply(sched.data_dir_report())

    @app.put("/api/data-dir")
    def data_dir_put(body: DataDirIn):
        try:
            sched.set_data_dir(body.path)
        except SchedulerError as e:
            fail(e)
        preset_cache.clear()
        health.invalidate()
        hub.publish("data_dir_changed", {"data_dir": sched.data_dir()})
        return reply(sched.data_dir_report())

    @app.get("/api/agent-prompt")
    def agent_prompt_():
        from . import agent_prompt

        return reply(agent_prompt.build(home=Path(sched.home), demo=demo))

    def preset_cache_summary(cf, name: str, plan) -> Dict[str, Any]:
        """How many of the preset's tasks already have each cached product (first 60 tasks), on the active backend."""
        key = f"{sched.active_backend}|{sched.data_dir_setting()}|{name}|{hash(tuple(plan.tasks[:60]))}"
        hit = preset_cache.get(key)
        if hit and time.time() - hit[0] < 20:
            return hit[1]
        sample = plan.tasks[:60]
        counts = {k: 0 for k in ("magnetogram", "footpoints", "pfss", "world", "relaxed_world")}
        for adapt, cr, nflux, _method in sample:
            ctx = sched.probe_context_for(name, cr, nflux, adapt, parsed=cf)
            if not ctx:
                continue
            for k, v in first_match_for_context(ctx).items():
                counts[k] += 1 if v else 0
        out = {"tasks": len(sample), "of": len(plan.tasks), "counts": counts}
        preset_cache[key] = (time.time(), out)
        return out

    @app.get("/api/profiles")
    def profiles(cache: bool = True):
        cf = sched.parsed()
        hidden = set(sched.presets.hidden())
        builtin = set(sched.presets.builtin_names())
        stats = sched.store.run_stats()
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
                    "own": {k: e.value for k, e in own.items()},
                    "runs": stats.get(name),
                    "origin": "builtin" if name in builtin else "user",
                    "hidden": name in hidden,
                    "cache": preset_cache_summary(cf, name, plan) if cache and not plan.errors else None,
                    "is_default_profile": cf.defaults.get("config_name") is not None
                    and cf.defaults["config_name"].value == name,
                }
            )
        return reply({"profiles": rows, "deleted": list(sched.presets.trashed_sections())})

    def preset_changed():
        preset_cache.clear()
        hub.publish("presets_changed", {})

    def preset_call(fn, *a, **kw):
        try:
            out = fn(*a, **kw)
        except PresetError as e:
            fail(e)
        preset_changed()
        return out

    @app.post("/api/profiles")
    def preset_create(body: PresetIn):
        name = preset_call(sched.presets.create, body.name, body.base, body.values)
        return reply({"name": name}, 201)

    @app.patch("/api/profiles/{name}")
    def preset_save(name: str, body: PresetSaveIn):
        preset_call(sched.presets.save, name, body.values, body.remove)
        return reply({"name": name})

    @app.post("/api/profiles/{name}/rename")
    def preset_rename(name: str, body: RenameIn):
        return reply({"name": preset_call(sched.presets.rename, name, body.name)})

    @app.post("/api/profiles/{name}/hidden")
    def preset_hide(name: str, body: HiddenIn):
        return reply({"hidden": preset_call(sched.presets.set_hidden, name, body.hidden)})

    @app.delete("/api/profiles/{name}")
    def preset_delete(name: str):
        preset_call(sched.presets.delete, name)
        return reply({"deleted": name})

    @app.post("/api/profiles-deleted/{name}/restore")
    def preset_restore(name: str):
        return reply({"name": preset_call(sched.presets.restore_deleted, name)})

    @app.get("/api/profiles-export")
    def preset_export(names: str):
        wanted = [n for n in names.split(",") if n]
        try:
            text = sched.presets.export(wanted)
        except PresetError as e:
            fail(e)
        fname = (wanted[0] if len(wanted) == 1 else "presets") + ".ini"
        return Response(
            text,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    @app.post("/api/profiles-import")
    def preset_import(body: ImportIn):
        return reply({"added": preset_call(sched.presets.import_text, body.text)}, 201)

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
        data_dir = sched.data_dir(name)
        tasks = []
        for adapt, cr, nflux, method in plan.tasks[:200]:
            ctx = sched.probe_context_for(name, cr, nflux, adapt, parsed=cf)
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
                "origin": sched.presets.origin(name),
                "hidden": name in set(sched.presets.hidden()),
                "batch_name": config_io.resolve(eff, "batch_name"),
                "backend": sched.active_backend,
                "sandboxed": sched.is_sandboxed(),
            }
        )

    @app.post("/api/jobs")
    def create_job(body: JobIn):
        extra: Dict[str, str] = {}
        backend = body.backend or sched.active_backend
        if body.scenario is not None or body.speed is not None:
            if not sched.is_sandboxed(backend):
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

    def job_or_404(job_id: str) -> Dict[str, Any]:
        job = sched.store.get_job(job_id)
        if not job:
            raise HTTPException(404, "unknown job")
        return job

    def job_effective(job: Dict[str, Any]):
        try:
            return config_io.read_file(job["snapshot_path"]).effective(job["profile"])
        except Exception as e:  # noqa: BLE001
            fail(ValueError(f"this job's saved settings cannot be read: {e}"), 410)

    def job_data_dir(job_id: str) -> str:
        eff = job_effective(job_or_404(job_id))
        return config_io.resolved_path(eff, "data_dir", str(paths.checkout_root()))

    @app.get("/api/jobs/{job_id}/config")
    def job_config(job_id: str):
        """The settings this job ran with (its frozen snapshot), marked against the preset as it is now."""
        job = job_or_404(job_id)
        eff = job_effective(job)
        try:
            now = sched.parsed().effective(job["profile"]) if job["profile"] in sched.parsed().sections else None
        except Exception:  # noqa: BLE001
            now = None
        entries = []
        pinned = ("data_dir", "base_dir")  # Plumber writes these into every snapshot; a difference is not a user change
        for key, e in eff.items():
            current = now[key].value if now is not None and key in now else None
            entries.append(
                {
                    "key": key,
                    "value": e.value,
                    "resolved": config_io.resolve(eff, key) if "${" in e.value else None,
                    "source": "DEFAULT" if e.section == "DEFAULT" else "profile",
                    "overrides_default": e.overrides_default,
                    "comment": e.comment,
                    "editable": key not in config_io.FORBIDDEN_KEYS,
                    "overridden": key in (job.get("overrides") or {}),
                    "preset_value": current,
                    "differs_from_preset": now is not None and current != e.value and key not in pinned,
                    "set_by_plumber": key in pinned,
                }
            )
        return reply(
            {
                "job_id": job_id,
                "profile": job["profile"],
                "preset_exists": now is not None,
                "overrides": job.get("overrides") or {},
                "data_dir": config_io.resolved_path(eff, "data_dir", str(paths.checkout_root())),
                "batch_name": config_io.resolve(eff, "batch_name"),
                "entries": entries,
            }
        )

    def results_for(t: Dict[str, Any]) -> Dict[str, Any]:
        from .results import task_results

        out = task_results(sched.probe_context(t), t.get("started"), t.get("ended"))
        try:
            stages = {s["id"]: s["status"] for s in sched.task_state(t["id"]).get("stages", [])}
        except Exception:  # noqa: BLE001
            stages = {}
        for prod in out["products"]:
            prod["stage_status"] = stages.get(PRODUCT_STAGE.get(prod["id"], prod["id"]))
        return out

    @app.get("/api/tasks/{task_id}/results")
    def task_results_view(task_id: str):
        t = sched.store.get_task(task_id)
        if not t:
            raise HTTPException(404, "unknown task")
        return reply({"task_id": task_id, "job_id": t["job_id"], **results_for(t)})

    @app.get("/api/jobs/{job_id}/files")
    def job_files(job_id: str):
        """Every file the job's tasks read or wrote (expected products plus plots), once each."""
        job_or_404(job_id)
        by_path: Dict[str, Dict[str, Any]] = {}
        missing = []
        data_dir = None
        for t in sched.store.tasks_for_job(job_id):
            r = results_for(t)
            data_dir = data_dir or r["data_dir"]
            for prod in r["products"]:
                if not prod["files"] and prod.get("stage_status") not in ("skipped", "pending", "running"):
                    missing.append(
                        {
                            "task_id": t["id"],
                            "cr": t["cr"],
                            "nflux": t["nflux"],
                            "stage": prod["id"],
                            "label": prod["label"],
                        }
                    )
                for f in prod["files"]:
                    by_path.setdefault(f["path"], {**f, "stage": prod["id"], "tasks": []})["tasks"].append(t["id"])
            for f in r["plots"]:
                by_path.setdefault(f["path"], {**f, "stage": "plots", "tasks": []})["tasks"].append(t["id"])
        current = os.path.realpath(sched.data_dir()) if data_dir else None
        return reply(
            {
                "job_id": job_id,
                "data_dir": data_dir,
                "is_current_data_dir": bool(data_dir) and os.path.realpath(data_dir) == current,
                "files": sorted(by_path.values(), key=lambda f: f["path"]),
                "missing": missing,
            }
        )

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

    GREP_MAX_LINES = 400
    GREP_LINE_CHARS = 2000

    def grep_log(path: str, size: int, needle: str, context: int) -> Dict[str, Any]:
        """Line windows around case-insensitive substring matches, as `N:line` (match) / `N-line` (context)."""
        wanted, needle = needle, needle.lower()
        before: deque = deque(maxlen=context)
        out: List[str] = []
        matches = 0
        after_left = 0
        last_emitted = 0
        truncated = False

        def emit(n: int, text: str, is_match: bool) -> bool:
            nonlocal last_emitted
            if len(out) >= GREP_MAX_LINES:
                return False
            if last_emitted and n > last_emitted + 1:
                out.append("--")
            out.append(f"{n}{':' if is_match else '-'}{text[:GREP_LINE_CHARS]}")
            last_emitted = n
            return True

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                line = line.rstrip("\r\n")
                if needle in line.lower():
                    matches += 1
                    ok = True
                    while before and ok:
                        bn, btext = before.popleft()
                        if bn > last_emitted:
                            ok = emit(bn, btext, False)
                    before.clear()
                    if ok:
                        ok = emit(n, line, True)
                    if not ok:
                        truncated = True
                        break
                    after_left = context
                elif after_left > 0:
                    after_left -= 1
                    if not emit(n, line, False):
                        truncated = True
                        break
                else:
                    before.append((n, line))
        lines = [x for x in out if x != "--"]
        return {
            "file": os.path.basename(path),
            "size": size,
            "grep": wanted,
            "context": context,
            "matches": matches,
            "lines": len(lines),
            "truncated": truncated,
            "text": "\n".join(out),
        }

    def log_file(t) -> Optional[str]:
        for name in ("run.log", "stdout.log"):
            p = os.path.join(t["run_dir"], name)
            if os.path.exists(p):
                return p
        return None

    @app.get("/api/tasks/{task_id}/log")
    def task_log(task_id: str, offset: int = -1, limit: int = 262144, grep: Optional[str] = None, context: int = 3):
        """Log text from byte `offset` (default: the last `limit` bytes). Returns the next offset.

        With `grep` (a case-insensitive plain substring), returns matching lines with `context` lines around
        each (0 to 20), numbered like `grep -n`, at most GREP_MAX_LINES lines in all.
        """
        t = sched.store.get_task(task_id)
        if not t:
            raise HTTPException(404, "unknown task")
        path = log_file(t)
        if path is None:
            return reply({"offset": 0, "size": 0, "text": "", "file": None})
        size = os.path.getsize(path)
        if grep:
            return reply(grep_log(path, size, grep, min(max(context, 0), 20)))
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
        return reply(
            {
                "offset": end,
                "start": start,
                "size": size,
                "text": data.decode("utf-8", "replace"),
                "file": os.path.basename(path),
            }
        )

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
                    "stages": [
                        {"id": s["id"], "status": s["status"], "source": s["source"]} for s in state.get("stages", [])
                    ],
                    "failure": next(
                        (s["detail"] for s in state.get("stages", []) if s["status"] == "failed" and s.get("detail")),
                        None,
                    ),
                }
            )
        return reply({"queue": sched.queue_state(), "running": running, "pending": pending, "recent": recent})

    # ------------------------------------------------------------------ cache
    @app.get("/api/cache")
    def cache_get(rescan: bool = False, profile: Optional[str] = None):
        c = cache_for(profile)
        summary = c.summary(rescan)
        records = c.records()[:20000]
        return reply(
            {
                **summary,
                "records": records,
                "truncated": summary["total_count"] > len(records),
                "presets": preset_scopes(),
                "sandboxed": sched.is_sandboxed(),
            }
        )

    def preset_scopes() -> List[Dict[str, Any]]:
        """For each runnable preset: the batch, rotations and instruments it reads and writes."""
        out = []
        try:
            cf = sched.parsed()
        except Exception:  # noqa: BLE001
            return out
        hidden = set(sched.presets.hidden())
        for name in cf.profiles:
            eff = cf.effective(name)
            plan = config_io.expand_tasks(eff)
            if plan.errors:
                continue
            out.append(
                {
                    "name": name,
                    "batch": config_io.resolve(eff, "batch_name"),
                    "crs": sorted({str(t[1]) for t in plan.tasks}),
                    "insts": sorted({config_io.task_instrument(t[0]) for t in plan.tasks}),
                    "hidden": name in hidden,
                }
            )
        return out

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
    def cache_file(rel: str) -> str:
        """Absolute path of a file in the current data folder (confined like the trash: no .., no escaping links)."""
        from .cache import _confine

        try:
            full = _confine(cache_for().data_dir, rel)
        except CacheError as e:
            fail(e, 403)
        if not os.path.isfile(full):
            raise HTTPException(404, "no such file (rescan the cache)")
        return full

    def render_python() -> str:
        local = next((b for b in sched.backends.values() if getattr(b, "kind", "") == "local"), None)
        return getattr(local, "python", None) or sys.executable

    def preview_reply(full: str, rel: str):
        from . import preview

        try:
            return reply(preview.describe(full, rel))
        except OSError as e:
            fail(e, 500)

    def raw_reply(full: str, download: bool):
        from . import preview

        mt = preview.media_type(full)
        inline = not download and mt != "application/octet-stream"
        headers = {"Cache-Control": "private, max-age=30", "X-Content-Type-Options": "nosniff"}
        if mt == "image/svg+xml":
            headers["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'; sandbox"
        return FileResponse(
            full,
            media_type=mt,
            headers=headers,
            filename=os.path.basename(full),
            content_disposition_type="inline" if inline else "attachment",
        )

    def render_reply(full: str, png_url: str):
        from . import preview

        if preview.kind_of(full) != "fits":
            fail(ValueError("only FITS files are rendered"), 415)
        try:
            out = preview.render_fits(full, str(sched.home), render_python())
        except preview.RenderError as e:
            fail(e, 422)
        return reply({"stats": out["stats"], "image": png_url})

    def render_png(full: str):
        from . import preview

        try:
            out = preview.render_fits(full, str(sched.home), render_python())
        except preview.RenderError as e:
            fail(e, 422)
        return FileResponse(out["png"], media_type="image/png", headers={"Cache-Control": "private, max-age=300"})

    @app.get("/api/cache/preview")
    def cache_preview(path: str):
        return preview_reply(cache_file(path), path)

    @app.get("/api/cache/file")
    def cache_raw(path: str, download: bool = False):
        return raw_reply(cache_file(path), download)

    @app.get("/api/cache/render")
    def cache_render(path: str):
        return render_reply(cache_file(path), f"/api/cache/render.png?path={quote(path)}")

    @app.get("/api/cache/render.png")
    def cache_render_png(path: str):
        return render_png(cache_file(path))

    # job-scoped twins: a job's snapshot can pin a data folder other than the current one
    def job_file(job_id: str, rel: str) -> str:
        from .cache import _confine

        data_dir = job_data_dir(job_id)
        try:
            full = _confine(data_dir, rel)
        except CacheError as e:
            fail(e, 403)
        if not os.path.isfile(full):
            raise HTTPException(404, "no such file")
        return full

    @app.get("/api/jobs/{job_id}/data/preview")
    def job_preview(job_id: str, path: str):
        return preview_reply(job_file(job_id, path), path)

    @app.get("/api/jobs/{job_id}/data/file")
    def job_raw(job_id: str, path: str, download: bool = False):
        return raw_reply(job_file(job_id, path), download)

    @app.get("/api/jobs/{job_id}/data/render")
    def job_render(job_id: str, path: str):
        return render_reply(job_file(job_id, path), f"/api/jobs/{quote(job_id)}/data/render.png?path={quote(path)}")

    @app.get("/api/jobs/{job_id}/data/render.png")
    def job_render_png(job_id: str, path: str):
        return render_png(job_file(job_id, path))

    @app.get("/api/artifacts")
    def artifact(path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext not in IMAGE_TYPES:
            raise HTTPException(415, "only images are served")
        real = os.path.realpath(os.path.expanduser(path))
        roots = [os.path.realpath(d) for d in sched.all_data_dirs()]
        roots.append(os.path.realpath(str(paths.runs_dir(sched.home))))
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
        if not host_allowed(host) or not authed(sock.cookies, sock.headers) or not origin_ok(sock.headers):
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
                        hub.publish(
                            "log_append",
                            {
                                "task_id": t["id"],
                                "start": pos,
                                "offset": pos + cut,
                                "text": data[:cut].decode("utf-8", "replace"),
                            },
                        )
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
                print(f"[plumber] tick failed: {e!r}")
            await asyncio.sleep(tick_interval)

    # ------------------------------------------------------------------ frontend
    @app.get("/login")
    def login_get():
        return login_page(status=200)

    @app.get("/manifest.webmanifest")
    def manifest():
        name = "Plumber (demo)" if demo else "Plumber"
        body = {
            "name": f"Fluxpype {name}",
            "short_name": name,
            "description": "Queue, watch and inspect fluxpype runs",
            "id": "/",
            "start_url": "/",
            "scope": "/",
            "display": "standalone",
            "background_color": "#0b0d12",
            "theme_color": "#0b0d12",
            "icons": [
                {"src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png"},
                {"src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png"},
                {"src": "/icons/maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"},
            ],
        }
        return JSONResponse(body, media_type="application/manifest+json", headers={"Cache-Control": "no-cache"})

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(404, "not found")
        if not WEB_DIR.is_dir():
            return HTMLResponse(
                "<h1>Fluxpype Plumber</h1><p>The frontend build is missing (fluxpype/plumber/web/dist). "
                "Run <code>npm install &amp;&amp; npm run build</code> in fluxpype/plumber/web.</p>",
                status_code=503,
            )
        target = (WEB_DIR / full_path).resolve()
        if full_path and WEB_DIR.resolve() in target.parents and target.is_file():
            headers = {"Cache-Control": "public, max-age=31536000, immutable"} if "/assets/" in f"/{full_path}" else {}
            return FileResponse(target, headers=headers)
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app
