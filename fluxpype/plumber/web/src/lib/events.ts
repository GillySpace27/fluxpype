import type { PlumberEvent } from "./types";

/** Kinds the inbox lists (notify.INBOX keeps the same set). */
export const INBOX = new Set([
  "task_failed",
  "task_timeout",
  "task_crashed",
  "job_ok",
  "job_partial",
  "job_failed",
  "task_stalled",
  "backend_down",
]);

const TITLE: Record<string, string> = {
  job_queued: "Job queued",
  task_started: "Task started",
  task_ok: "Task finished",
  task_failed: "Task failed",
  task_timeout: "Task timed out",
  task_crashed: "Task crashed",
  task_cancelled: "Task cancelled",
  job_ok: "Job succeeded",
  job_partial: "Job partly failed",
  job_failed: "Job failed",
  job_cancelled: "Job cancelled",
  task_stalled: "Task stalled",
  task_resumed: "Task output resumed",
  task_degraded: "Task events stopped",
  queue_paused: "Queue paused",
  queue_resumed: "Queue resumed",
  backend_down: "Backend unreachable",
  backend_up: "Backend reachable again",
  plumber_started: "Plumber started",
};

const ID = /^[a-z0-9-]{1,40}$/;
const int = (v: unknown) => (typeof v === "number" && Number.isInteger(v) ? v : null);

export const eventTitle = (e: PlumberEvent) => TITLE[e.kind] ?? "Event";

/** One short line from the event's ids and integers only (never log text: events carry none). */
export function eventDetail(e: PlumberEvent): string {
  const f = e.f ?? {};
  const out: string[] = [];
  const id = e.task_id ?? e.job_id;
  if (id && ID.test(id)) out.push(id);
  if (typeof f.stage === "string" && ID.test(f.stage)) out.push(`at ${f.stage}`);
  if (int(f.exit) !== null && e.kind !== "task_ok") out.push(`exit ${f.exit}`);
  if (f.reason === "submit") out.push("backend refused to start it");
  const quiet = int(f.quiet_s);
  if (quiet !== null) out.push(quiet < 120 ? `no output for ${quiet} s` : `no output for ${Math.round(quiet / 60)} min`);
  if (e.kind.startsWith("job_") && e.kind !== "job_queued") {
    const bad = ["failed", "timeout", "crashed"].reduce((n, k) => n + (int(f[k]) ?? 0), 0);
    out.push(`${int(f.ok) ?? 0} ok${bad ? `, ${bad} failed` : ""}${int(f.cancelled) ? `, ${f.cancelled} cancelled` : ""}`);
  }
  if (e.kind === "job_queued" && int(f.tasks) !== null) out.push(`${f.tasks} task${f.tasks === 1 ? "" : "s"}`);
  if (typeof f.backend === "string" && ID.test(f.backend)) out.push(f.backend);
  if (f.launcher_lost === 1) out.push("launcher lost");
  return out.join(" · ");
}

export const eventLink = (e: PlumberEvent): string | undefined =>
  e.job_id && ID.test(e.job_id)
    ? `/jobs/${e.job_id}${e.task_id && ID.test(e.task_id) ? `/${e.task_id}` : ""}`
    : undefined;

/** Worth a toast (and a browser notification): errors, stalls, and this tab's own successful jobs. */
export const toastWorthy = (e: PlumberEvent, tab: string) =>
  e.sev === "error" || e.kind === "task_stalled" || (e.kind === "job_ok" && !!tab && e.origin?.tab === tab);

/** Tasks whose latest stall has no later resume (events newest first). */
export function stalledTasks(events: PlumberEvent[]): Set<string> {
  const seen = new Set<string>();
  const out = new Set<string>();
  for (const e of events) {
    if (!e.task_id || seen.has(e.task_id)) continue;
    if (e.kind === "task_stalled") out.add(e.task_id);
    if (["task_stalled", "task_resumed", "task_ok", "task_failed", "task_timeout", "task_crashed", "task_cancelled"].includes(e.kind))
      seen.add(e.task_id);
  }
  return out;
}

const NOTIFY_KEY = "plumber.notify";
export const notifySupported = () => typeof window !== "undefined" && "Notification" in window && window.isSecureContext;
export function notifyEnabled(): boolean {
  try {
    return notifySupported() && Notification.permission === "granted" && localStorage.getItem(NOTIFY_KEY) === "1";
  } catch {
    return false;
  }
}
export function setNotifyEnabled(on: boolean) {
  try {
    localStorage.setItem(NOTIFY_KEY, on ? "1" : "0");
  } catch {
    /* private window: the setting just does not stick */
  }
}
