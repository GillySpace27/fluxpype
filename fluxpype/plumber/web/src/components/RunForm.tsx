import { ChevronRight, FlaskConical, Layers } from "lucide-react";
import { useState } from "react";
import { cn } from "../lib/cn";
import { instOf } from "../lib/format";
import { useHealth, useStatus } from "../lib/hooks";
import type { ProfileDetail, ProfileEntry } from "../lib/types";
import { Badge, Checkbox } from "./ui";

/** The knobs worth changing per run, shown first. */
export const COMMON = ["rotations", "fluxon_count", "adapts", "flow_method", "max_cycles", "relax_threshold", "do_steps", "light", "do_wind_calc"];

/** What a run will be launched with, before it exists. `picked` holds task keys; null means every task. */
export interface RunDraft {
  overrides: Record<string, string>;
  picked: string[] | null;
  scenario: string;
  speed: string;
}

export const emptyDraft = (): RunDraft => ({ overrides: {}, picked: null, scenario: "ok", speed: "20" });

type Task = ProfileDetail["tasks"][number];
export const taskKey = (t: { cr: string; nflux: string; adapt: string; method: string }) => `${t.cr}|${t.nflux}|${t.adapt}|${t.method}`;

/** Overrides that actually differ from the preset. */
export function changedOverrides(detail: ProfileDetail, d: RunDraft): [string, string][] {
  return Object.entries(d.overrides).filter(([k, v]) => v !== detail.entries.find((e) => e.key === k)?.value);
}

export function pickedTasks(detail: ProfileDetail, d: RunDraft): Task[] {
  if (d.picked === null) return detail.tasks;
  const want = new Set(d.picked);
  return detail.tasks.filter((t) => want.has(taskKey(t)));
}

/** The POST /api/jobs body for a draft. */
export function jobBody(detail: ProfileDetail, d: RunDraft, sim: boolean): Record<string, unknown> {
  const body: Record<string, unknown> = { profile: detail.name, overrides: Object.fromEntries(changedOverrides(detail, d)) };
  const tasks = pickedTasks(detail, d);
  if (tasks.length !== detail.tasks.length) body.tasks = tasks.map((t) => [t.cr, t.nflux, t.adapt, t.method]);
  if (sim) {
    body.scenario = d.scenario;
    body.speed = Number(d.speed) || 20;
  }
  return body;
}

/**
 * Backend note, simulated-run options, task picker and per-run settings for one preset.
 * Shared by the preset run drawer and the run view's Settings tab.
 */
export function RunForm({ detail, draft, onChange, wide }: { detail: ProfileDetail; draft: RunDraft; onChange: (d: RunDraft) => void; wide?: boolean }) {
  const { data: status } = useStatus();
  const { data: health } = useHealth();
  const [showAll, setShowAll] = useState(false);
  const sim = !!status?.sandboxed;
  const backend = health?.backends.find((b) => b.active);
  const editable = detail.entries.filter((e) => e.editable);
  const common = editable.filter((e) => COMMON.includes(e.key));
  const rest = editable.filter((e) => !COMMON.includes(e.key));
  const keys = detail.tasks.map(taskKey);
  const picked = new Set(draft.picked ?? keys);
  const allPicked = keys.every((k) => picked.has(k));
  const setPicked = (next: Set<string>) => onChange({ ...draft, picked: keys.every((k) => next.has(k)) ? null : keys.filter((k) => next.has(k)) });
  const toggle = (k: string) => {
    const n = new Set(picked);
    if (n.has(k)) n.delete(k);
    else n.add(k);
    setPicked(n);
  };

  const field = (e: ProfileEntry) => (
    <label key={e.key} className="block">
      <span className="flex items-baseline justify-between text-xs">
        <span className="font-mono text-ink/90">{e.key}</span>
        {draft.overrides[e.key] !== undefined && draft.overrides[e.key] !== e.value && (
          <button
            type="button"
            className="text-accent hover:underline"
            title={`Back to the preset value: ${e.value}`}
            onClick={(ev) => {
              ev.preventDefault();
              const o = { ...draft.overrides };
              delete o[e.key];
              onChange({ ...draft, overrides: o });
            }}
          >
            changed · reset
          </button>
        )}
      </span>
      <input className="input mt-1 font-mono" value={draft.overrides[e.key] ?? e.value} onChange={(ev) => onChange({ ...draft, overrides: { ...draft.overrides, [e.key]: ev.target.value } })} spellCheck={false} />
      {e.comment && <span className="mt-1 block text-[11px] leading-snug text-faint">{e.comment}</span>}
    </label>
  );

  const tasks = (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <p className="label">Tasks</p>
        <button className="text-xs text-muted hover:text-ink" onClick={() => setPicked(allPicked ? new Set() : new Set(keys))}>
          {allPicked ? "Select none" : "Select all"}
        </button>
      </div>
      <div className={cn("divide-y divide-line/50 overflow-y-auto rounded-lg border border-line", wide ? "max-h-[420px]" : "max-h-56")}>
        {detail.tasks.map((t) => {
          const k = taskKey(t);
          return (
            <div
              key={k}
              role="button"
              tabIndex={0}
              onClick={() => toggle(k)}
              onKeyDown={(ev) => ev.key === " " && ev.preventDefault()}
              className="flex cursor-pointer items-center gap-3 px-3 py-2 text-[13px] hover:bg-raised/50"
            >
              <Checkbox checked={picked.has(k)} label={`CR ${t.cr}`} onChange={() => toggle(k)} />
              <span className="flex-1">
                CR {t.cr} <span className="text-muted">· {t.nflux} · {instOf(t.adapt)} · {t.method}</span>
              </span>
              {t.cache.relaxed_world && <Badge tone="ok">cached</Badge>}
            </div>
          );
        })}
      </div>
      {detail.truncated && <p className="mt-1.5 text-[11px] text-faint">Showing the first {detail.tasks.length} of {detail.task_count} tasks.</p>}
    </section>
  );

  const settings = (
    <section className="space-y-3">
      <p className="label">Settings for this run</p>
      {common.map(field)}
      {rest.length > 0 && (
        <>
          <button className="flex items-center gap-1 text-xs text-muted hover:text-ink" onClick={() => setShowAll(!showAll)}>
            <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", showAll && "rotate-90")} />
            {showAll ? "Hide" : "Show"} {rest.length} more settings
          </button>
          {showAll && <div className="space-y-3">{rest.map(field)}</div>}
        </>
      )}
    </section>
  );

  return (
    <div className="space-y-6">
      {backend && (
        <div className={cn("flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-[13px]", sim ? "border-accent/40 bg-accent/10" : "border-line bg-raised/50")}>
          {sim ? <FlaskConical className="h-4 w-4 text-accent" /> : <Layers className="h-4 w-4 text-muted" />}
          <span>
            Runs on <span className="font-semibold">{backend.label}</span>
            {sim ? ": simulated, results go to the sandbox." : "."}
          </span>
          <span className="ml-auto text-[11px] text-faint">change it in the top bar</span>
        </div>
      )}
      {sim && (
        <section className="grid max-w-md grid-cols-[1fr_96px] gap-3">
          <label className="block text-xs">
            <span className="label">Scenario</span>
            <select className="input mt-1" value={draft.scenario} onChange={(e) => onChange({ ...draft, scenario: e.target.value })}>
              {(status?.scenarios ?? ["ok"]).map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="block text-xs">
            <span className="label">Speed x</span>
            <input className="input num mt-1" value={draft.speed} onChange={(e) => onChange({ ...draft, speed: e.target.value })} inputMode="decimal" />
          </label>
        </section>
      )}
      {wide ? (
        <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          {tasks}
          {settings}
        </div>
      ) : (
        <>
          {tasks}
          {settings}
        </>
      )}
    </div>
  );
}
