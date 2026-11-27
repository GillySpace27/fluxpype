import { ChevronRight, FlaskConical, Layers } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { instOf } from "../lib/format";
import { useCache, useHealth, useStatus } from "../lib/hooks";
import type { ProfileDetail, ProfileEntry } from "../lib/types";
import { Badge, Checkbox } from "./ui";

/** The knobs worth changing per run, shown first. */
export const COMMON = ["rotations", "fluxon_count", "adapts", "flow_method", "max_cycles", "relax_threshold", "do_steps", "light", "do_wind_calc"];

/** The "Compare with AIA" section's settings; they get their own controls instead of plain rows. */
export const AIA_KEYS = ["observation_date", "mag_file", "aia_channels"];
export const AIA_CHANNELS = ["171", "193", "211"];

const valueOf = (detail: ProfileDetail, d: RunDraft, key: string) => d.overrides[key] ?? detail.entries.find((e) => e.key === key)?.value ?? "";
const unset = (v: string) => ["", "none"].includes(v.trim().toLowerCase());

/** The section applies when this run renders FLUXlight (light = 1) and the config has the keys it sets. */
export const aiaSectionShown = (detail: ProfileDetail, d: RunDraft) => valueOf(detail, d, "light").trim() === "1" && AIA_KEYS.every((k) => detail.entries.some((e) => e.key === k && e.editable));

/** "[171, 193]" or "193" to ["171", "193"]. */
export const parseChannels = (v: string) => (v.match(/\d+/g) ?? []).filter((c) => AIA_CHANNELS.includes(c));
/** Channels back to the config list syntax; none is the unset sentinel `None`. */
export const formatChannels = (cs: string[]) => `[${AIA_CHANNELS.filter((c) => cs.includes(c)).join(", ")}]`;

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
  const aia = aiaSectionShown(detail, draft);
  const own = (e: ProfileEntry) => AIA_KEYS.includes(e.key) || (aia && e.key === "observer_lon");
  const common = editable.filter((e) => COMMON.includes(e.key) && !own(e));
  const rest = editable.filter((e) => !COMMON.includes(e.key) && !own(e));
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

  const set = (key: string, v: string) => onChange({ ...draft, overrides: { ...draft.overrides, [key]: v } });
  const firstCr = pickedTasks(detail, draft)[0]?.cr ?? detail.tasks[0]?.cr;
  const when = valueOf(detail, draft, "observation_date").trim();
  const derive = useQuery({
    queryKey: ["aia-observer", firstCr, unset(when) ? "" : when],
    queryFn: () => api.get<{ time: string; observer_lon: number }>(`/api/aia/observer?${new URLSearchParams({ ...(firstCr ? { cr: firstCr } : {}), ...(unset(when) ? {} : { time: when }) })}`),
    enabled: aia && !!firstCr,
    retry: false,
  });
  const cache = useCache();
  const mags = (cache.data?.records ?? []).filter((r) => r.kind === "magnetogram");
  const channels = parseChannels(valueOf(detail, draft, "aia_channels"));
  const magFile = valueOf(detail, draft, "mag_file");

  const aiaSection = aia && (
    <section className="space-y-3" data-testid="compare-aia">
      <p className="label">Compare with AIA</p>
      <label className="block">
        <span className="flex items-baseline justify-between text-xs">
          <span>Observation time (UTC)</span>
          {derive.data && (
            <button type="button" className="text-accent hover:underline" onClick={() => set("observation_date", derive.data.time)}>
              use mid-rotation {derive.data.time.slice(0, 10)}
            </button>
          )}
        </span>
        <input
          className="input mt-1 font-mono"
          placeholder="2014-05-18T20:00:00 (blank: keep the preset)"
          value={unset(when) ? "" : when}
          onChange={(ev) => set("observation_date", ev.target.value.trim() || "None")}
          spellCheck={false}
        />
      </label>
      <p className="text-xs text-muted">
        observer_lon:{" "}
        {unset(when) ? (
          <span className="font-mono">{valueOf(detail, draft, "observer_lon") || "pole-on"} (from the preset)</span>
        ) : derive.data ? (
          <span className="font-mono">{derive.data.observer_lon} (derived: Earth central meridian at that time)</span>
        ) : (
          <span className="text-faint">{derive.error ? "enter a valid ISO time" : "computing"}</span>
        )}
      </p>
      <div>
        <span className="text-xs">AIA channels</span>
        <div className="mt-1 flex gap-4">
          {AIA_CHANNELS.map((c) => (
            <label key={c} className="flex items-center gap-1.5 text-[13px]">
              <Checkbox
                checked={channels.includes(c)}
                label={`AIA ${c}`}
                onChange={(on) => set("aia_channels", formatChannels(on ? [...channels, c] : channels.filter((x) => x !== c)))}
              />
              {c}
            </label>
          ))}
        </div>
      </div>
      <label className="block text-xs">
        <span>Magnetogram file (optional)</span>
        <select className="input mt-1 font-mono" value={unset(magFile) ? "" : magFile} onChange={(ev) => set("mag_file", ev.target.value || "None")}>
          <option value="">none (the pipeline picks its own)</option>
          {mags.map((m) => (
            <option key={m.id} value={m.path}>
              {m.name}
            </option>
          ))}
        </select>
      </label>
    </section>
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
      {aiaSection}
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
