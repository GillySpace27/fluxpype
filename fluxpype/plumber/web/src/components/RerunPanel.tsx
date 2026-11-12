import { useState } from "react";
import { Link } from "react-router-dom";
import { ChevronDown, RotateCcw } from "lucide-react";
import { useRerunOptions, useRerunWind } from "../lib/hooks";
import type { RerunField, TaskState } from "../lib/types";

const NOTE = {
  light:
    "Recomputes the image from this task's relaxed world and replaces the cached FLUXlight result for that world. It can take many minutes. Images made with other settings keep their own names.",
  wind: "Queues a new run in the same batch that reads this task's relaxed world and redoes only the wind and the wind plots. Earlier wind files made with the same method are replaced and no copy is kept. The first stages are not rerun.",
};

function Field({ f, v, onChange }: { f: RerunField; v: string; onChange: (v: string) => void }) {
  return (
    <label className="block min-w-[10rem] flex-1 text-xs" title={f.hint}>
      <span className="text-muted">{f.label}</span>
      {f.kind === "choice" ? (
        <select className="input mt-1" value={v} onChange={(e) => onChange(e.target.value)}>
          {!f.choices!.includes(v) && <option value={v}>{v || "(job's value)"}</option>}
          {f.choices!.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      ) : (
        <input className="input num mt-1" value={v} onChange={(e) => onChange(e.target.value)} inputMode="decimal" spellCheck={false} placeholder="(job's value)" />
      )}
    </label>
  );
}

/** Regenerate the end of the pipeline ("wind" redoes the wind and its plots, "light" redoes FLUXlight) with other settings, without rerunning the rest. */
export function RerunPanel({ task, group, blocked, onLight }: { task: TaskState; group: "wind" | "light"; blocked: boolean; onLight: (settings: Record<string, string>) => void }) {
  const { data } = useRerunOptions(task.id);
  const wind = useRerunWind(task.id);
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState<Record<string, string>>({});
  const g = data?.[group];
  const live = task.status === "running" || task.status === "pending";
  if (!g) return null;
  const val = (f: RerunField) => draft[f.key] ?? f.value;
  const changed = Object.fromEntries(g.fields.filter((f) => draft[f.key] !== undefined && draft[f.key] !== f.value).map((f) => [f.key, draft[f.key]]));
  const why = live ? "Wait for the task to finish" : !g.can ? "Needs a relaxed world on disk" : blocked ? "Another run is going" : "";
  const run = () => (group === "light" ? onLight(changed) : wind.mutate(changed));
  return (
    <div className="rounded-xl border border-line/80 bg-bg/40">
      <button className="flex w-full items-center gap-2 px-3 py-2 text-left text-[13px] font-medium" onClick={() => setOpen(!open)} aria-expanded={open}>
        <RotateCcw className="h-3.5 w-3.5 text-accent" />
        {group === "light" ? "Regenerate FLUXlight with other settings" : "Re-run the wind and its plots with other settings"}
        <ChevronDown className={`ml-auto h-4 w-4 text-muted transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="space-y-3 border-t border-line/60 px-3 py-3">
          <div className="flex flex-wrap gap-3">
            {g.fields.map((f) => (
              <Field key={f.key} f={f} v={val(f)} onChange={(v) => setDraft({ ...draft, [f.key]: v })} />
            ))}
          </div>
          <p className="text-xs text-muted">{NOTE[group]}</p>
          <div className="flex flex-wrap items-center gap-3">
            <button className="btn btn-sm" disabled={!!why || wind.isPending} title={why || undefined} onClick={run}>
              {group === "light" ? "Regenerate FLUXlight" : "Re-run wind and plots"}
            </button>
            {Object.keys(changed).length > 0 && (
              <button className="text-xs text-accent hover:underline" onClick={() => setDraft({})}>
                Reset to this job's values
              </button>
            )}
            {why && <span className="text-xs text-faint">{why}</span>}
            {group === "wind" && wind.data && (
              <Link to={`/jobs/${wind.data.id}`} className="text-xs font-medium text-accent hover:underline">
                Open the new run
              </Link>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
