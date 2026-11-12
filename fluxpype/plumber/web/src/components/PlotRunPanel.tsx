import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { api } from "../lib/api";
import { runActive } from "../lib/hooks";
import { cn } from "../lib/cn";
import { duration } from "../lib/format";
import type { PlotRun } from "../lib/types";
import { StatusIcon } from "./ui";

/** The log of one plot run, polled while it runs. */
function RunLog({ run, open }: { run: PlotRun; open: boolean }) {
  const { data } = useQuery({
    queryKey: ["plotlog", run.task_id, run.id],
    queryFn: () => api.get<{ text: string }>(`/api/tasks/${run.task_id}/plots/${run.id}/log`),
    enabled: open,
    refetchInterval: open && runActive(run) ? 2000 : false,
  });
  return (
    <pre className="max-h-64 overflow-auto rounded-lg border border-line/70 bg-black/40 p-2 font-mono text-[11.5px] leading-snug text-ink/85">
      {data?.text?.trim() || "(no output yet)"}
    </pre>
  );
}

/** Progress of the latest plot run for a stage: one chip per plot, a Cancel button, and a log. */
export function PlotRunPanel({ runs, focus }: { runs: PlotRun[]; focus?: { id: string; n: number } | null }) {
  const qc = useQueryClient();
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const [logOpen, setLogOpen] = useState<string | null>(null);
  // "Show log" on a plot card: show that run and open its log (n changes on every click)
  useEffect(() => {
    if (focus) {
      setLogOpen(focus.id);
      setDismissed((d) => new Set([...d].filter((x) => x !== focus.id)));
    }
  }, [focus]);
  const run = runs.find((r) => r.id === focus?.id) ?? runs[0];
  if (!run || dismissed.has(run.id)) return null;
  const active = runActive(run);
  const failed = run.plots.some((p) => p.status === "failed") || ["failed", "crashed"].includes(run.status);
  const open = logOpen === run.id;
  const cancel = async () => {
    await api.post(`/api/tasks/${run.task_id}/plots/${run.id}/cancel`);
    qc.invalidateQueries({ queryKey: ["plotruns", run.task_id] });
  };
  return (
    <div className={cn("space-y-2 rounded-xl border p-3", failed ? "border-fail/40 bg-fail/5" : "border-line/80 bg-raised/20")} aria-label="Plot run">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <StatusIcon status={active ? "running" : run.status} className="h-4 w-4" />
        <span className={cn("text-[13px] font-medium", failed && "text-fail")}>
          {active ? "Making plots from the files on disk" : run.status === "ok" ? "Plots made" : run.status === "cancelled" ? "Plot run cancelled" : "Plot run failed"}
        </span>
        <span className="num text-xs text-faint">{run.id}</span>
        <span className="ml-auto flex items-center gap-1.5">
          {active && (
            <button className="btn btn-sm btn-danger" onClick={cancel}>
              Cancel
            </button>
          )}
          {!active && (
            <button className="btn btn-sm" aria-label="Dismiss" onClick={() => setDismissed((d) => new Set(d).add(run.id))}>
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </span>
      </div>
      <ul className="flex flex-wrap gap-1.5">
        {run.plots.map((p) => (
          <li key={p.id} title={p.detail ?? p.title} className={cn("flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11.5px]", p.status === "failed" ? "border-fail/40 text-fail" : "border-line/80 text-muted")}>
            <StatusIcon status={p.status} className="h-3 w-3" />
            {p.title}
            {p.elapsed_s != null && p.status !== "pending" && <span className="num text-faint">{duration(p.elapsed_s)}</span>}
          </li>
        ))}
      </ul>
      {run.plots.filter((p) => p.detail && p.status === "failed").map((p) => (
        <p key={p.id} className="text-xs text-fail">
          {p.title}: {p.detail}
        </p>
      ))}
      <details open={open} onToggle={(e) => setLogOpen((e.target as HTMLDetailsElement).open ? run.id : null)}>
        <summary className="cursor-pointer text-xs text-muted">Log</summary>
        <div className="mt-1.5">
          <RunLog run={run} open={open} />
        </div>
      </details>
    </div>
  );
}
