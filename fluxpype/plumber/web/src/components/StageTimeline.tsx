import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { clock, duration } from "../lib/format";
import { useNow } from "../lib/hooks";
import type { LogChunk, Stage, TaskState } from "../lib/types";
import { StatusIcon, Tip } from "./ui";
import { AlertCircle } from "lucide-react";
import { stageTip } from "./PipelineChips";

const CALC = /Calculated \w+: (\d+) of (\d+), [\d.]+\(s\) elapsed, ([\d.]+)\(s\) \[[\d.]+ mins\] remaining\.*/g;
const ANSI = /\x1b\[[0-9;]*[A-Za-z]/g;
const NOISE = /^(Read '|\(in cleanup\))/; // per-world reads and Gnuplot teardown chatter, not progress

/** What the task is doing now, from the log tail. The pipeline rewrites its progress line in place with \r, so the
 * file holds every update on one line: split on \r too, and the last piece is the newest. */
export function useProgress(task: TaskState) {
  const running = task.status === "running";
  const { data } = useQuery({
    queryKey: ["progress", task.id, task.status],
    queryFn: () => api.get<LogChunk>(`/api/tasks/${task.id}/log?limit=131072`),
    refetchInterval: running ? 2000 : false,
  });
  const text = (data?.text ?? "").replace(ANSI, "");
  const all = Array.from(text.matchAll(CALC));
  const m = all[all.length - 1];
  const wind = m ? { line: m[0].trim(), done: +m[1], total: +m[2], left: +m[3], frac: Math.min(+m[1] / Math.max(+m[2], 1), 1) } : null;
  const tail = text.split(/[\r\n]+/).map((l) => l.trim()).filter((l) => l && !NOISE.test(l)).slice(-5);
  return { wind, tail };
}

/** One row per stage; bars are positioned by when each stage ran, relative to the task. With onSelect the rows are the stage radio buttons. */
export function StageTimeline({ task, selected, onSelect }: { task: TaskState; selected?: string; onSelect?: (id: string) => void }) {
  const running = task.status === "running";
  const now = useNow(running ? 1000 : 60000);
  const { wind, tail } = useProgress(task);
  const cur = running ? task.stages.find((s) => s.status === "running") : undefined;
  const timed = task.stages.filter((s) => s.started);
  const t0 = Math.min(task.started ?? Infinity, ...timed.map((s) => s.started as number));
  const end = task.ended ?? Math.max(now, ...timed.map((s) => (s.ended ?? s.started ?? 0) as number));
  const total = Math.max(end - (isFinite(t0) ? t0 : end), 1);

  // arrow keys move between rows, as in the chip radiogroup
  const move = (e: React.KeyboardEvent, i: number) => {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d || !onSelect) return;
    e.preventDefault();
    const j = (i + d + task.stages.length) % task.stages.length;
    onSelect(task.stages[j].id);
    e.currentTarget.closest("[role=radiogroup]")?.querySelectorAll<HTMLElement>("[role=radio]")[j]?.focus();
  };
  return (
    <div className="space-y-1">
      <div role={onSelect ? "radiogroup" : undefined} aria-label="Pipeline stages" className="space-y-0.5">
        {task.stages.map((s, i) => (
          <Row key={s.id} s={s} t0={isFinite(t0) ? t0 : 0} total={total} now={now} sel={s.id === selected} onSelect={onSelect && (() => onSelect(s.id))} onKey={(e) => move(e, i)} />
        ))}
      </div>
      <div className="flex justify-between pl-[8.4rem] pr-[3.9rem] pt-1 text-[10.5px] text-faint num">
        <span>{isFinite(t0) ? clock(t0) : ""}</span>
        <span>{duration(total)}</span>
      </div>
      {(wind || (cur && tail.length > 0)) && (
        <div className="mt-2 space-y-2 border-t border-line/60 px-1.5 pt-3">
          {cur && (
            <p className="flex flex-wrap items-baseline gap-x-2 text-[13px]">
              <span className="label">Now running</span>
              <span className="font-medium">{cur.label}</span>
              <span className="num text-xs text-muted">{cur.started ? duration(now - cur.started) : ""}</span>
            </p>
          )}
          {wind && (
            <>
              <p className="num flex flex-wrap items-baseline gap-x-3 text-xs text-muted">
                <span className="text-[15px] font-semibold text-ink">{Math.round(wind.frac * 100)}%</span>
                <span>{wind.done} of {wind.total} open fluxons</span>
                {running && wind.frac < 1 && <span>done about {clock(now + wind.left)}</span>}
              </p>
              <div className="h-1.5 rounded bg-raised/70">
                <div className={cn("h-full rounded bg-gradient-to-r from-run/70 to-run", running && wind.frac < 1 && "animate-pulse")} style={{ width: `${wind.frac * 100}%` }} />
              </div>
              <p className="num break-words font-mono text-[12px] text-ink/85">{wind.line}</p>
            </>
          )}
          {cur && tail.length > 0 && (
            <pre className="max-h-28 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-bg/50 px-2.5 py-2 font-mono text-[11px] leading-relaxed text-muted">{tail.join("\n")}</pre>
          )}
        </div>
      )}
    </div>
  );
}

function Row({ s, t0, total, now, sel, onSelect, onKey }: { s: Stage; t0: number; total: number; now: number; sel: boolean; onSelect?: () => void; onKey: (e: React.KeyboardEvent) => void }) {
  const has = s.started != null;
  const stop = s.ended ?? (s.status === "running" ? now : s.started ?? 0);
  const left = has ? ((s.started! - t0) / total) * 100 : 0;
  const width = has ? Math.max(((stop - s.started!) / total) * 100, 0.8) : 0;
  const barTone =
    s.status === "running"
      ? "bg-gradient-to-r from-run/70 to-run"
      : s.status === "ok"
        ? "bg-gradient-to-r from-ok/55 to-ok/90"
        : s.status === "failed"
          ? "bg-gradient-to-r from-fail/60 to-fail"
          : "bg-skip/50";
  const elapsed = s.elapsed_s ?? (has && s.status === "running" ? now - s.started! : null);
  return (
    <Tip label={stageTip(s)} side="top">
      <div
        role={onSelect ? "radio" : undefined}
        aria-checked={onSelect ? sel : undefined}
        tabIndex={onSelect ? (sel ? 0 : -1) : undefined}
        onClick={onSelect}
        onKeyDown={onSelect && ((e) => (e.key === " " || e.key === "Enter" ? (e.preventDefault(), onSelect()) : onKey(e)))}
        className={cn(
          "group grid grid-cols-[7.5rem_1fr_3rem] items-center gap-2 rounded-md px-1.5 py-0.5 hover:bg-raised/60",
          onSelect && "cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-accent/70",
          sel && "bg-accent/10 ring-1 ring-accent/50",
        )}
      >
        <div className="flex min-w-0 items-center gap-2">
          <StatusIcon status={s.status} className="h-3.5 w-3.5 shrink-0" />
          <span className={cn("truncate text-[12.5px]", s.status === "pending" ? "text-faint" : "text-ink")}>{s.label}</span>
          {s.verified === false && <AlertCircle className="h-3.5 w-3.5 shrink-0 text-run" />}
        </div>
        <div className="relative h-3 rounded bg-raised/70">
          {has && s.status !== "skipped" && (
            <div
              className={cn("absolute top-0 h-full rounded", barTone, s.status === "running" && "animate-pulse")}
              style={{ left: `${Math.min(left, 99.2)}%`, width: `${Math.min(width, 100 - left)}%` }}
            />
          )}
          {s.status === "skipped" && <div className="absolute inset-x-0 top-1/2 h-px bg-skip/40" />}
        </div>
        <span className="num text-right text-xs text-muted">{s.status === "skipped" ? "skip" : elapsed != null ? duration(elapsed) : ""}</span>
      </div>
    </Tip>
  );
}
