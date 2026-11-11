import { AlertCircle } from "lucide-react";
import { cn } from "../lib/cn";
import { duration } from "../lib/format";
import type { Stage } from "../lib/types";
import { Tip } from "./ui";

const SHORT: Record<string, string> = {
  magnetogram: "Mag",
  footpoints: "Foot",
  pfss: "PFSS",
  world: "World",
  relax: "Relax",
  world_plots: "Plots",
  fieldmap: "Map",
  wind: "Wind",
  wind_plots: "Plots",
  fluxlight: "Light",
};

const SOURCE_NOTE: Record<string, string> = {
  event: "from structured events",
  log: "inferred from the log",
  file: "inferred from files on disk",
  none: "no evidence yet",
};

export function stageTip(s: Stage) {
  return (
    <div className="space-y-0.5">
      <p className="font-medium">{s.label}</p>
      <p className="text-muted">
        {s.status === "pending" ? "Not started" : s.status[0].toUpperCase() + s.status.slice(1)}
        {s.elapsed_s != null && s.status !== "pending" && ` · ${duration(s.elapsed_s)}`}
      </p>
      {s.source !== "none" && <p className="text-muted">Source: {SOURCE_NOTE[s.source]}</p>}
      {s.detail && <p className="text-ink">{s.detail}</p>}
      {s.verified === false && <p className="text-run">Events say it finished, but its output file was not found.</p>}
      {s.verified === true && <p className="text-ok">Output file found on disk.</p>}
    </div>
  );
}

/** The ten pipeline stages as chips. Filled when the state comes from events, outlined when it was inferred. */
export function PipelineChips({ stages, className, selected, onSelect, fill, reused }: { stages: Stage[]; className?: string; selected?: string; onSelect?: (id: string) => void; fill?: Record<string, number>; reused?: Set<string> }) {
  // with onSelect the chips are radio buttons: one stage at a time, arrow keys move between them
  const move = (e: React.KeyboardEvent, i: number) => {
    const d = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!d || !onSelect) return;
    e.preventDefault();
    const j = (i + d + stages.length) % stages.length;
    onSelect(stages[j].id);
    (e.currentTarget.closest("[role=radiogroup]")?.querySelectorAll<HTMLElement>("[role=radio]")[j])?.focus();
  };
  return (
    <ol className={cn("grid grid-cols-5 gap-1.5 sm:grid-cols-10", className)} aria-label="Pipeline stages" role={onSelect ? "radiogroup" : undefined}>
      {stages.map((s, i) => {
        const inferred = s.source !== "event" && s.source !== "none";
        const tone =
          s.status === "running"
            ? "border-run/50 text-run " + (inferred ? "bg-transparent" : "bg-run/15") + " animate-pulse-ring"
            : s.status === "ok"
              ? "border-ok/40 text-ok " + (inferred ? "bg-transparent" : "bg-ok/12")
              : s.status === "failed"
                ? "border-fail/50 text-fail " + (inferred ? "bg-transparent" : "bg-fail/12")
                : s.status === "skipped"
                  ? "border-line text-skip/80 bg-transparent"
                  : "border-dashed border-line text-faint bg-transparent";
        return (
          <li key={s.id}>
            <Tip
              label={
                <>
                  {stageTip(s)}
                  {reused?.has(s.id) && <p className="mt-0.5 text-info">Reused from the cache, not recomputed.</p>}
                </>
              }
            >
              <div
                role={onSelect ? "radio" : undefined}
                aria-checked={onSelect ? s.id === selected : undefined}
                tabIndex={onSelect ? (s.id === selected ? 0 : -1) : undefined}
                onClick={onSelect && (() => onSelect(s.id))}
                onKeyDown={onSelect && ((e) => (e.key === " " || e.key === "Enter" ? (e.preventDefault(), onSelect(s.id)) : move(e, i)))}
                className={cn(
                  "relative flex h-[26px] items-center justify-center rounded-md border text-[10.5px] font-medium leading-none tracking-wide",
                  onSelect ? "cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-accent/70" : "cursor-default",
                  tone,
                  inferred && "border-solid",
                  s.status === "skipped" && "opacity-70",
                  s.id === selected && "ring-2 ring-accent shadow-[0_0_0_3px_rgb(var(--accent)/0.18)]",
                )}
              >
                {fill?.[s.id] != null && <span className="pointer-events-none absolute inset-y-0 left-0 rounded-[inherit] bg-run/30" style={{ width: `${fill[s.id] * 100}%` }} />}
                {reused?.has(s.id) && <span className="hatch pointer-events-none absolute inset-0 rounded-[inherit]" />}
                <span className="relative">{SHORT[s.id] ?? s.label}</span>
                {s.verified === false && <AlertCircle className="absolute -right-1 -top-1 h-3 w-3 fill-bg text-run" />}
              </div>
            </Tip>
          </li>
        );
      })}
    </ol>
  );
}
