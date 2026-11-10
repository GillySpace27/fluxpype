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
export function PipelineChips({ stages, className }: { stages: Stage[]; className?: string }) {
  return (
    <ol className={cn("grid grid-cols-5 gap-1.5 sm:grid-cols-10", className)} aria-label="Pipeline stages">
      {stages.map((s) => {
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
            <Tip label={stageTip(s)}>
              <div
                className={cn(
                  "relative flex h-[26px] cursor-default items-center justify-center rounded-md border text-[10.5px] font-medium leading-none tracking-wide",
                  tone,
                  inferred && "border-solid",
                  s.status === "skipped" && "opacity-70",
                )}
              >
                {SHORT[s.id] ?? s.label}
                {s.verified === false && <AlertCircle className="absolute -right-1 -top-1 h-3 w-3 fill-bg text-run" />}
              </div>
            </Tip>
          </li>
        );
      })}
    </ol>
  );
}
