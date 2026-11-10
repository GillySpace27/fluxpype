import { AlertTriangle } from "lucide-react";
import { Link } from "react-router-dom";
import { cn } from "../lib/cn";
import { duration, instOf, num } from "../lib/format";
import { useNow } from "../lib/hooks";
import type { TaskState } from "../lib/types";
import { PipelineChips } from "./PipelineChips";
import { Sparkline } from "./Sparkline";
import { Badge, Card, StatusBadge, Tip } from "./ui";

/** Compact live card for one running task. */
export function TaskCard({ t }: { t: TaskState }) {
  const now = useNow();
  const elapsed = t.started ? (t.ended ?? now) - t.started : t.elapsed_s;
  const done = t.stages.filter((s) => s.status === "ok" || s.status === "skipped").length;
  const cur = t.stages.find((s) => s.id === t.current_stage);
  return (
    <Link to={`/jobs/${t.job_id}/${t.id}`} className="block focus:outline-none">
      <Card className="group p-4 transition-colors hover:border-faint/50">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            <p className="text-[15px] font-semibold tracking-tight">
              CR {t.cr} <span className="font-normal text-muted">· {t.nflux} fluxons</span>
            </p>
            <p className="mt-0.5 flex items-center gap-1.5 text-xs text-muted">
              <span>{t.profile}</span>
              <span className="text-faint">·</span>
              <span>{instOf(t.adapt)}</span>
              <span className="text-faint">·</span>
              <span>{t.method}</span>
            </p>
          </div>
          <div className="flex items-center gap-2">
            {t.degraded && (
              <Tip label={t.degraded_reason ?? "Structured events went quiet; progress is inferred from the log."}>
                <span>
                  <Badge tone="run">
                    <AlertTriangle className="h-3 w-3" /> inferred
                  </Badge>
                </span>
              </Tip>
            )}
            <StatusBadge status={t.status} />
          </div>
        </div>
        <PipelineChips stages={t.stages} className="mt-4" />
        <div className="mt-4 flex items-end justify-between gap-4">
          <div className="grid grid-cols-3 gap-x-6 gap-y-1 text-xs">
            <div>
              <p className="label">Elapsed</p>
              <p className="num mt-0.5 text-[13px] font-medium">{duration(elapsed)}</p>
            </div>
            <div>
              <p className="label">Stage</p>
              <p className="mt-0.5 text-[13px] font-medium">{cur ? cur.label : t.status === "ok" ? "Done" : "Starting"}</p>
            </div>
            <div>
              <p className="label">Stiffness</p>
              <p className="num mt-0.5 text-[13px] font-medium">{num(t.relax.latest?.stiffness)}</p>
            </div>
          </div>
          <div className={cn("hidden shrink-0 sm:block", !t.spark?.length && "opacity-0")}>
            <Sparkline values={t.spark ?? []} width={120} height={34} />
          </div>
        </div>
        <div className="mt-3 h-1 overflow-hidden rounded-full bg-raised">
          <div className="h-full rounded-full bg-gradient-to-r from-accent/70 to-accent transition-all duration-500" style={{ width: `${(done / Math.max(1, t.stages.length)) * 100}%` }} />
        </div>
      </Card>
    </Link>
  );
}
