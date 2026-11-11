import { Link } from "react-router-dom";
import { artifactUrl } from "../lib/api";
import { ago } from "../lib/format";
import { useNow, useOverview } from "../lib/hooks";
import type { TaskState } from "../lib/types";
import { StageStrip, toneOf } from "./StageStrip";
import { Card, StatusBadge } from "./ui";

/** Other recent runs of the same rotation and fluxon count: when, how far each got, and its headline image. Reads the overview's recent tasks, so only the latest twelve finished runs are considered. */
export function SameRotation({ task }: { task: TaskState }) {
  const { data } = useOverview();
  const now = useNow(60000);
  const others = (data?.recent ?? []).filter((r) => r.cr === task.cr && r.nflux === task.nflux && r.id !== task.id).slice(0, 6);
  if (others.length === 0) return null;
  return (
    <Card className="p-3">
      <details>
        <summary className="flex cursor-pointer items-center gap-2 text-[13px]">
          <span className="label">Other runs of CR {task.cr}</span>
          <span className="num text-xs text-muted">
            {others.length} with {task.nflux} fluxons
          </span>
        </summary>
        <div className="mt-2 divide-y divide-line/50">
          {others.map((r) => (
            <Link key={r.id} to={`/jobs/${r.job_id}/${r.id}`} className="flex items-center gap-3 rounded-lg px-1 py-2 hover:bg-raised/60">
              <div className="h-9 w-9 shrink-0 overflow-hidden rounded-md border border-line bg-raised">{r.thumb && <img src={artifactUrl(r.thumb)} alt="" loading="lazy" className="h-full w-full object-cover" />}</div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-[13px]">
                  {r.profile} <span className="text-muted">· {r.method} · attempt {r.attempt}</span>
                </p>
                <p className="num text-[11px] text-faint">
                  {ago(r.ended ?? r.created, now)} · {r.stages_ok}/10 stages{r.failure ? ` · ${r.failure}` : ""}
                </p>
              </div>
              <StageStrip items={r.stages.map((s) => ({ id: s.id, label: s.id, tone: toneOf(s.status) }))} labels={false} className="hidden w-36 !gap-0.5 sm:grid" />
              <StatusBadge status={r.status} />
            </Link>
          ))}
        </div>
      </details>
    </Card>
  );
}
