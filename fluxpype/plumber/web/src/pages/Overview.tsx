import { Activity, ArrowRight, CheckCircle2, Clock, Layers, Rocket } from "lucide-react";
import { Link } from "react-router-dom";
import { PipelineChips } from "../components/PipelineChips";
import { TaskCard } from "../components/TaskCard";
import { Card, EmptyState, ErrorNote, PageHeader, SectionTitle, Skeleton, StatusBadge } from "../components/ui";
import { artifactUrl } from "../lib/api";
import { ago, dateTime, instOf } from "../lib/format";
import { useNow, useOverview } from "../lib/hooks";
import type { RecentTask } from "../lib/types";

function Stat({ label, value, sub, icon }: { label: string; value: React.ReactNode; sub?: string; icon: React.ReactNode }) {
  return (
    <Card className="flex items-center gap-3.5 p-4">
      <div className="grid h-10 w-10 place-items-center rounded-xl border border-line bg-raised text-accent">{icon}</div>
      <div className="min-w-0">
        <p className="label">{label}</p>
        <p className="num mt-0.5 text-xl font-semibold leading-none tracking-tight">{value}</p>
        {sub && <p className="mt-1 truncate text-xs text-muted">{sub}</p>}
      </div>
    </Card>
  );
}

function RecentRow({ t, now }: { t: RecentTask; now: number }) {
  return (
    <Link to={`/jobs/${t.job_id}/${t.id}`} className="flex items-center gap-3.5 rounded-xl px-2 py-2.5 transition-colors hover:bg-raised/60 focus:outline-none">
      <div className="h-12 w-12 shrink-0 overflow-hidden rounded-lg border border-line bg-raised">
        {t.thumb ? <img src={artifactUrl(t.thumb)} alt="" loading="lazy" className="h-full w-full object-cover" /> : <div className="grid h-full place-items-center text-faint"><Layers className="h-4 w-4" /></div>}
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium">
          CR {t.cr} <span className="font-normal text-muted">· {t.nflux} fluxons · {instOf(t.adapt)} · {t.method}</span>
        </p>
        <p className="mt-0.5 truncate text-xs text-muted">{t.failure ? t.failure : `${t.stages_ok}/10 stages · ${t.profile}`}</p>
      </div>
      <div className="hidden w-[190px] sm:block">
        <PipelineChips stages={t.stages.map((s) => ({ ...s, label: s.id, started: null, ended: null, elapsed_s: null, detail: null, verified: null }))} className="!grid-cols-10 gap-0.5 [&_div]:!h-1.5 [&_div]:!text-[0px]" />
      </div>
      <div className="w-[88px] text-right">
        <StatusBadge status={t.status} />
        <p className="mt-1 text-[11px] text-faint">{ago(t.ended ?? t.created, now)}</p>
      </div>
    </Link>
  );
}

export function Overview() {
  const { data, error, refetch } = useOverview();
  const now = useNow(5000);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!data)
    return (
      <div className="space-y-4">
        <Skeleton className="h-9 w-48" />
        <div className="grid gap-3 sm:grid-cols-3">{[0, 1, 2].map((i) => <Skeleton key={i} className="h-[74px]" />)}</div>
        <Skeleton className="h-44" />
      </div>
    );
  const { queue, running, pending, recent } = data;
  const okCount = recent.filter((r) => r.status === "ok").length;
  return (
    <div className="space-y-8 animate-fade-in">
      <PageHeader
        title="Overview"
        sub="What is running, what is waiting, and what just finished."
        right={
          <Link to="/presets" className="btn btn-primary">
            <Rocket className="h-4 w-4" /> New run
          </Link>
        }
      />

      <div className="grid gap-3 sm:grid-cols-3">
        <Stat label="Running" value={running.length} sub={`${queue.slots_used} of ${queue.cpu_budget} CPU slots in use`} icon={<Activity className="h-5 w-5" />} />
        <Stat label="Queued" value={pending.length} sub={queue.paused ? "Queue is paused" : `up to ${queue.max_parallel} at once`} icon={<Clock className="h-5 w-5" />} />
        <Stat label="Recent success" value={`${okCount}/${recent.length}`} sub="of the latest finished tasks" icon={<CheckCircle2 className="h-5 w-5" />} />
      </div>

      <section>
        <SectionTitle title="Running now" />
        {running.length === 0 ? (
          <EmptyState
            icon={<Rocket className="h-5 w-5" />}
            title="Nothing is running"
            body="Pick a preset, review the tasks it expands to, and start a run. Progress shows up here live."
            action={
              <Link to="/presets" className="btn btn-primary">
                Choose a preset <ArrowRight className="h-4 w-4" />
              </Link>
            }
          />
        ) : (
          <div className="grid gap-3 lg:grid-cols-2">{running.map((t) => <TaskCard key={t.id} t={t} />)}</div>
        )}
      </section>

      {pending.length > 0 && (
        <section>
          <SectionTitle title="Up next" hint={`${pending.length} waiting for a free slot`} />
          <Card className="divide-y divide-line/60">
            {pending.slice(0, 6).map((t) => (
              <Link key={t.id} to={`/jobs/${t.job_id}`} className="flex items-center justify-between gap-3 px-4 py-2.5 text-[13px] hover:bg-raised/50">
                <span>
                  CR {t.cr} <span className="text-muted">· {t.nflux} fluxons · {instOf(t.adapt)} · {t.method}</span>
                </span>
                <span className="num text-xs text-faint">{t.slots} slot{t.slots === 1 ? "" : "s"}</span>
              </Link>
            ))}
            {pending.length > 6 && <p className="px-4 py-2 text-xs text-muted">and {pending.length - 6} more</p>}
          </Card>
        </section>
      )}

      <section>
        <SectionTitle title="Recently finished" right={<Link to="/jobs" className="text-xs text-muted hover:text-ink">All jobs</Link>} />
        {recent.length === 0 ? (
          <EmptyState title="No finished tasks yet" body="Completed runs appear here with a thumbnail of the last plot they produced." />
        ) : (
          <Card className="p-1.5">
            {recent.slice(0, 8).map((t) => <RecentRow key={t.id} t={t} now={now} />)}
          </Card>
        )}
      </section>
      <p className="text-center text-[11px] text-faint">Last updated {dateTime(now)}</p>
    </div>
  );
}
