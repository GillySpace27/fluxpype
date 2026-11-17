import { Activity, ArrowRight, CheckCircle2, Clock, Rocket } from "lucide-react";
import { Link } from "react-router-dom";
import { StageStrip, toneOf } from "../components/StageStrip";
import { ActivityList, AttentionPanel } from "../components/Notifications";
import { TaskCard } from "../components/TaskCard";
import { Card, EmptyState, ErrorNote, PageHeader, SectionTitle, Skeleton, StatusBadge } from "../components/ui";
import { ago, dateTime, duration, instOf } from "../lib/format";
import { cn } from "../lib/cn";
import { Cover, coverUrl } from "../components/Cover";
import { jobCoverVersion, jobHref, jobSummary, Segments } from "../components/RunCard";
import { useJobs, useNow, useOverview } from "../lib/hooks";
import type { RecentTask } from "../lib/types";

/** Why a queued task waits (Scheduler.queue_explain). */
export function waitLabel(reason: string | null | undefined): string {
  if (!reason) return "waiting";
  if (reason.startsWith("lock:")) return `waits for ${reason.slice(5)} (same output)`;
  return { paused: "queue paused", max_parallel: "all run slots busy", cpu_budget: "CPU budget full", next: "starts next" }[reason] ?? "waiting";
}

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

function RecentTile({ t, now }: { t: RecentTask; now: number }) {
  const scene = t.adapt === "scene";
  return (
    <Link to={`/jobs/${t.job_id}/${t.id}`} className="card lift group flex flex-col overflow-hidden focus:outline-none animate-rise">
      <Cover src={coverUrl.task(t.id, t.ended ?? t.status)} seed={t.id} className="zoom aspect-[16/10] border-b border-line/60">
        <div className="absolute left-2 top-2">
          <StatusBadge status={t.status} className="bg-black/55 backdrop-blur-sm" />
        </div>
      </Cover>
      <div className="flex flex-1 flex-col gap-1.5 p-3">
        <p className="truncate text-[13px] font-semibold">
          {scene ? t.cr : `CR ${t.cr}`} {!scene && <span className="font-normal text-muted">· {t.nflux} fluxons</span>}
        </p>
        <p className={cn("truncate text-[11.5px]", t.failure ? "text-fail" : "text-muted")}>{t.failure ? t.failure : `${scene ? "scene" : instOf(t.adapt)} · ${t.method} · ${t.profile}`}</p>
        <div className="mt-auto flex items-center gap-2 pt-1">
          <StageStrip items={t.stages.map((s) => ({ id: s.id, label: s.id, tone: toneOf(s.status) }))} labels={false} className="!gap-0.5 flex-1" />
          <span className="num shrink-0 text-[10.5px] text-faint">{ago(t.ended ?? t.created, now)}</span>
        </div>
      </div>
    </Link>
  );
}

/** The newest run as a large card at the top: the quickest way back to its page. */
function LatestRun({ now }: { now: number }) {
  const { data } = useJobs();
  const job = data?.jobs[0];
  if (!job) return null;
  const href = jobHref(job);
  const end = job.ended ?? now;
  const bad = ["failed", "partial", "timeout", "crashed"].includes(job.status);
  return (
    <section>
      <SectionTitle title="Latest run" right={<Link to="/jobs" className="text-xs text-muted hover:text-ink">All runs</Link>} />
      <Link to={href} className="card lift group grid overflow-hidden focus:outline-none md:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
        <Cover src={coverUrl.job(job.id, jobCoverVersion(job))} seed={job.id} className="zoom aspect-[16/9] md:aspect-auto md:min-h-[210px]" />
        <div className="flex flex-col gap-3 p-5">
          <div className="flex flex-wrap items-center gap-2.5">
            <h2 className="truncate text-[19px]">{job.profile}</h2>
            <StatusBadge status={job.status} />
          </div>
          <p className="text-[13px] text-muted">{jobSummary(job)}</p>
          <p className="num text-xs text-muted">
            Started {dateTime(job.created)} ({ago(job.created, now)}) · {job.started ? duration(end - job.started) : "not started"} · {job.tasks.length} task{job.tasks.length === 1 ? "" : "s"}
          </p>
          <Segments job={job} />
          <span className={cn("btn mt-auto self-start", bad ? "btn-danger" : "btn-primary")}>
            {bad ? "See what failed" : job.status === "running" ? "Watch it run" : "Open run"} <ArrowRight className="h-4 w-4" />
          </span>
        </div>
      </Link>
    </section>
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

      <AttentionPanel />

      {running.length === 0 && <LatestRun now={now} />}

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
              <Link key={t.id} to={`/jobs/${t.job_id}`} className="flex items-center justify-between gap-3 px-3 py-2 text-[13px] hover:bg-raised/50">
                <span className="flex min-w-0 items-center gap-2.5">
                  <Cover src={coverUrl.task(t.id)} seed={t.id} className="h-8 w-11 shrink-0 rounded-md border border-line/70" />
                  <span className="truncate">
                  CR {t.cr} <span className="text-muted">· {t.nflux} fluxons · {instOf(t.adapt)} · {t.method}</span>
                  </span>
                </span>
                <span className="num text-xs text-faint">
                  {waitLabel(t.wait_reason)} · {t.slots} slot{t.slots === 1 ? "" : "s"}
                </span>
              </Link>
            ))}
            {pending.length > 6 && <p className="px-4 py-2 text-xs text-muted">and {pending.length - 6} more</p>}
          </Card>
        </section>
      )}

      <section>
        <SectionTitle title="Recently finished" right={<Link to="/jobs" className="text-xs text-muted hover:text-ink">All runs</Link>} />
        {recent.length === 0 ? (
          <EmptyState title="No finished tasks yet" body="Completed runs appear here with a thumbnail of the last plot they produced." />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {recent.slice(0, 8).map((t) => <RecentTile key={t.id} t={t} now={now} />)}
          </div>
        )}
      </section>
      <section>
        <SectionTitle title="Activity" hint="The latest job events, kept whether or not a page was open." />
        <Card className="p-1">
          <ActivityList limit={10} />
        </Card>
      </section>
      <p className="text-center text-[11px] text-faint">Last updated {dateTime(now)}</p>
    </div>
  );
}
