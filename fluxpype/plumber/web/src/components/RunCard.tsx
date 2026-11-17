import { ArrowUpRight, Layers } from "lucide-react";
import { Link } from "react-router-dom";
import { cn } from "../lib/cn";
import { ago, duration, instOf } from "../lib/format";
import type { Job } from "../lib/types";
import { Cover, coverUrl } from "./Cover";
import { OriginTag } from "./Notifications";
import { StatusBadge } from "./ui";

/** "CR 2150, 2151 · 1000 fluxons · HMI" for a job: each axis once, long lists shortened. */
export function jobSummary(job: Job): string {
  const uniq = (xs: string[]) => [...new Set(xs)];
  const short = (xs: string[], n = 3) => (xs.length > n ? `${xs.slice(0, n).join(", ")} +${xs.length - n}` : xs.join(", "));
  if (job.tasks.some((t) => t.adapt === "scene")) return short(uniq(job.tasks.map((t) => t.cr)), 2);
  const crs = uniq(job.tasks.map((t) => t.cr));
  const nf = uniq(job.tasks.map((t) => t.nflux));
  const src = uniq(job.tasks.map((t) => instOf(t.adapt)));
  return [crs.length ? `CR ${short(crs)}` : "", nf.length ? `${short(nf)} fluxons` : "", short(src, 2)].filter(Boolean).join(" · ");
}

/** A version that changes when the job's picture could have changed (a task finished or moved on). */
export const jobCoverVersion = (job: Job) => `${job.status}-${job.ended ?? ""}-${job.counts.ok ?? 0}-${job.counts.running ?? 0}`;

/** Where a click on a job should land: its failed task if any, else the running one, else the job page. */
export function jobHref(job: Job): string {
  const t =
    job.tasks.find((r) => ["failed", "timeout", "crashed"].includes(r.status)) ?? job.tasks.find((r) => r.status === "running");
  return t && job.tasks.length > 1 ? `/jobs/${job.id}/${t.id}` : `/jobs/${job.id}`;
}

export function Segments({ job, className }: { job: Job; className?: string }) {
  const total = Math.max(1, job.tasks.length);
  const seg = (n: number, cls: string) => (n ? <div className={cls} style={{ width: `${(n / total) * 100}%` }} /> : null);
  const failed = (job.counts.failed || 0) + (job.counts.timeout || 0) + (job.counts.crashed || 0);
  return (
    <div className={cn("flex h-1 w-full overflow-hidden rounded-full bg-raised", className)}>
      {seg(job.counts.ok || 0, "bg-ok")}
      {seg(job.counts.running || 0, "bg-run animate-pulse")}
      {seg(failed, "bg-fail")}
      {seg(job.counts.cancelled || 0, "bg-skip/60")}
    </div>
  );
}

/**
 * One run as a gallery card: its picture, status and what it covers. The whole card opens the run page;
 * `actions` sit above the link so their buttons work on their own.
 */
export function RunCard({ job, now, actions, className }: { job: Job; now: number; actions?: React.ReactNode; className?: string }) {
  const end = job.ended ?? now;
  const n = job.tasks.length;
  return (
    <div className={cn("card lift group relative flex flex-col overflow-hidden", className)}>
      <Cover src={coverUrl.job(job.id, jobCoverVersion(job))} seed={job.id} className="zoom aspect-[16/9] border-b border-line/60">
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-16 bg-gradient-to-t from-black/70 to-transparent" />
        <div className="absolute left-2.5 top-2.5">
          <StatusBadge status={job.status} className="bg-black/55 backdrop-blur-sm" />
        </div>
        {n > 1 && (
          <span className="absolute right-2.5 top-2.5 inline-flex items-center gap-1 rounded-full bg-black/55 px-2 py-0.5 text-[11px] font-medium text-white/90 backdrop-blur-sm">
            <Layers className="h-3 w-3" /> {n} tasks
          </span>
        )}
        <p className="absolute inset-x-3 bottom-2 truncate text-[12px] font-medium text-white/90 drop-shadow">{jobSummary(job)}</p>
      </Cover>
      <div className="flex flex-1 flex-col gap-2 p-3.5">
        <div className="flex items-start justify-between gap-2">
          <p className="flex min-w-0 items-center gap-2 text-[14px] font-semibold tracking-tight">
            <span className="truncate">{job.profile}</span>
            <OriginTag origin={job.origin} className="relative z-10" />
          </p>
          <ArrowUpRight className="h-4 w-4 shrink-0 text-faint transition-colors group-hover:text-accent" />
        </div>
        <div className="flex min-h-7 items-center justify-between gap-2">
          <p className="num truncate text-xs text-muted">
            {ago(job.created, now)} · {job.started ? duration(end - job.started) : "not started"}
            {Object.keys(job.overrides).length > 0 && ` · ${Object.keys(job.overrides).length} override${Object.keys(job.overrides).length === 1 ? "" : "s"}`}
          </p>
          {actions && <div className="relative z-10 shrink-0 opacity-70 transition-opacity focus-within:opacity-100 group-hover:opacity-100">{actions}</div>}
        </div>
        <Segments job={job} className="mt-auto" />
      </div>
      <Link to={jobHref(job)} className="absolute inset-0 rounded-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/60" aria-label={`Open run ${job.profile}`} />
    </div>
  );
}

/** A compact row for side lists (the sidebar's recent runs, the job page's neighbours). */
export function RunMini({ job, active, className }: { job: Job; active?: boolean; className?: string }) {
  const dot =
    job.status === "ok" ? "bg-ok" : job.status === "running" ? "bg-run animate-pulse" : ["failed", "partial"].includes(job.status) ? "bg-fail" : "bg-faint";
  return (
    <Link
      to={jobHref(job)}
      className={cn(
        "group flex items-center gap-2.5 rounded-lg px-1.5 py-1.5 transition-colors",
        active ? "bg-raised text-ink" : "text-muted hover:bg-raised/60 hover:text-ink",
        className,
      )}
    >
      <Cover src={coverUrl.job(job.id, jobCoverVersion(job))} seed={job.id} className="h-8 w-11 shrink-0 rounded-md border border-line/70" />
      <span className="min-w-0 flex-1 leading-tight">
        <span className="flex items-center gap-1.5 text-[12.5px] font-medium">
          <span className={cn("h-1.5 w-1.5 shrink-0 rounded-full", dot)} />
          <span className="truncate">{job.profile}</span>
        </span>
        <span className="mt-0.5 block truncate text-[10.5px] text-faint">{jobSummary(job)}</span>
      </span>
    </Link>
  );
}
