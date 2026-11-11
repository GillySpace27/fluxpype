import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CircleDashed, Copy, FileX2, FolderOpen, Play, Plus, Search, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, Navigate, NavLink, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { FileViewer, useViewerMode } from "../components/FileViewer";
import { OriginTag, ResultsPane } from "../components/ResultsPane";
import { RunForm, changedOverrides, emptyDraft, jobBody, pickedTasks, taskKey, type RunDraft } from "../components/RunForm";
import { Badge, Banner, Card, Checkbox, EmptyState, ErrorNote, Skeleton, StatusBadge, Switch, useConfirm, useToast } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { ago, bytes, dateTime, kindLabel } from "../lib/format";
import { jobDataBase, useJob, useJobConfig, useJobFiles, useJobs, useProfile, useProfiles, useStatus, useTask, useWide } from "../lib/hooks";
import { loadDraft, saveDraft } from "../lib/layoutMode";
import type { CacheRecord, Job, JobFile, TaskRow } from "../lib/types";
import { StatusDot, TaskView } from "./JobDetail";
import { JobActions } from "./Jobs";

const enc = encodeURIComponent;
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

const TABS = [
  { id: "settings", label: "Settings" },
  { id: "progress", label: "Run" },
  { id: "cache", label: "Cache" },
  { id: "results", label: "Results" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function runLabel(j: Job) {
  const crs = [...new Set(j.tasks.map((t) => t.cr))];
  const n = [...new Set(j.tasks.map((t) => t.nflux))];
  return `${j.profile} · CR ${crs.slice(0, 3).join(", ")}${crs.length > 3 ? ` +${crs.length - 3}` : ""} · ${n.join("/")} fluxons`;
}

/* ------------------------------------------------------------------ header: which run */
function RunPicker({ current }: { current?: string }) {
  const { data } = useJobs();
  const nav = useNavigate();
  const jobs = data?.jobs ?? [];
  return (
    <div className="flex flex-wrap items-center gap-2">
      <select
        className="input h-9 w-auto min-w-0 max-w-full py-0 text-[13px] sm:max-w-[460px]"
        aria-label="Choose a run"
        value={current ?? ""}
        onChange={(e) => e.target.value && nav(`/run/${e.target.value}/progress`)}
      >
        {!current && <option value="">Choose a run...</option>}
        {jobs.map((j) => (
          <option key={j.id} value={j.id}>
            {dateTime(j.created)} · {runLabel(j)} · {j.status}
          </option>
        ))}
      </select>
      <Link to="/run/new" className="btn btn-sm">
        <Plus className="h-3.5 w-3.5" /> New run
      </Link>
    </div>
  );
}

function TabBar({ base, tab, disabled, search }: { base: string; tab: TabId; disabled?: boolean; search?: string }) {
  return (
    <div className="flex gap-1 overflow-x-auto border-b hair" role="tablist" aria-label="Run">
      {TABS.map((t) => {
        const off = disabled && t.id !== "settings";
        const cls = cn(
          "-mb-px shrink-0 border-b-2 px-3.5 py-2 text-[13px] font-medium transition-colors",
          t.id === tab ? "border-accent text-ink" : "border-transparent text-muted hover:text-ink",
          off && "pointer-events-none opacity-40",
        );
        return off ? (
          <span key={t.id} className={cls} title="Available once the run is launched" aria-disabled>
            {t.label}
          </span>
        ) : (
          <NavLink key={t.id} role="tab" aria-selected={t.id === tab} to={`${base}/${t.id}${search ?? ""}`} replace className={cls}>
            {t.label}
          </NavLink>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ Settings tab: what this run ran with */
function RunSettings({ job }: { job: Job }) {
  const { data, error, refetch } = useJobConfig(job.id);
  const [q, setQ] = useState("");
  const [onlyChanged, setOnlyChanged] = useState(Object.keys(job.overrides ?? {}).length > 0);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!data) return <Skeleton className="h-80" />;
  const needle = q.trim().toLowerCase();
  const rows = data.entries.filter(
    (e) => (!onlyChanged || e.overridden || e.differs_from_preset || e.source === "profile") && (!needle || `${e.key} ${e.value} ${e.comment ?? ""}`.toLowerCase().includes(needle)),
  );
  const nOver = Object.keys(data.overrides).length;
  const nDrift = data.entries.filter((e) => e.differs_from_preset && !e.overridden).length;
  return (
    <div className="space-y-4">
      <Card className="grid gap-4 p-4 sm:grid-cols-[1fr_auto]">
        <div className="min-w-0 space-y-1.5 text-[13px]">
          <p>
            Ran preset{" "}
            {data.preset_exists ? (
              <Link to={`/presets/${enc(data.profile)}`} className="font-semibold hover:text-accent">
                {data.profile}
              </Link>
            ) : (
              <span className="font-semibold">{data.profile}</span>
            )}
            {nOver > 0 ? ` with ${nOver} change${nOver === 1 ? "" : "s"} for this run.` : " as saved."}
          </p>
          <p className="flex items-center gap-1.5 text-muted">
            <FolderOpen className="h-3.5 w-3.5 shrink-0" />
            <span className="truncate font-mono text-[12px]" title={data.data_dir}>
              {data.data_dir}
            </span>
            <span className="shrink-0">· batch {data.batch_name}</span>
          </p>
          {nDrift > 0 && <p className="text-[12px] text-faint">{nDrift} setting{nDrift === 1 ? " has" : "s have"} changed in the preset since this run. The values below are what the run used.</p>}
        </div>
        <div className="flex items-start">
          <Link to={`/run/new?from=${job.id}`} className="btn btn-sm">
            <Copy className="h-3.5 w-3.5" /> Duplicate as new run
          </Link>
        </div>
      </Card>
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative min-w-[200px] flex-1">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-faint" />
          <input className="input pl-8" placeholder={`Filter ${data.entries.length} settings`} value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        <span className="flex items-center gap-2 text-xs text-muted">
          <Switch checked={onlyChanged} onChange={setOnlyChanged} label="Only what this preset or run sets" />
          Only what this preset or run sets
        </span>
      </div>
      <Card className="overflow-hidden">
        <table className="w-full text-[12.5px]">
          <thead className="border-b hair bg-raised/40 text-left">
            <tr>
              {["Setting", "Value used", ""].map((h) => (
                <th key={h} className="label px-3 py-2 font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => (
              <tr key={e.key} className="border-b border-line/40 align-top last:border-0">
                <td className="whitespace-nowrap px-3 py-1.5 font-mono font-medium">{e.key}</td>
                <td className="px-3 py-1.5">
                  <span className="break-all font-mono">{e.value}</span>
                  {e.resolved && <span className="block break-all font-mono text-[11px] text-faint">= {e.resolved}</span>}
                  {e.comment && <span className="block text-[11px] text-faint">{e.comment}</span>}
                </td>
                <td className="whitespace-nowrap px-3 py-1.5 text-right">
                  {e.overridden && <Badge tone="accent">changed for this run</Badge>}
                  {!e.overridden && e.differs_from_preset && (
                    <span title={`The preset now has: ${e.preset_value ?? "(not set)"}`}>
                      <Badge tone="info">preset now differs</Badge>
                    </span>
                  )}
                  {e.set_by_plumber && <span className="text-[11px] text-faint" title="Plumber sets this in every run's saved settings">set by Plumber</span>}
                  {!e.overridden && !e.differs_from_preset && !e.set_by_plumber && e.source === "profile" && <span className="text-[11px] text-faint">from preset</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <p className="px-3 py-6 text-center text-xs text-muted">No settings match.</p>}
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ Cache tab: only this run's files */
function RunFiles({ job, task }: { job: Job; task?: TaskRow }) {
  const active = job.status === "running" || job.status === "pending";
  const { data, error, refetch } = useJobFiles(job.id, active);
  const qc = useQueryClient();
  const toast = useToast();
  const { ask, node } = useConfirm();
  const [onlyTask, setOnlyTask] = useState(job.tasks.length > 1);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [open, setOpen] = useState<number | null>(null);
  const [mode, setMode] = useViewerMode();
  const wide = useWide();
  const rows: JobFile[] = useMemo(() => (data?.files ?? []).filter((f) => !onlyTask || !task || f.tasks.includes(task.id)), [data, onlyTask, task]);
  const missing = (data?.missing ?? []).filter((m) => !onlyTask || !task || m.task_id === task.id);
  const close = useCallback(() => setOpen(null), []);
  const trash = useMutation({
    mutationFn: (paths: string[]) => api.post("/api/cache/trash", { paths }),
    onSuccess: () => {
      toast("ok", `Moved ${sel.size} file(s) to the trash`);
      setSel(new Set());
      setOpen(null);
      qc.invalidateQueries({ queryKey: ["job-files", job.id] });
      qc.invalidateQueries({ queryKey: ["results"] });
      qc.invalidateQueries({ queryKey: ["cache"] });
      qc.invalidateQueries({ queryKey: ["trash"] });
    },
    onError: (e) => toast("error", errText(e)),
  });
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!data) return <Skeleton className="h-80" />;
  if (!data.data_dir) return <EmptyState icon={<FileX2 className="h-5 w-5" />} title="No files to show" body="This job's saved settings could not be read." />;
  const canTrash = data.is_current_data_dir && !active;
  const total = rows.reduce((a, f) => a + f.size, 0);
  const inline = open != null && wide && mode === "inline";
  const viewer =
    open != null && rows[open] ? (
      <FileViewer
        records={rows as CacheRecord[]}
        index={open}
        onIndex={setOpen}
        onClose={close}
        dataDir={data.data_dir}
        base={jobDataBase(job.id)}
        mode={wide ? mode : "popup"}
        onMode={setMode}
        className="lg:sticky lg:top-20"
        extraMeta={(r) => <OriginTag origin={(r as JobFile).origin} />}
      />
    ) : null;
  const allSel = rows.length > 0 && rows.every((f) => sel.has(f.path));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3 text-[13px]">
        <span className="text-muted">
          <span className="num font-semibold text-ink">{rows.length}</span> files · <span className="num">{bytes(total)}</span> · in{" "}
          <span className="font-mono text-[12px]" title={data.data_dir}>
            {data.data_dir}
          </span>
        </span>
        {job.tasks.length > 1 && task && (
          <span className="flex items-center gap-2 text-xs text-muted">
            <Switch checked={onlyTask} onChange={setOnlyTask} label={`Only CR ${task.cr} · ${task.nflux}`} />
            Only CR {task.cr} · {task.nflux}
          </span>
        )}
        <div className="ml-auto flex items-center gap-2">
          {sel.size > 0 && (
            <button
              className="btn btn-sm btn-danger"
              disabled={!canTrash || trash.isPending}
              onClick={async () => {
                if (await ask({ title: `Move ${sel.size} file(s) to the trash?`, body: "They go to the data folder's trash and can be restored from the Cache page. Later runs will rebuild them.", confirm: "Move to trash", danger: true }))
                  trash.mutate([...sel]);
              }}
            >
              <Trash2 className="h-3.5 w-3.5" /> Trash {sel.size}
            </button>
          )}
          <Link to={`/cache?cr=${task?.cr ?? job.tasks[0]?.cr ?? ""}`} className="btn btn-sm">
            Full Cache page
          </Link>
        </div>
      </div>
      {!data.is_current_data_dir && (
        <Banner tone="info" title="This run used another data folder">
          Its files can be viewed here but not trashed, because the trash works on the current data folder only.
        </Banner>
      )}
      {active && data.is_current_data_dir && <p className="text-[12px] text-faint">Trash is off while the run is going.</p>}
      {missing.length > 0 && (
        <Card className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-4 py-2.5 text-[12.5px]">
          <span className="flex items-center gap-1.5 text-muted">
            <CircleDashed className="h-3.5 w-3.5" /> Not on disk:
          </span>
          {missing.map((m) => (
            <span key={`${m.task_id}-${m.stage}`} className="rounded-md border border-line px-1.5 py-px text-[11.5px]">
              {m.label}
              {job.tasks.length > 1 && !onlyTask && <span className="text-faint"> · CR {m.cr} · {m.nflux}</span>}
            </span>
          ))}
        </Card>
      )}
      <div className={cn(inline && "grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]")}>
        <Card className="overflow-x-auto">
          <table className={cn("w-full text-[13px]", !inline && "min-w-[680px]")}>
            <thead className="border-b hair bg-raised/40 text-left">
              <tr>
                <th className="w-9 px-3 py-2">
                  <Checkbox checked={allSel} label="Select all" onChange={() => setSel(allSel ? new Set() : new Set(rows.map((f) => f.path)))} />
                </th>
                {(inline ? ["File", "Origin"] : ["File", "Type", "Origin", "Size", "Modified"]).map((h) => (
                  <th key={h} className="label px-3 py-2 font-medium">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((f, i) => (
                <tr key={f.path} onClick={() => setOpen(i)} className={cn("cursor-pointer border-b border-line/40 last:border-0 hover:bg-raised/50", open === i && "bg-raised")}>
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    <Checkbox
                      checked={sel.has(f.path)}
                      label={`Select ${f.name}`}
                      onChange={() =>
                        setSel((s) => {
                          const n = new Set(s);
                          if (n.has(f.path)) n.delete(f.path);
                          else n.add(f.path);
                          return n;
                        })
                      }
                    />
                  </td>
                  <td className={cn("truncate px-3 py-2 font-mono text-xs", inline ? "max-w-[220px]" : "max-w-[360px]")} title={f.path}>
                    {f.name}
                  </td>
                  {!inline && <td className="px-3 py-2 text-muted">{kindLabel[f.kind] ?? f.kind}</td>}
                  <td className="px-3 py-2">
                    <OriginTag origin={f.origin} />
                  </td>
                  {!inline && <td className="num px-3 py-2 text-muted">{bytes(f.size)}</td>}
                  {!inline && <td className="px-3 py-2 text-muted">{ago(f.mtime)}</td>}
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length === 0 && <p className="px-3 py-6 text-center text-xs text-muted">No files yet.</p>}
        </Card>
        {inline && viewer}
      </div>
      {open != null && !inline && viewer}
      {node}
    </div>
  );
}

/* ------------------------------------------------------------------ Results tab */
function RunResults({ row }: { row: TaskRow }) {
  const active = row.status === "running" || row.status === "pending";
  const { data: t, error, refetch } = useTask(row.id, active);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!t) return <Skeleton className="h-96" />;
  return <ResultsPane task={t} />;
}

/* ------------------------------------------------------------------ one run */
export function RunPage() {
  const { id, tab: tabParam } = useParams();
  const [params] = useSearchParams();
  const nav = useNavigate();
  const { data: job, error, refetch } = useJob(id);
  const tab = (TABS.some((t) => t.id === tabParam) ? tabParam : "progress") as TabId;
  const rows = job?.tasks ?? [];
  const taskId = params.get("task");
  const selected = useMemo(
    () => rows.find((r) => r.id === taskId) ?? rows.find((r) => r.status === "running") ?? rows.find((r) => ["failed", "timeout", "crashed"].includes(r.status)) ?? rows[0],
    [rows, taskId],
  );
  useEffect(() => {
    if (error && (error as { status?: number }).status === 404) nav("/run", { replace: true });
  }, [error, nav]);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!job) return <Skeleton className="h-96" />;
  const search = selected && rows.length > 1 ? `?task=${selected.id}` : "";
  return (
    <div className="animate-fade-in space-y-5">
      <RunPicker current={job.id} />
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="flex flex-wrap items-center gap-3 text-[22px]">
            {job.profile} <StatusBadge status={job.status} />
          </h1>
          <p className="mt-1 text-[13px] text-muted">
            Started {dateTime(job.created)} · backend {job.backend} · {rows.length} task{rows.length === 1 ? "" : "s"}
            {job.extra.scenario && ` · scenario ${job.extra.scenario}`}
          </p>
        </div>
        <JobActions job={job} onDeleted={() => nav("/run")} />
      </div>
      {rows.length > 1 && (
        <div className="-mx-1 flex gap-2 overflow-x-auto px-1 pb-1" role="tablist" aria-label="Tasks">
          {rows.map((r) => (
            <Link
              key={r.id}
              role="tab"
              aria-selected={r.id === selected?.id}
              to={`/run/${job.id}/${tab}?task=${r.id}`}
              replace
              className={cn("flex shrink-0 items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors", r.id === selected?.id ? "border-accent/50 bg-raised text-ink" : "border-line text-muted hover:text-ink")}
            >
              <StatusDot status={r.status} />
              CR {r.cr} · {r.nflux}
            </Link>
          ))}
        </div>
      )}
      <TabBar base={`/run/${job.id}`} tab={tab} search={search} />
      <div className="pt-1">
        {tab === "settings" && <RunSettings job={job} />}
        {tab === "progress" && (selected ? <TaskView key={selected.id} row={selected} results={false} /> : <EmptyState title="This run has no tasks" />)}
        {tab === "cache" && <RunFiles job={job} task={selected} />}
        {tab === "results" && (selected ? <RunResults key={selected.id} row={selected} /> : <EmptyState title="This run has no tasks" />)}
      </div>
    </div>
  );
}

/** /run: open the most recent run, or start a new one. */
export function RunIndex() {
  const { data, error, refetch } = useJobs();
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!data) return <Skeleton className="h-96" />;
  const latest = data.jobs[0];
  return <Navigate to={latest ? `/run/${latest.id}/progress` : "/run/new"} replace />;
}

/* ------------------------------------------------------------------ a run before launch */
interface StoredDraft extends RunDraft {
  preset: string;
}

export function NewRun() {
  const [params, setParams] = useSearchParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const { data: status } = useStatus();
  const { data: presets } = useProfiles();
  const fromJob = params.get("from") ?? undefined;
  const { data: srcCfg } = useJobConfig(fromJob);
  const { data: srcJob } = useJob(fromJob);
  const [draft, setDraftState] = useState<StoredDraft | null>(() => {
    if (params.get("preset") || fromJob) return null;
    return loadDraft<StoredDraft>();
  });
  const setDraft = useCallback((d: StoredDraft | null) => {
    setDraftState(d);
    saveDraft(d);
  }, []);

  // ?preset=X starts a fresh draft on that preset
  useEffect(() => {
    const p = params.get("preset");
    if (p) {
      setDraft({ preset: p, ...emptyDraft() });
      setParams({}, { replace: true });
    }
  }, [params, setParams, setDraft]);
  // ?from=<job> copies that run's preset, changes and tasks
  useEffect(() => {
    if (!fromJob || !srcCfg || !srcJob) return;
    setDraft({
      preset: srcCfg.profile,
      ...emptyDraft(),
      overrides: { ...srcCfg.overrides },
      picked: srcJob.tasks.map(taskKey),
      scenario: srcJob.extra.scenario ?? "ok",
      speed: srcJob.extra.speed ?? "20",
    });
    setParams({}, { replace: true });
    toast("info", `Copied ${srcJob.profile} (${srcJob.id})`);
  }, [fromJob, srcCfg, srcJob, setParams, setDraft, toast]);

  const visible = (presets?.profiles ?? []).filter((p) => !p.hidden || p.name === draft?.preset);
  const { data: detail, error } = useProfile(draft?.preset);
  const sim = !!status?.sandboxed;

  // picked tasks the preset no longer has are dropped; if none remain, fall back to all
  const effective: RunDraft | null = useMemo(() => {
    if (!draft || !detail) return null;
    if (draft.picked === null) return draft;
    const have = new Set(detail.tasks.map(taskKey));
    const kept = draft.picked.filter((k) => have.has(k));
    return { ...draft, picked: kept.length ? kept : null };
  }, [draft, detail]);

  const launch = useMutation({
    mutationFn: () => api.post<Job>("/api/jobs", jobBody(detail!, effective!, sim)),
    onSuccess: (job) => {
      setDraft(null);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["overview"] });
      toast("ok", `Started ${job.tasks.length} task${job.tasks.length === 1 ? "" : "s"}`);
      nav(`/run/${job.id}/progress`);
    },
    onError: (e) => toast("error", errText(e)),
  });

  if (fromJob && !draft) return <Skeleton className="h-96" />;
  const nTasks = detail && effective ? pickedTasks(detail, effective).length : 0;
  const nChanged = detail && effective ? changedOverrides(detail, effective).length : 0;

  return (
    <div className="animate-fade-in space-y-5">
      <RunPicker />
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-3 text-[22px]">
            New run <Badge tone="accent">draft</Badge>
          </h1>
          <p className="mt-1 text-[13px] text-muted">Kept in this browser until you launch it or discard it.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {draft && (
            <button className="btn btn-sm" onClick={() => setDraft(null)}>
              <X className="h-3.5 w-3.5" /> Discard draft
            </button>
          )}
          <button className="btn btn-primary" disabled={!detail || !effective || nTasks === 0 || launch.isPending || (detail?.errors.length ?? 0) > 0} onClick={() => launch.mutate()}>
            <Play className="h-3.5 w-3.5" /> {launch.isPending ? "Starting..." : `Launch${nTasks ? ` ${nTasks} task${nTasks === 1 ? "" : "s"}` : ""}`}
          </button>
        </div>
      </div>
      <TabBar base="/run/new" tab="settings" disabled />
      <div className="space-y-5 pt-1">
        <Card className="flex flex-wrap items-center gap-3 p-4">
          <label className="flex min-w-0 flex-1 items-center gap-3 text-[13px]">
            <span className="label shrink-0">Preset</span>
            <select
              className="input h-9 max-w-sm py-0"
              value={draft?.preset ?? ""}
              onChange={(e) => setDraft(e.target.value ? { preset: e.target.value, ...emptyDraft() } : null)}
            >
              <option value="">Choose a preset...</option>
              {visible.map((p) => (
                <option key={p.name} value={p.name} disabled={p.errors.length > 0}>
                  {p.name} ({p.task_count} task{p.task_count === 1 ? "" : "s"}){p.errors.length ? " · has errors" : ""}
                </option>
              ))}
            </select>
          </label>
          {draft && (
            <span className="text-[12px] text-muted">
              {nChanged > 0 ? `${nChanged} change${nChanged === 1 ? "" : "s"} from the preset` : "Preset as saved"}
              {" · "}
              <Link to={`/presets/${enc(draft.preset)}`} className="hover:text-ink">
                open preset
              </Link>
            </span>
          )}
        </Card>
        {!draft && <EmptyState title="Pick a preset to start from" body="Its settings fill in below. Change what you need for this run, choose tasks, then launch." />}
        {draft && error && <ErrorNote error={error} />}
        {draft && !error && !detail && <Skeleton className="h-80" />}
        {detail && detail.errors.length > 0 && (
          <Banner tone="error" title="This preset can't run">
            {detail.errors.join("; ")}
          </Banner>
        )}
        {detail && effective && <RunForm detail={detail} draft={effective} onChange={(d) => setDraft({ ...d, preset: draft!.preset })} wide />}
      </div>
    </div>
  );
}
