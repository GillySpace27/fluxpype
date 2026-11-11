import type { CacheRecord, Job, JobFile, TaskRow } from "../lib/types";
import { Badge, Banner, Card, Checkbox, EmptyState, ErrorNote, Skeleton, Switch, useConfirm, useToast } from "./ui";
import { CircleDashed, Copy, FileX2, FolderOpen, Search, Trash2 } from "lucide-react";
import { FileViewer, useViewerMode } from "./FileViewer";
import { FileIcons, ViewSwitch, useFileView } from "./FileBrowser";
import { Link } from "react-router-dom";
import { OriginTag } from "./ResultsPane";
import { ago, bytes, kindLabel } from "../lib/format";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { jobDataBase, useJobConfig, useJobFiles, useWide } from "../lib/hooks";
import { useCallback, useMemo, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";

const enc = encodeURIComponent;
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/* ------------------------------------------------------------------ Settings tab: what this run ran with */
export function RunSettings({ job }: { job: Job }) {
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
          <Link to={`/jobs/new?from=${job.id}`} className="btn btn-sm">
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
export function RunFiles({ job, task }: { job: Job; task?: TaskRow }) {
  const active = job.status === "running" || job.status === "pending";
  const { data, error, refetch } = useJobFiles(job.id, active);
  const qc = useQueryClient();
  const toast = useToast();
  const { ask, node } = useConfirm();
  const [onlyTask, setOnlyTask] = useState(job.tasks.length > 1);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [open, setOpen] = useState<number | null>(null);
  const [mode, setMode] = useViewerMode();
  const [fmode, setFmode, fsize, setFsize] = useFileView("runFiles");
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
          {fmode === "icons" && rows.length > 0 && (
            <span className="flex items-center gap-1.5 text-xs text-muted">
              <Checkbox checked={allSel} label="Select all" onChange={() => setSel(allSel ? new Set() : new Set(rows.map((f) => f.path)))} /> All
            </span>
          )}
          <ViewSwitch mode={fmode} onMode={setFmode} size={fsize} onSize={setFsize} />
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
        {fmode === "icons" ? (
          <div>
            <FileIcons
              records={rows}
              base={jobDataBase(job.id)}
              size={fsize}
              active={open}
              onOpen={setOpen}
              selected={sel}
              onToggle={(f) => setSel((s) => { const n = new Set(s); n.has(f.path) ? n.delete(f.path) : n.add(f.path); return n; })}
              tag={(f) => <OriginTag origin={f.origin} />}
            />
            {rows.length === 0 && <p className="px-3 py-6 text-center text-xs text-muted">No files yet.</p>}
          </div>
        ) : (
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
        )}
        {inline && viewer}
      </div>
      {open != null && !inline && viewer}
      {node}
    </div>
  );
}
