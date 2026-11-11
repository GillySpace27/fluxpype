import * as RMenu from "@radix-ui/react-dropdown-menu";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  ChevronRight,
  Copy,
  Database,
  Download,
  Eye,
  EyeOff,
  Layers,
  MoreHorizontal,
  Pencil,
  Play,
  Plus,
  RotateCcw,
  Save,
  Search,
  SlidersHorizontal,
  TextCursorInput,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { StageStrip, type StripItem } from "../components/StageStrip";
import { Badge, Banner, Card, Dialog, EmptyState, ErrorNote, PageHeader, Skeleton, Tip, useConfirm, useToast } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { instOf } from "../lib/format";
import { fuzzyMatch, type Field, type Match } from "../lib/fuzzy";
import { DataFolderBanner } from "../components/DataFolder";
import { RunForm, changedOverrides, emptyDraft, jobBody, pickedTasks, type RunDraft } from "../components/RunForm";
import { useProfile, useProfiles, useStatus } from "../lib/hooks";
import type { Job, PresetCache, ProfileDetail, ProfileSummary } from "../lib/types";

const CACHE_KINDS: [keyof PresetCache["counts"], string, string][] = [
  ["magnetogram", "Mag", "Magnetograms"],
  ["footpoints", "Foot", "Footpoints"],
  ["pfss", "PFSS", "PFSS solutions"],
  ["world", "World", "Initial worlds"],
  ["relaxed_world", "Relax", "Relaxed worlds"],
];


const enc = encodeURIComponent;
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

function useInvalidatePresets() {
  const qc = useQueryClient();
  return () => {
    qc.invalidateQueries({ queryKey: ["profiles"] });
    qc.invalidateQueries({ queryKey: ["profile"] });
  };
}

/* ------------------------------------------------------------------ cache status along a tile */
export function CacheBar({ cache, className }: { cache: PresetCache | null; className?: string }) {
  if (!cache || cache.tasks === 0) return null;
  const items: StripItem[] = CACHE_KINDS.map(([k, short, long]) => {
    const n = cache.counts[k];
    const frac = n / cache.tasks;
    return {
      id: k,
      label: short,
      tone: frac >= 1 ? "ok" : frac > 0 ? "run" : "idle",
      fill: Math.max(frac, n > 0 ? 0.12 : 0),
      tip: `${long}: ${n} of ${cache.tasks} task${cache.tasks === 1 ? "" : "s"} already on disk${cache.of > cache.tasks ? ` (first ${cache.tasks} of ${cache.of} checked)` : ""}`,
    };
  });
  return <StageStrip items={items} className={className} />;
}

/* ------------------------------------------------------------------ small dialogs */
function NameDialog({
  open,
  onOpenChange,
  title,
  description,
  initial,
  confirm,
  onSubmit,
  children,
}: {
  open: boolean;
  onOpenChange: (o: boolean) => void;
  title: string;
  description?: string;
  initial?: string;
  confirm: string;
  onSubmit: (name: string) => Promise<unknown>;
  children?: React.ReactNode;
}) {
  const [name, setName] = useState(initial ?? "");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    setErr(null);
    try {
      await onSubmit(name.trim());
      onOpenChange(false);
    } catch (e) {
      setErr(errText(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={title}
      description={description}
      footer={
        <>
          <button className="btn" onClick={() => onOpenChange(false)}>
            Cancel
          </button>
          <button className="btn btn-primary" disabled={!name.trim() || busy} onClick={submit}>
            {confirm}
          </button>
        </>
      }
    >
      <div className="space-y-3">
        <label className="block text-xs">
          <span className="label">Name</span>
          <input className="input mt-1 font-mono" value={name} autoFocus onChange={(e) => setName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && name.trim() && submit()} placeholder="my_preset" spellCheck={false} />
          <span className="mt-1 block text-[11px] text-faint">Letters, digits, _ - and . ; starts with a letter. This is the name runs use as FLUXPYPE_CONFIG.</span>
        </label>
        {children}
        {err && <p className="text-xs text-fail">{err}</p>}
      </div>
    </Dialog>
  );
}

function NewPresetDialog({ open, onOpenChange, presets, from }: { open: boolean; onOpenChange: (o: boolean) => void; presets: ProfileSummary[]; from?: string }) {
  const [base, setBase] = useState(from ?? "");
  const nav = useNavigate();
  const inval = useInvalidatePresets();
  const toast = useToast();
  return (
    <NameDialog
      open={open}
      onOpenChange={onOpenChange}
      title={from ? `Duplicate ${from}` : "New preset"}
      description="Your presets are saved by Plumber next to config.ini, which is never changed. They inherit everything in [DEFAULT]."
      initial={from ? `${from}_copy` : ""}
      confirm="Create"
      onSubmit={async (name) => {
        const r = await api.post<{ name: string }>("/api/profiles", { name, base: base || undefined });
        inval();
        toast("ok", `Created ${r.name}`);
        nav(`/presets/${enc(r.name)}`, { state: { edit: true } });
      }}
    >
      {!from && (
        <label className="block text-xs">
          <span className="label">Start from</span>
          <select className="input mt-1" value={base} onChange={(e) => setBase(e.target.value)}>
            <option value="">Blank: only the [DEFAULT] settings</option>
            {presets.map((p) => (
              <option key={p.name} value={p.name}>
                Copy of {p.name}
              </option>
            ))}
          </select>
        </label>
      )}
    </NameDialog>
  );
}

function DeletedDialog({ open, onOpenChange, names }: { open: boolean; onOpenChange: (o: boolean) => void; names: string[] }) {
  const inval = useInvalidatePresets();
  const toast = useToast();
  const restore = useMutation({
    mutationFn: (n: string) => api.post<{ name: string }>(`/api/profiles-deleted/${enc(n)}/restore`),
    onSuccess: (r) => {
      inval();
      toast("ok", `Restored ${r.name}`);
    },
    onError: (e) => toast("error", errText(e)),
  });
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title="Deleted presets" description="Presets you deleted are kept here until you restore them.">
      {names.length === 0 ? (
        <p className="py-6 text-center text-[13px] text-muted">Nothing deleted.</p>
      ) : (
        <div className="divide-y divide-line/50 rounded-lg border border-line">
          {names.map((n) => (
            <div key={n} className="flex items-center gap-3 px-3 py-2 text-[13px]">
              <span className="flex-1 font-mono text-xs">{n}</span>
              <button className="btn btn-sm" onClick={() => restore.mutate(n)}>
                <RotateCcw className="h-3.5 w-3.5" /> Restore
              </button>
            </div>
          ))}
        </div>
      )}
    </Dialog>
  );
}

/* ------------------------------------------------------------------ edit a user preset */
function EditDrawer({ detail, open, onOpenChange }: { detail: ProfileDetail; open: boolean; onOpenChange: (o: boolean) => void }) {
  const own = detail.entries.filter((e) => e.source === "profile");
  const [values, setValues] = useState<Record<string, string>>(() => Object.fromEntries(own.map((e) => [e.key, e.value])));
  const [removed, setRemoved] = useState<Set<string>>(new Set());
  const [adding, setAdding] = useState("");
  const inval = useInvalidatePresets();
  const toast = useToast();
  const addable = detail.entries.filter((e) => e.editable && !(e.key in values)).map((e) => e.key).sort();
  const changed = Object.entries(values).filter(([k, v]) => own.find((e) => e.key === k)?.value !== v);
  const dirty = changed.length > 0 || removed.size > 0;
  const save = useMutation({
    mutationFn: () => api.patch(`/api/profiles/${enc(detail.name)}`, { values: Object.fromEntries(changed), remove: [...removed] }),
    onSuccess: () => {
      inval();
      toast("ok", `Saved ${detail.name}`);
      onOpenChange(false);
    },
    onError: (e) => toast("error", errText(e)),
  });
  const byKey = new Map(detail.entries.map((e) => [e.key, e]));
  return (
    <Dialog
      side
      open={open}
      onOpenChange={onOpenChange}
      title={`Edit ${detail.name}`}
      description="Settings here override [DEFAULT]. Remove one to inherit the default again."
      footer={
        <>
          <button className="btn" onClick={() => onOpenChange(false)}>
            Cancel
          </button>
          <button className="btn btn-primary" disabled={!dirty || save.isPending} onClick={() => save.mutate()}>
            <Save className="h-3.5 w-3.5" /> {save.isPending ? "Saving..." : "Save preset"}
          </button>
        </>
      }
    >
      <div className="space-y-3">
        {Object.keys(values).length === 0 && <p className="text-[13px] text-muted">This preset uses only [DEFAULT]. Add a setting below to change it for this preset.</p>}
        {Object.entries(values).map(([k, v]) => {
          const def = byKey.get(k);
          return (
            <div key={k} className={cn(removed.has(k) && "opacity-40")}>
              <span className="flex items-baseline justify-between text-xs">
                <span className="font-mono text-ink/90">{k}</span>
                <button
                  className="text-[11px] text-muted hover:text-fail"
                  onClick={() => setRemoved((r) => (r.has(k) ? new Set([...r].filter((x) => x !== k)) : new Set(r).add(k)))}
                >
                  {removed.has(k) ? "keep" : "remove"}
                </button>
              </span>
              <input className="input mt-1 font-mono" disabled={removed.has(k)} value={v} onChange={(e) => setValues({ ...values, [k]: e.target.value })} spellCheck={false} />
              {def?.comment && <span className="mt-1 block text-[11px] leading-snug text-faint">{def.comment}</span>}
            </div>
          );
        })}
        <div className="flex gap-2 border-t hair pt-3">
          <select className="input flex-1" value={adding} onChange={(e) => setAdding(e.target.value)}>
            <option value="">Add a setting...</option>
            {addable.map((k) => (
              <option key={k} value={k}>
                {k}
              </option>
            ))}
          </select>
          <button
            className="btn"
            disabled={!adding}
            onClick={() => {
              setValues({ ...values, [adding]: byKey.get(adding)?.value ?? "" });
              setAdding("");
            }}
          >
            <Plus className="h-3.5 w-3.5" /> Add
          </button>
        </div>
      </div>
    </Dialog>
  );
}

/* ------------------------------------------------------------------ run drawer */
function RunDrawer({ detail, open, onOpenChange }: { detail: ProfileDetail; open: boolean; onOpenChange: (o: boolean) => void }) {
  const { data: status } = useStatus();
  const nav = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const inval = useInvalidatePresets();
  const [draft, setDraft] = useState<RunDraft>(emptyDraft);
  const [saveAs, setSaveAs] = useState(false);
  const sim = !!status?.sandboxed;
  const changed = changedOverrides(detail, draft);
  const nTasks = pickedTasks(detail, draft).length;

  const run = useMutation({
    mutationFn: () => api.post<Job>("/api/jobs", jobBody(detail, draft, sim)),
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["overview"] });
      toast("ok", `Started ${job.tasks.length} task${job.tasks.length === 1 ? "" : "s"}`);
      onOpenChange(false);
      nav(`/jobs/${job.id}`);
    },
    onError: (e) => toast("error", errText(e)),
  });

  return (
    <Dialog
      side
      open={open}
      onOpenChange={onOpenChange}
      title={`Run ${detail.name}`}
      description="Changes here apply to this run only, unless you save them as a new preset."
      footer={
        <>
          <span className="mr-auto text-xs text-muted">
            {nTasks} task{nTasks === 1 ? "" : "s"}
            {changed.length > 0 && ` · ${changed.length} change${changed.length === 1 ? "" : "s"}`}
          </span>
          <Link to={`/jobs/new?preset=${enc(detail.name)}`} className="btn" onClick={() => onOpenChange(false)}>
            Open as full page
          </Link>
          {changed.length > 0 && (
            <button className="btn" onClick={() => setSaveAs(true)}>
              <Save className="h-3.5 w-3.5" /> Save as preset
            </button>
          )}
          <button className="btn btn-primary" disabled={nTasks === 0 || run.isPending} onClick={() => run.mutate()}>
            <Play className="h-3.5 w-3.5" /> {run.isPending ? "Starting..." : "Start run"}
          </button>
        </>
      }
    >
      <RunForm detail={detail} draft={draft} onChange={setDraft} />
      {saveAs && (
        <NameDialog
          open={saveAs}
          onOpenChange={setSaveAs}
          title="Save as a new preset"
          description={`A copy of ${detail.name} with your ${changed.length} change${changed.length === 1 ? "" : "s"}.`}
          initial={`${detail.name}_copy`}
          confirm="Save preset"
          onSubmit={async (name) => {
            const r = await api.post<{ name: string }>("/api/profiles", { name, base: detail.name, values: Object.fromEntries(changed) });
            inval();
            toast("ok", `Saved ${r.name}`);
          }}
        />
      )}
    </Dialog>
  );
}

/* ------------------------------------------------------------------ preset detail */
function ActionsMenu({ detail, onEdit, onDuplicate, onRename }: { detail: ProfileDetail; onEdit: () => void; onDuplicate: () => void; onRename: () => void }) {
  const inval = useInvalidatePresets();
  const toast = useToast();
  const nav = useNavigate();
  const { ask, node } = useConfirm();
  const mine = detail.origin === "user";
  const hide = useMutation({
    mutationFn: (hidden: boolean) => api.post(`/api/profiles/${enc(detail.name)}/hidden`, { hidden }),
    onSuccess: (_r, hidden) => {
      inval();
      toast("ok", hidden ? `Hid ${detail.name}; "Show hidden" brings it back` : `${detail.name} is visible again`);
    },
    onError: (e) => toast("error", errText(e)),
  });
  const del = useMutation({
    mutationFn: () => api.del(`/api/profiles/${enc(detail.name)}`),
    onSuccess: () => {
      inval();
      toast("ok", `Deleted ${detail.name}; restore it from "Deleted"`);
      nav("/presets");
    },
    onError: (e) => toast("error", errText(e)),
  });
  const item = "flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] outline-none data-[highlighted]:bg-raised data-[disabled]:cursor-default data-[disabled]:opacity-40";
  return (
    <>
      <RMenu.Root>
        <RMenu.Trigger asChild>
          <button className="btn btn-icon" aria-label="More actions">
            <MoreHorizontal className="h-4 w-4" />
          </button>
        </RMenu.Trigger>
        <RMenu.Portal>
          <RMenu.Content align="end" sideOffset={6} className="z-50 w-56 animate-fade-in rounded-xl border border-line bg-surface p-1.5 shadow-pop">
            <RMenu.Item className={item} disabled={!mine} onSelect={onEdit}>
              <Pencil className="h-4 w-4 text-muted" /> Edit settings
            </RMenu.Item>
            <RMenu.Item className={item} onSelect={onDuplicate}>
              <Copy className="h-4 w-4 text-muted" /> Duplicate
            </RMenu.Item>
            <RMenu.Item className={item} disabled={!mine} onSelect={onRename}>
              <TextCursorInput className="h-4 w-4 text-muted" /> Rename
            </RMenu.Item>
            <RMenu.Item className={item} onSelect={() => (window.location.href = `/api/profiles-export?names=${enc(detail.name)}`)}>
              <Download className="h-4 w-4 text-muted" /> Export as .ini
            </RMenu.Item>
            <RMenu.Item className={item} onSelect={() => hide.mutate(!detail.hidden)}>
              {detail.hidden ? <Eye className="h-4 w-4 text-muted" /> : <EyeOff className="h-4 w-4 text-muted" />} {detail.hidden ? "Show in the list" : "Hide from the list"}
            </RMenu.Item>
            <RMenu.Separator className="my-1 h-px bg-line" />
            <RMenu.Item
              className={cn(item, "text-fail")}
              disabled={!mine}
              onSelect={async () => {
                if (await ask({ title: `Delete ${detail.name}?`, body: "It moves to Deleted presets, where you can restore it.", confirm: "Delete", danger: true })) del.mutate();
              }}
            >
              <Trash2 className="h-4 w-4" /> Delete
            </RMenu.Item>
            {!mine && <p className="px-2.5 pb-1 pt-1.5 text-[11px] leading-snug text-faint">Built-in presets come from config.ini. Duplicate one to edit, rename or delete it.</p>}
          </RMenu.Content>
        </RMenu.Portal>
      </RMenu.Root>
      {node}
    </>
  );
}

function PresetView({ name, presets }: { name: string; presets: ProfileSummary[] }) {
  const { data, error, refetch } = useProfile(name);
  const loc = useLocation();
  const nav = useNavigate();
  const inval = useInvalidatePresets();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [edit, setEdit] = useState(false);
  const [dup, setDup] = useState(false);
  const [rename, setRename] = useState(false);
  const [showDefault, setShowDefault] = useState(false);
  const summary = presets.find((p) => p.name === name);
  useEffect(() => {
    if ((loc.state as { edit?: boolean } | null)?.edit && data?.origin === "user") {
      setEdit(true);
      nav(loc.pathname, { replace: true, state: null });
    }
  }, [loc, data, nav]);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!data) return <Skeleton className="h-96" />;
  const own = data.entries.filter((e) => e.source === "profile");
  const shown = showDefault ? data.entries : own;
  const cacheLink = `/cache?preset=${enc(data.name)}`;
  return (
    <div className="animate-fade-in space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <Link to="/presets" className="mb-2 inline-flex items-center gap-1 text-xs text-muted hover:text-ink md:hidden">
            <ArrowLeft className="h-3.5 w-3.5" /> Presets
          </Link>
          <h1 className="flex flex-wrap items-center gap-2 text-[22px]">
            {data.name}
            {data.origin === "user" ? <Badge tone="accent">yours</Badge> : <Badge>built-in</Badge>}
            {data.hidden && <Badge>hidden</Badge>}
          </h1>
          <p className="mt-1 text-[13px] text-muted">
            {data.task_count} task{data.task_count === 1 ? "" : "s"} · {data.slots} CPU slot{data.slots === 1 ? "" : "s"} each · batch <span className="font-mono text-xs">{data.batch_name || "?"}</span>
          </p>
        </div>
        <div className="flex items-center gap-2">
          {data.origin === "user" && (
            <button className="btn" onClick={() => setEdit(true)}>
              <Pencil className="h-4 w-4" /> Edit
            </button>
          )}
          <ActionsMenu detail={data} onEdit={() => setEdit(true)} onDuplicate={() => setDup(true)} onRename={() => setRename(true)} />
          <button className="btn btn-primary" onClick={() => setOpen(true)} disabled={data.errors.length > 0 || data.task_count === 0}>
            <SlidersHorizontal className="h-4 w-4" /> Configure and run
          </button>
        </div>
      </div>

      {data.errors.map((e) => (
        <Banner key={e} tone="error" title="This preset cannot run as written">
          {e}. {data.origin === "user" ? "Edit it to fix this." : "Duplicate it and fix the copy (config.ini stays untouched)."}
        </Banner>
      ))}
      {data.warnings.map((e) => (
        <Banner key={e} tone="warn">
          {e}
        </Banner>
      ))}

      {summary?.cache && (
        <Card className="flex flex-wrap items-center gap-4 p-4">
          <div className="min-w-0 flex-1">
            <p className="flex items-center gap-2 text-[13px] font-medium">
              <Database className="h-4 w-4 text-muted" /> Already cached
              {data.sandboxed && <Badge tone="accent">sandbox</Badge>}
            </p>
            <p className="mt-0.5 truncate font-mono text-[11px] text-faint" title={data.data_dir}>
              {data.data_dir}
            </p>
          </div>
          <CacheBar cache={summary.cache} className="w-full max-w-sm" />
          <Link to={cacheLink} className="btn btn-sm">
            Manage in Cache <ChevronRight className="h-3.5 w-3.5" />
          </Link>
        </Card>
      )}

      <section>
        <div className="mb-2 flex items-end justify-between">
          <h2 className="text-[15px]">Tasks</h2>
          {data.truncated && <span className="text-xs text-muted">showing the first {data.tasks.length}</span>}
        </div>
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-[13px]">
              <thead>
                <tr className="border-b hair text-left">
                  {["Rotation", "Fluxons", "Source", "Method", "Already cached"].map((h) => (
                    <th key={h} className="label px-4 py-2.5 font-medium">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.tasks.map((t, i) => (
                  <tr key={i} className="border-b border-line/40 last:border-0">
                    <td className="num px-4 py-2 font-medium">CR {t.cr}</td>
                    <td className="num px-4 py-2">{t.nflux}</td>
                    <td className="px-4 py-2">{instOf(t.adapt)}</td>
                    <td className="px-4 py-2 text-muted">{t.method}</td>
                    <td className="px-4 py-2">
                      <Link to={`${cacheLink}&cr=${enc(t.cr)}`} className="flex gap-1" title="Open these files in Cache">
                        {CACHE_KINDS.map(([k, l, long]) => {
                          const have = (t.cache as Record<string, boolean>)[k];
                          return (
                            <Tip key={k} label={`${long}: ${have ? "on disk" : "not on disk"}`}>
                              <span className={cn("rounded px-1.5 py-0.5 text-[10px] font-medium", have ? "bg-ok/12 text-ok" : "bg-raised text-faint")}>{l}</span>
                            </Tip>
                          );
                        })}
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      <section>
        <div className="mb-2 flex items-end justify-between">
          <h2 className="text-[15px]">Settings</h2>
          <button className="text-xs text-muted hover:text-ink" onClick={() => setShowDefault(!showDefault)}>
            {showDefault ? "Only this preset's settings" : `Show all ${data.entries.length} (including DEFAULT)`}
          </button>
        </div>
        <Card className="divide-y divide-line/40">
          {shown.length === 0 && <p className="px-4 py-3 text-[13px] text-muted">This preset inherits everything from DEFAULT.</p>}
          {shown.map((e) => (
            <div key={e.key} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] items-baseline gap-3 px-4 py-2 text-[13px] sm:grid-cols-[200px_minmax(0,1fr)_auto]">
              <span className="truncate font-mono text-xs text-ink/90">{e.key}</span>
              <span className="truncate font-mono text-xs text-muted" title={e.resolved ?? e.value}>
                {e.value}
                {e.resolved && e.resolved !== e.value && <span className="text-faint"> → {e.resolved}</span>}
              </span>
              <span className="hidden sm:block">
                {e.source === "profile" ? e.overrides_default ? <Badge tone="accent">overrides default</Badge> : <Badge>this preset</Badge> : <Badge>default</Badge>}
              </span>
            </div>
          ))}
        </Card>
      </section>
      {open && <RunDrawer detail={data} open={open} onOpenChange={setOpen} />}
      {edit && <EditDrawer detail={data} open={edit} onOpenChange={setEdit} />}
      {dup && <NewPresetDialog open={dup} onOpenChange={setDup} presets={presets} from={data.name} />}
      {rename && (
        <NameDialog
          open={rename}
          onOpenChange={setRename}
          title={`Rename ${data.name}`}
          initial={data.name}
          confirm="Rename"
          onSubmit={async (n) => {
            const r = await api.post<{ name: string }>(`/api/profiles/${enc(data.name)}/rename`, { name: n });
            inval();
            toast("ok", `Renamed to ${r.name}`);
            nav(`/presets/${enc(r.name)}`, { replace: true });
          }}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ list */
/* ------------------------------------------------------------------ list filtering */
const AXIS_LABEL: Record<string, string> = { rotations: "CR", fluxon_count: "fluxons", adapts: "source", flow_method: "method" };

function presetFields(p: ProfileSummary): Field[] {
  const f: Field[] = [{ label: "name", text: p.name, weight: 3 }];
  if (p.batch_name) f.push({ label: "batch", text: p.batch_name, weight: 2, display: `batch ${p.batch_name}` });
  for (const [k, vals] of Object.entries(p.axes)) {
    const lab = AXIS_LABEL[k] ?? k;
    const shown = k === "adapts" ? vals.map((v) => instOf(v)) : vals;
    f.push({ label: lab, text: `${lab} ${shown.join(" ")} ${shown.map((v) => `${lab}${v}`).join(" ")}`, weight: 2, quiet: k === "rotations" || k === "fluxon_count", display: shown.join(", ") });
  }
  for (const [k, v] of Object.entries(p.own ?? {})) f.push({ label: k, text: `${k} ${v}`, display: `${k} = ${v}` });
  f.push({ label: "origin", text: p.origin === "user" ? "yours mine user" : "built-in builtin config.ini", quiet: true });
  if (p.hidden) f.push({ label: "state", text: "hidden", quiet: true });
  if (p.errors.length) f.push({ label: "problem", text: `can't run invalid broken ${p.errors.join(" ")}`, quiet: true });
  if (p.warnings.length) f.push({ label: "warning", text: p.warnings.join(" "), display: p.warnings[0] });
  if (p.runs) f.push({ label: "last run", text: `ran ${p.runs.last_status ?? ""}`, display: `last run ${p.runs.last_status ?? ""}` });
  return f;
}

const isComplete = (p: ProfileSummary) => !!p.cache && p.cache.tasks > 0 && p.cache.tasks === p.cache.of && p.cache.counts.relaxed_world >= p.cache.tasks;
const hasResults = (p: ProfileSummary) => !!p.cache && p.cache.counts.relaxed_world > 0;

type FilterKey = "runnable" | "complete" | "partial" | "fresh" | "ran";
const FILTERS: { key: FilterKey; label: string; tip: string; test: (p: ProfileSummary) => boolean }[] = [
  { key: "runnable", label: "Runnable", tip: "Only presets that can run as written", test: (p) => p.errors.length === 0 },
  { key: "complete", label: "Complete", tip: "Every task already has a relaxed world in the data folder", test: isComplete },
  { key: "partial", label: "Has results", tip: "At least one task has a relaxed world in the data folder", test: hasResults },
  { key: "fresh", label: "Not run yet", tip: "Runnable, and no task has a relaxed world yet", test: (p) => p.errors.length === 0 && !hasResults(p) },
  { key: "ran", label: "Ran in Plumber", tip: "Presets with at least one job finished OK from Plumber", test: (p) => (p.runs?.ok ?? 0) > 0 },
];

function loadFilters(): Set<FilterKey> {
  try {
    return new Set(JSON.parse(localStorage.getItem("plumber.presetFilters") || "[]"));
  } catch {
    return new Set();
  }
}

function PresetTile({ p, selected, match }: { p: ProfileSummary; selected: boolean; match?: Match | null }) {
  const rot = p.axes.rotations ?? [];
  return (
    <Link to={`/presets/${enc(p.name)}`} className="block focus:outline-none">
      <Card className={cn("p-3.5 transition-colors hover:border-faint/50", selected && "border-accent/50 bg-raised/40", p.hidden && "opacity-60")}>
        <div className="flex items-center justify-between gap-2">
          <p className="flex min-w-0 items-center gap-1.5 truncate text-[13.5px] font-semibold">
            <span className="truncate">{p.name}</span>
            {p.origin === "user" && <Badge tone="accent">yours</Badge>}
            {p.hidden && <EyeOff className="h-3.5 w-3.5 shrink-0 text-faint" />}
          </p>
          {p.errors.length > 0 ? <Badge tone="fail">can't run</Badge> : <Badge>{p.task_count} task{p.task_count === 1 ? "" : "s"}</Badge>}
        </div>
        <p className="mt-1 truncate text-xs text-muted">
          {rot.length ? `CR ${rot.slice(0, 3).join(", ")}${rot.length > 3 ? "..." : ""}` : "no rotations"} · {p.axes.fluxon_count?.join(", ") || "?"} fluxons
          {p.runs && p.runs.ok > 0 && <span className="text-faint"> · ran {p.runs.ok}x here</span>}
        </p>
        {match?.hint && (
          <p className="mt-1 truncate font-mono text-[10.5px] text-accent/90" title={match.hint.display ?? match.hint.text}>
            {match.hint.display ?? match.hint.text}
          </p>
        )}
        {p.errors.length > 0 ? (
          <p className="mt-2 line-clamp-2 text-[11.5px] leading-snug text-fail/90">{p.errors[0]}</p>
        ) : (
          p.cache && <CacheBar cache={p.cache} className="mt-3 border-t border-line/50 pt-2.5" />
        )}
      </Card>
    </Link>
  );
}

export function PresetsPage() {
  const { name } = useParams();
  const { data, error, refetch } = useProfiles();
  const [showHidden, setShowHidden] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deletedOpen, setDeletedOpen] = useState(false);
  const [q, setQ] = useState("");
  const file = useRef<HTMLInputElement>(null);
  const inval = useInvalidatePresets();
  const toast = useToast();
  const presets = useMemo(() => data?.profiles ?? [], [data]);
  const hiddenCount = presets.filter((p) => p.hidden).length;
  const [filters, setFilters] = useState<Set<FilterKey>>(loadFilters);
  const toggleFilter = (k: FilterKey) =>
    setFilters((cur) => {
      const n = new Set(cur);
      if (n.has(k)) n.delete(k);
      else {
        n.add(k);
        if (k === "fresh") ["complete", "partial"].forEach((x) => n.delete(x as FilterKey)); // contradictory
        if (k === "complete" || k === "partial") n.delete("fresh");
      }
      try {
        localStorage.setItem("plumber.presetFilters", JSON.stringify([...n]));
      } catch {
        /* per-browser convenience only */
      }
      return n;
    });
  const matches = useMemo(() => {
    const m = new Map<string, Match | null>();
    for (const p of presets) m.set(p.name, q.trim() ? fuzzyMatch(q, presetFields(p)) : { score: 0 });
    return m;
  }, [presets, q]);
  const active = FILTERS.filter((f) => filters.has(f.key));
  const visible = presets
    .filter((p) => (showHidden || !p.hidden || p.name === name) && matches.get(p.name) && active.every((f) => f.test(p)))
    .sort((a, b) => (q.trim() ? (matches.get(b.name)?.score ?? 0) - (matches.get(a.name)?.score ?? 0) : 0));
  const filteredOut = presets.filter((p) => showHidden || !p.hidden).length - visible.filter((p) => showHidden || !p.hidden).length;
  const mine = visible.filter((p) => p.origin === "user");
  const builtin = visible.filter((p) => p.origin === "builtin");

  const onImport = async (f: File | undefined) => {
    if (!f) return;
    try {
      const r = await api.post<{ added: string[] }>("/api/profiles-import", { text: await f.text() });
      inval();
      toast("ok", `Imported ${r.added.join(", ")}`);
    } catch (e) {
      toast("error", errText(e));
    } finally {
      if (file.current) file.current.value = "";
    }
  };

  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  const group = (title: string, rows: ProfileSummary[]) =>
    rows.length > 0 && (
      <div className="space-y-2">
        <p className="px-1 text-[11px] uppercase tracking-[0.12em] text-faint">{title}</p>
        {rows.map((p) => (
          <PresetTile key={p.name} p={p} selected={name === p.name} match={q.trim() ? matches.get(p.name) : null} />
        ))}
      </div>
    );
  const list = (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <button className="btn btn-primary btn-sm" onClick={() => setCreating(true)}>
          <Plus className="h-3.5 w-3.5" /> New
        </button>
        <button className="btn btn-sm" onClick={() => file.current?.click()}>
          <Upload className="h-3.5 w-3.5" /> Import
        </button>
        <input ref={file} type="file" accept=".ini,.cfg,.txt,text/plain" className="hidden" onChange={(e) => onImport(e.target.files?.[0])} />
        {hiddenCount > 0 && (
          <button className="btn btn-sm btn-ghost" onClick={() => setShowHidden(!showHidden)}>
            {showHidden ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />} {showHidden ? "Hide" : "Show"} hidden ({hiddenCount})
          </button>
        )}
        {(data?.deleted.length ?? 0) > 0 && (
          <button className="btn btn-sm btn-ghost" onClick={() => setDeletedOpen(true)}>
            <Trash2 className="h-3.5 w-3.5" /> Deleted ({data?.deleted.length})
          </button>
        )}
      </div>
      <div className="space-y-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-2 h-4 w-4 text-faint" />
          <input
            className="input pl-8 pr-8"
            placeholder="Search name, batch, CR, fluxons, any setting"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            aria-label="Search presets"
          />
          {q && (
            <button className="absolute right-2 top-2 text-faint hover:text-ink" aria-label="Clear search" onClick={() => setQ("")}>
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {FILTERS.map((f) => {
            const on = filters.has(f.key);
            const n = presets.filter((p) => (showHidden || !p.hidden) && f.test(p)).length;
            return (
              <Tip key={f.key} label={f.tip}>
                <button
                  onClick={() => toggleFilter(f.key)}
                  aria-pressed={on}
                  className={cn(
                    "rounded-full border px-2.5 py-0.5 text-[11.5px] transition-colors",
                    on ? "border-accent/60 bg-accent/15 text-ink" : "border-line text-muted hover:border-faint/60 hover:text-ink",
                  )}
                >
                  {f.label} <span className="num text-faint">{n}</span>
                </button>
              </Tip>
            );
          })}
        </div>
        {data && filteredOut > 0 && (q.trim() || active.length > 0) && (
          <p className="flex items-center justify-between px-1 text-[11px] text-faint">
            {visible.length} shown, {filteredOut} filtered out
            <button className="hover:text-ink hover:underline" onClick={() => { setQ(""); active.forEach((f) => toggleFilter(f.key)); }}>
              Clear
            </button>
          </p>
        )}
      </div>
      {!data && [0, 1, 2].map((i) => <Skeleton key={i} className="h-[96px]" />)}
      {data && visible.length === 0 && <p className="px-1 py-6 text-center text-[13px] text-muted">No presets match.</p>}
      {group("Your presets", mine)}
      {group(mine.length ? "Built-in (config.ini)" : "Presets (config.ini)", builtin)}
    </div>
  );
  return (
    <div className="animate-fade-in">
      {!name && <PageHeader title="Presets" sub="A preset is a saved recipe for a set of runs: rotations x fluxon counts x sources x methods, plus every setting they use." />}
      <DataFolderBanner className="mb-4" />
      <div className="grid gap-6 md:grid-cols-[300px_minmax(0,1fr)]">
        <div className={cn(name && "hidden md:block")}>{list}</div>
        <div className={cn(!name && "hidden md:block")}>
          {name ? (
            <PresetView name={name} presets={presets} />
          ) : (
            <EmptyState
              icon={<Layers className="h-5 w-5" />}
              title="Select a preset"
              body="Choose one on the left to see its tasks, what is already cached, and its settings. Or make your own."
              action={
                <button className="btn btn-primary" onClick={() => setCreating(true)}>
                  <Plus className="h-4 w-4" /> New preset
                </button>
              }
            />
          )}
        </div>
      </div>
      {creating && <NewPresetDialog open={creating} onOpenChange={setCreating} presets={presets} />}
      {deletedOpen && <DeletedDialog open={deletedOpen} onOpenChange={setDeletedOpen} names={data?.deleted ?? []} />}
    </div>
  );
}
