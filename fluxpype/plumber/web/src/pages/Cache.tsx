import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Database, FlaskConical, Layers, RotateCcw, Search, Trash2, X } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge, Card, Checkbox, Dialog, EmptyState, ErrorNote, PageHeader, Skeleton, useConfirm, useToast } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { ago, bytes, kindLabel } from "../lib/format";
import { useCache, useProfile, useStatus, useTrash } from "../lib/hooks";
import type { CacheRecord, PresetScope } from "../lib/types";
import { DataFolderBanner } from "../components/DataFolder";
import { FileViewer, useViewerMode } from "../components/FileViewer";
import { FileIcons, ViewSwitch, useFileView } from "../components/FileBrowser";
import { useWide } from "../lib/hooks";

const UNUSED = "__unused__";

/** Presets whose runs read or write this file: same rotation, same source, and same batch for batch outputs. */
function usedBy(r: CacheRecord, scopes: PresetScope[]): PresetScope[] {
  if (!r.cr) return [];
  return scopes.filter((p) => p.crs.includes(r.cr!) && (!r.inst || p.insts.includes(r.inst)) && (!r.batch || r.batch === p.batch));
}

const LIMIT = 200;

function TrashDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (o: boolean) => void }) {
  const { data } = useTrash();
  const qc = useQueryClient();
  const toast = useToast();
  const { ask, node } = useConfirm();
  const [sel, setSel] = useState<Set<string>>(new Set());
  const done = () => {
    qc.invalidateQueries({ queryKey: ["trash"] });
    qc.invalidateQueries({ queryKey: ["cache"] });
  };
  const restore = useMutation({
    mutationFn: (ids: string[]) => api.post("/api/cache/trash/restore", { ids }),
    onSuccess: () => { toast("ok", "Restored to its original place"); setSel(new Set()); done(); },
    onError: (e) => toast("error", String(e)),
  });
  const empty = useMutation({
    mutationFn: () => api.del("/api/cache/trash"),
    onSuccess: () => { toast("ok", "Trash emptied"); done(); },
    onError: (e) => toast("error", String(e)),
  });
  const items = data?.items ?? [];
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      wide
      title="Trash"
      description="Files stay here until you empty the trash. Restoring puts them back exactly where they were."
      footer={
        <>
          <button className="btn btn-danger mr-auto" disabled={!items.length} onClick={async () => { if (await ask({ title: "Empty the trash?", body: `${items.length} item(s), ${bytes(data?.size)}, will be deleted permanently. This cannot be undone.`, confirm: "Delete forever", danger: true })) empty.mutate(); }}>
            <Trash2 className="h-3.5 w-3.5" /> Empty trash
          </button>
          <button className="btn" disabled={!sel.size} onClick={() => restore.mutate([...sel])}>
            <RotateCcw className="h-3.5 w-3.5" /> Restore {sel.size || ""}
          </button>
        </>
      }
    >
      {items.length === 0 ? (
        <p className="py-8 text-center text-[13px] text-muted">The trash is empty.</p>
      ) : (
        <div className="divide-y divide-line/50 rounded-lg border border-line">
          {items.map((i) => (
            <label key={i.id} className="flex cursor-pointer items-center gap-3 px-3 py-2 text-[13px] hover:bg-raised/50">
              <Checkbox checked={sel.has(i.id)} label={i.path} onChange={() => setSel((s) => { const n = new Set(s); n.has(i.id) ? n.delete(i.id) : n.add(i.id); return n; })} />
              <span className="min-w-0 flex-1">
                <span className="block truncate font-mono text-xs">{i.path}</span>
                <span className="text-[11px] text-faint">{kindLabel[i.kind] ?? i.kind}</span>
              </span>
              <span className="num text-xs text-muted">{bytes(i.size)}</span>
            </label>
          ))}
        </div>
      )}
      {node}
    </Dialog>
  );
}

function UsedBy({ presets }: { presets: PresetScope[] }) {
  if (!presets.length) return <span className="text-[11px] text-faint">no preset</span>;
  const shown = presets.slice(0, 2);
  return (
    <span className="flex flex-wrap items-center gap-1">
      {shown.map((p) => (
        <Link key={p.name} to={`/presets/${encodeURIComponent(p.name)}`} className="rounded-md border border-line px-1.5 py-px text-[11px] text-muted hover:border-accent/50 hover:text-ink">
          {p.name}
        </Link>
      ))}
      {presets.length > 2 && (
        <span className="text-[11px] text-faint" title={presets.slice(2).map((p) => p.name).join(", ")}>
          +{presets.length - 2}
        </span>
      )}
    </span>
  );
}

export function CachePage() {
  const { data, error, refetch } = useCache();
  const qc = useQueryClient();
  const toast = useToast();
  const { ask, node } = useConfirm();
  const [kind, setKind] = useState<string | null>(null);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [trashOpen, setTrashOpen] = useState(false);
  const [viewing, setViewing] = useState<number | null>(null);
  const [viewerMode, setViewerMode] = useViewerMode();
  const [fmode, setFmode, fsize, setFsize] = useFileView("cache");
  const wide = useWide();
  useEffect(() => {
    if (viewing !== null) document.querySelector(`[data-row="${viewing}"]`)?.scrollIntoView({ block: "nearest" });
  }, [viewing]);
  const [limit, setLimit] = useState(LIMIT);
  const [params, setParams] = useSearchParams();
  const presetName = params.get("preset") ?? undefined;
  const crParam = params.get("cr");
  const { data: preset } = useProfile(presetName === UNUSED ? undefined : presetName);
  const { data: status } = useStatus();

  const scopes = useMemo(() => data?.presets ?? [], [data]);
  const users = useMemo(() => new Map((data?.records ?? []).map((r) => [r.id, usedBy(r, scopes)])), [data, scopes]);
  const unusedOnly = presetName === UNUSED;
  const usedByOf = useCallback((r: CacheRecord) => users.get(r.id) ?? [], [users]);
  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const scope = presetName && !unusedOnly ? scopes.find((p) => p.name === presetName) : undefined;
    return (data?.records ?? []).filter((r) => {
      const u = users.get(r.id) ?? [];
      return (
        (!kind || r.kind === kind) &&
        (!needle || `${r.name} ${r.cr ?? ""} ${r.batch ?? ""} ${u.map((p) => p.name).join(" ")}`.toLowerCase().includes(needle)) &&
        (!crParam || r.cr === crParam) &&
        (!unusedOnly || u.length === 0) &&
        (!presetName || unusedOnly || (scope ? u.includes(scope) : false))
      );
    });
  }, [data, kind, q, crParam, presetName, unusedOnly, scopes, users]);
  const pickPreset = (name: string) => {
    const next = new URLSearchParams(params);
    if (name) next.set("preset", name);
    else next.delete("preset");
    next.delete("cr");
    setParams(next);
    setLimit(LIMIT);
  };
  const shown = rows.slice(0, limit);
  const byId = useMemo(() => new Map((data?.records ?? []).map((r) => [r.id, r])), [data]);
  const selSize = [...sel].reduce((a, id) => a + (byId.get(id)?.size ?? 0), 0);

  const trash = useMutation({
    mutationFn: (paths: string[]) => api.post("/api/cache/trash", { paths }),
    onSuccess: () => {
      toast("ok", `Moved ${sel.size} item(s) to the trash`);
      setSel(new Set());
      qc.invalidateQueries({ queryKey: ["cache"] });
      qc.invalidateQueries({ queryKey: ["trash"] });
    },
    onError: (e) => toast("error", e instanceof Error ? e.message : String(e)),
  });

  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  const viewer =
    viewing !== null && data && rows[viewing] ? (
      <FileViewer
        records={rows}
        index={viewing}
        onIndex={(i) => {
          setViewing(i);
          if (i >= limit) setLimit(i + 1);
        }}
        onClose={() => setViewing(null)}
        dataDir={data.data_dir}
        usedBy={usedByOf}
        mode={wide ? viewerMode : "popup"}
        onMode={wide ? setViewerMode : undefined}
        className="lg:sticky lg:top-4"
      />
    ) : null;
  const inlineOpen = !!viewer && wide && viewerMode === "inline";
  const toggle = (r: CacheRecord) => setSel((s) => { const n = new Set(s); n.has(r.id) ? n.delete(r.id) : n.add(r.id); return n; });
  const allShown = shown.length > 0 && shown.every((r) => sel.has(r.id));

  return (
    <div className="animate-fade-in space-y-6">
      <PageHeader
        title="Cache"
        sub={
          data ? (
            <span className="flex flex-wrap items-center gap-2">
              {status?.sandboxed && (
                <Badge tone="accent">
                  <FlaskConical className="mr-1 inline h-3 w-3" />
                  sandbox for simulated runs
                </Badge>
              )}
              <span className="font-mono text-xs">{data.data_dir}</span>
            </span>
          ) : undefined
        }
        right={
          <button className="btn" onClick={() => setTrashOpen(true)}>
            <Trash2 className="h-3.5 w-3.5" /> Trash {data && data.trash.count > 0 && <Badge>{data.trash.count}</Badge>}
          </button>
        }
      />
      <DataFolderBanner />
      {!data ? (
        <Skeleton className="h-64" />
      ) : !data.exists || data.total_count === 0 ? (
        <EmptyState icon={<Database className="h-5 w-5" />} title="Nothing cached yet" body="Magnetograms, footpoints, worlds and relaxed worlds saved by runs will be listed here, so you can reuse or clean them up." />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-5">
            {data.tiles.map((t) => (
              <button key={t.kind} onClick={() => { setKind(kind === t.kind ? null : t.kind); setLimit(LIMIT); }} className="text-left focus:outline-none">
                <Card className={cn("p-3.5 transition-colors hover:border-faint/50", kind === t.kind && "border-accent/60 bg-raised/50")}>
                  <p className="label">{kindLabel[t.kind] ?? t.kind}</p>
                  <p className="num mt-1.5 text-xl font-semibold leading-none">{t.count}</p>
                  <p className="num mt-1 text-xs text-muted">{bytes(t.size)}</p>
                </Card>
              </button>
            ))}
          </div>

          {presetName && !unusedOnly && (
            <div className="flex flex-wrap items-center gap-2 rounded-xl border border-accent/40 bg-accent/10 px-3 py-2 text-[13px]">
              <Layers className="h-4 w-4 text-accent" />
              Files for preset
              <Link to={`/presets/${encodeURIComponent(presetName)}`} className="font-semibold hover:underline">
                {presetName}
              </Link>
              {crParam ? <span className="text-muted">· CR {crParam}</span> : preset && <span className="text-muted">· CR {(preset.axes.rotations ?? []).join(", ")}</span>}
              {preset?.batch_name && <span className="text-muted">· batch {preset.batch_name}</span>}
              <button className="btn btn-ghost btn-sm ml-auto" onClick={() => setParams({})}>
                <X className="h-3.5 w-3.5" /> Show everything
              </button>
            </div>
          )}
          <div className={cn(inlineOpen && "grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]")}>
          <Card className="min-w-0">
            <div className="flex flex-wrap items-center gap-3 border-b hair p-3">
              <div className="relative min-w-[180px] flex-1">
                <Search className="pointer-events-none absolute left-2.5 top-2 h-4 w-4 text-faint" />
                <input className="input pl-8" placeholder="Filter by name or rotation" value={q} onChange={(e) => setQ(e.target.value)} />
              </div>
              <select
                className="input w-auto max-w-[220px] text-[13px]"
                value={presetName ?? ""}
                onChange={(e) => pickPreset(e.target.value)}
                aria-label="Show files for a preset"
              >
                <option value="">All files</option>
                <option value={UNUSED}>Not used by any preset</option>
                <optgroup label="Used by preset">
                  {scopes.filter((p) => !p.hidden || p.name === presetName).map((p) => (
                    <option key={p.name} value={p.name}>{p.name}</option>
                  ))}
                </optgroup>
              </select>
              <span className="num text-xs text-muted">Click a file to preview it · {rows.length} of {data.total_count} files · {bytes(data.total_size)} total</span>
              <div className="flex items-center gap-2">
              {fmode === "icons" && (
                <span className="flex items-center gap-1.5 text-xs text-muted">
                  <Checkbox checked={allShown} indeterminate={!allShown && sel.size > 0} label="Select all" onChange={() => setSel(allShown ? new Set() : new Set(shown.map((r) => r.id)))} /> All
                </span>
              )}
              <ViewSwitch mode={fmode} onMode={setFmode} size={fsize} onSize={setFsize} />
              </div>
              <button className="btn btn-danger btn-sm" disabled={!sel.size}
                onClick={async () => { if (await ask({ title: `Move ${sel.size} item(s) to the trash?`, body: `${bytes(selSize)} will be moved aside. Nothing is deleted until you empty the trash, and you can restore it first.`, confirm: "Move to trash", danger: true })) trash.mutate([...sel].map((id) => byId.get(id)!.path)); }}>
                <Trash2 className="h-3.5 w-3.5" /> Trash {sel.size || ""}
              </button>
            </div>
            {fmode === "icons" ? (
              <div className="p-3">
                <FileIcons records={shown} size={fsize} active={viewing} onOpen={setViewing} idOf={(r) => r.id} selected={sel} onToggle={toggle} tag={(r) => <Badge>{kindLabel[r.kind] ?? r.kind}{r.cr ? ` · CR ${r.cr}` : ""}</Badge>} />
                {rows.length === 0 && <p className="px-4 py-8 text-center text-[13px] text-muted">No files match this filter.</p>}
              </div>
            ) : (
            <div className="overflow-x-auto">
              <table className={cn("w-full text-[13px]", !inlineOpen && "min-w-[760px]")}>
                <thead>
                  <tr className="border-b hair text-left">
                    <th className="w-10 px-3 py-2"><Checkbox checked={allShown} indeterminate={!allShown && sel.size > 0} label="Select all" onChange={() => setSel(allShown ? new Set() : new Set(shown.map((r) => r.id)))} /></th>
                    {(inlineOpen ? ["File", "Type", "Rotation", "Size"] : ["File", "Type", "Rotation", "Used by", "Size", "Modified"]).map((h) => <th key={h} className="label px-3 py-2 font-medium">{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {shown.map((r, i) => (
                    <tr key={r.id} data-row={i} onClick={() => setViewing(i)} title="Preview" className={cn("cursor-pointer border-b border-line/40 last:border-0 hover:bg-raised/40", sel.has(r.id) && "bg-accent/5", viewing === i && "bg-accent/10 shadow-[inset_2px_0_0_rgb(var(--accent))]")}>
                      <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}><Checkbox checked={sel.has(r.id)} label={r.name} onChange={() => toggle(r)} /></td>
                      <td className={cn("truncate px-3 py-2 font-mono text-xs", inlineOpen ? "max-w-[220px]" : "max-w-[360px]")} title={r.path}>{r.name}</td>
                      <td className="px-3 py-2"><Badge>{kindLabel[r.kind] ?? r.kind}</Badge></td>
                      <td className="num px-3 py-2 text-muted">{r.cr ? `CR ${r.cr}` : "-"}</td>
                      {!inlineOpen && (
                        <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                          <UsedBy presets={users.get(r.id) ?? []} />
                        </td>
                      )}
                      <td className="num px-3 py-2">{bytes(r.size)}</td>
                      {!inlineOpen && <td className="px-3 py-2 text-muted">{ago(r.mtime)}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
              {rows.length === 0 && <p className="px-4 py-8 text-center text-[13px] text-muted">No files match this filter.</p>}
            </div>
            )}
            {rows.length > shown.length && (
              <div className="border-t hair p-3 text-center">
                <button className="btn btn-sm" onClick={() => setLimit(limit + LIMIT)}>Show {Math.min(LIMIT, rows.length - shown.length)} more</button>
              </div>
            )}
          </Card>
          {inlineOpen && viewer}
          </div>
        </>
      )}
      {trashOpen && <TrashDialog open={trashOpen} onOpenChange={setTrashOpen} />}
      {viewing !== null && !inlineOpen && viewer}
      {node}
    </div>
  );
}
