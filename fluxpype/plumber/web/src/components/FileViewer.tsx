import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Copy, Download, ExternalLink, Loader2, PanelRightClose, PanelRightOpen, X } from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useDatTable } from "../lib/hooks";
import { cn } from "../lib/cn";
import { ago, bytes, kindLabel, num } from "../lib/format";
import type { CacheRecord, FilePreview, FitsCard, FitsRender, PresetScope } from "../lib/types";
import { DatPlot, DatTable } from "./DatViewer";
import { Badge, Dialog, ErrorNote, Skeleton, Tip, useToast } from "./ui";

const FluxViewer = lazy(() => import("./FluxViewer").then((m) => ({ default: m.FluxViewer }))); // three.js: loaded when a .flux file is opened

const q = (path: string) => encodeURIComponent(path);
/** Where the viewer reads files: the current data folder (`/api/cache`) or one job's own folder (`/api/jobs/<id>/data`). */
export const CACHE_BASE = "/api/cache";
export const fileUrl = (path: string, download = false, base = CACHE_BASE) => `${base}/file?path=${q(path)}${download ? "&download=1" : ""}`;

function errText(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

function FitsImage({ path, base }: { path: string; base: string }) {
  const { data, error, isLoading } = useQuery({
    queryKey: ["fits-render", base, path],
    queryFn: () => api.get<FitsRender>(`${base}/render?path=${q(path)}`),
    staleTime: Infinity,
    retry: false,
  });
  if (isLoading)
    return (
      <div className="flex h-56 flex-col items-center justify-center gap-2 rounded-xl border border-line bg-bg/40 text-[13px] text-muted">
        <Loader2 className="h-5 w-5 animate-spin" />
        Rendering with the FLUX Python...
      </div>
    );
  if (error) return <p className="rounded-xl border border-run/40 bg-run/10 px-3 py-2.5 text-[13px]">No image: {errText(error)}</p>;
  if (!data) return null;
  const s = data.stats;
  return (
    <figure className="space-y-1.5">
      <a href={data.image} target="_blank" rel="noreferrer" className="block overflow-hidden rounded-xl border border-line bg-black">
        <img src={data.image} alt={`Image in ${path}`} className="mx-auto max-h-[60vh] w-full object-contain" />
      </a>
      <figcaption className="num text-[11px] text-faint">
        {s.hdu && `HDU ${s.hdu} · `}
        {s.shape && `${s.shape[1]} x ${s.shape[0]} px · `}
        data {num(s.min)} to {num(s.max)} · shown {num(s.vmin)} to {num(s.vmax)} (gray, 99.5th percentile{s.min != null && s.min < 0 ? ", symmetric" : ""})
        {s.nan ? ` · ${s.nan} non-finite pixels` : ""}
      </figcaption>
    </figure>
  );
}

function HeaderTable({ cards }: { cards: FitsCard[] }) {
  const [filter, setFilter] = useState("");
  const f = filter.trim().toLowerCase();
  const rows = f ? cards.filter((c) => `${c.key} ${c.value} ${c.comment}`.toLowerCase().includes(f)) : cards;
  return (
    <div className="space-y-2">
      <input className="input text-xs" placeholder={`Filter ${cards.length} header cards`} value={filter} onChange={(e) => setFilter(e.target.value)} />
      <div className="max-h-[50vh] overflow-auto rounded-lg border border-line">
        <table className="w-full text-[11.5px]">
          <tbody>
            {rows.map((c, i) => (
              <tr key={i} className="border-b border-line/40 last:border-0 align-top">
                <td className="whitespace-nowrap px-2.5 py-1 font-mono font-medium text-ink">{c.key}</td>
                <td className="break-all px-2.5 py-1 font-mono">{c.value}</td>
                <td className="px-2.5 py-1 text-faint">{c.comment}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && <p className="px-3 py-4 text-center text-xs text-muted">No cards match.</p>}
      </div>
    </div>
  );
}

function FitsView({ p, base }: { p: FilePreview; base: string }) {
  const hdus = p.hdus ?? [];
  const [hdu, setHdu] = useState(0);
  return (
    <div className="space-y-4">
      {p.renderable ? <FitsImage path={p.path} base={base} /> : <p className="text-[13px] text-muted">No 2-D image in the first HDUs.</p>}
      <div className="space-y-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {hdus.map((h, i) => (
            <button
              key={i}
              onClick={() => setHdu(i)}
              className={cn("rounded-full border px-2.5 py-0.5 text-[11.5px]", i === hdu ? "border-accent/60 bg-accent/15" : "border-line text-muted hover:text-ink")}
            >
              {h.name}
              {h.dims.length > 0 && <span className="num text-faint"> {h.dims.join("x")}</span>}
            </button>
          ))}
        </div>
        {hdus[hdu] && <HeaderTable key={hdu} cards={hdus[hdu].cards} />}
      </div>
    </div>
  );
}

function TextView({ p }: { p: FilePreview }) {
  const all = p.lines ?? [];
  const skip = Math.max(0, all.findIndex((l) => l.trim() !== "")); // world files start with blank lines
  const lines = all.slice(skip);
  return (
    <div className="space-y-1.5">
      <pre className="max-h-[70vh] overflow-auto rounded-xl border border-line bg-bg/60 py-2 text-[11.5px] leading-[1.55]">
        {lines.map((l, i) => (
          <div key={i} className="flex">
            <span className="num w-12 shrink-0 select-none pr-3 text-right text-faint/70">{i + 1 + skip}</span>
            <span className="whitespace-pre pr-3 font-mono">{l || " "}</span>
          </div>
        ))}
      </pre>
      {p.truncated && <p className="text-[11px] text-faint">Showing the first {all.length} lines. Download the file for the rest.</p>}
    </div>
  );
}

type Sibling = { path: string; name: string };

/** A .flux world file: 3-D view first, the raw text as the fallback tab. */
function FluxView({ p, base, others }: { p: FilePreview; base: string; others: Sibling[] }) {
  const [tab, setTab] = useState<"3d" | "text">("3d");
  return (
    <div className="space-y-3">
      <div className="inline-flex rounded-lg border border-line bg-bg/50 p-0.5" role="radiogroup" aria-label="View">
        {(
          [
            ["3d", "3D"],
            ["text", "Text"],
          ] as const
        ).map(([v, label]) => (
          <button key={v} role="radio" aria-checked={tab === v} onClick={() => setTab(v)} className={cn("rounded-md px-2.5 py-1 text-xs font-medium transition-colors", tab === v ? "bg-raised text-ink shadow-sm" : "text-muted hover:text-ink")}>
            {label}
          </button>
        ))}
      </div>
      {tab === "3d" ? (
        <Suspense fallback={<Skeleton className="h-72" />}>
          <FluxViewer path={p.path} base={base} mtime={p.mtime} others={others} />
        </Suspense>
      ) : (
        <TextView p={p} />
      )}
    </div>
  );
}

/** A numeric .dat table: a chart chosen by the file kind, the table with column statistics, the raw text as fallback. */
function DatView({ p, base }: { p: FilePreview; base: string }) {
  const [tab, setTab] = useState<"plot" | "table" | "text">("plot");
  const { data: d, error, isLoading, refetch } = useDatTable(p.path, base, p.mtime);
  return (
    <div className="space-y-3">
      <div className="inline-flex rounded-lg border border-line bg-bg/50 p-0.5" role="radiogroup" aria-label="View">
        {(
          [
            ["plot", "Plot"],
            ["table", "Table"],
            ["text", "Text"],
          ] as const
        ).map(([v, label]) => (
          <button key={v} role="radio" aria-checked={tab === v} onClick={() => setTab(v)} className={cn("rounded-md px-2.5 py-1 text-xs font-medium transition-colors", tab === v ? "bg-raised text-ink shadow-sm" : "text-muted hover:text-ink")}>
            {label}
          </button>
        ))}
      </div>
      {tab === "text" ? <TextView p={p} /> : error ? <ErrorNote error={error} retry={() => refetch()} /> : isLoading || !d ? <Skeleton className="h-72" /> : tab === "plot" ? <DatPlot d={d} /> : <DatTable d={d} />}
    </div>
  );
}

function Body({ p, base, others }: { p: FilePreview; base: string; others: Sibling[] }) {
  if (p.view === "image")
    return (
      <a href={fileUrl(p.path, false, base)} target="_blank" rel="noreferrer" className="block overflow-hidden rounded-xl border border-line bg-[repeating-conic-gradient(rgb(128_128_128/.12)_0%_25%,transparent_0%_50%)] bg-[length:16px_16px]">
        <img src={fileUrl(p.path, false, base)} alt={p.name} className="mx-auto max-h-[75vh] w-full object-contain" />
      </a>
    );
  if (p.view === "pdf") return <iframe src={fileUrl(p.path, false, base)} title={p.name} className="h-[75vh] w-full rounded-xl border border-line bg-white" />;
  if (p.view === "fits") return <FitsView p={p} base={base} />;
  if (p.view === "text") {
    if (/\.flux$/i.test(p.name)) return <FluxView key={p.path} p={p} base={base} others={others} />;
    if (/\.dat$/i.test(p.name)) return <DatView key={p.path} p={p} base={base} />;
    return <TextView p={p} />;
  }
  return <p className="rounded-xl border border-line bg-bg/40 px-3 py-6 text-center text-[13px] text-muted">{p.note ?? "No preview for this file."}</p>;
}

export type ViewerMode = "inline" | "popup";

/** Remembered per browser: whether previews open beside the list or in a panel over the page. */
export function useViewerMode(): [ViewerMode, (m: ViewerMode) => void] {
  const [mode, setMode] = useState<ViewerMode>(() => {
    try {
      return localStorage.getItem("plumber.viewerMode") === "popup" ? "popup" : "inline";
    } catch {
      return "inline";
    }
  });
  const set = (m: ViewerMode) => {
    setMode(m);
    try {
      localStorage.setItem("plumber.viewerMode", m);
    } catch {
      /* convenience only */
    }
  };
  return [mode, set];
}

/** Previews one file of a list, inline (a pane beside the list) or as a popup panel; previous/next and arrow keys move through the list. */
export function FileViewer({
  records,
  index,
  onIndex,
  onClose,
  dataDir,
  usedBy,
  mode = "popup",
  onMode,
  className,
  base = CACHE_BASE,
  extraMeta,
}: {
  records: CacheRecord[];
  index: number;
  onIndex: (i: number) => void;
  onClose: () => void;
  dataDir: string;
  usedBy?: (r: CacheRecord) => PresetScope[];
  mode?: ViewerMode;
  onMode?: (m: ViewerMode) => void;
  className?: string;
  /** API prefix for preview, file and render; defaults to the current data folder. */
  base?: string;
  /** Extra tags shown with the file details (for example "made by this run"). */
  extraMeta?: (r: CacheRecord) => React.ReactNode;
}) {
  const r = records[index];
  const toast = useToast();
  const { data, error, refetch } = useQuery({
    queryKey: ["preview", base, r?.path, r?.mtime],
    queryFn: () => api.get<FilePreview>(`${base}/preview?path=${q(r!.path)}`),
    enabled: !!r,
    staleTime: 60_000,
  });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
      if (e.key === "ArrowRight" || e.key === "ArrowDown" || e.key === "j") {
        if (index < records.length - 1) {
          e.preventDefault();
          onIndex(index + 1);
        }
      }
      if (e.key === "ArrowLeft" || e.key === "ArrowUp" || e.key === "k") {
        if (index > 0) {
          e.preventDefault();
          onIndex(index - 1);
        }
      }
      if (e.key === "Escape" && mode === "inline") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, records.length, onIndex, onClose, mode]);
  const users = useMemo(() => (r && usedBy ? usedBy(r) : []), [r, usedBy]);
  const fluxSiblings = useMemo(() => records.filter((x) => x.path !== r?.path && /\.flux$/i.test(x.name)), [records, r?.path]);
  if (!r) return null;
  const full = `${dataDir.replace(/\/$/, "")}/${r.path}`;
  const title = <span className="break-all font-mono text-[13.5px]">{r.name}</span>;
  const meta = (
    <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
      <Badge>{kindLabel[r.kind] ?? r.kind}</Badge>
      {r.cr && <span>CR {r.cr}</span>}
      {r.batch && <span>batch {r.batch}</span>}
      <span className="num">{bytes(r.size)}</span>
      <span>modified {ago(r.mtime)}</span>
      {extraMeta?.(r)}
      {users.length > 0 && (
        <span className="flex flex-wrap items-center gap-1">
          used by
          {users.map((p) => (
            <Link key={p.name} to={`/presets/${encodeURIComponent(p.name)}`} className="rounded-md border border-line px-1.5 text-[11px] hover:border-accent/50 hover:text-ink">
              {p.name}
            </Link>
          ))}
        </span>
      )}
    </span>
  );
  const actions = (
    <>
      <span className="num mr-auto text-xs text-faint">
        {index + 1} of {records.length} · arrow keys move
      </span>
      <button className="btn btn-sm btn-icon" aria-label="Previous file" disabled={index === 0} onClick={() => onIndex(index - 1)}>
        <ChevronLeft className="h-4 w-4" />
      </button>
      <button className="btn btn-sm btn-icon" aria-label="Next file" disabled={index >= records.length - 1} onClick={() => onIndex(index + 1)}>
        <ChevronRight className="h-4 w-4" />
      </button>
      <button
        className="btn btn-sm"
        onClick={() =>
          navigator.clipboard?.writeText(full).then(
            () => toast("ok", "Path copied"),
            () => toast("error", "Could not copy"),
          )
        }
      >
        <Copy className="h-3.5 w-3.5" /> Copy path
      </button>
      {(data?.view === "image" || data?.view === "pdf") && (
        <a className="btn btn-sm" href={fileUrl(r.path, false, base)} target="_blank" rel="noreferrer">
          <ExternalLink className="h-3.5 w-3.5" /> Open
        </a>
      )}
      <a className="btn btn-sm" href={fileUrl(r.path, true, base)}>
        <Download className="h-3.5 w-3.5" /> Download
      </a>
    </>
  );
  const body = error ? <ErrorNote error={error} retry={() => refetch()} /> : !data ? <Skeleton className="h-72" /> : <Body p={data} base={base} others={fluxSiblings} />;
  const modeButton = onMode && (
    <Tip label={mode === "inline" ? "Open previews in a panel over the page" : "Show previews beside the list"}>
      <button className="btn btn-ghost btn-icon" aria-label={mode === "inline" ? "Pop out" : "Dock beside the list"} onClick={() => onMode(mode === "inline" ? "popup" : "inline")}>
        {mode === "inline" ? <PanelRightOpen className="h-4 w-4" /> : <PanelRightClose className="h-4 w-4" />}
      </button>
    </Tip>
  );

  if (mode === "inline")
    return (
      <section className={cn("flex max-h-[calc(100vh-6rem)] flex-col rounded-2xl border border-line bg-surface", className)} aria-label="File preview">
        <div className="flex items-start gap-2 border-b hair px-4 py-3">
          <div className="min-w-0 flex-1">
            <p className="font-semibold">{title}</p>
            <div className="mt-1 text-[12.5px] text-muted">{meta}</div>
          </div>
          {modeButton}
          <button className="btn btn-ghost btn-icon" aria-label="Close preview" onClick={onClose}>
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">{body}</div>
        <div className="flex flex-wrap items-center justify-end gap-2 border-t hair px-4 py-2.5">{actions}</div>
      </section>
    );
  return (
    <Dialog
      side
      xl
      open
      onOpenChange={(o) => !o && onClose()}
      title={
        <span className="flex items-start gap-2">
          <span className="min-w-0 flex-1">{title}</span>
          {modeButton}
        </span>
      }
      description={meta}
      footer={actions}
    >
      {body}
    </Dialog>
  );
}
