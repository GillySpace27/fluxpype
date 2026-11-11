import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Copy, Download, ExternalLink, Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { ago, bytes, kindLabel, num } from "../lib/format";
import type { CacheRecord, FilePreview, FitsCard, FitsRender, PresetScope } from "../lib/types";
import { Badge, Dialog, ErrorNote, Skeleton, useToast } from "./ui";

const q = (path: string) => encodeURIComponent(path);
export const fileUrl = (path: string, download = false) => `/api/cache/file?path=${q(path)}${download ? "&download=1" : ""}`;

function errText(e: unknown) {
  return e instanceof Error ? e.message : String(e);
}

function FitsImage({ path }: { path: string }) {
  const { data, error, isLoading } = useQuery({
    queryKey: ["fits-render", path],
    queryFn: () => api.get<FitsRender>(`/api/cache/render?path=${q(path)}`),
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

function FitsView({ p }: { p: FilePreview }) {
  const hdus = p.hdus ?? [];
  const [hdu, setHdu] = useState(0);
  return (
    <div className="space-y-4">
      {p.renderable ? <FitsImage path={p.path} /> : <p className="text-[13px] text-muted">No 2-D image in the first HDUs.</p>}
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

function Body({ p }: { p: FilePreview }) {
  if (p.view === "image")
    return (
      <a href={fileUrl(p.path)} target="_blank" rel="noreferrer" className="block overflow-hidden rounded-xl border border-line bg-[repeating-conic-gradient(rgb(128_128_128/.12)_0%_25%,transparent_0%_50%)] bg-[length:16px_16px]">
        <img src={fileUrl(p.path)} alt={p.name} className="mx-auto max-h-[75vh] w-full object-contain" />
      </a>
    );
  if (p.view === "pdf") return <iframe src={fileUrl(p.path)} title={p.name} className="h-[75vh] w-full rounded-xl border border-line bg-white" />;
  if (p.view === "fits") return <FitsView p={p} />;
  if (p.view === "text") return <TextView p={p} />;
  return <p className="rounded-xl border border-line bg-bg/40 px-3 py-6 text-center text-[13px] text-muted">{p.note ?? "No preview for this file."}</p>;
}

/** Side panel that previews one cache file, with previous/next through the current list (arrow keys too). */
export function FileViewer({
  records,
  index,
  onIndex,
  onClose,
  dataDir,
  usedBy,
}: {
  records: CacheRecord[];
  index: number;
  onIndex: (i: number) => void;
  onClose: () => void;
  dataDir: string;
  usedBy: (r: CacheRecord) => PresetScope[];
}) {
  const r = records[index];
  const toast = useToast();
  const { data, error, refetch } = useQuery({
    queryKey: ["preview", r?.path, r?.mtime],
    queryFn: () => api.get<FilePreview>(`/api/cache/preview?path=${q(r!.path)}`),
    enabled: !!r,
    staleTime: 60_000,
  });
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      if (e.key === "ArrowRight" || e.key === "j") index < records.length - 1 && onIndex(index + 1);
      if (e.key === "ArrowLeft" || e.key === "k") index > 0 && onIndex(index - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, records.length, onIndex]);
  const users = useMemo(() => (r ? usedBy(r) : []), [r, usedBy]);
  if (!r) return null;
  const full = `${dataDir.replace(/\/$/, "")}/${r.path}`;
  return (
    <Dialog
      side
      xl
      open
      onOpenChange={(o) => !o && onClose()}
      title={<span className="break-all font-mono text-[13.5px]">{r.name}</span>}
      description={
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1">
          <Badge>{kindLabel[r.kind] ?? r.kind}</Badge>
          {r.cr && <span>CR {r.cr}</span>}
          {r.batch && <span>batch {r.batch}</span>}
          <span className="num">{bytes(r.size)}</span>
          <span>modified {ago(r.mtime)}</span>
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
      }
      footer={
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
            <a className="btn btn-sm" href={fileUrl(r.path)} target="_blank" rel="noreferrer">
              <ExternalLink className="h-3.5 w-3.5" /> Open
            </a>
          )}
          <a className="btn btn-sm" href={fileUrl(r.path, true)}>
            <Download className="h-3.5 w-3.5" /> Download
          </a>
        </>
      }
    >
      {error ? <ErrorNote error={error} retry={() => refetch()} /> : !data ? <Skeleton className="h-72" /> : <Body p={data} />}
    </Dialog>
  );
}
