import { File, FileText, LayoutGrid, List, Sun } from "lucide-react";
import { useRef, useState, type ReactNode } from "react";
import { ago, bytes } from "../lib/format";
import { cn } from "../lib/cn";
import type { CacheRecord } from "../lib/types";
import { CACHE_BASE, fileUrl } from "./FileViewer";
import { Checkbox, Segmented } from "./ui";

export type FileMode = "list" | "icons";
type Surface = "cache" | "runFiles" | "stage" | "results";

const DEFAULT_MODE: Record<Surface, FileMode> = { cache: "list", runFiles: "list", stage: "icons", results: "icons" };
const IMG = /\.(png|jpe?g|svg)$/i;
const FITS = /\.(fits?|fts)(\.gz)?$/i;

/** An image tile; a FITS file shows the server's cached render. Falls back to the icon if the image will not load. */
function Thumb({ src, alt, Icon, ext, contain }: { src?: string; alt: string; Icon: typeof File; ext: string; contain?: boolean }) {
  const [bad, setBad] = useState(false);
  if (src && !bad) return <img src={src} alt={alt} loading="lazy" onError={() => setBad(true)} className={cn("h-full w-full transition-transform duration-300 group-hover:scale-[1.03]", contain ? "object-contain" : "object-cover")} />;
  return (
    <span className="flex flex-col items-center gap-1 text-muted">
      <Icon className="h-7 w-7" />
      <span className="text-[11px] font-medium">{ext || "file"}</span>
    </span>
  );
}
const GAP = 12; // px, matches gap-3

/** Remembered per browser and per surface: list or icons, and the icon size. Not in the URL: Cache keeps its filters there. */
export function useFileView(surface: Surface): [FileMode, (m: FileMode) => void, number, (n: number) => void] {
  const key = `plumber.files.${surface}`;
  const [v, setV] = useState<{ mode: FileMode; size: number }>(() => {
    try {
      const s = JSON.parse(localStorage.getItem(key) ?? "null");
      if (s && (s.mode === "list" || s.mode === "icons") && Number.isFinite(s.size)) return { mode: s.mode, size: Math.min(320, Math.max(96, s.size)) };
    } catch {
      /* convenience only */
    }
    return { mode: DEFAULT_MODE[surface], size: 160 };
  });
  const set = (n: { mode: FileMode; size: number }) => {
    setV(n);
    try {
      localStorage.setItem(key, JSON.stringify(n));
    } catch {
      /* convenience only */
    }
  };
  return [v.mode, (mode) => set({ ...v, mode }), v.size, (size) => set({ ...v, size })];
}

export function ViewSwitch({ mode, onMode, size, onSize }: { mode: FileMode; onMode: (m: FileMode) => void; size: number; onSize: (n: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      {mode === "icons" && <input type="range" min={96} max={320} step={16} value={size} onChange={(e) => onSize(+e.target.value)} aria-label="Icon size" className="w-24 accent-[rgb(var(--accent))]" />}
      <Segmented
        label="File view"
        value={mode}
        onChange={onMode}
        options={[
          { value: "list", label: <span className="flex items-center gap-1.5"><List className="h-3.5 w-3.5" /><span className="hidden sm:inline">List</span></span> },
          { value: "icons", label: <span className="flex items-center gap-1.5"><LayoutGrid className="h-3.5 w-3.5" /><span className="hidden sm:inline">Icons</span></span> },
        ]}
      />
    </div>
  );
}

/** A grid of tiles. Arrows move between tiles; Space or Enter opens one. While a viewer is open (`active` set) the viewer owns the arrow keys. */
export function FileIcons<T extends CacheRecord>({
  records,
  base = CACHE_BASE,
  size,
  active,
  onOpen,
  idOf = (r) => r.path,
  selected,
  onToggle,
  tag,
}: {
  records: T[];
  base?: string;
  size: number;
  active: number | null;
  onOpen: (i: number) => void;
  idOf?: (r: T) => string;
  selected?: Set<string>;
  onToggle?: (r: T) => void;
  tag?: (r: T) => ReactNode;
}) {
  const grid = useRef<HTMLDivElement>(null);
  const [cur, setCur] = useState(0);
  const onKey = (e: React.KeyboardEvent) => {
    if (active != null) return;
    const cols = Math.max(1, Math.floor(((grid.current?.offsetWidth ?? 0) + GAP) / (size + GAP)));
    const step = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -cols, ArrowDown: cols }[e.key];
    if (!step) return;
    e.preventDefault();
    const n = Math.min(records.length - 1, Math.max(0, cur + step));
    setCur(n);
    grid.current?.querySelectorAll<HTMLElement>("[data-tile]")[n]?.focus();
  };
  return (
    <div ref={grid} role="list" onKeyDown={onKey} className="grid gap-3" style={{ gridTemplateColumns: `repeat(auto-fill,minmax(${size}px,1fr))` }}>
      {records.map((r, i) => {
        const ext = r.name.includes(".") ? r.name.split(".").pop()!.toUpperCase() : "";
        const Icon = ext === "PDF" ? FileText : ext === "FITS" || ext === "FIT" ? Sun : File;
        return (
          <div key={idOf(r)} role="listitem" className="relative">
            <button
              data-tile
              tabIndex={i === Math.min(cur, records.length - 1) ? 0 : -1}
              onFocus={() => setCur(i)}
              onClick={() => onOpen(i)}
              className={cn("group block w-full overflow-hidden rounded-xl border bg-bg/50 text-left transition-colors hover:border-accent/50", active === i ? "border-accent/60" : "border-line/80", selected?.has(idOf(r)) && "bg-accent/5")}
            >
              <div className="grid aspect-[4/3] place-items-center overflow-hidden bg-black/40">
                <Thumb
                  src={IMG.test(r.name) ? fileUrl(r.path, false, base) : FITS.test(r.name) ? `${base}/render.png?path=${encodeURIComponent(r.path)}` : undefined}
                  alt={r.name}
                  contain={FITS.test(r.name)} // a 2:1 synoptic map would lose its sides to a 4:3 crop
                  Icon={Icon}
                  ext={ext}
                />
              </div>
              <div className="space-y-1 px-2.5 py-2">
                <p className="truncate text-xs text-muted" title={r.path}>
                  {r.name}
                </p>
                {tag?.(r)}
              </div>
            </button>
            {selected && onToggle && (
              <div className="absolute left-1.5 top-1.5 rounded bg-bg/80 p-0.5">
                <Checkbox checked={selected.has(idOf(r))} label={`Select ${r.name}`} onChange={() => onToggle(r)} />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/** One row per file, for the surfaces that have no table of their own. */
export function FileRows<T extends CacheRecord>({ records, active, onOpen, tag }: { records: T[]; active: number | null; onOpen: (i: number) => void; tag?: (r: T) => ReactNode }) {
  return (
    <div className="divide-y divide-line/60 rounded-xl border border-line/80 bg-surface/40 p-1">
      {records.map((r, i) => (
        <button key={r.path} onClick={() => onOpen(i)} className={cn("flex w-full flex-wrap items-center gap-x-3 gap-y-1 rounded-md px-3 py-1.5 text-left hover:bg-raised/60", active === i && "bg-raised")}>
          <span className="min-w-0 truncate font-mono text-[12px]" title={r.path}>
            {r.name}
          </span>
          {tag?.(r)}
          <span className="num ml-auto text-[11.5px] text-faint">
            {bytes(r.size)} · {ago(r.mtime)}
          </span>
        </button>
      ))}
    </div>
  );
}
