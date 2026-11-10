import { ChevronLeft, ChevronRight, ImageOff } from "lucide-react";
import { useEffect, useState } from "react";
import { artifactUrl } from "../lib/api";
import { clock } from "../lib/format";
import type { Artifact } from "../lib/types";
import { Badge, Dialog, EmptyState } from "./ui";

const name = (p: string) => p.split("/").pop() ?? p;

export function ArtifactGrid({ artifacts }: { artifacts: Artifact[] }) {
  const [open, setOpen] = useState<number | null>(null);
  const items = [...artifacts].reverse();

  useEffect(() => {
    if (open == null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") setOpen((i) => (i == null ? i : Math.min(i + 1, items.length - 1)));
      if (e.key === "ArrowLeft") setOpen((i) => (i == null ? i : Math.max(i - 1, 0)));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, items.length]);

  if (items.length === 0) {
    return <EmptyState icon={<ImageOff className="h-5 w-5" />} title="No images yet" body="Plots and renders show up here as the pipeline writes them." />;
  }
  const cur = open != null ? items[open] : null;
  return (
    <>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
        {items.map((a, i) => (
          <button key={a.path} onClick={() => setOpen(i)} className="group overflow-hidden rounded-xl border border-line/80 bg-bg/50 text-left transition-colors hover:border-accent/50">
            <div className="aspect-[4/3] overflow-hidden bg-black/40">
              <img src={artifactUrl(a.path)} alt={name(a.path)} loading="lazy" className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]" />
            </div>
            <div className="flex items-center justify-between gap-2 px-2.5 py-2">
              <span className="truncate text-xs text-muted" title={a.path}>
                {name(a.path)}
              </span>
              <Badge tone={a.source === "event" ? "accent" : "neutral"} className="shrink-0 px-1 text-[10px]">
                {a.source === "event" ? "event" : "file"}
              </Badge>
            </div>
          </button>
        ))}
      </div>
      <Dialog open={cur != null} onOpenChange={(o) => !o && setOpen(null)} title={cur ? name(cur.path) : ""} description={cur ? `${clock(cur.t)} · ${cur.path}` : ""} wide>
        {cur && (
          <div className="relative">
            <img src={artifactUrl(cur.path)} alt={name(cur.path)} className="mx-auto max-h-[64vh] rounded-lg" />
            <div className="mt-3 flex items-center justify-between">
              <button className="btn btn-sm" disabled={open === 0} onClick={() => setOpen((i) => Math.max((i ?? 0) - 1, 0))}>
                <ChevronLeft className="h-4 w-4" /> Previous
              </button>
              <span className="num text-xs text-muted">
                {(open ?? 0) + 1} / {items.length}
              </span>
              <button className="btn btn-sm" disabled={open === items.length - 1} onClick={() => setOpen((i) => Math.min((i ?? 0) + 1, items.length - 1))}>
                Next <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}
      </Dialog>
    </>
  );
}
