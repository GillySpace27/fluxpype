import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, FolderOpen, RotateCcw } from "lucide-react";
import { useState } from "react";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { useDataDir, useStatus } from "../lib/hooks";
import type { DataDirReport, DataFolder } from "../lib/types";
import { Badge, Card, Skeleton, useToast } from "./ui";

export function useSetDataDir() {
  const qc = useQueryClient();
  const toast = useToast();
  return useMutation({
    mutationFn: (path: string | null) => api.put<DataDirReport>("/api/data-dir", { path }),
    onSuccess: (r) => {
      qc.setQueryData(["datadir"], r);
      qc.invalidateQueries();
      toast("ok", r.source === "plumber" ? `Real runs now use ${r.current.path}` : "Back to the data folder in config.ini");
    },
    onError: (e) => toast("error", e instanceof Error ? e.message : String(e)),
  });
}

export function folderSummary(f: DataFolder): string {
  if (!f.exists) return "does not exist";
  if (!f.has_data) return "empty";
  const parts = [`${f.magnetograms} magnetogram${f.magnetograms === 1 ? "" : "s"}`, `${f.pfss} PFSS`, `${f.batches} batch${f.batches === 1 ? "" : "es"}`];
  return parts.join(" · ");
}

/** The better folder to offer when the current one holds no runs, or null. */
export function betterFolder(d: DataDirReport | undefined): DataFolder | null {
  if (!d || d.current.has_data) return null;
  return d.candidates.find((c) => c.has_data && c.real !== d.current.real) ?? null;
}

/** Shown on Presets and Cache when the real data folder is empty but another folder on this computer holds runs. */
export function DataFolderBanner({ className }: { className?: string }) {
  const { data: status } = useStatus();
  const { data } = useDataDir();
  const set = useSetDataDir();
  const better = betterFolder(data);
  if (!data || !better || status?.sandboxed) return null;
  return (
    <div className={cn("flex flex-wrap items-center gap-x-3 gap-y-2 rounded-xl border border-run/40 bg-run/10 px-3.5 py-2.5 text-[13px]", className)}>
      <FolderOpen className="h-4 w-4 shrink-0 text-run" />
      <span className="min-w-0 flex-1 leading-snug">
        Plumber is reading <span className="break-all font-mono text-xs">{data.current.path}</span> ({folderSummary(data.current)}). Your runs look like they are in{" "}
        <span className="break-all font-mono text-xs">{better.path}</span> ({folderSummary(better)}).
      </span>
      <button className="btn btn-primary btn-sm" disabled={set.isPending} onClick={() => set.mutate(better.path)}>
        Use that folder
      </button>
    </div>
  );
}

/** Settings card: which folder real runs read and write, with the folders found on this computer. */
export function DataFolderCard() {
  const { data } = useDataDir();
  const set = useSetDataDir();
  const [custom, setCustom] = useState("");
  if (!data) return <Skeleton className="h-40" />;
  const current = data.current;
  return (
    <Card className="divide-y divide-line/60">
      <div className="px-4 py-3.5">
        <p className="flex flex-wrap items-center gap-2 text-[13px] font-medium">
          Real runs read and write
          <Badge tone={data.source === "plumber" ? "accent" : "neutral"}>{data.source === "plumber" ? "chosen here" : "from config.ini"}</Badge>
        </p>
        <p className="mt-1 break-all font-mono text-xs">{current.path}</p>
        <p className={cn("mt-0.5 text-xs", current.has_data ? "text-muted" : "text-run")}>{folderSummary(current)}</p>
        {data.source === "plumber" && (
          <button className="btn btn-sm btn-ghost mt-2" onClick={() => set.mutate(null)}>
            <RotateCcw className="h-3.5 w-3.5" /> Use config.ini's folder ({data.config_value})
          </button>
        )}
      </div>
      <div className="px-4 py-3">
        <p className="label mb-2">Found on this computer</p>
        <div className="space-y-1.5">
          {data.candidates.map((c) => {
            const active = c.real === current.real;
            return (
              <div key={c.real} className={cn("flex items-center gap-3 rounded-lg border px-3 py-2", active ? "border-accent/50 bg-accent/5" : "border-line")}>
                <div className="min-w-0 flex-1">
                  <p className="break-all font-mono text-xs">{c.path}</p>
                  <p className="mt-0.5 text-[11px] text-muted">
                    {folderSummary(c)}
                    {c.aliases?.length ? ` · also reached as ${c.aliases.join(", ")}` : ""}
                  </p>
                </div>
                {active ? (
                  <Check className="h-4 w-4 shrink-0 text-accent" aria-label="In use" />
                ) : (
                  <button className="btn btn-sm" disabled={set.isPending} onClick={() => set.mutate(c.path)}>
                    Use this
                  </button>
                )}
              </div>
            );
          })}
          {data.candidates.length === 0 && <p className="text-xs text-muted">No other FLUX data folders found next to this checkout.</p>}
        </div>
        <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (custom.trim()) set.mutate(custom.trim());
          }}
        >
          <input className="input flex-1 font-mono text-xs" placeholder="Or type a folder path, e.g. ~/data/flux" value={custom} onChange={(e) => setCustom(e.target.value)} />
          <button className="btn btn-sm" disabled={!custom.trim() || set.isPending}>
            Use path
          </button>
        </form>
        <p className="mt-2 text-[11px] leading-snug text-faint">
          Applies to new runs and to the cache status on presets. Simulated runs always use their own sandbox. Running jobs keep the folder they started with.
        </p>
      </div>
    </Card>
  );
}
