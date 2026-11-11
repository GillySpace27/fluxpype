import { useQuery } from "@tanstack/react-query";
import { CircleDashed, Pause, Play } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { cn } from "../lib/cn";
import { ago, bytes, duration } from "../lib/format";
import { jobDataBase, useTaskResults } from "../lib/hooks";
import type { CacheRecord, ResultFile, TaskState } from "../lib/types";
import { FileViewer, fileUrl } from "./FileViewer";
import { OriginTag, RelaxViews } from "./ResultsPane";
import { useProgress } from "./StageTimeline";
import { Badge, Card, StatusIcon } from "./ui";

/** One line per stage, from the pipeline's own banners and the stage names. */
const ABOUT: Record<string, string> = {
  magnetogram: "Finds the HMI magnetogram for this rotation, downloading it from JSOC if it is not on disk.",
  footpoints: "Places footpoints on the magnetogram, in the order of a Hilbert (space-filling) curve.",
  pfss: "Traces the footpoints through a PFSS model into open and closed field lines.",
  world: "Converts the PFSS field lines into a FLUX world of fluxons.",
  relax: "Relaxes the world until its stiffness falls below the target.",
  world_plots: "Plots the world before and after relaxation.",
  fieldmap: "Draws the footpoint topology figure.",
  wind: "Computes the solar wind along each open fluxon.",
  wind_plots: "Plots the wind solutions and field maps.",
  fluxlight: "Renders FLUXlight images from the relaxed world.",
};
const PRODUCT: Record<string, string> = { relax: "relaxed_world" }; // stage id -> results product id (the rest match)
const IMG = /\.(png|jpe?g|svg)$/i;
const EMPTY: Record<string, string> = {
  pending: "Not reached yet.",
  running: "In progress; files appear here as the stage writes them.",
  skipped: "Skipped in this run.",
  failed: "The stage failed before writing files.",
  ok: "The stage finished but no matching file was found.",
};

/** The footpoint file's rows replayed in file order: it follows the Hilbert curve, so the path fills the map locally. */
function HilbertReplay({ pts }: { pts: number[][] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [n, setN] = useState(pts.length);
  const [playing, setPlaying] = useState(false);
  const bounds = useMemo(() => {
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  }, [pts]);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let at = n >= pts.length ? 0 : n;
    const step = () => {
      at = Math.min(at + Math.max(1, pts.length / 360), pts.length); // about six seconds end to end
      setN(Math.floor(at));
      if (at < pts.length) raf = requestAnimationFrame(step);
      else setPlaying(false);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, pts.length]);

  useEffect(() => {
    const c = canvas.current;
    const g = c?.getContext("2d");
    if (!c || !g) return;
    const { x0, x1, y0, y1 } = bounds;
    const pad = 10;
    const sx = (c.width - 2 * pad) / Math.max(x1 - x0, 1);
    const sy = (c.height - 2 * pad) / Math.max(y1 - y0, 1);
    const at = (p: number[]) => [pad + (p[0] - x0) * sx, c.height - pad - (p[1] - y0) * sy];
    g.clearRect(0, 0, c.width, c.height);
    g.strokeStyle = "rgba(160,170,190,0.35)";
    g.lineWidth = 1;
    g.beginPath();
    pts.slice(0, n).forEach((p, i) => {
      const [x, y] = at(p);
      if (i) g.lineTo(x, y);
      else g.moveTo(x, y);
    });
    g.stroke();
    pts.slice(0, n).forEach((p) => {
      const [x, y] = at(p);
      g.fillStyle = p[2] > 0 ? "#ff9f43" : "#5fb3ff";
      g.beginPath();
      g.arc(x, y, 2.4, 0, 6.2832);
      g.fill();
    });
    if (n > 0 && n <= pts.length) {
      const [x, y] = at(pts[n - 1]);
      g.strokeStyle = "#fff";
      g.lineWidth = 2;
      g.beginPath();
      g.arc(x, y, 6, 0, 6.2832);
      g.stroke();
    }
  }, [n, pts, bounds]);

  return (
    <div className="space-y-2">
      <canvas ref={canvas} width={900} height={380} className="w-full rounded-lg border border-line/70 bg-bg/40" aria-label="Footpoints in placement order" />
      <div className="flex items-center gap-3">
        <button className="btn btn-sm" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>
          {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />} {playing ? "Pause" : "Replay"}
        </button>
        <input type="range" min={0} max={pts.length} value={n} onChange={(e) => { setPlaying(false); setN(+e.target.value); }} className="min-w-0 flex-1 accent-[rgb(var(--accent))]" aria-label="Footpoints shown" />
        <span className="num text-xs text-muted">
          {n} of {pts.length}
        </span>
      </div>
      <p className="text-[11.5px] text-faint">A replay of the footpoint file in row order, which follows the Hilbert curve. It is not recorded live. Orange is positive flux, blue is negative.</p>
    </div>
  );
}

function useFootpoints(file: ResultFile | undefined, base: string) {
  return useQuery({
    queryKey: ["floc", base, file?.path, file?.mtime],
    enabled: !!file,
    queryFn: async () => {
      const r = await fetch(fileUrl(file!.path, false, base));
      if (!r.ok) throw new Error(`${r.status}`);
      return (await r.text())
        .split("\n")
        .map((l) => l.trim().split(/\s+/).map(Number))
        .filter((p) => p.length >= 3 && p.every((v) => Number.isFinite(v)));
    },
    staleTime: Infinity,
  });
}

/** Everything one stage of a task has to show: its state, its progress, and the files it made (or reused). */
export function StagePanel({ task, stageId, relaxNode, onSelect }: { task: TaskState; stageId: string; relaxNode: ReactNode; onSelect: (id: string) => void }) {
  const stage = task.stages.find((s) => s.id === stageId);
  const live = task.status === "running" || task.status === "pending";
  const { data } = useTaskResults(task.id, live);
  const { wind } = useProgress(task);
  const base = jobDataBase(task.job_id);
  const [open, setOpen] = useState<number | null>(null);
  const [older, setOlder] = useState(false);

  const all: ResultFile[] = useMemo(() => {
    if (!data) return [];
    const pid = PRODUCT[stageId] ?? stageId;
    return [...data.products.filter((p) => p.id === pid).flatMap((p) => p.files), ...data.plots.filter((p) => p.stage === stageId)];
  }, [data, stageId]);
  // old FLUXlight images from earlier runs would pass for this run's result: show only what this run made unless asked
  const files = stageId === "fluxlight" && !older ? all.filter((f) => f.origin === "made") : all;
  const hidden = all.length - files.length;
  useEffect(() => setOpen(null), [stageId]);

  const fits = stageId === "magnetogram" ? files.find((f) => /\.fits?$/i.test(f.name)) : undefined;
  const floc = stageId === "footpoints" ? files.find((f) => /^floc_.*\.dat$/i.test(f.name)) : undefined;
  const { data: pts } = useFootpoints(floc, base);
  const images = files.filter((f) => IMG.test(f.name));
  const others = files.filter((f) => !IMG.test(f.name));
  const show = (f: ResultFile) => setOpen(files.indexOf(f));

  return (
    <section className="space-y-4" aria-label={`${stage?.label ?? stageId} stage`}>
      <Card className="space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          {stage && <StatusIcon status={stage.status} className="h-4 w-4" />}
          <h3 className="text-[15px]">{stage?.label ?? stageId}</h3>
          {stage && (
            <span className="num text-xs text-muted">
              {stage.status === "pending" ? "not started" : stage.status}
              {stage.elapsed_s != null && stage.status !== "pending" && ` · ${duration(stage.elapsed_s)}`}
            </span>
          )}
          {stage && stage.source !== "none" && <Badge>{stage.source === "event" ? "from events" : stage.source === "log" ? "from the log" : "from files"}</Badge>}
        </div>
        <p className="text-xs text-muted">{ABOUT[stageId]}</p>
        {stage?.detail && <p className={cn("text-[13px]", stage.status === "failed" && "text-fail")}>{stage.detail}</p>}
        {stage?.verified === false && <p className="text-xs text-run">Events say it finished, but its output file was not found.</p>}

        {stageId === "wind" && wind && (
          <div className="space-y-1.5">
            <div className="h-1.5 rounded bg-raised/70">
              <div className="h-full rounded bg-gradient-to-r from-run/70 to-run" style={{ width: `${wind.frac * 100}%` }} />
            </div>
            <p className="num break-words font-mono text-[12px] text-ink/85">{wind.line}</p>
          </div>
        )}
        {stageId === "footpoints" && (
          <p className="text-xs text-muted">
            The footpoint topology figure is drawn by the Map stage.{" "}
            <button className="font-medium text-accent hover:underline" onClick={() => onSelect("fieldmap")}>
              Show it
            </button>
          </p>
        )}
      </Card>

      {stageId === "relax" && relaxNode}
      {stageId === "world_plots" && <RelaxViews task={task} />}
      {fits && (
        <Card className="overflow-hidden bg-black p-0">
          <button className="block w-full" onClick={() => show(fits)} aria-label={`Open ${fits.name}`}>
            <img src={`${base}/render.png?path=${encodeURIComponent(fits.path)}`} alt={`Rendered ${fits.name}`} className="mx-auto max-h-[420px] w-full object-contain" />
          </button>
        </Card>
      )}
      {stageId === "footpoints" && pts && pts.length > 1 && (
        <Card className="p-4">
          <HilbertReplay pts={pts} />
        </Card>
      )}

      {files.length === 0 ? (
        !["relax", "world_plots"].includes(stageId) && (
          <p className="flex items-center gap-1.5 px-1 text-[12.5px] text-faint">
            <CircleDashed className="h-3.5 w-3.5" /> {EMPTY[stage?.status ?? "pending"]}
          </p>
        )
      ) : (
          <div className="space-y-3">
            {images.length > 0 && stageId !== "world_plots" && (
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
                {images.map((f) => (
                  <button key={f.path} onClick={() => show(f)} className="group overflow-hidden rounded-xl border border-line/80 bg-bg/50 text-left transition-colors hover:border-accent/50">
                    <div className="grid aspect-[4/3] place-items-center overflow-hidden bg-black/40">
                      <img src={fileUrl(f.path, false, base)} alt={f.name} loading="lazy" className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]" />
                    </div>
                    <div className="space-y-1 px-2.5 py-2">
                      <p className="truncate text-xs text-muted" title={f.path}>{f.name}</p>
                      <OriginTag origin={f.origin} />
                    </div>
                  </button>
                ))}
              </div>
            )}
            {others.length > 0 && (
              <Card className="divide-y divide-line/60 p-1">
                {others.map((f) => (
                  <button key={f.path} onClick={() => show(f)} className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 rounded-md px-3 py-1.5 text-left hover:bg-raised/60">
                    <span className="min-w-0 truncate font-mono text-[12px]" title={f.path}>{f.name}</span>
                    <OriginTag origin={f.origin} />
                    <span className="num ml-auto text-[11.5px] text-faint">{bytes(f.size)} · {ago(f.mtime)}</span>
                  </button>
                ))}
              </Card>
            )}
          </div>
      )}
      {hidden > 0 && (
        <p className="px-1 text-xs text-muted">
          {hidden} older image{hidden === 1 ? "" : "s"} from earlier runs hidden (this run made none).{" "}
          <button className="font-medium text-accent hover:underline" onClick={() => setOlder(true)}>
            Show them
          </button>
        </p>
      )}
      {older && stageId === "fluxlight" && (
        <p className="px-1 text-xs text-muted">
          Showing older images too.{" "}
          <button className="font-medium text-accent hover:underline" onClick={() => setOlder(false)}>
            Hide them
          </button>
        </p>
      )}

      {open != null && files[open] && data?.data_dir && (
        <FileViewer records={files as CacheRecord[]} index={open} onIndex={setOpen} onClose={() => setOpen(null)} dataDir={data.data_dir} base={base} mode="popup" extraMeta={(r) => <OriginTag origin={(r as ResultFile).origin} />} />
      )}
    </section>
  );
}
