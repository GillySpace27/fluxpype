import { CircleDashed, Database, FileX2, ImageOff } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { cn } from "../lib/cn";
import { ago, bytes, duration, int, num } from "../lib/format";
import { jobDataBase, useTaskResults, useWide } from "../lib/hooks";
import type { CacheRecord, ResultFile, TaskState } from "../lib/types";
import { FileViewer, fileUrl, useViewerMode } from "./FileViewer";
import { Badge, Card, EmptyState, ErrorNote, SectionTitle, Skeleton } from "./ui";

/** Why a product has no file, from its stage's status. */
const EMPTY: Record<string, string> = {
  skipped: "skipped in this run",
  pending: "not reached yet",
  running: "in progress",
  failed: "stage failed",
  ok: "missing (the stage finished but no file matches)",
};

const IMG = /\.(png|jpe?g|svg)$/i;

/** Where a file came from, judged by its clock against the task's run window. */
export function OriginTag({ origin }: { origin: ResultFile["origin"] | undefined }) {
  if (origin === "made") return <Badge tone="ok" className="px-1.5 text-[10.5px]">made by this run</Badge>;
  if (origin === "reused") return <Badge tone="info" className="px-1.5 text-[10.5px]">reused from cache</Badge>;
  if (origin === "later")
    return (
      <span title="Written after this run ended, usually by a later run of the same rotation and fluxon count.">
        <Badge tone="run" className="px-1.5 text-[10.5px]">rewritten later</Badge>
      </span>
    );
  return null;
}

function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="min-w-0">
      <p className="label truncate">{label}</p>
      <p className="num mt-0.5 truncate text-[17px] font-semibold tracking-tight">{value}</p>
      {sub && <p className="truncate text-[11px] text-faint">{sub}</p>}
    </div>
  );
}

/**
 * Everything one task produced, in one place: the headline image with the outcome numbers,
 * then each product by stage (tagged made here or reused), then the other plots.
 * Files open in the shared viewer, reading the job's own data folder.
 */
export function ResultsPane({ task }: { task: TaskState }) {
  const live = task.status === "running" || task.status === "pending";
  const { data, error, refetch } = useTaskResults(task.id, live);
  const [open, setOpen] = useState<number | null>(null);
  const [mode, setMode] = useViewerMode();
  const wide = useWide();
  const base = jobDataBase(task.job_id);

  // one flat list, in display order, so previous/next walks the whole page
  const flat: ResultFile[] = useMemo(() => {
    if (!data) return [];
    return [...data.products.flatMap((p) => p.files), ...data.plots];
  }, [data]);
  const indexOf = useCallback((path: string) => flat.findIndex((f) => f.path === path), [flat]);
  const show = (path: string) => setOpen(indexOf(path));
  const close = useCallback(() => setOpen(null), []);

  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!data) return <Skeleton className="h-96" />;
  if (!data.data_dir) return <EmptyState icon={<FileX2 className="h-5 w-5" />} title="No results to show" body="This job's saved settings could not be read, so Plumber can't tell where its files are." />;

  const head = data.headline ? flat.find((f) => f.path === data.headline) : undefined;
  const nOut = task.metrics.find((m) => m.name === "n_actual");
  const made = flat.filter((f) => f.origin === "made").length;
  const reused = flat.filter((f) => f.origin === "reused").length;
  const later = flat.filter((f) => f.origin === "later").length;
  const elapsed = task.started ? (task.ended ?? Date.now() / 1000) - task.started : task.elapsed_s;
  const inline = open != null && wide && mode === "inline";
  const viewer =
    open != null && flat[open] ? (
      <FileViewer
        records={flat as CacheRecord[]}
        index={open}
        onIndex={setOpen}
        onClose={close}
        dataDir={data.data_dir}
        base={base}
        mode={wide ? mode : "popup"}
        onMode={setMode}
        className="lg:sticky lg:top-20"
        extraMeta={(r) => <OriginTag origin={(r as ResultFile).origin} />}
      />
    ) : null;

  const list = (
    <div className="space-y-6">
      <Card className="overflow-hidden">
        <div className={cn("grid gap-0", head && !inline && "md:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]")}>
          {head ? (
            <button onClick={() => show(head.path)} className="group block bg-black" aria-label={`Open ${head.name}`}>
              <img src={fileUrl(head.path, false, base)} alt={head.name} className="mx-auto max-h-[420px] w-full object-contain transition-opacity group-hover:opacity-90" />
            </button>
          ) : (
            <div className="hidden" />
          )}
          <div className="space-y-4 p-5">
            {head ? (
              <div>
                <p className="label">Headline</p>
                <p className="mt-0.5 break-all font-mono text-[12.5px]">{head.name}</p>
                <div className="mt-1">
                  <OriginTag origin={head.origin} />
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2 text-[13px] text-muted">
                <ImageOff className="h-4 w-4" /> No image yet for this task.
              </div>
            )}
            <div className={cn("grid grid-cols-2 gap-4", inline && "sm:grid-cols-3")}>
              <Stat label="Outcome" value={task.status} sub={task.exit_code != null ? `exit ${task.exit_code}` : undefined} />
              <Stat label="Elapsed" value={duration(elapsed)} />
              <Stat label="Fluxons out" value={nOut ? String(nOut.value) : "-"} sub={`of ${task.nflux} asked`} />
              <Stat label="Final stiffness" value={num(task.relax.latest?.stiffness)} sub={task.relax.threshold != null ? `target ${num(task.relax.threshold)}` : undefined} />
              <Stat label="Relax steps" value={int(task.relax.latest?.step)} sub={task.relax.cycles ? `${task.relax.cycles} cycles` : undefined} />
              <Stat label="Files" value={`${made} made here`} sub={`${reused} reused${later ? ` · ${later} rewritten later` : ""}`} />
            </div>
          </div>
        </div>
      </Card>

      <section>
        <SectionTitle title="Products by stage" hint="What each stage left on disk for this task." />
        <Card className="divide-y divide-line/60">
          {data.products.map((p) => (
            <div key={p.id} className="flex flex-col gap-1.5 px-4 py-3 sm:flex-row sm:items-start">
              <p className="w-36 shrink-0 pt-0.5 text-[13px] font-medium">{p.label}</p>
              <div className="min-w-0 flex-1 space-y-1">
                {p.files.length === 0 && (
                  <p className={cn("flex items-center gap-1.5 text-[12.5px]", p.stage_status === "failed" ? "text-fail" : "text-faint")}>
                    <CircleDashed className="h-3.5 w-3.5" /> {EMPTY[p.stage_status ?? ""] ?? "missing"}
                  </p>
                )}
                {p.files.map((f) => (
                  <button key={f.path} onClick={() => show(f.path)} className={cn("flex w-full flex-wrap items-center gap-x-3 gap-y-1 rounded-md px-2 py-1 text-left hover:bg-raised/60", open != null && flat[open]?.path === f.path && "bg-raised")}>
                    <span className="min-w-0 truncate font-mono text-[12px]" title={f.path}>
                      {f.name}
                    </span>
                    <OriginTag origin={f.origin} />
                    <span className="num ml-auto text-[11.5px] text-faint">
                      {bytes(f.size)} · {ago(f.mtime)}
                    </span>
                  </button>
                ))}
                {p.more > 0 && <p className="px-2 text-[11.5px] text-faint">and {p.more} more (see the Cache page)</p>}
              </div>
            </div>
          ))}
        </Card>
      </section>

      <section>
        <SectionTitle
          title="Plots"
          hint="Other images and PDFs under this rotation's batch folder for this fluxon count."
          right={
            <Link to={`/cache?cr=${task.cr}`} className="inline-flex items-center gap-1 text-xs text-muted hover:text-ink">
              <Database className="h-3.5 w-3.5" /> All files for CR {task.cr}
            </Link>
          }
        />
        {data.plots.length === 0 ? (
          <EmptyState icon={<ImageOff className="h-5 w-5" />} title="No plots" body="Plots show up here as the pipeline writes them." />
        ) : (
          <div className={cn("grid grid-cols-2 gap-3 sm:grid-cols-3", !inline && "xl:grid-cols-4")}>
            {data.plots.map((f) => (
              <button key={f.path} onClick={() => show(f.path)} className={cn("group overflow-hidden rounded-xl border bg-bg/50 text-left transition-colors hover:border-accent/50", open != null && flat[open]?.path === f.path ? "border-accent/60" : "border-line/80")}>
                <div className="grid aspect-[4/3] place-items-center overflow-hidden bg-black/40">
                  {IMG.test(f.name) ? (
                    <img src={fileUrl(f.path, false, base)} alt={f.name} loading="lazy" className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]" />
                  ) : (
                    <span className="text-xs text-muted">PDF</span>
                  )}
                </div>
                <div className="space-y-1 px-2.5 py-2">
                  <p className="truncate text-xs text-muted" title={f.path}>
                    {f.name}
                  </p>
                  <OriginTag origin={f.origin} />
                </div>
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );

  return (
    <>
      <div className={cn(inline && "grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]")}>
        {list}
        {inline && viewer}
      </div>
      {open != null && !inline && viewer}
    </>
  );
}

// world plots: .../imgs/world/<n>_<zoom>/cr2150_f1000_<initial|relaxed>_<zoom>.png
const WORLD_VIEW = /\/imgs\/world\/([^/]+)\/[^/]*_(initial|relaxed)_[^/]*\.(?:png|jpe?g)$/i;

/** The world before and after relaxation, side by side, with a switch between the zoom levels the pipeline plots. */
export function RelaxViews({ task }: { task: TaskState }) {
  const live = task.status === "running" || task.status === "pending";
  const { data } = useTaskResults(task.id, live);
  const base = jobDataBase(task.job_id);
  const [sel, setSel] = useState(0);
  const views = useMemo(() => {
    const m = new Map<string, Partial<Record<"initial" | "relaxed", ResultFile>>>();
    for (const f of data?.plots ?? []) {
      const r = WORLD_VIEW.exec("/" + f.path);
      if (r) m.set(r[1], { ...m.get(r[1]), [r[2].toLowerCase()]: f });
    }
    return Array.from(m.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [data]);
  if (!views.length) return null;
  const [, pair] = views[Math.min(sel, views.length - 1)];
  const label = (n: string) => n.replace(/^\d+_/, "");
  return (
    <section>
      <SectionTitle
        title="Before and after relaxation"
        hint="The world plots the pipeline wrote at each zoom level. Click an image to open it full size."
        right={
          <div className="inline-flex rounded-lg border border-line bg-bg/50 p-0.5" role="radiogroup" aria-label="Zoom level">
            {views.map(([n], i) => (
              <button
                key={n}
                role="radio"
                aria-checked={i === Math.min(sel, views.length - 1)}
                onClick={() => setSel(i)}
                className={cn("rounded-md px-2.5 py-1 text-xs font-medium capitalize transition-colors", i === Math.min(sel, views.length - 1) ? "bg-raised text-ink shadow-sm" : "text-muted hover:text-ink")}
              >
                {label(n)}
              </button>
            ))}
          </div>
        }
      />
      <Card className="grid gap-4 p-4 md:grid-cols-2">
        {(["initial", "relaxed"] as const).map((k) => {
          const f = pair[k];
          return (
            <figure key={k} className="min-w-0">
              <figcaption className="mb-2 flex items-center gap-2 text-[13px] font-medium">
                {k === "initial" ? "Before (initial world)" : "After (relaxed world)"}
                <OriginTag origin={f?.origin} />
              </figcaption>
              {f ? (
                <a href={fileUrl(f.path, false, base)} target="_blank" rel="noreferrer" className="block overflow-hidden rounded-lg border border-line/70 bg-bg/40">
                  <img src={fileUrl(f.path, false, base)} alt={f.name} loading="lazy" className="max-h-[460px] w-full object-contain" />
                </a>
              ) : (
                <p className="rounded-lg border border-dashed border-line/70 px-3 py-10 text-center text-xs text-faint">no {k} plot at this zoom level</p>
              )}
            </figure>
          );
        })}
      </Card>
    </section>
  );
}
