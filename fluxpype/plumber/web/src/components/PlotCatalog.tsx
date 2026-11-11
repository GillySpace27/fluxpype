import { ImageOff } from "lucide-react";
import { cn } from "../lib/cn";
import type { CatalogPlot, ResultFile } from "../lib/types";
import { fileUrl } from "./FileViewer";
import { OriginTag } from "./ResultsPane";
import { Badge, SectionTitle } from "./ui";

const IMG = /\.(png|jpe?g|svg)$/i;

/** One defined plot: its newest file as a thumbnail, or a labelled placeholder when nothing has been rendered. */
function PlotCard({ c, base, visible, onOpen }: { c: CatalogPlot; base: string; visible: (f: ResultFile) => boolean; onOpen: (f: ResultFile) => void }) {
  const shown = c.files.filter(visible);
  const head = shown.find((f) => IMG.test(f.name)) ?? shown[0];
  const hiddenOlder = c.files.length - shown.length;
  return (
    <div className="flex min-w-0 flex-col overflow-hidden rounded-xl border border-line/80 bg-bg/50">
      {head ? (
        <button onClick={() => onOpen(head)} className="group relative grid aspect-[4/3] place-items-center overflow-hidden bg-black/40" aria-label={`Open ${head.name}`}>
          {IMG.test(head.name) ? <img src={fileUrl(head.path, false, base)} alt={head.name} loading="lazy" className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]" /> : <span className="text-xs text-muted">{head.name}</span>}
          {c.n_files > 1 && <span className="num absolute right-1.5 top-1.5 rounded bg-bg/80 px-1.5 py-0.5 text-[10.5px] text-muted">{c.n_files} files</span>}
        </button>
      ) : (
        <div className="grid h-32 place-items-center border-b border-dashed border-line/70 bg-raised/20 px-3 text-center">
          <div className="space-y-1 text-faint">
            <ImageOff className="mx-auto h-5 w-5" />
            <p className="text-[11.5px] font-medium">{hiddenOlder > 0 ? "Only older images exist" : "Not rendered"}</p>
            <p className="text-[10.5px] leading-snug">{c.gate ? `Note: ${c.gate}` : "No file found for this run."}</p>
          </div>
        </div>
      )}
      <div className="space-y-1 px-2.5 py-2">
        <p className="text-[12.5px] font-medium leading-snug">{c.title}</p>
        <p className="line-clamp-3 text-[11.5px] leading-snug text-muted" title={c.summary}>
          {c.summary}
        </p>
        <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
          {head && <OriginTag origin={head.origin} />}
          {c.legacy && <Badge className="px-1.5 text-[10.5px]">not made now</Badge>}
        </div>
        <p className="truncate font-mono text-[10px] text-faint" title={`${c.writer}${c.call ? `\ncalled from ${c.call}` : ""}`}>
          {c.writer}
        </p>
      </div>
    </div>
  );
}

/** Every plot the pipeline defines for a stage, rendered or as a placeholder, with the ones the current pipeline no longer makes folded away. */
export function PlotCatalog({ entries, base, visible, onOpen }: { entries: CatalogPlot[]; base: string; visible: (f: ResultFile) => boolean; onOpen: (f: ResultFile) => void }) {
  if (entries.length === 0) return null;
  const live = entries.filter((c) => !c.legacy);
  const legacy = entries.filter((c) => c.legacy);
  const made = live.filter((c) => c.files.some(visible)).length;
  const grid = "grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3";
  return (
    <section className="space-y-3">
      {live.length > 0 && (
        <>
          <SectionTitle title="Plots this stage defines" hint={`${made} of ${live.length} rendered for this task. Empty cards are plots the pipeline can make but did not for this task; the note says when each one is made.`} />
          <div className={grid}>
            {live.map((c) => (
              <PlotCard key={c.id} c={c} base={base} visible={visible} onOpen={onOpen} />
            ))}
          </div>
        </>
      )}
      {legacy.length > 0 && (
        <details className={cn("rounded-xl border border-line/70 p-3", live.length === 0 && "open:bg-raised/10")} open={live.length === 0}>
          <summary className="cursor-pointer text-[13px] text-muted">
            Defined in the code but not made by the current pipeline <span className="num text-faint">({legacy.length})</span>
          </summary>
          <p className="mt-1 text-xs text-faint">Their calls are commented out, unreachable or manual only. Older files from earlier runs show if they exist.</p>
          <div className={cn("mt-3", grid)}>
            {legacy.map((c) => (
              <PlotCard key={c.id} c={c} base={base} visible={visible} onOpen={onOpen} />
            ))}
          </div>
        </details>
      )}
    </section>
  );
}
