import { useQuery } from "@tanstack/react-query";
import { Boxes } from "lucide-react";
import { lazy, Suspense, useState } from "react";
import { Badge, Card, EmptyState, ErrorNote, PageHeader, Skeleton } from "../components/ui";
import { api } from "../lib/api";
import { ago } from "../lib/format";

const FluxViewer = lazy(() => import("../components/FluxViewer").then((m) => ({ default: m.FluxViewer }))); // three.js

interface SceneRow {
  id: string;
  hash: string;
  world: string;
  lines: number;
  lid_height: number | null;
  notes: string;
  modified: number;
}
interface SceneDoc {
  id: string;
  hash: string;
  world: string;
  spec: Record<string, unknown>;
}

function SceneCard({ s, open, onToggle }: { s: SceneRow; open: boolean; onToggle: () => void }) {
  const doc = useQuery({ queryKey: ["scene", s.id], queryFn: () => api.get<SceneDoc>(`/api/scenes/${encodeURIComponent(s.id)}`), enabled: open });
  const spec = doc.data?.spec ?? {};
  return (
    <Card className="p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-2.5 text-[14px] font-semibold">
          {s.id}
          <Badge>planar arcade</Badge>
          <Badge tone="info">connectivity hand-set</Badge>
        </p>
        <button className="btn btn-sm" onClick={onToggle}>
          {open ? "Hide" : "Open"}
        </button>
      </div>
      <p className="mt-1 text-xs text-muted">
        {s.lines} lines · {s.lid_height != null ? `lid at z = ${s.lid_height}` : "no lid"} · saved {ago(s.modified)} · hash {s.hash}
      </p>
      {s.notes && <p className="mt-2 text-[13px]">{s.notes}</p>}
      {open && (
        <div className="mt-3 space-y-3">
          <div className="h-[420px] overflow-hidden rounded-lg border border-line">
            <Suspense fallback={<Skeleton className="h-full" />}>
              <FluxViewer path={s.world} base="/api/cache" mtime={s.modified} />
            </Suspense>
          </div>
          <details>
            <summary className="cursor-pointer text-xs text-muted">Scene file</summary>
            <pre className="mt-2 max-h-72 overflow-auto rounded bg-raised p-2 text-[11.5px]">{JSON.stringify(spec, null, 1)}</pre>
          </details>
          <p className="text-xs text-muted">
            This is a construction, not a result: nothing has been relaxed here. Relaxed energies of two scenes are only comparable when footpoints, photosphere, lid and vertex counts match, and they depend on the lid height.
          </p>
        </div>
      )}
    </Card>
  );
}

export function ScenesPage() {
  const { data, error, refetch } = useQuery({ queryKey: ["scenes"], queryFn: () => api.get<{ scenes: SceneRow[] }>("/api/scenes"), refetchInterval: 10_000 });
  const [open, setOpen] = useState<string | null>(null);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  return (
    <div className="animate-fade-in">
      <PageHeader title="Scenes" sub="Planar worlds saved from the Ask bar or the API: a scene file and its compiled .flux, in the data folder." />
      {!data ? (
        <Skeleton className="h-24" />
      ) : data.scenes.length === 0 ? (
        <EmptyState icon={<Boxes className="h-5 w-5" />} title="No scenes yet" body="Ask Plum for a planar arcade over a polarity inversion line, and run the card she proposes." />
      ) : (
        <div className="space-y-2.5">
          {data.scenes.map((s) => (
            <SceneCard key={s.id} s={s} open={open === s.id} onToggle={() => setOpen(open === s.id ? null : s.id)} />
          ))}
        </div>
      )}
    </div>
  );
}
