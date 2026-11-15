import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Boxes, Play } from "lucide-react";
import { lazy, Suspense, useState } from "react";
import { Link } from "react-router-dom";
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

/** Queue a relaxation of this scene (or a fair pair with another, at one or several lid heights) as an ordinary job. */
function RelaxForm({ s, others }: { s: SceneRow; others: SceneRow[] }) {
  const qc = useQueryClient();
  const [other, setOther] = useState("");
  const [lids, setLids] = useState("");
  const [cycles, setCycles] = useState("150");
  const [steps, setSteps] = useState("100");
  const heights = lids.split(/[\s,]+/).filter(Boolean).map(Number);
  const badLids = heights.some((h) => !(h > 0));
  const go = useMutation({
    mutationFn: () =>
      api.post<{ job_id: string }>("/api/scene-jobs", {
        ...(other ? { mode: "compare", a: s.id, b: other } : { mode: "run", scene: s.id }),
        ...(heights.length ? { lids: heights } : {}),
        cycles: Number(cycles),
        steps: Number(steps),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["scenes"] });
    },
  });
  const field = "input h-7 w-full text-[12px]";
  return (
    <div className="rounded-lg border border-line p-3">
      <p className="text-xs font-medium">Relax in the queue</p>
      <div className="mt-2 grid gap-2 sm:grid-cols-4">
        <label className="text-[11px] text-muted sm:col-span-2">
          Compare with (empty: this scene alone)
          <select className={field} value={other} onChange={(e) => setOther(e.target.value)}>
            <option value="">none</option>
            {others.map((o) => (
              <option key={o.id} value={o.id}>
                {o.id}
              </option>
            ))}
          </select>
        </label>
        <label className="text-[11px] text-muted">
          Cycles
          <input className={field} inputMode="numeric" value={cycles} onChange={(e) => setCycles(e.target.value)} />
        </label>
        <label className="text-[11px] text-muted">
          Steps per cycle
          <input className={field} inputMode="numeric" value={steps} onChange={(e) => setSteps(e.target.value)} />
        </label>
        <label className="text-[11px] text-muted sm:col-span-4">
          Lid heights to sweep, comma separated (empty: the lid as saved). Each makes a saved variant scene.
          <input className={field} placeholder="for example 8, 12, 16" value={lids} onChange={(e) => setLids(e.target.value)} />
        </label>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-3">
        <button className="btn btn-sm btn-primary" onClick={() => go.mutate()} disabled={go.isPending || badLids || !(Number(cycles) >= 1) || !(Number(steps) >= 1)}>
          <Play className="h-3 w-3" /> {go.isPending ? "Queuing..." : "Relax"}
        </button>
        {go.data && (
          <Link className="text-xs underline" to={`/jobs/${go.data.job_id}`}>
            Queued: open job {go.data.job_id}
          </Link>
        )}
        {badLids && <span className="text-xs text-fail">Lid heights are positive numbers.</span>}
        {go.isError && <span className="text-xs text-fail">{go.error instanceof Error ? go.error.message : String(go.error)}</span>}
      </div>
    </div>
  );
}

function SceneCard({ s, open, onToggle, others }: { s: SceneRow; open: boolean; onToggle: () => void; others: SceneRow[] }) {
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
          <RelaxForm s={s} others={others} />
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
            <SceneCard key={s.id} s={s} open={open === s.id} onToggle={() => setOpen(open === s.id ? null : s.id)} others={data.scenes.filter((o) => o.id !== s.id)} />
          ))}
        </div>
      )}
    </div>
  );
}
