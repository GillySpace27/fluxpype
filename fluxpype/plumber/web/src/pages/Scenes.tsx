import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Boxes, ChevronDown, Play, Plus } from "lucide-react";
import { lazy, Suspense, useState } from "react";
import { Link } from "react-router-dom";
import { Badge, Card, EmptyState, ErrorNote, PageHeader, Skeleton } from "../components/ui";
import { Cover, coverUrl } from "../components/Cover";
import { cn } from "../lib/cn";
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
      api.post<{ id?: string; job_id?: string }>("/api/scene-jobs", {
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
        {go.data && (go.data.id ?? go.data.job_id) && (
          <Link className="text-xs underline" to={`/jobs/${go.data.id ?? go.data.job_id}`}>
            Queued: open job {go.data.id ?? go.data.job_id}
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
  if (!open)
    return (
      <button onClick={onToggle} aria-expanded={false} className="card lift group flex flex-col overflow-hidden text-left focus:outline-none animate-rise">
        <Cover src={coverUrl.scene(s.id, s.modified)} seed={s.id} fit="contain" className="zoom aspect-[16/9] w-full border-b border-line/60">
          <div className="absolute left-2 top-2 flex gap-1.5">
            <Badge className="bg-black/55 text-white/85 backdrop-blur-sm">planar arcade</Badge>
          </div>
        </Cover>
        <div className="flex w-full flex-1 flex-col gap-1 p-3">
          <div className="flex items-center justify-between gap-2">
            <p className="truncate text-[14px] font-semibold tracking-tight">{s.id}</p>
            <span className="font-mono text-[10.5px] text-faint">{s.hash.slice(0, 8)}</span>
          </div>
          <p className="text-xs text-muted">
            {s.lines} lines · {s.lid_height != null ? `lid at z = ${s.lid_height}` : "no lid"} · saved {ago(s.modified)}
          </p>
          {s.notes && <p className="line-clamp-2 text-[12.5px] text-ink/85">{s.notes}</p>}
          <Badge tone="info" className="mt-auto self-start">connectivity hand-set</Badge>
        </div>
      </button>
    );
  return (
    <Card className={cn("col-span-full p-4 animate-fade-in")}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-2.5 text-[14px] font-semibold">
          {s.id}
          <Badge>planar arcade</Badge>
          <Badge tone="info">connectivity hand-set</Badge>
        </p>
        <button className="btn btn-sm" onClick={onToggle}>
          <ChevronDown className="h-3.5 w-3.5 rotate-180" /> Close
        </button>
      </div>
      <p className="mt-1 text-xs text-muted">
        {s.lines} lines · {s.lid_height != null ? `lid at z = ${s.lid_height}` : "no lid"} · saved {ago(s.modified)} · hash {s.hash}
      </p>
      {s.notes && <p className="mt-2 text-[13px]">{s.notes}</p>}
      {(
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

interface ValidateOut {
  ok: boolean;
  problem?: string;
  name?: string;
  content_hash?: string;
  report?: Record<string, unknown>;
  fair_pair_with?: { scene: string; fair: boolean; why_not: string[] | string | null };
  note?: string;
}
interface PreviewOut {
  name: string;
  lines: number;
  rows: Record<string, unknown>[];
  [k: string]: unknown;
}

const STARTER = JSON.stringify({ name: "my-arcade", n_lines: 4, lid_height: 4 }, null, 2);

/** Write or paste a scene spec (the same JSON as create_scene), check it and see its compiled world, then save it.
 *  Validate and Preview are read-only and cost no Claude turn. */
function NewScene({ others }: { others: SceneRow[] }) {
  const qc = useQueryClient();
  const [text, setText] = useState(STARTER);
  const [cmp, setCmp] = useState("");
  let spec: Record<string, unknown> | null = null;
  let parseErr = "";
  try {
    const v = JSON.parse(text);
    if (v && typeof v === "object" && !Array.isArray(v)) spec = v;
    else parseErr = "A scene is a JSON object.";
  } catch (e) {
    parseErr = e instanceof Error ? e.message : String(e);
  }
  const check = useMutation({ mutationFn: () => api.post<ValidateOut>("/api/scenes/validate", { spec, ...(cmp ? { compare_with: cmp } : {}) }) });
  const view = useMutation({ mutationFn: () => api.post<PreviewOut>("/api/scenes/preview", { spec }) });
  const save = useMutation({
    mutationFn: () => api.post<{ id: string }>("/api/scenes", { spec }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["scenes"] }),
  });
  const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));
  return (
    <Card className="mb-3 p-4">
      <p className="text-sm font-medium">New scene</p>
      <p className="mt-0.5 text-xs text-muted">Edit the scene file, check it and look at the compiled world. Nothing is saved until you press Save.</p>
      <textarea
        aria-label="Scene file (JSON)"
        className="input mt-2 h-48 w-full font-mono text-[12px]"
        spellCheck={false}
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      {parseErr && <p className="text-xs text-fail">Not valid JSON: {parseErr}</p>}
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <label className="text-[11px] text-muted">
          Fair pair with{" "}
          <select className="input h-7 text-[12px]" value={cmp} onChange={(e) => setCmp(e.target.value)}>
            <option value="">none</option>
            {others.map((o) => (
              <option key={o.id} value={o.id}>
                {o.id}
              </option>
            ))}
          </select>
        </label>
        <button className="btn btn-sm" disabled={!spec || check.isPending} onClick={() => check.mutate()}>
          Validate
        </button>
        <button className="btn btn-sm" disabled={!spec || view.isPending} onClick={() => view.mutate()}>
          Preview
        </button>
        <button className="btn btn-sm" disabled={!spec || save.isPending} onClick={() => save.mutate()}>
          Save
        </button>
      </div>
      {check.data &&
        (check.data.ok ? (
          <div className="mt-2 text-xs">
            <p className="text-ok">Valid: {check.data.name}</p>
            <pre className="mt-1 max-h-48 overflow-auto rounded bg-raised p-2 text-[11.5px]">{JSON.stringify(check.data.report, null, 1)}</pre>
            {check.data.fair_pair_with && (
              <p className="mt-1">
                {check.data.fair_pair_with.fair ? "A fair pair" : "Not a fair pair"} with {check.data.fair_pair_with.scene}
                {check.data.fair_pair_with.fair ? "." : `: ${JSON.stringify(check.data.fair_pair_with.why_not)}`}
              </p>
            )}
            {check.data.note && <p className="mt-1 text-muted">{check.data.note}</p>}
          </div>
        ) : (
          <p className="mt-2 text-xs text-fail">{check.data.problem}</p>
        ))}
      {check.isError && <p className="mt-2 text-xs text-fail">{msg(check.error)}</p>}
      {view.data && <pre className="mt-2 max-h-72 overflow-auto rounded bg-raised p-2 text-[11.5px]">{JSON.stringify(view.data, null, 1)}</pre>}
      {view.isError && <p className="mt-2 text-xs text-fail">{msg(view.error)}</p>}
      {save.data && <p className="mt-2 text-xs text-ok">Saved as {save.data.id}.</p>}
      {save.isError && <p className="mt-2 text-xs text-fail">{msg(save.error)}</p>}
    </Card>
  );
}

export function ScenesPage() {
  const { data, error, refetch } = useQuery({ queryKey: ["scenes"], queryFn: () => api.get<{ scenes: SceneRow[] }>("/api/scenes"), refetchInterval: 10_000 });
  const [open, setOpen] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  return (
    <div className="animate-fade-in">
      <PageHeader title="Scenes" sub="Planar worlds saved from the Ask bar, the form below or the API: a scene file and its compiled .flux, in the data folder." />
      <button className="btn btn-sm mb-3" onClick={() => setCreating(!creating)} aria-expanded={creating}>
        <Plus className="h-3.5 w-3.5" /> New scene
      </button>
      {creating && <NewScene others={data?.scenes ?? []} />}
      {!data ? (
        <Skeleton className="h-24" />
      ) : data.scenes.length === 0 ? (
        <EmptyState icon={<Boxes className="h-5 w-5" />} title="No scenes yet" body="Press New scene to write one, or ask Plum for a planar arcade over a polarity inversion line and run the card she proposes." />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {data.scenes.map((s) => (
            <SceneCard key={s.id} s={s} open={open === s.id} onToggle={() => setOpen(open === s.id ? null : s.id)} others={data.scenes.filter((o) => o.id !== s.id)} />
          ))}
        </div>
      )}
    </div>
  );
}
