import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, ChevronRight, Layers, Play, SlidersHorizontal } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { Badge, Banner, Card, Checkbox, Dialog, EmptyState, ErrorNote, PageHeader, Skeleton, Tip, useToast } from "../components/ui";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { instOf } from "../lib/format";
import { useProfile, useProfiles, useStatus } from "../lib/hooks";
import type { Job, ProfileDetail, ProfileEntry } from "../lib/types";

const CACHE_LABEL: Record<string, string> = { magnetogram: "Mag", footpoints: "Foot", pfss: "PFSS", world: "World", relaxed_world: "Relaxed" };

/** The knobs worth changing per run, shown first in the run drawer. */
const COMMON = ["rotations", "fluxon_count", "adapts", "flow_method", "max_cycles", "relax_threshold", "do_steps", "light", "do_wind_calc"];

function RunDrawer({ detail, open, onOpenChange }: { detail: ProfileDetail; open: boolean; onOpenChange: (o: boolean) => void }) {
  const { data: status } = useStatus();
  const nav = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [picked, setPicked] = useState<Set<number>>(() => new Set(detail.tasks.map((_, i) => i)));
  const [scenario, setScenario] = useState("ok");
  const [speed, setSpeed] = useState("20");
  const [showAll, setShowAll] = useState(false);

  const demo = !!status?.demo;
  const editable = detail.entries.filter((e) => e.editable);
  const common = editable.filter((e) => COMMON.includes(e.key));
  const rest = editable.filter((e) => !COMMON.includes(e.key));
  const changed = Object.entries(overrides).filter(([k, v]) => v !== detail.entries.find((e) => e.key === k)?.value);
  const allPicked = picked.size === detail.tasks.length;

  const run = useMutation({
    mutationFn: () => {
      const body: Record<string, unknown> = { profile: detail.name, overrides: Object.fromEntries(changed) };
      if (!allPicked) body.tasks = [...picked].sort((a, b) => a - b).map((i) => [detail.tasks[i].cr, detail.tasks[i].nflux, detail.tasks[i].adapt, detail.tasks[i].method]);
      if (demo) {
        body.backend = "fake";
        body.scenario = scenario;
        body.speed = Number(speed) || 20;
      }
      return api.post<Job>("/api/jobs", body);
    },
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["overview"] });
      toast("ok", `Started ${job.tasks.length} task${job.tasks.length === 1 ? "" : "s"}`);
      onOpenChange(false);
      nav(`/jobs/${job.id}`);
    },
    onError: (e) => toast("error", e instanceof Error ? e.message : String(e)),
  });

  const field = (e: ProfileEntry) => (
    <label key={e.key} className="block">
      <span className="flex items-baseline justify-between text-xs">
        <span className="font-mono text-ink/90">{e.key}</span>
        {overrides[e.key] !== undefined && overrides[e.key] !== e.value && <span className="text-accent">changed</span>}
      </span>
      <input className="input mt-1 font-mono" value={overrides[e.key] ?? e.value} onChange={(ev) => setOverrides({ ...overrides, [e.key]: ev.target.value })} spellCheck={false} />
      {e.comment && <span className="mt-1 block text-[11px] leading-snug text-faint">{e.comment}</span>}
    </label>
  );

  return (
    <Dialog
      side
      open={open}
      onOpenChange={onOpenChange}
      title={`Run ${detail.name}`}
      description="Overrides apply to this run only. The saved profile file is not touched."
      footer={
        <>
          <span className="mr-auto text-xs text-muted">
            {picked.size} task{picked.size === 1 ? "" : "s"}
            {changed.length > 0 && ` · ${changed.length} override${changed.length === 1 ? "" : "s"}`}
          </span>
          <button className="btn" onClick={() => onOpenChange(false)}>
            Cancel
          </button>
          <button className="btn btn-primary" disabled={picked.size === 0 || run.isPending} onClick={() => run.mutate()}>
            <Play className="h-3.5 w-3.5" /> {run.isPending ? "Starting..." : "Start run"}
          </button>
        </>
      }
    >
      <div className="space-y-6">
        {demo && (
          <Banner tone="info" title="Demo mode">
            Tasks run on a simulated pipeline. Pick a scenario to see how the UI reacts to problems.
          </Banner>
        )}
        {demo && (
          <section className="grid grid-cols-[1fr_96px] gap-3">
            <label className="block text-xs">
              <span className="label">Scenario</span>
              <select className="input mt-1" value={scenario} onChange={(e) => setScenario(e.target.value)}>
                {(status?.scenarios ?? ["ok"]).map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-xs">
              <span className="label">Speed x</span>
              <input className="input mt-1 num" value={speed} onChange={(e) => setSpeed(e.target.value)} inputMode="decimal" />
            </label>
          </section>
        )}

        <section>
          <div className="mb-2 flex items-center justify-between">
            <p className="label">Tasks</p>
            <button className="text-xs text-muted hover:text-ink" onClick={() => setPicked(allPicked ? new Set() : new Set(detail.tasks.map((_, i) => i)))}>
              {allPicked ? "Select none" : "Select all"}
            </button>
          </div>
          <div className="max-h-56 divide-y divide-line/50 overflow-y-auto rounded-lg border border-line">
            {detail.tasks.map((t, i) => (
              <div
                key={i}
                role="button"
                tabIndex={0}
                onClick={() => setPicked((p) => (p.has(i) ? new Set([...p].filter((x) => x !== i)) : new Set(p).add(i)))}
                onKeyDown={(ev) => ev.key === " " && ev.preventDefault()}
                className="flex cursor-pointer items-center gap-3 px-3 py-2 text-[13px] hover:bg-raised/50"
              >
                <Checkbox
                  checked={picked.has(i)}
                  label={`CR ${t.cr}`}
                  onChange={() => setPicked((p) => (p.has(i) ? new Set([...p].filter((x) => x !== i)) : new Set(p).add(i)))}
                />
                <span className="flex-1">
                  CR {t.cr} <span className="text-muted">· {t.nflux} · {instOf(t.adapt)} · {t.method}</span>
                </span>
                {t.cache.relaxed_world && <Badge tone="ok">cached</Badge>}
              </div>
            ))}
          </div>
        </section>

        <section className="space-y-3">
          <p className="label">Settings for this run</p>
          {common.map(field)}
          {rest.length > 0 && (
            <>
              <button className="flex items-center gap-1 text-xs text-muted hover:text-ink" onClick={() => setShowAll(!showAll)}>
                <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", showAll && "rotate-90")} />
                {showAll ? "Hide" : "Show"} {rest.length} more settings
              </button>
              {showAll && <div className="space-y-3">{rest.map(field)}</div>}
            </>
          )}
        </section>
      </div>
    </Dialog>
  );
}

function ProfileView({ name }: { name: string }) {
  const { data, error, refetch } = useProfile(name);
  const [open, setOpen] = useState(false);
  const [showDefault, setShowDefault] = useState(false);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  if (!data) return <Skeleton className="h-96" />;
  const own = data.entries.filter((e) => e.source === "profile");
  const shown = showDefault ? data.entries : own;
  return (
    <div className="animate-fade-in space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <Link to="/profiles" className="mb-2 inline-flex items-center gap-1 text-xs text-muted hover:text-ink md:hidden">
            <ArrowLeft className="h-3.5 w-3.5" /> Profiles
          </Link>
          <h1 className="text-[22px]">{data.name}</h1>
          <p className="mt-1 text-[13px] text-muted">
            {data.task_count} task{data.task_count === 1 ? "" : "s"} · {data.slots} CPU slot{data.slots === 1 ? "" : "s"} each
          </p>
        </div>
        <button className="btn btn-primary" onClick={() => setOpen(true)} disabled={data.errors.length > 0 || data.task_count === 0}>
          <SlidersHorizontal className="h-4 w-4" /> Configure and run
        </button>
      </div>

      {data.errors.map((e) => (
        <Banner key={e} tone="error">{e}</Banner>
      ))}
      {data.warnings.map((e) => (
        <Banner key={e} tone="warn">{e}</Banner>
      ))}

      <section>
        <div className="mb-2 flex items-end justify-between">
          <h2 className="text-[15px]">Tasks</h2>
          {data.truncated && <span className="text-xs text-muted">showing the first {data.tasks.length}</span>}
        </div>
        <Card className="overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-[13px]">
              <thead>
                <tr className="border-b hair text-left">
                  {["Rotation", "Fluxons", "Source", "Method", "Already cached"].map((h) => (
                    <th key={h} className="label px-4 py-2.5 font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.tasks.map((t, i) => (
                  <tr key={i} className="border-b border-line/40 last:border-0">
                    <td className="num px-4 py-2 font-medium">CR {t.cr}</td>
                    <td className="num px-4 py-2">{t.nflux}</td>
                    <td className="px-4 py-2">{instOf(t.adapt)}</td>
                    <td className="px-4 py-2 text-muted">{t.method}</td>
                    <td className="px-4 py-2">
                      <div className="flex gap-1">
                        {Object.entries(CACHE_LABEL).map(([k, l]) => (
                          <Tip key={k} label={`${l}: ${(t.cache as Record<string, boolean>)[k] ? "found on disk" : "not on disk"}`}>
                            <span className={cn("rounded px-1.5 py-0.5 text-[10px] font-medium", (t.cache as Record<string, boolean>)[k] ? "bg-ok/12 text-ok" : "bg-raised text-faint")}>{l}</span>
                          </Tip>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </section>

      <section>
        <div className="mb-2 flex items-end justify-between">
          <h2 className="text-[15px]">Settings</h2>
          <button className="text-xs text-muted hover:text-ink" onClick={() => setShowDefault(!showDefault)}>
            {showDefault ? "Only this profile's settings" : `Show all ${data.entries.length} (including DEFAULT)`}
          </button>
        </div>
        <Card className="divide-y divide-line/40">
          {shown.length === 0 && <p className="px-4 py-3 text-[13px] text-muted">This profile inherits everything from DEFAULT.</p>}
          {shown.map((e) => (
            <div key={e.key} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] items-baseline gap-3 px-4 py-2 text-[13px] sm:grid-cols-[200px_minmax(0,1fr)_auto]">
              <span className="truncate font-mono text-xs text-ink/90">{e.key}</span>
              <span className="truncate font-mono text-xs text-muted" title={e.resolved ?? e.value}>
                {e.value}
                {e.resolved && e.resolved !== e.value && <span className="text-faint"> → {e.resolved}</span>}
              </span>
              <span className="hidden sm:block">{e.source === "profile" ? e.overrides_default ? <Badge tone="accent">overrides</Badge> : <Badge>profile</Badge> : <Badge>default</Badge>}</span>
            </div>
          ))}
        </Card>
      </section>
      {open && <RunDrawer detail={data} open={open} onOpenChange={setOpen} />}
    </div>
  );
}

export function ProfilesPage() {
  const { name } = useParams();
  const { data, error, refetch } = useProfiles();
  const profiles = useMemo(() => data?.profiles ?? [], [data]);
  if (error) return <ErrorNote error={error} retry={() => refetch()} />;
  const list = (
    <div className="space-y-2">
      {!data && [0, 1, 2].map((i) => <Skeleton key={i} className="h-[76px]" />)}
      {profiles.map((p) => (
        <Link key={p.name} to={`/profiles/${encodeURIComponent(p.name)}`} className="block focus:outline-none">
          <Card className={cn("p-3.5 transition-colors hover:border-faint/50", name === p.name && "border-accent/50 bg-raised/40")}>
            <div className="flex items-center justify-between gap-2">
              <p className="truncate text-[13.5px] font-semibold">{p.name}</p>
              {p.errors.length > 0 ? <Badge tone="fail">error</Badge> : <Badge>{p.task_count} task{p.task_count === 1 ? "" : "s"}</Badge>}
            </div>
            <p className="mt-1 truncate text-xs text-muted">
              {p.axes.rotations?.length ? `CR ${p.axes.rotations.slice(0, 3).join(", ")}${p.axes.rotations.length > 3 ? "..." : ""}` : "no rotations"} · {p.axes.fluxon_count?.join(", ") ?? "?"} fluxons
            </p>
          </Card>
        </Link>
      ))}
    </div>
  );
  return (
    <div className="animate-fade-in">
      {!name && <PageHeader title="Profiles" sub="Each profile expands to a grid of tasks: rotations x fluxon counts x sources x methods." />}
      <div className="grid gap-6 md:grid-cols-[280px_minmax(0,1fr)]">
        <div className={cn(name && "hidden md:block")}>{list}</div>
        <div className={cn(!name && "hidden md:block")}>
          {name ? <ProfileView name={name} /> : <EmptyState icon={<Layers className="h-5 w-5" />} title="Select a profile" body="Choose one on the left to see the tasks it expands to and what is already cached." />}
        </div>
      </div>
    </div>
  );
}
