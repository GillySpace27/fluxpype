import type { Job } from "../lib/types";
import { ArrowLeft, Play, X } from "lucide-react";
import { Badge, Banner, Card, EmptyState, ErrorNote, Skeleton, useToast } from "../components/ui";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { RunForm, changedOverrides, emptyDraft, jobBody, pickedTasks, taskKey, type RunDraft } from "../components/RunForm";
import { api } from "../lib/api";
import { loadDraft, saveDraft } from "../lib/runDraft";
import { useCallback, useEffect, useMemo, useState } from "react";
import { useJob, useJobConfig, useProfile, useProfiles, useStatus } from "../lib/hooks";
import { useMutation, useQueryClient } from "@tanstack/react-query";

const enc = encodeURIComponent;
const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/* ------------------------------------------------------------------ a run before launch */
interface StoredDraft extends RunDraft {
  preset: string;
}

export function NewRun() {
  const [params, setParams] = useSearchParams();
  const nav = useNavigate();
  const qc = useQueryClient();
  const toast = useToast();
  const { data: status } = useStatus();
  const { data: presets } = useProfiles();
  const fromJob = params.get("from") ?? undefined;
  const { data: srcCfg } = useJobConfig(fromJob);
  const { data: srcJob } = useJob(fromJob);
  const [draft, setDraftState] = useState<StoredDraft | null>(() => {
    if (params.get("preset") || fromJob) return null;
    return loadDraft<StoredDraft>();
  });
  const setDraft = useCallback((d: StoredDraft | null) => {
    setDraftState(d);
    saveDraft(d);
  }, []);

  // ?preset=X starts a fresh draft on that preset
  useEffect(() => {
    const p = params.get("preset");
    if (p) {
      setDraft({ preset: p, ...emptyDraft() });
      setParams({}, { replace: true });
    }
  }, [params, setParams, setDraft]);
  // ?from=<job> copies that run's preset, changes and tasks
  useEffect(() => {
    if (!fromJob || !srcCfg || !srcJob) return;
    setDraft({
      preset: srcCfg.profile,
      ...emptyDraft(),
      overrides: { ...srcCfg.overrides },
      picked: srcJob.tasks.map(taskKey),
      scenario: srcJob.extra.scenario ?? "ok",
      speed: srcJob.extra.speed ?? "20",
    });
    setParams({}, { replace: true });
    toast("info", `Copied ${srcJob.profile} (${srcJob.id})`);
  }, [fromJob, srcCfg, srcJob, setParams, setDraft, toast]);

  const visible = (presets?.profiles ?? []).filter((p) => !p.hidden || p.name === draft?.preset);
  const { data: detail, error } = useProfile(draft?.preset);
  const sim = !!status?.sandboxed;

  // picked tasks the preset no longer has are dropped; if none remain, fall back to all
  const effective: RunDraft | null = useMemo(() => {
    if (!draft || !detail) return null;
    if (draft.picked === null) return draft;
    const have = new Set(detail.tasks.map(taskKey));
    const kept = draft.picked.filter((k) => have.has(k));
    return { ...draft, picked: kept.length ? kept : null };
  }, [draft, detail]);

  const launch = useMutation({
    mutationFn: () => api.post<Job>("/api/jobs", jobBody(detail!, effective!, sim)),
    onSuccess: (job) => {
      setDraft(null);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["overview"] });
      toast("ok", `Started ${job.tasks.length} task${job.tasks.length === 1 ? "" : "s"}`);
      nav(`/jobs/${job.id}`);
    },
    onError: (e) => toast("error", errText(e)),
  });

  if (fromJob && !draft) return <Skeleton className="h-96" />;
  const nTasks = detail && effective ? pickedTasks(detail, effective).length : 0;
  const nChanged = detail && effective ? changedOverrides(detail, effective).length : 0;

  return (
    <div className="animate-fade-in space-y-5">
      <Link to="/jobs" className="inline-flex items-center gap-1 text-xs text-muted hover:text-ink">
        <ArrowLeft className="h-3.5 w-3.5" /> Jobs
      </Link>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="flex items-center gap-3 text-[22px]">
            New run <Badge tone="accent">draft</Badge>
          </h1>
          <p className="mt-1 text-[13px] text-muted">Kept in this browser until you launch it or discard it.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {draft && (
            <button className="btn btn-sm" onClick={() => setDraft(null)}>
              <X className="h-3.5 w-3.5" /> Discard draft
            </button>
          )}
          <button className="btn btn-primary" disabled={!detail || !effective || nTasks === 0 || launch.isPending || (detail?.errors.length ?? 0) > 0} onClick={() => launch.mutate()}>
            <Play className="h-3.5 w-3.5" /> {launch.isPending ? "Starting..." : `Launch${nTasks ? ` ${nTasks} task${nTasks === 1 ? "" : "s"}` : ""}`}
          </button>
        </div>
      </div>
      <div className="space-y-5 pt-1">
        <Card className="flex flex-wrap items-center gap-3 p-4">
          <label className="flex min-w-0 flex-1 items-center gap-3 text-[13px]">
            <span className="label shrink-0">Preset</span>
            <select
              className="input h-9 max-w-sm py-0"
              value={draft?.preset ?? ""}
              onChange={(e) => setDraft(e.target.value ? { preset: e.target.value, ...emptyDraft() } : null)}
            >
              <option value="">Choose a preset...</option>
              {visible.map((p) => (
                <option key={p.name} value={p.name} disabled={p.errors.length > 0}>
                  {p.name} ({p.task_count} task{p.task_count === 1 ? "" : "s"}){p.errors.length ? " · has errors" : ""}
                </option>
              ))}
            </select>
          </label>
          {draft && (
            <span className="text-[12px] text-muted">
              {nChanged > 0 ? `${nChanged} change${nChanged === 1 ? "" : "s"} from the preset` : "Preset as saved"}
              {" · "}
              <Link to={`/presets/${enc(draft.preset)}`} className="hover:text-ink">
                open preset
              </Link>
            </span>
          )}
        </Card>
        {!draft && <EmptyState title="Pick a preset to start from" body="Its settings fill in below. Change what you need for this run, choose tasks, then launch." />}
        {draft && error && <ErrorNote error={error} />}
        {draft && !error && !detail && <Skeleton className="h-80" />}
        {detail && detail.errors.length > 0 && (
          <Banner tone="error" title="This preset can't run">
            {detail.errors.join("; ")}
          </Banner>
        )}
        {detail && effective && <RunForm detail={detail} draft={effective} onChange={(d) => setDraft({ ...d, preset: draft!.preset })} wide />}
      </div>
    </div>
  );
}
