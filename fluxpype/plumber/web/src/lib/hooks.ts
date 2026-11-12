import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "./api";
import { useToast } from "../components/ui";
import { useLive } from "./live";
import type {
  CacheData,
  DatData,
  DataDirReport,
  HealthReport,
  FluxWorld,
  Job,
  JobConfig,
  JobFiles,
  PlotPlanItem,
  PlotRun,
  Overview,
  ProfileDetail,
  ProfileSummary,
  QueueState,
  RelaxSeries,
  StatusInfo,
  TaskResults,
  TaskState,
  TrashItem,
} from "./types";

/** Re-render every `ms` (for live elapsed timers). */
export function useNow(ms = 1000): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now() / 1000), ms);
    return () => window.clearInterval(id);
  }, [ms]);
  return now;
}

/** Poll more slowly while the WebSocket is delivering pushes. */
function useRefetch(fast: number, slow: number) {
  const { connected } = useLive();
  return connected ? slow : fast;
}

export const useStatus = (poll?: number) => useQuery({ queryKey: ["status"], queryFn: () => api.get<StatusInfo>("/api/status"), staleTime: 60_000, refetchInterval: poll });

export const useOverview = () => {
  const refetchInterval = useRefetch(2000, 8000);
  return useQuery({ queryKey: ["overview"], queryFn: () => api.get<Overview>("/api/overview"), refetchInterval });
};

export const useQueue = () => {
  const refetchInterval = useRefetch(3000, 10000);
  return useQuery({ queryKey: ["queue"], queryFn: () => api.get<QueueState>("/api/queue"), refetchInterval });
};

export const useJobs = () => {
  const refetchInterval = useRefetch(3000, 10000);
  return useQuery({ queryKey: ["jobs"], queryFn: () => api.get<{ jobs: Job[] }>("/api/jobs"), refetchInterval });
};

export const useJob = (id: string | undefined) => {
  const refetchInterval = useRefetch(2500, 10000);
  return useQuery({
    queryKey: ["job", id],
    queryFn: () => api.get<Job>(`/api/jobs/${id}`),
    enabled: !!id,
    refetchInterval,
  });
};

export const useTask = (id: string | undefined, active: boolean) => {
  const refetchInterval = useRefetch(1500, 6000);
  return useQuery({
    queryKey: ["task", id],
    queryFn: () => api.get<TaskState>(`/api/tasks/${id}`),
    enabled: !!id,
    refetchInterval: active ? refetchInterval : false,
  });
};

export const useRelax = (id: string | undefined, nPoints: number, active: boolean) => {
  const refetchInterval = useRefetch(2000, 6000);
  return useQuery({
    queryKey: ["relax", id],
    queryFn: () => api.get<RelaxSeries>(`/api/tasks/${id}/relax`),
    enabled: !!id,
    refetchInterval: active ? refetchInterval : false,
    // a new point count means new data; keep the previous series on screen while refetching
    placeholderData: (prev) => prev,
    meta: { nPoints },
  });
};

export const useProfiles = () =>
  useQuery({ queryKey: ["profiles"], queryFn: () => api.get<{ profiles: ProfileSummary[]; deleted: string[] }>("/api/profiles"), staleTime: 10_000 });

export const useHealth = () =>
  useQuery({ queryKey: ["health"], queryFn: () => api.get<HealthReport>("/api/health"), refetchInterval: 60_000, staleTime: 30_000 });

export const useProfile = (name: string | undefined) =>
  useQuery({ queryKey: ["profile", name], queryFn: () => api.get<ProfileDetail>(`/api/profiles/${encodeURIComponent(name!)}`), enabled: !!name });

export const useDataDir = () =>
  useQuery({ queryKey: ["datadir"], queryFn: () => api.get<DataDirReport>("/api/data-dir"), staleTime: 30_000 });

export const useCache = () => useQuery({ queryKey: ["cache"], queryFn: () => api.get<CacheData>("/api/cache"), staleTime: 15_000 });

export const useTrash = () =>
  useQuery({ queryKey: ["trash"], queryFn: () => api.get<{ items: TrashItem[]; count: number; size: number }>("/api/cache/trash") });

/** True on screens wide enough for a list and a preview side by side (Tailwind lg). */
export function useWide(query = "(min-width: 1024px)"): boolean {
  const [wide, setWide] = useState(() => typeof window !== "undefined" && window.matchMedia(query).matches);
  useEffect(() => {
    const m = window.matchMedia(query);
    const on = () => setWide(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, [query]);
  return wide;
}

/** What one task produced; refreshed while it runs. */
export const useTaskResults = (id: string | undefined, active: boolean) =>
  useQuery({
    queryKey: ["results", id],
    queryFn: () => api.get<TaskResults>(`/api/tasks/${id}/results`),
    enabled: !!id,
    refetchInterval: active ? 5000 : false,
    placeholderData: (prev) => prev,
  });

/** The settings a job ran with (its frozen snapshot). */
export const useJobConfig = (id: string | undefined) =>
  useQuery({ queryKey: ["job-config", id], queryFn: () => api.get<JobConfig>(`/api/jobs/${id}/config`), enabled: !!id, staleTime: Infinity });

/** Every file a job's tasks read or wrote. */
export const useJobFiles = (id: string | undefined, active: boolean) =>
  useQuery({
    queryKey: ["job-files", id],
    queryFn: () => api.get<JobFiles>(`/api/jobs/${id}/files`),
    enabled: !!id,
    refetchInterval: active ? 6000 : false,
    placeholderData: (prev) => prev,
  });

/** API prefix for the file viewer when it reads a job's own data folder. */
export const jobDataBase = (jobId: string) => `/api/jobs/${encodeURIComponent(jobId)}/data`;

/** A .flux world file parsed by the server; `base` is the API prefix (current data folder or a job's). */
export const useFluxWorld = (path: string | undefined, base = "/api/cache", mtime?: number) =>
  useQuery({
    queryKey: ["flux", base, path, mtime],
    queryFn: () => api.get<FluxWorld>(`${base}/flux?path=${encodeURIComponent(path!)}`),
    enabled: !!path,
    staleTime: 60_000,
    retry: false,
  });

/** A numeric .dat table parsed by the server; same bases as `useFluxWorld`. */
export const useDatTable = (path: string | undefined, base = "/api/cache", mtime?: number) =>
  useQuery({
    queryKey: ["dat", base, path, mtime],
    queryFn: () => api.get<DatData>(`${base}/dat?path=${encodeURIComponent(path!)}`),
    enabled: !!path,
    staleTime: 60_000,
    retry: false,
  });

const ACTIVE_RUN = ["starting", "running"];
export const runActive = (r: PlotRun) => ACTIVE_RUN.includes(r.status);

/** The post-hoc plot runs of a task, newest first; polled while one is active. */
export const usePlotRuns = (taskId: string | undefined) =>
  useQuery({
    queryKey: ["plotruns", taskId],
    queryFn: () => api.get<{ task_id: string; runs: PlotRun[] }>(`/api/tasks/${taskId}/plots`).then((r) => r.runs),
    enabled: !!taskId,
    refetchInterval: (q) => (q.state.data?.some(runActive) ? 2000 : false),
  });

/** Start a plot run. A dry run first; if it would replace files or use inputs changed since the task ended, ask before overwriting. */
export function useGeneratePlots(taskId: string) {
  const qc = useQueryClient();
  const toast = useToast();
  return useMutation({
    mutationFn: async ({ plots, stage }: { plots: string[] | "all_possible"; stage?: string }) => {
      const base = { plots, stage: stage ?? null, include_legacy: false };
      const { plan } = await api.post<{ plan: PlotPlanItem[] }>(`/api/tasks/${taskId}/plots`, { ...base, dry_run: true });
      const clash = plan.filter((p) => p.conflicts.length || p.inputs_later.length);
      let overwrite = false;
      if (clash.length) {
        const lines = clash.map(
          (p) =>
            `${p.title}\n` +
            [...p.conflicts.map((f) => `  replaces ${f}`), ...p.inputs_later.map((f) => `  input changed since the run: ${f}`)].join("\n") +
            (p.overwrite_risk ? `\n  Risk: ${p.overwrite_risk}` : "")
        );
        if (!window.confirm(`This will overwrite files (copies are kept in the plot run's replaced folder):\n\n${lines.join("\n\n")}\n\nContinue?`)) return null;
        overwrite = true;
      }
      return api.post<PlotRun>(`/api/tasks/${taskId}/plots`, { ...base, overwrite });
    },
    onSuccess: (run) => {
      if (run) toast("info", `Making ${run.plots.length} plot${run.plots.length === 1 ? "" : "s"} from the files on disk`);
      qc.invalidateQueries({ queryKey: ["plotruns", taskId] });
    },
    onError: (e) => toast("error", e instanceof Error ? e.message : "Could not start the plot run"),
  });
}
