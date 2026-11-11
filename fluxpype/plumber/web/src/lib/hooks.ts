import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";
import { useLive } from "./live";
import type {
  CacheData,
  DataDirReport,
  HealthReport,
  Job,
  Overview,
  ProfileDetail,
  ProfileSummary,
  QueueState,
  RelaxSeries,
  StatusInfo,
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

export const useStatus = () => useQuery({ queryKey: ["status"], queryFn: () => api.get<StatusInfo>("/api/status"), staleTime: 60_000 });

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
