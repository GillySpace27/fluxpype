import { Pause, Play } from "lucide-react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { useQueue } from "../lib/hooks";
import { cn } from "../lib/cn";
import { Tip, useToast } from "./ui";

export function QueuePill() {
  const { data: q } = useQueue();
  const qc = useQueryClient();
  const toast = useToast();
  const toggle = useMutation({
    mutationFn: (paused: boolean) => api.patch("/api/queue", { paused }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["queue"] }),
    onError: (e) => toast("error", String(e)),
  });
  if (!q) return null;
  const busy = q.running > 0;
  return (
    <div className="flex items-center gap-2 rounded-full border border-line bg-raised/70 py-1 pl-3 pr-1 text-xs">
      <span className={cn("h-1.5 w-1.5 rounded-full", q.paused ? "bg-run" : busy ? "bg-ok animate-pulse" : "bg-faint")} />
      <span className="num text-muted">
        <span className="text-ink">{q.running}</span> running · <span className="text-ink">{q.pending}</span> queued
        {q.paused && <span className="ml-1.5 text-run">paused</span>}
      </span>
      <Tip label={q.paused ? "Resume starting queued tasks" : "Pause: running tasks continue, nothing new starts"}>
        <button className="btn btn-ghost btn-icon !h-6 !w-6 rounded-full" aria-label={q.paused ? "Resume queue" : "Pause queue"} onClick={() => toggle.mutate(!q.paused)}>
          {q.paused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
        </button>
      </Tip>
    </div>
  );
}
