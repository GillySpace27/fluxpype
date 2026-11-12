import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Check, Play } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { cn } from "../lib/cn";

interface Action {
  tool: string;
  args: Record<string, unknown>;
  why?: string;
}

const str = (v: unknown) => (typeof v === "boolean" ? (v ? "1" : "0") : Array.isArray(v) ? `[${v.join(", ")}]` : String(v)); // as the MCP server writes overrides

/** The four changes the model may propose; the click is the approval. Returns a line for the person and a link if there is one. */
const RUNNERS: Record<string, { title: string; danger?: boolean; run: (a: Record<string, any>) => Promise<{ done: string; to?: string }> }> = {
  submit_job: {
    title: "Queue a job",
    run: async (a) => {
      const body: Record<string, unknown> = {
        profile: a.profile,
        overrides: Object.fromEntries(Object.entries(a.overrides ?? {}).map(([k, v]) => [k, str(v)])),
      };
      if (a.tasks) body.tasks = (a.tasks as unknown[][]).map((t) => t.map(String));
      if (a.backend) body.backend = a.backend;
      const job = await api.post<{ id: string }>("/api/jobs", body);
      return { done: `Queued job ${job.id}`, to: `/jobs/${job.id}` };
    },
  },
  cancel: {
    title: "Cancel",
    danger: true,
    run: async (a) => {
      if (a.job_id) await api.post(`/api/jobs/${encodeURIComponent(a.job_id)}/cancel`);
      else await api.post(`/api/tasks/${encodeURIComponent(a.task_id)}/cancel`);
      return { done: "Cancelled", to: a.job_id ? `/jobs/${a.job_id}` : undefined };
    },
  },
  retry_failed: {
    title: "Retry failed tasks",
    run: async (a) => {
      await api.post(`/api/jobs/${encodeURIComponent(a.job_id)}/retry`);
      return { done: "Re-queued", to: `/jobs/${a.job_id}` };
    },
  },
  set_queue: {
    title: "Change the queue",
    run: async (a) => {
      const { paused, max_parallel, cpu_budget } = a;
      await api.patch("/api/queue", Object.fromEntries(Object.entries({ paused, max_parallel, cpu_budget }).filter(([, v]) => v != null)));
      return { done: "Queue updated" };
    },
  },
};

function ActionCard({ action }: { action: Action }) {
  const r = RUNNERS[action.tool];
  const qc = useQueryClient();
  const [dismissed, setDismissed] = useState(false);
  const go = useMutation({
    mutationFn: () => r.run(action.args ?? {}),
    onSuccess: () => qc.invalidateQueries(),
  });
  if (dismissed) return null;
  return (
    <div className={cn("my-2 rounded-xl border p-3", r.danger ? "border-fail/40 bg-fail/5" : "border-accent/40 bg-accent/5")}>
      <p className="text-[13px] font-semibold">{r.title}</p>
      {action.why && <p className="mt-0.5 text-xs text-muted">{action.why}</p>}
      <pre className="mt-2 overflow-x-auto rounded-lg bg-bg/60 px-2.5 py-1.5 font-mono text-[11px]">{JSON.stringify(action.args ?? {}, null, 1)}</pre>
      <div className="mt-2.5 flex items-center gap-2">
        {go.isSuccess ? (
          <p className="flex items-center gap-1.5 text-xs text-ok">
            <Check className="h-3.5 w-3.5" /> {go.data.done}
            {go.data.to && (
              <Link to={go.data.to} className="ml-1 underline">
                open
              </Link>
            )}
          </p>
        ) : (
          <>
            <button className={cn("btn btn-sm", r.danger ? "border-fail/50 text-fail" : "btn-primary")} onClick={() => go.mutate()} disabled={go.isPending}>
              <Play className="h-3 w-3" /> {go.isPending ? "Working..." : "Run"}
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => setDismissed(true)}>
              Dismiss
            </button>
            {go.isError && <span className="text-xs text-fail">{go.error instanceof Error ? go.error.message : String(go.error)}</span>}
          </>
        )}
      </div>
    </div>
  );
}

/** `code` and **bold** inside a line. */
function Inline({ text }: { text: string }) {
  return (
    <>
      {text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).map((p, i) =>
        p.startsWith("`") && p.endsWith("`") && p.length > 1 ? (
          <code key={i} className="rounded bg-bg/70 px-1 py-px font-mono text-[12px]">
            {p.slice(1, -1)}
          </code>
        ) : p.startsWith("**") && p.endsWith("**") && p.length > 4 ? (
          <strong key={i}>{p.slice(2, -2)}</strong>
        ) : (
          p
        ),
      )}
    </>
  );
}

function Prose({ text }: { text: string }) {
  return (
    <>
      {text
        .split(/\n{2,}/)
        .filter((b) => b.trim())
        .map((b, i) => {
          const lines = b.split("\n");
          return lines.every((l) => /^\s*[-*] /.test(l)) ? (
            <ul key={i} className="my-1.5 list-disc space-y-0.5 pl-5">
              {lines.map((l, j) => (
                <li key={j}>
                  <Inline text={l.replace(/^\s*[-*] /, "")} />
                </li>
              ))}
            </ul>
          ) : (
            <p key={i} className="my-1.5 whitespace-pre-wrap">
              <Inline text={b} />
            </p>
          );
        })}
    </>
  );
}

/** The model's answer: paragraphs, lists, inline code, code blocks, and `flux-action` blocks as cards. */
export function AskAnswer({ text, streaming }: { text: string; streaming?: boolean }) {
  const parts = text.split(/(```[\w-]*\n[\s\S]*?(?:```|$))/g);
  return (
    <div className="text-[13.5px] leading-relaxed">
      {parts.map((p, i) => {
        const m = /^```([\w-]*)\n([\s\S]*?)(?:```)?$/.exec(p);
        if (!m) return <Prose key={i} text={p} />;
        const closed = p.trimEnd().endsWith("```") && p.length > 7;
        if (m[1] === "flux-action" && !streaming && closed) {
          try {
            const a = JSON.parse(m[2]) as Action;
            if (a && RUNNERS[a.tool]) return <ActionCard key={i} action={a} />;
          } catch {
            /* shown as code below */
          }
        }
        if (m[1] === "flux-action" && streaming) return null; // the card appears once the block is complete
        return (
          <pre key={i} className="my-2 overflow-x-auto rounded-lg bg-bg/60 px-3 py-2 font-mono text-[11.5px]">
            {m[2]}
          </pre>
        );
      })}
    </div>
  );
}
