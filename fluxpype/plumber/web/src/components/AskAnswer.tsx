import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Play, X } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { cn } from "../lib/cn";

interface Proposal {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  summary: string;
  diff: { key: string; from: string | null; to: string }[];
  status: "pending" | "running" | "approved" | "rejected" | "expired" | "failed";
  result: Record<string, unknown> | null;
  error: string | null;
}

/** The changes the model may propose. The server stored and checked them; Run executes the stored arguments once. */
const TITLES: Record<string, { title: string; danger?: boolean }> = {
  submit_job: { title: "Queue a job" },
  cancel: { title: "Cancel", danger: true },
  retry_failed: { title: "Retry failed tasks" },
  set_queue: { title: "Change the queue" },
  create_preset: { title: "Save a preset" },
};

const linkOf = (p: Proposal) =>
  p.result?.job_id ? `/jobs/${p.result.job_id}` : p.result?.name ? `/presets/${encodeURIComponent(String(p.result.name))}` : undefined;

function ProposalCard({ p }: { p: Proposal }) {
  const t = TITLES[p.tool] ?? { title: p.tool };
  const qc = useQueryClient();
  const decide = useMutation({
    mutationFn: (d: "approve" | "reject") => api.post<Proposal>(`/api/proposals/${encodeURIComponent(p.id)}/${d}`),
    onSettled: () => qc.invalidateQueries(),
  });
  const to = linkOf(p);
  return (
    <div className={cn("my-2 rounded-xl border p-3", t.danger ? "border-fail/40 bg-fail/5" : "border-accent/40 bg-accent/5")}>
      <p className="text-[13px] font-semibold">{t.title}</p>
      <p className="mt-0.5 text-xs text-muted">{p.summary}</p>
      {p.diff.length > 0 && (
        <table className="mt-2 w-full font-mono text-[11px]">
          <tbody>
            {p.diff.map((d) => (
              <tr key={d.key}>
                <td className="pr-2 align-top">{d.key}</td>
                <td className="pr-2 align-top text-faint line-through">{d.from ?? "(unset)"}</td>
                <td className="align-top">{d.to}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <pre className="mt-2 overflow-x-auto rounded-lg bg-bg/60 px-2.5 py-1.5 font-mono text-[11px]">{JSON.stringify(p.args, null, 1)}</pre>
      <div className="mt-2.5 flex items-center gap-2">
        {p.status === "pending" ? (
          <>
            <button className={cn("btn btn-sm", t.danger ? "border-fail/50 text-fail" : "btn-primary")} onClick={() => decide.mutate("approve")} disabled={decide.isPending}>
              <Play className="h-3 w-3" /> {decide.isPending ? "Working..." : "Run"}
            </button>
            <button className="btn btn-ghost btn-sm" onClick={() => decide.mutate("reject")} disabled={decide.isPending}>
              <X className="h-3 w-3" /> Reject
            </button>
            {decide.isError && <span className="text-xs text-fail">{decide.error instanceof Error ? decide.error.message : String(decide.error)}</span>}
          </>
        ) : (
          <p className={cn("flex items-center gap-1.5 text-xs", p.status === "approved" ? "text-ok" : p.status === "failed" ? "text-fail" : "text-muted")}>
            {p.status === "approved" && <Check className="h-3.5 w-3.5" />}
            {p.status === "approved" ? "Done" : p.status === "failed" ? `Failed: ${p.error ?? ""}` : p.status[0].toUpperCase() + p.status.slice(1)}
            {to && (
              <Link to={to} className="ml-1 underline">
                open
              </Link>
            )}
          </p>
        )}
      </div>
    </div>
  );
}

/** Cards for the proposals this answer made, drawn from what the server stored (never from the model's text). */
function ProposalCards({ ids }: { ids: string[] }) {
  const q = useQuery({ queryKey: ["proposals"], queryFn: () => api.get<{ proposals: Proposal[] }>("/api/proposals") });
  const byId = new Map((q.data?.proposals ?? []).map((p) => [p.id, p]));
  return (
    <>
      {ids
        .map((id) => byId.get(id))
        .filter((p): p is Proposal => !!p)
        .map((p) => (
          <ProposalCard key={p.id} p={p} />
        ))}
    </>
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

/** The model's answer: paragraphs, lists, inline code and code blocks. Plumber appends a `flux-proposals` block (ids)
 * after the model's text; the last one becomes cards drawn from the server's record of those proposals. */
export function AskAnswer({ text, streaming }: { text: string; streaming?: boolean }) {
  const parts = text.split(/(```[\w-]*\n[\s\S]*?(?:```|$))/g);
  let ids: string[] = [];
  const shown = parts.map((p, i) => {
    const m = /^```([\w-]*)\n([\s\S]*?)(?:```)?$/.exec(p);
    if (!m) return <Prose key={i} text={p} />;
    if (m[1] === "flux-proposals") {
      try {
        const v = JSON.parse(m[2]);
        ids = Array.isArray(v) ? v.map(String) : [];
      } catch {
        /* incomplete while streaming */
      }
      return null;
    }
    return (
      <pre key={i} className="my-2 overflow-x-auto rounded-lg bg-bg/60 px-3 py-2 font-mono text-[11.5px]">
        {m[2]}
      </pre>
    );
  });
  return (
    <div className="text-[13.5px] leading-relaxed">
      {shown}
      {!streaming && ids.length > 0 && <ProposalCards ids={ids} />}
    </div>
  );
}
