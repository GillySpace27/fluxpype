import { cn } from "../lib/cn";
import type { TaskState } from "../lib/types";
import { Badge, Banner } from "./ui";

function Row({ tone, title, children, right }: { tone: "ok" | "info" | "faint" | "fail" | "run"; title: string; children: React.ReactNode; right?: React.ReactNode }) {
  const dot = { ok: "bg-ok", info: "bg-info", faint: "bg-faint/60", fail: "bg-fail", run: "bg-run" }[tone];
  return (
    <div className="flex items-start gap-3 py-2.5">
      <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", dot)} />
      <div className="min-w-0 flex-1">
        <p className="text-[13px] font-medium">{title}</p>
        <p className="text-xs leading-relaxed text-muted">{children}</p>
      </div>
      {right}
    </div>
  );
}

const age = (s: number | null) => (s == null ? "never" : s < 2 ? "just now" : `${Math.round(s)} s ago`);

/** Which of the three sources is feeding this task's state, and how fresh each one is. */
export function SourcesPanel({ task }: { task: TaskState }) {
  const { events, log, files, process } = task.sources;
  const unverified = task.stages.filter((s) => s.verified === false);
  const finished = task.outcome != null;
  return (
    <div>
      {task.degraded && (
        <div className="mb-3">
          <Banner tone="info" title="Degraded: inferred from the log">
            {task.degraded_reason}. Stage and relaxation fields now come from the log and files until events resume.
          </Banner>
        </div>
      )}
      <div className="divide-y divide-line/60">
        <Row tone={!events.present ? "faint" : task.degraded ? "info" : "ok"} title="Structured events" right={<Badge tone={task.mode === "events" ? "accent" : "neutral"}>{task.mode === "events" ? "primary" : task.mode === "degraded" ? "stale" : "absent"}</Badge>}>
          {events.present ? (
            <>
              {events.lines.toLocaleString()} events{events.bad_lines ? `, ${events.bad_lines} unreadable` : ""}. {finished ? "Run finished." : `Last one ${age(events.age_s)}.`}
            </>
          ) : (
            "This run is not emitting events (started from a terminal, or an older install). The log and files are used instead."
          )}
        </Row>
        <Row tone={log.present ? "ok" : "faint"} title="Log" right={<Badge tone={task.mode === "log" || task.mode === "degraded" ? "info" : "neutral"}>{task.mode === "log" ? "primary" : "fallback"}</Badge>}>
          {log.present ? (
            <>
              {log.lines.toLocaleString()} lines parsed for banners and relaxation steps. {finished ? "" : `Last line ${age(log.age_s)}.`}
            </>
          ) : (
            "No log output yet."
          )}
        </Row>
        <Row tone={files.checked ? (files.missing.length && finished ? "run" : "ok") : "faint"} title="Output files" right={<Badge>probe</Badge>}>
          {files.checked ? (
            <>
              {files.ok.length} of {files.ok.length + files.missing.length} expected outputs on disk
              {files.missing.length > 0 && <>; not found: {files.missing.join(", ")}</>}.
            </>
          ) : (
            "Not checked yet."
          )}
        </Row>
        <Row tone={process.alive == null ? "faint" : process.alive ? "ok" : "fail"} title="Process">
          {process.alive == null ? (finished ? "Finished." : "Unknown.") : process.alive ? "Running." : "Not running."}
          {task.exit_code != null && <> Exit code {task.exit_code}.</>}
        </Row>
      </div>
      {unverified.length > 0 && (
        <div className="mt-3">
          <Banner tone="warn" title="Verification failed">
            Events reported {unverified.map((s) => s.label).join(", ")} as finished, but the expected output file is not on disk.
          </Banner>
        </div>
      )}
    </div>
  );
}
