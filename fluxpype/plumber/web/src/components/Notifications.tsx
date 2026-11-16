import { Bell, Bot, Check, CheckCheck, Globe, MessageSquare } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { cn } from "../lib/cn";
import { eventDetail, eventLink, eventTitle, INBOX } from "../lib/events";
import { ago } from "../lib/format";
import { useNow } from "../lib/hooks";
import { useEvents } from "../lib/live";
import type { Origin, PlumberEvent, Sev } from "../lib/types";
import { Dialog, Segmented, Tab, TabList, TabPanel, Tabs, Tip } from "./ui";

const SEV_DOT: Record<Sev, string> = { info: "bg-info", success: "bg-ok", warn: "bg-run", error: "bg-fail" };

/** Who started a job: a label, not proof that a person acted. */
export function OriginTag({ origin, className }: { origin?: Origin; className?: string }) {
  if (!origin?.kind) return null;
  const { icon: Icon, label } = {
    web: { icon: Globe, label: "Started from the web app" },
    mcp: { icon: Bot, label: "Started by an AI agent (MCP)" },
    ask: { icon: MessageSquare, label: "Started from an Ask bar card" },
  }[origin.kind];
  return (
    <Tip label={label + (origin.retried_by ? `; retried from ${origin.retried_by}` : "")}>
      <span className={cn("inline-flex items-center gap-1 text-[11px] text-faint", className)}>
        <Icon className="h-3 w-3" />
        {origin.kind === "ask" ? "Ask" : null}
      </span>
    </Tip>
  );
}

export function EventRow({ e, now, onAck, onClick }: { e: PlumberEvent; now: number; onAck?: () => void; onClick?: () => void }) {
  const to = eventLink(e);
  const body = (
    <>
      <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", SEV_DOT[e.sev] ?? "bg-line")} />
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2 text-[13px] font-medium">
          {eventTitle(e)}
          <OriginTag origin={e.origin} />
        </span>
        <span className="block truncate text-xs text-muted">{eventDetail(e)}</span>
      </span>
      <span className="shrink-0 text-[11px] text-faint">{ago(e.t, now)}</span>
    </>
  );
  return (
    <div className={cn("flex items-start gap-2.5 px-3 py-2", e.acked && "opacity-60")}>
      {to ? (
        <Link to={to} onClick={onClick} className="flex min-w-0 flex-1 items-start gap-2.5 hover:text-ink">
          {body}
        </Link>
      ) : (
        <div className="flex min-w-0 flex-1 items-start gap-2.5">{body}</div>
      )}
      {onAck && !e.acked && (
        <Tip label="Acknowledge">
          <button className="btn btn-ghost btn-icon btn-sm" aria-label="Acknowledge" onClick={onAck}>
            <Check className="h-3.5 w-3.5" />
          </button>
        </Tip>
      )}
    </div>
  );
}

const needsAttention = (e: PlumberEvent) => INBOX.has(e.kind) && (e.sev === "error" || e.sev === "warn") && !e.acked;

/** The bell in the top bar: unread count, red while an error is unacknowledged; opens the inbox. */
export function NotificationsBell() {
  const ev = useEvents();
  const [open, setOpen] = useState(false);
  return (
    <>
      <Tip label={ev.attention ? `${ev.attention} error${ev.attention === 1 ? "" : "s"} need attention` : "Notifications"}>
        <button className="btn btn-ghost btn-icon relative" aria-label="Notifications" onClick={() => setOpen(true)}>
          <Bell className={cn("h-4 w-4", ev.attention > 0 && "text-fail")} />
          {ev.unread > 0 && (
            <span className={cn("absolute -right-0.5 -top-0.5 min-w-[16px] rounded-full px-1 text-center text-[10px] font-semibold leading-4", ev.attention > 0 ? "bg-fail text-white" : "bg-accent text-accent-ink")}>
              {ev.unread > 99 ? "99+" : ev.unread}
            </span>
          )}
        </button>
      </Tip>
      {open && <NotificationsDrawer onClose={() => setOpen(false)} />}
    </>
  );
}

function NotificationsDrawer({ onClose }: { onClose: () => void }) {
  const ev = useEvents();
  const now = useNow(10000);
  const [sev, setSev] = useState<"all" | "warn" | "error">("all");
  const [who, setWho] = useState<"any" | "web" | "mcp" | "ask">("any");
  const newest = ev.events[0]?.seq ?? 0;
  useEffect(() => {
    if (newest > ev.readUpto) ev.markRead(newest); // opening the inbox reads what it shows
  }, [newest]); // eslint-disable-line react-hooks/exhaustive-deps
  const attention = useMemo(() => ev.events.filter(needsAttention), [ev.events]);
  const errors = useMemo(() => ev.events.filter((e) => e.sev === "error"), [ev.events]);
  const all = useMemo(
    () =>
      ev.events.filter(
        (e) =>
          (sev === "all" || e.sev === sev || (sev === "warn" && e.sev === "error")) &&
          (who === "any" || e.origin?.kind === who),
      ),
    [ev.events, sev, who],
  );
  const list = (rows: PlumberEvent[], empty: string, ackable = false) =>
    rows.length === 0 ? (
      <p className="px-3 py-8 text-center text-[13px] text-muted">{empty}</p>
    ) : (
      <div className="divide-y divide-line/60">
        {rows.map((e) => (
          <EventRow key={e.seq} e={e} now={now} onClick={onClose} onAck={ackable ? () => ev.ack({ seqs: [e.seq] }) : undefined} />
        ))}
      </div>
    );
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} side title="Notifications" description="Job events from this Plumber, kept whether or not a page was open.">
      <Tabs defaultValue={attention.length ? "attention" : "all"}>
        <TabList>
          <Tab value="attention">Attention{attention.length ? ` (${attention.length})` : ""}</Tab>
          <Tab value="all">All</Tab>
          <Tab value="errors">Errors</Tab>
        </TabList>
        <TabPanel value="attention" className="pt-3">
          {attention.length > 0 && (
            <div className="mb-2 flex justify-end">
              <button className="btn btn-sm" onClick={() => ev.ack({ all: true })}>
                <CheckCheck className="h-3.5 w-3.5" /> Acknowledge all
              </button>
            </div>
          )}
          {list(attention, "Nothing needs attention.", true)}
        </TabPanel>
        <TabPanel value="all" className="pt-3">
          <div className="mb-2 flex flex-wrap gap-2">
            <Segmented label="Severity" value={sev} onChange={setSev} options={[{ value: "all", label: "All" }, { value: "warn", label: "Warnings" }, { value: "error", label: "Errors" }]} />
            <Segmented label="Started by" value={who} onChange={setWho} options={[{ value: "any", label: "Anyone" }, { value: "web", label: "Web" }, { value: "mcp", label: "Agent" }, { value: "ask", label: "Ask" }]} />
          </div>
          {list(all, "No events yet.", true)}
          {ev.more && (
            <button className="btn btn-sm mt-3 w-full" onClick={ev.loadOlder}>
              Load older
            </button>
          )}
        </TabPanel>
        <TabPanel value="errors" className="pt-3">
          {list(errors, "No errors.", true)}
        </TabPanel>
      </Tabs>
    </Dialog>
  );
}

/** Overview: unacknowledged errors, each with Acknowledge. */
export function AttentionPanel() {
  const ev = useEvents();
  const now = useNow(10000);
  const rows = ev.events.filter((e) => e.sev === "error" && !e.acked);
  if (!rows.length) return null;
  return (
    <div className="card divide-y divide-line/60 border-fail/30">
      <div className="flex items-center justify-between px-3 py-2">
        <p className="text-[13px] font-medium text-fail">
          {Math.max(ev.attention, rows.length)} error{Math.max(ev.attention, rows.length) === 1 ? "" : "s"} need attention
        </p>
        <button className="btn btn-sm" onClick={() => ev.ack({ all: true })}>
          <CheckCheck className="h-3.5 w-3.5" /> Acknowledge all
        </button>
      </div>
      {rows.slice(0, 6).map((e) => (
        <EventRow key={e.seq} e={e} now={now} onAck={() => ev.ack({ seqs: [e.seq] })} />
      ))}
    </div>
  );
}

/** The latest events of any severity, optionally for one job. */
export function ActivityList({ limit = 10, job }: { limit?: number; job?: string }) {
  const ev = useEvents();
  const now = useNow(10000);
  const rows = ev.events.filter((e) => !job || e.job_id === job).slice(0, limit);
  return rows.length === 0 ? (
    <p className="px-3 py-6 text-center text-[13px] text-muted">No activity yet.</p>
  ) : (
    <div className="divide-y divide-line/60">
      {rows.map((e) => (
        <EventRow key={e.seq} e={e} now={now} />
      ))}
    </div>
  );
}
