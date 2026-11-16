import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useToast } from "../components/ui";
import { api, TAB_ID } from "./api";
import { eventDetail, eventTitle, notifyEnabled, toastWorthy } from "./events";
import type { EventPage, PlumberEvent, TaskState } from "./types";

const KEEP = 500; // events held in memory; older pages load on demand

type LogListener = (m: { task_id: string; start: number; offset: number; text: string }) => void;

/** The latest request an AI agent made through the MCP server; `ts` is browser time. */
export interface AgentPing {
  ts: number;
  tool: string;
  count: number;
}

/** Plumber's event log as this tab knows it: newest first, at most KEEP rows, plus the server's counts. */
export interface EventsState {
  events: PlumberEvent[];
  unread: number;
  attention: number;
  readUpto: number;
  more: boolean;
  markRead: (upto: number) => void;
  ack: (what: { seqs?: number[]; all?: boolean }) => void;
  loadOlder: () => void;
}

interface LiveCtx {
  connected: boolean;
  agent: AgentPing | null;
  onLog: (fn: LogListener) => () => void;
  ev: EventsState;
}

const NO_EVENTS: EventsState = { events: [], unread: 0, attention: 0, readUpto: 0, more: false, markRead: () => {}, ack: () => {}, loadOlder: () => {} };
const Ctx = createContext<LiveCtx>({ connected: false, agent: null, onLog: () => () => {}, ev: NO_EVENTS });

const mergeEvents = (cur: PlumberEvent[], add: PlumberEvent[]) => {
  const bySeq = new Map(cur.map((e) => [e.seq, e]));
  add.forEach((e) => bySeq.set(e.seq, e));
  return [...bySeq.values()].sort((a, b) => b.seq - a.seq).slice(0, KEEP);
};

/** One WebSocket for the whole app: pushes update the query cache; polling is only the fallback. */
export function LiveProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [connected, setConnected] = useState(false);
  const [agent, setAgent] = useState<AgentPing | null>(null);
  const listeners = useRef(new Set<LogListener>());
  const toast = useToast();
  const [events, setEvents] = useState<PlumberEvent[]>([]);
  const [counts, setCounts] = useState({ unread: 0, attention: 0, readUpto: 0 });
  const [more, setMore] = useState(false);
  const cursor = useRef<{ seq: number | null; log: string | null }>({ seq: null, log: null });
  const syncing = useRef(false);

  const takeCounts = (p: { unread: number; attention: number; read_upto: number }) =>
    setCounts({ unread: p.unread, attention: p.attention, readUpto: p.read_upto });

  /** REST catch-up: the newest page on first load or a new log, else everything after the last seq seen. */
  const catchUp = useCallback(async () => {
    if (syncing.current) return;
    syncing.current = true;
    try {
      const c = cursor.current;
      let page = await api.get<EventPage>(c.seq === null ? "/api/events?limit=100" : `/api/events?since=${c.seq}&limit=500`);
      if (c.log !== null && page.log_id !== c.log) {
        page = await api.get<EventPage>("/api/events?limit=100");
        setEvents([]);
        c.seq = null;
      }
      const fresh = c.seq === null;
      c.log = page.log_id;
      c.seq = Math.max(c.seq ?? 0, page.last_seq);
      setEvents((cur) => mergeEvents(fresh ? [] : cur, page.events));
      if (fresh) setMore(page.events.length >= 100);
      takeCounts(page);
    } catch {
      /* offline: the next socket open retries */
    } finally {
      syncing.current = false;
    }
  }, []);

  /** A live event from the socket: toasts and browser notifications only for these, never for catch-up rows. */
  const arrive = useCallback(
    (e: PlumberEvent) => {
      const c = cursor.current;
      const gap = c.seq !== null && e.seq > c.seq + 1; // missed some (asleep, reconnecting): fetch from the old cursor
      if (c.seq !== null && !gap) c.seq = Math.max(c.seq, e.seq);
      setEvents((cur) => mergeEvents(cur, [e]));
      setCounts((n) => ({
        ...n,
        unread: n.unread + (e.sev === "warn" || e.sev === "error" ? 1 : 0),
        attention: n.attention + (e.sev === "error" && !e.acked ? 1 : 0),
      }));
      if (gap) void catchUp();
      if (Date.now() - e.t * 1000 >= 60_000 || !toastWorthy(e, TAB_ID)) return;
      const text = `${eventTitle(e)}${eventDetail(e) ? `: ${eventDetail(e)}` : ""}`;
      toast(e.sev === "error" ? "error" : e.sev === "success" ? "ok" : "info", text);
      if (document.hidden && notifyEnabled()) {
        try {
          new Notification(eventTitle(e), { body: eventDetail(e), tag: e.job_id ?? e.kind });
        } catch {
          /* some browsers only allow notifications from a service worker */
        }
      }
    },
    [catchUp, toast],
  );
  const arriveRef = useRef(arrive);
  arriveRef.current = arrive;

  const markRead = useCallback((upto: number) => {
    setCounts((n) => (upto > n.readUpto ? { ...n, unread: 0, readUpto: upto } : n));
    api.post<EventPage>("/api/events/read", { upto }).then(takeCounts, () => undefined);
  }, []);
  const ack = useCallback((what: { seqs?: number[]; all?: boolean }) => {
    setEvents((cur) => cur.map((e) => (what.all || what.seqs?.includes(e.seq) ? { ...e, acked: true } : e)));
    api.post<EventPage>("/api/events/ack", what).then(takeCounts, () => undefined);
  }, []);
  const eventsRef = useRef(events);
  eventsRef.current = events;
  const loadOlder = useCallback(() => {
    const cur = eventsRef.current;
    if (!cur.length) return;
    api.get<EventPage>(`/api/events?before=${cur[cur.length - 1].seq}&limit=100`).then(
      (p) => {
        setMore(p.events.length >= 100);
        setEvents((now) => {
          const all = new Map(now.map((e) => [e.seq, e]));
          p.events.forEach((e) => all.set(e.seq, e));
          return [...all.values()].sort((a, b) => b.seq - a.seq); // a page asked for may go past KEEP
        });
      },
      () => undefined,
    );
  }, []);

  const baseTitle = useRef(document.title);
  useEffect(() => {
    document.title = counts.attention > 0 ? `(${counts.attention}) Plumber` : baseTitle.current;
  }, [counts.attention]);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;
    let attempt = 0;
    let invalidateTimer: number | undefined;

    // keys pile up until the timer fires, so a busy stream of events cannot starve an earlier invalidation
    const pending = new Map<string, unknown[]>();
    const soon = (keys: unknown[][]) => {
      keys.forEach((k) => pending.set(JSON.stringify(k), k));
      window.clearTimeout(invalidateTimer);
      invalidateTimer = window.setTimeout(() => {
        const todo = [...pending.values()];
        pending.clear();
        todo.forEach((k) => qc.invalidateQueries({ queryKey: k }));
      }, 120);
    };

    const open = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/api/ws`);
      ws.onopen = () => {
        attempt = 0;
        setConnected(true);
        qc.invalidateQueries();
        void catchUp();
      };
      ws.onmessage = (ev) => {
        let msg: { type: string; data: any };
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        switch (msg.type) {
          case "task_state": {
            const s = msg.data as TaskState;
            qc.setQueryData(["task", s.id], s);
            soon([["overview"], ["job", s.job_id], ["relax", s.id]]);
            break;
          }
          case "job_state":
            qc.setQueryData(["job", msg.data.id], (old: any) => (old ? { ...old, ...msg.data } : old));
            soon([["jobs"], ["overview"], ["queue"]]);
            break;
          case "plot_run": {
            const d = msg.data as { task_id: string; job_id: string; status: string };
            const keys: unknown[][] = [["plotruns", d.task_id]];
            if (!["starting", "running"].includes(d.status)) keys.push(["results", d.task_id], ["job-files", d.job_id]);
            soon(keys);
            break;
          }
          case "job_removed":
            soon([["jobs"], ["overview"]]);
            break;
          case "queue_state":
            soon([["queue"], ["overview"]]);
            break;
          case "cache_changed":
            soon([["cache"], ["trash"], ["profiles"], ["profile"]]);
            break;
          case "presets_changed":
            soon([["profiles"], ["profile"]]);
            break;
          case "data_dir_changed":
            soon([["status"], ["health"], ["datadir"], ["cache"], ["trash"], ["profiles"], ["profile"]]);
            break;
          case "backend_changed":
            soon([["status"], ["health"], ["queue"], ["cache"], ["trash"], ["profiles"], ["profile"]]);
            break;
          case "log_append":
            listeners.current.forEach((fn) => fn(msg.data));
            break;
          case "agent_call":
            soon([["agent-log"]]);
            break;
          case "proposals":
            soon([["proposals"]]);
            break;
          case "agent_activity":
            setAgent({ ts: Date.now(), tool: msg.data.tool, count: msg.data.count });
            break;
          case "event":
            arriveRef.current(msg.data as PlumberEvent);
            break;
          case "events_read":
            takeCounts(msg.data);
            break;
        }
      };
      ws.onclose = () => {
        setConnected(false);
        if (closed) return;
        attempt += 1;
        timer = window.setTimeout(open, Math.min(1000 * 2 ** Math.min(attempt, 4), 10000));
      };
      ws.onerror = () => ws?.close();
    };
    open();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      window.clearTimeout(invalidateTimer);
      ws?.close();
    };
  }, [qc, catchUp]);

  const value = useRef<LiveCtx>({
    connected: false,
    agent: null,
    ev: NO_EVENTS,
    onLog: (fn) => {
      listeners.current.add(fn);
      return () => listeners.current.delete(fn);
    },
  });
  value.current.connected = connected;
  const ev: EventsState = { events, ...counts, more, markRead, ack, loadOlder };
  return <Ctx.Provider value={{ connected, agent, onLog: value.current.onLog, ev }}>{children}</Ctx.Provider>;
}

export const useLive = () => useContext(Ctx);
export const useEvents = () => useContext(Ctx).ev;
