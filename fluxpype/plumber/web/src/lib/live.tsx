import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { TaskState } from "./types";

type LogListener = (m: { task_id: string; start: number; offset: number; text: string }) => void;

/** The latest request an AI agent made through the MCP server; `ts` is browser time. */
export interface AgentPing {
  ts: number;
  tool: string;
  count: number;
}

interface LiveCtx {
  connected: boolean;
  agent: AgentPing | null;
  onLog: (fn: LogListener) => () => void;
}

const Ctx = createContext<LiveCtx>({ connected: false, agent: null, onLog: () => () => {} });

/** One WebSocket for the whole app: pushes update the query cache; polling is only the fallback. */
export function LiveProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [connected, setConnected] = useState(false);
  const [agent, setAgent] = useState<AgentPing | null>(null);
  const listeners = useRef(new Set<LogListener>());

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
          case "agent_activity":
            setAgent({ ts: Date.now(), tool: msg.data.tool, count: msg.data.count });
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
  }, [qc]);

  const value = useRef<LiveCtx>({
    connected: false,
    agent: null,
    onLog: (fn) => {
      listeners.current.add(fn);
      return () => listeners.current.delete(fn);
    },
  });
  value.current.connected = connected;
  return <Ctx.Provider value={{ connected, agent, onLog: value.current.onLog }}>{children}</Ctx.Provider>;
}

export const useLive = () => useContext(Ctx);
