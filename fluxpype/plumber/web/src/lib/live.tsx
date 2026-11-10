import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { TaskState } from "./types";

type LogListener = (m: { task_id: string; start: number; offset: number; text: string }) => void;

interface LiveCtx {
  connected: boolean;
  onLog: (fn: LogListener) => () => void;
}

const Ctx = createContext<LiveCtx>({ connected: false, onLog: () => () => {} });

/** One WebSocket for the whole app: pushes update the query cache; polling is only the fallback. */
export function LiveProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [connected, setConnected] = useState(false);
  const listeners = useRef(new Set<LogListener>());

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let closed = false;
    let attempt = 0;
    let invalidateTimer: number | undefined;

    const soon = (keys: unknown[][]) => {
      window.clearTimeout(invalidateTimer);
      invalidateTimer = window.setTimeout(() => keys.forEach((k) => qc.invalidateQueries({ queryKey: k })), 120);
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
          case "job_removed":
            soon([["jobs"], ["overview"]]);
            break;
          case "queue_state":
            soon([["queue"], ["overview"]]);
            break;
          case "cache_changed":
            soon([["cache"], ["trash"]]);
            break;
          case "log_append":
            listeners.current.forEach((fn) => fn(msg.data));
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
    onLog: (fn) => {
      listeners.current.add(fn);
      return () => listeners.current.delete(fn);
    },
  });
  value.current.connected = connected;
  return <Ctx.Provider value={{ connected, onLog: value.current.onLog }}>{children}</Ctx.Provider>;
}

export const useLive = () => useContext(Ctx);
