import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowDownToLine, Search, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { useLive } from "../lib/live";
import type { LogChunk } from "../lib/types";
import { Switch } from "./ui";

const ANSI = /\x1b\[[0-9;]*[A-Za-z]/g;
const MAX_LINES = 20000;
const ANCHOR = /^\s*\((pdl|py)\) (.+)$/;
const ROW = 18;

function lineTone(line: string): string {
  if (/failed|Error|Traceback|TOO MANY|Timeout|terminated/i.test(line)) return "text-fail";
  if (/^\s*\((pdl|py)\) /.test(line) || /Program Complete/.test(line)) return "text-accent font-medium";
  if (/^step \d+:|^\(\d+ vertices|Cycle \d+/.test(line)) return "text-muted";
  if (/^~{10,}$|^`{5,}|^-{10,}$/.test(line.trim())) return "text-faint/70";
  return "text-ink/85";
}

function Highlight({ text, q }: { text: string; q: string }) {
  if (!q) return <>{text}</>;
  const i = text.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, i)}
      <mark className="rounded-sm bg-accent/35 px-px text-ink">{text.slice(i, i + q.length)}</mark>
      <Highlight text={text.slice(i + q.length)} q={q} />
    </>
  );
}

export function LogPane({ taskId, active }: { taskId: string; active: boolean }) {
  const { connected, onLog } = useLive();
  const [lines, setLines] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState("");
  const [follow, setFollow] = useState(true);
  const [file, setFile] = useState<string | null>(null);
  const offset = useRef(0);
  const carry = useRef("");
  const scroller = useRef<HTMLDivElement>(null);

  const ingest = useCallback((text: string, replace = false) => {
    const clean = text.replace(ANSI, "");
    const all = (replace ? "" : carry.current) + clean;
    const parts = all.split("\n");
    carry.current = parts.pop() ?? "";
    const fresh = parts.map((l) => l.replace(/\r/g, ""));
    setLines((cur) => {
      const next = replace ? fresh : cur.concat(fresh);
      return next.length > MAX_LINES ? next.slice(next.length - MAX_LINES) : next;
    });
  }, []);

  const fetchFrom = useCallback(
    async (from: number | null) => {
      const url = `/api/tasks/${taskId}/log` + (from == null ? "" : `?offset=${from}`);
      const chunk = await api.get<LogChunk>(url);
      setFile(chunk.file);
      if (from == null) {
        offset.current = chunk.offset;
        ingest(chunk.text, true);
      } else if (chunk.text) {
        offset.current = chunk.offset;
        ingest(chunk.text);
      }
    },
    [taskId, ingest],
  );

  // initial load (and when the task changes)
  useEffect(() => {
    setLines([]);
    setLoading(true);
    carry.current = "";
    offset.current = 0;
    fetchFrom(null)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [taskId, fetchFrom]);

  // live appends over the WebSocket; polling only when it is down
  useEffect(() => {
    if (!active) return;
    const off = onLog((m) => {
      if (m.task_id !== taskId) return;
      if (m.start === offset.current) {
        offset.current = m.offset;
        ingest(m.text);
      } else if (m.start > offset.current) {
        fetchFrom(offset.current).catch(() => {});
      }
    });
    let timer: number | undefined;
    if (!connected) timer = window.setInterval(() => fetchFrom(offset.current).catch(() => {}), 2000);
    return () => {
      off();
      window.clearInterval(timer);
    };
  }, [active, connected, taskId, onLog, ingest, fetchFrom]);

  const view = useMemo(() => {
    if (!q) return lines.map((text, i) => ({ text, i }));
    const needle = q.toLowerCase();
    return lines.reduce<{ text: string; i: number }[]>((acc, text, i) => {
      if (text.toLowerCase().includes(needle)) acc.push({ text, i });
      return acc;
    }, []);
  }, [lines, q]);

  const anchors = useMemo(() => {
    const out: { label: string; at: number }[] = [];
    lines.forEach((l, i) => {
      const m = ANCHOR.exec(l);
      if (m) out.push({ label: m[2].replace(/ for CR\d+.*/, "").slice(0, 34), at: i });
    });
    return out;
  }, [lines]);

  const virt = useVirtualizer({ count: view.length, getScrollElement: () => scroller.current, estimateSize: () => ROW, overscan: 30 });

  useEffect(() => {
    if (follow && view.length) virt.scrollToIndex(view.length - 1, { align: "end" });
  }, [view.length, follow, virt]);

  const jump = (at: number) => {
    setFollow(false);
    setQ("");
    requestAnimationFrame(() => virt.scrollToIndex(at, { align: "start" }));
  };

  return (
    <div className="flex flex-col overflow-hidden rounded-xl border border-line/80 bg-bg/60">
      <div className="flex flex-wrap items-center gap-2 border-b hair px-3 py-2">
        <div className="relative min-w-[10rem] flex-1 sm:max-w-xs">
          <Search className="pointer-events-none absolute left-2.5 top-2 h-3.5 w-3.5 text-faint" />
          <input className="input h-7 pl-8 pr-7 text-xs" placeholder="Filter log lines" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Filter log lines" />
          {q && (
            <button className="absolute right-1.5 top-1.5 text-faint hover:text-ink" onClick={() => setQ("")} aria-label="Clear filter">
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
        <span className="num text-[11px] text-faint">{q ? `${view.length} of ${lines.length} lines` : `${lines.length.toLocaleString()} lines`}</span>
        <div className="ml-auto flex items-center gap-2">
          {file && <span className="hidden font-mono text-[11px] text-faint sm:inline">{file}</span>}
          <label className="flex items-center gap-2 text-xs text-muted">
            <ArrowDownToLine className="h-3.5 w-3.5" />
            Follow
            <Switch checked={follow} onChange={setFollow} label="Follow the end of the log" />
          </label>
        </div>
      </div>
      {anchors.length > 0 && !q && (
        <div className="flex gap-1.5 overflow-x-auto border-b hair px-3 py-1.5">
          <span className="label shrink-0 self-center whitespace-nowrap pr-1">Jump to</span>
          {anchors.map((a) => (
            <button key={a.at} onClick={() => jump(a.at)} className="shrink-0 rounded-md border border-line bg-raised px-2 py-0.5 text-[11px] text-muted transition-colors hover:border-accent/50 hover:text-ink">
              {a.label}
            </button>
          ))}
        </div>
      )}
      <div
        ref={scroller}
        className="h-[360px] overflow-auto font-mono"
        onWheel={(e) => e.deltaY < 0 && follow && setFollow(false)}
        role="log"
        aria-label="Task log"
        tabIndex={0}
      >
        {loading ? (
          <p className="p-4 text-xs text-faint">Loading log...</p>
        ) : view.length === 0 ? (
          <p className="p-4 text-xs text-faint">{q ? "No lines match." : active ? "Waiting for output..." : "This task wrote no log."}</p>
        ) : (
          <div style={{ height: virt.getTotalSize(), minWidth: "max-content", position: "relative" }}>
            {virt.getVirtualItems().map((v) => {
              const row = view[v.index];
              return (
                <div key={v.key} className="absolute left-0 flex w-full" style={{ top: 0, height: ROW, transform: `translateY(${v.start}px)` }}>
                  <span className="logline num sticky left-0 w-12 shrink-0 select-none bg-bg/90 pr-2 text-right text-faint/60">{row.i + 1}</span>
                  <span className={cn("logline pl-2", lineTone(row.text))}>
                    <Highlight text={row.text || " "} q={q} />
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
