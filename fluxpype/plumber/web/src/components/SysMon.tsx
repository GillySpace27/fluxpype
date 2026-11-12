import { useId, useState } from "react";
import { cn } from "../lib/cn";
import { bytes } from "../lib/format";
import { useSystem } from "../lib/hooks";
import type { SysSample, SysSnapshot } from "../lib/types";
import { Badge, Dialog } from "./ui";

type Tone = "accent" | "info" | "ok" | "run" | "fail";
const rgb = (t: Tone) => `rgb(var(--${t}))`;
/** Accent while calm, amber then red as a percentage climbs. */
const hot = (p: number | null | undefined): Tone => (p == null ? "accent" : p >= 90 ? "fail" : p >= 70 ? "run" : "accent");
const pct = (v: number | null | undefined) => (v == null ? "n/a" : `${v.toFixed(0)}%`);
const rate = (v: number | null | undefined) => (v == null ? "n/a" : `${bytes(Math.round(v))}/s`);
const sum = (a: number | null, b: number | null) => (a == null || b == null ? null : a + b);

/** Lines over the recent samples; `max` fixes the scale (100 for percentages), else it follows the data. */
export function Spark({ series, max, height = 18, className }: { series: { values: (number | null)[]; tone: Tone }[]; max?: number; height?: number; className?: string }) {
  const id = useId();
  const W = 100;
  const n = Math.max(0, ...series.map((s) => s.values.length));
  const top = max ?? Math.max(1, ...series.flatMap((s) => s.values.map((v) => v ?? 0))) * 1.15;
  return (
    <svg viewBox={`0 0 ${W} ${height}`} preserveAspectRatio="none" className={cn("w-full overflow-visible", className)} style={{ height }} aria-hidden>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="currentColor" stopOpacity="0.28" />
          <stop offset="100%" stopColor="currentColor" stopOpacity="0" />
        </linearGradient>
      </defs>
      <line x1="0" y1={height - 0.5} x2={W} y2={height - 0.5} stroke="rgb(var(--line))" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      {n > 1 &&
        series.map((s, k) => {
          let d = "";
          let first = -1;
          let last = 0;
          s.values.forEach((v, i) => {
            if (v == null) return;
            const x = (i / (n - 1)) * W;
            d += `${first < 0 ? "M" : "L"}${x.toFixed(1)},${(height - 1 - Math.min(1, v / top) * (height - 2)).toFixed(1)}`;
            if (first < 0) first = x;
            last = x;
          });
          if (!d) return null;
          return (
            <g key={k} style={{ color: rgb(s.tone) }}>
              {k === 0 && <path d={`${d} L${last.toFixed(1)},${height} L${first.toFixed(1)},${height} Z`} fill={`url(#${id})`} />}
              <path d={d} fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            </g>
          );
        })}
    </svg>
  );
}

const col = (h: SysSample[], k: keyof SysSample) => h.map((s) => s[k] as number | null);

function Row({ label, value, tone, children }: { label: string; value: string; tone?: Tone; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[30px_1fr_74px] items-center gap-2">
      <span className="text-[10px] font-semibold uppercase tracking-[0.1em] text-faint">{label}</span>
      <div className="min-w-0">{children}</div>
      <span className="num text-right text-[11.5px] font-medium" style={{ color: tone ? rgb(tone) : undefined }}>
        {value}
      </span>
    </div>
  );
}

/** The sidebar monitor: one line per resource. Click for the full picture. */
export function SysMonPanel() {
  const { data } = useSystem(false);
  const [open, setOpen] = useState(false);
  const h = data?.history ?? [];
  const now = data?.now;
  const data1 = data?.storage[0];
  return (
    <>
      <button
        className="w-full space-y-1.5 rounded-xl border border-line bg-raised/40 px-2.5 py-2 text-left transition-colors hover:bg-raised/70 focus:outline-none focus-visible:ring-1 focus-visible:ring-accent/60"
        onClick={() => setOpen(true)}
        aria-label="Open the system monitor"
      >
        <Row label="CPU" value={pct(now?.cpu)} tone={hot(now?.cpu)}>
          <Spark series={[{ values: col(h, "cpu"), tone: hot(now?.cpu) }]} max={100} />
        </Row>
        <Row label="MEM" value={pct(now?.mem)} tone={hot(now?.mem)}>
          <Spark series={[{ values: col(h, "mem"), tone: hot(now?.mem) }]} max={100} />
        </Row>
        <Row label="GPU" value={pct(now?.gpu)} tone={hot(now?.gpu)}>
          <Spark series={[{ values: col(h, "gpu"), tone: hot(now?.gpu) }]} max={100} />
        </Row>
        <Row label="NET" value={rate(sum(now?.net_rx ?? null, now?.net_tx ?? null))}>
          <Spark series={[{ values: col(h, "net_rx"), tone: "accent" }, { values: col(h, "net_tx"), tone: "info" }]} />
        </Row>
        <Row label="DSK" value={rate(sum(now?.disk_r ?? null, now?.disk_w ?? null))}>
          <Spark series={[{ values: col(h, "disk_r"), tone: "accent" }, { values: col(h, "disk_w"), tone: "info" }]} />
        </Row>
        <Row label="STO" value={pct(data1?.pct)} tone={hot(data1?.pct)}>
          <div className="h-1.5 overflow-hidden rounded-full bg-line">
            <div className="h-full rounded-full" style={{ width: `${data1?.pct ?? 0}%`, background: rgb(hot(data1?.pct)) }} />
          </div>
        </Row>
      </button>
      {open && <SysMonDialog onOpenChange={setOpen} />}
    </>
  );
}

function Card({ title, value, sub, children }: { title: string; value: React.ReactNode; sub?: React.ReactNode; children?: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-line bg-bg/40 p-3">
      <div className="flex items-baseline justify-between gap-2">
        <p className="text-[11px] uppercase tracking-[0.12em] text-faint">{title}</p>
        <p className="num text-[15px] font-semibold">{value}</p>
      </div>
      {sub && <p className="mt-0.5 text-[11px] text-muted">{sub}</p>}
      <div className="mt-2">{children}</div>
    </section>
  );
}

function Legend({ items }: { items: [string, Tone][] }) {
  return (
    <p className="mt-1 flex gap-3 text-[10.5px] text-faint">
      {items.map(([l, t]) => (
        <span key={l} className="flex items-center gap-1">
          <span className="h-1.5 w-3 rounded-full" style={{ background: rgb(t) }} /> {l}
        </span>
      ))}
    </p>
  );
}

function Cores({ cores }: { cores: number[] }) {
  return (
    <div className="mt-2 flex h-8 items-end gap-[3px]" title="Each bar is one core">
      {cores.map((c, i) => (
        <div key={i} className="flex h-full flex-1 items-end rounded-sm bg-line/60">
          <div className="w-full rounded-sm" style={{ height: `${Math.max(3, c)}%`, background: rgb(hot(c)) }} />
        </div>
      ))}
    </div>
  );
}

function Body({ d }: { d: SysSnapshot }) {
  const h = d.history;
  const n = d.now;
  const m = d.memory;
  return (
    <div className="space-y-4">
      {!d.psutil && (
        <p className="rounded-xl border border-run/30 bg-run/10 p-3 text-xs leading-relaxed">
          Only the load average and storage are available. Install <span className="font-mono">psutil</span> in Plumber's Python (<span className="font-mono">pip install psutil</span>) for memory, network, disk and per-process figures, then restart Plumber.
        </p>
      )}
      <div className="grid gap-3 sm:grid-cols-2">
        <Card title={d.psutil ? "CPU" : "CPU load"} value={pct(n.cpu)} sub={`${d.cpu_count} cores · load ${d.load.map((x) => x.toFixed(2)).join(" ")}`}>
          <Spark series={[{ values: col(h, "cpu"), tone: hot(n.cpu) }]} max={100} height={52} />
          {d.cores && <Cores cores={d.cores} />}
        </Card>
        <Card title="Memory" value={pct(n.mem)} sub={m ? `${bytes(m.used)} of ${bytes(m.total)} · swap ${bytes(m.swap_used)}` : "n/a"}>
          <Spark series={[{ values: col(h, "mem"), tone: hot(n.mem) }]} max={100} height={52} />
        </Card>
        <Card title="GPU" value={pct(n.gpu)} sub={d.gpu ? d.gpu.name + (d.gpu.mem_total ? ` · ${bytes(d.gpu.mem_used)} of ${bytes(d.gpu.mem_total)}` : "") : "No GPU readout on this machine"}>
          <Spark series={[{ values: col(h, "gpu"), tone: hot(n.gpu) }]} max={100} height={52} />
        </Card>
        <Card title="Network" value={rate(sum(n.net_rx, n.net_tx))} sub={`down ${rate(n.net_rx)} · up ${rate(n.net_tx)}`}>
          <Spark series={[{ values: col(h, "net_rx"), tone: "accent" }, { values: col(h, "net_tx"), tone: "info" }]} height={52} />
          <Legend items={[["down", "accent"], ["up", "info"]]} />
        </Card>
        <Card title="Disk I/O" value={rate(sum(n.disk_r, n.disk_w))} sub={`read ${rate(n.disk_r)} · write ${rate(n.disk_w)}`}>
          <Spark series={[{ values: col(h, "disk_r"), tone: "accent" }, { values: col(h, "disk_w"), tone: "info" }]} height={52} />
          <Legend items={[["read", "accent"], ["write", "info"]]} />
        </Card>
        <Card title="Storage" value={pct(d.storage[0]?.pct)} sub="Volumes Plumber uses">
          <div className="space-y-2.5">
            {d.storage.map((v) => (
              <div key={v.path}>
                <div className="flex justify-between text-[11.5px]">
                  <span className="font-medium">{v.label}</span>
                  <span className="num text-muted">
                    {bytes(v.free)} free of {bytes(v.total)}
                  </span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-line">
                  <div className="h-full rounded-full" style={{ width: `${v.pct}%`, background: rgb(hot(v.pct)) }} />
                </div>
                <p className="mt-0.5 truncate font-mono text-[10.5px] text-faint">{v.path}</p>
              </div>
            ))}
          </div>
        </Card>
      </div>
      {d.processes && (
        <>
          {d.processes.tasks.length > 0 && (
            <Table title="Plumber tasks running now" head={["Task", "Processes", "CPU", "Memory"]} rows={d.processes.tasks.map((t) => [t.label, String(t.procs), `${t.cpu.toFixed(0)}%`, bytes(t.rss)])} />
          )}
          <Table
            title="Busiest processes"
            head={["Process", "PID", "CPU", "Memory"]}
            rows={d.processes.top.map((p) => [
              <span key="n" className="flex items-center gap-2">
                <span className="truncate">{p.name}</span>
                {p.task && <Badge tone="accent">{p.task}</Badge>}
              </span>,
              String(p.pid),
              `${p.cpu.toFixed(0)}%`,
              bytes(p.rss),
            ])}
          />
          <p className="text-[11px] text-faint">CPU is per core, so a busy multi-threaded process can pass 100%. It reads zero until Plumber has measured twice.</p>
        </>
      )}
    </div>
  );
}

function Table({ title, head, rows }: { title: string; head: string[]; rows: React.ReactNode[][] }) {
  return (
    <section>
      <p className="mb-1.5 px-1 text-[11px] uppercase tracking-[0.12em] text-faint">{title}</p>
      <div className="overflow-hidden rounded-xl border border-line bg-bg/40">
        <table className="w-full text-[12px]">
          <thead>
            <tr className="border-b border-line/60 text-left text-[10.5px] uppercase tracking-wider text-faint">
              {head.map((x, i) => (
                <th key={x} className={cn("px-3 py-1.5 font-medium", i > 0 && "text-right")}>
                  {x}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-line/50">
            {rows.map((r, i) => (
              <tr key={i}>
                {r.map((c, j) => (
                  <td key={j} className={cn("max-w-[260px] px-3 py-1.5", j > 0 && "num text-right text-muted")}>
                    {c}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SysMonDialog({ onOpenChange }: { onOpenChange: (o: boolean) => void }) {
  const { data } = useSystem(true);
  return (
    <Dialog open wide onOpenChange={onOpenChange} title="System monitor" description="This computer, as Plumber sees it. Charts show the last few minutes; figures refresh every two seconds.">
      {data ? <Body d={data} /> : <p className="text-sm text-muted">Reading the machine...</p>}
    </Dialog>
  );
}
