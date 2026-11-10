import { useEffect, useMemo, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { useTheme, token } from "../lib/theme";
import { num } from "../lib/format";
import type { RelaxCycle, RelaxPoint } from "../lib/types";
import { EmptyState } from "./ui";
import { Activity } from "lucide-react";

const FONT = '11px "Inter Variable", ui-sans-serif, system-ui, sans-serif';

/** Step (x) at which each cycle ended: the last relax point at or before the cycle event. */
function boundaries(points: RelaxPoint[], cycles: RelaxCycle[]): { cycle: RelaxCycle; step: number }[] {
  const out: { cycle: RelaxCycle; step: number }[] = [];
  for (const c of cycles) {
    let step: number | null = null;
    if (c.t != null) {
      for (const p of points) {
        if (p.t != null && p.t <= c.t) step = p.step;
        else break;
      }
    }
    if (step == null) continue;
    out.push({ cycle: c, step });
  }
  return out;
}

function thresholdSeries(points: RelaxPoint[], b: { cycle: RelaxCycle; step: number }[]): (number | null)[] {
  return points.map((p) => {
    const k = b.findIndex((x) => p.step <= x.step);
    const idx = k === -1 ? b.length : k; // cycle the point belongs to
    if (idx === 0) {
      const first = b[0]?.cycle.threshold;
      return first ? first / 1.1 : null;
    }
    return b[idx - 1]?.cycle.threshold ?? null;
  });
}

interface ChartDef {
  title: string;
  hint: string;
  series: { label: string; color: string; values: (number | null)[]; dash?: number[]; stepped?: boolean; width?: number }[];
  markers: number[];
  yLabel: string;
}

function Chart({ def, xs, height, onCursor }: { def: ChartDef; xs: number[]; height: number; onCursor: (i: number | null) => void }) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const { theme } = useTheme();
  const markersRef = useRef<number[]>([]);
  markersRef.current = def.markers;
  const cursorRef = useRef(onCursor);
  cursorRef.current = onCursor;
  const shape = def.series.map((s) => s.label).join("|");

  // (re)create when the theme or the series layout changes
  useEffect(() => {
    if (!host.current) return;
    const grid = token("line", 0.55);
    const text = token("muted");
    const opts: uPlot.Options = {
      width: host.current.clientWidth || 400,
      height,
      padding: [10, 10, 0, 0],
      legend: { show: false },
      cursor: { sync: { key: "relax", setSeries: false }, drag: { x: true, y: false, setScale: true }, points: { size: 7, width: 2 } },
      scales: { x: { time: false }, y: { distr: 3, log: 10 } },
      axes: [
        { stroke: text, font: FONT, grid: { stroke: grid, width: 1 }, ticks: { stroke: grid, size: 4 }, gap: 6, size: 30 },
        {
          stroke: text,
          font: FONT,
          grid: { stroke: grid, width: 1 },
          ticks: { stroke: grid, size: 4 },
          gap: 6,
          size: 52,
          values: (_u, v) => v.map((x) => (x == null ? "" : num(x, 2))),
        },
      ],
      series: [
        {},
        ...def.series.map((s) => ({
          label: s.label,
          stroke: s.color,
          width: s.width ?? 1.8,
          dash: s.dash,
          points: { show: false },
          spanGaps: true,
          paths: s.stepped ? uPlot.paths.stepped!({ align: 1 }) : undefined,
        })),
      ],
      hooks: {
        draw: [
          (u) => {
            const ctx = u.ctx;
            ctx.save();
            ctx.strokeStyle = token("faint", 0.5);
            ctx.setLineDash([2, 4]);
            ctx.lineWidth = 1;
            for (const x of markersRef.current) {
              const px = Math.round(u.valToPos(x, "x", true));
              if (px < u.bbox.left || px > u.bbox.left + u.bbox.width) continue;
              ctx.beginPath();
              ctx.moveTo(px, u.bbox.top);
              ctx.lineTo(px, u.bbox.top + u.bbox.height);
              ctx.stroke();
            }
            ctx.restore();
          },
        ],
        setCursor: [(u) => cursorRef.current(u.cursor.idx ?? null)],
      },
    };
    const data = [xs, ...def.series.map((s) => s.values)] as uPlot.AlignedData;
    const p = new uPlot(opts, data, host.current);
    plot.current = p;
    const ro = new ResizeObserver(() => host.current && p.setSize({ width: host.current.clientWidth, height }));
    ro.observe(host.current);
    return () => {
      ro.disconnect();
      p.destroy();
      plot.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, shape, height]);

  useEffect(() => {
    plot.current?.setData([xs, ...def.series.map((s) => s.values)] as uPlot.AlignedData);
  }, [xs, def]);

  return <div ref={host} className="w-full" style={{ height }} onMouseLeave={() => onCursor(null)} />;
}

export function RelaxCharts({ points, cycles, height = 210 }: { points: RelaxPoint[]; cycles: RelaxCycle[]; height?: number }) {
  const { theme } = useTheme();
  const [hover, setHover] = useState<number | null>(null);
  const pts = useMemo(() => points.filter((p) => p.step != null), [points]);
  const xs = useMemo(() => pts.map((p) => p.step), [pts]);
  const bounds = useMemo(() => boundaries(pts, cycles), [pts, cycles]);
  const markers = useMemo(() => bounds.map((b) => b.step), [bounds]);

  const defs = useMemo(() => {
    const pos = (v: number | null | undefined) => (v != null && isFinite(v) && v > 0 ? v : null);
    const stiff: ChartDef = {
      title: "Stiffness",
      hint: "f_av / fs_av, percent. Relaxation stops when it falls below the threshold line.",
      yLabel: "%",
      markers,
      series: [
        { label: "Stiffness", color: token("accent"), values: pts.map((p) => pos(p.stiffness)), width: 2 },
        { label: "Threshold", color: token("info"), values: thresholdSeries(pts, bounds), dash: [5, 4], stepped: true, width: 1.4 },
      ],
    };
    const forces: ChartDef = {
      title: "Forces",
      hint: "fmax, f_av and fs_av in code units.",
      yLabel: "",
      markers,
      series: [
        { label: "fmax", color: token("fail"), values: pts.map((p) => pos(p.fmax)), width: 1.6 },
        { label: "f_av", color: token("ok"), values: pts.map((p) => pos(p.f_av)), width: 1.6 },
        { label: "fs_av", color: token("run"), values: pts.map((p) => pos(p.fs_av)), width: 1.6 },
      ],
    };
    return [stiff, forces];
    // theme is in the deps so token colors are re-read after a switch
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pts, bounds, markers, theme]);

  if (pts.length < 2) {
    return <EmptyState icon={<Activity className="h-5 w-5" />} title="Waiting for relaxation data" body="Charts appear as soon as the relaxation stage starts reporting steps." />;
  }
  const h = hover != null && hover < pts.length ? pts[hover] : pts[pts.length - 1];
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {defs.map((d, i) => (
        <div key={d.title} className="min-w-0 rounded-xl border border-line/70 bg-bg/30 p-3">
          <div className="mb-1 flex items-baseline justify-between gap-2 px-1">
            <div className="min-w-0">
              <h3 className="text-[13px]">{d.title}</h3>
              <p className="truncate text-[11px] text-faint">{d.hint}</p>
            </div>
            <div className="num shrink-0 text-right text-[11px] text-muted">
              <span className="text-faint">step </span>
              {h.step}
              {i === 0 ? (
                <>
                  <span className="text-faint"> · </span>
                  <span style={{ color: token("accent") }}>{num(h.stiffness, 3)}%</span>
                </>
              ) : (
                <>
                  <span className="text-faint"> · fmax </span>
                  {num(h.fmax, 2)}
                </>
              )}
            </div>
          </div>
          <Chart def={d} xs={xs} height={height} onCursor={setHover} />
          <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-[11px] text-muted">
            {d.series.map((s) => (
              <span key={s.label} className="inline-flex items-center gap-1.5">
                <span className="inline-block h-0.5 w-3.5 rounded" style={{ background: s.color, opacity: s.dash ? 0.9 : 1 }} />
                {s.label}
              </span>
            ))}
            {i === 0 && markers.length > 0 && <span className="text-faint">dotted: cycle ends</span>}
          </div>
        </div>
      ))}
    </div>
  );
}
