import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "../lib/cn";
import { useTheme } from "../lib/theme";
import type { DatData, DatPlotSpec } from "../lib/types";

/** Plot and table views of a numeric .dat file (see plumber/datfile.py). The default chart comes from the file kind the
 *  server recognised; any file can be re-plotted by choosing columns. */

const VIRIDIS = [
  [0.267, 0.005, 0.329],
  [0.231, 0.322, 0.545],
  [0.129, 0.569, 0.549],
  [0.369, 0.788, 0.384],
  [0.993, 0.906, 0.144],
];
const rgb = (c: number[]) => `rgb(${c.map((a) => Math.round(a * 255)).join(" ")})`;
const css = `linear-gradient(to right, ${VIRIDIS.map(rgb).join(", ")})`;
const POS = "rgb(245 115 128)"; // positive flux and the upper of two categories, as in the FLUX viewer
const NEG = "rgb(97 166 250)";

function viridis(t: number) {
  const x = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const k = Math.min(VIRIDIS.length - 2, Math.floor(x));
  return rgb(VIRIDIS[k].map((a, j) => a + (x - k) * (VIRIDIS[k + 1][j] - a)));
}

export function fmt(v: number | null | undefined) {
  if (v == null) return "nan";
  const a = Math.abs(v);
  return a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(3) : String(+v.toPrecision(6));
}

const colName = (d: DatData, i: number) => {
  const k = d.kind?.columns?.[i];
  return k ? (k.unit ? `${k.name} (${k.unit})` : k.name) : d.columns[i].name;
};

/** Round tick values covering [lo, hi]. */
function ticks(lo: number, hi: number, n = 6) {
  const raw = (hi - lo) / n || 1;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function Select({ label, value, onChange, items }: { label: string; value: number; onChange: (v: number) => void; items: [number, string][] }) {
  return (
    <label className="flex items-center gap-1.5 text-xs text-muted">
      {label}
      <select className="input w-auto max-w-[11rem] py-0.5 text-xs" value={value} onChange={(e) => onChange(+e.target.value)}>
        {items.map(([v, t]) => (
          <option key={v} value={v}>
            {t}
          </option>
        ))}
      </select>
    </label>
  );
}

function Chart({ d, s }: { d: DatData; s: DatPlotSpec }) {
  const { theme } = useTheme();
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState<[number, number]>([0, 0]);
  useEffect(() => {
    const c = canvas.current!;
    const ro = new ResizeObserver(() => setSize([c.clientWidth, c.clientHeight]));
    ro.observe(c);
    return () => ro.disconnect();
  }, []);

  // the colour column decides: two or a few distinct values are categories, otherwise a viridis ramp
  const colour = useMemo(() => {
    if (s.color == null) return null;
    const vals = d.data[s.color].filter((v): v is number => v != null);
    const uniq = [...new Set(vals)].sort((a, b) => a - b);
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const cat = uniq.length <= 4;
    const at = (v: number | null) => (v == null ? "gray" : cat ? (uniq.length === 2 ? (v === uniq[1] ? POS : NEG) : `hsl(${(uniq.indexOf(v) * 360) / uniq.length} 70% 60%)`) : viridis((v - lo) / (hi - lo || 1)));
    return { at, cat, uniq, lo, hi };
  }, [d, s.color]);

  useEffect(() => {
    const c = canvas.current!;
    const g = c.getContext("2d");
    const [w, h] = size;
    if (!g || !w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr;
    c.height = h * dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const tok = (n: string) => `rgb(${getComputedStyle(document.documentElement).getPropertyValue(`--${n}`).trim()})`;
    const [ink, muted, line] = [tok("ink"), tok("muted"), tok("line")];
    const xs = d.data[s.x];
    const ys = d.data[s.y];
    const ly = !!s.logy;
    const ok = (i: number) => xs[i] != null && ys[i] != null && (!ly || (ys[i] as number) > 0);
    const yv = (v: number) => (ly ? Math.log10(v) : v);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < xs.length; i++)
      if (ok(i)) {
        x0 = Math.min(x0, xs[i]!); x1 = Math.max(x1, xs[i]!);
        y0 = Math.min(y0, yv(ys[i]!)); y1 = Math.max(y1, yv(ys[i]!));
      }
    g.font = "11px ui-sans-serif, system-ui, sans-serif";
    if (!(x0 <= x1)) {
      g.fillStyle = muted;
      g.fillText("No finite points for these columns.", 16, 24);
      return;
    }
    const px = (x1 - x0) * 0.04 || 1;
    const py = (y1 - y0) * 0.04 || 1;
    [x0, x1, y0, y1] = [x0 - px, x1 + px, y0 - py, y1 + py];
    const L = 62, R = 14, T = 10, B = 38;
    const X = (v: number) => L + ((v - x0) / (x1 - x0)) * (w - L - R);
    const Y = (v: number) => h - B - ((yv(v) - y0) / (y1 - y0)) * (h - T - B);
    // grid, ticks and axis titles
    g.strokeStyle = line;
    g.fillStyle = muted;
    g.lineWidth = 1;
    g.textAlign = "center";
    for (const t of ticks(x0, x1)) {
      const x = X(t);
      g.globalAlpha = 0.4;
      g.beginPath(); g.moveTo(x, T); g.lineTo(x, h - B); g.stroke();
      g.globalAlpha = 1;
      g.fillText(fmt(t), x, h - B + 14);
    }
    g.textAlign = "right";
    const yt = ly ? ticks(Math.ceil(y0), Math.floor(y1), 8).filter(Number.isInteger) : ticks(y0, y1);
    for (const t of yt) {
      const y = h - B - ((t - y0) / (y1 - y0)) * (h - T - B);
      g.globalAlpha = 0.4;
      g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke();
      g.globalAlpha = 1;
      g.fillText(ly ? `1e${t}` : fmt(t), L - 6, y + 3);
    }
    g.strokeStyle = muted;
    g.strokeRect(L, T, w - L - R, h - T - B);
    g.fillStyle = ink;
    g.textAlign = "center";
    g.fillText(colName(d, s.x), L + (w - L - R) / 2, h - 6);
    g.save();
    g.translate(12, T + (h - T - B) / 2);
    g.rotate(-Math.PI / 2);
    g.fillText(colName(d, s.y) + (ly ? ", log scale" : ""), 0, 0);
    g.restore();
    // data, clipped to the frame
    g.save();
    g.beginPath();
    g.rect(L, T, w - L - R, h - T - B);
    g.clip();
    const accent = tok("accent");
    if (s.type === "lines") {
      const grp = s.group != null ? d.data[s.group] : null;
      const hue = (k: number) => `hsl(${((k * 137.508) % 360)} 65% ${theme === "dark" ? 62 : 42}%)`;
      g.lineWidth = 1;
      g.globalAlpha = 0.75;
      let k = -1;
      let prev: number | null | undefined;
      let open = false;
      for (let i = 0; i < xs.length; i++) {
        if (!ok(i)) { open = false; continue; }
        if (!grp || grp[i] !== prev) {
          if (open) g.stroke();
          k++;
          prev = grp ? grp[i] : null;
          g.beginPath();
          g.strokeStyle = colour ? colour.at(d.data[s.color!][i]) : grp ? hue(k) : accent;
          g.moveTo(X(xs[i]!), Y(ys[i]!));
          open = true;
        } else g.lineTo(X(xs[i]!), Y(ys[i]!));
      }
      if (open) g.stroke();
    } else {
      g.globalAlpha = 0.8;
      for (let i = 0; i < xs.length; i++) {
        if (!ok(i)) continue;
        g.fillStyle = colour ? colour.at(d.data[s.color!][i]) : accent;
        g.beginPath();
        g.arc(X(xs[i]!), Y(ys[i]!), 2.4, 0, 6.2832);
        g.fill();
      }
    }
    g.restore();
  }, [d, s, size, colour, theme]);

  return (
    <div className="space-y-1.5">
      <canvas ref={canvas} className="h-[46vh] min-h-[280px] w-full rounded-xl border border-line bg-bg/60" aria-label={`${colName(d, s.y)} against ${colName(d, s.x)}`} />
      {colour && s.color != null && (
        <div className="num flex flex-wrap items-center gap-2 text-[11px] text-muted">
          {colName(d, s.color)}
          {colour.cat ? (
            colour.uniq.map((v) => (
              <span key={v} className="flex items-center gap-1">
                <span className="h-2 w-2 rounded-full" style={{ background: colour.at(v) }} />
                {fmt(v)}
              </span>
            ))
          ) : (
            <>
              {fmt(colour.lo)}
              <span className="h-1.5 w-24 rounded-full" style={{ background: css }} />
              {fmt(colour.hi)}
            </>
          )}
        </div>
      )}
    </div>
  );
}

function Notes({ d }: { d: DatData }) {
  return (
    <div className="space-y-0.5 text-[11.5px] text-muted">
      <p>
        {d.kind ? <span className="font-medium text-ink">{d.kind.label}</span> : <span className="font-medium text-ink">Unrecognised table</span>}
        <span className="num">
          {" "}
          · {d.rows} rows · {d.columns.length} columns
        </span>
        {d.bad_rows > 0 && <span className="text-run"> · {d.bad_rows} malformed rows skipped</span>}
        {d.decimated && <span className="text-run"> · showing every {d.decimated.step}th row ({d.decimated.rows_kept} of {d.decimated.rows_total})</span>}
      </p>
      {d.kind ? (
        <p>
          Written by <span className="font-mono">science/{d.kind.writer}</span>.
        </p>
      ) : (
        <p>No column names or units are known for this file; columns are shown as numbered.</p>
      )}
      {d.kind?.notes.map((n) => <p key={n}>{n}</p>)}
    </div>
  );
}

export function DatPlot({ d }: { d: DatData }) {
  const n = d.columns.length;
  const start: DatPlotSpec = d.kind?.plot ?? { type: "scatter", x: 0, y: Math.min(1, n - 1) };
  const [s, setS] = useState<DatPlotSpec>(start);
  const set = (p: Partial<DatPlotSpec>) => setS((o) => ({ ...o, ...p }));
  const cols: [number, string][] = d.columns.map((_, i) => [i, colName(d, i)]);
  const opt = (v: number | undefined) => (v == null ? -1 : v);
  if (n < 2 || d.rows === 0) return <p className="rounded-xl border border-line bg-bg/40 px-3 py-6 text-center text-[13px] text-muted">This file has no two numeric columns to plot; the Table tab shows what was read.</p>;
  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <div className="inline-flex rounded-lg border border-line bg-bg/50 p-0.5" role="radiogroup" aria-label="Chart type">
          {(["scatter", "lines"] as const).map((t) => (
            <button key={t} role="radio" aria-checked={s.type === t} onClick={() => set({ type: t })} className={cn("rounded-md px-2 py-0.5 text-xs font-medium capitalize transition-colors", s.type === t ? "bg-raised text-ink shadow-sm" : "text-muted hover:text-ink")}>
              {t}
            </button>
          ))}
        </div>
        <Select label="x" value={s.x} onChange={(x) => set({ x })} items={cols} />
        <Select label="y" value={s.y} onChange={(y) => set({ y })} items={cols} />
        <Select label="one line per" value={opt(s.group)} onChange={(v) => set({ group: v < 0 ? undefined : v })} items={[[-1, "none"], ...cols]} />
        <Select label="colour" value={opt(s.color)} onChange={(v) => set({ color: v < 0 ? undefined : v })} items={[[-1, "none"], ...cols]} />
        <label className="flex items-center gap-1.5 text-xs text-muted">
          <input type="checkbox" checked={!!s.logy} onChange={(e) => set({ logy: e.target.checked })} /> log y
        </label>
        {d.kind && <button className="btn btn-sm" onClick={() => setS(start)}>Default</button>}
      </div>
      <Chart d={d} s={s} />
      <Notes d={d} />
    </div>
  );
}

const SHOWN = 500; // rows drawn in the table

export function DatTable({ d }: { d: DatData }) {
  const [sort, setSort] = useState<{ c: number; dir: 1 | -1 } | null>(null);
  const order = useMemo(() => {
    const idx = Array.from({ length: d.data[0]?.length ?? 0 }, (_, i) => i);
    if (sort) {
      const col = d.data[sort.c];
      idx.sort((a, b) => ((col[a] ?? Infinity) - (col[b] ?? Infinity)) * sort.dir || a - b); // nan last
    }
    return idx.slice(0, SHOWN);
  }, [d, sort]);
  return (
    <div className="space-y-1.5">
      <div className="max-h-[60vh] overflow-auto rounded-xl border border-line bg-bg/60">
        <table className="num w-full border-collapse text-[11.5px]">
          <thead className="sticky top-0 bg-surface">
            <tr>
              <th className="px-2 py-1 text-right font-normal text-faint">#</th>
              {d.columns.map((_, i) => (
                <th key={i} className="whitespace-nowrap px-2 py-1 text-right font-medium">
                  <button className="hover:text-accent" onClick={() => setSort((o) => (o?.c === i ? (o.dir === 1 ? { c: i, dir: -1 } : null) : { c: i, dir: 1 }))} aria-label={`Sort by ${colName(d, i)}`}>
                    {colName(d, i)} {sort?.c === i ? (sort.dir === 1 ? "▲" : "▼") : ""}
                  </button>
                </th>
              ))}
            </tr>
            {(
              [
                ["min", "min"],
                ["max", "max"],
                ["mean", "mean"],
              ] as const
            ).map(([k, label]) => (
              <tr key={k} className="text-faint">
                <td className="px-2 text-right">{label}</td>
                {d.columns.map((c, i) => (
                  <td key={i} className="px-2 text-right">
                    {fmt(c[k])}
                  </td>
                ))}
              </tr>
            ))}
            {d.columns.some((c) => c.nonfinite > 0) && (
              <tr className="text-run">
                <td className="px-2 text-right">nan</td>
                {d.columns.map((c, i) => (
                  <td key={i} className="px-2 text-right">
                    {c.nonfinite || ""}
                  </td>
                ))}
              </tr>
            )}
          </thead>
          <tbody>
            {order.map((r) => (
              <tr key={r} className="border-t border-line/40 hover:bg-raised/50">
                <td className="px-2 text-right text-faint">{r + 1}</td>
                {d.data.map((col, i) => (
                  <td key={i} className="px-2 text-right">
                    {fmt(col[r])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-faint">
        {order.length < (d.data[0]?.length ?? 0) ? `First ${order.length} rows${sort ? " of the sorted table" : ""}. ` : ""}Min, max and mean are over every row in the file.
      </p>
    </div>
  );
}
