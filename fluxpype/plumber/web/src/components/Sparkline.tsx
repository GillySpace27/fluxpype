import { useId } from "react";

/** Tiny SVG line with an area fill. `log` plots on a log scale (stiffness spans decades). */
export function Sparkline({ values, width = 120, height = 34, log = true, className }: { values: number[]; width?: number; height?: number; log?: boolean; className?: string }) {
  const id = useId();
  const v = values.filter((x) => isFinite(x) && x > 0);
  if (v.length < 2) {
    return (
      <svg width={width} height={height} className={className} aria-hidden>
        <line x1="0" y1={height - 2} x2={width} y2={height - 2} stroke="rgb(var(--line))" strokeDasharray="3 4" />
      </svg>
    );
  }
  const f = (x: number) => (log ? Math.log10(x) : x);
  const lo = Math.min(...v.map(f));
  const hi = Math.max(...v.map(f));
  const span = hi - lo || 1;
  const pad = 3;
  const pts = v.map((x, i) => [(i / (v.length - 1)) * (width - 2) + 1, pad + (1 - (f(x) - lo) / span) * (height - pad * 2)] as const);
  const line = pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = `${line} L${pts[pts.length - 1][0].toFixed(1)},${height} L${pts[0][0].toFixed(1)},${height} Z`;
  const [lx, ly] = pts[pts.length - 1];
  return (
    <svg width={width} height={height} className={className} aria-hidden>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgb(var(--accent))" stopOpacity="0.32" />
          <stop offset="100%" stopColor="rgb(var(--accent))" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${id})`} />
      <path d={line} fill="none" stroke="rgb(var(--accent))" strokeWidth="1.6" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={lx} cy={ly} r="2.4" fill="rgb(var(--accent))" />
    </svg>
  );
}
