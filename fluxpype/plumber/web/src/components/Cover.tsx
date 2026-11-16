import { useState, type ReactNode } from "react";
import { cn } from "../lib/cn";

/** Where each list's picture comes from (server: covers.py). `v` busts the browser cache when a run moves on. */
export const coverUrl = {
  job: (id: string, v?: string | number | null) => `/api/covers/job/${encodeURIComponent(id)}${v != null ? `?v=${encodeURIComponent(String(v))}` : ""}`,
  task: (id: string, v?: string | number | null) => `/api/covers/task/${encodeURIComponent(id)}${v != null ? `?v=${encodeURIComponent(String(v))}` : ""}`,
  preset: (name: string, v?: string | number | null) => `/api/covers/preset/${encodeURIComponent(name)}${v != null ? `?v=${encodeURIComponent(String(v))}` : ""}`,
  scene: (id: string, v?: string | number | null) => `/api/covers/scene/${encodeURIComponent(id)}${v != null ? `?v=${encodeURIComponent(String(v))}` : ""}`,
};

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

function rng(seed: string) {
  let x = hash(seed) || 1;
  return () => {
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    return ((x >>> 0) % 10000) / 10000;
  };
}

/**
 * A stand-in picture while a run has no image yet: a limb with field lines coloured the FLUX way
 * (red open north, blue open south, green closed loops), varied by `seed` so neighbours differ.
 * Decoration only; it shows no data.
 */
export function FieldArt({ seed, className, dim }: { seed: string; className?: string; dim?: boolean }) {
  const r = rng(seed);
  const tilt = (r() - 0.5) * 30;
  const cx = 160 + (r() - 0.5) * 60;
  const R = 70 + r() * 20;
  const cy = 120 + R * 0.55;
  const loops = 3 + Math.floor(r() * 3);
  const open = 3 + Math.floor(r() * 3);
  const paths: { d: string; c: string; w: number }[] = [];
  for (let i = 0; i < loops; i++) {
    const a = 0.25 + i * 0.13 + r() * 0.05;
    const x0 = cx - R * Math.sin(a), x1 = cx + R * Math.sin(a * 0.9 + 0.05);
    const y0 = cy - R * Math.cos(a), y1 = cy - R * Math.cos(a * 0.9 + 0.05);
    const h = 30 + i * 16 + r() * 10;
    paths.push({ d: `M ${x0.toFixed(1)} ${y0.toFixed(1)} C ${(x0 - 10).toFixed(1)} ${(y0 - h).toFixed(1)}, ${(x1 + 10).toFixed(1)} ${(y1 - h).toFixed(1)}, ${x1.toFixed(1)} ${y1.toFixed(1)}`, c: "var(--fa-g)", w: 1.4 });
  }
  for (let i = 0; i < open; i++) {
    const side = i % 2 === 0 ? -1 : 1;
    const a = side * (0.95 + Math.floor(i / 2) * 0.22 + r() * 0.08);
    const x0 = cx + R * Math.sin(a), y0 = cy - R * Math.cos(a);
    const x2 = cx + (R + 220) * Math.sin(a * 1.15), y2 = cy - (R + 220) * Math.cos(a * 1.15);
    paths.push({ d: `M ${x0.toFixed(1)} ${y0.toFixed(1)} Q ${(x0 + (x2 - x0) * 0.3).toFixed(1)} ${(y0 - 60).toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}`, c: side < 0 ? "var(--fa-r)" : "var(--fa-b)", w: 1.2 });
  }
  return (
    <svg viewBox="0 0 320 180" preserveAspectRatio="xMidYMid slice" aria-hidden="true" className={cn("block h-full w-full fa", dim && "opacity-60", className)}>
      <defs>
        <radialGradient id={`fa-glow-${hash(seed)}`} cx="50%" cy="75%" r="70%">
          <stop offset="0%" stopColor="var(--fa-glow)" stopOpacity="0.35" />
          <stop offset="100%" stopColor="var(--fa-glow)" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width="320" height="180" fill="var(--fa-bg)" />
      <rect width="320" height="180" fill={`url(#fa-glow-${hash(seed)})`} />
      <g transform={`rotate(${tilt.toFixed(1)} ${cx.toFixed(1)} ${cy.toFixed(1)})`} fill="none" strokeLinecap="round">
        {paths.map((p, i) => (
          <path key={i} d={p.d} stroke={p.c} strokeWidth={p.w} opacity={0.85} />
        ))}
        <circle cx={cx} cy={cy} r={R} fill="var(--fa-disk)" stroke="var(--fa-limb)" strokeWidth="1.2" />
      </g>
    </svg>
  );
}

/**
 * The picture at the head of a list item: the server's cover image when there is one, the field-line
 * art while it loads or when there is none. Children are overlaid (badges, a gradient, a caption).
 */
export function Cover({
  src,
  seed,
  className,
  children,
  imgClassName,
  fit = "cover",
}: {
  src?: string | null;
  seed: string;
  className?: string;
  children?: ReactNode;
  imgClassName?: string;
  fit?: "cover" | "contain";
}) {
  const [state, setState] = useState<{ src?: string | null; s: "loading" | "ok" | "none" }>({ src, s: src ? "loading" : "none" });
  const s = state.src === src ? state.s : src ? "loading" : "none";
  return (
    <div className={cn("relative isolate overflow-hidden bg-[rgb(var(--fa-bg-rgb))]", className)}>
      {s !== "ok" && <FieldArt seed={seed} dim={s === "loading"} />}
      {s === "loading" && <div className="pointer-events-none absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />}
      {src && s !== "none" && (
        <img
          src={src}
          alt=""
          loading="lazy"
          decoding="async"
          onLoad={() => setState({ src, s: "ok" })}
          onError={() => setState({ src, s: "none" })}
          className={cn(
            "absolute inset-0 h-full w-full transition-[opacity,transform] duration-500 ease-out",
            fit === "cover" ? "object-cover" : "object-contain",
            s === "ok" ? "opacity-100" : "opacity-0",
            imgClassName,
          )}
        />
      )}
      {children}
    </div>
  );
}
