export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !isFinite(seconds)) return "-";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, "0")}m`;
}

export function bytes(n: number | null | undefined): string {
  if (n == null) return "-";
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)} ${units[i]}`;
}

export function ago(epoch: number | null | undefined, now = Date.now() / 1000): string {
  if (!epoch) return "-";
  const d = now - epoch;
  if (d < 5) return "just now";
  if (d < 60) return `${Math.floor(d)}s ago`;
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
}

export function clock(epoch: number | null | undefined): string {
  if (!epoch) return "-";
  return new Date(epoch * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function dateTime(epoch: number | null | undefined): string {
  if (!epoch) return "-";
  return new Date(epoch * 1000).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function num(v: number | null | undefined, digits = 3): string {
  if (v == null || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-2)) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toString();
}

export function int(v: number | null | undefined): string {
  return v == null ? "-" : Math.round(v).toLocaleString();
}

export const instOf = (adapt: string) => (["", "0", "false", "none"].includes(String(adapt).toLowerCase()) ? "HMI" : "ADAPT");

export const taskTitle = (t: { cr: string; nflux: string; method: string }) => `CR ${t.cr} · ${t.nflux} fluxons · ${t.method}`;

export const kindLabel: Record<string, string> = {
  magnetogram: "Magnetograms",
  pfss_pickle: "PFSS pickles",
  footpoints: "Footpoints",
  world: "Worlds",
  relaxed_world: "Relaxed worlds",
  wind: "Wind",
  plots: "Plots",
  fluxlight: "FLUXlight",
  other: "Other",
};
