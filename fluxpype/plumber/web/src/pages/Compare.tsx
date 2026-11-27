import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { Card, ErrorNote, PageHeader, Skeleton } from "../components/ui";
import { api } from "../lib/api";

/** GET /api/tasks/{id}/compare (A4-03): both images at a display size, packed as base64 float32. */
type Packed = { file: string; data?: string; prep?: string | null };
type Compare = {
  task_id: string;
  channel: string | null;
  channels: string[];
  available: boolean;
  note: string | null;
  aia_missing?: boolean;
  shape?: [number, number];
  wcs?: Record<string, number | string>;
  unit?: string | null;
  model?: Packed;
  aia?: Packed | null;
  metrics?: Record<string, number | null> | null;
  stretch?: [number, number] | null;
};

type View = { s: number; tx: number; ty: number }; // screen px per image px, and where the image's top-left sits
type Hover = { u: number; v: number } | null; // image coordinates (pixels, top-left origin, north up)
type Mode = "log" | "sqrt" | "linear";

const SIZE = 480; // canvas pixels per pane (CSS scales it)

function decode(b64: string): Float32Array {
  const bin = atob(b64);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Float32Array(buf.buffer);
}

/** An image canvas the size of the array, rows flipped so north is up. */
function paint(n: [number, number], pixel: (i: number) => [number, number, number]): HTMLCanvasElement {
  const [ny, nx] = n;
  const c = document.createElement("canvas");
  c.width = nx;
  c.height = ny;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(nx, ny);
  for (let row = 0; row < ny; row++) {
    for (let col = 0; col < nx; col++) {
      const [r, g, b] = pixel(row * nx + col);
      const o = ((ny - 1 - row) * nx + col) * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function level(v: number, lo: number, hi: number, mode: Mode): number {
  if (!Number.isFinite(v)) return 0;
  if (mode === "log") return v <= 0 ? 0 : (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
  const t = (v - lo) / (hi - lo);
  return mode === "sqrt" ? Math.sqrt(Math.max(t, 0)) : t;
}

/** Blue (model low) through white to red (model high), -2 to +2 decades. */
function diverging(m: number, a: number): [number, number, number] {
  if (!(m > 0 && a > 0)) return [0, 0, 0];
  const t = Math.max(-1, Math.min(1, Math.log10(m / a) / 2));
  return [255 - 255 * Math.max(-t, 0), 255 * (1 - Math.abs(t)), 255 - 255 * Math.max(t, 0)];
}

type PaneProps = { title: string; src: HTMLCanvasElement; view: View; setView: (v: View) => void; hover: Hover; setHover: (h: Hover) => void; limb: number | null; n: [number, number] };

/** One pane. The view and the cursor are the parent's, so every pane shows the same region. */
function Pane({ title, src, view, setView, hover, setHover, limb, n }: PaneProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const viewRef = useRef(view);
  viewRef.current = view;

  useEffect(() => {
    const c = ref.current!;
    const ctx = c.getContext("2d")!;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.imageSmoothingEnabled = view.s < 1;
    ctx.drawImage(src, view.tx, view.ty, n[1] * view.s, n[0] * view.s);
    if (limb) {
      ctx.strokeStyle = "rgba(255,255,255,0.7)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(view.tx + (n[1] / 2) * view.s, view.ty + (n[0] / 2) * view.s, limb * view.s, 0, 2 * Math.PI);
      ctx.stroke();
    }
    if (hover) {
      const x = view.tx + hover.u * view.s;
      const y = view.ty + hover.v * view.s;
      ctx.strokeStyle = "rgba(255,200,0,0.9)";
      ctx.beginPath();
      ctx.moveTo(x - 6, y);
      ctx.lineTo(x + 6, y);
      ctx.moveTo(x, y - 6);
      ctx.lineTo(x, y + 6);
      ctx.stroke();
    }
  }, [src, view, hover, limb, n]);

  // wheel zoom about the cursor; a native listener because React's wheel handler is passive
  useEffect(() => {
    const c = ref.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const px = ((e.clientX - r.left) * SIZE) / r.width;
      const py = ((e.clientY - r.top) * SIZE) / r.height;
      const v = viewRef.current;
      const s = Math.max(0.2, Math.min(80, v.s * Math.exp(-e.deltaY * 0.0015)));
      setView({ s, tx: px - ((px - v.tx) / v.s) * s, ty: py - ((py - v.ty) / v.s) * s });
    };
    c.addEventListener("wheel", onWheel, { passive: false });
    return () => c.removeEventListener("wheel", onWheel);
  }, [setView]);

  const at = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return { x: ((e.clientX - r.left) * SIZE) / r.width, y: ((e.clientY - r.top) * SIZE) / r.height };
  };
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 text-xs text-muted">{title}</figcaption>
      <canvas
        ref={ref}
        width={SIZE}
        height={SIZE}
        className="aspect-square w-full cursor-grab touch-none rounded-lg border border-line bg-black active:cursor-grabbing"
        aria-label={`${title}; drag to pan, scroll to zoom, linked with the other panes`}
        onPointerDown={(e) => {
          ref.current!.setPointerCapture(e.pointerId);
          drag.current = at(e);
        }}
        onPointerUp={() => (drag.current = null)}
        onPointerLeave={() => setHover(null)}
        onDoubleClick={() => setView({ s: SIZE / Math.max(n[0], n[1]), tx: 0, ty: 0 })}
        onPointerMove={(e) => {
          const p = at(e);
          if (drag.current) {
            setView({ ...view, tx: view.tx + p.x - drag.current.x, ty: view.ty + p.y - drag.current.y });
            drag.current = p;
          }
          setHover({ u: (p.x - view.tx) / view.s, v: (p.y - view.ty) / view.s });
        }}
      />
    </figure>
  );
}

const fmt = (v: number | null | undefined, d = 3) => (v == null || !Number.isFinite(v) ? "n/a" : Number(v).toPrecision(d));

export function ComparePage() {
  const { id, taskId } = useParams();
  const [sp, setSp] = useSearchParams();
  const channel = sp.get("channel") ?? undefined;
  // static once written: no interval polling (the WebSocket and refetch-on-focus decide when to look again)
  const { data, error, isLoading } = useQuery({
    queryKey: ["compare", taskId, channel],
    queryFn: () => api.get<Compare>(`/api/tasks/${taskId}/compare${channel ? `?channel=${channel}` : ""}`),
    staleTime: 60_000,
  });
  const back = `/jobs/${id}/${taskId}?tab=progress&stage=euv`;

  const [mode, setMode] = useState<Mode>("log");
  const [lo, setLo] = useState<string>("");
  const [hi, setHi] = useState<string>("");
  const [view, setView] = useState<View>({ s: 1, tx: 0, ty: 0 });
  const [hover, setHover] = useState<Hover>(null);
  const n = data?.shape;
  useEffect(() => {
    if (!n) return;
    setView({ s: SIZE / Math.max(n[0], n[1]), tx: 0, ty: 0 });
    if (data?.stretch) {
      setLo(String(data.stretch[0]));
      setHi(String(data.stretch[1]));
    }
  }, [data?.channel, n?.[0], n?.[1]]); // eslint-disable-line react-hooks/exhaustive-deps

  const model = useMemo(() => (data?.model?.data ? decode(data.model.data) : null), [data?.model?.data]);
  const aia = useMemo(() => (data?.aia?.data ? decode(data.aia.data) : null), [data?.aia?.data]);
  const l = Number(lo);
  const h = Number(hi);
  const okRange = Number.isFinite(l) && Number.isFinite(h) && h > l && (mode !== "log" || l > 0);
  const gray = (arr: Float32Array) =>
    n && okRange ? paint(n, (i) => { const g = 255 * Math.max(0, Math.min(1, level(arr[i], l, h, mode))); return [g, g, g]; }) : null;
  const modelImg = useMemo(() => (model ? gray(model) : null), [model, n, l, h, mode]); // eslint-disable-line react-hooks/exhaustive-deps
  const aiaImg = useMemo(() => (aia ? gray(aia) : null), [aia, n, l, h, mode]); // eslint-disable-line react-hooks/exhaustive-deps
  const ratioImg = useMemo(() => (model && aia && n ? paint(n, (i) => diverging(model[i], aia[i])) : null), [model, aia, n]);

  const w = data?.wcs;
  const limb = w && w.RSUN_OBS && w.CDELT1 ? Number(w.RSUN_OBS) / Math.abs(Number(w.CDELT1)) : null;
  // the cursor in arcsec and solar radii, from the shared WCS (CRVAL is 0; pixel centres are at integer FITS pixels)
  const readout = (() => {
    if (!hover || !n || !w) return null;
    const col = Math.floor(hover.u);
    const row = n[0] - 1 - Math.floor(hover.v);
    if (col < 0 || row < 0 || col >= n[1] || row >= n[0]) return null;
    const x = (hover.u + 0.5 - Number(w.CRPIX1)) * Number(w.CDELT1);
    const y = (n[0] - hover.v + 0.5 - Number(w.CRPIX2)) * Number(w.CDELT2);
    const rs = Number(w.RSUN_OBS);
    const m = model?.[row * n[1] + col];
    const a = aia?.[row * n[1] + col];
    return { x, y, r: rs ? Math.hypot(x, y) / rs : null, m, a };
  })();

  return (
    <div className="space-y-4">
      <PageHeader title="FLUXlight against AIA" sub={`Task ${taskId}${data?.channel ? `, AIA ${data.channel}` : ""}`} />
      <p className="text-xs"><Link className="text-accent hover:underline" to={back}>Back to the task</Link></p>
      {isLoading && <Skeleton className="h-64 w-full" />}
      {error && <ErrorNote error={error} />}
      {data && !data.available && <Card className="p-4 text-[13px]">{data.note}</Card>}
      {data?.available && n && (
        <>
          {data.aia_missing && <Card className="border-run/50 p-4 text-[13px]">{data.note}</Card>}
          <Card className="flex flex-wrap items-end gap-3 p-3 text-[13px]">
            {data.channels.length > 1 && (
              <label className="space-y-1">
                <span className="block text-xs text-muted">Channel</span>
                <select className="input" value={data.channel ?? ""} onChange={(e) => setSp({ channel: e.target.value })}>
                  {data.channels.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
            )}
            <label className="space-y-1">
              <span className="block text-xs text-muted">Stretch (both images)</span>
              <select className="input" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
                <option value="log">log</option>
                <option value="sqrt">square root</option>
                <option value="linear">linear</option>
              </select>
            </label>
            <label className="space-y-1">
              <span className="block text-xs text-muted">Min ({data.unit ?? "units"})</span>
              <input className="input w-28" value={lo} onChange={(e) => setLo(e.target.value)} inputMode="decimal" aria-invalid={!okRange} />
            </label>
            <label className="space-y-1">
              <span className="block text-xs text-muted">Max</span>
              <input className="input w-28" value={hi} onChange={(e) => setHi(e.target.value)} inputMode="decimal" aria-invalid={!okRange} />
            </label>
            <button className="btn btn-sm" onClick={() => setView({ s: SIZE / Math.max(n[0], n[1]), tx: 0, ty: 0 })}>Reset view</button>
            <p className="text-xs text-muted">Drag to pan, scroll to zoom: all panes move together. White circle: the solar limb.</p>
          </Card>
          {!okRange && <p className="text-[13px] text-fail">Min must be below max{mode === "log" ? " and above 0 for a log stretch" : ""}.</p>}
          <div className={`grid gap-3 ${aia ? "md:grid-cols-3" : "md:grid-cols-1 md:max-w-xl"}`}>
            {modelImg && <Pane title={`FLUXlight ${data.channel} (${data.model?.file})`} src={modelImg} {...{ view, setView, hover, setHover, limb, n }} />}
            {aiaImg && <Pane title={`AIA ${data.channel}, on the model grid (${data.aia?.file})`} src={aiaImg} {...{ view, setView, hover, setHover, limb, n }} />}
            {ratioImg && <Pane title="log10 model / AIA (blue: model lower, red: model higher, +-2)" src={ratioImg} {...{ view, setView, hover, setHover, limb, n }} />}
          </div>
          <p className="num h-5 font-mono text-[12px] text-muted" aria-live="off">
            {readout
              ? `x ${fmt(readout.x, 5)}"  y ${fmt(readout.y, 5)}"  r ${fmt(readout.r, 3)} R_sun  model ${fmt(readout.m)}${readout.a != null ? `  AIA ${fmt(readout.a)}` : ""} ${data.unit ?? ""}`
              : "Move over an image for the position and values."}
          </p>
          {data.metrics && (
            <Card className="p-3 text-[13px]">
              <h3 className="mb-1 text-[13px] font-medium">How close (from science/aia.py compare)</h3>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-4">
                <div><dt className="text-xs text-muted">Pixels compared</dt><dd className="num font-mono">{data.metrics.pixels_compared}</dd></div>
                <div><dt className="text-xs text-muted">Model / AIA, total</dt><dd className="num font-mono">{fmt(data.metrics.model_over_aia_total)}</dd></div>
                <div><dt className="text-xs text-muted">Median log10 ratio</dt><dd className="num font-mono">{fmt(data.metrics.median_log_ratio)}</dd></div>
                <div><dt className="text-xs text-muted">Log correlation</dt><dd className="num font-mono">{fmt(data.metrics.log_correlation)}</dd></div>
              </dl>
              {data.aia?.prep && <p className="mt-2 text-xs text-muted">AIA prep steps: {data.aia.prep}</p>}
            </Card>
          )}
        </>
      )}
    </div>
  );
}
