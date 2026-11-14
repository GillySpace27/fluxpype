import {
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Points,
  PointsMaterial,
  Quaternion,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
  WireframeGeometry,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { RotateCcw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { cn } from "../lib/cn";
import { useFluxWorld } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import type { FluxWorld } from "../lib/types";
import { ErrorNote, Skeleton } from "./ui";

/** Interactive 3-D view of .flux world files: fluxons as polylines, concentrations as points, the boundary sphere.
 *  `base` is the API prefix, as for `fileUrl` in FileViewer ("/api/cache" or `jobDataBase(id)`). Loads three.js, so
 *  import it lazily where the bundle size matters. */
type Vec3 = [number, number, number];
type Fit = "all" | "sun";
interface Opts {
  fluxons: boolean;
  conc: boolean;
  sphere: boolean;
  colorBy: "id" | "radius" | "theta" | "polarity";
  width: number;
  fit: Fit;
}
interface Cam {
  camera: PerspectiveCamera;
  controls: OrbitControls;
  R: number; // radius of the data
  R0: number; // radius of the boundary sphere
  placed: boolean;
  draw: () => void;
}
/** Cameras that move together (compare mode). */
interface Link {
  views: Set<Cam>;
  busy: boolean;
}

const HOME: Vec3 = [1, -1, 0.6];
// z is solar north; the tiny offsets keep the z views off the pole singularity of the orbit controls
const PRESETS: [string, Vec3][] = [
  ["+x", [1, 0, 0]],
  ["-x", [-1, 0, 0]],
  ["+y", [0, 1, 0]],
  ["-y", [0, -1, 0]],
  ["+z", [0, -1e-3, 1]],
  ["-z", [0, 1e-3, -1]],
];
// viridis at 0, .25, .5, .75, 1 (sRGB)
const VIRIDIS = [
  [0.267, 0.005, 0.329],
  [0.231, 0.322, 0.545],
  [0.129, 0.569, 0.549],
  [0.369, 0.788, 0.384],
  [0.993, 0.906, 0.144],
];
// diverging blue, grey, red (RdBu, with a mid grey that stays visible on both themes): inward 180 deg, tangential 90, outward 0
const DIVERGING = [
  [0.129, 0.4, 0.675],
  [0.263, 0.576, 0.765],
  [0.557, 0.557, 0.557],
  [0.839, 0.376, 0.302],
  [0.698, 0.094, 0.169],
];
const css = (stops: number[][]) => `linear-gradient(to right, ${stops.map(([r, g, b]) => `rgb(${r * 255 | 0} ${g * 255 | 0} ${b * 255 | 0})`).join(", ")})`;
const POS: Vec3 = [0.96, 0.45, 0.5]; // positive flux (also the concentration points)
const NEG: Vec3 = [0.38, 0.65, 0.98];
const SHUT: Vec3 = [0.6, 0.6, 0.6]; // closed loops have a footpoint of each sign

function ramp(stops: number[][], t: number, c: Color) {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const k = Math.min(stops.length - 2, Math.floor(x));
  const f = x - k;
  c.setRGB(...(stops[k].map((a, j) => a + f * (stops[k + 1][j] - a)) as Vec3), SRGBColorSpace);
}

/** Polarity of each fluxon: the sign of its photospheric end. FLUX stores every fluxon from its positive end (fc0) to its
 *  negative end (fc1), so the vertex order is the field direction. Only concentrations listed in the file are
 *  photospheric (the special open and plasmoid ones are not). An open fluxon has one listed end; a closed one has two,
 *  so it gets 0. */
function polarities(w: FluxWorld): number[] {
  const sign = new Map(w.concentrations.id.map((id, i) => [id, Math.sign(w.concentrations.flux[i])] as const));
  return w.fluxons.id.map((_, f) => {
    const a = sign.get(w.fluxons.fc0[f]);
    const b = sign.get(w.fluxons.fc1[f]);
    return a != null && b == null ? a : b != null && a == null ? b : 0;
  });
}

function place(link: Link, dir: Vec3, sun: boolean) {
  const vs = [...link.views];
  if (!vs.length) return;
  const R = Math.max(...vs.map((v) => (sun ? 1.8 * v.R0 : v.R)));
  link.busy = true;
  for (const v of vs) {
    v.controls.target.set(0, 0, 0);
    v.camera.position.set(...dir).normalize().multiplyScalar(R * 3.9); // fov 30: R / sin(15 deg) fits the sphere of radius R
    v.camera.zoom = 1;
    v.camera.updateProjectionMatrix();
    v.controls.update();
    v.draw();
  }
  link.busy = false;
}

function tokenColor(name: string): Color {
  const v = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim().split(/\s+/).map(Number);
  return new Color().setRGB(v[0] / 255, v[1] / 255, v[2] / 255, SRGBColorSpace);
}

/** Smallest and largest distance from the origin among the drawn fluxon vertices. */
function radiusRange(w: FluxWorld): [number, number] {
  const a = w.fluxons.xyz;
  let lo = Infinity;
  let hi = 0;
  for (let i = 0; i < a.length; i += 3) {
    const r = Math.hypot(a[i], a[i + 1], a[i + 2]);
    if (r < lo) lo = r;
    if (r > hi) hi = r;
  }
  return lo <= hi ? [lo, hi] : [0, 1];
}

function View({ w, o, link, range }: { w: FluxWorld; o: Opts; link: Link; range: [number, number] }) {
  const { theme } = useTheme();
  const host = useRef<HTMLDivElement>(null);
  const S = useRef<{ scene: Scene; cam: Cam; mat: LineMaterial; fl?: LineSegments2; conc?: Points; sph?: Group } | null>(null);

  // renderer, camera and controls: once
  useEffect(() => {
    const el = host.current!;
    const renderer = new WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new Scene();
    const camera = new PerspectiveCamera(30, 1, 0.01, 2000);
    camera.up.set(0, 0, 1);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.screenSpacePanning = true;
    controls.minDistance = 0.05;
    controls.maxDistance = 500;
    let raf = 0;
    const draw = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => renderer.render(scene, camera));
    };
    const cam: Cam = { camera, controls, R: 1, R0: 1, placed: false, draw };
    const mat = new LineMaterial({ vertexColors: true, linewidth: 1.5, worldUnits: false });
    S.current = { scene, cam, mat };
    link.views.add(cam);
    controls.addEventListener("change", () => {
      draw();
      if (link.busy) return;
      link.busy = true;
      for (const v of link.views)
        if (v !== cam) {
          v.camera.position.copy(camera.position);
          v.camera.zoom = camera.zoom;
          v.camera.updateProjectionMatrix();
          v.controls.target.copy(controls.target);
          v.camera.lookAt(v.controls.target);
          v.draw();
        }
      link.busy = false;
    });
    const ro = new ResizeObserver(() => {
      const { clientWidth: cw, clientHeight: ch } = el;
      if (!cw || !ch) return;
      renderer.setSize(cw, ch);
      camera.aspect = cw / ch;
      camera.updateProjectionMatrix();
      mat.resolution.set(cw, ch);
      draw();
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      cancelAnimationFrame(raf);
      link.views.delete(cam);
      controls.dispose();
      mat.dispose();
      scene.traverse((x) => {
        const m = x as Mesh;
        m.geometry?.dispose();
        (m.material as { dispose?: () => void } | undefined)?.dispose?.();
      });
      renderer.dispose();
      renderer.domElement.remove();
      S.current = null;
    };
  }, [link]);

  // fluxons: rebuilt when the file, the colouring or the theme changes
  useEffect(() => {
    const s = S.current!;
    s.fl?.removeFromParent();
    s.fl?.geometry.dispose();
    const { offsets, xyz } = w.fluxons;
    const id = w.fluxons.id;
    const pol = o.colorBy === "polarity" ? polarities(w) : [];
    const ctr = w.boundary?.type === "SPHERE" ? w.boundary.args : [0, 0, 0];
    const nSeg = xyz.length / 3 - id.length;
    const pos = new Float32Array(nSeg * 6);
    const col = new Float32Array(nSeg * 6);
    const c = new Color();
    const [r0, r1] = range;
    const dark = theme === "dark";
    const vcol = (i: number, f: number, out: number) => {
      if (o.colorBy === "id") c.setHSL((((id[f] * 0.61803) % 1) + 1) % 1, 0.7, dark ? 0.62 : 0.42, SRGBColorSpace);
      else if (o.colorBy === "polarity") c.setRGB(...(pol[f] > 0 ? POS : pol[f] < 0 ? NEG : SHUT), SRGBColorSpace);
      else if (o.colorBy === "theta") {
        // angle between the field direction (vertex order, positive to negative end) and the radial unit vector:
        // 0 outward, 90 tangential, 180 inward
        const lo = Math.max(offsets[f], i - 1);
        const hi = Math.min(offsets[f + 1] - 1, i + 1);
        const rx = xyz[3 * i] - ctr[0];
        const ry = xyz[3 * i + 1] - ctr[1];
        const rz = xyz[3 * i + 2] - ctr[2];
        const dx = xyz[3 * hi] - xyz[3 * lo];
        const dy = xyz[3 * hi + 1] - xyz[3 * lo + 1];
        const dz = xyz[3 * hi + 2] - xyz[3 * lo + 2];
        const cos = (dx * rx + dy * ry + dz * rz) / (Math.hypot(dx, dy, dz) * Math.hypot(rx, ry, rz) || 1);
        ramp(DIVERGING, 1 - Math.acos(Math.min(1, Math.max(-1, cos))) / Math.PI, c);
      } else ramp(VIRIDIS, (Math.hypot(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]) - r0) / (r1 - r0 || 1), c);
      col[out] = c.r;
      col[out + 1] = c.g;
      col[out + 2] = c.b;
    };
    let k = 0;
    for (let f = 0; f < id.length; f++)
      for (let j = offsets[f]; j < offsets[f + 1] - 1; j++, k += 6) {
        pos.set(xyz.slice(3 * j, 3 * j + 6), k);
        vcol(j, f, k);
        vcol(j + 1, f, k + 3);
      }
    const g = new LineSegmentsGeometry();
    if (nSeg > 0) {
      g.setPositions(pos);
      g.setColors(col);
    }
    const m = new LineSegments2(g, s.mat);
    m.visible = o.fluxons;
    s.scene.add(m);
    s.fl = m;
    s.cam.R = Math.max(range[1], 1);
    if (!s.cam.placed) {
      s.cam.placed = true;
      const other = [...link.views].find((v) => v !== s.cam && v.placed);
      if (other) {
        s.cam.camera.position.copy(other.camera.position);
        s.cam.controls.target.copy(other.controls.target);
        s.cam.camera.zoom = other.camera.zoom;
        s.cam.camera.updateProjectionMatrix();
        s.cam.controls.update();
      } else place(link, HOME, o.fit === "sun");
    }
    s.cam.draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w, o.colorBy, theme, range]);

  // concentrations
  useEffect(() => {
    const s = S.current!;
    s.conc?.removeFromParent();
    s.conc?.geometry.dispose();
    const { xyz, flux } = w.concentrations;
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(xyz, 3));
    const pos = new Color().setRGB(...POS, SRGBColorSpace);
    const neg = new Color().setRGB(...NEG, SRGBColorSpace);
    g.setAttribute("color", new Float32BufferAttribute(flux.flatMap((f) => (f > 0 ? pos : neg).toArray()), 3));
    const p = new Points(g, new PointsMaterial({ size: 4, sizeAttenuation: false, vertexColors: true }));
    p.visible = o.conc;
    s.scene.add(p);
    s.conc = p;
    s.cam.draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w]);

  // boundary sphere
  useEffect(() => {
    const s = S.current!;
    s.sph?.removeFromParent();
    const b = w.boundary;
    s.cam.R0 = b?.type === "SPHERE" && b.args[3] > 0 ? b.args[3] : 1;
    const grp = new Group();
    const planes = w.planes ?? [];
    if (b?.type !== "SPHERE" && planes.length) {
      // planar worlds: draw each boundary plane as a translucent square, sized to the data, with a grid
      const xyz = w.fluxons.xyz;
      let ext = 0;
      for (let i = 0; i < xyz.length; i += 3) ext = Math.max(ext, Math.abs(xyz[i]), Math.abs(xyz[i + 1]));
      ext = (ext || 1) * 1.15;
      s.cam.R0 = ext;
      for (const p of planes) {
        const color = p.role === "b2" ? tokenColor("accent") : tokenColor("muted");
        const g = new Group();
        g.add(new Mesh(new PlaneGeometry(2 * ext, 2 * ext), new MeshBasicMaterial({ color, transparent: true, opacity: p.role === "b2" ? 0.08 : 0.14, depthWrite: false, side: 2 })));
        g.add(new LineSegments(new WireframeGeometry(new PlaneGeometry(2 * ext, 2 * ext, 10, 10)), new LineBasicMaterial({ color, transparent: true, opacity: 0.25 })));
        g.quaternion.copy(new Quaternion().setFromUnitVectors(new Vector3(0, 0, 1), new Vector3(...(p.normal as Vec3)).normalize()));
        g.position.set(p.origin[0], p.origin[1], p.origin[2]);
        grp.add(g);
      }
      grp.visible = o.sphere;
      s.scene.add(grp);
      s.sph = grp;
    } else if (b?.type === "SPHERE") {
      const color = tokenColor("muted");
      grp.add(new Mesh(new SphereGeometry(s.cam.R0, 48, 24), new MeshBasicMaterial({ color, transparent: true, opacity: 0.12, depthWrite: false })));
      grp.add(new LineSegments(new WireframeGeometry(new SphereGeometry(s.cam.R0, 24, 12)), new LineBasicMaterial({ color, transparent: true, opacity: 0.3 })));
      grp.position.set(b.args[0], b.args[1], b.args[2]);
      grp.visible = o.sphere;
      s.scene.add(grp);
      s.sph = grp;
    }
    s.cam.draw();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [w, theme]);

  useEffect(() => {
    const s = S.current!;
    if (s.fl) s.fl.visible = o.fluxons;
    if (s.conc) s.conc.visible = o.conc;
    if (s.sph) s.sph.visible = o.sphere;
    s.mat.linewidth = o.width;
    s.cam.draw();
  }, [o.fluxons, o.conc, o.sphere, o.width]);

  return <div ref={host} className="h-full w-full cursor-grab active:cursor-grabbing" />;
}

function Pane({ path, base, mtime, o, link }: { path: string; base: string; mtime?: number; o: Opts; link: Link }) {
  const { data: w, error, isLoading, refetch } = useFluxWorld(path, base, mtime);
  const range = useMemo(() => (w ? radiusRange(w) : ([0, 1] as [number, number])), [w]);
  return (
    <div className="min-w-0 space-y-1.5">
      <div className="flex min-h-[2.25rem] flex-wrap content-start items-baseline gap-x-3 text-[11.5px] text-muted">
        <span className="break-all font-mono text-ink">{path.split("/").pop()}</span>
        {w && (
          <span className="num">
            {w.counts.fluxons} fluxons · {w.counts.vertices} vertices · {w.counts.concentrations} concentrations
          </span>
        )}
        {w?.decimated && (
          <span className="text-run">
            showing every {w.decimated.step}th fluxon ({w.decimated.fluxons_kept} of {w.decimated.fluxons_total})
          </span>
        )}
      </div>
      <div className="relative h-[52vh] min-h-[300px] overflow-hidden rounded-xl border border-line bg-bg/60">
        {error ? <ErrorNote error={error} retry={() => refetch()} /> : isLoading || !w ? <Skeleton className="h-full" /> : <View w={w} o={o} link={link} range={range} />}
        {w && o.colorBy === "radius" && (
          <div className="num pointer-events-none absolute bottom-2 left-2 flex items-center gap-1.5 text-[10.5px] text-muted">
            r {range[0].toFixed(1)}
            <span className="h-1.5 w-24 rounded-full" style={{ background: css(VIRIDIS) }} />
            {range[1].toFixed(1)}
          </div>
        )}
        {w && o.colorBy === "theta" && (
          <div className="num pointer-events-none absolute bottom-2 left-2 flex items-center gap-1.5 text-[10.5px] text-muted" title="Angle between the field direction and the radial direction">
            180° in
            <span className="h-1.5 w-24 rounded-full" style={{ background: css([...DIVERGING].reverse()) }} />
            0° out <span className="text-faint">(90° sideways)</span>
          </div>
        )}
        {w && o.colorBy === "polarity" && (
          <div className="pointer-events-none absolute bottom-2 left-2 flex items-center gap-3 text-[10.5px] text-muted">
            {([[POS, "+ footpoint, field out"], [NEG, "- footpoint, field in"], [SHUT, "closed loop"]] as [Vec3, string][]).map(([v, text]) => (
              <span key={text} className="flex items-center gap-1">
                <span className="h-1.5 w-4 rounded-full" style={{ background: `rgb(${v.map((a) => (a * 255) | 0).join(" ")})` }} />
                {text}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Seg<T extends string>({ value, onChange, items, label }: { value: T; onChange: (v: T) => void; items: [T, string][]; label: string }) {
  return (
    <div className="inline-flex rounded-lg border border-line bg-bg/50 p-0.5" role="radiogroup" aria-label={label}>
      {items.map(([v, text]) => (
        <button
          key={v}
          role="radio"
          aria-checked={value === v}
          onClick={() => onChange(v)}
          className={cn("rounded-md px-2 py-0.5 text-xs font-medium transition-colors", value === v ? "bg-raised text-ink shadow-sm" : "text-muted hover:text-ink")}
        >
          {text}
        </button>
      ))}
    </div>
  );
}

export function FluxViewer({ path, base = "/api/cache", mtime, others = [] }: { path: string; base?: string; mtime?: number; others?: { path: string; name: string }[] }) {
  const [o, setO] = useState<Opts>({ fluxons: true, conc: true, sphere: true, colorBy: "id", width: 1.5, fit: "all" });
  const [cmp, setCmp] = useState("");
  const [link] = useState<Link>(() => ({ views: new Set(), busy: false }));
  const set = <K extends keyof Opts>(k: K, v: Opts[K]) => setO((p) => ({ ...p, [k]: v }));
  const toggle = (k: "fluxons" | "conc" | "sphere", text: string) => (
    <button key={k} aria-pressed={o[k]} className={cn("btn btn-sm", o[k] && "border-accent/60 bg-accent/15")} onClick={() => set(k, !o[k])}>
      {text}
    </button>
  );
  const sun = o.fit === "sun";
  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <div className="flex items-center gap-1">
          {PRESETS.map(([name, dir]) => (
            <button key={name} className="btn btn-sm num px-2" title={`Look along ${name}, z up`} onClick={() => place(link, dir, sun)}>
              {name}
            </button>
          ))}
          <button className="btn btn-sm btn-icon" aria-label="Reset view" title="Reset view" onClick={() => place(link, HOME, sun)}>
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>
        <Seg label="Fit" value={o.fit} onChange={(v) => set("fit", v)} items={[["all", "Fit all"], ["sun", "Fit sun"]]} />
        <div className="flex items-center gap-1">{[toggle("fluxons", "Fluxons"), toggle("conc", "Points"), toggle("sphere", "Boundary")]}</div>
        <Seg label="Colour by" value={o.colorBy} onChange={(v) => set("colorBy", v)} items={[["id", "Fluxon id"], ["radius", "Radius"], ["theta", "Deviation"], ["polarity", "Polarity"]]} />
        <label className="flex items-center gap-1.5 text-xs text-muted">
          Width
          <input type="range" min={1} max={6} step={0.5} value={o.width} onChange={(e) => set("width", +e.target.value)} className="w-20 accent-accent" />
        </label>
        {others.length > 0 && (
          <select className="input w-auto max-w-[16rem] text-xs" value={cmp} onChange={(e) => setCmp(e.target.value)} aria-label="Compare with another file">
            <option value="">Compare with...</option>
            {others.map((x) => (
              <option key={x.path} value={x.path}>
                {x.path}
              </option>
            ))}
          </select>
        )}
      </div>
      <div className={cn("grid gap-2", cmp && "grid-cols-2")}>
        <Pane path={path} base={base} mtime={mtime} o={o} link={link} />
        {cmp && <Pane key={cmp} path={cmp} base={base} o={o} link={link} />}
      </div>
      <p className="text-[11px] text-faint">Drag to rotate, wheel to zoom, right-drag or shift-drag to pan. Points: red positive flux, blue negative. Deviation: angle between the field and the radial direction, field running from the positive to the negative end. Axes: z is up in the side views.</p>
    </div>
  );
}
