import { useQuery } from "@tanstack/react-query";
import { api } from "../lib/api";
import { num } from "../lib/format";
import { Card } from "./ui";

interface Point {
  cycle: number;
  e_simple2?: number | null;
  d_e_simple2?: number | null;
}
interface SceneResultDoc {
  mode: string;
  status: string;
  done?: boolean;
  lid?: number | null;
  cycles?: number;
  worlds: { scene: string; cycles_done: number; final_e_simple2: number | null; final_top_z: number | null; final_stiffness: number | null; energy_settled_at_cycle: number | null; series: Point[] }[];
  D?: { definition: string; final: number | null; settled_at_cycle?: number | null; value_at_settle?: number | null; series: Point[] };
}

/** A small line chart of one or two series over cycles, y fitted to the data (the energies are only comparable within a pair). */
function Lines({ lines }: { lines: { label: string; pts: [number, number][]; cls: string }[] }) {
  const all = lines.flatMap((l) => l.pts);
  if (all.length < 2) return <p className="text-xs text-faint">Waiting for cycles...</p>;
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  if (y1 - y0 < 1e-12) (y0 -= 1), (y1 += 1);
  const W = 420, H = 120, P = 6;
  const sx = (x: number) => P + ((x - x0) / Math.max(x1 - x0, 1e-9)) * (W - 2 * P);
  const sy = (y: number) => H - P - ((y - y0) / (y1 - y0)) * (H - 2 * P);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={lines.map((l) => l.label).join(" and ")}>
        {lines.map((l) => (
          <polyline key={l.label} fill="none" strokeWidth="1.6" className={l.cls} points={l.pts.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ")} />
        ))}
      </svg>
      <p className="num flex flex-wrap gap-x-3 text-[11px] text-muted">
        {lines.map((l) => (
          <span key={l.label}>
            <span className={l.cls.replace("stroke-", "text-")}>&#9632;</span> {l.label}
          </span>
        ))}
        <span className="ml-auto">
          y {num(y0)} to {num(y1)}, cycles {x0} to {x1}
        </span>
      </p>
    </div>
  );
}

/** The energy series of a scene relaxation task: per world, the settled cycle and final numbers, and for a compare
 * the difference D = E(B) - E(A). Read from the run directory, so it fills in while the task runs. */
export function SceneResult({ taskId, running }: { taskId: string; running: boolean }) {
  const q = useQuery({ queryKey: ["scene-result", taskId], queryFn: () => api.get<SceneResultDoc>(`/api/tasks/${taskId}/scene-result`), refetchInterval: running ? 3000 : false });
  const r = q.data;
  if (!r || !r.worlds.length) return null;
  const cls = ["stroke-accent", "stroke-run"];
  return (
    <Card className="space-y-3 p-4">
      <p className="flex flex-wrap items-baseline gap-x-3 text-[13px] font-semibold">
        Relaxation energy <span className="text-xs font-normal text-muted">{r.mode}{r.lid != null ? `, lid at z = ${r.lid}` : ""}{r.done ? "" : ", running"}</span>
      </p>
      <Lines lines={r.worlds.map((w, i) => ({ label: w.scene, cls: cls[i % 2], pts: w.series.filter((p) => p.e_simple2 != null).map((p) => [p.cycle, p.e_simple2 as number]) }))} />
      <table className="num w-full text-xs">
        <thead>
          <tr className="text-left text-faint">
            <th className="font-normal">World</th>
            <th className="font-normal">Cycles</th>
            <th className="font-normal">Final energy</th>
            <th className="font-normal">Top z</th>
            <th className="font-normal">Settled at</th>
          </tr>
        </thead>
        <tbody>
          {r.worlds.map((w) => (
            <tr key={w.scene}>
              <td className="pr-2 font-mono">{w.scene}</td>
              <td>{w.cycles_done}</td>
              <td>{num(w.final_e_simple2)}</td>
              <td>{num(w.final_top_z)}</td>
              <td>{w.energy_settled_at_cycle != null ? `cycle ${w.energy_settled_at_cycle}` : "not yet"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {r.D && (
        <div>
          <p className="text-xs">
            <span className="font-medium">D = E(B) - E(A)</span>: {num(r.D.final)}
            {r.D.settled_at_cycle != null ? `, settled at cycle ${r.D.settled_at_cycle} (${num(r.D.value_at_settle)})` : ", not settled yet"}
          </p>
          <Lines lines={[{ label: "D", cls: "stroke-ok", pts: r.D.series.filter((p) => p.d_e_simple2 != null).map((p) => [p.cycle, p.d_e_simple2 as number]) }]} />
        </div>
      )}
      <p className="text-[11px] text-faint">Energies are relaxed values of the world as built; compare them only within a fair pair at the same lid height. They are not a model of a measured region.</p>
    </Card>
  );
}
