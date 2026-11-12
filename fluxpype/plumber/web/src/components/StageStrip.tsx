import type { ReactNode } from "react";
import { cn } from "../lib/cn";
import type { StageStatus } from "../lib/types";
import { Tip } from "./ui";

export type StripTone = "ok" | "run" | "fail" | "skip" | "idle";

export interface StripItem {
  id: string;
  label: string;
  tone: StripTone;
  /** 0 to 1 of the bar to fill (default all) */
  fill?: number;
  /** drawn purple: the stage reused earlier output instead of computing it */
  hatched?: boolean;
  tip?: ReactNode;
}

const BAR: Record<StripTone, string> = { ok: "bg-ok", run: "bg-run", fail: "bg-fail", skip: "bg-skip/60", idle: "bg-line" };
const TEXT: Record<StripTone, string> = { ok: "text-ok", run: "text-run", fail: "text-fail", skip: "text-skip", idle: "text-faint" };

export const toneOf = (s: StageStatus): StripTone => (s === "ok" ? "ok" : s === "running" ? "run" : s === "failed" ? "fail" : s === "skipped" ? "skip" : "idle");

/** The thin segmented strip used on cards and list rows: one bar per stage, labelled or bare. The interactive chip version is PipelineChips. */
export function StageStrip({ items, labels = true, className }: { items: StripItem[]; labels?: boolean; className?: string }) {
  return (
    <div className={cn("grid gap-1.5", className)} style={{ gridTemplateColumns: `repeat(${items.length}, minmax(0, 1fr))` }}>
      {items.map((it) => (
        <Tip key={it.id} label={it.tip ?? it.label}>
          <div className="min-w-0">
            <div className="h-1 overflow-hidden rounded-full bg-line/60">
              <div className={cn("h-full rounded-full", BAR[it.tone], it.tone === "run" && "animate-pulse", it.hatched && "!bg-info")} style={{ width: `${(it.fill ?? 1) * 100}%` }} />
            </div>
            {labels && <p className={cn("mt-1 truncate text-[10px] font-medium", TEXT[it.tone])}>{it.label}</p>}
          </div>
        </Tip>
      ))}
    </div>
  );
}
