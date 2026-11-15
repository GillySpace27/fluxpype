import { useEffect, useLayoutEffect, useRef, type RefObject } from "react";
import { createPortal } from "react-dom";
import { FEET_Y, HEAD_BOX, Plum } from "./Plum";

/**
 * The one Plum that travels. At rest in the search bar a plain head-only <Plum> sits in the slot; when Ask opens, this
 * actor takes over from exactly that spot, hops along an arc to the top edge of the Ask panel while her body unfolds
 * from behind the head, and lands with a squash. Closing the panel plays it back. Everything that moves is a CSS
 * transform or opacity (compositor friendly); there are no filters and no layout reads while she is in the air.
 */

type Pose = { x: number; y: number; s: number };

const HEAD_CX = HEAD_BOX.x + HEAD_BOX.w / 2;
const HEAD_CY = HEAD_BOX.y + HEAD_BOX.h / 2;
const HOP_MS = 620;
const SETTLE_MS = 420;
const CROUCH_MS = 100;
/** The Ask panel's top edge, as a share of the viewport height. Keep in step with AskBar. */
export const PANEL_TOP = 0.12;

const css = (p: Pose) => `translate3d(${p.x.toFixed(2)}px, ${p.y.toFixed(2)}px, 0) scale(${p.s.toFixed(4)})`;
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Where the actor must be so that her head exactly covers the search-bar slot. */
function barPose(slot: HTMLElement): Pose {
  const r = slot.getBoundingClientRect();
  if (!r.width) return { x: window.innerWidth / 2 - 13, y: 4, s: 0.4 }; // the search bar is not shown at this width
  const s = Math.min(r.width / HEAD_BOX.w, r.height / HEAD_BOX.h);
  return { x: r.left + r.width / 2 - s * HEAD_CX, y: r.top + r.height / 2 - s * HEAD_CY, s };
}

/** Standing on the top edge of the panel, near its left corner. */
function perchPose(): Pose {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const w = Math.min(720, vw - 32);
  const left = (vw - w) / 2;
  const top = vh * PANEL_TOP;
  const s = Math.max(0.7, Math.min(1.2, (top + 8) / FEET_Y));
  return { x: left + 20 - s * 6, y: top + 5 - s * FEET_Y, s };
}

function currentPose(el: HTMLElement): Pose {
  const m = new DOMMatrix(getComputedStyle(el).transform);
  return { x: m.e, y: m.f, s: m.a || 1 };
}

/** A small fan of drops and sparkles where her feet land. */
function burst(layer: HTMLElement, x: number, y: number, n: number) {
  const colors = ["#6cc4ff", "#f08c1e", "#b66fd1", "#ffffff", "#2fb5a3"];
  for (let i = 0; i < n; i++) {
    const d = document.createElement("div");
    const size = 4 + Math.random() * 5;
    const shape = i % 3 === 0 ? "50% 0 50% 50%" : i % 3 === 1 ? "50%" : "1px";
    d.style.cssText = `position:absolute;left:${x}px;top:${y}px;width:${size}px;height:${size}px;margin:${-size / 2}px 0 0 ${-size / 2}px;background:${colors[i % colors.length]};border-radius:${shape};will-change:transform,opacity`;
    layer.appendChild(d);
    const a = -Math.PI * (0.08 + 0.84 * Math.random());
    const r = 20 + Math.random() * 30;
    const dx = Math.cos(a) * r;
    const dy = Math.sin(a) * r * 0.7;
    const spin = (Math.random() - 0.5) * 220;
    const anim = d.animate(
      [
        { transform: "translate(0,0) scale(1) rotate(0deg)", opacity: 1 },
        { transform: `translate(${dx}px,${dy}px) scale(1.1) rotate(${spin / 2}deg)`, opacity: 1, offset: 0.42 },
        { transform: `translate(${dx * 1.25}px,${dy + 30}px) scale(0) rotate(${spin}deg)`, opacity: 0 },
      ],
      { duration: 520 + Math.random() * 200, easing: "cubic-bezier(0.2, 0.7, 0.3, 1)" },
    );
    const gone = () => d.remove();
    anim.addEventListener("finish", gone);
    anim.addEventListener("cancel", gone);
  }
}

export function PlumActor({ slot, open, onAway }: { slot: RefObject<HTMLElement>; open: boolean; onAway: (away: boolean) => void }) {
  const wrap = useRef<HTMLDivElement>(null);
  const layer = useRef<HTMLDivElement>(null);
  const last = useRef(false);
  const busy = useRef(false);
  const timers = useRef<number[]>([]);

  const later = (fn: () => void, ms: number) => {
    timers.current.push(window.setTimeout(fn, ms));
  };
  const stop = () => {
    timers.current.forEach((t) => window.clearTimeout(t));
    timers.current = [];
    const el = wrap.current;
    el?.getAnimations().forEach((a) => a.cancel());
    el?.firstElementChild?.getAnimations().forEach((a) => a.cancel());
  };

  useLayoutEffect(() => {
    if (last.current === open) return; // first render, and React strict-mode re-runs
    last.current = open;
    const el = wrap.current;
    const sl = slot.current;
    const sq = el?.firstElementChild as HTMLElement | null;
    const svg = el?.querySelector("svg");
    if (!el || !sl || !sq || !svg) return;

    const flying = el.style.visibility === "visible";
    const from = flying ? currentPose(el) : barPose(sl);
    const to = open ? perchPose() : barPose(sl);
    stop();
    busy.current = true;
    el.style.visibility = "visible";
    el.style.transform = css(to); // the resting value once the animation is over
    if (open) onAway(true);

    svg.classList.remove("plum-land", "plum-wave");
    if (open) svg.classList.add("plum-full");
    else later(() => svg.classList.remove("plum-full"), 170);

    const finish = () => {
      busy.current = false;
      if (open) {
        el.style.transform = css(perchPose());
      } else {
        el.style.visibility = "hidden";
        svg.classList.remove("plum-full");
        onAway(false);
      }
    };

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      finish();
      return;
    }

    // the hop: crouch, launch, swoop (the bar is near the top of the window, so there is little room above it), land,
    // rebound once, settle
    const dx = to.x - from.x;
    const apexY = Math.max(4, Math.min(from.y, to.y) - (44 + Math.min(36, Math.abs(to.y - from.y) * 0.12)));
    const total = HOP_MS + SETTLE_MS;
    const mid = lerp(from.s, to.s, 0.6);
    const at = (ms: number) => ms / total;
    const rebound = (open ? 15 : 7) * to.s;
    el.animate(
      [
        { transform: css(from), offset: 0, easing: "ease-in-out" },
        { transform: css({ x: from.x, y: from.y + 2 * from.s, s: from.s }), offset: at(CROUCH_MS), easing: "cubic-bezier(0.15, 0.75, 0.3, 1)" },
        { transform: css({ x: from.x + dx * 0.58, y: apexY, s: mid }), offset: at(HOP_MS * 0.55), easing: "cubic-bezier(0.55, 0.05, 0.85, 0.45)" },
        { transform: css(to), offset: at(HOP_MS), easing: "cubic-bezier(0.2, 0.7, 0.4, 1)" },
        { transform: css({ ...to, y: to.y - rebound }), offset: at(HOP_MS + 130), easing: "cubic-bezier(0.6, 0, 0.9, 0.5)" },
        { transform: css(to), offset: at(HOP_MS + 260) },
        { transform: css(to), offset: 1 },
      ],
      { duration: total },
    );
    // squash and stretch about the feet
    sq.animate(
      [
        { transform: "scale(1, 1)", offset: 0, easing: "ease-out" },
        { transform: "scale(1.14, 0.84)", offset: at(CROUCH_MS) },
        { transform: "scale(0.88, 1.18)", offset: at(CROUCH_MS + 90), easing: "ease-in-out" },
        { transform: "scale(0.98, 1.04)", offset: at(HOP_MS * 0.55) },
        { transform: "scale(0.94, 1.1)", offset: at(HOP_MS - 30) },
        { transform: "scale(1.22, 0.78)", offset: at(HOP_MS + 10), easing: "ease-out" },
        { transform: "scale(0.93, 1.1)", offset: at(HOP_MS + 130) },
        { transform: "scale(1.08, 0.93)", offset: at(HOP_MS + 262), easing: "ease-out" },
        { transform: "scale(0.99, 1.02)", offset: at(HOP_MS + 340) },
        { transform: "scale(1, 1)", offset: 1 },
      ],
      { duration: total },
    );

    later(() => {
      const lx = to.x + to.s * 32;
      const ly = to.y + to.s * FEET_Y;
      if (layer.current) burst(layer.current, lx, ly, open ? 11 : 5);
      svg.classList.add("plum-land");
      if (open) later(() => svg.classList.add("plum-wave"), 300);
      later(() => svg.classList.remove("plum-land"), 1100);
    }, HOP_MS);
    later(finish, total);
  }, [open, slot, onAway]);

  // keep her on the panel's corner if the window is resized while she is perched
  useEffect(() => {
    if (!open) return;
    const fix = () => {
      if (!busy.current && wrap.current) wrap.current.style.transform = css(perchPose());
    };
    window.addEventListener("resize", fix);
    return () => window.removeEventListener("resize", fix);
  }, [open]);

  useEffect(() => stop, []);

  return createPortal(
    <>
      <div ref={wrap} className="plum-actor" aria-hidden="true">
        <div className="plum-squash">
          <Plum pose="auto" className="h-full w-full" />
        </div>
      </div>
      <div ref={layer} className="pointer-events-none fixed inset-0 z-[61]" aria-hidden="true" />
    </>,
    document.body,
  );
}
