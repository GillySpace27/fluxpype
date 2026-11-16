import { cn } from "../lib/cn";

/**
 * Plum, the Ask agent: a small chibi plumber with plum-coloured twin tails, a water-drop cap, dungarees and a wrench.
 * Pure SVG, no images.
 *
 * The drawing is split into groups so it can be animated part by part (see PlumActor and the `.plum-*` rules in
 * styles.css): the body parts fold up behind the head and unfold again, and the tails, eyes and wrench move on their
 * own. Everything is drawn in a 64 x 72 box. `pose` picks how much of it is shown:
 *   "full" the whole figure, "head" a crop of the head only, "auto" whatever the surrounding `plum-full` class says.
 */
export type PlumPose = "full" | "head" | "auto";

/** The head, tails and cap sit inside this part of the 64 x 72 box. */
export const HEAD_BOX = { x: 4, y: 1, w: 56, h: 44 } as const;
/** Where the feet touch the ground in the 64 x 72 box. */
export const FEET_Y = 71.8;

export function Plum({ className, title = "Plum", pose = "full" }: { className?: string; title?: string; pose?: PlumPose }) {
  const head = pose === "head";
  const viewBox = head ? `${HEAD_BOX.x} ${HEAD_BOX.y} ${HEAD_BOX.w} ${HEAD_BOX.h}` : "0 0 64 72";
  return (
    <svg
      viewBox={viewBox}
      className={cn("plum shrink-0", pose === "full" && "plum-full", head && "plum-head-only", className)}
      role="img"
      aria-label={title}
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* legs and boots */}
      <g className="plum-part plum-legs">
        <rect x="25.5" y="60" width="5.5" height="8" rx="2" fill="#3a6fd8" />
        <rect x="33" y="60" width="5.5" height="8" rx="2" fill="#3a6fd8" />
        <ellipse cx="27.6" cy="69.2" rx="4.6" ry="2.6" fill="#6b4423" />
        <ellipse cx="36" cy="69.2" rx="4.6" ry="2.6" fill="#6b4423" />
      </g>
      {/* torso: shirt, then dungaree bib and straps, with the hanging left arm and the neck */}
      <g className="plum-part plum-torso">
        <rect x="22" y="44" width="20" height="18" rx="6" fill="#2fb5a3" />
        <rect x="24" y="49" width="16" height="13" rx="3.5" fill="#3a6fd8" />
        <rect x="25" y="44" width="3" height="6" rx="1.2" fill="#3a6fd8" />
        <rect x="36" y="44" width="3" height="6" rx="1.2" fill="#3a6fd8" />
        <rect x="29" y="53" width="6" height="4.5" rx="1" fill="#2c57b0" />
        <circle cx="26.5" cy="50.5" r="0.9" fill="#f08c1e" />
        <circle cx="37.5" cy="50.5" r="0.9" fill="#f08c1e" />
        <rect x="28.8" y="40" width="6.4" height="5" rx="2" fill="#ffe3d0" />
      </g>
      <g className="plum-part plum-arm-l">
        <line x1="22.5" y1="47.5" x2="17.5" y2="57" stroke="#2fb5a3" strokeWidth="5" strokeLinecap="round" />
        <circle cx="17" cy="58.6" r="2.6" fill="#ffe3d0" />
      </g>
      {/* head, drawn smaller and higher */}
      <g transform="translate(3.8 -1) scale(0.88)">
        <g className="plum-tail plum-tail-l">
          <ellipse cx="9" cy="36" rx="6.5" ry="13" fill="#7a3790" transform="rotate(12 9 36)" />
        </g>
        <g className="plum-tail plum-tail-r">
          <ellipse cx="55" cy="36" rx="6.5" ry="13" fill="#7a3790" transform="rotate(-12 55 36)" />
        </g>
        <circle cx="32" cy="32" r="18" fill="#ffe3d0" />
        <path d="M14 30 C14 14 50 14 50 30 C46 24 40 23 35 27 C32 24 28 24 25 27 C20 23 16 25 14 30 Z" fill="#8e44a8" />
        <path d="M15 22 C15 4 49 4 49 22 Z" fill="#f08c1e" />
        <ellipse cx="32" cy="22" rx="19" ry="3.6" fill="#d77300" />
        <path d="M32 8.5 C34.6 12.4 35.6 14.2 32 16.8 C28.4 14.2 29.4 12.4 32 8.5 Z" fill="#6cc4ff" stroke="#fff" strokeWidth="0.8" />
        <g className="plum-eyes">
          <ellipse cx="25.5" cy="35" rx="3.3" ry="4.4" fill="#2a1a3a" />
          <ellipse cx="38.5" cy="35" rx="3.3" ry="4.4" fill="#2a1a3a" />
          <circle cx="26.6" cy="33.4" r="1.3" fill="#fff" />
          <circle cx="39.6" cy="33.4" r="1.3" fill="#fff" />
          <circle cx="24.6" cy="36.6" r="0.6" fill="#fff" />
          <circle cx="37.6" cy="36.6" r="0.6" fill="#fff" />
        </g>
        <ellipse cx="20.5" cy="41.5" rx="3" ry="1.8" fill="#ff9fb0" opacity="0.65" />
        <ellipse cx="43.5" cy="41.5" rx="3" ry="1.8" fill="#ff9fb0" opacity="0.65" />
        <path d="M29.2 43.6 Q32 46.4 34.8 43.6" fill="none" stroke="#b5524a" strokeWidth="1.2" strokeLinecap="round" />
        <circle cx="14.5" cy="27" r="2" fill="#f08c1e" />
        <circle cx="49.5" cy="27" r="2" fill="#f08c1e" />
      </g>
      {/* raised arm, wrench and hand, in front of the hair */}
      <g className="plum-part plum-arm-r">
        <line x1="41.5" y1="47.5" x2="47.5" y2="52" stroke="#2fb5a3" strokeWidth="5" strokeLinecap="round" />
        <line x1="49" y1="52" x2="53.4" y2="39.8" stroke="#aeb8cb" strokeWidth="2.4" strokeLinecap="round" />
        <path d="M57.2 35.5 A3.2 3.2 0 1 1 53.4 34" fill="none" stroke="#aeb8cb" strokeWidth="2.2" strokeLinecap="round" />
        <circle cx="49" cy="52.4" r="2.6" fill="#ffe3d0" />
      </g>
    </svg>
  );
}
