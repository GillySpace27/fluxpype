import { cn } from "../lib/cn";

/** Plum, the Ask agent: a small chibi plumber with plum-coloured twin tails, a water-drop cap, dungarees and a wrench. Pure SVG, no images. */
export function Plum({ className, title = "Plum" }: { className?: string; title?: string }) {
  return (
    <svg viewBox="0 0 64 72" className={cn("shrink-0", className)} role="img" aria-label={title} xmlns="http://www.w3.org/2000/svg">
      {/* legs and boots */}
      <rect x="25.5" y="60" width="5.5" height="8" rx="2" fill="#3a6fd8" />
      <rect x="33" y="60" width="5.5" height="8" rx="2" fill="#3a6fd8" />
      <ellipse cx="27.6" cy="69.2" rx="4.6" ry="2.6" fill="#6b4423" />
      <ellipse cx="36" cy="69.2" rx="4.6" ry="2.6" fill="#6b4423" />
      {/* torso: shirt, then dungaree bib and straps */}
      <rect x="22" y="44" width="20" height="18" rx="6" fill="#2fb5a3" />
      <rect x="24" y="49" width="16" height="13" rx="3.5" fill="#3a6fd8" />
      <rect x="25" y="44" width="3" height="6" rx="1.2" fill="#3a6fd8" />
      <rect x="36" y="44" width="3" height="6" rx="1.2" fill="#3a6fd8" />
      <rect x="29" y="53" width="6" height="4.5" rx="1" fill="#2c57b0" />
      <circle cx="26.5" cy="50.5" r="0.9" fill="#f08c1e" />
      <circle cx="37.5" cy="50.5" r="0.9" fill="#f08c1e" />
      {/* arms: left hangs, right is raised with the wrench */}
      <line x1="22.5" y1="47.5" x2="17.5" y2="57" stroke="#2fb5a3" strokeWidth="5" strokeLinecap="round" />
      <circle cx="17" cy="58.6" r="2.6" fill="#ffe3d0" />
      <line x1="41.5" y1="47.5" x2="47.5" y2="52" stroke="#2fb5a3" strokeWidth="5" strokeLinecap="round" />
      {/* neck */}
      <rect x="28.8" y="40" width="6.4" height="5" rx="2" fill="#ffe3d0" />
      {/* head, drawn smaller and higher */}
      <g transform="translate(3.8 -1) scale(0.88)">
        <ellipse cx="9" cy="36" rx="6.5" ry="13" fill="#7a3790" transform="rotate(12 9 36)" />
        <ellipse cx="55" cy="36" rx="6.5" ry="13" fill="#7a3790" transform="rotate(-12 55 36)" />
        <circle cx="32" cy="32" r="18" fill="#ffe3d0" />
        <path d="M14 30 C14 14 50 14 50 30 C46 24 40 23 35 27 C32 24 28 24 25 27 C20 23 16 25 14 30 Z" fill="#8e44a8" />
        <path d="M15 22 C15 4 49 4 49 22 Z" fill="#f08c1e" />
        <ellipse cx="32" cy="22" rx="19" ry="3.6" fill="#d77300" />
        <path d="M32 8.5 C34.6 12.4 35.6 14.2 32 16.8 C28.4 14.2 29.4 12.4 32 8.5 Z" fill="#6cc4ff" stroke="#fff" strokeWidth="0.8" />
        <ellipse cx="25.5" cy="35" rx="3.3" ry="4.4" fill="#2a1a3a" />
        <ellipse cx="38.5" cy="35" rx="3.3" ry="4.4" fill="#2a1a3a" />
        <circle cx="26.6" cy="33.4" r="1.3" fill="#fff" />
        <circle cx="39.6" cy="33.4" r="1.3" fill="#fff" />
        <circle cx="24.6" cy="36.6" r="0.6" fill="#fff" />
        <circle cx="37.6" cy="36.6" r="0.6" fill="#fff" />
        <ellipse cx="20.5" cy="41.5" rx="3" ry="1.8" fill="#ff9fb0" opacity="0.65" />
        <ellipse cx="43.5" cy="41.5" rx="3" ry="1.8" fill="#ff9fb0" opacity="0.65" />
        <path d="M29.2 43.6 Q32 46.4 34.8 43.6" fill="none" stroke="#b5524a" strokeWidth="1.2" strokeLinecap="round" />
        <circle cx="14.5" cy="27" r="2" fill="#f08c1e" />
        <circle cx="49.5" cy="27" r="2" fill="#f08c1e" />
      </g>
      {/* wrench and raised hand, in front of the hair */}
      <line x1="49" y1="52" x2="53.4" y2="39.8" stroke="#aeb8cb" strokeWidth="2.4" strokeLinecap="round" />
      <path d="M57.2 35.5 A3.2 3.2 0 1 1 53.4 34" fill="none" stroke="#aeb8cb" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="49" cy="52.4" r="2.6" fill="#ffe3d0" />
    </svg>
  );
}
