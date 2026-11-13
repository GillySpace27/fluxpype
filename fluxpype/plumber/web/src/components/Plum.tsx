import { cn } from "../lib/cn";

/** Plum, the Ask agent: a small chibi plumber with plum-coloured twin tails and a water-drop cap. Pure SVG, no images. */
export function Plum({ className, title = "Plum" }: { className?: string; title?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={cn("shrink-0", className)} role="img" aria-label={title} xmlns="http://www.w3.org/2000/svg">
      {/* twin tails behind the head */}
      <ellipse cx="9" cy="36" rx="6.5" ry="13" fill="#7a3790" transform="rotate(12 9 36)" />
      <ellipse cx="55" cy="36" rx="6.5" ry="13" fill="#7a3790" transform="rotate(-12 55 36)" />
      {/* dungarees */}
      <path d="M17 64 C17 52 47 52 47 64 Z" fill="#3a6fd8" />
      <rect x="25" y="52" width="4" height="8" rx="1.5" fill="#2c57b0" />
      <rect x="35" y="52" width="4" height="8" rx="1.5" fill="#2c57b0" />
      {/* head */}
      <circle cx="32" cy="32" r="18" fill="#ffe3d0" />
      {/* bangs */}
      <path d="M14 30 C14 14 50 14 50 30 C46 24 40 23 35 27 C32 24 28 24 25 27 C20 23 16 25 14 30 Z" fill="#8e44a8" />
      {/* cap */}
      <path d="M15 22 C15 4 49 4 49 22 Z" fill="#f08c1e" />
      <ellipse cx="32" cy="22" rx="19" ry="3.6" fill="#d77300" />
      <path d="M32 8.5 C34.6 12.4 35.6 14.2 32 16.8 C28.4 14.2 29.4 12.4 32 8.5 Z" fill="#6cc4ff" stroke="#fff" strokeWidth="0.8" />
      {/* eyes */}
      <ellipse cx="25.5" cy="35" rx="3.3" ry="4.4" fill="#2a1a3a" />
      <ellipse cx="38.5" cy="35" rx="3.3" ry="4.4" fill="#2a1a3a" />
      <circle cx="26.6" cy="33.4" r="1.3" fill="#fff" />
      <circle cx="39.6" cy="33.4" r="1.3" fill="#fff" />
      <circle cx="24.6" cy="36.6" r="0.6" fill="#fff" />
      <circle cx="37.6" cy="36.6" r="0.6" fill="#fff" />
      {/* cheeks and mouth */}
      <ellipse cx="20.5" cy="41.5" rx="3" ry="1.8" fill="#ff9fb0" opacity="0.65" />
      <ellipse cx="43.5" cy="41.5" rx="3" ry="1.8" fill="#ff9fb0" opacity="0.65" />
      <path d="M29.2 43.6 Q32 46.4 34.8 43.6" fill="none" stroke="#b5524a" strokeWidth="1.2" strokeLinecap="round" />
      {/* hair ties */}
      <circle cx="14.5" cy="27" r="2" fill="#f08c1e" />
      <circle cx="49.5" cy="27" r="2" fill="#f08c1e" />
    </svg>
  );
}
