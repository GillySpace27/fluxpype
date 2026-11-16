/** Plum's skin tone: a per-browser choice, applied as CSS variables that the `.plum-*` rules in styles.css read. */
export type PlumSkin = { id: string; label: string; skin: string; blush: string; mouth: string; eyeRim?: string };

export const PLUM_SKINS: PlumSkin[] = [
  { id: "light", label: "Light", skin: "#ffe3d0", blush: "#ff9fb0", mouth: "#b5524a" },
  { id: "light-warm", label: "Light warm", skin: "#f6cfae", blush: "#ff8f9c", mouth: "#a8483f" },
  { id: "tan", label: "Tan", skin: "#e0ac7e", blush: "#f07f7f", mouth: "#8f3b34" },
  { id: "brown", label: "Brown", skin: "#b9783f", blush: "#e06a5e", mouth: "#6e2620" },
  { id: "deep-brown", label: "Deep brown", skin: "#8a5529", blush: "#d0604c", mouth: "#4a1712", eyeRim: "#c99a6e" },
  { id: "dark", label: "Dark", skin: "#5c3a22", blush: "#b4513f", mouth: "#2e0e0b", eyeRim: "#a87b54" },
  { id: "lavender", label: "Lavender", skin: "#dccbf5", blush: "#f59ad0", mouth: "#7a3790" },
  { id: "sunlit-gold", label: "Sunlit gold", skin: "#ffd98a", blush: "#ff9a6b", mouth: "#b5524a" },
  { id: "mint", label: "Mint", skin: "#bdeede", blush: "#ff9fb0", mouth: "#2a8f80" },
  { id: "sky", label: "Sky", skin: "#c4e2ff", blush: "#ff9fb0", mouth: "#3a6fd8" },
];

export const DEFAULT_PLUM_SKIN = "sunlit-gold";
const KEY = "plumber.plumSkin";

const find = (id: string | null) => PLUM_SKINS.find((s) => s.id === id);

export function plumSkin(): string {
  try {
    const v = find(localStorage.getItem(KEY));
    if (v) return v.id;
  } catch {
    /* storage can be blocked */
  }
  return DEFAULT_PLUM_SKIN;
}

export function applyPlumSkin(id: string): void {
  const s = find(id) ?? find(DEFAULT_PLUM_SKIN)!;
  const st = document.documentElement.style;
  st.setProperty("--plum-skin", s.skin);
  st.setProperty("--plum-blush", s.blush);
  st.setProperty("--plum-mouth", s.mouth);
  st.setProperty("--plum-eye-rim", s.eyeRim ?? "transparent");
}

export function setPlumSkin(id: string): void {
  applyPlumSkin(id);
  try {
    localStorage.setItem(KEY, id);
  } catch {
    /* ignore */
  }
}
