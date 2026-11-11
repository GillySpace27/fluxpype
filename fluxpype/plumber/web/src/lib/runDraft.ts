/** Which arrangement the app opens in: presets first (launch several jobs) or one run start to finish. */
export type LayoutMode = "presets" | "run";

const KEY = "plumber.layout";

export function storedLayout(): LayoutMode {
  try {
    return localStorage.getItem(KEY) === "run" ? "run" : "presets";
  } catch {
    return "presets";
  }
}

export function storeLayout(m: LayoutMode): void {
  try {
    localStorage.setItem(KEY, m);
  } catch {
    /* convenience only */
  }
}

export const layoutOf = (pathname: string): LayoutMode => (pathname === "/run" || pathname.startsWith("/run/") ? "run" : "presets");

/* the unsent new-run draft, kept per browser */
const DRAFT = "plumber.runDraft";

export function loadDraft<T>(): T | null {
  try {
    const raw = localStorage.getItem(DRAFT);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

export function saveDraft(d: unknown | null): void {
  try {
    if (d === null) localStorage.removeItem(DRAFT);
    else localStorage.setItem(DRAFT, JSON.stringify(d));
  } catch {
    /* convenience only */
  }
}
