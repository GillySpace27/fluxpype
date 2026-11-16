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
