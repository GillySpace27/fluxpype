/** Small fuzzy matcher for lists of labelled fields (no dependency). */

export interface Field {
  label: string;
  text: string;
  weight?: number;
  /** What to show when this field is why a result matched; leave out to show `text`. */
  display?: string;
  /** Already visible on the result, so never shown as the match hint. */
  quiet?: boolean;
}

export interface Match {
  score: number;
  /** The best-matching field other than the first (the name), for a hint under the result. */
  hint?: Field;
}

const compact = (s: string) => s.toLowerCase().replace(/[^a-z0-9.]+/g, "");

/** Score of one token in one string: substring beats word start beats a tight subsequence. 0 = no match. */
function tokenScore(token: string, text: string): number {
  const low = text.toLowerCase();
  const at = low.indexOf(token);
  if (at >= 0) return at === 0 || /[^a-z0-9]/.test(low[at - 1]) ? 10 : 7;
  const c = compact(text);
  const ct = compact(token);
  if (ct && c.includes(ct)) return 6; // "cr2225" matches "CR 2225", "fluxoncount" matches "fluxon_count"
  if (ct.length < 3) return 0;
  // subsequence, but only within a short span so long fields do not match everything
  let i = 0;
  let first = -1;
  for (let j = 0; j < c.length && i < ct.length; j++) {
    if (c[j] === ct[i]) {
      if (first < 0) first = j;
      i++;
      if (i === ct.length) {
        const span = j - first + 1;
        return span <= ct.length * 2 + 1 ? 3 : 0;
      }
    }
  }
  return 0;
}

/** Every whitespace-separated token must match some field. Returns null when the query does not match. */
export function fuzzyMatch(query: string, fields: Field[]): Match | null {
  const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!tokens.length) return { score: 0 };
  let total = 0;
  let hint: Field | undefined;
  let hintScore = 0;
  for (const t of tokens) {
    let best = 0;
    let bestField: Field | undefined;
    fields.forEach((f, idx) => {
      const s = tokenScore(t, f.text) * (f.weight ?? 1);
      if (s > best) {
        best = s;
        bestField = idx === 0 ? undefined : f;
      }
    });
    if (!best) return null;
    total += best;
    if (bestField && !bestField.quiet && best > hintScore) {
      hint = bestField;
      hintScore = best;
    }
  }
  return { score: total, hint };
}
