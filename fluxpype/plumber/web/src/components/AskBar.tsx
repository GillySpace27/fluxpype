import * as RDialog from "@radix-ui/react-dialog";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Bot, CornerDownLeft, Database, Gauge, Layers, ListChecks, Loader2, Moon, Pause, Play, Plus, Search, Settings, Sparkles, Sun } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ComponentType } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { api } from "../lib/api";
import { cn } from "../lib/cn";
import { fuzzyMatch } from "../lib/fuzzy";
import { useCache, useJobs, useProfiles, useQueue } from "../lib/hooks";
import { useTheme } from "../lib/theme";
import { AskAnswer } from "./AskAnswer";

const MAC = typeof navigator !== "undefined" && /mac/i.test(navigator.platform);
const KEY = MAC ? "⌘K" : "Ctrl K";

interface Item {
  id: string;
  group: string;
  title: string;
  hint?: string;
  icon: ComponentType<{ className?: string }>;
  /** extra words the search also looks in */
  words?: string;
  run: () => void;
}

/** Open a dialog that lives elsewhere (the system monitor, the agent panel). */
export const openThing = (what: "sysmon" | "agent") => window.dispatchEvent(new CustomEvent("flux:open", { detail: what }));

interface Turn {
  q: string;
  text: string;
  tools: string[];
  state: "running" | "done" | "error";
  error?: string;
  cost?: number;
}

/** What the box can do without a model: go places, flip switches, jump to a job or preset. */
function useItems(close: () => void): Item[] {
  const nav = useNavigate();
  const qc = useQueryClient();
  const { data: jobs } = useJobs();
  const { data: profiles } = useProfiles();
  const { data: cache } = useCache();
  const { data: queue } = useQueue();
  const { theme, setChoice } = useTheme();
  const pause = useMutation({
    mutationFn: (paused: boolean) => api.patch("/api/queue", { paused }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["queue"] }),
  });
  return useMemo(() => {
    const go = (to: string) => () => {
      close();
      nav(to);
    };
    const items: Item[] = [
      { id: "p-overview", group: "Pages", title: "Overview", icon: Gauge, words: "home dashboard running", run: go("/") },
      { id: "p-presets", group: "Pages", title: "Presets", icon: Layers, words: "profiles recipes", run: go("/presets") },
      { id: "p-jobs", group: "Pages", title: "Jobs", icon: ListChecks, words: "runs history", run: go("/jobs") },
      { id: "p-new", group: "Pages", title: "New run", icon: Plus, words: "submit queue start job", run: go("/jobs/new") },
      { id: "p-cache", group: "Pages", title: "Cache", icon: Database, words: "files data folder magnetograms worlds trash", run: go("/cache") },
      { id: "p-settings", group: "Pages", title: "Settings", icon: Settings, words: "data folder backend update", run: go("/settings") },
      {
        id: "a-queue",
        group: "Actions",
        title: queue?.paused ? "Resume the queue" : "Pause the queue",
        hint: queue?.paused ? undefined : "running tasks continue",
        icon: queue?.paused ? Play : Pause,
        words: "pause resume stop start queue",
        run: () => {
          close();
          pause.mutate(!queue?.paused);
        },
      },
      { id: "a-theme", group: "Actions", title: theme === "dark" ? "Switch to the light theme" : "Switch to the dark theme", icon: theme === "dark" ? Sun : Moon, words: "theme dark light mode", run: () => setChoice(theme === "dark" ? "light" : "dark") },
      { id: "a-sys", group: "Actions", title: "Open the system monitor", icon: Gauge, words: "cpu memory gpu network disk storage processes", run: () => (close(), openThing("sysmon")) },
      { id: "a-agent", group: "Actions", title: "Open the agent panel", icon: Bot, words: "ai agent mcp activity log connect", run: () => (close(), openThing("agent")) },
    ];
    for (const j of (jobs?.jobs ?? []).slice(0, 40)) {
      items.push({
        id: `j-${j.id}`,
        group: "Jobs",
        title: `${j.profile} · ${j.id}`,
        hint: `${j.status}, ${j.tasks.length} task${j.tasks.length === 1 ? "" : "s"}`,
        icon: ListChecks,
        words: `${j.status} ${j.backend} ${[...new Set(j.tasks.map((t) => `CR${t.cr}`))].join(" ")}`,
        run: go(`/jobs/${j.id}`),
      });
    }
    for (const p of profiles?.profiles ?? []) {
      items.push({ id: `pr-${p.name}`, group: "Presets", title: p.name, hint: `${p.task_count} task${p.task_count === 1 ? "" : "s"}`, icon: Layers, words: p.batch_name, run: go(`/presets/${encodeURIComponent(p.name)}`) });
    }
    const crs = [...new Set((cache?.records ?? []).map((r) => r.cr).filter(Boolean))] as string[];
    for (const cr of crs.slice(0, 60)) items.push({ id: `c-${cr}`, group: "Cache", title: `Cache files for CR ${cr}`, icon: Database, words: `cr${cr} rotation`, run: go(`/cache?cr=${cr}`) });
    return items;
  }, [jobs, profiles, cache, queue, theme, nav, close, pause, setChoice]);
}

function rank(items: Item[], q: string): Item[] {
  const t = q.trim();
  if (!t) return items.filter((i) => i.group === "Pages" || i.group === "Actions");
  return items
    .map((i) => ({ i, m: fuzzyMatch(t, [{ label: "name", text: i.title, weight: 3 }, { label: "words", text: i.words ?? "", quiet: true }, { label: "group", text: i.group, quiet: true }]) }))
    .filter((x) => x.m)
    .sort((a, b) => b.m!.score - a.m!.score)
    .slice(0, 9)
    .map((x) => x.i);
}

async function* events(res: Response): AsyncGenerator<Record<string, any>> {
  const reader = res.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let at: number;
    while ((at = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, at);
      buf = buf.slice(at + 2);
      if (chunk.startsWith("data: ")) yield JSON.parse(chunk.slice(6));
    }
  }
}

function Palette({ onClose }: { onClose: () => void }) {
  const loc = useLocation();
  const items = useItems(onClose);
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const [turns, setTurns] = useState<Turn[]>([]);
  const abort = useRef<AbortController | null>(null);
  const body = useRef<HTMLDivElement>(null);
  const hits = useMemo(() => rank(items, q), [items, q]);
  const rows = q.trim() ? [...hits, null] : hits; // null is the "Ask Claude" row
  const asking = turns.length > 0;
  const running = turns[turns.length - 1]?.state === "running";

  useEffect(() => {
    // a question, or nothing that matches, goes to Ask by default
    setSel(q.trim() && (/\?\s*$/.test(q) || !hits.length) ? hits.length : 0);
  }, [q, hits.length]);
  useEffect(() => () => abort.current?.abort(), []);
  useEffect(() => {
    body.current?.scrollTo({ top: body.current.scrollHeight });
  }, [turns]);

  const ask = async (question: string) => {
    abort.current?.abort();
    const ctrl = (abort.current = new AbortController());
    const history = turns.filter((t) => t.state === "done").map((t) => ({ q: t.q, a: t.text }));
    const patch = (p: Partial<Turn> | ((t: Turn) => Partial<Turn>)) =>
      setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? { ...t, ...(typeof p === "function" ? p(t) : p) } : t)));
    setTurns((ts) => [...ts, { q: question, text: "", tools: [], state: "running" }]);
    setQ("");
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, page: loc.pathname, history }),
        signal: ctrl.signal,
      });
      if (!res.ok) throw new Error((await res.json().catch(() => null))?.detail ?? res.statusText);
      for await (const e of events(res)) {
        if (e.type === "text") patch((t) => ({ text: t.text + e.text }));
        else if (e.type === "tool") patch((t) => ({ tools: [...t.tools, e.name] }));
        else if (e.type === "done") patch({ state: "done", cost: e.cost });
        else if (e.type === "error") patch({ state: "error", error: e.message });
      }
      patch((t) => (t.state === "running" ? { state: "done" } : {}));
    } catch (e) {
      if (!ctrl.signal.aborted) patch({ state: "error", error: e instanceof Error ? e.message : String(e) });
    }
  };

  const choose = (i: number) => {
    const row = rows[i];
    if (row === null || (row === undefined && q.trim())) ask(q.trim());
    else row?.run();
  };
  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      setSel((s) => Math.max(0, Math.min(rows.length - 1, s + (e.key === "ArrowDown" ? 1 : -1))));
    } else if (e.key === "Enter" && !e.nativeEvent.isComposing) {
      e.preventDefault();
      if (running) return;
      if (asking) q.trim() && ask(q.trim()); // a follow-up, never a command
      else if (q.trim() || rows.length) choose(sel);
    }
  };

  let lastGroup = "";
  return (
    <RDialog.Root open onOpenChange={(o) => !o && onClose()}>
      <RDialog.Portal>
        <RDialog.Overlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-[2px] data-[state=open]:animate-fade-in" />
        <RDialog.Content
          aria-describedby={undefined}
          className="fixed left-1/2 top-[10vh] z-50 flex max-h-[78vh] w-[min(720px,calc(100vw-2rem))] -translate-x-1/2 animate-rise flex-col overflow-hidden rounded-2xl border border-line bg-surface shadow-pop focus:outline-none"
        >
          <RDialog.Title className="sr-only">Ask or search</RDialog.Title>
          <div className="flex items-center gap-3 border-b hair px-4 py-3">
            {asking ? (
              <button className="btn btn-ghost btn-icon !h-6 !w-6" aria-label="Back to search" onClick={() => (abort.current?.abort(), setTurns([]))}>
                <ArrowLeft className="h-4 w-4" />
              </button>
            ) : (
              <Search className="h-4 w-4 text-faint" />
            )}
            <input
              autoFocus
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={onKey}
              placeholder={asking ? "Ask a follow-up" : "Search, run a command, or ask about FLUX and Plumber"}
              className="min-w-0 flex-1 bg-transparent text-[14px] text-ink placeholder:text-faint focus:outline-none focus-visible:ring-0"
              aria-label="Ask or search"
              spellCheck={false}
            />
            {running && <Loader2 className="h-4 w-4 animate-spin text-accent" />}
            <kbd className="rounded border border-line px-1.5 py-0.5 text-[10px] text-faint">esc</kbd>
          </div>
          <div ref={body} className="min-h-0 flex-1 overflow-y-auto">
            {asking ? (
              <div className="space-y-5 px-5 py-4">
                {turns.map((t, i) => (
                  <div key={i}>
                    <p className="text-xs font-medium text-muted">{t.q}</p>
                    {t.tools.length > 0 && <p className="mt-1 text-[11px] text-faint">Looked at: {[...new Set(t.tools)].join(", ")}</p>}
                    {t.text ? <AskAnswer text={t.text} streaming={t.state === "running"} /> : t.state === "running" && <p className="mt-2 text-xs text-faint">Thinking...</p>}
                    {t.state === "error" && <p className="mt-2 rounded-lg border border-fail/30 bg-fail/10 px-3 py-2 text-xs">{t.error}</p>}
                    {t.state === "done" && t.cost != null && <p className="mt-1 text-[10.5px] text-faint">Answered by Sonnet from the FLUX reference; check anything you will act on. About ${t.cost.toFixed(3)}.</p>}
                  </div>
                ))}
              </div>
            ) : (
              <ul role="listbox" className="p-1.5">
                {rows.map((r, i) => {
                  const head = r === null ? "Ask" : r.group !== lastGroup ? r.group : "";
                  if (r) lastGroup = r.group;
                  const Icon = r === null ? Sparkles : r.icon;
                  return (
                    <li key={r === null ? "ask" : r.id} role="option" aria-selected={i === sel}>
                      {head && <p className="px-3 pb-1 pt-2 text-[10.5px] uppercase tracking-[0.12em] text-faint">{head}</p>}
                      <button
                        className={cn("flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[13px]", i === sel ? "bg-raised" : "hover:bg-raised/60")}
                        onMouseEnter={() => setSel(i)}
                        onClick={() => choose(i)}
                      >
                        <Icon className={cn("h-4 w-4 shrink-0", r === null ? "text-accent" : "text-faint")} />
                        <span className="min-w-0 flex-1 truncate">{r === null ? `Ask Claude: ${q.trim()}` : r.title}</span>
                        {r && r.hint && <span className="shrink-0 text-[11px] text-faint">{r.hint}</span>}
                        {i === sel && <CornerDownLeft className="h-3.5 w-3.5 shrink-0 text-faint" />}
                      </button>
                    </li>
                  );
                })}
                {!rows.length && <li className="px-4 py-6 text-center text-xs text-muted">Type to search, or ask a question.</li>}
              </ul>
            )}
          </div>
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}

/** The box in the middle of the top bar. Cmd+K (Ctrl+K) opens it from anywhere. */
export function AskBar() {
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex h-9 w-full max-w-[420px] items-center gap-2.5 rounded-xl border border-line bg-raised/70 px-3 text-left text-[13px] text-faint transition-colors hover:bg-raised focus:outline-none focus-visible:ring-1 focus-visible:ring-accent/60"
        aria-label="Ask or search"
      >
        <Sparkles className="h-4 w-4 shrink-0 text-accent" />
        <span className="min-w-0 flex-1 truncate">Ask or search</span>
        <kbd className="hidden rounded border border-line px-1.5 py-0.5 text-[10px] sm:inline">{KEY}</kbd>
      </button>
      {open && <Palette onClose={() => setOpen(false)} />}
    </>
  );
}
