import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpCircle, Loader2, RefreshCw } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { ago } from "../lib/format";
import type { StatusInfo, UpdateStatus } from "../lib/types";
import { Badge, Banner, Card, Skeleton, useConfirm, useToast } from "./ui";

/** Asks the remote once an hour (the server caches the fetch); "Check now" asks again at once. */
export const useUpdates = () =>
  useQuery({
    queryKey: ["update"],
    queryFn: () => api.get<UpdateStatus>("/api/update?fetch=1"),
    staleTime: 3_600_000,
    refetchInterval: 3_600_000,
    retry: false,
  });

/** Wait for the restarted server to answer with the new commit, then reload the page. */
async function waitForRestart(target: string | null, onTick: (s: string) => void) {
  const end = Date.now() + 90_000;
  await new Promise((r) => setTimeout(r, 1500));
  while (Date.now() < end) {
    try {
      const st = await api.get<StatusInfo>("/api/status");
      if (!target || st.commit === target) {
        window.location.reload();
        return;
      }
    } catch {
      onTick("Waiting for Plumber to come back...");
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  onTick("Plumber has not come back after 90 s. Check <home>/server.log, or start it again.");
}

export function UpdatePill() {
  const { data } = useUpdates();
  if (!data?.supported || !data.behind) return null;
  return (
    <Link
      to="/settings#updates"
      className="hidden h-8 items-center gap-1.5 rounded-full border border-accent/50 bg-accent/10 px-3 text-xs text-ink hover:bg-accent/20 md:flex"
      title={`${data.behind} new commit(s) on ${data.upstream}`}
    >
      <ArrowUpCircle className="h-3.5 w-3.5 text-accent" /> Update available
    </Link>
  );
}

export function UpdatesCard() {
  const { data, isLoading, error } = useUpdates();
  const qc = useQueryClient();
  const toast = useToast();
  const { ask, node } = useConfirm();
  const [checking, setChecking] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);

  const check = async () => {
    setChecking(true);
    try {
      qc.setQueryData(["update"], await api.get<UpdateStatus>("/api/update?fetch=1"));
    } catch (e) {
      toast("error", e instanceof Error ? e.message : String(e));
    } finally {
      setChecking(false);
    }
  };
  const update = async () => {
    if (!data) return;
    const ok = await ask({
      title: `Update to ${data.upstream} and restart?`,
      body: `${data.behind} new commit(s) will be pulled (fast-forward only), then Plumber restarts. Running FLUX jobs keep running; the page reloads when Plumber is back.${
        data.deps_changed?.length ? ` Dependencies changed (${data.deps_changed.join(", ")}): reinstall afterwards if Plumber reports missing packages.` : ""
      }`,
      confirm: "Update and restart",
    });
    if (!ok) return;
    setBusy("Updating...");
    try {
      const r = await api.post<{ to: string; restarting: boolean }>("/api/update", { restart: true });
      setBusy("Restarting Plumber...");
      await waitForRestart(r.to, setBusy);
    } catch (e) {
      setBusy(null);
      toast("error", e instanceof Error ? e.message : String(e));
      qc.invalidateQueries({ queryKey: ["update"] });
    }
  };

  if (isLoading) return <Skeleton className="h-28" />;
  if (error) return <Banner tone="warn" title="Could not check for updates">{error instanceof Error ? error.message : String(error)}</Banner>;
  if (!data) return null;
  if (!data.supported) return <Card className="px-4 py-3.5 text-[13px] text-muted">{data.reason}</Card>;
  return (
    <Card className="divide-y divide-line/60">
      <div className="flex flex-wrap items-start justify-between gap-3 px-4 py-3.5">
        <div className="min-w-0">
          <p className="flex flex-wrap items-center gap-2 text-[13px] font-medium">
            {data.behind ? (
              <>
                <ArrowUpCircle className="h-4 w-4 text-accent" /> {data.behind} update{data.behind === 1 ? "" : "s"} available
              </>
            ) : (
              "Plumber is up to date"
            )}
            {data.ahead ? <Badge tone="run">{data.ahead} local commit{data.ahead === 1 ? "" : "s"}</Badge> : null}
          </p>
          <p className="mt-1 font-mono text-[11.5px] text-muted">
            {data.branch ?? "detached"} at {data.head}
            {data.upstream && <> · tracking {data.upstream}</>}
          </p>
          <p className="mt-0.5 text-[11px] text-faint">
            {data.fetched === false ? `Could not reach the remote: ${data.fetch_error}` : `Checked ${ago(data.checked_at)}`} · {data.repo}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn btn-sm" onClick={check} disabled={checking || !!busy}>
            <RefreshCw className={checking ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"} /> Check now
          </button>
          {data.behind ? (
            <button className="btn btn-primary btn-sm" onClick={update} disabled={!data.can_update || !!busy} title={data.can_update ? undefined : data.reason ?? undefined}>
              {busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ArrowUpCircle className="h-3.5 w-3.5" />} {busy ?? "Update and restart"}
            </button>
          ) : null}
        </div>
      </div>
      {data.behind && !data.can_update && data.reason ? (
        <div className="px-4 py-2.5 text-xs text-run">
          Can't update from here: {data.reason}.
          {data.dirty?.length ? <span className="mt-1 block font-mono text-[11px] text-muted">{data.dirty.join(", ")}</span> : null}
        </div>
      ) : null}
      {data.commits && data.commits.length > 0 && (
        <div className="px-4 py-3">
          <p className="label mb-1.5">What's new</p>
          <ul className="space-y-1">
            {data.commits.map((c) => (
              <li key={c.sha} className="flex gap-2 text-[12.5px]">
                <span className="num shrink-0 font-mono text-[11px] text-faint">{c.sha}</span>
                <span className="min-w-0 flex-1">{c.subject}</span>
                <span className="shrink-0 text-[11px] text-faint">{ago(c.time)}</span>
              </li>
            ))}
          </ul>
          {data.deps_changed?.length ? (
            <p className="mt-2 text-[11.5px] text-run">
              Dependencies changed ({data.deps_changed.join(", ")}). If Plumber reports a missing package after the update, reinstall it into Plumber's environment.
            </p>
          ) : null}
        </div>
      )}
      {busy && <div className="px-4 py-2.5 text-xs text-muted">{busy}</div>}
      {node}
    </Card>
  );
}
