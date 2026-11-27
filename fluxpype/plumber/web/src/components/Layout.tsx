import { Boxes, Database, Gauge, Layers, ListChecks, Settings } from "lucide-react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { cn } from "../lib/cn";
import { useJobs, useStatus } from "../lib/hooks";
import { RunMini } from "./RunCard";
import { useLive } from "../lib/live";
import { FlaskConical } from "lucide-react";
import { QueuePill } from "./QueuePill";
import { AskBar } from "./AskBar";
import { SysMonPanel } from "./SysMon";
import { ConnectAgentButton } from "./ConnectAgent";
import { NotificationsBell } from "./Notifications";
import { BackendSwitcher, StatusLight } from "./SystemBar";
import { UpdatePill } from "./Updates";

const NAV_PRESETS = [
  { to: "/", label: "Overview", icon: Gauge, end: true },
  { to: "/presets", label: "Presets", icon: Layers },
  { to: "/jobs", label: "Runs", icon: ListChecks },
  { to: "/scenes", label: "Scenes", icon: Boxes },
  { to: "/cache", label: "Cache", icon: Database },
  { to: "/settings", label: "Settings", icon: Settings },
];

function Logo({ compact }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5 px-2">
      <div className="h-8 w-8 shrink-0 overflow-hidden rounded-xl shadow-[0_6px_20px_-6px_rgb(var(--accent)/0.6)]">
        {/* The dark mark is the app's identity in both themes (dock, favicon, sidebar). */}
        <img src="/icons/plumber-dark-small.svg" alt="" className="h-full w-full" />
      </div>
      <div className={cn("leading-none", compact && "hidden")}>
        <p className="text-[15px] font-semibold tracking-tight">Fluxpype Plumber</p>
        <p className="mt-1 text-[10.5px] uppercase tracking-[0.14em] text-faint">fluxpype</p>
      </div>
    </div>
  );
}

/** The last few runs under the navigation, so a run's page is one click away from anywhere. */
function RecentRunsNav() {
  const { data } = useJobs();
  const { pathname } = useLocation();
  const jobs = data?.jobs.slice(0, 5) ?? [];
  if (!jobs.length) return null;
  const open = pathname.startsWith("/jobs/") ? pathname.split("/")[2] : null;
  return (
    <div className="mt-6">
      <div className="mb-1 flex items-center justify-between px-2.5">
        <p className="text-[10.5px] font-medium uppercase tracking-[0.14em] text-faint">Recent runs</p>
        <Link to="/jobs" className="text-[10.5px] text-faint hover:text-ink">
          All
        </Link>
      </div>
      <div className="flex flex-col gap-0.5">
        {jobs.map((j) => (
          <RunMini key={j.id} job={j} active={open === j.id} />
        ))}
      </div>
    </div>
  );
}

export function Layout() {
  const { data: status } = useStatus();
  const { connected, restarting } = useLive();
  const NAV = NAV_PRESETS;
  const { data: jobs } = useJobs();
  const active = jobs?.jobs.filter((j) => j.status === "running").length ?? 0;
  return (
    <div className="min-h-screen md:pl-[220px]">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-[220px] flex-col overflow-y-auto border-r hair bg-surface/70 px-3 py-5 backdrop-blur md:flex">
        <Logo />
        <nav className="mt-8 flex flex-col gap-0.5">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.end}
              className={({ isActive }) =>
                cn(
                  "group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13px] font-medium transition-colors",
                  isActive ? "bg-raised text-ink shadow-sm" : "text-muted hover:bg-raised/60 hover:text-ink",
                )
              }
            >
              {({ isActive }) => (
                <>
                  <n.icon className={cn("h-4 w-4", isActive ? "text-accent" : "text-faint group-hover:text-muted")} />
                  {n.label}
                  {n.to === "/jobs" && active > 0 && (
                    <span className="num ml-auto inline-flex items-center gap-1 rounded-full bg-run/15 px-1.5 text-[10.5px] font-semibold text-run">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-run" />
                      {active}
                    </span>
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>
        <RecentRunsNav />
        <div className="mt-auto space-y-3 pt-4">
          <SysMonPanel />
          <div className="space-y-2 px-2 text-[11px] text-faint">
            <div className="flex items-center gap-2">
              <span className={cn("h-1.5 w-1.5 rounded-full", connected ? "bg-ok" : "bg-run animate-pulse")} />
              {connected ? "Live" : "Reconnecting..."}
            </div>
            {status && <p className="num">v{status.version}</p>}
          </div>
        </div>
      </aside>

      <header className="sticky top-0 z-20 border-b hair bg-bg/70 backdrop-blur">
        <div className="flex h-14 items-center justify-between gap-3 px-4 md:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <div className="md:hidden">
              <Logo compact />
            </div>
            <BackendSwitcher />
          </div>
          <div className="hidden min-w-0 flex-1 justify-center px-2 md:flex">
            <AskBar />
          </div>
          <div className="flex items-center gap-2">
            <UpdatePill />
            <StatusLight />
            <NotificationsBell />
            <ConnectAgentButton live />
            <QueuePill />
          </div>
        </div>
        {restarting && (
          <div role="status" className="flex items-center gap-2 border-t border-accent/25 bg-accent/10 px-4 py-1.5 text-xs text-ink/90 md:px-8">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-run" />
            Plumber is restarting. This page reloads when it is back.
          </div>
        )}
        {status?.sandboxed && (
          <div className="flex items-center gap-2 border-t border-accent/25 bg-accent/10 px-4 py-1.5 text-xs text-ink/90 md:px-8">
            <FlaskConical className="h-3.5 w-3.5 text-accent" />
            Simulated runs: nothing real is computed, and results go to a sandbox, not your data folder.
          </div>
        )}
      </header>

      <main className="mx-auto w-full max-w-[1280px] px-4 pb-28 pt-6 md:px-8 md:pb-12">
        <Outlet />
      </main>

      <nav className="fixed inset-x-0 bottom-0 z-30 flex border-t hair bg-surface/90 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden" aria-label="Primary">
        {NAV.map((n) => (
          <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => cn("flex flex-1 flex-col items-center gap-1 py-2.5 text-[10.5px] font-medium", isActive ? "text-accent" : "text-muted")}>
            <n.icon className="h-5 w-5" />
            {n.label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
