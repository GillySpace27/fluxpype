import { lazy } from "react";
import { Navigate, Route, Routes, useLocation, useParams } from "react-router-dom";
import { Layout } from "./components/Layout";

// Pages load on first visit; the shell (Layout, live provider) stays in the main chunk.
const CachePage = lazy(() => import("./pages/Cache").then((m) => ({ default: m.CachePage })));
const ComparePage = lazy(() => import("./pages/Compare").then((m) => ({ default: m.ComparePage })));
const JobDetail = lazy(() => import("./pages/JobDetail").then((m) => ({ default: m.JobDetail })));
const JobsPage = lazy(() => import("./pages/Jobs").then((m) => ({ default: m.JobsPage })));
const NewRun = lazy(() => import("./pages/NewRun").then((m) => ({ default: m.NewRun })));
const Overview = lazy(() => import("./pages/Overview").then((m) => ({ default: m.Overview })));
const PresetsPage = lazy(() => import("./pages/Presets").then((m) => ({ default: m.PresetsPage })));
const ScenesPage = lazy(() => import("./pages/Scenes").then((m) => ({ default: m.ScenesPage })));
const SettingsPage = lazy(() => import("./pages/Settings").then((m) => ({ default: m.SettingsPage })));

/** The old Run layout's tab names, as tabs of the job page. */
const RUN_TAB: Record<string, string> = { progress: "progress", settings: "settings", cache: "files", results: "results" };

/** /run/:id/:tab?task=X&stage=Y (the old Run layout) becomes /jobs/:id/X?tab=...&stage=Y. */
function RunRedirect() {
  const { id, tab } = useParams();
  const { search } = useLocation();
  const q = new URLSearchParams(search);
  const task = q.get("task");
  q.delete("task");
  if (tab && RUN_TAB[tab]) q.set("tab", RUN_TAB[tab]);
  const rest = q.toString();
  return <Navigate to={`/jobs/${id}${task ? `/${task}` : ""}${rest ? `?${rest}` : ""}`} replace />;
}

/** /run/new keeps its ?preset= and ?from= parameters. */
function RunNewRedirect() {
  const { search } = useLocation();
  return <Navigate to={`/jobs/new${search}`} replace />;
}

function OldProfile() {
  const { name } = useParams();
  return <Navigate to={`/presets/${encodeURIComponent(name ?? "")}`} replace />;
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="overview" element={<Overview />} />
        <Route path="presets" element={<PresetsPage />} />
        <Route path="presets/:name" element={<PresetsPage />} />
        <Route path="profiles" element={<Navigate to="/presets" replace />} />
        <Route path="profiles/:name" element={<OldProfile />} />
        <Route path="jobs" element={<JobsPage />} />
        <Route path="jobs/new" element={<NewRun />} />
        <Route path="jobs/:id" element={<JobDetail />} />
        <Route path="jobs/:id/:taskId" element={<JobDetail />} />
        <Route path="jobs/:id/:taskId/compare" element={<ComparePage />} />
        <Route path="scenes" element={<ScenesPage />} />
        <Route path="cache" element={<CachePage />} />
        <Route path="run" element={<Navigate to="/jobs" replace />} />
        <Route path="run/new" element={<RunNewRedirect />} />
        <Route path="run/:id" element={<RunRedirect />} />
        <Route path="run/:id/:tab" element={<RunRedirect />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
