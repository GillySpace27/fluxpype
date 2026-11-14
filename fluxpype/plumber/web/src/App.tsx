import { Navigate, Route, Routes, useLocation, useParams } from "react-router-dom";
import { Layout } from "./components/Layout";
import { CachePage } from "./pages/Cache";
import { JobDetail } from "./pages/JobDetail";
import { JobsPage } from "./pages/Jobs";
import { NewRun } from "./pages/NewRun";
import { Overview } from "./pages/Overview";
import { PresetsPage } from "./pages/Presets";
import { ScenesPage } from "./pages/Scenes";
import { SettingsPage } from "./pages/Settings";

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
