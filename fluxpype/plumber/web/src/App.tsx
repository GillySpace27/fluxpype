import { Navigate, Route, Routes, useParams } from "react-router-dom";
import { Layout } from "./components/Layout";
import { CachePage } from "./pages/Cache";
import { JobDetail } from "./pages/JobDetail";
import { JobsPage } from "./pages/Jobs";
import { Overview } from "./pages/Overview";
import { PresetsPage } from "./pages/Presets";
import { NewRun, RunIndex, RunPage } from "./pages/RunView";
import { storedLayout } from "./lib/layoutMode";
import { SettingsPage } from "./pages/Settings";

/** The overview opens in the remembered layout. */
function Home() {
  return storedLayout() === "run" ? <Navigate to="/run" replace /> : <Overview />;
}

function RunTabRedirect() {
  const { id } = useParams();
  return <Navigate to={`/run/${id}/progress`} replace />;
}

function OldProfile() {
  const { name } = useParams();
  return <Navigate to={`/presets/${encodeURIComponent(name ?? "")}`} replace />;
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="overview" element={<Overview />} />
        <Route path="presets" element={<PresetsPage />} />
        <Route path="presets/:name" element={<PresetsPage />} />
        <Route path="profiles" element={<Navigate to="/presets" replace />} />
        <Route path="profiles/:name" element={<OldProfile />} />
        <Route path="jobs" element={<JobsPage />} />
        <Route path="jobs/:id" element={<JobDetail />} />
        <Route path="jobs/:id/:taskId" element={<JobDetail />} />
        <Route path="cache" element={<CachePage />} />
        <Route path="run" element={<RunIndex />} />
        <Route path="run/new" element={<NewRun />} />
        <Route path="run/:id" element={<RunTabRedirect />} />
        <Route path="run/:id/:tab" element={<RunPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
