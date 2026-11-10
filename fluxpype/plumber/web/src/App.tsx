import { Navigate, Route, Routes, useParams } from "react-router-dom";
import { Layout } from "./components/Layout";
import { CachePage } from "./pages/Cache";
import { JobDetail } from "./pages/JobDetail";
import { JobsPage } from "./pages/Jobs";
import { Overview } from "./pages/Overview";
import { PresetsPage } from "./pages/Presets";
import { SettingsPage } from "./pages/Settings";

function OldProfile() {
  const { name } = useParams();
  return <Navigate to={`/presets/${encodeURIComponent(name ?? "")}`} replace />;
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="presets" element={<PresetsPage />} />
        <Route path="presets/:name" element={<PresetsPage />} />
        <Route path="profiles" element={<Navigate to="/presets" replace />} />
        <Route path="profiles/:name" element={<OldProfile />} />
        <Route path="jobs" element={<JobsPage />} />
        <Route path="jobs/:id" element={<JobDetail />} />
        <Route path="jobs/:id/:taskId" element={<JobDetail />} />
        <Route path="cache" element={<CachePage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
