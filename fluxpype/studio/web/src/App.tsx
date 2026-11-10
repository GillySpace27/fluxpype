import { Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { CachePage } from "./pages/Cache";
import { JobDetail } from "./pages/JobDetail";
import { JobsPage } from "./pages/Jobs";
import { Overview } from "./pages/Overview";
import { ProfilesPage } from "./pages/Profiles";
import { SettingsPage } from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="profiles" element={<ProfilesPage />} />
        <Route path="profiles/:name" element={<ProfilesPage />} />
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
