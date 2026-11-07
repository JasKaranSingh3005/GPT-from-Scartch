import { Navigate, Route, Routes } from "react-router-dom";
import { Playground } from "../pages/Playground";
import { Dashboard } from "../pages/Dashboard";
import { UnderDevelopment } from "../pages/UnderDevelopment";
import { Layout } from "./Layout";
import { NAV_ITEMS } from "./nav";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="playground" element={<Playground />} />
        {NAV_ITEMS.filter((item) => !item.implemented).map((item) => (
          <Route key={item.path} path={item.path} element={<UnderDevelopment title={item.label} />} />
        ))}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}