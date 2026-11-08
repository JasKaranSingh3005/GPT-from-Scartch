import { Navigate, Route, Routes } from "react-router-dom";
import { Playground } from "../pages/Playground";
import { Dashboard } from "../pages/Dashboard";
import { UnderDevelopment } from "../pages/UnderDevelopment";
import { Layout } from "./Layout";
import { NAV_ITEMS } from "./nav";
import { AttentionVisualizer } from "../pages/AttentionVisualizer";
import { TokenizerExplorer } from "../pages/TokenizerExplorer";
import { ArchitectureExplorer } from "../pages/ArchitectureExplorer";
import { TrainingLab } from "../pages/TrainingLab";

export function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="playground" element={<Playground />} />
        <Route path="tokenizer" element={<TokenizerExplorer />} />
        <Route path="attention" element={<AttentionVisualizer />} />
        <Route path="architecture" element={<ArchitectureExplorer />} />
        <Route path="training" element={<TrainingLab />} />
        {NAV_ITEMS.filter((item) => !item.implemented).map((item) => (
          <Route key={item.path} path={item.path} element={<UnderDevelopment title={item.label} />} />
        ))}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}