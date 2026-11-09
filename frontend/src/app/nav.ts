import {
  Activity,
  BookOpen,
  Boxes,
  FlaskConical,
  GitCompare,
  LayoutDashboard,
  type LucideIcon,
  MessageSquare,
  Network,
  Settings,
  Type,
} from "lucide-react";

export type NavGroup = "Overview" | "Build" | "Inspect" | "Learn" | "System";
export const NAV_GROUPS: NavGroup[] = ["Overview", "Build", "Inspect", "Learn", "System"];

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
  group: NavGroup;
  /** Pages that are not built yet render an honest "under development" state. */
  implemented: boolean;
}

// Keep `implemented` as it is in your current file for each path; the values below are the
// current state of the project (Settings and Learning Center are still to come).
export const NAV_ITEMS: NavItem[] = [
  { path: "/", label: "Dashboard", icon: LayoutDashboard, group: "Overview", implemented: true },
  { path: "/playground", label: "GPT Playground", icon: MessageSquare, group: "Build", implemented: true },
  { path: "/training", label: "Training Lab", icon: Activity, group: "Build", implemented: true },
  { path: "/experiments", label: "Experiment Lab", icon: FlaskConical, group: "Build", implemented: true },
  { path: "/architecture", label: "Architecture Explorer", icon: Network, group: "Inspect", implemented: true },
  { path: "/attention", label: "Attention Visualizer", icon: Boxes, group: "Inspect", implemented: true },
  { path: "/tokenizer", label: "Tokenizer Explorer", icon: Type, group: "Inspect", implemented: true },
  { path: "/compare", label: "Model Comparison", icon: GitCompare, group: "Inspect", implemented: true },
  { path: "/learn", label: "Learning Center", icon: BookOpen, group: "Learn", implemented: false },
  { path: "/settings", label: "Settings", icon: Settings, group: "System", implemented: false },
];