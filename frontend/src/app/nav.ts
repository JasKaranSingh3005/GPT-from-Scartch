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

export interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
  /** Pages that are not built yet render an honest "under development" state. */
  implemented: boolean;
}

export const NAV_ITEMS: NavItem[] = [
  { path: "/", label: "Dashboard", icon: LayoutDashboard, implemented: true },
  { path: "/playground", label: "GPT Playground", icon: MessageSquare, implemented: true },
  { path: "/architecture", label: "Architecture Explorer", icon: Network, implemented: true },
  { path: "/attention", label: "Attention Visualizer", icon: Boxes, implemented: true },
  { path: "/tokenizer", label: "Tokenizer Explorer", icon: Type, implemented: true },
  { path: "/training", label: "Training Lab", icon: Activity, implemented: true },
  { path: "/experiments", label: "Experiment Lab", icon: FlaskConical, implemented: true },
  { path: "/compare", label: "Model Comparison", icon: GitCompare, implemented: true },
  { path: "/learn", label: "Learning Center", icon: BookOpen, implemented: false },
  { path: "/settings", label: "Settings", icon: Settings, implemented: false },
];