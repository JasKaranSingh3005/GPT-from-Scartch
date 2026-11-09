import { create } from "zustand";

const KEY = "gptlab-sidebar-collapsed";

function initialCollapsed(): boolean {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "1") return true;
    if (saved === "0") return false;
  } catch {
    /* storage unavailable */
  }
  return typeof window !== "undefined" && window.innerWidth < 1024; // tablets start collapsed
}

interface UiState {
  collapsed: boolean;
  toggleSidebar: () => void;
}

export const useUiStore = create<UiState>((set, get) => ({
  collapsed: initialCollapsed(),
  toggleSidebar: () => {
    const next = !get().collapsed;
    try {
      localStorage.setItem(KEY, next ? "1" : "0");
    } catch {
      /* ignore */
    }
    set({ collapsed: next });
  },
}));