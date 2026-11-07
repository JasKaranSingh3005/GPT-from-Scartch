import { create } from "zustand";

interface ModelState {
  selectedModelId: string | null;
  select: (id: string | null) => void;
}

/** The model the user is inspecting. Later pages (Playground, Attention) share it. */
export const useModelStore = create<ModelState>((set) => ({
  selectedModelId: null,
  select: (id) => set({ selectedModelId: id }),
}));