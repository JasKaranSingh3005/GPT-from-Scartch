import { create } from "zustand";

import type { FinishedRun } from "../hooks/useGenerationStream";

export interface PlaygroundSettings {
  modelId: string | null;
  prompt: string;
  maxNewTokens: number;
  temperature: number;
  topK: number | null;
  seed: number | null;
}

export const DEFAULT_SETTINGS: PlaygroundSettings = {
  modelId: null,
  prompt: "ROMEO:",
  maxNewTokens: 200,
  temperature: 0.8,
  topK: 40,
  seed: null,
};

export interface HistoryEntry extends FinishedRun {
  id: number;
  at: string;
}

const MAX_HISTORY = 20;
let nextId = 1;

interface PlaygroundState {
  settings: PlaygroundSettings;
  history: HistoryEntry[];
  update: (patch: Partial<PlaygroundSettings>) => void;
  addHistory: (run: FinishedRun) => void;
  clearHistory: () => void;
  restore: (entry: HistoryEntry) => void;
}

/** History lives in memory only: it is lost on reload. */
export const usePlaygroundStore = create<PlaygroundState>((set) => ({
  settings: DEFAULT_SETTINGS,
  history: [],
  update: (patch) => set((s) => ({ settings: { ...s.settings, ...patch } })),
  addHistory: (run) =>
    set((s) => ({
      history: [{ ...run, id: nextId++, at: new Date().toISOString() }, ...s.history].slice(
        0,
        MAX_HISTORY,
      ),
    })),
  clearHistory: () => set({ history: [] }),
  restore: (entry) =>
    set({
      settings: {
        modelId: entry.request.model_id,
        prompt: entry.request.prompt,
        maxNewTokens: entry.request.max_new_tokens,
        temperature: entry.request.temperature,
        topK: entry.request.top_k,
        seed: entry.request.seed,
      },
    }),
}));