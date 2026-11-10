import { create } from "zustand";

export type Theme = "dark" | "light";
export type Rgb = readonly [number, number, number];
export type Tokens = Record<string, string>;

/**
 * THE PALETTE. Values are space-separated "R G B" so Tailwind can add opacity
 * (bg-violet/15). Legacy Tailwind names map onto these tokens (see tailwind.config.ts):
 *   navy-*  = surface scale   slate-* = text scale   violet = accent (blue)   cyan = secondary (neutral gray)
 */
export const DARK: Tokens = {
  "surface-950": "10 10 11", "surface-900": "18 18 20", "surface-800": "26 26 29",
  "surface-700": "39 39 43", "surface-600": "63 63 70", "field-border": "113 113 122",
  "fg-50": "250 250 250", "fg-100": "244 244 245", "fg-200": "228 228 231",
  "fg-300": "212 212 216", "fg-400": "161 161 170", "fg-500": "143 143 153", "fg-600": "113 113 122",
  accent: "37 99 235", "accent-soft": "96 165 250", "accent-strong": "29 78 216",
  secondary: "228 228 231",
  success: "74 222 128", warning: "250 204 21", danger: "248 113 113", "danger-fg": "254 202 202",
  "heat-from": "39 39 43", "heat-to": "96 165 250", "heat-masked": "18 18 20",
};

export const LIGHT: Tokens = {
  "surface-950": "244 244 245", "surface-900": "250 250 250", "surface-800": "255 255 255",
  "surface-700": "244 244 245", "surface-600": "212 212 216", "field-border": "113 113 122",
  "fg-50": "24 24 27", "fg-100": "39 39 42", "fg-200": "63 63 70",
  "fg-300": "82 82 91", "fg-400": "113 113 122", "fg-500": "107 107 116", "fg-600": "161 161 170",
  accent: "37 99 235", "accent-soft": "37 99 235", "accent-strong": "29 78 216",
  secondary: "63 63 70",
  success: "21 128 61", warning: "161 98 7", danger: "220 38 38", "danger-fg": "153 27 27",
  "heat-from": "244 244 245", "heat-to": "37 99 235", "heat-masked": "250 250 250",
};

export const TOKENS: Record<Theme, Tokens> = { dark: DARK, light: LIGHT };

const SERIES: Record<Theme, string[]> = {
  dark: ["#60A5FA", "#E4E4E7", "#A1A1AA", "#4ADE80", "#FACC15", "#F87171"],
  light: ["#2563EB", "#27272A", "#71717A", "#15803D", "#A16207", "#DC2626"],
};

const STORAGE_KEY = "gptlab-theme";

// ---- applying a theme ---------------------------------------------------------

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  for (const [name, value] of Object.entries(TOKENS[theme])) {
    root.style.setProperty(`--${name}`, value);
  }
  root.classList.toggle("dark", theme === "dark");
  root.dataset["theme"] = theme;
  root.style.colorScheme = theme;
}

function readStoredTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    /* storage unavailable */
  }
  if (typeof window !== "undefined" && typeof window.matchMedia === "function") {
    return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  return "dark";
}

interface ThemeState {
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggle: () => void;
}

export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: readStoredTheme(),
  setTheme: (theme) => {
    applyTheme(theme);
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* ignore */
    }
    set({ theme });
  },
  toggle: () => get().setTheme(get().theme === "dark" ? "light" : "dark"),
}));

// ---- colors for things Tailwind classes cannot reach (SVG, charts, React Flow) --

export interface ChartColors {
  grid: string;
  axis: string;
  text: string;
  accent: string;
  secondary: string;
  cursor: string;
  selectedFill: string;
  panel: string;
  border: string;
  masked: string;
  cellStroke: string;
  heatFrom: Rgb;
  heatTo: Rgb;
  series: string[];
  tooltip: { background: string; border: string; color: string };
}

const css = (t: string) => `rgb(${t.split(" ").join(", ")})`;
const rgba = (t: string, a: number) => `rgba(${t.split(" ").join(", ")}, ${a})`;
const nums = (t: string): Rgb => {
  const [r = 0, g = 0, b = 0] = t.split(" ").map(Number);
  return [r, g, b];
};

export function chartTheme(theme: Theme): ChartColors {
  const t = TOKENS[theme];
  const tok = (k: string) => t[k] ?? "0 0 0";
  return {
    grid: css(tok("surface-600")),
    axis: css(tok("fg-400")),
    text: css(tok("fg-200")),
    accent: css(tok("accent-soft")),
    secondary: css(tok("secondary")),
    cursor: rgba(tok("accent"), 0.15),
    selectedFill: rgba(tok("accent"), 0.25),
    panel: css(tok("surface-800")),
    border: css(tok("surface-600")),
    masked: css(tok("heat-masked")),
    cellStroke: css(tok("surface-800")),
    heatFrom: nums(tok("heat-from")),
    heatTo: nums(tok("heat-to")),
    series: SERIES[theme],
    tooltip: {
      background: css(tok("surface-800")),
      border: `1px solid ${css(tok("surface-600"))}`,
      color: css(tok("fg-200")),
    },
  };
}