import { create } from "zustand";

export type Theme = "dark" | "light";
export type Rgb = readonly [number, number, number];
export type Tokens = Record<string, string>;

/**
 * THE PALETTE. Values are space-separated "R G B" so Tailwind can add opacity
 * (bg-violet/15). Legacy Tailwind names map onto these tokens (see tailwind.config.ts):
 *   navy-*  = surface scale   slate-* = text scale   violet = accent (caramel)   cyan = secondary (cream / cocoa)
 */
export const DARK: Tokens = {
  "surface-950": "24 15 6", "surface-900": "33 21 9", "surface-800": "56 36 13",
  "surface-700": "78 52 22", "surface-600": "104 72 38", "field-border": "142 122 90",
  "fg-50": "253 251 212", "fg-100": "245 240 204", "fg-200": "232 224 190",
  "fg-300": "208 196 160", "fg-400": "186 172 138", "fg-500": "168 152 118", "fg-600": "112 96 68",
  accent: "192 88 0", "accent-soft": "238 158 84", "accent-strong": "158 74 0",
  secondary: "253 251 212",
  success: "140 200 110", warning: "246 200 60", danger: "239 83 80", "danger-fg": "254 202 202",
  "heat-from": "78 52 22", "heat-to": "238 158 84", "heat-masked": "33 21 9",
};

export const LIGHT: Tokens = {
  "surface-950": "244 238 190", "surface-900": "253 251 212", "surface-800": "255 254 244",
  "surface-700": "250 246 214", "surface-600": "222 212 160", "field-border": "150 128 90",
  "fg-50": "42 27 9", "fg-100": "56 36 13", "fg-200": "74 50 22",
  "fg-300": "96 68 34", "fg-400": "117 88 52", "fg-500": "128 98 60", "fg-600": "166 142 100",
  accent: "192 88 0", "accent-soft": "154 72 0", "accent-strong": "113 54 0",
  secondary: "113 54 0",
  success: "63 115 30", warning: "138 102 0", danger: "200 40 40", "danger-fg": "140 25 25",
  "heat-from": "247 242 200", "heat-to": "150 70 0", "heat-masked": "253 251 212",
};

export const TOKENS: Record<Theme, Tokens> = { dark: DARK, light: LIGHT };

const SERIES: Record<Theme, string[]> = {
  dark: ["#EE9E54", "#FDFBD4", "#D96C5A", "#9DBB6A", "#C99BD8", "#6FB7C9"],
  light: ["#C05800", "#713600", "#B83A4B", "#4D7C0F", "#7E4AA8", "#1F7A8C"],
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