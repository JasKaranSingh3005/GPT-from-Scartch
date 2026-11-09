import { applyTheme, chartTheme, DARK, LIGHT, TOKENS, useThemeStore, type Theme } from "./theme";

const rgb = (t: string | undefined): [number, number, number] => {
  const [r = 0, g = 0, b = 0] = (t ?? "").split(" ").map(Number);
  return [r, g, b];
};
const channel = (v: number) => {
  const s = v / 255;
  return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
};
const luminance = (t: string | undefined) => {
  const [r, g, b] = rgb(t);
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
};
const contrast = (a: string | undefined, b: string | undefined) => {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (hi + 0.05) / (lo + 0.05);
};
const WHITE = "255 255 255";

test("both themes define exactly the same valid tokens", () => {
  expect(Object.keys(DARK).sort()).toEqual(Object.keys(LIGHT).sort());
  for (const tokens of Object.values(TOKENS)) {
    for (const [name, value] of Object.entries(tokens)) {
      const parts = value.split(" ").map(Number);
      expect(parts, name).toHaveLength(3);
      expect(parts.every((n) => Number.isInteger(n) && n >= 0 && n <= 255), name).toBe(true);
    }
  }
});

describe.each<Theme>(["dark", "light"])("%s theme contrast (WCAG AA, 4.5:1)", (theme) => {
  const t = TOKENS[theme];
  const pairs: [string, string | undefined, string | undefined][] = [
    ["strong text on panel", t["fg-300"], t["surface-800"]],
    ["muted text on panel", t["fg-400"], t["surface-800"]],
    ["muted text on canvas", t["fg-400"], t["surface-900"]],
    ["hint text on panel", t["fg-500"], t["surface-800"]],
    ["hint text on canvas", t["fg-500"], t["surface-900"]],
    ["accent text on panel", t["accent-soft"], t["surface-800"]],
    ["secondary text on panel", t["secondary"], t["surface-800"]],
    ["warning text on panel", t["warning"], t["surface-800"]],
    ["danger text on panel", t["danger-fg"], t["surface-800"]],
    ["button label on accent", WHITE, t["accent"]],
    ["button label on hover", WHITE, t["accent-strong"]],
  ];
  test.each(pairs)("%s", (_name, fg, bg) => {
    expect(contrast(fg, bg)).toBeGreaterThanOrEqual(4.5);
  });
});

test("chart colors follow the theme", () => {
  const d = chartTheme("dark");
  const l = chartTheme("light");
  expect(d.grid).not.toBe(l.grid);
  expect(d.series).toHaveLength(6);
  expect(new Set(d.series).size).toBe(6);
  expect(d.tooltip.border).toContain("rgb(");
});

test("setTheme applies the theme to the document and remembers it", () => {
  useThemeStore.getState().setTheme("light");
  expect(document.documentElement.dataset["theme"]).toBe("light");
  expect(document.documentElement.classList.contains("dark")).toBe(false);
  expect(localStorage.getItem("gptlab-theme")).toBe("light");
  useThemeStore.getState().toggle();
  expect(useThemeStore.getState().theme).toBe("dark");
  expect(document.documentElement.classList.contains("dark")).toBe(true);
  applyTheme("dark");
});