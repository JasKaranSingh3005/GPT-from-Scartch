import type { Config } from "tailwindcss";

/** Every color is a CSS variable set by src/theme/theme.ts, so one class works in both themes. */
const v = (name: string) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Legacy names kept so components need no edits. The names no longer describe the colors:
        // violet is the accent (blue), cyan is the secondary (neutral gray). Values live in theme.ts.
        navy: { 950: v("surface-950"), 900: v("surface-900"), 800: v("surface-800"), 700: v("surface-700"), 600: v("surface-600") },
        slate: { 50: v("fg-50"), 100: v("fg-100"), 200: v("fg-200"), 300: v("fg-300"), 400: v("fg-400"), 500: v("fg-500"), 600: v("fg-600") },
        // Form-field outlines: 3:1 against both the panel and the field background (WCAG 1.4.11).
        field: v("field-border"),
        violet: { DEFAULT: v("accent"), soft: v("accent-soft"), strong: v("accent-strong") },
        cyan: { DEFAULT: v("secondary"), soft: v("secondary") },
        amber: { 300: v("warning") },
        red: { 200: v("danger-fg"), 500: v("danger") },
        emerald: { 400: v("success") },
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;