import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        navy: { 950: "#080c1a", 900: "#0b1020", 800: "#111833", 700: "#1a2447", 600: "#26325c" },
        violet: { DEFAULT: "#8b5cf6", soft: "#a78bfa" },
        cyan: { DEFAULT: "#22d3ee", soft: "#67e8f9" },
      },
      fontFamily: {
        sans: ["Inter Variable", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono Variable", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;