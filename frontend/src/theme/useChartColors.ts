import { useMemo } from "react";

import { type ChartColors, chartTheme, useThemeStore } from "./theme";

export function useChartColors(): ChartColors {
  const theme = useThemeStore((s) => s.theme);
  return useMemo(() => chartTheme(theme), [theme]);
}