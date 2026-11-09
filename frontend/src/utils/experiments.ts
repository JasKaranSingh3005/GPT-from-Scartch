import type { CurvePoint } from "../types/api";

/** One chart row per iteration, one column per variant: [{iter, base: 2.1, heads_1: 2.2}, ...]. */
export function mergeCurves(
  curves: Record<string, CurvePoint[]>,
  names: string[],
  field: "train_loss" | "val_loss",
): Record<string, number>[] {
  const byIter = new Map<number, Record<string, number>>();
  for (const name of names) {
    for (const p of curves[name] ?? []) {
      const entry = byIter.get(p.iter) ?? { iter: p.iter };
      entry[name] = p[field];
      byIter.set(p.iter, entry);
    }
  }
  return [...byIter.values()].sort((a, b) => (a["iter"] ?? 0) - (b["iter"] ?? 0));
}

/** Largest minus smallest of the numbers that are present; null if fewer than two. */
export function spread(values: (number | null | undefined)[]): number | null {
  const v = values.filter((x): x is number => typeof x === "number");
  return v.length < 2 ? null : Math.max(...v) - Math.min(...v);
}