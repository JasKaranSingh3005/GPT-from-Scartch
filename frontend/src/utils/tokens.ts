import type { EncodedToken } from "../types/api";

export interface CharCount {
  char: string;
  count: number;
}

/** Characters of the given tokens, most frequent first (ties broken alphabetically). */
export function charFrequencies(tokens: EncodedToken[]): CharCount[] {
  const counts = new Map<string, number>();
  for (const t of tokens) counts.set(t.char, (counts.get(t.char) ?? 0) + 1);
  return [...counts.entries()]
    .map(([char, count]) => ({ char, count }))
    .sort((a, b) => b.count - a.count || a.char.localeCompare(b.char));
}

/** Element-wise mean over heads: weights[head][query][key] -> [query][key]. */
export function averageHeads(weights: number[][][]): number[][] {
  const first = weights[0];
  if (!first) return [];
  return first.map((row, q) =>
    row.map((_, k) => weights.reduce((sum, head) => sum + (head[q]?.[k] ?? 0), 0) / weights.length),
  );
}

/** How many leading tokens fall outside a context window of `blockSize`. */
export function tokensOutsideWindow(length: number, blockSize: number | null): number {
  return blockSize === null ? 0 : Math.max(0, length - blockSize);
}

/** Parse "1, 2 3" into integers; returns an error message for anything else. */
export function parseIds(text: string): { ids: number[] } | { error: string } {
  const parts = text.split(/[\s,]+/).filter((p) => p !== "");
  if (parts.length === 0) return { error: "Enter at least one token ID" };
  const ids: number[] = [];
  for (const p of parts) {
    if (!/^\d+$/.test(p)) return { error: `"${p}" is not a non-negative integer` };
    ids.push(Number(p));
  }
  return { ids };
}