export const DASH = "\u2014";

export function formatInt(n: number | null | undefined): string {
  return n === null || n === undefined ? DASH : n.toLocaleString("en-US");
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(1)} GB`;
}

export function formatLoss(n: number | null | undefined): string {
  return n === null || n === undefined ? DASH : n.toFixed(4);
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return DASH;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  return `${m} min ${Math.round(seconds - m * 60)} s`;
}

/** Make whitespace characters visible in token displays. */
export function visibleChar(c: string): string {
  if (c === " ") return "\u2423";
  if (c === "\n") return "\u21b5";
  if (c === "\t") return "\u21e5";
  return c;
}