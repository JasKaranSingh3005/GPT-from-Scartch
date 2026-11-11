/** Equal weight over the current and all earlier positions: the "running average" idea. */
export function causalAverage(values: number[]): { weights: number[][]; output: number[] } {
  const weights = values.map((_, i) => values.map((__, j) => (j <= i ? 1 / (i + 1) : 0)));
  const output = values.map((_, i) => {
    let sum = 0;
    for (let j = 0; j <= i; j++) sum += values[j] ?? 0;
    return sum / (i + 1);
  });
  return { weights, output };
}

export function parseNumberList(
  text: string, min = 2, max = 8,
): { values: number[] } | { error: string } {
  const parts = text.split(/[\s,]+/).filter((p) => p !== "");
  if (parts.length < min) return { error: `Enter at least ${min} numbers` };
  if (parts.length > max) return { error: `Use at most ${max} numbers` };
  const values: number[] = [];
  for (const p of parts) {
    const v = Number(p);
    if (!Number.isFinite(v)) return { error: `"${p}" is not a number` };
    values.push(v);
  }
  return { values };
}