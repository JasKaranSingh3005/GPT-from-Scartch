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

export function layerNorm(values: number[], eps = 1e-5): { mean: number; std: number; output: number[] } {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / n; // population variance
  const denom = Math.sqrt(variance + eps);
  return { mean, std: Math.sqrt(variance), output: values.map((v) => (v - mean) / denom) };
}

/** What the model sees and must predict at each position of a text. */
export function nextCharPairs(text: string): { context: string; target: string }[] {
  const chars = [...text];
  return chars.slice(1).map((target, i) => ({ context: chars.slice(0, i + 1).join(""), target }));
}