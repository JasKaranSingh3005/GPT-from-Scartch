import { causalAverage, layerNorm, nextCharPairs, parseNumberList } from "./learn";

test("causalAverage is the running mean, with zero weight on the future", () => {
  const { weights, output } = causalAverage([1, 3, 5]);
  expect(output).toEqual([1, 2, 3]);
  expect(weights[0]).toEqual([1, 0, 0]);
  expect(weights[2]?.every((w) => Math.abs(w - 1 / 3) < 1e-12)).toBe(true);
  weights.forEach((row) => expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1));
});

test("parseNumberList validates count and content", () => {
  expect(parseNumberList("1, 2 3")).toEqual({ values: [1, 2, 3] });
  expect(parseNumberList("1")).toHaveProperty("error");
  expect(parseNumberList("1, x")).toHaveProperty("error");
  expect(parseNumberList("1 2 3 4 5 6 7 8 9")).toHaveProperty("error");
});

test("layerNorm gives zero mean and (almost) unit spread whatever the input scale", () => {
  for (const scale of [0.5, 3, 40]) {
    const { output } = layerNorm([1, 2, 3, 4, 5, 6, 7, 8].map((v) => v * scale + 100));
    const mean = output.reduce((a, b) => a + b, 0) / output.length;
    const std = Math.sqrt(output.reduce((a, b) => a + (b - mean) ** 2, 0) / output.length);
    expect(Math.abs(mean)).toBeLessThan(1e-9);
    expect(std).toBeGreaterThan(0.99);
    expect(std).toBeLessThanOrEqual(1);
  }
});

test("nextCharPairs pairs every prefix with the character that follows it", () => {
  expect(nextCharPairs("abc")).toEqual([
    { context: "a", target: "b" },
    { context: "ab", target: "c" },
  ]);
  expect(nextCharPairs("a")).toEqual([]);
  expect(nextCharPairs("")).toEqual([]);
});