import { causalAverage, parseNumberList } from "./learn";

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