import { averageHeads, charFrequencies, parseIds, tokensOutsideWindow } from "./tokens";

const tokens = [..."banana"].map((char, id) => ({ char, id }));

test("charFrequencies counts and sorts, ties alphabetically", () => {
  expect(charFrequencies(tokens)).toEqual([
    { char: "a", count: 3 },
    { char: "n", count: 2 },
    { char: "b", count: 1 },
  ]);
});

test("averageHeads is the element-wise mean", () => {
  const heads = [[[1, 0], [0.5, 0.5]], [[1, 0], [0.1, 0.9]]];
  const avg = averageHeads(heads);
  expect(avg[0]).toEqual([1, 0]);
  expect(avg[1]?.[0]).toBeCloseTo(0.3);
  expect(avg[1]?.[1]).toBeCloseTo(0.7);
  expect(averageHeads([])).toEqual([]);
});

test("tokensOutsideWindow", () => {
  expect(tokensOutsideWindow(100, 64)).toBe(36);
  expect(tokensOutsideWindow(10, 64)).toBe(0);
  expect(tokensOutsideWindow(100, null)).toBe(0);
});

test("parseIds accepts commas and spaces and rejects junk", () => {
  expect(parseIds("1, 2 3")).toEqual({ ids: [1, 2, 3] });
  expect(parseIds("")).toHaveProperty("error");
  expect(parseIds("1, x")).toHaveProperty("error");
  expect(parseIds("-3")).toHaveProperty("error");
});