import { DASH, formatBytes, formatDuration, formatInt, formatLoss } from "./format";

test("formatInt groups digits and handles missing values", () => {
  expect(formatInt(211777)).toBe("211,777");
  expect(formatInt(null)).toBe(DASH);
  expect(formatInt(undefined)).toBe(DASH);
});

test("formatBytes picks a sensible unit", () => {
  expect(formatBytes(512)).toBe("512 B");
  expect(formatBytes(2048)).toBe("2.0 KB");
  expect(formatBytes(5 * 1024 ** 2)).toBe("5.0 MB");
});

test("formatLoss never invents a number", () => {
  expect(formatLoss(1.66584)).toBe("1.6658");
  expect(formatLoss(null)).toBe(DASH);
});

test("formatDuration", () => {
  expect(formatDuration(12)).toBe("12.0 s");
  expect(formatDuration(125)).toBe("2 min 5 s");
  expect(formatDuration(null)).toBe(DASH);
});