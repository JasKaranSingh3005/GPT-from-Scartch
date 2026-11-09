import { mergeCurves, spread } from "./experiments";

const p = (iter: number, v: number) => ({ iter, train_loss: v + 0.1, val_loss: v });

test("mergeCurves builds one row per iteration with a column per variant", () => {
  const rows = mergeCurves(
    { a: [p(10, 2), p(0, 4)], b: [p(0, 5)], c: [p(0, 9)] }, ["a", "b"], "val_loss");
  expect(rows).toEqual([{ iter: 0, a: 4, b: 5 }, { iter: 10, a: 2 }]);
  expect(mergeCurves({}, ["missing"], "val_loss")).toEqual([]);
});

test("spread ignores missing values and needs two numbers", () => {
  expect(spread([2.0, 2.1, null])).toBeCloseTo(0.1);
  expect(spread([2.0, null])).toBeNull();
  expect(spread([])).toBeNull();
});