import { mergeRow } from "./useTrainingSocket";

const r = (iter: number, v = 1) => ({ iter, train_loss: v, val_loss: v, elapsed_s: iter });

test("mergeRow keeps rows sorted and replaces a replayed iteration", () => {
  let rows = [r(10), r(0)];
  rows = mergeRow(rows, r(5));
  expect(rows.map((x) => x.iter)).toEqual([0, 5, 10]);
  rows = mergeRow(rows, r(5, 9));
  expect(rows).toHaveLength(3);
  expect(rows[1]?.val_loss).toBe(9);
});