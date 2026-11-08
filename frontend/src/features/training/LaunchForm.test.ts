import { validateForm } from "./LaunchForm";

const ok = { name: "run1", model_type: "gpt" as const, n_embd: 64, n_head: 4, n_layer: 4, block_size: 64,
  dropout: 0.1, batch_size: 32, learning_rate: 0.001, max_iters: 1000, eval_interval: 100, eval_iters: 50, seed: 1 };

test("accepts a valid form", () => expect(validateForm(ok)).toBeNull());
test("rejects bad names, indivisible heads and oversized eval intervals", () => {
  expect(validateForm({ ...ok, name: "../x" })).toMatch(/Name/);
  expect(validateForm({ ...ok, n_head: 3 })).toMatch(/divisible/);
  expect(validateForm({ ...ok, eval_interval: 5000 })).toMatch(/Eval interval/);
  expect(validateForm({ ...ok, model_type: "bigram", n_head: 3 })).toBeNull();
});