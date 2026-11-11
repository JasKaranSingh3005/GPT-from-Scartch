import { BACKEND_DEMOS, DEMO_KEYS, MODULES } from "./modules";

const ORDER = [
  "language-modelling", "tokenization", "embeddings", "positional-encoding", "self-attention",
  "causal-masking", "multi-head-attention", "feed-forward", "residual-connections", "layernorm",
  "autoregressive-generation", "training",
];

test("lists the twelve topics from the brief, in order, with unique ids", () => {
  expect(MODULES.map((m) => m.id)).toEqual(ORDER);
  expect(new Set(MODULES.map((m) => m.id)).size).toBe(12);
});

test("written modules have content; unwritten ones have none, so nothing is faked", () => {
  for (const m of MODULES) {
    if (m.written) {
      expect(m.sections.length, m.id).toBeGreaterThan(0);
      expect(m.equations.length, m.id).toBeGreaterThan(0);
    } else {
      expect(m.sections, m.id).toEqual([]);
      expect(m.demo, m.id).toBeUndefined();
    }
  }
});

test("every demo key is known and backend demos are a subset", () => {
  for (const m of MODULES) if (m.demo) expect(DEMO_KEYS).toContain(m.demo);
  for (const d of BACKEND_DEMOS) expect(DEMO_KEYS).toContain(d);
});