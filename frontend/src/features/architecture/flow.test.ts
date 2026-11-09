import type { ArchitectureStage } from "../../types/api";
import { statsForStage } from "../../utils/architecture";
import { buildFlow } from "./flow";
import { chartTheme } from "../../theme/theme";

const stage = (key: string, parameters = 0): ArchitectureStage => ({
  key, label: key, input_shape: "", output_shape: "", parameters, details: {},
});
const STAGES = ["tokenizer", "token_embedding", "position_embedding", "blocks", "ln_f", "lm_head", "softmax"]
  .map((k) => stage(k, k === "blocks" ? 100 : 0));

test("builds one node per stage and a chain that skips the side node", () => {
  const { nodes, edges } = buildFlow(STAGES, null);
  expect(nodes).toHaveLength(7);
  const ids = edges.map((e) => e.id);
  expect(ids).toContain("tokenizer->token_embedding");
  expect(ids).toContain("token_embedding->blocks");
  expect(ids).toContain("position_embedding->blocks"); // positions are added before the blocks
  expect(ids).not.toContain("token_embedding->position_embedding");
  expect(edges).toHaveLength(6);
});

test("the positional embedding sits beside the token embedding", () => {
  const { nodes } = buildFlow(STAGES, null);
  const tok = nodes.find((n) => n.id === "token_embedding");
  const pos = nodes.find((n) => n.id === "position_embedding");
  expect(pos?.position.y).toBe(tok?.position.y);
  expect(pos?.position.x).toBeGreaterThan(tok?.position.x ?? 0);
});

test("only the selected node is highlighted and parameter counts appear in labels", () => {
  const accent = chartTheme("dark").accent;
  const { nodes } = buildFlow(STAGES, "blocks");
  const border = (id: string) => String(nodes.find((n) => n.id === id)?.style?.border);
  expect(border("blocks")).toContain(accent);
  expect(border("ln_f")).not.toContain(accent);
  expect(String(nodes.find((n) => n.id === "blocks")?.data.label)).toContain("100 parameters");
});

test("statsForStage groups every block under 'blocks'", () => {
  const stat = (name: string) => ({ name, shape: [1], mean: 0, std: 0, min: 0, max: 0 });
  const all = [stat("token_embedding"), stat("blocks.0"), stat("blocks.1"), stat("ln_f")];
  expect(statsForStage("blocks", all).map((s) => s.name)).toEqual(["blocks.0", "blocks.1"]);
  expect(statsForStage("ln_f", all)).toHaveLength(1);
  expect(statsForStage("softmax", all)).toEqual([]);
});