import type { Edge, Node } from "@xyflow/react";

import type { ArchitectureStage } from "../../types/api";
import { formatInt } from "../../utils/format";

const ROW_HEIGHT = 110;
const SIDE_X = 300;

/**
 * Vertical chain of stages. The positional embedding sits beside the token embedding
 * and feeds the stage after it, because its output is added to the token embedding.
 */
export function buildFlow(
  stages: ArchitectureStage[],
  selectedKey: string | null,
): { nodes: Node[]; edges: Edge[] } {
  let row = -1;
  const nodes: Node[] = stages.map((s) => {
    const side = s.key === "position_embedding";
    if (!side) row += 1;
    const selected = s.key === selectedKey;
    return {
      id: s.key,
      position: { x: side ? SIDE_X : 0, y: row * ROW_HEIGHT },
      data: { label: s.parameters > 0 ? `${s.label}\n${formatInt(s.parameters)} parameters` : s.label },
      draggable: false,
      style: {
        width: 250,
        whiteSpace: "pre-line",
        fontSize: 12,
        color: "#e2e8f0",
        background: selected ? "#2a1f5c" : "#111833",
        border: `1px solid ${selected ? "#8b5cf6" : "#26325c"}`,
        borderRadius: 8,
      },
    };
  });

  const main = stages.filter((s) => s.key !== "position_embedding");
  const edges: Edge[] = [];
  for (let i = 0; i + 1 < main.length; i++) {
    const a = main[i];
    const b = main[i + 1];
    if (a && b) edges.push({ id: `${a.key}->${b.key}`, source: a.key, target: b.key });
  }
  const pos = stages.find((s) => s.key === "position_embedding");
  const tok = main.findIndex((s) => s.key === "token_embedding");
  const after = tok >= 0 ? main[tok + 1] : undefined;
  if (pos && after) {
    edges.push({ id: `${pos.key}->${after.key}`, source: pos.key, target: after.key, label: "+ positions" });
  }
  return { nodes, edges };
}