import { Background, Controls, ReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";

import type { ArchitectureStage } from "../../types/api";
import { buildFlow } from "./flow";

interface DiagramProps {
  stages: ArchitectureStage[];
  selectedKey: string | null;
  onSelect: (key: string) => void;
}

export function Diagram({ stages, selectedKey, onSelect }: DiagramProps) {
  const { nodes, edges } = useMemo(() => buildFlow(stages, selectedKey), [stages, selectedKey]);
  return (
    <div className="h-[640px] w-full rounded border border-navy-600" data-testid="diagram">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        colorMode="dark"
        fitView
        nodesDraggable={false}
        nodesConnectable={false}
        onNodeClick={(_, node) => onSelect(node.id)}
      >
        <Background />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}