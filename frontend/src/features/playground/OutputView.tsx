import clsx from "clsx";

import type { StreamToken } from "../../types/api";
import { visibleChar } from "../../utils/format";

interface OutputViewProps {
  prompt: string;
  tokens: StreamToken[];
  showTokens: boolean;
  streaming: boolean;
}

export function OutputView({ prompt, tokens, showTokens, streaming }: OutputViewProps) {
  if (showTokens) {
    return (
      <div className="min-h-32 rounded border border-navy-600 bg-navy-900 p-3 font-mono text-sm">
        <div className="mb-2 text-xs text-slate-500">Prompt: {prompt || "(newline start)"}</div>
        <div className="flex flex-wrap gap-1" data-testid="output">
          {tokens.map((t) => (
            <span
              key={t.index}
              title={`token id ${t.id}`}
              className="rounded bg-violet/15 px-1.5 py-0.5 text-violet-soft"
            >
              {visibleChar(t.text)}
              <sub className="ml-0.5 text-[9px] text-slate-500">{t.id}</sub>
            </span>
          ))}
        </div>
      </div>
    );
  }
  return (
    <pre
      data-testid="output"
      className={clsx(
        "min-h-32 whitespace-pre-wrap break-words rounded border border-navy-600",
        "bg-navy-900 p-3 font-mono text-sm",
        streaming && "border-violet/50",
      )}
    >
      <span className="text-slate-500">{prompt}</span>
      <span className="text-slate-100">{tokens.map((t) => t.text).join("")}</span>
    </pre>
  );
}