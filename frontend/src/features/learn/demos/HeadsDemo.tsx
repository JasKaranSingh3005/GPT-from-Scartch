import { useState } from "react";

import { ErrorState } from "../../../components/ui/states";
import { useAttention, useModelConfig } from "../../../hooks/useApi";
import { useDebounced } from "../../../hooks/useDebounced";
import { Heatmap } from "../../attention/Heatmap";

const FIELD = "w-full rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";

export function HeadsDemo({ modelId }: { modelId: string | null }) {
  const [prompt, setPrompt] = useState("ROMEO:\nBut");
  const config = useModelConfig(modelId);
  const query = useAttention(modelId, useDebounced(prompt, 400), 0);
  const kw = config.data?.model_kwargs ?? {};
  const nEmbd = typeof kw["n_embd"] === "number" ? kw["n_embd"] : null;
  const nHead = typeof kw["n_head"] === "number" ? kw["n_head"] : null;
  const headSize = config.data?.head_size ?? null;
  const data = query.data;

  return (
    <div className="space-y-3">
      {nEmbd !== null && nHead !== null && headSize !== null && (
        <p className="font-mono text-sm text-slate-200">
          n_embd {nEmbd} ÷ {nHead} heads = head size {headSize}; {nHead} × {headSize} = {nEmbd}
        </p>
      )}
      <label htmlFor="learn-heads" className="block text-sm text-slate-400">Prompt (up to 24 characters)</label>
      <input id="learn-heads" className={FIELD} maxLength={24} value={prompt}
        onChange={(e) => setPrompt(e.target.value)} />
      {config.error && <ErrorState error={config.error} />}
      {query.error && <ErrorState error={query.error} />}
      {data && (
        <>
          <div className="flex flex-wrap gap-4">
            {data.weights.map((w, h) => (
              <div key={h}>
                <div className="mb-1 font-mono text-xs text-slate-400">Head {h}</div>
                <Heatmap tokens={data.tokens} matrix={w} cell={data.tokens.length <= 12 ? 16 : 10}
                  label={`Head ${h} attention`} />
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-500">
            Layer 0 of the selected checkpoint: the same prompt, one real attention pattern per head.
            The patterns differ, but one prompt cannot say what any head is for.
          </p>
        </>
      )}
    </div>
  );
}