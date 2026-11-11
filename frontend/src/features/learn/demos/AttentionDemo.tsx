import { useState } from "react";
import { Link } from "react-router-dom";

import { ErrorState } from "../../../components/ui/states";
import { useAttention } from "../../../hooks/useApi";
import { useDebounced } from "../../../hooks/useDebounced";
import { visibleChar } from "../../../utils/format";
import { Heatmap, type HoverInfo } from "../../attention/Heatmap";

const FIELD = "rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";

export function AttentionDemo({ modelId }: { modelId: string | null }) {
  const [prompt, setPrompt] = useState("ROMEO:\nBut");
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState(0);
  const [hover, setHover] = useState<HoverInfo | null>(null);
  const query = useAttention(modelId, useDebounced(prompt, 400), layer);
  const data = query.data;
  const nLayers = data?.n_layers ?? 1;
  const nHeads = data?.n_heads ?? 1;
  const effLayer = Math.min(layer, nLayers - 1);
  const effHead = Math.min(head, nHeads - 1);
  const n = data?.tokens.length ?? 0;
  const tok = (i: number) => visibleChar(data?.tokens[i]?.char ?? "");

  return (
    <div className="space-y-3">
      <label htmlFor="learn-att" className="block text-sm text-slate-400">Prompt (up to 32 characters)</label>
      <textarea id="learn-att" rows={2} maxLength={32} className={`${FIELD} w-full`} value={prompt}
        onChange={(e) => { setPrompt(e.target.value); setHover(null); }} />
      <div className="flex gap-3 text-sm">
        <label className="text-slate-400">Layer
          <select className={`${FIELD} ml-2`} value={effLayer}
            onChange={(e) => { setLayer(Number(e.target.value)); setHover(null); }}>
            {Array.from({ length: nLayers }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
        <label className="text-slate-400">Head
          <select className={`${FIELD} ml-2`} value={effHead}
            onChange={(e) => { setHead(Number(e.target.value)); setHover(null); }}>
            {Array.from({ length: nHeads }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
      </div>
      {query.error && <ErrorState error={query.error} />}
      {data && (
        <>
          <div className="overflow-x-auto">
            <Heatmap tokens={data.tokens} matrix={data.weights[effHead] ?? []} cell={n <= 16 ? 24 : 14}
              label={`Attention weights, layer ${effLayer}, head ${effHead}`} showLabels={n <= 24}
              onHover={setHover} />
          </div>
          <div className="min-h-8 rounded border border-navy-600 bg-navy-900 p-2 font-mono text-xs text-slate-300">
            {hover
              ? `"${tok(hover.query)}" (position ${hover.query}) reads "${tok(hover.key)}" (position ${hover.key}) with weight ${hover.weight.toFixed(4)}`
              : "Hover a cell to see the exact weight."}
          </div>
          <p className="text-xs text-slate-500">
            Real weights from a forward pass of the selected checkpoint. The full tool, with all
            heads side by side, is the <Link className="text-violet-soft underline" to="/attention">Attention Visualizer</Link>.
          </p>
        </>
      )}
    </div>
  );
}