import { useMemo, useState } from "react";

import { Card } from "../components/ui/Card";
import { Skeleton } from "../components/ui/Skeleton";
import { ErrorState } from "../components/ui/states";
import { Heatmap, type HoverInfo } from "../features/attention/Heatmap";
import { useAttention } from "../hooks/useApi";
import { useModelChoice } from "../hooks/useModelChoice";
import { visibleChar } from "../utils/format";
import { averageHeads } from "../utils/tokens";
import { BUTTON_PRIMARY, FIELD_BASE } from "../components/ui/controls";

const DEFAULT_PROMPT = "ROMEO:\nBut soft, what";
const FIELD = `${FIELD_BASE} w-full`;

const mainCell = (n: number) => (n <= 16 ? 28 : n <= 32 ? 16 : n <= 48 ? 11 : 8);
const smallCell = (n: number) => (n <= 16 ? 12 : n <= 32 ? 7 : 4);

export function AttentionVisualizer() {
  const { models, modelId, setModelId } = useModelChoice();
  const [draft, setDraft] = useState(DEFAULT_PROMPT);
  const [submitted, setSubmitted] = useState(DEFAULT_PROMPT);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState<number | "avg">(0);
  const [hover, setHover] = useState<HoverInfo | null>(null);

  const query = useAttention(modelId, submitted, layer);
  const data = query.data;
  const nHeads = data?.n_heads ?? 1;
  const nLayers = data?.n_layers ?? 1;
  const effLayer = Math.min(layer, nLayers - 1);
  const effHead = head === "avg" || head < nHeads ? head : 0;

  const matrix = useMemo(() => {
    if (!data) return [];
    return effHead === "avg" ? averageHeads(data.weights) : (data.weights[effHead] ?? []);
  }, [data, effHead]);

  const n = data?.tokens.length ?? 0;
  const headLabel = effHead === "avg" ? "average of heads" : `head ${effHead}`;
  const tok = (i: number) => data?.tokens[i];

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-slate-50">Attention Visualizer</h1>
      <p className="max-w-3xl text-sm text-slate-400">
        Real attention weights from a forward pass of the selected model. Each row is a position
        choosing how much to look at each earlier position (and itself). The dark upper triangle is
        the causal mask: future positions are never attended to.
      </p>

      <Card title="Controls">
        <div className="grid gap-3 md:grid-cols-3">
          <div>
            <label htmlFor="av-model" className="mb-1 block text-sm text-slate-400">Model</label>
            <select id="av-model" className={FIELD} value={modelId ?? ""}
              onChange={(e) => { setModelId(e.target.value || null); setHover(null); }}>
              {models.data?.models.map((m) => (
                <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="av-layer" className="mb-1 block text-sm text-slate-400">Layer</label>
            <select id="av-layer" className={FIELD} value={effLayer}
              onChange={(e) => { setLayer(Number(e.target.value)); setHover(null); }}>
              {Array.from({ length: nLayers }, (_, i) => (
                <option key={i} value={i}>Layer {i}</option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="av-head" className="mb-1 block text-sm text-slate-400">Head</label>
            <select id="av-head" className={FIELD} value={String(effHead)}
              onChange={(e) => { setHead(e.target.value === "avg" ? "avg" : Number(e.target.value)); setHover(null); }}>
              {Array.from({ length: nHeads }, (_, i) => (
                <option key={i} value={i}>Head {i}</option>
              ))}
              <option value="avg">Average of heads (computed here)</option>
            </select>
          </div>
        </div>
        <label htmlFor="av-prompt" className="mb-1 mt-3 block text-sm text-slate-400">
          Prompt (the last {data?.block_size ?? "block_size"} characters are used, at most 64 shown)
        </label>
        <textarea id="av-prompt" rows={3} maxLength={4000} className={FIELD} value={draft}
          onChange={(e) => setDraft(e.target.value)} />
        <button className={`mt-3 ${BUTTON_PRIMARY}`}
          disabled={modelId === null || draft.length === 0}
          onClick={() => { setSubmitted(draft); setHover(null); }}>
          Visualize
        </button>
      </Card>

      {query.error && <ErrorState error={query.error} />}
      {query.isPending && query.fetchStatus !== "idle" && <Skeleton className="h-64 w-full" />}

      {data && (
        <>
          {data.cropped && (
            <p role="status" className="text-sm text-amber-300">
              The prompt was longer than the window, so only its last {n} characters are shown.
            </p>
          )}
          <Card title={`Layer ${data.layer}, ${headLabel}`}>
            <p className="mb-3 text-xs text-slate-500">
              Layer {data.layer} of {data.n_layers} | {data.n_heads} heads | {n} tokens | weights
              rounded to 5 decimals, square-root color scale (small weights look brighter than they are; hover for exact values) 
            </p>
            <div className="overflow-x-auto">
              <Heatmap tokens={data.tokens} matrix={matrix} cell={mainCell(n)}
                label={`Attention heatmap, ${headLabel}`} showLabels={n <= 40}
                onHover={setHover} />
            </div>
            <div data-testid="readout" className="mt-3 min-h-10 rounded border border-navy-600 bg-navy-900 p-2 font-mono text-xs text-slate-300">
              {hover && tok(hover.query) && tok(hover.key) ? (
                <>
                  query &ldquo;{visibleChar(tok(hover.query)?.char ?? "")}&rdquo; (position {hover.query})
                  {" \u2192 "}key &ldquo;{visibleChar(tok(hover.key)?.char ?? "")}&rdquo; (position {hover.key}):
                  weight {hover.weight.toFixed(4)}
                </>
              ) : (
                "Hover a cell to see the exact weight."
              )}
            </div>
          </Card>

          <Card title="Compare heads in this layer">
            <div className="flex flex-wrap gap-4">
              {data.weights.map((w, h) => (
                <button key={h} onClick={() => setHead(h)} aria-label={`Select head ${h}`}
                  className={`rounded border p-2 text-left ${effHead === h ? "border-violet" : "border-navy-600 hover:border-slate-500"}`}>
                  <div className="mb-1 font-mono text-xs text-slate-400">Head {h}</div>
                  <Heatmap tokens={data.tokens} matrix={w} cell={smallCell(n)}
                    label={`Head ${h} overview`} />
                </button>
              ))}
            </div>
          </Card>

          <Card title="How to read this">
            <ul className="list-disc space-y-1 pl-5 text-sm text-slate-400">
              <li>Every row sums to 1, since it is a softmax. Position 0 can only see itself, so its weight is always 1.</li>
              <li>Different heads in one layer can learn different patterns. What a head &ldquo;means&rdquo; cannot be read off a single prompt.</li>
              <li>Try the same prompt in layer 0 and in the last layer, and in a barely trained checkpoint against a fully trained one.</li>
            </ul>
          </Card>
        </>
      )}
    </div>
  );
}