import { useState } from "react";

import { Card } from "../../components/ui/Card";
import { BUTTON_PRIMARY, BUTTON_SECONDARY, FIELD_BASE } from "../../components/ui/controls";
import { ErrorState } from "../../components/ui/states";
import { useCompare, useModels, useSystemInfo } from "../../hooks/useApi";
import { usePlaygroundStore } from "../../stores/playgroundStore";
import type { CompareRequest } from "../../types/api";
import { parseOptionalInt } from "./SettingsPanel";

interface VariantRow {
  label: string;
  modelId: string; // "" = use the page's model
  temperature: string; // "" = use the page's temperature
  topK: string; // "" = use the page's top-k
}

const LABELS = ["A", "B", "C", "D"];
const FIELD = FIELD_BASE;

export function CompareView() {
  const settings = usePlaygroundStore((s) => s.settings);
  const models = useModels();
  const system = useSystemInfo();
  const compare = useCompare();
  const [seed, setSeed] = useState(42);
  const [rows, setRows] = useState<VariantRow[]>([
    { label: "A", modelId: "", temperature: "", topK: "" },
    { label: "B", modelId: "", temperature: "1.2", topK: "" },
  ]);
  const [lastRequest, setLastRequest] = useState<CompareRequest | null>(null);

  const limit = system.data?.limits.max_new_tokens ?? null;
  const overLimit = limit !== null && settings.maxNewTokens > limit;

  const patch = (i: number, p: Partial<VariantRow>) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...p } : r)));

  const run = () => {
    if (settings.modelId === null) return;
    const request: CompareRequest = {
      model_id: settings.modelId,
      prompt: settings.prompt,
      max_new_tokens: settings.maxNewTokens,
      temperature: settings.temperature,
      top_k: settings.topK,
      seed,
      variants: rows.map((r) => ({
        label: r.label,
        model_id: r.modelId || null,
        temperature: r.temperature === "" ? null : Number(r.temperature),
        top_k: parseOptionalInt(r.topK),
      })),
    };
    setLastRequest(request);
    compare.mutate(request);
  };

  return (
    <div className="space-y-4">
      <Card title="Controlled comparison">
        <p className="mb-3 text-sm text-slate-400">
          Every variant uses the same seed and the page settings (left). Blank fields inherit them.
          Two variants with identical settings produce identical text, which is the control.
        </p>
        <div className="space-y-2">
          {rows.map((r, i) => (
            <div key={r.label} className="flex flex-wrap items-center gap-2 text-sm">
              <span className="w-6 font-mono text-violet-soft">{r.label}</span>
              <select aria-label={`Model for ${r.label}`} className={FIELD} value={r.modelId}
                onChange={(e) => patch(i, { modelId: e.target.value })}>
                <option value="">same model</option>
                {models.data?.models.map((m) => (
                  <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
                ))}
              </select>
              <input aria-label={`Temperature for ${r.label}`} placeholder="temperature"
                type="number" step={0.05} min={0.05} max={5} className={`${FIELD} w-32`}
                value={r.temperature} onChange={(e) => patch(i, { temperature: e.target.value })} />
              <input aria-label={`Top-k for ${r.label}`} placeholder="top-k" type="number"
                min={1} className={`${FIELD} w-24`} value={r.topK}
                onChange={(e) => patch(i, { topK: e.target.value })} />
              {rows.length > 2 && (
                <button className="text-xs text-slate-400 hover:text-slate-200"
                  onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}>Remove</button>
              )}
            </div>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
          {rows.length < 4 && (
            <button className={BUTTON_SECONDARY}
              onClick={() => setRows((rs) => [...rs, {
                label: LABELS.find((l) => !rs.some((r) => r.label === l)) ?? "X",
                modelId: "", temperature: "", topK: "",
              }])}>
              Add variant
            </button>
          )}
          <label className="flex items-center gap-2 text-slate-400">
            Seed
            <input type="number" min={0} className={`${FIELD} w-28`} value={seed}
              onChange={(e) => setSeed(parseOptionalInt(e.target.value) ?? 0)} />
          </label>
          <button
            className={BUTTON_PRIMARY}
            disabled={settings.modelId === null || compare.isPending || overLimit} onClick={run}>
            {compare.isPending ? "Running..." : "Run comparison"}
          </button>
          {overLimit && <span role="alert" className="text-xs text-amber-300">
            Max new tokens exceeds the server limit ({limit}).</span>}
        </div>
      </Card>

      {compare.error && <ErrorState error={compare.error} />}

      {compare.data && lastRequest && (
        <div className="grid gap-4 lg:grid-cols-2">
          {compare.data.results.map(({ label, result }) => {
            const v = lastRequest.variants.find((x) => x.label === label);
            return (
              <Card key={label} title={`${label}: ${result.model_id}`}>
                <div className="mb-2 text-xs text-slate-500">
                  T {v?.temperature ?? lastRequest.temperature} | top-k {v?.top_k ?? lastRequest.top_k ?? "off"} |
                  seed {result.seed} | {result.device}
                </div>
                <pre className="whitespace-pre-wrap break-words rounded border border-navy-600 bg-navy-900 p-3 font-mono text-sm">
                  <span className="text-slate-500">{result.prompt}</span>
                  <span className="text-slate-100">{result.completion}</span>
                </pre>
                <div className="mt-2 text-xs text-slate-400">
                  {result.tokens_generated} tokens in {result.seconds.toFixed(2)} s
                  ({result.tokens_per_second.toFixed(1)} tok/s)
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}