import { useState } from "react";

import { ErrorState } from "../../../components/ui/states";
import { useDistribution } from "../../../hooks/useApi";
import { useDebounced } from "../../../hooks/useDebounced";
import { visibleChar } from "../../../utils/format";

const FIELD = "rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";

export function TemperatureDemo({ modelId }: { modelId: string | null }) {
  const [prompt, setPrompt] = useState("ROMEO:");
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState("");
  const parsed = parseInt(topK, 10);
  const k = Number.isNaN(parsed) || parsed < 1 ? null : parsed;
  const query = useDistribution(
    modelId, useDebounced(prompt, 300), useDebounced(temperature, 200), useDebounced(k, 300),
  );
  const data = query.data;

  return (
    <div className="space-y-3">
      <label htmlFor="learn-temp-prompt" className="block text-sm text-slate-400">Prompt</label>
      <textarea id="learn-temp-prompt" rows={2} maxLength={500} className={`${FIELD} w-full`} value={prompt}
        onChange={(e) => setPrompt(e.target.value)} />
      <div className="flex flex-wrap items-end gap-4 text-sm">
        <label className="text-slate-400">Temperature {temperature.toFixed(2)}
          <input type="range" min={0.1} max={2} step={0.05} className="mt-1 block w-48 accent-violet"
            value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
        </label>
        <label className="text-slate-400">Top-k (empty = off)
          <input type="number" min={1} className={`${FIELD} mt-1 block w-24`} value={topK}
            onChange={(e) => setTopK(e.target.value)} />
        </label>
      </div>
      {query.error && <ErrorState error={query.error} />}
      {data && (
        <>
          <ul aria-label="Next-character probabilities" className="space-y-1">
            {data.entries.map((e) => (
              <li key={e.id} className="flex items-center gap-2 font-mono text-xs">
                <span className="w-6 text-right text-slate-200">{visibleChar(e.char)}</span>
                <div className="h-3 flex-1 rounded bg-navy-700">
                  <div className="h-full rounded bg-violet" style={{ width: `${e.probability * 100}%` }} />
                </div>
                <span className="w-14 text-right text-slate-300">{(e.probability * 100).toFixed(1)}%</span>
              </li>
            ))}
          </ul>
          <p className="text-xs text-slate-500">
            The {data.entries.length} most likely characters cover {(data.shown_mass * 100).toFixed(1)}% of
            the probability. The model saw {data.prompt_tokens} characters
            {data.cropped ? " (the end of your prompt, cropped to its context window)" : ""}. Real
            probabilities from the selected checkpoint, after temperature and top-k.
          </p>
        </>
      )}
    </div>
  );
}