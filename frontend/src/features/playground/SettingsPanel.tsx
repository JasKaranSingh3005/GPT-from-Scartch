import { Card } from "../../components/ui/Card";
import { useModelDetail, useModels, useSystemInfo } from "../../hooks/useApi";
import { usePlaygroundStore } from "../../stores/playgroundStore";
import { useState } from "react";

const FIELD = "w-full rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";

export function parseOptionalInt(value: string): number | null {
  const n = parseInt(value, 10);
  return Number.isNaN(n) ? null : n;
}

export function SettingsPanel() {
  const settings = usePlaygroundStore((s) => s.settings);
  const update = usePlaygroundStore((s) => s.update);
  const models = useModels();
  const system = useSystemInfo();
  const detail = useModelDetail(settings.modelId);
  const limit = system.data?.limits.max_new_tokens;
  const vocab = detail.data?.vocab_size;
  const [maxText, setMaxText] = useState(String(settings.maxNewTokens));

  return (
    <Card title="Settings">
      <div className="space-y-4 text-sm">
        <div>
          <label htmlFor="pg-model" className="mb-1 block text-slate-400">Model / checkpoint</label>
          <select
            id="pg-model"
            className={FIELD}
            value={settings.modelId ?? ""}
            onChange={(e) => update({ modelId: e.target.value || null })}
          >
            {settings.modelId === null && <option value="">Select a model</option>}
            {models.data?.models.map((m) => (
              <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
            ))}
          </select>
          <p className="mt-1 text-xs text-slate-500">Loaded automatically on first use.</p>
        </div>

        <div>
          <label htmlFor="pg-temp" className="mb-1 flex justify-between text-slate-400">
            <span>Temperature</span>
            <span className="font-mono text-slate-100">{settings.temperature.toFixed(2)}</span>
          </label>
          <input
            id="pg-temp" type="range" min={0.1} max={2} step={0.05}
            className="w-full accent-violet"
            value={settings.temperature}
            onChange={(e) => update({ temperature: Number(e.target.value) })}
          />
        </div>

        <div>
          <label htmlFor="pg-topk" className="mb-1 block text-slate-400">
            Top-k {vocab ? `(1 to ${vocab}, empty = off)` : "(empty = off)"}
          </label>
          <input
            id="pg-topk" type="number" min={1} max={vocab} placeholder="off" className={FIELD}
            value={settings.topK ?? ""}
            onChange={(e) => update({ topK: parseOptionalInt(e.target.value) })}
          />
        </div>

        <div>
          <label htmlFor="pg-max" className="mb-1 block text-slate-400">
            Max new tokens {limit ? `(server limit ${limit})` : ""}
          </label>
          <input
            id="pg-max" type="number" min={1} max={limit} className={FIELD}
            value={maxText}
            onChange={(e) => {
              setMaxText(e.target.value);
              const n = parseOptionalInt(e.target.value);
              if (n !== null && n >= 1) update({ maxNewTokens: n });
            }}
          />
        </div>

        <div>
          <label htmlFor="pg-seed" className="mb-1 block text-slate-400">Seed (empty = random)</label>
          <input
            id="pg-seed" type="number" min={0} placeholder="random" className={FIELD}
            value={settings.seed ?? ""}
            onChange={(e) => update({ seed: parseOptionalInt(e.target.value) })}
          />
        </div>
      </div>
    </Card>
  );
}