import { useEffect, useState } from "react";

import { Card } from "../components/ui/Card";
import { ErrorState } from "../components/ui/states";
import { useCompare, useModelConfig, useModelDetail, useModels } from "../hooks/useApi";
import { DASH, formatDuration, formatInt, formatLoss } from "../utils/format";

const FIELD = "rounded border border-field bg-navy-900 px-2 py-1.5 text-sm font-mono";

function FactsRow({ id, speed }: { id: string; speed: number | undefined }) {
  const detail = useModelDetail(id);
  const config = useModelConfig(id);
  const d = detail.data;
  if (detail.error) {
    return <tr><td colSpan={8}><ErrorState error={detail.error} /></td></tr>;
  }
  return (
    <tr>
      <td className="pr-3">{id}</td>
      <td className="pr-3">{d?.model_type ?? DASH}</td>
      <td className="pr-3">{formatInt(d?.parameters)}</td>
      <td className="pr-3">{formatInt(config.data?.context_length)}</td>
      <td className="pr-3">{formatInt(d?.iteration)}</td>
      <td className="pr-3">{formatLoss(d?.run_best_val_loss)}</td>
      <td className="pr-3">{formatDuration(d?.run_training_time_s)}</td>
      <td>{speed === undefined ? DASH : `${speed.toFixed(1)} tok/s`}</td>
    </tr>
  );
}

export function ModelComparison() {
  const models = useModels();
  const compare = useCompare();
  const [ids, setIds] = useState<string[]>([]);
  const [prompt, setPrompt] = useState("ROMEO:");
  const [seed, setSeed] = useState(42);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState<string>("40");
  const [maxTokens, setMaxTokens] = useState(200);

  useEffect(() => {
    const all = models.data?.models.map((m) => m.model_id) ?? [];
    if (ids.length === 0 && all.length >= 2) {
      const preferred = ["bigram_baseline:best", "gpt_small:best"].filter((p) => all.includes(p));
      setIds(preferred.length === 2 ? preferred : all.slice(0, 2));
    }
  }, [models.data, ids.length]);

  const toggle = (id: string) =>
    setIds((cur) => (cur.includes(id) ? cur.filter((x) => x !== id) : cur.length < 4 ? [...cur, id] : cur));

  const run = () => {
    const first = ids[0];
    if (first === undefined) return;
    const k = parseInt(topK, 10);
    compare.mutate({
      model_id: first, prompt, max_new_tokens: maxTokens, temperature,
      top_k: Number.isNaN(k) ? null : k, seed,
      variants: ids.map((id) => ({ label: id, model_id: id, temperature: null, top_k: null })),
    });
  };

  const results = compare.data?.results ?? [];
  const speedOf = (id: string) => results.find((r) => r.label === id)?.result.tokens_per_second;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-slate-50">Model Comparison</h1>
      <p className="max-w-3xl text-sm text-slate-400">
        Compare 2 to 4 checkpoints. Every model gets the same prompt, settings and seed, so
        differences in the text come from the models.
      </p>

      <Card title="Models">
        {models.error && <ErrorState error={models.error} />}
        <div className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
          {models.data?.models.map((m) => (
            <label key={m.model_id} className="flex items-center gap-1.5 font-mono text-slate-300">
              <input type="checkbox" checked={ids.includes(m.model_id)} onChange={() => toggle(m.model_id)} />
              {m.model_id}
            </label>
          ))}
        </div>
        <div className="mt-3 grid gap-3 md:grid-cols-5 text-sm">
          <label className="text-slate-400 md:col-span-5">Prompt
            <input className={`${FIELD} mt-1 block w-full`} value={prompt} maxLength={4000}
              onChange={(e) => setPrompt(e.target.value)} />
          </label>
          <label className="text-slate-400">Temperature
            <input type="number" step={0.05} min={0.05} max={5} className={`${FIELD} mt-1 block w-full`}
              value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
          </label>
          <label className="text-slate-400">Top-k (empty = off)
            <input type="number" min={1} className={`${FIELD} mt-1 block w-full`} value={topK}
              onChange={(e) => setTopK(e.target.value)} />
          </label>
          <label className="text-slate-400">New tokens
            <input type="number" min={1} className={`${FIELD} mt-1 block w-full`} value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))} />
          </label>
          <label className="text-slate-400">Seed
            <input type="number" min={0} className={`${FIELD} mt-1 block w-full`} value={seed}
              onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
        </div>
        <button
          className="mt-3 rounded bg-violet px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-strong disabled:opacity-50"
          disabled={ids.length < 2 || compare.isPending} onClick={run}>
          {compare.isPending ? "Generating..." : "Compare"}
        </button>
        {ids.length < 2 && <span role="status" className="ml-3 text-xs text-amber-300">Select at least two models.</span>}
        {compare.error && <div className="mt-2"><ErrorState error={compare.error} /></div>}
      </Card>

      {ids.length >= 2 && (
        <Card title="Facts">
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs" aria-label="Model facts">
              <thead className="text-slate-500">
                <tr>
                  {["Model", "Type", "Parameters", "Context", "Iteration", "Run best val", "Run train time", "Generation speed"].map((h) => (
                    <th key={h} className="pr-3 pb-1">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>{ids.map((id) => <FactsRow key={id} id={id} speed={speedOf(id)} />)}</tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Parameters, context and iteration come from the checkpoint. Validation loss and
            training time belong to the whole run, so the best and last checkpoints of one run
            show the same values. Speed is one measurement from the last comparison; the first
            model to run can include GPU warm-up, and for models this small per-token overhead
            dominates, so do not read it as a benchmark.
          </p>
        </Card>
      )}

      {results.length > 0 && (
        <div className="grid gap-4 lg:grid-cols-2">
          {results.map(({ label, result }) => (
            <Card key={label} title={label}>
              <div className="mb-2 text-xs text-slate-500">
                seed {result.seed} | {result.device} | {result.tokens_generated} tokens in {result.seconds.toFixed(2)} s
              </div>
              <pre className="whitespace-pre-wrap break-words rounded border border-navy-600 bg-navy-900 p-3 font-mono text-sm">
                <span className="text-slate-500">{result.prompt}</span>
                <span className="text-slate-100">{result.completion}</span>
              </pre>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}