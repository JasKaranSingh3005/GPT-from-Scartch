import { useState } from "react";

import { Card } from "../../components/ui/Card";
import { ErrorState } from "../../components/ui/states";
import { useCreateJob } from "../../hooks/useApi";
import type { TrainingJobRequest } from "../../types/api";

const FIELD = "w-full rounded border border-field bg-navy-900 px-2 py-1.5 text-sm font-mono";

const DEFAULTS: TrainingJobRequest = {
  name: "", model_type: "gpt", n_embd: 64, n_head: 4, n_layer: 4, block_size: 64, dropout: 0.1,
  batch_size: 32, learning_rate: 0.001, max_iters: 1000, eval_interval: 100, eval_iters: 50, seed: 1337,
};

type NumKey = Exclude<keyof TrainingJobRequest, "name" | "model_type">;
const NUM_FIELDS: [NumKey, string][] = [
  ["n_embd", "Embedding dim"], ["n_head", "Attention heads"], ["n_layer", "Layers"],
  ["block_size", "Context length"], ["dropout", "Dropout"], ["batch_size", "Batch size"],
  ["learning_rate", "Learning rate"], ["max_iters", "Max iterations"],
  ["eval_interval", "Eval interval"], ["eval_iters", "Eval batches"], ["seed", "Seed"],
];

/** Client-side check mirroring the server's rules; the server stays the authority. */
export function validateForm(f: TrainingJobRequest): string | null {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(f.name)) {
    return "Name: 1-64 letters, digits, '-' or '_', starting with a letter or digit";
  }
  if (f.model_type === "gpt" && f.n_embd % f.n_head !== 0) {
    return `Embedding dim (${f.n_embd}) must be divisible by heads (${f.n_head})`;
  }
  if (f.eval_interval > f.max_iters) return "Eval interval cannot exceed max iterations";
  return null;
}

export function LaunchForm({ onCreated }: { onCreated: (jobId: string) => void }) {
  const [form, setForm] = useState<TrainingJobRequest>(DEFAULTS);
  const create = useCreateJob();
  const problem = validateForm(form);
  const isGpt = form.model_type === "gpt";

  return (
    <Card title="New training run">
      <div className="grid gap-3 md:grid-cols-3">
        <div>
          <label htmlFor="tl-name" className="mb-1 block text-sm text-slate-400">Run name</label>
          <input id="tl-name" className={FIELD} value={form.name} placeholder="my_run"
            onChange={(e) => setForm({ ...form, name: e.target.value })} />
        </div>
        <div>
          <label htmlFor="tl-type" className="mb-1 block text-sm text-slate-400">Architecture</label>
          <select id="tl-type" className={FIELD} value={form.model_type}
            onChange={(e) => setForm({ ...form, model_type: e.target.value as "gpt" | "bigram" })}>
            <option value="gpt">GPT</option>
            <option value="bigram">Bigram baseline</option>
          </select>
        </div>
        {NUM_FIELDS.filter(([k]) => isGpt || !["n_embd", "n_head", "n_layer", "dropout"].includes(k))
          .map(([key, label]) => (
            <div key={key}>
              <label htmlFor={`tl-${key}`} className="mb-1 block text-sm text-slate-400">{label}</label>
              <input id={`tl-${key}`} type="number" className={FIELD} value={form[key]}
                step={key === "learning_rate" || key === "dropout" ? "any" : 1}
                onChange={(e) => setForm({ ...form, [key]: Number(e.target.value) })} />
            </div>
          ))}
      </div>
      {!isGpt && (
        <p className="mt-2 text-xs text-slate-500">
          The bigram ignores width, heads, layers and dropout. Use a learning rate near 0.01.
        </p>
      )}
      {problem && form.name !== "" && <p role="alert" className="mt-2 text-xs text-amber-300">{problem}</p>}
      {create.error && <div className="mt-2"><ErrorState error={create.error} /></div>}
      <button
        className="mt-3 rounded bg-violet px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-strong disabled:opacity-50"
        disabled={problem !== null || create.isPending}
        onClick={() => create.mutate(form, { onSuccess: (job) => onCreated(job.job_id) })}>
        {create.isPending ? "Queuing..." : "Start training"}
      </button>
    </Card>
  );
}