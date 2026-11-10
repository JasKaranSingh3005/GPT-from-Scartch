import clsx from "clsx";

import { Card } from "../components/ui/Card";
import { MetricCard } from "../components/ui/MetricCard";
import { Skeleton } from "../components/ui/Skeleton";
import { EmptyState, ErrorState } from "../components/ui/states";
import {
  useLoadModel,
  useModelConfig,
  useModelDetail,
  useModelParameters,
  useModels,
  useSystemInfo,
  useUnloadModel,
} from "../hooks/useApi";
import { useModelStore } from "../stores/modelStore";
import type { ModelSummary } from "../types/api";
import { DASH, formatBytes, formatDuration, formatInt, formatLoss } from "../utils/format";

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 py-1.5 text-sm">
      <dt className="text-slate-400">{label}</dt>
      <dd className="text-right font-mono text-slate-100">{value}</dd>
    </div>
  );
}

function SystemCard() {
  const { data, error, isPending } = useSystemInfo();
  return (
    <Card title="System">
      {isPending && <Skeleton className="h-32 w-full" />}
      {error && <ErrorState error={error} />}
      {data && (
        <dl className="divide-y divide-navy-600">
          <Row label="Inference device" value={data.device} />
          <Row label="CUDA" value={data.cuda_available ? `Available (${data.cuda_version ?? DASH})` : "Not available"} />
          <Row label="PyTorch" value={data.torch_version} />
          <Row label="Python" value={data.python_version} />
          <Row label="CPU cores" value={formatInt(data.cpu_count)} />
          {data.gpus.map((g) => (
            <Row
              key={g.index}
              label={`GPU ${g.index}`}
              value={`${g.name} (${formatBytes(g.total_memory_bytes)})`}
            />
          ))}
          <Row label="Models loaded" value={`${data.models_loaded} / ${data.limits.max_loaded_models}`} />
          <Row label="Max new tokens" value={formatInt(data.limits.max_new_tokens)} />
        </dl>
      )}
    </Card>
  );
}

function ModelTable({ models }: { models: ModelSummary[] }) {
  const selected = useModelStore((s) => s.selectedModelId);
  const select = useModelStore((s) => s.select);
  const load = useLoadModel();
  const unload = useUnloadModel();
  const busy = load.isPending || unload.isPending;
  const mutationError = load.error ?? unload.error;

  return (
    <>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs font-medium text-slate-500">
            <tr>
              <th className="py-2 pr-4 font-medium">Model</th>
              <th className="py-2 pr-4 font-medium">Size</th>
              <th className="py-2 pr-4 font-medium">Status</th>
              <th className="py-2 font-medium" />
            </tr>
          </thead>
          <tbody className="divide-y divide-navy-600">
            {models.map((m) => {
              const isSelected = m.model_id === selected;
              return (
                <tr key={m.model_id} className={clsx("transition-colors hover:bg-navy-700/50", isSelected && "bg-violet/10")}>
                  <td className="py-2 pr-4 font-mono">{m.model_id}</td>
                  <td className="py-2 pr-4 text-slate-400">{formatBytes(m.size_bytes)}</td>
                  <td className="py-2 pr-4">
                    <span className="inline-flex items-center gap-2">
                      <span
                        aria-hidden
                        className={clsx("h-2 w-2 rounded-full", m.loaded ? "bg-emerald-400" : "bg-slate-600")}
                      />
                      <span className={m.loaded ? "text-slate-100" : "text-slate-400"}>
                        {m.loaded ? "Loaded" : "Not loaded"}
                      </span>
                    </span>
                  </td>
                  <td className="space-x-2 py-2 text-right whitespace-nowrap">
                    <button
                      type="button"
                      aria-pressed={isSelected}
                      className={clsx(
                        "rounded-md border px-2.5 py-1 text-xs font-medium transition-colors focus-visible:outline-none",
                        isSelected
                          ? "border-violet-soft text-violet-soft"
                          : "border-navy-600 text-slate-200 hover:bg-navy-700",
                      )}
                      onClick={() => select(m.model_id)}
                    >
                      {isSelected ? "Inspecting" : "Inspect"}
                    </button>
                    {m.loaded ? (
                      <button
                        type="button"
                        disabled={busy}
                        className="rounded-md border border-navy-600 px-2.5 py-1 text-xs font-medium text-slate-200 transition-colors hover:bg-navy-700 disabled:cursor-not-allowed disabled:opacity-50"
                        onClick={() => unload.mutate(m.model_id)}
                      >
                        Unload
                      </button>
                    ) : (
                      <button
                        type="button"
                        disabled={busy}
                        className="rounded-md bg-violet px-2.5 py-1 text-xs font-medium text-white transition-colors hover:bg-violet-strong disabled:cursor-not-allowed disabled:opacity-50"
                        onClick={() => load.mutate(m.model_id)}
                      >
                        Load
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {mutationError && (
        <div className="mt-3">
          <ErrorState error={mutationError} />
        </div>
      )}
    </>
  );
}

function CheckpointsCard() {
  const { data, error, isPending } = useModels();
  return (
    <Card title="Checkpoints">
      {isPending && <Skeleton className="h-24 w-full" />}
      {error && <ErrorState error={error} />}
      {data && data.models.length === 0 && (
        <EmptyState title="No checkpoints found">
          Train a model first, for example <code className="font-mono">python train.py --name gpt_small</code>.
        </EmptyState>
      )}
      {data && data.models.length > 0 && <ModelTable models={data.models} />}
    </Card>
  );
}

function InspectedModel() {
  const id = useModelStore((s) => s.selectedModelId);
  const detail = useModelDetail(id);
  const config = useModelConfig(id);
  const params = useModelParameters(id);

  if (id === null) {
    return (
      <EmptyState title="No model selected">
        Choose &ldquo;Inspect&rdquo; on a checkpoint to see its configuration.
      </EmptyState>
    );
  }
  const failure = detail.error ?? config.error ?? params.error;
  if (failure) return <ErrorState error={failure} />;

  const loading = detail.isPending || config.isPending || params.isPending;
  const kw = config.data?.model_kwargs ?? {};
  const num = (key: string): string => (typeof kw[key] === "number" ? formatInt(kw[key] as number) : "n/a");

  return (
    <div className="space-y-3">
      <div className="text-sm text-slate-400">
        Showing <span className="font-mono text-slate-100">{id}</span>
        {detail.data && ` (${detail.data.model_type}, iteration ${formatInt(detail.data.iteration)})`}
      </div>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricCard label="Parameters" loading={loading} value={formatInt(params.data?.total)}
          hint={params.data?.breakdown_matches_actual === false ? "breakdown mismatch" : undefined} />
        <MetricCard label="Layers" loading={loading} value={num("n_layer")} />
        <MetricCard label="Attention heads" loading={loading} value={num("n_head")} />
        <MetricCard label="Embedding dim" loading={loading} value={num("n_embd")} />
        <MetricCard label="Context length" loading={loading} value={formatInt(config.data?.context_length)} />
        <MetricCard label="Vocabulary" loading={loading} value={formatInt(detail.data?.vocab_size)} />
        <MetricCard label="Best val loss" loading={loading} value={formatLoss(detail.data?.run_best_val_loss)}
          hint="from the run's summary.json" />
        <MetricCard label="Training time" loading={loading} value={formatDuration(detail.data?.run_training_time_s)}
          hint="wall-clock, can be noisy" />
      </div>
    </div>
  );
}

export function Dashboard() {
  return (
    <div className="mx-auto max-w-[1500px] space-y-8">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-slate-50">Dashboard</h1>
        <p className="mt-1.5 max-w-xl text-sm text-slate-400">Check your runtime, inspect checkpoints and see how each model was trained.</p>
      </div>
      <div className="grid gap-6 xl:grid-cols-2">
        <SystemCard />
        <CheckpointsCard />
      </div>
      <Card title="Model details">
        <InspectedModel />
      </Card>
    </div>
  );
}
