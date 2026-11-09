import { useState } from "react";

import { Card } from "../components/ui/Card";
import { Skeleton } from "../components/ui/Skeleton";
import { ErrorState } from "../components/ui/states";
import { Diagram } from "../features/architecture/Diagram";
import { StageDetail } from "../features/architecture/StageDetail";
import { useArchitecture, useForwardPass } from "../hooks/useApi";
import { useModelChoice } from "../hooks/useModelChoice";
import { formatInt } from "../utils/format";

const FIELD = "w-full rounded border border-field bg-navy-900 px-2 py-1.5 text-sm font-mono";

export function ArchitectureExplorer() {
  const { models, modelId, setModelId } = useModelChoice();
  const architecture = useArchitecture(modelId);
  const forward = useForwardPass();
  const [selected, setSelected] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("ROMEO:");

  const arch = architecture.data;
  const stage = arch?.stages.find((s) => s.key === selected) ?? null;
  // Ignore a result that belongs to a previously selected model.
  const stats = forward.data && forward.data.model_id === modelId ? forward.data.stages : null;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-slate-50">Architecture Explorer</h1>
      <p className="max-w-3xl text-sm text-slate-400">
        The selected checkpoint&apos;s real structure. Click a stage for its shapes, parameter count
        and the math it performs.
      </p>

      <Card title="Model">
        <div className="grid gap-3 md:grid-cols-[minmax(0,320px)_1fr]">
          <div>
            <label htmlFor="ae-model" className="mb-1 block text-sm text-slate-400">Checkpoint</label>
            <select id="ae-model" className={FIELD} value={modelId ?? ""}
              onChange={(e) => { setModelId(e.target.value || null); setSelected(null); }}>
              {models.data?.models.map((m) => (
                <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
              ))}
            </select>
          </div>
          {arch && (
            <dl className="flex flex-wrap items-end gap-x-6 gap-y-1 font-mono text-xs text-slate-300">
              {Object.entries(arch.config).map(([k, v]) => (
                <div key={k}><span className="text-slate-500">{k}</span> {String(v)}</div>
              ))}
              <div><span className="text-slate-500">parameters</span> {formatInt(arch.parameters_total)}</div>
            </dl>
          )}
        </div>
        {arch && !arch.stage_sum_matches_total && (
          <p role="alert" className="mt-2 text-xs text-amber-300">
            The stage parameter counts do not add up to the checkpoint total.
          </p>
        )}
      </Card>

      {models.error && <ErrorState error={models.error} />}
      {architecture.error && <ErrorState error={architecture.error} />}
      {architecture.isPending && architecture.fetchStatus !== "idle" && <Skeleton className="h-64 w-full" />}

      {arch && (
        <>
          <Card title="Run a forward pass (optional)">
            <div className="flex flex-wrap items-start gap-2">
              <label htmlFor="ae-prompt" className="sr-only">Prompt</label>
              <input id="ae-prompt" className={`${FIELD} max-w-md`} value={prompt} maxLength={4000}
                onChange={(e) => setPrompt(e.target.value)} />
              <button
                className="rounded bg-violet px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-strong disabled:opacity-50"
                disabled={modelId === null || prompt.length === 0 || forward.isPending}
                onClick={() => modelId !== null && forward.mutate({ modelId, prompt })}>
                {forward.isPending ? "Running..." : "Run forward pass"}
              </button>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Records each module&apos;s real output shape and statistics for this prompt.
            </p>
            {forward.error && <div className="mt-2"><ErrorState error={forward.error} /></div>}
          </Card>

          <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
            <Diagram stages={arch.stages} selectedKey={selected} onSelect={setSelected} />
            {stage ? (
              <StageDetail stage={stage} architecture={arch} stats={stats} />
            ) : (
              <Card title="Details">
                <p className="text-sm text-slate-400">Select a stage in the diagram.</p>
              </Card>
            )}
          </div>
        </>
      )}
    </div>
  );
}