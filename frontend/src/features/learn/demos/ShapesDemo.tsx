import { useState } from "react";

import { ErrorState } from "../../../components/ui/states";
import { useForwardPass } from "../../../hooks/useApi";
import { formatShape } from "../../../utils/architecture";

const FIELD = "w-full rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";

function wanted(name: string, focus?: string[]): boolean {
  if (!focus) return true;
  return focus.some((f) => (f.endsWith(".") ? name.startsWith(f) : name === f));
}

export function ShapesDemo({ modelId, focus }: { modelId: string | null; focus?: string[] }) {
  const [prompt, setPrompt] = useState("ROMEO:");
  const run = useForwardPass();
  const data = run.data && run.data.model_id === modelId ? run.data : null;
  const rows = data ? data.stages.filter((s) => wanted(s.name, focus)) : [];

  return (
    <div className="space-y-3">
      <label htmlFor="learn-shapes" className="block text-sm text-slate-400">Prompt</label>
      <div className="flex flex-wrap gap-2">
        <input id="learn-shapes" className={`${FIELD} max-w-md`} maxLength={64} value={prompt}
          onChange={(e) => setPrompt(e.target.value)} />
        <button
          className="rounded bg-violet px-3 py-1.5 text-sm font-medium text-white hover:bg-violet-strong disabled:opacity-50"
          disabled={modelId === null || prompt.length === 0 || run.isPending}
          onClick={() => modelId !== null && run.mutate({ modelId, prompt })}>
          {run.isPending ? "Running..." : "Run forward pass"}
        </button>
      </div>
      {run.error && <ErrorState error={run.error} />}
      {data && (
        <>
          <table className="w-full text-left font-mono text-xs" aria-label="Forward pass statistics">
            <thead className="text-slate-500">
              <tr><th className="pr-3">Module</th><th className="pr-3">Output shape</th>
                <th className="pr-3">Mean</th><th>Std</th></tr>
            </thead>
            <tbody>
              {rows.map((s) => (
                <tr key={s.name}>
                  <td className="pr-3">{s.name}</td>
                  <td className="pr-3">{formatShape(s.shape)}</td>
                  <td className="pr-3">{s.mean.toFixed(3)}</td>
                  <td>{s.std.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-slate-500">
            Measured on one forward pass of {data.tokens.length} characters
            {data.cropped ? " (your prompt was cropped to the model's window)" : ""}. These describe
            this prompt, not the model in general.
          </p>
        </>
      )}
      {!data && !run.error && (
        <p className="text-xs text-slate-500">Nothing is shown until you run a pass, so every number is measured.</p>
      )}
    </div>
  );
}