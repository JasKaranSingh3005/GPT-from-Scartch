import clsx from "clsx";
import { Link, Navigate, useParams } from "react-router-dom";

import { Card } from "../components/ui/Card";
import { ErrorState } from "../components/ui/states";
import { ModuleView } from "../features/learn/ModuleView";
import { BACKEND_DEMOS, FIRST_MODULE_ID, MODULES } from "../features/learn/modules";
import { useModelChoice } from "../hooks/useModelChoice";

const FIELD = "rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";

export function LearningCenter() {
  const { moduleId } = useParams();
  const { models, modelId, setModelId } = useModelChoice();
  const current = MODULES.find((m) => m.id === (moduleId ?? FIRST_MODULE_ID));
  if (!current) return <Navigate to="/learn" replace />;

  const writtenCount = MODULES.filter((m) => m.written).length;
  const needsModel = current.demo !== undefined && BACKEND_DEMOS.includes(current.demo);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-slate-50">Learning Center</h1>
      <p className="max-w-3xl text-sm text-slate-400">
        How the model works, one idea at a time. Demos that involve a model call the real backend.
        {" "}{writtenCount} of {MODULES.length} modules are written; the rest are marked and will
        say so if you open them.
      </p>

      <div className="grid gap-4 lg:grid-cols-[240px_minmax(0,1fr)]">
        <nav aria-label="Modules">
          <Card title="Modules">
            <ol className="space-y-0.5">
              {MODULES.map((m, i) => (
                <li key={m.id}>
                  <Link to={`/learn/${m.id}`} aria-current={m.id === current.id ? "page" : undefined}
                    className={clsx(
                      "flex items-baseline gap-2 rounded px-2 py-1 text-[13px]",
                      m.id === current.id ? "bg-violet/15 text-violet-soft" : "text-slate-400 hover:bg-navy-700 hover:text-slate-200",
                    )}>
                    <span className="w-4 shrink-0 text-right font-mono text-[11px] text-slate-500">{i + 1}</span>
                    <span className="flex-1">{m.title}</span>
                    {!m.written && <span className="text-[10px] text-slate-500">soon</span>}
                  </Link>
                </li>
              ))}
            </ol>
          </Card>
        </nav>

        <div className="min-w-0 space-y-4">
          {needsModel && (
            <div className="flex items-center gap-2 text-sm">
              <label htmlFor="learn-model" className="text-slate-400">Checkpoint for the demo</label>
              <select id="learn-model" className={FIELD} value={modelId ?? ""}
                onChange={(e) => setModelId(e.target.value || null)}>
                {models.data?.models.map((m) => (
                  <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
                ))}
              </select>
            </div>
          )}
          {needsModel && models.error && <ErrorState error={models.error} />}
          <ModuleView module={current} modelId={modelId} />
        </div>
      </div>
    </div>
  );
}