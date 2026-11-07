import clsx from "clsx";
import { useEffect, useState } from "react";

import { CompareView } from "../features/playground/CompareView";
import { HistoryPanel } from "../features/playground/HistoryPanel";
import { SettingsPanel } from "../features/playground/SettingsPanel";
import { SingleView } from "../features/playground/SingleView";
import { useModels } from "../hooks/useApi";
import { usePlaygroundStore } from "../stores/playgroundStore";

const PREFERRED_MODEL = "gpt_small:best";

export function Playground() {
  const [mode, setMode] = useState<"single" | "compare">("single");
  const models = useModels();
  const modelId = usePlaygroundStore((s) => s.settings.modelId);
  const update = usePlaygroundStore((s) => s.update);

  useEffect(() => {
    const ids = models.data?.models.map((m) => m.model_id) ?? [];
    if (modelId === null && ids.length > 0) {
      update({ modelId: ids.includes(PREFERRED_MODEL) ? PREFERRED_MODEL : (ids[0] ?? null) });
    }
  }, [models.data, modelId, update]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold text-slate-50">GPT Playground</h1>
        <div role="tablist" className="flex rounded-md border border-navy-600 text-sm">
          {(["single", "compare"] as const).map((m) => (
            <button
              key={m} role="tab" aria-selected={mode === m}
              className={clsx("px-3 py-1.5 capitalize",
                mode === m ? "bg-violet/20 text-violet-soft" : "text-slate-400 hover:text-slate-200")}
              onClick={() => setMode(m)}
            >
              {m}
            </button>
          ))}
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-[280px_1fr]">
        <SettingsPanel />
        <div className="min-w-0 space-y-4">
          {mode === "single" ? <SingleView /> : <CompareView />}
          <HistoryPanel />
        </div>
      </div>
    </div>
  );
}