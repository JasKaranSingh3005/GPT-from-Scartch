import { useState } from "react";

import { Card } from "../../components/ui/Card";
import { BUTTON_PRIMARY, BUTTON_SECONDARY } from "../../components/ui/controls";
import { ErrorState } from "../../components/ui/states";
import { useGenerationStream } from "../../hooks/useGenerationStream";
import { useSystemInfo } from "../../hooks/useApi";
import { usePlaygroundStore } from "../../stores/playgroundStore";
import { OutputView } from "./OutputView";
export function SingleView() {
  const settings = usePlaygroundStore((s) => s.settings);
  const update = usePlaygroundStore((s) => s.update);
  const addHistory = usePlaygroundStore((s) => s.addHistory);
  const system = useSystemInfo();
  const { state, start, stop } = useGenerationStream();
  const [showTokens, setShowTokens] = useState(false);
  const [copied, setCopied] = useState<"yes" | "failed" | null>(null);

  const limit = system.data?.limits.max_new_tokens ?? null;
  const overLimit = limit !== null && settings.maxNewTokens > limit;
  const streaming = state.status === "streaming";
  const canRun = settings.modelId !== null && !overLimit && !streaming;

  const run = () => {
    if (settings.modelId === null) return;
    setCopied(null);
    void start(
      {
        model_id: settings.modelId,
        prompt: settings.prompt,
        max_new_tokens: settings.maxNewTokens,
        temperature: settings.temperature,
        top_k: settings.topK,
        seed: settings.seed,
      },
      addHistory,
    );
  };

  const copy = async () => {
    const text = (state.request?.prompt ?? "") + state.tokens.map((t) => t.text).join("");
    try {
      await navigator.clipboard.writeText(text);
      setCopied("yes");
    } catch {
      setCopied("failed");
    }
  };

  const hasOutput = state.status !== "idle";
  const seconds = state.done?.seconds;

  return (
    <div className="space-y-4">
      <Card title="Prompt">
        <label htmlFor="pg-prompt" className="sr-only">Prompt</label>
        <textarea
          id="pg-prompt" rows={4}
          className="w-full rounded border border-field bg-navy-900 p-2 font-mono text-sm"
          value={settings.prompt}
          onChange={(e) => update({ prompt: e.target.value })}
        />
        <p className="mt-1 text-xs text-slate-500">
          Only characters from the model&apos;s training vocabulary are accepted. An empty prompt
          starts from a newline.
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button className={BUTTON_PRIMARY}
            disabled={!canRun} onClick={run}>
            Generate
          </button>
          <button className={BUTTON_SECONDARY}
            disabled={!streaming} onClick={stop}>
            Stop
          </button>
          <button className={BUTTON_SECONDARY}
            disabled={!canRun || !hasOutput} onClick={run}>
            Regenerate
          </button>
          {overLimit && (
            <span role="alert" className="text-xs text-amber-300">
              Max new tokens exceeds the server limit ({limit}).
            </span>
          )}
        </div>
      </Card>

      {state.error && <ErrorState error={new Error(state.error)} />}

      {hasOutput && (
        <Card
          title="Output"
          action={
            <div className="flex items-center gap-3 text-xs">
              <label className="flex items-center gap-1.5 text-slate-400">
                <input type="checkbox" checked={showTokens}
                  onChange={(e) => setShowTokens(e.target.checked)} />
                Show tokens
              </label>
              <button className="rounded border border-navy-600 px-2 py-1 hover:bg-navy-700"
                disabled={state.tokens.length === 0} onClick={() => void copy()}>
                {copied === "yes" ? "Copied" : copied === "failed" ? "Copy failed" : "Copy"}
              </button>
            </div>
          }
        >
          <OutputView prompt={state.request?.prompt ?? ""} tokens={state.tokens}
            showTokens={showTokens} streaming={streaming} />
          <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-400" data-testid="stats">
            <div>Status: <span className="font-mono text-slate-200">{state.status}</span></div>
            <div>Tokens: <span className="font-mono text-slate-200">{state.tokens.length}</span></div>
            {state.start && (
            <div>Device: <span className="font-mono text-slate-200">{state.start.device}</span></div>
            )}
            {seconds !== undefined && (
              <div>Server time: <span className="font-mono text-slate-200">{seconds.toFixed(2)} s</span></div>
            )}
            {state.done && (
              <div>Speed: <span className="font-mono text-slate-200">
                {state.done.tokens_per_second.toFixed(1)} tok/s</span></div>
            )}
            {state.start?.seed !== null && state.start?.seed !== undefined && (
              <div>Seed: <span className="font-mono text-slate-200">{state.start.seed}</span>
                {" "}(reproducible on the same device type only)</div>
            )}
          </dl>
        </Card>
      )}
    </div>
  );
}