import { Card } from "../../components/ui/Card";
import { EmptyState } from "../../components/ui/states";
import { usePlaygroundStore } from "../../stores/playgroundStore";
import { DASH } from "../../utils/format";

export function HistoryPanel() {
  const history = usePlaygroundStore((s) => s.history);
  const restore = usePlaygroundStore((s) => s.restore);
  const clear = usePlaygroundStore((s) => s.clearHistory);

  return (
    <Card
      title="History (this session)"
      action={
        history.length > 0 ? (
          <button className="text-xs text-slate-400 hover:text-slate-200" onClick={clear}>
            Clear
          </button>
        ) : undefined
      }
    >
      {history.length === 0 ? (
        <EmptyState title="No generations yet" />
      ) : (
        <ul className="divide-y divide-navy-600">
          {history.map((h) => (
            <li key={h.id} className="flex items-start justify-between gap-3 py-2 text-sm">
              <div className="min-w-0">
                <div className="truncate font-mono text-slate-200">
                  {(h.request.prompt + h.completion).replace(/\n/g, " \u21b5 ").slice(0, 120)}
                </div>
                <div className="mt-0.5 text-xs text-slate-500">
                  {h.request.model_id} | T {h.request.temperature} | top-k {h.request.top_k ?? "off"} |
                  seed {h.request.seed ?? "random"} | {h.tokens} tokens
                  {h.tokensPerSecond !== null ? ` | ${h.tokensPerSecond.toFixed(0)} tok/s` : ` | ${DASH}`}
                  {h.status === "stopped" ? " | stopped" : ""}
                </div>
              </div>
              <button
                className="shrink-0 rounded border border-navy-600 px-2 py-1 text-xs hover:bg-navy-700"
                onClick={() => restore(h)}
              >
                Restore settings
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}