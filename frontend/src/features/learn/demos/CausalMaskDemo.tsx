import { useState } from "react";

import { Heatmap } from "../../attention/Heatmap";
import { causalAverage, parseNumberList } from "../../../utils/learn";

const FIELD = "w-full rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";

export function CausalMaskDemo() {
  const [text, setText] = useState("1, 3, 5, 7");
  const parsed = parseNumberList(text);
  const result = "values" in parsed ? causalAverage(parsed.values) : null;
  const tokens = "values" in parsed ? parsed.values.map((_, i) => ({ char: String(i), id: i })) : [];

  return (
    <div className="space-y-3">
      <label htmlFor="learn-nums" className="block text-sm text-slate-400">Numbers (2 to 8, comma separated)</label>
      <input id="learn-nums" aria-label="Numbers" className={FIELD} value={text}
        onChange={(e) => setText(e.target.value)} />
      {"error" in parsed && <p role="alert" className="text-xs text-amber-300">{parsed.error}</p>}
      {result && "values" in parsed && (
        <div className="flex flex-wrap items-start gap-6">
          <Heatmap tokens={tokens} matrix={result.weights} cell={28} label="Equal weights over the past"
            showLabels />
          <table className="font-mono text-xs" data-testid="causal-output">
            <thead className="text-slate-500">
              <tr><th className="pr-3 text-left">Position</th><th className="pr-3 text-left">Input</th><th className="text-left">Output</th></tr>
            </thead>
            <tbody>
              {parsed.values.map((v, i) => (
                <tr key={i}>
                  <td className="pr-3">{i}</td>
                  <td className="pr-3">{v}</td>
                  <td>{(result.output[i] ?? 0).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="text-xs text-slate-500">
        A worked example computed in your browser from the numbers above. It is not data from a
        model. Each output is the average of its own input and everything before it; the hatched
        cells are positions it may not read.
      </p>
    </div>
  );
}