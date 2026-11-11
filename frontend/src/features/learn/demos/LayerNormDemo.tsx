import { useState } from "react";

import { layerNorm } from "../../../utils/learn";

const BASE = [-1.2, 0.3, 0.9, -0.4, 1.5, -0.8, 0.1, 0.6];
const fix = (v: number) => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);

export function LayerNormDemo() {
  const [shift, setShift] = useState(3);
  const [spread, setSpread] = useState(4);
  const before = BASE.map((b) => b * spread + shift);
  const { output, mean, std } = layerNorm(before);
  const after = layerNorm(output);

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-6 text-sm">
        <label className="text-slate-400">Mean of the input: {shift.toFixed(1)}
          <input type="range" min={-10} max={10} step={0.5} className="mt-1 block w-48 accent-violet"
            aria-label="Mean of the input" value={shift} onChange={(e) => setShift(Number(e.target.value))} />
        </label>
        <label className="text-slate-400">Spread of the input: {spread.toFixed(1)}
          <input type="range" min={0.5} max={10} step={0.5} className="mt-1 block w-48 accent-violet"
            aria-label="Spread of the input" value={spread} onChange={(e) => setSpread(Number(e.target.value))} />
        </label>
      </div>
      <table className="font-mono text-xs" aria-label="LayerNorm example">
        <tbody>
          <tr><th className="pr-3 text-left text-slate-500">before</th>
            {before.map((v, i) => <td key={i} className="pr-3">{v.toFixed(2)}</td>)}</tr>
          <tr><th className="pr-3 text-left text-slate-500">after</th>
            {output.map((v, i) => <td key={i} className="pr-3">{v.toFixed(2)}</td>)}</tr>
        </tbody>
      </table>
      <p data-testid="ln-stats" className="font-mono text-xs text-slate-300">
        before: mean {fix(mean)}, std {fix(std)} | after: mean {fix(after.mean)}, std {fix(after.std)}
      </p>
      <p className="text-xs text-slate-500">
        A worked example computed in your browser on eight made-up numbers. It is not data from a
        model, and it uses a scale of 1 and a shift of 0. The standard deviation after is just under
        1 because of the small constant added for safety.
      </p>
    </div>
  );
}