import { useState } from "react";

import { visibleChar } from "../../../utils/format";
import { nextCharPairs } from "../../../utils/learn";

const FIELD = "w-full rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";
const show = (s: string) => [...s].map(visibleChar).join("");

export function ShiftDemo() {
  const [text, setText] = useState("To be, or not");
  const pairs = nextCharPairs(text);

  return (
    <div className="space-y-3">
      <label htmlFor="learn-shift" className="block text-sm text-slate-400">A window of text (up to 24 characters)</label>
      <input id="learn-shift" className={FIELD} maxLength={24} value={text}
        onChange={(e) => setText(e.target.value)} />
      {pairs.length === 0 ? (
        <p role="status" className="text-xs text-amber-300">Enter at least two characters.</p>
      ) : (
        <table className="w-full text-left font-mono text-xs" aria-label="Training pairs">
          <thead className="text-slate-500">
            <tr><th className="pr-3">Position</th><th className="pr-3">The model sees</th><th>It must predict</th></tr>
          </thead>
          <tbody>
            {pairs.map((p, i) => (
              <tr key={i}>
                <td className="pr-3">{i}</td>
                <td className="pr-3">{show(p.context)}</td>
                <td className="text-violet-soft">{visibleChar(p.target)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="text-xs text-slate-500">
        A worked example computed in your browser, not data from a model. In training the model
        sees the whole window at once, and the causal mask makes position {pairs.length > 0 ? "i" : "t"} use
        only the characters up to it, so one window gives all of these predictions in a single pass.
      </p>
    </div>
  );
}