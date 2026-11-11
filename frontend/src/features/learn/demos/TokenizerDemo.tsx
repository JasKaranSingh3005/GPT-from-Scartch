import { useState } from "react";

import { ErrorState } from "../../../components/ui/states";
import { useEncode } from "../../../hooks/useApi";
import { useDebounced } from "../../../hooks/useDebounced";
import { visibleChar } from "../../../utils/format";

const FIELD = "w-full rounded border border-navy-600 bg-navy-900 px-2 py-1.5 text-sm font-mono";

export function TokenizerDemo({ modelId }: { modelId: string | null }) {
  const [text, setText] = useState("To be, or not to be");
  const query = useEncode(modelId, useDebounced(text, 300));
  const data = query.data;

  return (
    <div className="space-y-3">
      <label htmlFor="learn-tok" className="block text-sm text-slate-400">Text</label>
      <textarea id="learn-tok" rows={2} maxLength={500} className={FIELD} value={text}
        onChange={(e) => setText(e.target.value)} />
      {query.error && <ErrorState error={query.error} />}
      {data && (
        <>
          <div className="flex flex-wrap gap-1" data-testid="learn-chips">
            {data.tokens.map((t, i) => (
              <span key={i} title={`token id ${t.id}`}
                className="rounded bg-violet/15 px-1.5 py-0.5 font-mono text-sm text-violet-soft">
                {visibleChar(t.char)}
                <sub className="ml-0.5 text-[9px] text-slate-500">{t.id}</sub>
              </span>
            ))}
          </div>
          <p className="text-xs text-slate-500">
            {data.length} characters become {data.length} tokens. The vocabulary has {data.vocab_size} entries
            and comes from the selected checkpoint.
          </p>
        </>
      )}
    </div>
  );
}