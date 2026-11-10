import { useMemo, useState } from "react";
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { Card } from "../components/ui/Card";
import { BUTTON_PRIMARY, FIELD_BASE } from "../components/ui/controls";
import { EmptyState, ErrorState } from "../components/ui/states";
import { useDecode, useEncode, useVocabulary } from "../hooks/useApi";
import { useDebounced } from "../hooks/useDebounced";
import { useModelChoice } from "../hooks/useModelChoice";
import { charFrequencies, parseIds, tokensOutsideWindow } from "../utils/tokens";
import { formatInt, visibleChar } from "../utils/format";
import { useChartColors } from "../theme/useChartColors";

const MAX_CHIPS = 500;
const FIELD = `${FIELD_BASE} w-full`;

export function TokenizerExplorer() {
  const c = useChartColors();
  const { models, modelId, setModelId } = useModelChoice();
  const [text, setText] = useState("To be, or not to be");
  const debounced = useDebounced(text, 300);
  const encode = useEncode(modelId, debounced);
  const vocabulary = useVocabulary(modelId);
  const decode = useDecode();
  const [idsText, setIdsText] = useState("");
  const [idsError, setIdsError] = useState<string | null>(null);

  const data = encode.data;
  const outside = data ? tokensOutsideWindow(data.length, data.block_size) : 0;
  const frequencies = useMemo(
    () =>
      data
        ? charFrequencies(data.tokens)
            .slice(0, 20)
            .map((f) => ({ label: visibleChar(f.char), count: f.count }))
        : [],
    [data],
  );

  const runDecode = () => {
    const parsed = parseIds(idsText);
    if ("error" in parsed) {
      setIdsError(parsed.error);
      return;
    }
    setIdsError(null);
    if (modelId !== null) decode.mutate({ modelId, ids: parsed.ids });
  };

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-semibold text-slate-50">Tokenizer Explorer</h1>
      <p className="max-w-3xl text-sm text-slate-400">
        This model uses a <strong className="text-slate-200">character-level</strong> tokenizer: every
        distinct character is one token and its ID is its position in the sorted vocabulary. It is
        not BPE, so there are no subword tokens here.
      </p>

      <Card title="Input">
        <div className="mb-3 max-w-sm">
          <label htmlFor="tk-model" className="mb-1 block text-sm text-slate-400">
            Vocabulary from checkpoint
          </label>
          <select id="tk-model" className={FIELD} value={modelId ?? ""}
            onChange={(e) => setModelId(e.target.value || null)}>
            {models.data?.models.map((m) => (
              <option key={m.model_id} value={m.model_id}>{m.model_id}</option>
            ))}
          </select>
        </div>
        <label htmlFor="tk-text" className="sr-only">Text to tokenize</label>
        <textarea id="tk-text" rows={3} maxLength={20000} className={FIELD} value={text}
          onChange={(e) => setText(e.target.value)} />
      </Card>

      {encode.error && <ErrorState error={encode.error} />}
      {models.error && <ErrorState error={models.error} />}

      {data && (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[
              ["Characters", formatInt(text.length)],
              ["Tokens", formatInt(data.length)],
              ["Vocabulary size", formatInt(data.vocab_size)],
              ["Context window", data.block_size === null ? "none" : formatInt(data.block_size)],
            ].map(([label, value]) => (
              <div key={label} className="rounded-lg border border-navy-600 bg-navy-800 p-4">
                <div className="text-xs text-slate-400">{label}</div>
                <div className="mt-1 font-mono text-xl text-slate-50">{value}</div>
              </div>
            ))}
          </div>

          <Card title="Tokens and context window">
            <div className="flex flex-wrap gap-1" data-testid="chips">
              {data.tokens.slice(0, MAX_CHIPS).map((t, i) => (
                <span key={i} title={`token id ${t.id}`}
                  className={
                    i < outside
                      ? "rounded bg-navy-700 px-1.5 py-0.5 font-mono text-sm text-slate-500"
                      : "rounded bg-violet/15 px-1.5 py-0.5 font-mono text-sm text-violet-soft"
                  }>
                  {visibleChar(t.char)}
                  <sub className="ml-0.5 text-[9px] text-slate-500">{t.id}</sub>
                </span>
              ))}
            </div>
            {data.tokens.length > MAX_CHIPS && (
              <p className="mt-2 text-xs text-slate-500">
                Showing the first {MAX_CHIPS} of {formatInt(data.tokens.length)} tokens.
              </p>
            )}
            <p className="mt-3 text-xs text-slate-400">
              {data.block_size === null
                ? "This model has no context limit (it only looks at the previous token)."
                : outside > 0
                  ? `${outside} leading token(s) are dimmed: they exceed the ${data.block_size}-token window, so generation would only see the last ${data.block_size}.`
                  : `All ${data.length} tokens fit inside the ${data.block_size}-token window.`}
            </p>
            <p className="mt-2 break-all font-mono text-xs text-slate-500">
              IDs: [{data.ids.slice(0, 200).join(", ")}{data.ids.length > 200 ? ", ..." : ""}]
            </p>
          </Card>

          <Card title="Characters in your text">
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={frequencies}>
                  <XAxis dataKey="label" stroke={c.axis} />
                  <YAxis allowDecimals={false} stroke={c.axis} />
                  <Tooltip
                    cursor={{ fill: c.cursor }}
                    contentStyle={c.tooltip}
                    labelStyle={{ color: c.text }}
                  />
                  <Bar dataKey="count" fill={c.accent} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Counted from the text above (top 20), not from the training data.
            </p>
          </Card>
        </>
      )}

      {!data && !encode.error && text.length === 0 && (
        <EmptyState title="Nothing to tokenize">Type some text above.</EmptyState>
      )}

      <Card title="Decode token IDs">
        <div className="flex flex-wrap items-start gap-2">
          <label htmlFor="tk-ids" className="sr-only">Token IDs</label>
          <input id="tk-ids" className={`${FIELD} max-w-md`} placeholder="e.g. 30, 27, 25"
            value={idsText} onChange={(e) => setIdsText(e.target.value)} />
          <button className={BUTTON_PRIMARY}
            disabled={modelId === null || decode.isPending} onClick={runDecode}>
            Decode
          </button>
        </div>
        {idsError && <div className="mt-2"><ErrorState error={new Error(idsError)} /></div>}
        {decode.error && <div className="mt-2"><ErrorState error={decode.error} /></div>}
        {decode.data && !idsError && (
          <pre data-testid="decoded" className="mt-2 whitespace-pre-wrap rounded border border-navy-600 bg-navy-900 p-2 font-mono text-sm">
            {decode.data.text}
          </pre>
        )}
      </Card>

      <Card title={`Vocabulary${vocabulary.data ? ` (${vocabulary.data.vocab_size})` : ""}`}>
        {vocabulary.error && <ErrorState error={vocabulary.error} />}
        {vocabulary.data && (
          <details>
            <summary className="cursor-pointer text-sm text-slate-400">Show all entries</summary>
            <div className="mt-2 grid grid-cols-2 gap-1 font-mono text-xs sm:grid-cols-4 lg:grid-cols-6">
              {vocabulary.data.entries.map((e) => (
                <div key={e.id} className="rounded border border-navy-600 px-2 py-1">
                  <span className="text-violet-soft">{visibleChar(e.char)}</span>{" "}
                  <span className="text-slate-400">{e.id}</span>{" "}
                  <span className="text-slate-500">{e.codepoint}</span>
                </div>
              ))}
            </div>
          </details>
        )}
      </Card>
    </div>
  );
}