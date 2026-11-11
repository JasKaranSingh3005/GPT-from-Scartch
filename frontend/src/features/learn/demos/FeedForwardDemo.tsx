import { ErrorState } from "../../../components/ui/states";
import { useModelParameters } from "../../../hooks/useApi";
import { formatInt } from "../../../utils/format";

export function FeedForwardDemo({ modelId }: { modelId: string | null }) {
  const query = useModelParameters(modelId);
  const data = query.data;
  const ffn = data?.tensors.filter((t) => t.name.startsWith("blocks.0.ffn.")) ?? [];
  const first = ffn.find((t) => t.name.endsWith("ffn.net.0.weight"));
  const width = first ? first.shape[1] : undefined;
  const hidden = first ? first.shape[0] : undefined;
  const total = ffn.reduce((sum, t) => sum + t.numel, 0);

  return (
    <div className="space-y-3">
      {query.error && <ErrorState error={query.error} />}
      {data && ffn.length === 0 && !query.error && (
        <p role="status" className="text-sm text-amber-300">This checkpoint has no feed-forward layers.</p>
      )}
      {data && ffn.length > 0 && (
        <>
          {width !== undefined && hidden !== undefined && (
            <p className="font-mono text-sm text-slate-200">
              {width} → {hidden} → {width} per position
            </p>
          )}
          <table className="w-full text-left font-mono text-xs" aria-label="Feed-forward tensors">
            <thead className="text-slate-500">
              <tr><th className="pr-3">Tensor</th><th className="pr-3">Shape</th><th>Parameters</th></tr>
            </thead>
            <tbody>
              {ffn.map((t) => (
                <tr key={t.name}>
                  <td className="pr-3">{t.name}</td>
                  <td className="pr-3">[{t.shape.join(", ")}]</td>
                  <td>{formatInt(t.numel)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-slate-500">
            This block's feed-forward network holds {formatInt(total)} of the model's {formatInt(data.total)}{" "}
            parameters ({((100 * total) / data.total).toFixed(1)}%). Read from the checkpoint's tensors.
          </p>
        </>
      )}
    </div>
  );
}