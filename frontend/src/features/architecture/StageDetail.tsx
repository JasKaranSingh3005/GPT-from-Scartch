import { Card } from "../../components/ui/Card";
import type { ArchitectureResponse, ArchitectureStage, TensorStat } from "../../types/api";
import { formatInt } from "../../utils/format";
import { formatShape, STAGE_INFO, statsForStage } from "../../utils/architecture";

interface StageDetailProps {
  stage: ArchitectureStage;
  architecture: ArchitectureResponse;
  stats: TensorStat[] | null;
}

export function StageDetail({ stage, architecture, stats }: StageDetailProps) {
  const info = STAGE_INFO[stage.key];
  const share =
    architecture.parameters_total > 0
      ? ((100 * stage.parameters) / architecture.parameters_total).toFixed(1)
      : null;
  const measured = stats ? statsForStage(stage.key, stats) : [];

  return (
    <Card title={stage.label}>
      <div className="space-y-3 text-sm">
        {info && <p className="text-slate-400">{info.summary}</p>}
        {info && (
          <pre className="overflow-x-auto rounded border border-navy-600 bg-navy-900 p-2 font-mono text-xs text-cyan">
            {info.equation}
          </pre>
        )}
        <dl className="divide-y divide-navy-600">
          <Row label="Input shape" value={stage.input_shape} />
          <Row label="Output shape" value={stage.output_shape} />
          <Row
            label="Parameters"
            value={`${formatInt(stage.parameters)}${share !== null && stage.parameters > 0 ? ` (${share}%)` : ""}`}
          />
          {Object.entries(stage.details).map(([k, v]) => (
            <Row key={k} label={k} value={typeof v === "number" ? formatInt(v) : v} />
          ))}
        </dl>
        <p className="text-xs text-slate-500">
          Shapes use B = batch, T = sequence length. Values come from the checkpoint&apos;s config
          and tensors.
        </p>

        <div>
          <h3 className="mb-1 text-xs font-medium text-slate-400">
            Measured on your prompt
          </h3>
          {stats === null && (
            <p className="text-xs text-slate-500">Run a forward pass above to see real statistics.</p>
          )}
          {stats !== null && measured.length === 0 && (
            <p className="text-xs text-slate-500">
              This stage has no tensor output of its own in the recorded pass.
            </p>
          )}
          {measured.length > 0 && (
            <table className="w-full text-left font-mono text-xs">
              <thead className="text-slate-500">
                <tr><th className="pr-2">Module</th><th className="pr-2">Shape</th>
                  <th className="pr-2">Mean</th><th className="pr-2">Std</th>
                  <th className="pr-2">Min</th><th>Max</th></tr>
              </thead>
              <tbody>
                {measured.map((m) => (
                  <tr key={m.name}>
                    <td className="pr-2">{m.name}</td>
                    <td className="pr-2">{formatShape(m.shape)}</td>
                    <td className="pr-2">{m.mean.toFixed(3)}</td>
                    <td className="pr-2">{m.std.toFixed(3)}</td>
                    <td className="pr-2">{m.min.toFixed(3)}</td>
                    <td>{m.max.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 py-1.5">
      <dt className="text-slate-400">{label}</dt>
      <dd className="text-right font-mono text-slate-100">{value}</dd>
    </div>
  );
}