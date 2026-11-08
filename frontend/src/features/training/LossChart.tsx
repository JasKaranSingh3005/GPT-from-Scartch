import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { MetricRow } from "../../types/api";

export function LossChart({ rows }: { rows: MetricRow[] }) {
  return (
    <div className="h-72 w-full" data-testid="loss-chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows}>
          <CartesianGrid stroke="#26325c" strokeDasharray="3 3" />
          <XAxis dataKey="iter" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip contentStyle={{ background: "#111833", border: "1px solid #26325c" }}
            labelStyle={{ color: "#e2e8f0" }} formatter={(v: number) => v.toFixed(4)} />
          <Legend />
          <Line type="monotone" dataKey="train_loss" name="train" stroke="#22d3ee" dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="val_loss" name="validation" stroke="#8b5cf6" dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}