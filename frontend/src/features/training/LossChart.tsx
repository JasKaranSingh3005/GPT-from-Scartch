import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { useChartColors } from "../../theme/useChartColors";
import type { MetricRow } from "../../types/api";

export function LossChart({ rows }: { rows: MetricRow[] }) {
  const c = useChartColors();
  return (
    <div className="h-72 w-full" data-testid="loss-chart">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows}>
          <CartesianGrid stroke={c.grid} strokeDasharray="3 3" />
          <XAxis dataKey="iter" stroke={c.axis} />
          <YAxis stroke={c.axis} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip contentStyle={c.tooltip} labelStyle={{ color: c.text }} formatter={(v: number) => v.toFixed(4)} />
          <Legend />
          <Line type="monotone" dataKey="train_loss" name="train" stroke={c.secondary} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="val_loss" name="validation" stroke={c.accent} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}