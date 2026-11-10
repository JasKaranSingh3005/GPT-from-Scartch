import { Skeleton } from "./Skeleton";

interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
  loading?: boolean;
}

export function MetricCard({ label, value, hint, loading }: MetricCardProps) {
  return (
    <div className="rounded-lg border border-navy-600 bg-navy-900 p-4">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-400">{label}</div>
      {loading ? (
        <Skeleton className="mt-1.5 h-6 w-24" />
      ) : (
        <div className="mt-1.5 font-mono text-xl font-medium tabular-nums text-slate-50">{value}</div>
      )}
      {hint && !loading && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}
