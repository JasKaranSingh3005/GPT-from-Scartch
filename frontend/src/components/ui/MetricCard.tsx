import { Skeleton } from "./Skeleton";

interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
  loading?: boolean;
}

export function MetricCard({ label, value, hint, loading }: MetricCardProps) {
  return (
    <div className="rounded-md border border-navy-600 bg-navy-900 p-3">
      <div className="text-xs text-slate-400">{label}</div>
      {loading ? (
        <Skeleton className="mt-1.5 h-6 w-24" />
      ) : (
        <div className="mt-1 font-mono text-lg text-slate-50">{value}</div>
      )}
      {hint && !loading && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}
