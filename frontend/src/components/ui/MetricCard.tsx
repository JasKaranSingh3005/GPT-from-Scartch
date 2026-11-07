import { Skeleton } from "./Skeleton";

interface MetricCardProps {
  label: string;
  value: string;
  hint?: string;
  loading?: boolean;
}

export function MetricCard({ label, value, hint, loading }: MetricCardProps) {
  return (
    <div className="rounded-lg border border-navy-600 bg-navy-800 p-4">
      <div className="text-xs uppercase tracking-wide text-slate-400">{label}</div>
      {loading ? (
        <Skeleton className="mt-2 h-7 w-24" />
      ) : (
        <div className="mt-1 font-mono text-xl text-slate-50">{value}</div>
      )}
      {hint && !loading && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}