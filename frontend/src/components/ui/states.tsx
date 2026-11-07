import { AlertTriangle, Inbox } from "lucide-react";
import type { ReactNode } from "react";

export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : "Something went wrong";
  return (
    <div
      role="alert"
      className="flex items-start gap-2 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-sm text-red-200"
    >
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
      <span>{message}</span>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-navy-600 p-8 text-center">
      <Inbox className="h-6 w-6 text-slate-500" aria-hidden />
      <div className="text-sm font-medium text-slate-200">{title}</div>
      {children && <div className="max-w-md text-sm text-slate-400">{children}</div>}
    </div>
  );
}