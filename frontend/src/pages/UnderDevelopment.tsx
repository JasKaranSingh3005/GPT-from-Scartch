import { EmptyState } from "../components/ui/states";

export function UnderDevelopment({ title }: { title: string }) {
  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold text-slate-50">{title}</h1>
      <EmptyState title="Under development">
        This page is not implemented yet. Nothing here is simulated.
      </EmptyState>
    </div>
  );
}