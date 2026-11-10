import clsx from "clsx";
import type { ReactNode } from "react";

interface CardProps {
  title?: string;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function Card({ title, action, className, children }: CardProps) {
  return (
    <section className={clsx("rounded-xl border border-navy-600 bg-navy-800 p-6 shadow-sm", className)}>
      {(title || action) && (
        <div className="mb-4 flex items-center justify-between gap-3">
          {title && <h2 className="text-base font-semibold tracking-tight text-slate-50">{title}</h2>}
          {action}
        </div>
      )}
      {children}
    </section>
  );
}
