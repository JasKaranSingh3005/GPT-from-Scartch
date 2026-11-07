import clsx from "clsx";
import { NavLink, Outlet } from "react-router-dom";

import { useHealth } from "../hooks/useApi";
import { NAV_ITEMS } from "./nav";

function BackendStatus() {
  const { data, isError, isPending } = useHealth();
  const state = isPending ? "checking" : isError ? "offline" : "online";
  const color = { checking: "bg-slate-500", offline: "bg-red-500", online: "bg-emerald-400" }[state];
  return (
    <div className="flex items-center gap-2 text-xs text-slate-300" role="status">
      <span className={clsx("h-2 w-2 rounded-full", color)} aria-hidden />
      <span>Backend {state}</span>
      {data && <span className="font-mono text-slate-500">v{data.version}</span>}
    </div>
  );
}

export function Layout() {
  return (
    <div className="flex h-full">
      <aside className="flex w-60 shrink-0 flex-col border-r border-navy-600 bg-navy-950">
        <div className="px-4 py-5">
          <div className="text-lg font-semibold text-slate-50">
            GPT <span className="text-violet-soft">LAB</span>
          </div>
          <div className="text-[11px] uppercase tracking-widest text-slate-500">
            Build. Train. Understand.
          </div>
        </div>
        <nav aria-label="Main" className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-4">
          {NAV_ITEMS.map(({ path, label, icon: Icon, implemented }) => (
            <NavLink
              key={path}
              to={path}
              end={path === "/"}
              className={({ isActive }) =>
                clsx(
                  "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                  isActive
                    ? "bg-violet/15 text-violet-soft"
                    : "text-slate-400 hover:bg-navy-800 hover:text-slate-200",
                )
              }
            >
              <Icon className="h-4 w-4" aria-hidden />
              <span className="flex-1">{label}</span>
              {!implemented && <span className="text-[10px] text-slate-600">soon</span>}
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center justify-end border-b border-navy-600 px-6">
          <BackendStatus />
        </header>
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}