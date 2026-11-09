import clsx from "clsx";
import { ChevronsLeft, ChevronsRight, Menu } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router-dom";

import { BRAND } from "../brand";
import { BrandMark } from "../components/BrandMark";
import { ThemeToggle } from "../components/ThemeToggle";
import { useHealth } from "../hooks/useApi";
import { useUiStore } from "../stores/uiStore";
import { NAV_GROUPS, NAV_ITEMS, type NavItem } from "./nav";

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

function useCurrentItem(): NavItem | undefined {
  const { pathname } = useLocation();
  return NAV_ITEMS.find((item) =>
    item.path === "/" ? pathname === "/" : pathname === item.path || pathname.startsWith(`${item.path}/`),
  );
}

export function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const collapsed = useUiStore((s) => s.collapsed);
  const toggleSidebar = useUiStore((s) => s.toggleSidebar);
  const current = useCurrentItem();
  // The mobile drawer is always shown in full; the icon rail only applies on large screens.
  const showLabels = !collapsed || mobileOpen;
  const CurrentIcon = current?.icon;

  return (
    <div className="flex h-full overflow-hidden bg-navy-900">
      {mobileOpen && (
        <button
          type="button"
          aria-label="Close navigation"
          className="fixed inset-0 z-20 bg-black/60 lg:hidden"
          onClick={() => setMobileOpen(false)}
        />
      )}

      <aside
        className={clsx(
          "fixed inset-y-0 left-0 z-30 flex shrink-0 flex-col border-r border-navy-600 bg-navy-950",
          "transition-[width,transform] duration-200 motion-reduce:transition-none lg:static lg:translate-x-0",
          showLabels ? "w-64" : "w-16",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className={clsx("flex h-14 shrink-0 items-center gap-3 border-b border-navy-600", showLabels ? "px-5" : "justify-center")}>
          <BrandMark className="h-7 w-7 shrink-0" />
          {showLabels && (
            <div className="min-w-0">
              <div className="text-base font-semibold leading-tight tracking-tight text-slate-50">{BRAND.name}</div>
              <div className="truncate text-xs text-slate-400">{BRAND.tagline}</div>
            </div>
          )}
        </div>

        <nav aria-label="Main" className="flex-1 overflow-y-auto px-2 py-4">
          {NAV_GROUPS.map((group, index) => {
            const items = NAV_ITEMS.filter((item) => item.group === group);
            if (items.length === 0) return null;
            return (
              <div key={group} className={clsx(index > 0 && "mt-5")}>
                {showLabels ? (
                  <div className="px-3 pb-1.5 text-xs font-medium text-slate-500">{group}</div>
                ) : (
                  index > 0 && <div className="mx-3 mb-3 border-t border-navy-600" aria-hidden />
                )}
                <ul className="space-y-0.5">
                  {items.map(({ path, label, icon: Icon, implemented }) => (
                    <li key={path}>
                      <NavLink
                        to={path}
                        end={path === "/"}
                        aria-label={label}
                        title={showLabels ? undefined : label}
                        onClick={() => setMobileOpen(false)}
                        className={({ isActive }) =>
                          clsx(
                            "relative flex items-center gap-3 rounded-md py-2 text-sm font-medium transition-colors motion-reduce:transition-none",
                            showLabels ? "px-3" : "justify-center",
                            isActive ? "bg-navy-700 text-slate-50" : "text-slate-400 hover:bg-navy-800 hover:text-slate-100",
                          )
                        }
                      >
                        {({ isActive }) => (
                          <>
                            {isActive && <span aria-hidden className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-violet-soft" />}
                            <Icon className={clsx("h-[18px] w-[18px] shrink-0", isActive && "text-violet-soft")} aria-hidden />
                            {showLabels && <span className="flex-1 truncate">{label}</span>}
                            {showLabels && !implemented && (
                              <span className="rounded border border-navy-600 px-1.5 text-[11px] font-normal text-slate-500">Soon</span>
                            )}
                          </>
                        )}
                      </NavLink>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </nav>

        <div className="hidden border-t border-navy-600 p-2 lg:block">
          <button
            type="button"
            onClick={toggleSidebar}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className={clsx(
              "flex w-full items-center gap-3 rounded-md py-2 text-sm text-slate-400 transition-colors hover:bg-navy-800 hover:text-slate-100 motion-reduce:transition-none",
              collapsed ? "justify-center" : "px-3",
            )}
          >
            {collapsed ? (
              <ChevronsRight className="h-[18px] w-[18px]" aria-hidden />
            ) : (
              <>
                <ChevronsLeft className="h-[18px] w-[18px]" aria-hidden />
                <span>Collapse</span>
              </>
            )}
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 shrink-0 items-center gap-3 border-b border-navy-600 px-4 sm:px-6">
          <button
            type="button"
            className="rounded-md p-2 text-slate-400 hover:bg-navy-800 hover:text-slate-100 lg:hidden"
            aria-label="Open navigation"
            onClick={() => setMobileOpen(true)}
          >
            <Menu className="h-5 w-5" aria-hidden />
          </button>
          {current && CurrentIcon && (
            <div className="flex items-center gap-2 text-sm font-medium text-slate-200">
              <CurrentIcon className="h-4 w-4 text-slate-400" aria-hidden />
              <span>{current.label}</span>
            </div>
          )}
          <div className="ml-auto flex items-center gap-4">
            <BackendStatus />
            <ThemeToggle />
          </div>
        </header>
        <main className="flex-1 overflow-y-auto px-4 py-5 sm:px-6 sm:py-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
