import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { vi } from "vitest";

import { useThemeStore } from "../theme/theme";
import { useUiStore } from "../stores/uiStore";
import { Layout } from "./Layout";

function renderLayout(path = "/playground") {
  vi.stubGlobal("fetch", vi.fn(async () =>
    new Response(JSON.stringify({ status: "ok", version: "0.1.0", uptime_seconds: 1 }), { status: 200 })));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes><Route element={<Layout />}><Route path="*" element={<div>page</div>} /></Route></Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  useUiStore.setState({ collapsed: false });
  useThemeStore.getState().setTheme("dark");
});

test("groups navigation, shows the current page and links every item", () => {
  renderLayout("/training");
  for (const group of ["Overview", "Build", "Inspect", "Learn", "System"]) {
    expect(screen.getByText(group)).toBeInTheDocument();
  }
  expect(screen.getAllByText("Training Lab").length).toBeGreaterThan(0);
  expect(screen.getAllByRole("link")).toHaveLength(10);
});

test("the sidebar collapses to icons and keeps accessible names", async () => {
  renderLayout();
  await userEvent.click(screen.getByRole("button", { name: "Collapse sidebar" }));
  expect(screen.queryByText("Build")).not.toBeInTheDocument();
  expect(screen.getByRole("link", { name: "GPT Playground" })).toBeInTheDocument();
});

test("the theme toggle switches the document theme", async () => {
  renderLayout();
  await userEvent.click(screen.getByRole("button", { name: "Switch to light theme" }));
  expect(document.documentElement.dataset["theme"]).toBe("light");
  expect(screen.getByRole("button", { name: "Switch to dark theme" })).toBeInTheDocument();
});