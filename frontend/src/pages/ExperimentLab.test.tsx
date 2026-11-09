import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { ExperimentLab } from "./ExperimentLab";

const DEFS = {
  base_model: {},
  experiments: [{ key: "heads", title: "Attention heads", variants: [
    { name: "base", label: "GPT base", model_type: "gpt", overrides: {}, model: {} }] }],
};
const RUNS = { runs: [{ batch_id: "it100_s1", max_iters: 100, seed: 1, variants_finished: 1,
  variants_total: 14, modified: "2026-01-01T00:00:00Z" }] };
const DETAIL = {
  batch_id: "it100_s1", max_iters: 100, seed: 1, batch: null,
  curves: { base: [{ iter: 0, train_loss: 4, val_loss: 4.1 }, { iter: 10, train_loss: 2, val_loss: 2.5 }] },
  rows: [{ run: "base", label: "GPT base", parameters: 211777, training_time_s: 12, final_train_loss: 2.0,
    final_val_loss: 2.1, best_val_loss: 2.0123, best_iter: 10, val_minus_train: 0.1 }],
};
const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });

test("shows measured results from the API and links the CSV export", async () => {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.includes("/experiments/runs/")) return json(DETAIL);
    if (url.endsWith("/experiments/runs")) return json(RUNS);
    if (url.endsWith("/experiments")) return json(DEFS);
    if (url.endsWith("/training/jobs")) return json({ jobs: [] });
    return json({});
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><MemoryRouter><ExperimentLab /></MemoryRouter></QueryClientProvider>);

  const table = await screen.findByRole("table", { name: "Experiment results" });
  expect(within(table).getByText("211,777")).toBeInTheDocument();
  expect(within(table).getByText("2.0123")).toBeInTheDocument();
  const link = screen.getByRole("link", { name: /Download CSV/ });
  expect(link.getAttribute("href")).toMatch(/\/experiments\/export\/it100_s1$/);
  expect(screen.getByText(/Tick two to four batches/)).toBeInTheDocument();
});