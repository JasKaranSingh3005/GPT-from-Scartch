import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { useModelStore } from "../stores/modelStore";
import { Dashboard } from "./Dashboard";

const SYSTEM = {
  app_version: "0.1.0", python_version: "3.11.0", platform: "test", torch_version: "9.9.9+test",
  cuda_available: false, cuda_version: null, device: "cpu", gpus: [], cpu_count: 8,
  runs_dir_exists: true, checkpoints_found: 1, models_loaded: 0,
  limits: { max_new_tokens: 1000, max_loaded_models: 3 },
};
const MODELS = {
  models: [{ model_id: "tiny:best", run: "tiny", kind: "best", size_bytes: 2048,
    modified: "2026-01-01T00:00:00Z", loaded: false }],
};
const DETAIL = { ...MODELS.models[0], model_type: "gpt", iteration: 10, parameters: 4321,
  vocab_size: 7, run_best_val_loss: null, run_final_val_loss: null, run_training_time_s: null };
const CONFIG = { model_id: "tiny:best", model_type: "gpt", head_size: 8, context_length: 8,
  model_kwargs: { n_embd: 16, n_head: 2, n_layer: 2, block_size: 8 } };
const PARAMS = { model_id: "tiny:best", total: 4321, breakdown: null,
  breakdown_matches_actual: null, tensors: [] };

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200 });
}

function stubBackend() {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.endsWith("/system/info")) return json(SYSTEM);
    if (url.endsWith("/models")) return json(MODELS);
    if (url.endsWith("/config")) return json(CONFIG);
    if (url.endsWith("/parameters")) return json(PARAMS);
    if (url.includes("/models/")) return json(DETAIL);
    return json({ status: "ok", version: "0.1.0", uptime_seconds: 1 });
  }));
}

function renderDashboard() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter><Dashboard /></MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => useModelStore.setState({ selectedModelId: null }));

test("shows real system and checkpoint data from the API", async () => {
  stubBackend();
  renderDashboard();
  expect(await screen.findByText("tiny:best")).toBeInTheDocument();
  expect(await screen.findByText("9.9.9+test")).toBeInTheDocument();
  expect(screen.getByText("Not available")).toBeInTheDocument();
  expect(screen.getByText("No model selected")).toBeInTheDocument();
});

test("inspecting a checkpoint shows its configuration and an honest dash for missing loss", async () => {
  stubBackend();
  renderDashboard();
  await userEvent.click(await screen.findByRole("button", { name: "Inspect" }));
  expect(await screen.findByText("4,321")).toBeInTheDocument();
  expect(screen.getByText("Best val loss").nextSibling).toHaveTextContent("\u2014");
});

test("shows an error state when the backend is unreachable", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("network"); }));
  renderDashboard();
  const alerts = await screen.findAllByRole("alert");
  expect(alerts.length).toBeGreaterThan(0);
  expect(alerts[0]).toHaveTextContent(/Cannot reach the backend/);
});