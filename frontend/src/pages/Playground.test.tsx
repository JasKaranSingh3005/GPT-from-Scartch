import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { DEFAULT_SETTINGS, usePlaygroundStore } from "../stores/playgroundStore";
import { Playground } from "./Playground";

const SYSTEM = {
  app_version: "0.1.0", python_version: "3.11", platform: "t", torch_version: "t", cuda_available: false,
  cuda_version: null, device: "cpu", gpus: [], cpu_count: 4, runs_dir_exists: true,
  checkpoints_found: 1, models_loaded: 0, limits: { max_new_tokens: 100, max_loaded_models: 3 },
};
const SUMMARY = { model_id: "tiny:best", run: "tiny", kind: "best", size_bytes: 1, modified: "2026-01-01T00:00:00Z", loaded: false };
const DETAIL = { ...SUMMARY, model_type: "gpt", iteration: 1, parameters: 1, vocab_size: 7,
  run_best_val_loss: null, run_final_val_loss: null, run_training_time_s: null };

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

function sseResponse(events: [string, object][]): Response {
  const text = events.map(([e, d]) => `event: ${e}\ndata: ${JSON.stringify(d)}\n\n`).join("");
  const stream = new ReadableStream<Uint8Array>({
    start(c) {
      c.enqueue(new TextEncoder().encode(text));
      c.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

function stub(streamResponse: () => Response) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.endsWith("/generate/stream")) return streamResponse();
    if (url.endsWith("/system/info")) return json(SYSTEM);
    if (url.endsWith("/models")) return json({ models: [SUMMARY] });
    if (url.endsWith("/config")) return json({ model_id: "tiny:best", model_type: "gpt", model_kwargs: {}, head_size: 8, context_length: 8 });
    return json(DETAIL);
  }));
}

function renderPlayground() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter><Playground /></MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() =>
  usePlaygroundStore.setState({
    settings: { ...DEFAULT_SETTINGS, maxNewTokens: 50 },
    history: [],
  }),
);

test("streams tokens from the backend into the output and records history", async () => {
  stub(() => sseResponse([
    ["start", { model_id: "tiny:best", device: "cpu", seed: null, max_new_tokens: 2 }],
    ["token", { id: 1, text: "a", index: 1 }],
    ["token", { id: 2, text: "b", index: 2 }],
    ["done", { tokens_generated: 2, seconds: 0.5, tokens_per_second: 4 }],
  ]));
  renderPlayground();
  const button = await screen.findByRole("button", { name: "Generate" });
  await waitFor(() => expect(button).toBeEnabled());
  await userEvent.click(button);

  expect(await screen.findByTestId("output")).toHaveTextContent("ROMEO:ab");
  await waitFor(() => expect(screen.getByTestId("stats")).toHaveTextContent("done"));
  expect(screen.getByTestId("stats")).toHaveTextContent("Tokens: 2");
  expect(usePlaygroundStore.getState().history).toHaveLength(1);
  expect(usePlaygroundStore.getState().history[0]?.completion).toBe("ab");
});

test("shows the server's validation message instead of output", async () => {
  stub(() => json({ detail: "Characters not in vocabulary: ['x']" }, 422));
  renderPlayground();
  const button = await screen.findByRole("button", { name: "Generate" });
  await waitFor(() => expect(button).toBeEnabled());
  await userEvent.click(button);
  expect(await screen.findByRole("alert")).toHaveTextContent("Characters not in vocabulary");
  expect(usePlaygroundStore.getState().history).toHaveLength(0);
});

test("a stream that ends without 'done' is reported as an error", async () => {
  stub(() => sseResponse([["start", { model_id: "tiny:best", device: "cpu", seed: null, max_new_tokens: 2 }]]));
  renderPlayground();
  const button = await screen.findByRole("button", { name: "Generate" });
  await waitFor(() => expect(button).toBeEnabled());
  await userEvent.click(button);
  expect(await screen.findByRole("alert")).toHaveTextContent(/ended before generation finished/);
});

test("warns and disables Generate above the server token limit", async () => {
  stub(() => json({}));
  renderPlayground();
  const max = await screen.findByLabelText(/Max new tokens/);
  await screen.findByText(/server limit 100/);
  await userEvent.clear(max);
  await userEvent.type(max, "500");
  expect(await screen.findByRole("alert")).toHaveTextContent(/exceeds the server limit/);
  expect(screen.getByRole("button", { name: "Generate" })).toBeDisabled();
});