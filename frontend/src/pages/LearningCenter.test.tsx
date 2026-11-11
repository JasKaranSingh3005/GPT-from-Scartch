import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { vi } from "vitest";

import { ModuleView } from "../features/learn/ModuleView";
import { LearningCenter } from "./LearningCenter";

const SUMMARY = { model_id: "tiny:best", run: "tiny", kind: "best", size_bytes: 1, modified: "2026-01-01T00:00:00Z", loaded: false };
const ENCODE = {
  model_id: "tiny:best", ids: [1, 2, 0], length: 3, vocab_size: 7, block_size: 8,
  tokens: [{ char: "a", id: 1 }, { char: "b", id: 2 }, { char: "\n", id: 0 }],
};
const HEAD = [[1, 0, 0], [0.5, 0.5, 0], [0.2, 0.3, 0.5]];
const ATTENTION = {
  model_id: "tiny:best", layer: 0, n_layers: 2, n_heads: 2, block_size: 8, cropped: false,
  tokens: ENCODE.tokens, weights: [HEAD, HEAD],
};
const DIST = {
  model_id: "tiny:best", prompt_tokens: 6, cropped: false, temperature: 0.8, top_k: null, shown_mass: 0.9,
  entries: [{ id: 1, char: "a", probability: 0.6 }, { id: 0, char: "\n", probability: 0.3 }],
};
const CONFIG = {
  model_id: "tiny:best", model_type: "gpt", head_size: 8, context_length: 8,
  model_kwargs: { n_embd: 16, n_head: 2, n_layer: 2, block_size: 8 },
};
const tensor = (name: string, shape: number[]) => ({ name, shape, numel: shape.reduce((a, b) => a * b, 1) });
const PARAMS = {
  model_id: "tiny:best", total: 4321, breakdown: null, breakdown_matches_actual: null,
  tensors: [
    tensor("blocks.0.ffn.net.0.weight", [64, 16]), tensor("blocks.0.ffn.net.0.bias", [64]),
    tensor("blocks.0.ffn.net.2.weight", [16, 64]), tensor("blocks.0.ffn.net.2.bias", [16]),
    tensor("blocks.0.ln1.weight", [16]),
  ],
};
const stat = (name: string, shape: number[]) => ({ name, shape, mean: 0.1, std: 0.5, min: -1, max: 1 });
const FORWARD = {
  model_id: "tiny:best", cropped: false, tokens: ENCODE.tokens,
  stages: [
    stat("token_embedding", [1, 6, 16]), stat("position_embedding", [6, 16]),
    stat("blocks.0", [1, 6, 16]), stat("ln_f", [1, 6, 16]),
  ],
};
const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });

function renderAt(path: string) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.endsWith("/tokenizer/encode")) return json(ENCODE);
    if (url.endsWith("/visualize/attention")) return json(ATTENTION);
    if (url.endsWith("/visualize/forward-pass")) return json(FORWARD);
    if (url.endsWith("/generate/distribution")) return json(DIST);
    if (url.endsWith("/config")) return json(CONFIG);
    if (url.endsWith("/parameters")) return json(PARAMS);
    if (url.endsWith("/models")) return json({ models: [SUMMARY] });
    return json({});
  }));
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="learn" element={<LearningCenter />} />
          <Route path="learn/:moduleId" element={<LearningCenter />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

test("opens the first module and lists all twelve, none marked as missing", () => {
  renderAt("/learn");
  expect(screen.getByRole("heading", { name: "What is a language model?" })).toBeInTheDocument();
  expect(within(screen.getByRole("navigation", { name: "Modules" })).getAllByRole("link")).toHaveLength(12);
  expect(screen.queryByText("soon")).not.toBeInTheDocument();
});

test("an unknown module redirects to the first one", () => {
  renderAt("/learn/nope");
  expect(screen.getByRole("heading", { name: "What is a language model?" })).toBeInTheDocument();
});

test("a module that is not written says so instead of showing filler", () => {
  render(<ModuleView modelId={null}
    module={{ id: "x", title: "X", summary: "About X.", written: false, sections: [], equations: [] }} />);
  expect(screen.getByText(/has not been written yet/)).toBeInTheDocument();
});

test("the causal-mask demo computes running averages and rejects bad input", async () => {
  renderAt("/learn/causal-masking");
  expect(screen.getByTestId("causal-output")).toHaveTextContent("4.00");
  const input = screen.getByLabelText("Numbers");
  await userEvent.clear(input);
  await userEvent.type(input, "5");
  expect(screen.getByRole("alert")).toHaveTextContent(/at least 2/);
});

test("the tokenizer demo shows the backend's tokens", async () => {
  renderAt("/learn/tokenization");
  await waitFor(() => expect(screen.getByTestId("learn-chips").children).toHaveLength(3));
});

test("the attention demo draws the backend's weights", async () => {
  renderAt("/learn/self-attention");
  expect(await screen.findByRole("img", { name: /Attention weights, layer 0, head 0/ })).toBeInTheDocument();
});

test("the temperature demo shows the backend's probabilities", async () => {
  renderAt("/learn/autoregressive-generation");
  const list = await screen.findByLabelText("Next-character probabilities");
  expect(within(list).getAllByRole("listitem")).toHaveLength(2);
  expect(list).toHaveTextContent("60.0%");
});

test("the embeddings demo shows only the measured token embedding after a forward pass", async () => {
  renderAt("/learn/embeddings");
  expect(screen.queryByRole("table", { name: "Forward pass statistics" })).not.toBeInTheDocument();
  const button = screen.getByRole("button", { name: "Run forward pass" });
  await waitFor(() => expect(button).toBeEnabled());
  await userEvent.click(button);
  const table = await screen.findByRole("table", { name: "Forward pass statistics" });
  expect(within(table).getByText("(1, 6, 16)")).toBeInTheDocument();
  expect(within(table).getByText("token_embedding")).toBeInTheDocument();
  expect(within(table).queryByText("ln_f")).not.toBeInTheDocument();
});

test("the residual demo shows the blocks and the positional demo adds the position table", async () => {
  renderAt("/learn/residual-connections");
  const button = screen.getByRole("button", { name: "Run forward pass" });
  await waitFor(() => expect(button).toBeEnabled());
  await userEvent.click(button);
  const table = await screen.findByRole("table", { name: "Forward pass statistics" });
  expect(within(table).getByText("blocks.0")).toBeInTheDocument();
  expect(within(table).queryByText("position_embedding")).not.toBeInTheDocument();
});

test("the multi-head demo shows the real head size and one map per head", async () => {
  renderAt("/learn/multi-head-attention");
  expect(await screen.findByText(/n_embd 16 ÷ 2 heads = head size 8/)).toBeInTheDocument();
  expect(await screen.findAllByRole("img", { name: /Head \d attention/ })).toHaveLength(2);
});

test("the feed-forward demo reads the checkpoint's real tensors", async () => {
  renderAt("/learn/feed-forward");
  expect(await screen.findByText(/16 → 64 → 16 per position/)).toBeInTheDocument();
  const table = screen.getByRole("table", { name: "Feed-forward tensors" });
  expect(within(table).getAllByRole("row")).toHaveLength(5); // header + the 4 ffn tensors
  expect(screen.getByText(/2,128 of the model's 4,321/)).toBeInTheDocument();
});

test("the LayerNorm demo reports a normalised result and says it is a worked example", () => {
  renderAt("/learn/layernorm");
  expect(screen.getByTestId("ln-stats")).toHaveTextContent("after: mean 0.000");
  expect(screen.getByText(/worked example computed in your browser/)).toBeInTheDocument();
});

test("the training demo pairs each prefix with the next character", async () => {
  renderAt("/learn/training");
  const input = screen.getByLabelText(/A window of text/);
  await userEvent.clear(input);
  await userEvent.type(input, "abc");
  const table = screen.getByRole("table", { name: "Training pairs" });
  expect(within(table).getAllByRole("row")).toHaveLength(3); // header + 2 pairs
  await userEvent.clear(input);
  await userEvent.type(input, "a");
  expect(screen.getByRole("status")).toHaveTextContent(/at least two/);
});