import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { vi } from "vitest";

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
const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });

function renderAt(path: string) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.endsWith("/tokenizer/encode")) return json(ENCODE);
    if (url.endsWith("/visualize/attention")) return json(ATTENTION);
    if (url.endsWith("/generate/distribution")) return json(DIST);
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

test("opens the first module and lists all twelve", () => {
  renderAt("/learn");
  expect(screen.getByRole("heading", { name: "What is a language model?" })).toBeInTheDocument();
  expect(within(screen.getByRole("navigation", { name: "Modules" })).getAllByRole("link")).toHaveLength(12);
});

test("an unknown module redirects to the first one", () => {
  renderAt("/learn/nope");
  expect(screen.getByRole("heading", { name: "What is a language model?" })).toBeInTheDocument();
});

test("an unwritten module says so instead of showing filler", () => {
  renderAt("/learn/embeddings");
  expect(screen.getByText(/has not been written yet/)).toBeInTheDocument();
});

test("the causal-mask demo computes running averages and rejects bad input", async () => {
  renderAt("/learn/causal-masking");
  expect(screen.getByTestId("causal-output")).toHaveTextContent("4.00"); // default 1, 3, 5, 7
  const input = screen.getByLabelText("Numbers");
  await userEvent.clear(input);
  await userEvent.type(input, "2, 4, 6");
  expect(screen.getByTestId("causal-output")).toHaveTextContent("4.00"); // mean of 2, 4, 6
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