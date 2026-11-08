import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { TokenizerExplorer } from "./TokenizerExplorer";

const SUMMARY = { model_id: "tiny:best", run: "tiny", kind: "best", size_bytes: 1, modified: "2026-01-01T00:00:00Z", loaded: false };
const ENCODE = {
  model_id: "tiny:best", ids: [1, 2, 0], length: 3, vocab_size: 7, block_size: 8,
  tokens: [{ char: "a", id: 1 }, { char: "b", id: 2 }, { char: "\n", id: 0 }],
};
const VOCAB = { model_id: "tiny:best", vocab_size: 7, entries: [{ id: 0, char: "\n", codepoint: "U+000A" }] };
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

function stub(encode: () => Response) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.endsWith("/tokenizer/encode")) return encode();
    if (url.includes("/tokenizer/vocabulary")) return json(VOCAB);
    if (url.endsWith("/models")) return json({ models: [SUMMARY] });
    return json({});
  }));
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter><TokenizerExplorer /></MemoryRouter>
    </QueryClientProvider>,
  );
}

test("shows one chip per character and says the tokenizer is character-level", async () => {
  stub(() => json(ENCODE));
  renderPage();
  expect(screen.getByText(/character-level/i)).toBeInTheDocument();
  await waitFor(() => expect(screen.getByTestId("chips").children).toHaveLength(3));
  expect(screen.getByText(/All 3 tokens fit inside the 8-token window/)).toBeInTheDocument();
});

test("shows the server's error for characters outside the vocabulary", async () => {
  stub(() => json({ detail: "Characters not in vocabulary: ['x']" }, 422));
  renderPage();
  expect(await screen.findByRole("alert")).toHaveTextContent("Characters not in vocabulary");
});