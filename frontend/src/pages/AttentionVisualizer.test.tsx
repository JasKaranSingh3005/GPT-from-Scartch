import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { AttentionVisualizer } from "./AttentionVisualizer";

const SUMMARY = { model_id: "tiny:best", run: "tiny", kind: "best", size_bytes: 1, modified: "2026-01-01T00:00:00Z", loaded: false };
const HEAD = [[1, 0, 0], [0.5, 0.5, 0], [0.2, 0.3, 0.5]];
const ATTENTION = {
  model_id: "tiny:best", layer: 0, n_layers: 2, n_heads: 2, block_size: 8, cropped: false,
  tokens: [{ char: "a", id: 1 }, { char: "b", id: 2 }, { char: "c", id: 3 }],
  weights: [HEAD, HEAD],
};
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

function stub(attention: () => Response) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.endsWith("/visualize/attention")) return attention();
    if (url.endsWith("/models")) return json({ models: [SUMMARY] });
    return json({});
  }));
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter><AttentionVisualizer /></MemoryRouter>
    </QueryClientProvider>,
  );
}

test("renders the backend's weights and shows the exact value on hover", async () => {
  stub(() => json(ATTENTION));
  renderPage();
  const main = await screen.findByRole("img", { name: "Attention heatmap, head 0" });
  expect(await screen.findByText(/3 tokens/)).toBeInTheDocument();
  expect(screen.getByRole("img", { name: "Head 1 overview" })).toBeInTheDocument();
  await userEvent.hover(main.querySelector("rect[data-q='2'][data-k='1']") as Element);
  expect(screen.getByTestId("readout")).toHaveTextContent("weight 0.3000");
});

test("shows the server's explanation instead of a heatmap on an error", async () => {
  stub(() => json({ detail: "This view needs a GPT checkpoint" }, 422));
  renderPage();
  expect(await screen.findByRole("alert")).toHaveTextContent("needs a GPT checkpoint");
  expect(screen.queryByRole("img", { name: /Attention heatmap/ })).not.toBeInTheDocument();
});