import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";

import { ArchitectureExplorer } from "./ArchitectureExplorer";

// React Flow needs browser layout APIs that jsdom lacks, so the diagram is replaced here.
vi.mock("../features/architecture/Diagram", () => ({
  Diagram: ({ stages, onSelect }: { stages: { key: string; label: string }[]; onSelect: (k: string) => void }) => (
    <div>
      {stages.map((s) => (
        <button key={s.key} onClick={() => onSelect(s.key)}>{s.label}</button>
      ))}
    </div>
  ),
}));

const SUMMARY = { model_id: "tiny:best", run: "tiny", kind: "best", size_bytes: 1, modified: "2026-01-01T00:00:00Z", loaded: false };
const ARCH = {
  model_id: "tiny:best",
  config: { n_embd: 16, n_head: 2, n_layer: 2, block_size: 8, vocab_size: 7, dropout: 0 },
  parameters_total: 1000, stage_sum_matches_total: true,
  stages: [
    { key: "token_embedding", label: "Token embedding", input_shape: "(B, T)", output_shape: "(B, T, 16)", parameters: 112, details: { table: "7 x 16" } },
    { key: "blocks", label: "Transformer blocks x 2", input_shape: "(B, T, 16)", output_shape: "(B, T, 16)", parameters: 888, details: { head_size: 8 } },
  ],
};
const FORWARD = {
  model_id: "tiny:best", cropped: false, tokens: [],
  stages: [{ name: "token_embedding", shape: [1, 3, 16], mean: 0.1, std: 0.5, min: -1, max: 1 }],
};
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

function stub(architecture: () => Response) {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url.includes("/visualize/architecture")) return architecture();
    if (url.endsWith("/visualize/forward-pass")) return json(FORWARD);
    if (url.endsWith("/models")) return json({ models: [SUMMARY] });
    return json({});
  }));
}

function renderPage() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter><ArchitectureExplorer /></MemoryRouter>
    </QueryClientProvider>,
  );
}

test("shows real stage details and measured statistics only after a forward pass", async () => {
  stub(() => json(ARCH));
  renderPage();
  await userEvent.click(await screen.findByRole("button", { name: "Transformer blocks x 2" }));
  expect(screen.getByText("head_size")).toBeInTheDocument();
  expect(screen.getByText("888 (88.8%)")).toBeInTheDocument();
  expect(screen.getByText(/Run a forward pass above/)).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "Token embedding" }));
  await userEvent.click(screen.getByRole("button", { name: "Run forward pass" }));
  expect(await screen.findByText("(1, 3, 16)")).toBeInTheDocument();
});

test("shows the server's explanation for a model without blocks", async () => {
  stub(() => json({ detail: "This view needs a GPT checkpoint" }, 422));
  renderPage();
  expect(await screen.findByRole("alert")).toHaveTextContent("needs a GPT checkpoint");
});