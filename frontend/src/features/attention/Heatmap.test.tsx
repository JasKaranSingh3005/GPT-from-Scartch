import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi } from "vitest";

import { colorFor, Heatmap } from "./Heatmap";

const tokens = [{ char: "a", id: 1 }, { char: "b", id: 2 }, { char: "c", id: 3 }];
const matrix = [[1, 0, 0], [0.25, 0.75, 0], [0.2, 0.3, 0.5]];

test("draws every cell and marks exactly the upper triangle as masked", () => {
  const { container } = render(<Heatmap tokens={tokens} matrix={matrix} cell={10} label="h" />);
  expect(container.querySelectorAll("rect")).toHaveLength(9);
  const masked = container.querySelectorAll("rect[data-masked='true']");
  expect(masked).toHaveLength(3);
  masked.forEach((r) => expect(Number(r.getAttribute("data-k"))).toBeGreaterThan(Number(r.getAttribute("data-q"))));
});

test("hovering an allowed cell reports its real weight; masked cells report nothing", async () => {
  const onHover = vi.fn();
  const { container } = render(<Heatmap tokens={tokens} matrix={matrix} cell={10} label="h" onHover={onHover} />);
  await userEvent.hover(container.querySelector("rect[data-q='2'][data-k='1']") as Element);
  expect(onHover).toHaveBeenLastCalledWith({ query: 2, key: 1, weight: 0.3 });
  onHover.mockClear();
  await userEvent.hover(container.querySelector("rect[data-q='0'][data-k='2']") as Element);
  expect(onHover).not.toHaveBeenCalled();
});

test("colorFor clamps and is monotonic in the weight", () => {
  expect(colorFor(-1)).toBe(colorFor(0));
  expect(colorFor(2)).toBe(colorFor(1));
  expect(colorFor(0)).not.toBe(colorFor(1));
});