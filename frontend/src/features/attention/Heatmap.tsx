import { useId } from "react";

import { useChartColors } from "../../theme/useChartColors";
import type { Rgb } from "../../theme/theme";
import type { EncodedToken } from "../../types/api";
import { visibleChar } from "../../utils/format";

export interface HoverInfo {
  query: number;
  key: number;
  weight: number;
}

interface HeatmapProps {
  tokens: EncodedToken[];
  matrix: number[][]; // [query][key]
  cell: number;
  label: string;
  showLabels?: boolean;
  onHover?: (info: HoverInfo | null) => void;
}

export interface Ramp {
  from: Rgb;
  to: Rgb;
}

/** Matches the dark theme's heat tokens. The component itself uses the live theme's ramp. */
const DEFAULT_RAMP: Ramp = { from: [78, 52, 22], to: [238, 158, 84] };
const mix = (a: number, b: number, t: number) => Math.round(a + (b - a) * t);

/** Weight to color on a square-root scale, so small weights stay visible (exact values are in the hover readout). */
export function colorFor(weight: number, ramp: Ramp = DEFAULT_RAMP): string {
  const t = Math.sqrt(Math.min(1, Math.max(0, weight)));
  const [fr, fg, fb] = ramp.from;
  const [tr, tg, tb] = ramp.to;
  return `rgb(${mix(fr, tr, t)},${mix(fg, tg, t)},${mix(fb, tb, t)})`;
}

/**
 * Row = the position doing the attending (query), column = the position attended to (key).
 * Cells above the diagonal are the causal mask: future positions, always weight 0.
 * They are hatched, not just darker, so "masked" is never told apart from "weight 0" by color alone.
 */
export function Heatmap({ tokens, matrix, cell, label, showLabels = false, onHover }: HeatmapProps) {
  const c = useChartColors();
  // Several heatmaps share one page (the Attention page draws five), so each needs its own pattern id.
  // React ids contain colons, which are awkward inside url(#...), so strip them.
  const patternId = `masked-${useId().replace(/:/g, "")}`;
  const ramp: Ramp = { from: c.heatFrom, to: c.heatTo };
  const n = tokens.length;
  const pad = showLabels ? 22 : 0;
  const size = pad + n * cell;
  const fontSize = Math.min(12, cell * 0.7);

  return (
    <svg
      role="img"
      aria-label={label}
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      style={{ maxWidth: "100%", height: "auto" }}
      onMouseLeave={() => onHover?.(null)}
    >
      <defs>
        {/* The hatch background is a <path>, not a <rect>: tests (and any CSS) treat every <rect> as a data cell. */}
        <pattern id={patternId} width={4} height={4} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <path d="M0 0h4v4H0z" fill={c.masked} />
          <line x1={0} y1={0} x2={0} y2={4} stroke={c.axis} strokeOpacity={0.5} strokeWidth={1} />
        </pattern>
      </defs>
      {showLabels &&
        tokens.map((t, i) => (
          <g key={i} fill={c.axis} fontFamily="monospace" fontSize={fontSize}>
            <text x={pad + i * cell + cell / 2} y={pad - 6} textAnchor="middle">
              {visibleChar(t.char)}
            </text>
            <text x={pad - 5} y={pad + i * cell + cell * 0.7} textAnchor="end">
              {visibleChar(t.char)}
            </text>
          </g>
        ))}
      {tokens.map((_, q) =>
        tokens.map((__, k) => {
          const masked = k > q;
          const weight = masked ? 0 : (matrix[q]?.[k] ?? 0);
          return (
            <rect
              key={`${q}-${k}`}
              data-q={q}
              data-k={k}
              data-masked={masked}
              x={pad + k * cell}
              y={pad + q * cell}
              width={cell}
              height={cell}
              fill={masked ? `url(#${patternId})` : colorFor(weight, ramp)}
              stroke={c.cellStroke}
              strokeWidth={0.5}
              onMouseEnter={masked ? undefined : () => onHover?.({ query: q, key: k, weight })}
            />
          );
        }),
      )}
    </svg>
  );
}