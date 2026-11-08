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

const mix = (a: number, b: number, t: number) => Math.round(a + (b - a) * t);

/** Navy (weight 0) to violet (weight 1), linear in the weight. */
export function colorFor(weight: number): string {
  const t = Math.sqrt(Math.min(1, Math.max(0, weight)));
  return `rgb(${mix(17, 139, t)},${mix(24, 92, t)},${mix(51, 246, t)})`;
}

/**
 * Row = the position doing the attending (query), column = the position attended to (key).
 * Cells above the diagonal are the causal mask: future positions, always weight 0.
 */
export function Heatmap({ tokens, matrix, cell, label, showLabels = false, onHover }: HeatmapProps) {
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
      {showLabels &&
        tokens.map((t, i) => (
          <g key={i} fill="#94a3b8" fontFamily="monospace" fontSize={fontSize}>
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
              fill={masked ? "#0b1020" : colorFor(weight)}
              stroke="#080c1a"
              strokeWidth={0.5}
              onMouseEnter={masked ? undefined : () => onHover?.({ query: q, key: k, weight })}
            />
          );
        }),
      )}
    </svg>
  );
}