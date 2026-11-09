/**
 * The logo: a causal attention mask. Each token can only look at itself and earlier
 * tokens, so the grid is lower-triangular; the diagonal (a token attending to itself)
 * carries the secondary color. Colors come from theme tokens, so it works in both themes.
 */
const WEIGHTS: number[][] = [
  [1],
  [0.35, 1],
  [0.2, 0.55, 1],
  [0.6, 0.15, 0.4, 1],
];

const CELL = 4.5;
const STEP = 5.5;
const INSET = 1.25;

export function BrandMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden focusable="false">
      {WEIGHTS.flatMap((row, i) =>
        row.map((weight, j) => (
          <rect
            key={`${i}-${j}`}
            x={INSET + j * STEP}
            y={INSET + i * STEP}
            width={CELL}
            height={CELL}
            rx={1}
            className={i === j ? "fill-cyan" : "fill-violet-soft"}
            fillOpacity={weight}
          />
        )),
      )}
    </svg>
  );
}
