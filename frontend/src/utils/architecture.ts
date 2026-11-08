import type { TensorStat } from "../types/api";

export interface StageInfo {
  summary: string;
  equation: string;
}

/** Fixed teaching text per stage key. Numbers are never stored here. */
export const STAGE_INFO: Record<string, StageInfo> = {
  tokenizer: {
    summary:
      "Character-level tokenizer: every distinct character is one token, and its ID is its position in the sorted vocabulary.",
    equation: "id_t = sorted_vocab.index(char_t)",
  },
  token_embedding: {
    summary: "A learned lookup table turns each token ID into a vector.",
    equation: "x_tok[t] = E_tok[id_t]",
  },
  position_embedding: {
    summary:
      "A second learned table encodes position, because attention alone has no notion of order. Its output is added to the token embedding.",
    equation: "x[t] = E_tok[id_t] + E_pos[t]",
  },
  blocks: {
    summary:
      "Each block applies multi-head causal self-attention, then a feed-forward network, each with LayerNorm before it and a residual connection around it (pre-norm).",
    equation: "x = x + MHA(LN(x));  x = x + FFN(LN(x))",
  },
  ln_f: {
    summary: "A final LayerNorm normalizes each position's feature vector before the output projection.",
    equation: "LN(x) = gamma * (x - mean) / sqrt(var + eps) + beta",
  },
  lm_head: {
    summary: "A linear layer maps each position's vector to one score (logit) per vocabulary entry.",
    equation: "logits = x W + b",
  },
  softmax: {
    summary:
      "At generation time only the last position's logits are used. Temperature rescales them, top-k optionally masks the rest, softmax turns them into probabilities, and one token is sampled.",
    equation: "p_i = exp(z_i / T) / sum_j exp(z_j / T)",
  },
};

/** Which forward-hook entries belong to a diagram stage. */
export function statsForStage(key: string, stats: TensorStat[]): TensorStat[] {
  if (key === "blocks") return stats.filter((s) => s.name.startsWith("blocks."));
  return stats.filter((s) => s.name === key);
}

export function formatShape(shape: number[]): string {
  return `(${shape.join(", ")})`;
}