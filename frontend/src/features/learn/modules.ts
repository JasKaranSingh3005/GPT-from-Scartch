export type DemoKey = "tokenizer" | "causal-mask" | "attention" | "temperature";
export const DEMO_KEYS: DemoKey[] = ["tokenizer", "causal-mask", "attention", "temperature"];
/** Demos that call the backend and therefore need a model chosen. */
export const BACKEND_DEMOS: DemoKey[] = ["tokenizer", "attention", "temperature"];

export interface LearnModule {
  id: string;
  title: string;
  summary: string;
  /** false = listed but not written yet; the page says so instead of showing filler. */
  written: boolean;
  sections: { heading: string; paragraphs: string[] }[];
  equations: { label: string; text: string }[];
  demo?: DemoKey;
}

const todo = (id: string, title: string, summary: string): LearnModule => ({
  id, title, summary, written: false, sections: [], equations: [],
});

export const MODULES: LearnModule[] = [
  {
    id: "language-modelling",
    title: "What is a language model?",
    summary: "Predicting what comes next, and how that is measured.",
    written: true,
    sections: [
      {
        heading: "The task",
        paragraphs: [
          "A language model reads the text so far and outputs a probability for every possible next token. Everything else, including generating text, is built from that one ability.",
          "Generating is just repetition: predict a distribution, pick one token from it, append it to the text, and predict again.",
        ],
      },
      {
        heading: "How training measures it",
        paragraphs: [
          "During training the model sees real text. For each position it is scored on how much probability it gave to the character that actually came next. The loss is the average of the negative logarithm of that probability, so giving the true character more probability lowers the loss.",
          "A model that knows nothing gives all 65 characters of the vocabulary the same probability, which is a loss of ln 65, about 4.17. A freshly initialised model in this project starts close to that value, which you can see as the first point of any run in the Training Lab.",
          "The Experiment Lab shows how far real runs get below that starting value, and how that depends on model size, context length and the other settings.",
        ],
      },
    ],
    equations: [
      { label: "Training objective (average negative log-likelihood)", text: "L = -(1/N) * sum_t log p(x[t+1] | x[1..t])" },
      { label: "A model with no knowledge, 65 characters", text: "L = ln(65) = 4.17" },
    ],
  },
  {
    id: "tokenization",
    title: "Tokenization",
    summary: "Turning text into the integers a network can use.",
    written: true,
    demo: "tokenizer",
    sections: [
      {
        heading: "Characters as tokens",
        paragraphs: [
          "A neural network works on numbers, so text must be turned into integers first. This project uses the simplest scheme: every distinct character in the training text is one token, and its ID is its position in the sorted list of those characters. The Tiny Shakespeare vocabulary has 65 of them.",
          "This is not BPE or any subword scheme. There is no merging of frequent letter groups into larger tokens, so one token is always exactly one character.",
          "The trade-off is that sequences are long: a context window of 64 tokens is only 64 characters. In return the vocabulary is tiny and every word is built from known pieces.",
          "A model can only handle characters that were in its vocabulary. Text containing anything else is rejected with an error that names the characters, because silently dropping them would change what the model sees.",
        ],
      },
    ],
    equations: [{ label: "Encoding", text: "id[t] = sorted_vocabulary.index(char[t])" }],
  },
  todo("embeddings", "Embeddings", "How token IDs become vectors."),
  todo("positional-encoding", "Positional encoding", "How the model knows the order of tokens."),
  {
    id: "self-attention",
    title: "Self-attention",
    summary: "How each position decides which earlier positions to read.",
    written: true,
    demo: "attention",
    sections: [
      {
        heading: "Queries, keys and values",
        paragraphs: [
          "Each position's vector is turned into three new vectors by three learned linear layers: a query (what this position is looking for), a key (what it offers to others) and a value (the information it passes on).",
          "The score between position i and position j is the dot product of i's query with j's key, divided by the square root of the head size. The division keeps the scores at a size where the softmax does not collapse onto a single position.",
          "A softmax over the scores turns them into weights that sum to 1, and the output for position i is the weighted sum of the value vectors. The heatmap below shows those weights from a real forward pass of the model you choose.",
        ],
      },
      {
        heading: "Reading the picture",
        paragraphs: [
          "Row i of the heatmap is position i choosing how to spread its attention over positions up to and including itself. The hatched upper triangle is the causal mask, covered in the next module.",
          "Different heads in one layer can learn different patterns, but a single prompt cannot tell you what a head means, so treat any story about one head as a hypothesis.",
        ],
      },
    ],
    equations: [
      { label: "Scaled dot-product attention", text: "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + M) V" },
    ],
  },
  {
    id: "causal-masking",
    title: "Causal masking",
    summary: "Why a position may not look at the future.",
    written: true,
    demo: "causal-mask",
    sections: [
      {
        heading: "No peeking",
        paragraphs: [
          "The model is trained to predict the next character. If position t could read position t+1, it could simply copy the answer and learn nothing useful.",
          "The mask is added to the scores before the softmax. Allowed positions get 0 and future positions get minus infinity, so after the softmax their weights are exactly 0, and each row still sums to 1 over the visible positions.",
          "The simplest possible version gives every visible position the same weight, which makes each output the running average of the inputs so far. Attention keeps this lower-triangular structure but learns the weights instead.",
          "This project tests the property directly: changing the later tokens of a sequence must not change the outputs at earlier positions.",
        ],
      },
    ],
    equations: [
      { label: "The mask", text: "M[i][j] = 0 if j <= i, otherwise -infinity" },
      { label: "Equal weights over the past", text: "y[i] = (1 / (i + 1)) * sum over j <= i of x[j]" },
    ],
  },
  todo("multi-head-attention", "Multi-head attention", "Several attention patterns in parallel."),
  todo("feed-forward", "Feed-forward networks", "The per-position computation after attention."),
  todo("residual-connections", "Residual connections", "Why blocks add their input back."),
  todo("layernorm", "LayerNorm", "Keeping activations in a workable range."),
  {
    id: "autoregressive-generation",
    title: "Autoregressive generation",
    summary: "From probabilities to text: temperature, top-k and sampling.",
    written: true,
    demo: "temperature",
    sections: [
      {
        heading: "The loop",
        paragraphs: [
          "To generate, the model is given the text so far (cropped to the last block-size characters, because that is all its position table covers). It produces scores, called logits, for every character at the last position.",
          "Temperature divides the logits before the softmax. Below 1 the distribution becomes sharper and the text safer and more repetitive. Above 1 it flattens, which gives more surprising text and more mistakes.",
          "Top-k keeps only the k highest scores and sets every other probability to exactly 0, so unlikely characters can never be picked. One character is then sampled, appended to the text, and the loop repeats.",
          "With the same seed on the same kind of device (CPU or GPU), the same settings give the same text. Different device types draw different random numbers from the same seed.",
        ],
      },
      {
        heading: "The demo",
        paragraphs: [
          "The bars below are the model's real probabilities for the next character after your prompt, computed with the same function the sampler uses. Nothing is sampled here, so moving the sliders only changes the shape of the distribution.",
        ],
      },
    ],
    equations: [
      { label: "Temperature softmax", text: "p[i] = exp(z[i] / T) / sum over j of exp(z[j] / T)" },
    ],
  },
  todo("training", "Training and backpropagation", "How the weights are adjusted."),
];

export const FIRST_MODULE_ID = "language-modelling";