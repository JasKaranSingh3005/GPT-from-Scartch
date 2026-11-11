export type DemoKey =
  | "tokenizer" | "causal-mask" | "attention" | "temperature"
  | "shapes" | "heads" | "ffn" | "layernorm" | "shift";
export const DEMO_KEYS: DemoKey[] = [
  "tokenizer", "causal-mask", "attention", "temperature",
  "shapes", "heads", "ffn", "layernorm", "shift",
];
/** Demos that call the backend and therefore need a model chosen. */
export const BACKEND_DEMOS: DemoKey[] = [
  "tokenizer", "attention", "temperature", "shapes", "heads", "ffn",
];

export interface LearnModule {
  id: string;
  title: string;
  summary: string;
  /** false = listed but not written yet; the page says so instead of showing filler. */
  written: boolean;
  sections: { heading: string; paragraphs: string[] }[];
  equations: { label: string; text: string }[];
  demo?: DemoKey;
  /** For the "shapes" demo: which forward-pass modules to show (a trailing "." matches a prefix). */
  focus?: string[];
}

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
  {
    id: "embeddings",
    title: "Embeddings",
    summary: "How token IDs become vectors.",
    written: true,
    demo: "shapes",
    focus: ["token_embedding"],
    sections: [
      {
        heading: "A lookup table",
        paragraphs: [
          "A token ID is just a label, so the first layer replaces it with a vector. The token embedding is a table with one row per vocabulary entry and n_embd columns. Looking up a token means reading its row. For the default model that is 65 rows of 64 numbers, 4,160 parameters.",
          "The rows start as small random numbers (a normal distribution with standard deviation 0.02) and are learned during training like every other weight. Nothing about them is meaningful at the start, and whatever structure they end up with comes from the next-character objective.",
          "The vector for each position is what flows through the rest of the network, so n_embd is the width of the whole model.",
        ],
      },
      {
        heading: "The demo",
        paragraphs: [
          "Run a forward pass of the selected checkpoint on a prompt. The table shows the real output shape and statistics of the token embedding: a batch of 1, one vector per character, and n_embd numbers per vector.",
        ],
      },
    ],
    equations: [{ label: "Token embedding", text: "x_tok[t] = E_tok[id[t]]    (E_tok has shape vocab_size x n_embd)" }],
  },
  {
    id: "positional-encoding",
    title: "Positional encoding",
    summary: "How the model knows the order of tokens.",
    written: true,
    demo: "shapes",
    focus: ["token_embedding", "position_embedding"],
    sections: [
      {
        heading: "Telling positions apart",
        paragraphs: [
          "Self-attention compares vectors by content, and the token embedding says nothing about where a token sits in the text. Apart from the causal mask, attention has no built-in notion of order, so this model adds an explicit position signal.",
          "It uses a second learned table with one row per position, from 0 up to the context length minus one. The vector for position t is added to the token embedding of the character at position t, so the same character at two places gets two different vectors.",
          "Because the table has exactly block_size rows, the model cannot represent positions beyond it. That is why generation feeds the model only the last block_size characters, and why a longer prompt is cropped.",
          "The original Transformer paper used fixed sine and cosine patterns instead of a learned table. This project learns the table, which is the simpler choice.",
        ],
      },
      {
        heading: "The demo",
        paragraphs: [
          "The position embedding has no batch dimension: its output has one row per position and n_embd columns. Adding it to the token embedding keeps the shape of (1, positions, n_embd).",
        ],
      },
    ],
    equations: [{ label: "Adding position information", text: "x[t] = E_tok[id[t]] + E_pos[t]" }],
  },
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
  {
    id: "multi-head-attention",
    title: "Multi-head attention",
    summary: "Several attention patterns in parallel.",
    written: true,
    demo: "heads",
    sections: [
      {
        heading: "Splitting the width",
        paragraphs: [
          "Instead of one attention pattern per layer, the layer runs several heads side by side. Each head has its own query, key and value layers and works on a smaller vector of size n_embd divided by the number of heads.",
          "The heads' outputs are concatenated back to the full width and mixed by one more linear layer. Concatenating n_head vectors of size head_size gives exactly n_embd numbers again.",
          "More heads therefore does not mean more weights: each head is smaller. The parameter count of this model is the same for 1, 2, 4 or 8 heads, and a test in this project checks that.",
          "In this project the heads run in a plain Python loop over separate modules. That is clearer to read than one fused operation, but slower.",
        ],
      },
      {
        heading: "Do more heads help?",
        paragraphs: [
          "In the Experiment Lab, the number of heads (1, 2, 4 and 8) made no detectable difference to validation loss at this model size. Across two seeds, the spread between head counts (0.018 to 0.027) was the same size as the seed-to-seed variation of a single setting, so a small benefit cannot be ruled in or out. The effect may appear in larger models, but these experiments do not show it.",
        ],
      },
    ],
    equations: [
      { label: "Multi-head attention", text: "MHA(x) = Concat(head_1, ..., head_h) W_O" },
      { label: "Head size", text: "head_size = n_embd / n_head" },
    ],
  },
  {
    id: "feed-forward",
    title: "Feed-forward networks",
    summary: "The per-position computation after attention.",
    written: true,
    demo: "ffn",
    sections: [
      {
        heading: "Think per position",
        paragraphs: [
          "Attention moves information between positions. The feed-forward network then processes each position on its own, with the same weights at every position.",
          "It is two linear layers with a ReLU between them. The hidden layer is four times wider than the model, so a vector of n_embd numbers is expanded to 4 x n_embd, passed through the ReLU, and projected back down to n_embd.",
          "This is where most of a block's parameters live. Per layer it holds 8 x n_embd squared plus 5 x n_embd parameters. For the default model that is 33,088 per layer and 132,352 for all four layers, which is the feed-forward line you see in the Architecture Explorer.",
        ],
      },
      {
        heading: "The demo",
        paragraphs: [
          "The table lists the real tensors of the first block's feed-forward network, read from the selected checkpoint, with their shapes and parameter counts.",
        ],
      },
    ],
    equations: [{ label: "Feed-forward network", text: "FFN(x) = W2 ReLU(W1 x + b1) + b2" }],
  },
  {
    id: "residual-connections",
    title: "Residual connections",
    summary: "Why blocks add their input back.",
    written: true,
    demo: "shapes",
    focus: ["token_embedding", "blocks."],
    sections: [
      {
        heading: "Adding instead of replacing",
        paragraphs: [
          "Each sublayer in a block, the attention and the feed-forward network, does not replace its input. Its output is added to the input: x becomes x plus the sublayer's result.",
          "This gives gradients a direct path back through every block during training, and it lets a block start by doing little and learn only a correction. A test in this project checks the property: if both sublayers output zero, the block returns its input unchanged.",
          "Because the output is added to the input, the shape never changes inside a block. The vectors that go in and come out always have n_embd numbers per position.",
        ],
      },
      {
        heading: "The demo",
        paragraphs: [
          "The forward-pass table shows the real output of every block for your prompt. The shape is the same after each one, and the statistics show how the size of the activations changes as they pass through the stack.",
        ],
      },
    ],
    equations: [
      { label: "One block (pre-norm)", text: "x = x + MHA(LN(x));  x = x + FFN(LN(x))" },
    ],
  },
  {
    id: "layernorm",
    title: "LayerNorm",
    summary: "Keeping activations in a workable range.",
    written: true,
    demo: "layernorm",
    sections: [
      {
        heading: "Normalising one position",
        paragraphs: [
          "Layer normalisation rescales the vector at each position on its own: it subtracts the mean of the n_embd numbers, divides by their standard deviation (with a tiny constant added for safety), and then applies a learned scale and a learned shift.",
          "The learned scale starts at 1 and the shift at 0, so at the start it only standardises. During training the model can change both.",
          "This model normalises before each sublayer rather than after it. There is a LayerNorm in front of the attention, one in front of the feed-forward network, and a final one before the output layer. For the default model that is 4 x 4 x 64 + 128 = 1,152 parameters in total.",
        ],
      },
      {
        heading: "The demo",
        paragraphs: [
          "Change the mean and spread of a small vector and watch the result. It uses a scale of 1 and a shift of 0, as a freshly created LayerNorm does, and it is computed in your browser, not by the model.",
        ],
      },
    ],
    equations: [{ label: "Layer normalisation", text: "LN(x) = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta" }],
  },
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
  {
    id: "training",
    title: "Training and backpropagation",
    summary: "How the weights are adjusted.",
    written: true,
    demo: "shift",
    sections: [
      {
        heading: "Inputs and targets",
        paragraphs: [
          "Each training step takes random windows of the training text. The target is the same window shifted one character to the right, so at every position the model is asked to predict the character that really comes next. One window of length T therefore gives T separate prediction problems.",
          "The last tenth of the text is held back and never trained on. Its loss, the validation loss, shows how well the model handles text it has not seen, and a growing gap between training and validation loss is the sign of overfitting.",
        ],
      },
      {
        heading: "Backpropagation and the update",
        paragraphs: [
          "The loss is a single number computed from all the weights. Backpropagation applies the chain rule backwards through the network to find how much each weight contributed, which gives the gradient of the loss with respect to every weight.",
          "The optimiser, AdamW, uses those gradients to move every weight a small step that should lower the loss. In this project the learning rate is constant (there is no schedule), weight decay is 0.01, and the gradient norm is clipped to 1.0 so one unusual batch cannot cause a huge update.",
          "Every evaluation interval the loss is measured over several batches for both the training and the validation text, with dropout switched off. The Training Lab plots those measurements live.",
        ],
      },
    ],
    equations: [
      { label: "One AdamW step (per weight w, gradient g)", text: "w <- w - lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * w )" },
      { label: "Targets are the input shifted by one", text: "x = text[i : i + T],   y = text[i + 1 : i + T + 1]" },
    ],
  },
];

export const FIRST_MODULE_ID = "language-modelling";