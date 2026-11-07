from models.tokenizer import CharTokenizer

CHARS = CharTokenizer.from_text("abcdef\n").chars  # 7 characters
GPT_KWARGS = dict(vocab_size=7, n_embd=16, n_head=2, n_layer=2, block_size=8, dropout=0.0)