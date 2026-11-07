import torch

from models.bigram import BigramLanguageModel


def test_forward_shapes():
    model = BigramLanguageModel(vocab_size=10)
    idx = torch.randint(10, (3, 7))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 7, 10)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)


def test_forward_without_targets_has_no_loss():
    model = BigramLanguageModel(vocab_size=10)
    _, loss = model(torch.randint(10, (2, 4)))
    assert loss is None


def test_generate_length_and_range():
    model = BigramLanguageModel(vocab_size=10)
    out = model.generate(torch.zeros((2, 1), dtype=torch.long), max_new_tokens=15, top_k=3)
    assert out.shape == (2, 16)
    assert out.min() >= 0 and out.max() < 10


def test_training_reduces_loss():
    torch.manual_seed(0)
    model = BigramLanguageModel(vocab_size=5)
    data = torch.tensor([0, 1, 2, 3, 4] * 20)
    x, y = data[:-1].unsqueeze(0), data[1:].unsqueeze(0)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    first = model(x, y)[1].item()
    for _ in range(50):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first