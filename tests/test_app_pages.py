from pathlib import Path
import pytest
import torch
from streamlit.testing.v1 import AppTest
from app.components.attention_page import attention_for_prompt
from app.components.common import discover_checkpoints
from app.components.generator_page import next_char_distribution
from models.gpt import GPT
from models.sampling import encode_prompt, next_token_probs
from models.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
TOKENIZER = CharTokenizer.from_text("abcdef\n")
V, BS = TOKENIZER.vocab_size, 8


def tiny_gpt() -> GPT:
    torch.manual_seed(0)
    return GPT(vocab_size=V, n_embd=16, n_head=2, n_layer=2, block_size=BS, dropout=0.0).eval()


def test_attention_for_prompt_shapes_and_causality():
    result = attention_for_prompt(tiny_gpt(), TOKENIZER, "abcab")
    assert result.chars == list("abcab") and not result.cropped
    assert len(result.weights) == 2
    w = result.weights[0]
    assert w.shape == (2, 5, 5)
    assert torch.allclose(w.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert torch.all(torch.triu(w, diagonal=1) == 0)  # nothing attends to the future


def test_attention_prompt_is_cropped_to_block_size():
    result = attention_for_prompt(tiny_gpt(), TOKENIZER, "abcdefabcdef")  # 12 > 8
    assert result.cropped and len(result.chars) == BS
    assert "".join(result.chars) == "abcdefabcdef"[-BS:]


def test_attention_rejects_empty_and_unknown_prompts():
    with pytest.raises(ValueError):
        attention_for_prompt(tiny_gpt(), TOKENIZER, "")
    with pytest.raises(ValueError):
        attention_for_prompt(tiny_gpt(), TOKENIZER, "xyz")


def test_next_token_probs_sums_to_one_and_respects_top_k():
    logits = torch.tensor([[5.0, 4.0, 0.0, 0.0, -1.0]])
    probs = next_token_probs(logits, temperature=0.7, top_k=2)
    assert torch.allclose(probs.sum(-1), torch.ones(1))
    assert torch.all(probs[0, 2:] == 0) and torch.all(probs[0, :2] > 0)


def test_higher_temperature_flattens_the_distribution():
    logits = torch.tensor([[3.0, 1.0, 0.0]])
    cold = next_token_probs(logits, temperature=0.5)
    hot = next_token_probs(logits, temperature=2.0)
    assert cold.max() > hot.max()


def test_encode_prompt_empty_uses_newline():
    assert encode_prompt(TOKENIZER, "") == TOKENIZER.encode("\n")
    assert encode_prompt(TOKENIZER, "ab") == TOKENIZER.encode("ab")


def test_next_char_distribution_is_a_distribution():
    probs = next_char_distribution(tiny_gpt(), TOKENIZER, "abc", 1.0, 3)
    assert probs.shape == (V,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (probs > 0).sum() <= 3 + 1  # top-3 (a rare exact tie may keep one more)


def test_next_char_distribution_crops_long_prompts():
    model = tiny_gpt()
    long = next_char_distribution(model, TOKENIZER, "abcdefabcdef", 1.0, None)
    short = next_char_distribution(model, TOKENIZER, "abcdefabcdef"[-BS:], 1.0, None)
    assert torch.allclose(long, short, atol=1e-6)


HAVE_GPT = any(c.run != "" for c in discover_checkpoints())


@pytest.mark.skipif(not HAVE_GPT, reason="no trained checkpoints in outputs/runs")
def test_attention_and_generator_pages_render():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key="nav_page").set_value("Attention Visualizer").run()
    assert not at.exception
    at.radio(key="nav_page").set_value("Text Generator").run()
    assert not at.exception
    at.slider(key="gen_len").set_value(50)
    at.button(key="gen_button").click().run()
    assert not at.exception
    assert len(at.session_state["gen_result"]["text"]) >= 50