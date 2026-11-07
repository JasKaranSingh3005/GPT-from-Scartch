from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from app.components.common import discover_checkpoints
from app.components.tokenizer_page import SRC_CUSTOM, chip_html

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
DATASET = ROOT / "data" / "input.txt"


def test_discover_checkpoints_finds_only_best_and_last(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "best.pt").write_bytes(b"x")
    (tmp_path / "a" / "last.pt").write_bytes(b"x")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "notes.txt").write_text("x")
    (tmp_path / "stray.pt").write_bytes(b"x")
    found = discover_checkpoints(tmp_path)
    assert [(c.run, c.kind) for c in found] == [("a", "best"), ("a", "last")]
    assert found[0].label == "a (best)"


def test_discover_checkpoints_missing_dir(tmp_path):
    assert discover_checkpoints(tmp_path / "nope") == []


def test_chip_html_escapes_special_characters():
    assert "&lt;" in chip_html("<", 3)
    assert "&amp;" in chip_html("&", 4)


def test_chip_html_shows_whitespace_symbols():
    assert "↵" in chip_html("\n", 0)
    assert "␣" in chip_html(" ", 1)


def test_home_page_renders():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


@pytest.mark.skipif(not DATASET.is_file(), reason="dataset not downloaded")
def test_tokenizer_page_renders_in_both_modes():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.radio(key="nav_page").set_value("Tokenizer Explorer").run()
    assert not at.exception and not at.error
    at.radio(key="tok_source").set_value(SRC_CUSTOM).run()
    at.text_area(key="tok_text").input("hello hello").run()
    assert not at.exception and not at.error