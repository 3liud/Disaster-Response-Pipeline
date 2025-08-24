from src.processing.clean_text import normalize_text


def test_normalize_basic():
    txt = "Need Water & Food!!! Visit https://help.org now; Email: aid@foo.bar"
    out = normalize_text(txt)
    assert "http" not in out and "@" not in out
    assert "need" in out and "water" in out and "food" in out
