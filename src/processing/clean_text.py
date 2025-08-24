import re
from typing import Iterable


_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+?@[\w\.-]+\.\w+\b")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Basic, language-agnostic cleaning."""
    t = text.strip().lower()
    t = _URL_RE.sub(" ", t)
    t = _EMAIL_RE.sub(" ", t)
    t = _NON_WORD_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip()


def batch_normalize_text(texts: Iterable[str]) -> list[str]:
    return [normalize_text(t) for t in texts]
