from typing import Iterable, List

# Placeholder: if your model pipeline already contains a TfidfVectorizer,
# you won't need this. Kept here for future custom features.


def identity(texts: Iterable[str]) -> List[str]:
    """No-op featurizer (useful with sklearn Pipeline where vectorizer is inside)."""
    return list(texts)
