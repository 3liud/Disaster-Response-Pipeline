from pathlib import Path
from typing import Tuple, List
import json

import joblib
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from src.processing.clean_text import normalize_text

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MESSAGES_CSV = DATA_DIR / "disaster_messages.csv"
CATEGORIES_CSV = DATA_DIR / "disaster_categories.csv"
MODEL_PATH = MODELS_DIR / "classifier.pkl"
LABELS_PATH = MODELS_DIR / "label_names.json"
GLOBAL_FREQS_PATH = MODELS_DIR / "global_token_freqs.json"


def load_and_prepare() -> Tuple[pd.Series, pd.DataFrame]:
    messages = pd.read_csv(MESSAGES_CSV)
    categories = pd.read_csv(CATEGORIES_CSV)
    df = messages.merge(categories, on="id", how="inner")

    cats = df["categories"].str.split(";", expand=True)
    first_row = cats.iloc[0]
    cat_colnames: List[str] = first_row.apply(lambda s: s.split("-")[0]).tolist()
    cats.columns = cat_colnames
    for c in cat_colnames:
        cats[c] = cats[c].str.split("-").str[-1].astype(int).clip(0, 1)
    if "related" in cats.columns:
        cats["related"] = cats["related"].replace(2, 1)

    X = df["message"].astype(str).map(normalize_text)
    Y = cats.astype(int)

    keep = [c for c in Y.columns if Y[c].sum() > 5]
    if not keep:
        raise ValueError("All labels are too rare (sum <= 5).")
    Y = Y[keep]
    return X, Y


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=200, solver="liblinear", class_weight="balanced"
                    )
                ),
            ),
        ]
    )


def _attach_label_names(pipe: Pipeline, label_names: List[str]) -> None:
    try:
        pipe.label_names = list(label_names)
    except Exception:
        pass
    try:
        if "clf" in pipe.named_steps:
            pipe.named_steps["clf"].label_names = list(label_names)
    except Exception:
        pass


def _compute_and_save_global_freqs(all_texts: pd.Series) -> None:
    # Unigram counts across the WHOLE dataset; remove English stopwords; drop very rare tokens
    cv = CountVectorizer(stop_words="english", min_df=5)
    Xc = cv.fit_transform(all_texts)  # shape: (N_docs, V)
    freqs = Xc.sum(axis=0).A1  # total counts per token
    vocab = cv.get_feature_names_out()
    data = {tok: int(freqs[i]) for i, tok in enumerate(vocab) if freqs[i] > 0}
    GLOBAL_FREQS_PATH.write_text(json.dumps(data, ensure_ascii=False))


def main() -> None:
    X, Y = load_and_prepare()

    # Save global token frequencies (for the dataset-level word cloud)
    _compute_and_save_global_freqs(X)

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(msss.split(X, Y))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    label_names = list(y_train.columns)
    _attach_label_names(pipe, label_names)
    LABELS_PATH.write_text(json.dumps(label_names, ensure_ascii=False))

    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.resolve()}")
    print(f"Saved label names to {LABELS_PATH.resolve()}")
    print(f"Saved global freqs to {GLOBAL_FREQS_PATH.resolve()}")


if __name__ == "__main__":
    main()
