from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report

from src.processing.clean_text import normalize_text

DATA_DIR = Path("data")
MESSAGES_CSV = DATA_DIR / "disaster_messages.csv"
CATEGORIES_CSV = DATA_DIR / "disaster_categories.csv"
MODEL_PATH = Path("models/classifier.pkl")


def _load_eval_split():
    # Simple eval on full data (for demo). Prefer a proper saved split.
    messages = pd.read_csv(MESSAGES_CSV)
    categories = pd.read_csv(CATEGORIES_CSV)

    df = messages.merge(categories, on="id", how="inner")
    cats = df["categories"].str.split(";", expand=True)
    cat_colnames = cats.iloc[0].apply(lambda s: s.split("-")[0]).tolist()
    cats.columns = cat_colnames
    for c in cat_colnames:
        cats[c] = cats[c].str.split("-").str[-1].astype(int).clip(0, 1)
    if "related" in cats.columns:
        cats["related"] = cats["related"].replace(2, 1)

    X = df["message"].astype(str).map(normalize_text)
    Y = cats.astype(int)
    keep = [c for c in Y.columns if Y[c].sum() > 5]
    return X, Y[keep]


def main():
    model = joblib.load(MODEL_PATH)
    X, Y = _load_eval_split()
    preds = model.predict(X)
    print(classification_report(Y, preds, zero_division=0))


if __name__ == "__main__":
    main()
