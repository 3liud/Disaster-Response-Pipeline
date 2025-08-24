from pathlib import Path
from typing import Dict, Optional, List, Set
import json
import base64
import io

import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.schemas.message import MessageIn
from src.schemas.prediction import PredictionOut
from src.processing.clean_text import normalize_text

# Optional: wordcloud; degrade gracefully if missing
try:
    from wordcloud import WordCloud, STOPWORDS  # type: ignore
except Exception:  # pragma: no cover
    WordCloud = None
    STOPWORDS = set()

app = FastAPI(title="Disaster Response API", version="0.3.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

_MODEL_PATH = Path("models/classifier.pkl")
_LABELS_PATH = Path("models/label_names.json")
_FREQS_PATH = Path("models/global_token_freqs.json")

_model = None
_label_names: Optional[List[str]] = None
_global_freqs: Optional[Dict[str, float]] = None

NEED_LABELS: Set[str] = {
    "water",
    "food",
    "shelter",
    "medical_help",
    "medical_products",
    "sanitation",
    "clothing",
    "money",
    "search_and_rescue",
    "transport",
    "electricity",
    "tools",
    "hospitals",
    "aid_centers",
    "missing_people",
    "refugees",
    "request",
    "aid_related",
}
MIN_SCORE = 0.30  # table threshold


def _load_model():
    global _model
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {_MODEL_PATH.resolve()}")
        _model = joblib.load(_MODEL_PATH)
    return _model


def _load_label_names() -> List[str]:
    global _label_names
    if _label_names is not None:
        return _label_names
    if _LABELS_PATH.exists():
        _label_names = list(json.loads(_LABELS_PATH.read_text()))
        return _label_names
    m = _load_model()
    if hasattr(m, "label_names"):
        _label_names = list(getattr(m, "label_names"))
        return _label_names
    if hasattr(m, "named_steps") and "clf" in m.named_steps:
        clf = m.named_steps["clf"]
        if hasattr(clf, "label_names"):
            _label_names = list(getattr(clf, "label_names"))
            return _label_names
    return []


def _load_global_freqs() -> Optional[Dict[str, float]]:
    global _global_freqs
    if _global_freqs is not None:
        return _global_freqs
    if _FREQS_PATH.exists():
        _global_freqs = json.loads(_FREQS_PATH.read_text())
        return _global_freqs
    return None


def _positive_probs(model, text: str) -> List[float]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])
        if isinstance(probs, list):
            return [float(a[0][1]) for a in probs]
        return [float(v) for v in probs[0]]
    if hasattr(model, "decision_function"):
        scores = model.decision_function([text])
        if getattr(scores, "ndim", 1) == 1:
            return [float(scores[0])]
        return [float(v) for v in scores[0]]
    preds = model.predict([text])[0]
    return [float(v) for v in (preds if hasattr(preds, "__len__") else [preds])]


def _needs_from_probs(names: List[str], pos: List[float]) -> Dict[str, float]:
    needs = {}
    for i, name in enumerate(names or [f"label_{i}" for i in range(len(pos))]):
        if name in NEED_LABELS and pos[i] >= MIN_SCORE:
            needs[name] = pos[i]
    return needs


def _tokens_to_highlight_from_labels(labels: List[str]) -> Set[str]:
    # Highlight literal label tokens (split underscores). Add a few synonyms for clarity.
    base = set()
    SYN = {
        "medical_help": {"medical", "help", "doctor", "medicine"},
        "medical_products": {"medical", "medicine", "medicines", "drugs"},
        "search_and_rescue": {"search", "rescue"},
        "missing_people": {"missing", "people"},
        "aid_related": {"aid"},
        "aid_centers": {"aid", "center", "centers"},
        "electricity": {"electricity", "power"},
        "water": {"water"},
        "food": {"food", "hunger", "hungry", "meal", "meals"},
        "shelter": {"shelter", "tents"},
        "money": {"money", "cash", "funds"},
        "transport": {"transport", "truck", "trucks"},
        "sanitation": {"sanitation", "toilet", "latrine"},
        "clothing": {"clothing", "clothes"},
        "refugees": {"refugees"},
        "hospitals": {"hospital", "hospitals"},
        "tools": {"tools"},
    }
    for lab in labels:
        base |= set(lab.split("_"))
        base |= SYN.get(lab, set())
    # Don’t highlight stopwords
    return {t for t in base if t and t not in STOPWORDS}


def _global_wordcloud_data_uri(highlight_words: Set[str]) -> Optional[str]:
    if WordCloud is None:
        return None
    freqs = _load_global_freqs()
    if not freqs:
        return None

    # Boost highlighted words so they’re larger, and color them differently
    boost = 3.0 if highlight_words else 1.0
    boosted: Dict[str, float] = {}
    for tok, f in freqs.items():
        if tok in highlight_words:
            boosted[tok] = float(f) * boost
        else:
            boosted[tok] = float(f)

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        # red-ish for highlights, muted grey/blue for others
        if word in highlight_words:
            return "hsl(5, 90%, 40%)"
        return "hsl(210, 15%, 28%)"

    wc = WordCloud(
        width=900, height=500, background_color="white", color_func=color_func
    )
    img = wc.generate_from_frequencies(boosted).to_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# --------- Routes ---------


class HealthOut(BaseModel):
    status: str


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    # Default: dataset cloud with no highlights
    wc_uri = _global_wordcloud_data_uri(set()) or None
    return templates.TemplateResponse(
        "go.html",
        {"request": request, "ids": [], "graphJSON": "[]", "wordcloud_uri": wc_uri},
    )


@app.get("/go", response_class=HTMLResponse)
def go(request: Request, query: Optional[str] = None):
    ctx = {"request": request, "ids": [], "graphJSON": "[]"}
    # Always render the dataset cloud; maybe highlight depending on query
    highlight: Set[str] = set()
    needs_table: Dict[str, float] = {}

    if query:
        model = _load_model()
        names = _load_label_names()
        pos = _positive_probs(model, query)
        needs_table = _needs_from_probs(names, pos)
        if needs_table:
            highlight = _tokens_to_highlight_from_labels(list(needs_table.keys()))
        ctx["input_text"] = query

    ctx["prediction"] = needs_table if needs_table else None
    ctx["wordcloud_uri"] = _global_wordcloud_data_uri(highlight) or None
    return templates.TemplateResponse("go.html", ctx)


@app.post("/predict", response_model=PredictionOut)
def predict(payload: MessageIn) -> PredictionOut:
    model = _load_model()
    names = _load_label_names()
    pos = _positive_probs(model, payload.text)
    if not names:
        names = [f"label_{i}" for i in range(len(pos))]
    labels = {str(names[i]): float(p) for i, p in enumerate(pos[: len(names)])}
    return PredictionOut(labels=labels, top_k=None)
