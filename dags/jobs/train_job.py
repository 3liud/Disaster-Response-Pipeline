from dagster import job, op, Out
from pathlib import Path

from src.modeling.train import load_and_prepare, build_pipeline
import joblib


@op(out={"X": Out(), "Y": Out()})
def load_data_op():
    X, Y = load_and_prepare()
    return X, Y


@op
def train_op(context, X, Y):
    pipe = build_pipeline()
    context.log.info(f"Training on {len(X)} samples, {Y.shape[1]} labels")
    pipe.fit(X, Y)
    return pipe


@op
def persist_model_op(model):
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "classifier.pkl"
    joblib.dump(model, path)
    return str(path)


@job
def train_job():
    X, Y = load_data_op()
    model = train_op(X, Y)
    persist_model_op(model)
