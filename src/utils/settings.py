import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL", "postgresql+psycopg2://user:password@localhost:5432/disaster"
    )
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    models_dir: str = os.getenv("MODELS_DIR", "models")


settings = Settings()
