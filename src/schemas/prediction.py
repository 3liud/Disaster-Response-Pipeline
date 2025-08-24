from typing import Dict, List, Tuple
from pydantic import BaseModel


class PredictionOut(BaseModel):
    labels: Dict[str, float]
    top_k: List[Tuple[str, float]] | None = None
