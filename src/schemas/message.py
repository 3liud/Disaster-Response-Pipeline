from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    text: str = Field(..., min_length=1, description="Free-text disaster message")
    lang: str | None = Field(default=None, description="Optional ISO language hint")
