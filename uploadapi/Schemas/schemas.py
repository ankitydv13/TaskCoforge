from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    page_content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

class LLMResponse(BaseModel):
    query : str
    answer : str
    chunks : list[str]
    scores : list[float]

class ChatSessionResponse(BaseModel):
    session_id: str

upload_progress = {}