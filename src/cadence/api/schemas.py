"""Request/response models for the OpenAI-compatible surface.

Only the fields the gateway actually honours are typed; the rest are accepted
and ignored, because an OpenAI client sends more than this and rejecting it
would defeat the point of being drop-in compatible.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = "qwen"
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = None
    top_p: float | None = None
    user: str | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "cadence"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
