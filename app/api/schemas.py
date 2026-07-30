from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessCheck(BaseModel):
    name: str
    status: Literal["ok", "not_configured", "unavailable"]
    detail: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: list[ReadinessCheck]
