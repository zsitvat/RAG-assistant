from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_backend: Literal["ollama", "dummy"] = "ollama"
    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"

    api_base_url: str = "http://api:8000"
    redis_url: str = "redis://redis:6379/0"

    langfuse_enabled: bool = True
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
