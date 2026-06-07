"""Configuration for zzk harness."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_timeout_seconds: float = 30.0
    knowledge_provider: Literal["auto", "http", "sqlite", "mock"] = "auto"
    knowledge_base_url: str = "http://127.0.0.1:8000/knowledge"
    knowledge_api_key: str = ""
    knowledge_timeout_seconds: float = 10.0
    knowledge_sqlite_path: str = ""
    search_provider: Literal["duckduckgo", "serpapi"] = "duckduckgo"
    search_api_key: str = ""
    search_timeout_seconds: float = 10.0
    prompt_version: Literal["v1", "v2"] = "v2"
    memory_compress_mode: Literal["deterministic", "llm"] = "deterministic"
    memory_summary_max_tokens: int = 512
    enable_user_skills: bool = False
    app_name: str = "zzk"

    model_config = SettingsConfigDict(
        env_prefix="ZZK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
