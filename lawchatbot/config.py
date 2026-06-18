from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    weaviate_url: str
    weaviate_api_key: str
    openai_api_key: Optional[str] = None
    weaviate_class: str = "JustiaFederalCases"
    text_key: str = "text"
    metadata_attributes: list[str] = ["text"]
    semantic_k: int = 10
    bm25_k: int = 5
    alpha: float = 0.5

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

def load_config() -> AppConfig:
    """
    Loads configuration from environment variables or .env file.
    """
    return AppConfig()