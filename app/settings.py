# app/settings.py — configuration and environment setup

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application configuration values loaded from environment variables."""
    data_dir: str = os.getenv("DATA_DIR", ".data")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text")
    llm_model: str = os.getenv("LLM_MODEL", "llama3")


settings = Settings()
