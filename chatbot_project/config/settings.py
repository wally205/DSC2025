"""Configuration settings for the chatbot application."""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings configuration."""
    
    # API Keys
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    langchain_api_key: str = Field(default="", env="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="langraph-chatbot", env="LANGCHAIN_PROJECT")
    weather_api_key: str = Field(default="", env="WEATHER_API_KEY")
    
    # Model configuration
    gemini_model: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")  # Use flash for lower quota
    max_output_tokens: int = Field(default=8192, env="MAX_OUTPUT_TOKENS")  # Increased for longer responses
    temperature: float = Field(default=0.1, env="TEMPERATURE")  # Low temperature for factual responses
    
    # Database configuration
    chroma_db_path: str = Field(default="./vectordb", env="CHROMA_DB_PATH")
    chroma_collection_name: str = Field(default="documents", env="CHROMA_COLLECTION_NAME")
    
    # PDF processing
    pdf_data_path: str = Field(default="./data", env="PDF_DATA_PATH")
    chunk_size: int = Field(default=600, env="CHUNK_SIZE")  # Further increased for more comprehensive content
    chunk_overlap: int = Field(default=150, env="CHUNK_OVERLAP")  # Reduced overlap
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")  # Increased limit
    
    # API configuration
    api_host: str = Field(default="127.0.0.1", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    
    # Streamlit configuration
    streamlit_port: int = Field(default=8501, env="STREAMLIT_PORT")
    
    # Logging configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


# Global settings instance
settings = get_settings()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_path() -> Path:
    """Get the data directory path."""
    return get_project_root() / "data"


def get_vectordb_path() -> Path:
    """Get the vector database path."""
    return get_project_root() / "vectordb"


def ensure_directories() -> None:
    """Ensure necessary directories exist."""
    dirs_to_create = [
        get_data_path(),
        get_vectordb_path(),
        get_project_root() / "logs",
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)