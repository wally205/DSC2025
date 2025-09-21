"""Root package initialization."""

# Package metadata
__version__ = "1.0.0"
__author__ = "LangGraph Chatbot Team"
__description__ = "A professional chatbot built with LangGraph, ChromaDB, and FastAPI"

# Main imports
from config import configure_logging, get_logger
from graph import ChatbotGraphBuilder

# Configure logging on import
configure_logging()

__all__ = [
    "ChatbotGraphBuilder",
    "configure_logging",
    "get_logger",
]