"""FastAPI main application."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import configure_logging, get_logger, settings, ensure_directories
from .routes import router

# Configure logging
configure_logging()

# Ensure necessary directories exist
ensure_directories()

# Initialize logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LangGraph Chatbot API",
    description="A professional chatbot API built with LangGraph, ChromaDB, and FastAPI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["chatbot"])


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting LangGraph Chatbot API")
    logger.info("API documentation available at http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down LangGraph Chatbot API")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LangGraph Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    run_server()