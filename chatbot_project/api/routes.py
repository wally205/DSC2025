"""FastAPI routes for the chatbot API."""

from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from config import get_logger
from graph import ChatbotGraphBuilder
from ingest import DataIngester

# Initialize logger
logger = get_logger(__name__)

# Initialize chatbot
chatbot = ChatbotGraphBuilder()

# Initialize data ingester
data_ingester = DataIngester()

# Create router
router = APIRouter()


# Request/Response models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Bot response")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Confidence score")
    response_type: str = Field(..., description="Type of response")
    sources: List[Dict[str, str]] = Field(default=[], description="Source documents")
    timestamp: str = Field(..., description="Response timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")


class IngestionRequest(BaseModel):
    """Data ingestion request model."""
    clear_existing: bool = Field(default=False, description="Clear existing data before ingestion")


class IngestionResponse(BaseModel):
    """Data ingestion response model."""
    success: bool = Field(..., description="Ingestion success status")
    documents_processed: int = Field(..., description="Number of documents processed")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")


# API Routes
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a chat message and return response.
    
    Args:
        request: Chat request containing user message
        
    Returns:
        Chat response with bot reply and metadata
    """
    try:
        logger.info("Processing chat request", message=request.message[:100])
        
        # Process query through chatbot pipeline
        result = chatbot.process_query(
            user_query=request.message,
            session_id=request.session_id
        )
        
        # Create response
        response = ChatResponse(
            response=result.get("response", "Xin lỗi, tôi không thể xử lý yêu cầu của bạn."),
            intent=result.get("intent", "unknown"),
            confidence=result.get("confidence", 0.0),
            response_type=result.get("response_type", "unknown"),
            sources=result.get("sources", []),
            timestamp=result.get("timestamp", datetime.utcnow().isoformat()),
            session_id=request.session_id
        )
        
        logger.info(
            "Chat request processed successfully",
            intent=response.intent,
            confidence=response.confidence
        )
        
        return response
        
    except Exception as e:
        logger.error("Chat request failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Đã xảy ra lỗi khi xử lý tin nhắn: {str(e)}"
        )


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_data(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
) -> IngestionResponse:
    """
    Ingest PDF data into the vector database.
    
    Args:
        request: Ingestion request
        background_tasks: Background task handler
        
    Returns:
        Ingestion response with status
    """
    try:
        logger.info("Starting data ingestion", clear_existing=request.clear_existing)
        
        # Run ingestion in background
        def run_ingestion():
            try:
                documents_count = data_ingester.ingest_from_directory(
                    clear_existing=request.clear_existing
                )
                logger.info("Data ingestion completed", documents_count=documents_count)
            except Exception as e:
                logger.error("Background ingestion failed", error=str(e))
        
        background_tasks.add_task(run_ingestion)
        
        return IngestionResponse(
            success=True,
            documents_processed=0,  # Will be updated in background
            message="Data ingestion started in background"
        )
        
    except Exception as e:
        logger.error("Data ingestion failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi thực hiện ingestion: {str(e)}"
        )


@router.get("/ingest/status")
async def get_ingestion_status() -> Dict[str, Any]:
    """
    Get current ingestion status.
    
    Returns:
        Ingestion status information
    """
    try:
        status = data_ingester.get_ingestion_status()
        logger.info("Retrieved ingestion status", status=status)
        return {
            "success": True,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get ingestion status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy trạng thái ingestion: {str(e)}"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        Health status information
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@router.get("/graph/visualization")
async def get_graph_visualization() -> Dict[str, str]:
    """
    Get chatbot graph visualization.
    
    Returns:
        Graph structure visualization
    """
    try:
        visualization = chatbot.get_graph_visualization()
        return {
            "visualization": visualization,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get graph visualization", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi lấy visualization: {str(e)}"
        )