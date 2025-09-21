"""Application constants."""

from enum import Enum
from typing import Final

# Agent types
INTENT_ANALYZER_AGENT: Final[str] = "intent_analyzer"
ACTION_EXECUTOR_AGENT: Final[str] = "action_executor"

# Node names in the graph
INTENT_ANALYSIS_NODE: Final[str] = "intent_analysis"
ACTION_EXECUTION_NODE: Final[str] = "action_execution"
END_NODE: Final[str] = "end"

# State keys
USER_QUERY_KEY: Final[str] = "user_query"
INTENT_KEY: Final[str] = "intent"
CONFIDENCE_KEY: Final[str] = "confidence"
SEARCH_RESULTS_KEY: Final[str] = "search_results"
RESPONSE_KEY: Final[str] = "response"
METADATA_KEY: Final[str] = "metadata"

# Intent types
class IntentType(str, Enum):
    """Enumeration of supported intent types."""
    SEARCH_DOCUMENT = "search_document"
    GENERAL_QUESTION = "general_question"
    WEATHER_QUERY = "weather_query"
    WEATHER_AGRICULTURE = "weather_agriculture"
    UNKNOWN = "unknown"

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.8
MEDIUM_CONFIDENCE_THRESHOLD: Final[float] = 0.5
LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.3

# Vector search parameters
DEFAULT_SEARCH_LIMIT: Final[int] = 5
MAX_SEARCH_LIMIT: Final[int] = 20
SIMILARITY_THRESHOLD: Final[float] = 0.7

# File processing
SUPPORTED_FILE_EXTENSIONS: Final[tuple] = (".pdf", ".txt", ".md")
MAX_FILE_SIZE_MB: Final[int] = 10

# HTTP status codes
HTTP_OK: Final[int] = 200
HTTP_BAD_REQUEST: Final[int] = 400
HTTP_INTERNAL_SERVER_ERROR: Final[int] = 500

# Response templates
ERROR_RESPONSES: Final[dict] = {
    "no_results": "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn.",
    "low_confidence": "Tôi không chắc chắn về câu trả lời. Bạn có thể hỏi cụ thể hơn không?",
    "processing_error": "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
    "unknown_intent": "Tôi không hiểu câu hỏi của bạn. Bạn có thể diễn đạt lại không?"
}