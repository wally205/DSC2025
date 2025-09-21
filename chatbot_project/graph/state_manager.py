"""State management for the LangGraph chatbot pipeline."""

from typing import Dict, Any, List, Optional, TypedDict

from config import get_logger, LoggerMixin


class ChatbotState(TypedDict):
    """State schema for the chatbot pipeline."""
    # Input
    user_query: str
    session_id: Optional[str]
    
    # Conversation History & Context
    conversation_history: List[Dict[str, Any]]
    previous_context: str
    last_weather_data: Optional[Dict[str, Any]]
    last_location: Optional[str]
    
    # Intent analysis results
    intent: Optional[str]
    confidence: float
    keywords: List[str]
    reasoning: str
    
    # Search results
    search_results: Optional[Dict[str, Any]]
    context_used: str
    sources: List[Dict[str, str]]
    
    # Response
    response: str
    response_type: str
    
    # Metadata
    timestamp: str
    execution_path: List[str]
    errors: List[str]
    action_completed: bool


class StateManager(LoggerMixin):
    """Manages state transitions and validation for the chatbot pipeline."""
    
    def __init__(self):
        """Initialize state manager."""
        pass
    
    def create_initial_state(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        previous_state: Optional[ChatbotState] = None
    ) -> ChatbotState:
        """
        Create initial state for a new conversation turn.
        
        Args:
            user_query: User's input query
            session_id: Optional session identifier
            previous_state: Previous conversation state for context
            
        Returns:
            Initial state dictionary
        """
        self.logger.info("Creating initial state", query=user_query[:100])
        
        # Get conversation history from previous state
        conversation_history = []
        previous_context = ""
        last_weather_data = None
        last_location = None
        
        if previous_state:
            conversation_history = previous_state.get("conversation_history", [])
            previous_context = previous_state.get("context_used", "")
            last_weather_data = previous_state.get("last_weather_data")
            last_location = previous_state.get("last_location")
            
            # Add previous turn to history
            if previous_state.get("user_query") and previous_state.get("response"):
                conversation_history.append({
                    "user_query": previous_state["user_query"],
                    "response": previous_state["response"],
                    "intent": previous_state.get("intent"),
                    "timestamp": previous_state.get("timestamp")
                })
                
                # Keep only last 5 turns to avoid context overload
                conversation_history = conversation_history[-5:]
        
        state = ChatbotState(
            # Input
            user_query=user_query.strip(),
            session_id=session_id,
            
            # Conversation History & Context
            conversation_history=conversation_history,
            previous_context=previous_context,
            last_weather_data=last_weather_data,
            last_location=last_location,
            
            # Intent analysis results
            intent=None,
            confidence=0.0,
            keywords=[],
            reasoning="",
            
            # Search results
            search_results=None,
            context_used="",
            sources=[],
            
            # Response
            response="",
            response_type="",
            
            # Metadata
            timestamp=self._get_timestamp(),
            execution_path=[],
            errors=[],
            action_completed=False
        )
        
        return state
    
    def update_state(
        self,
        current_state: ChatbotState,
        updates: Dict[str, Any],
        step_name: str
    ) -> ChatbotState:
        """
        Update state with new information.
        
        Args:
            current_state: Current state
            updates: Dictionary of updates to apply
            step_name: Name of the processing step
            
        Returns:
            Updated state
        """
        self.logger.info("Updating state", step=step_name, updates=list(updates.keys()))
        
        # Create new state with updates
        new_state = {**current_state, **updates}
        
        # Add step to execution path
        execution_path = new_state.get("execution_path", [])
        execution_path.append(step_name)
        new_state["execution_path"] = execution_path
        
        # Validate updated state
        validation_errors = self.validate_state(new_state)
        if validation_errors:
            self.logger.warning("State validation errors", errors=validation_errors)
            errors = new_state.get("errors", [])
            errors.extend(validation_errors)
            new_state["errors"] = errors
        
        return new_state
    
    def validate_state(self, state: Dict[str, Any]) -> List[str]:
        """
        Validate state consistency and completeness.
        
        Args:
            state: State to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check required fields
        required_fields = ["user_query", "timestamp"]
        for field in required_fields:
            if not state.get(field):
                errors.append(f"Missing required field: {field}")
        
        # Validate confidence range
        confidence = state.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
            errors.append(f"Invalid confidence value: {confidence}")
        
        # Validate intent if present
        intent = state.get("intent")
        if intent is not None:
            valid_intents = ["search_document", "general_question", "weather_query", "weather_agriculture", "unknown"]
            if intent not in valid_intents:
                errors.append(f"Invalid intent: {intent}")
        
        # Check search results consistency
        search_results = state.get("search_results")
        if search_results is not None:
            if not isinstance(search_results, dict):
                errors.append("search_results must be a dictionary")
            elif "has_results" not in search_results:
                errors.append("search_results missing 'has_results' field")
        
        # Validate sources format
        sources = state.get("sources", [])
        if not isinstance(sources, list):
            errors.append("sources must be a list")
        else:
            for i, source in enumerate(sources):
                if not isinstance(source, dict):
                    errors.append(f"source[{i}] must be a dictionary")
                elif "filename" not in source:
                    errors.append(f"source[{i}] missing 'filename' field")
        
        return errors
    
    def is_complete(self, state: Dict[str, Any]) -> bool:
        """
        Check if the state represents a completed conversation turn.
        
        Args:
            state: State to check
            
        Returns:
            True if conversation turn is complete
        """
        required_completion_fields = [
            "user_query",
            "intent", 
            "response",
            "response_type"
        ]
        
        return all(
            state.get(field) is not None and state.get(field) != ""
            for field in required_completion_fields
        )
    
    def get_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of the current state for logging/debugging.
        
        Args:
            state: State to summarize
            
        Returns:
            Summary dictionary
        """
        return {
            "user_query": state.get("user_query", "")[:50] + "...",
            "intent": state.get("intent"),
            "confidence": state.get("confidence", 0.0),
            "has_response": bool(state.get("response")),
            "has_search_results": bool(state.get("search_results")),
            "sources_count": len(state.get("sources", [])),
            "execution_path": state.get("execution_path", []),
            "errors_count": len(state.get("errors", [])),
            "is_complete": self.is_complete(state)
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.utcnow().isoformat()