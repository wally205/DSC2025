"""LangGraph pipeline builder for the chatbot with conversation support."""

from typing import Dict, Any, Literal, Optional, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from config import (
    get_logger, LoggerMixin, 
    INTENT_ANALYSIS_NODE, ACTION_EXECUTION_NODE, END_NODE
)
from agents import IntentAnalyzer, ActionExecutor
from .state_manager import StateManager, ChatbotState
from chat.history_manager import ChatHistoryManager, ChatSession


class ChatbotGraphBuilder(LoggerMixin):
    """Builds and manages the LangGraph-based chatbot pipeline."""
    
    def __init__(self):
        """Initialize graph builder with conversation support."""
        self.intent_analyzer = IntentAnalyzer()
        self.action_executor = ActionExecutor()
        self.state_manager = StateManager()
        self.history_manager = ChatHistoryManager()
        self._graph = None
        self.current_session: Optional[ChatSession] = None
        
    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph pipeline.
        
        Returns:
            Compiled StateGraph instance
        """
        self.logger.info("Building chatbot graph")
        
        # Create state graph
        workflow = StateGraph(ChatbotState)
        
        # Add nodes
        workflow.add_node(INTENT_ANALYSIS_NODE, self._intent_analysis_node)
        workflow.add_node(ACTION_EXECUTION_NODE, self._action_execution_node)
        
        # Set entry point
        workflow.set_entry_point(INTENT_ANALYSIS_NODE)
        
        # Add edges
        workflow.add_conditional_edges(
            INTENT_ANALYSIS_NODE,
            self._route_after_intent_analysis,
            {
                ACTION_EXECUTION_NODE: ACTION_EXECUTION_NODE,
                END: END
            }
        )
        
        workflow.add_edge(ACTION_EXECUTION_NODE, END)
        
        # Compile graph
        self._graph = workflow.compile()
        
        self.logger.info("Chatbot graph built successfully")
        return self._graph
    
    def _intent_analysis_node(self, state: ChatbotState) -> Dict[str, Any]:
        """
        Intent analysis node function.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with intent analysis results
        """
        self.logger.info("Executing intent analysis node")
        
        try:
            user_query = state["user_query"]
            
            # Perform intent analysis
            analysis_result = self.intent_analyzer.analyze_intent(user_query)
            
            # Update state
            updates = {
                "intent": analysis_result["intent"].value,
                "confidence": analysis_result["confidence"],
                "keywords": analysis_result.get("keywords", []),
                "reasoning": analysis_result.get("reasoning", "")
            }
            
            # Add any errors from analysis
            if analysis_result.get("errors"):
                updates["errors"] = state.get("errors", []) + analysis_result["errors"]
            
            updated_state = self.state_manager.update_state(
                state, updates, INTENT_ANALYSIS_NODE
            )
            
            self.logger.info(
                "Intent analysis completed",
                intent=analysis_result["intent"].value,
                confidence=analysis_result["confidence"]
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error("Intent analysis node failed", error=str(e))
            
            # Return state with error
            error_updates = {
                "intent": "unknown",
                "confidence": 0.0,
                "errors": state.get("errors", []) + [f"Intent analysis error: {str(e)}"]
            }
            
            return self.state_manager.update_state(
                state, error_updates, INTENT_ANALYSIS_NODE
            )
    
    async def _action_execution_node(self, state: ChatbotState) -> Dict[str, Any]:
        """
        Action execution node function.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with action execution results
        """
        self.logger.info("Executing action execution node")
        
        try:
            # Execute action based on intent
            result = await self.action_executor.execute_action(state)
            
            # Update state with execution results
            updated_state = self.state_manager.update_state(
                state, result, ACTION_EXECUTION_NODE
            )
            
            self.logger.info(
                "Action execution completed",
                response_type=result.get("response_type"),
                has_response=bool(result.get("response"))
            )
            
            return updated_state
            
        except Exception as e:
            self.logger.error("Action execution node failed", error=str(e))
            
            # Return state with error response
            error_updates = {
                "response": "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại.",
                "response_type": "error",
                "action_completed": False,
                "errors": state.get("errors", []) + [f"Action execution error: {str(e)}"]
            }
            
            return self.state_manager.update_state(
                state, error_updates, ACTION_EXECUTION_NODE
            )
    
    def _route_after_intent_analysis(
        self, state: ChatbotState
    ) -> str:
        """
        Routing function after intent analysis.
        
        Args:
            state: Current state
            
        Returns:
            Next node to execute
        """
        errors = state.get("errors", [])
        intent = state.get("intent")
        confidence = state.get("confidence", 0.0)
        
        # If there are critical errors, end the conversation
        critical_errors = [
            error for error in errors 
            if "validation" in error.lower() or "critical" in error.lower()
        ]
        
        if critical_errors:
            self.logger.warning("Critical errors detected, ending conversation", errors=critical_errors)
            return END
        
        # If intent analysis failed completely, end with error
        if intent is None or confidence == 0.0:
            self.logger.warning("Intent analysis failed, ending conversation")
            return END
        
        # Otherwise, proceed to action execution
        self.logger.info("Routing to action execution", intent=intent, confidence=confidence)
        return ACTION_EXECUTION_NODE
    
    async def process_query(
        self, 
        user_query: str, 
        session_id: str = None,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete pipeline.
        
        Args:
            user_query: User's input query
            session_id: Optional session ID for context
            conversation_context: Optional conversation history
            
        Returns:
            Processing result
        """
        self.logger.info("Processing user query", query=user_query[:100])
        
        if self._graph is None:
            self.build_graph()
        
        try:
            # Extract previous state from conversation context if available
            previous_state = None
            if conversation_context:
                # Parse conversation context to extract previous state
                previous_state = {
                    "conversation_history": conversation_context,
                    "last_weather_data": None,  # Will be extracted from history
                    "last_location": None  # Will be extracted from history
                }
                
                # Extract last weather data and location from conversation history
                for msg in reversed(conversation_context):
                    if msg.get("intent") in ["weather_query", "weather_agriculture"]:
                        # Extract location and weather data from previous weather responses
                        response = msg.get("response", "")
                        if "°C" in response:  # Basic check for weather data
                            previous_state["last_weather_data"] = {
                                "timestamp": msg.get("timestamp", 0),
                                "location": "detected_location"  # Could be improved
                            }
                        break
            
            # Create initial state
            initial_state = self.state_manager.create_initial_state(
                user_query=user_query,
                session_id=session_id,
                previous_state=previous_state
            )
            
            # Execute graph
            final_state = await self._graph.ainvoke(initial_state)
            
            # Log final state summary
            summary = self.state_manager.get_state_summary(final_state)
            self.logger.info("Query processing completed", summary=summary)
            
            return final_state
            
        except Exception as e:
            self.logger.error("Query processing failed", error=str(e))
            
            # Return error state
            return {
                "user_query": user_query,
                "intent": "unknown",
                "confidence": 0.0,
                "response": "Đã xảy ra lỗi hệ thống. Vui lòng thử lại sau.",
                "response_type": "system_error",
                "action_completed": False,
                "errors": [f"System error: {str(e)}"],
                "timestamp": self.state_manager._get_timestamp()
            }
    
    def start_new_session(self, title: Optional[str] = None) -> str:
        """
        Start a new chat session.
        
        Args:
            title: Optional session title
            
        Returns:
            Session ID
        """
        self.current_session = self.history_manager.create_session(title)
        self.logger.info("Started new chat session", session_id=self.current_session.session_id)
        return self.current_session.session_id
    
    def load_session(self, session_id: str) -> bool:
        """
        Load existing chat session.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            True if session loaded successfully
        """
        session = self.history_manager.load_session(session_id)
        if session:
            self.current_session = session
            self.logger.info("Loaded chat session", session_id=session_id)
            return True
        else:
            self.logger.warning("Failed to load session", session_id=session_id)
            return False
    
    async def process_query_with_history(self, user_query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process query with conversation context.
        
        Args:
            user_query: User's query
            session_id: Optional session ID, creates new if not provided
            
        Returns:
            Processing result with conversation context
        """
        # Ensure we have a session
        if not self.current_session or (session_id and self.current_session.session_id != session_id):
            if session_id:
                if not self.load_session(session_id):
                    self.start_new_session()
            else:
                self.start_new_session()
        
        # Add user message to history
        user_message = self.current_session.add_message(user_query, is_user=True)
        
        # Get conversation context for intent analysis
        conversation_context = self.current_session.get_context(max_messages=10)
        
        # Process with context
        try:
            result = await self.process_query(user_query, conversation_context=conversation_context)
            
            # Add bot response to history
            bot_message = self.current_session.add_message(
                result["response"], 
                is_user=False,
                metadata={
                    "intent": result.get("intent"),
                    "confidence": result.get("confidence"),
                    "response_type": result.get("response_type"),
                    "sources": result.get("sources", [])
                }
            )
            
            # Save session
            self.history_manager.save_session(self.current_session)
            
            # Add session info to result
            result.update({
                "session_id": self.current_session.session_id,
                "session_title": self.current_session.title,
                "message_id": bot_message.id,
                "conversation_length": len(self.current_session.messages)
            })
            
            return result
            
        except Exception as e:
            self.logger.error("Error processing query with history", error=str(e))
            
            # Still save user message even if processing failed
            error_response = f"Xin lỗi, đã xảy ra lỗi: {str(e)}"
            self.current_session.add_message(error_response, is_user=False)
            self.history_manager.save_session(self.current_session)
            
            return {
                "user_query": user_query,
                "response": error_response,
                "session_id": self.current_session.session_id,
                "error": str(e)
            }
    
    def get_session_history(self) -> Optional[Dict[str, Any]]:
        """Get current session history."""
        if self.current_session:
            return self.current_session.to_dict()
        return None
    
    def list_sessions(self, limit: int = 20) -> List[Dict[str, str]]:
        """List recent chat sessions."""
        return self.history_manager.list_sessions(limit)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        success = self.history_manager.delete_session(session_id)
        if success and self.current_session and self.current_session.session_id == session_id:
            self.current_session = None
        return success
    
    def get_graph_visualization(self) -> str:
        """
        Get a text representation of the graph structure.
        
        Returns:
            Graph structure as string
        """
        return f"""
        Chatbot LangGraph Pipeline:
        
        START
          ↓
        {INTENT_ANALYSIS_NODE}
          ↓
        Route Decision
          ↓
        {ACTION_EXECUTION_NODE}
          ↓
        END
        
        Nodes:
        - {INTENT_ANALYSIS_NODE}: Analyzes user intent and extracts information
        - {ACTION_EXECUTION_NODE}: Executes appropriate action based on intent
        
        Routing Logic:
        - If critical errors → END
        - If intent analysis failed → END  
        - Otherwise → {ACTION_EXECUTION_NODE}
        """