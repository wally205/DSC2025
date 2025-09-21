"""Streamlit UI for the chatbot demo."""

import sys
import os
import json
import requests
import asyncio
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import settings, get_logger, configure_logging

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="LangGraph Chatbot Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = f"http://{settings.api_host}:{settings.api_port}/api/v1"


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "current_session_title" not in st.session_state:
        st.session_state.current_session_title = "New Chat"


from graph.graph_builder import ChatbotGraphBuilder

# Initialize chatbot directly
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = ChatbotGraphBuilder()

def call_chatbot_api(message: str) -> Dict[str, Any]:
    """
    Call the chatbot with conversation history support.
    
    Args:
        message: User message
        
    Returns:
        Chatbot response
    """
    try:
        # Use conversation-aware processing with asyncio
        response = asyncio.run(st.session_state.chatbot.process_query_with_history(
            message, 
            session_id=st.session_state.session_id
        ))
        
        # Update session info
        if "session_id" in response:
            st.session_state.session_id = response["session_id"]
        if "session_title" in response:
            st.session_state.current_session_title = response["session_title"]
        
        if response:
            return {
                "response": response.get('response', 'Không có phản hồi'),
                "intent": response.get('intent', 'unknown'),
                "confidence": response.get('confidence', 0.0),
                "response_type": response.get('response_type', 'direct'),
                "sources": response.get('sources', []),
                "execution_time": response.get('execution_time', 0.0),
                "timestamp": response.get('timestamp', datetime.now().isoformat()),
                "session_id": response.get('session_id'),
                "conversation_length": response.get('conversation_length', 0)
            }
        else:
            return {
                "response": "Xin lỗi, tôi không thể xử lý yêu cầu này.",
                "intent": "error",
                "confidence": 0.0,
                "response_type": "error",
                "sources": [],
                "execution_time": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error("Chatbot error", error=str(e))
        return {
            "response": f"Lỗi xử lý: {str(e)}",
            "intent": "error",
            "confidence": 0.0,
            "response_type": "error",
            "sources": [],
            "execution_time": 0.0,
            "timestamp": datetime.now().isoformat()
        }


def get_health_status() -> Dict[str, Any]:
    """Get chatbot health status."""
    try:
        # Check if chatbot is initialized
        if hasattr(st.session_state, 'chatbot') and st.session_state.chatbot:
            return {
                "status": "healthy",
                "message": "Chatbot đã sẵn sàng",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "initializing", 
                "message": "Chatbot đang khởi tạo...",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Lỗi: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        response.raise_for_status()
        return response.json()
    except:
        return {"status": "unhealthy"}


def get_ingestion_status() -> Dict[str, Any]:
    """Get data ingestion status."""
    try:
        response = requests.get(f"{API_BASE_URL}/ingest/status", timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return {"success": False, "error": "Could not fetch ingestion status"}


def trigger_ingestion(clear_existing: bool = False) -> Dict[str, Any]:
    """Trigger data ingestion."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ingest",
            json={"clear_existing": clear_existing},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_sidebar():
    """Render the sidebar with controls, chat history and information."""
    with st.sidebar:
        st.title("🤖 LangGraph Chatbot")
        st.markdown("---")
        
        # Current session info
        st.subheader("� Current Chat")
        if st.session_state.session_id:
            st.text(f"Session: {st.session_state.current_session_title}")
            st.text(f"Messages: {len(st.session_state.messages)}")
        else:
            st.text("No active session")
        
        # New chat button
        if st.button("🆕 New Chat"):
            st.session_state.messages = []
            st.session_state.session_id = None
            st.session_state.current_session_title = "New Chat"
            st.rerun()
        
        # Chat history
        st.markdown("---")
        st.subheader("📚 Chat History")
        try:
            if hasattr(st.session_state, 'chatbot') and st.session_state.chatbot:
                sessions = st.session_state.chatbot.list_sessions(limit=10)
                
                if sessions:
                    for session in sessions:
                        session_title = session['title'][:30] + "..." if len(session['title']) > 30 else session['title']
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if st.button(f"� {session_title}", key=f"load_{session['session_id']}"):
                                # Load session
                                if st.session_state.chatbot.load_session(session['session_id']):
                                    st.session_state.session_id = session['session_id']
                                    st.session_state.current_session_title = session['title']
                                    
                                    # Load messages from session
                                    session_data = st.session_state.chatbot.get_session_history()
                                    if session_data:
                                        st.session_state.messages = []
                                        for msg in session_data['messages']:
                                            st.session_state.messages.append({
                                                "content": msg['content'],
                                                "is_user": msg['is_user'],
                                                "timestamp": msg['timestamp'],
                                                **msg.get('metadata', {})
                                            })
                                    
                                    st.rerun()
                        
                        with col2:
                            if st.button("🗑️", key=f"del_{session['session_id']}"):
                                if st.session_state.chatbot.delete_session(session['session_id']):
                                    st.success("Deleted!")
                                    st.rerun()
                else:
                    st.text("No chat history")
        except Exception as e:
            st.error(f"Error loading history: {e}")
        
        # System status
        st.markdown("---")
        st.subheader("📊 System Status")
        health = get_health_status()
        
        if health.get("status") == "healthy":
            st.success("✅ System Online")
        else:
            st.error("❌ System Offline")
        
        # Clear conversation
        st.markdown("---")
        if st.button("🗑️ Clear Current Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Instructions
        st.markdown("---")
        st.subheader("💡 Usage Guide")
        st.markdown("""
        **New Features:**
        - 💬 **Chat History**: All conversations saved
        - 🔄 **Context Aware**: Remembers previous messages
        - 📁 **Multiple Sessions**: Switch between chats
        
        **Ask about:**
        - 🌾 Rice cultivation: "kỹ thuật trồng lúa"
        - ☕ Coffee farming: "bệnh cà phê"
        - 🥔 Potato growing: "chăm sóc khoai tây"
        - 🌶️ Pepper cultivation: "hồ tiêu"
        """)
        
        # Data management (simplified)
        st.markdown("---")
        st.subheader("⚙️ Data Management")
        ingestion_status = get_ingestion_status()
        if ingestion_status.get("success"):
            status_data = ingestion_status.get("status", {})
            st.info(f"📄 Documents: {status_data.get('document_count', 0)}")
        else:
            st.warning("⚠️ Status unavailable")


def render_message(message: Dict[str, Any], is_user: bool = False):
    """Render a chat message."""
    if is_user:
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            
            # Show metadata in expander
            with st.expander("📊 Response Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Intent", message.get("intent", "N/A"))
                    st.metric("Confidence", f"{message.get('confidence', 0):.2f}")
                
                with col2:
                    st.metric("Response Type", message.get("response_type", "N/A"))
                    st.metric("Sources", len(message.get("sources", [])))
                
                # Show conversation info
                if message.get("conversation_length"):
                    st.metric("Conversation Length", message["conversation_length"])
                
                # Show sources if available
                sources = message.get("sources", [])
                if sources:
                    st.subheader("📄 Source Documents")
                    for i, source in enumerate(sources, 1):
                        st.text(f"{i}. {source.get('filename', 'Unknown')} ({source.get('chunk_count', 0)} chunks)")
                
                # Show session info if available
                if message.get("session_id"):
                    st.text(f"Session ID: {message['session_id'][:8]}...")


def main():
    """Main application function."""
    init_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("💬 Smart Agricultural Assistant")
    st.markdown("🌾 AI-powered farming advisor with conversation memory and hybrid search")
    
    # Show current session info
    if st.session_state.session_id:
        st.info(f"💭 **{st.session_state.current_session_title}** • {len(st.session_state.messages)} messages")
    
    # Display chat messages
    for message in st.session_state.messages:
        render_message(message, is_user=message.get("is_user", False))
    
    # Chat input
    if prompt := st.chat_input("Ask about farming, crop diseases, cultivation techniques..."):
        # Add user message to chat
        user_message = {
            "content": prompt,
            "is_user": True,
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Display user message
        render_message(user_message, is_user=True)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                response = call_chatbot_api(prompt)
                
                # Display bot response
                st.write(response["response"])
                
                # Show metadata
                with st.expander("📊 Response Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Intent", response.get("intent", "N/A"))
                        st.metric("Confidence", f"{response.get('confidence', 0):.2f}")
                    
                    with col2:
                        st.metric("Response Type", response.get("response_type", "N/A"))
                        st.metric("Sources", len(response.get("sources", [])))
                    
                    # Show conversation info
                    if response.get("conversation_length"):
                        st.metric("Conversation Length", response["conversation_length"])
                    
                    # Show sources
                    sources = response.get("sources", [])
                    if sources:
                        st.subheader("📄 Source Documents")
                        for i, source in enumerate(sources, 1):
                            st.text(f"{i}. {source.get('filename', 'Unknown')} ({source.get('chunk_count', 0)} chunks)")
        
        # Add bot message to chat
        bot_message = {
            "content": response["response"],
            "is_user": False,
            "intent": response.get("intent"),
            "confidence": response.get("confidence"),
            "response_type": response.get("response_type"),
            "sources": response.get("sources", []),
            "conversation_length": response.get("conversation_length"),
            "session_id": response.get("session_id"),
            "timestamp": response.get("timestamp")
        }
        st.session_state.messages.append(bot_message)
        
        # Update session info
        if response.get("session_id"):
            st.session_state.session_id = response["session_id"]


if __name__ == "__main__":
    main()