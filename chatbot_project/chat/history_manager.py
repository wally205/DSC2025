"""Chat history management for persistent conversations."""

import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import uuid

from config import get_logger, LoggerMixin


class ChatMessage:
    """Represents a single chat message."""
    
    def __init__(self, content: str, is_user: bool, metadata: Optional[Dict] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.is_user = is_user
        self.timestamp = datetime.now().isoformat()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'is_user': self.is_user,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create message from dictionary."""
        msg = cls(data['content'], data['is_user'], data.get('metadata', {}))
        msg.id = data['id']
        msg.timestamp = data['timestamp']
        return msg


class ChatSession:
    """Represents a chat session with message history."""
    
    def __init__(self, session_id: Optional[str] = None, title: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.title = title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.messages: List[ChatMessage] = []
    
    def add_message(self, content: str, is_user: bool, metadata: Optional[Dict] = None) -> ChatMessage:
        """Add a message to the session."""
        message = ChatMessage(content, is_user, metadata)
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
        
        # Auto-generate title from first user message
        if is_user and len(self.messages) == 1 and self.title.startswith("Chat "):
            self.title = content[:50] + "..." if len(content) > 50 else content
        
        return message
    
    def get_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent messages as context for LLM."""
        recent_messages = self.messages[-max_messages:] if max_messages > 0 else self.messages
        
        context = []
        for msg in recent_messages:
            role = "user" if msg.is_user else "assistant"
            context.append({
                "role": role,
                "content": msg.content
            })
        
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'messages': [msg.to_dict() for msg in self.messages]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create session from dictionary."""
        session = cls(data['session_id'], data['title'])
        session.created_at = data['created_at']
        session.updated_at = data['updated_at']
        session.messages = [ChatMessage.from_dict(msg_data) for msg_data in data['messages']]
        return session


class ChatHistoryManager(LoggerMixin):
    """Manages chat history with SQLite persistence."""
    
    def __init__(self, db_path: str = "chat_history.db"):
        """Initialize chat history manager."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        messages_json TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_updated_at 
                    ON chat_sessions(updated_at DESC)
                """)
                
                conn.commit()
                
            self.logger.info("Chat history database initialized", db_path=str(self.db_path))
            
        except Exception as e:
            self.logger.error("Failed to initialize database", error=str(e))
            raise
    
    def create_session(self, title: Optional[str] = None) -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(title=title)
        self.save_session(session)
        self.logger.info("Created new chat session", session_id=session.session_id)
        return session
    
    def save_session(self, session: ChatSession) -> None:
        """Save session to database."""
        try:
            session.updated_at = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO chat_sessions 
                    (session_id, title, created_at, updated_at, messages_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    json.dumps([msg.to_dict() for msg in session.messages])
                ))
                
                conn.commit()
                
            self.logger.info("Session saved", session_id=session.session_id)
            
        except Exception as e:
            self.logger.error("Failed to save session", session_id=session.session_id, error=str(e))
            raise
    
    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT session_id, title, created_at, updated_at, messages_json
                    FROM chat_sessions WHERE session_id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                
                if row:
                    session_id, title, created_at, updated_at, messages_json = row
                    messages_data = json.loads(messages_json)
                    
                    session = ChatSession(session_id, title)
                    session.created_at = created_at
                    session.updated_at = updated_at
                    session.messages = [ChatMessage.from_dict(msg_data) for msg_data in messages_data]
                    
                    self.logger.info("Session loaded", session_id=session_id)
                    return session
                
                return None
                
        except Exception as e:
            self.logger.error("Failed to load session", session_id=session_id, error=str(e))
            return None
    
    def list_sessions(self, limit: int = 50) -> List[Dict[str, str]]:
        """List recent chat sessions."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT session_id, title, created_at, updated_at
                    FROM chat_sessions 
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))
                
                sessions = []
                for row in cursor.fetchall():
                    session_id, title, created_at, updated_at = row
                    sessions.append({
                        'session_id': session_id,
                        'title': title,
                        'created_at': created_at,
                        'updated_at': updated_at
                    })
                
                self.logger.info("Sessions listed", count=len(sessions))
                return sessions
                
        except Exception as e:
            self.logger.error("Failed to list sessions", error=str(e))
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM chat_sessions WHERE session_id = ?
                """, (session_id,))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    self.logger.info("Session deleted", session_id=session_id)
                    return True
                else:
                    self.logger.warning("Session not found for deletion", session_id=session_id)
                    return False
                
        except Exception as e:
            self.logger.error("Failed to delete session", session_id=session_id, error=str(e))
            return False
    
    def clear_old_sessions(self, days: int = 30) -> int:
        """Clear sessions older than specified days."""
        try:
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)
            cutoff_str = cutoff_date.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM chat_sessions WHERE updated_at < ?
                """, (cutoff_str,))
                
                conn.commit()
                deleted_count = cursor.rowcount
                
            self.logger.info("Old sessions cleared", deleted_count=deleted_count, days=days)
            return deleted_count
            
        except Exception as e:
            self.logger.error("Failed to clear old sessions", error=str(e))
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM chat_sessions")
                total_sessions = cursor.fetchone()[0]
                
                cursor = conn.execute("""
                    SELECT session_id, messages_json FROM chat_sessions
                """)
                
                total_messages = 0
                for _, messages_json in cursor.fetchall():
                    messages = json.loads(messages_json)
                    total_messages += len(messages)
                
                return {
                    'total_sessions': total_sessions,
                    'total_messages': total_messages,
                    'db_path': str(self.db_path),
                    'db_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
                }
                
        except Exception as e:
            self.logger.error("Failed to get stats", error=str(e))
            return {}