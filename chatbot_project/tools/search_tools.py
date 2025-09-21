"""Search and utility tools for agents."""

import re
from typing import Dict, List, Optional, Any

from config import get_logger, LoggerMixin, IntentType
from .document_retriever import DocumentRetriever


class SearchTools(LoggerMixin):
    """Collection of search and utility tools for agents."""
    
    def __init__(self):
        """Initialize search tools."""
        self.document_retriever = DocumentRetriever()
        
    def search_knowledge_base(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: Search query
            limit: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            Dictionary with search results and metadata
        """
        self.logger.info(
            "Searching knowledge base",
            query=query[:100],
            limit=limit,
            min_score=min_score
        )
        
        try:
            # Get relevant context
            context = self.document_retriever.get_relevant_context(
                query=query,
                limit=limit,
                min_score=min_score
            )
            
            # Get document sources
            sources = self.document_retriever.get_document_sources(
                query=query,
                limit=limit
            )
            
            # Get results with scores for confidence calculation
            results_with_scores = self.document_retriever.search_with_scores(
                query=query,
                limit=limit
            )
            
            # Calculate average confidence
            if results_with_scores:
                scores = [score for _, score in results_with_scores]
                avg_confidence = sum(scores) / len(scores)
                max_confidence = max(scores)
            else:
                avg_confidence = 0.0
                max_confidence = 0.0
            
            search_result = {
                "context": context,
                "sources": sources,
                "results_count": len(results_with_scores),
                "avg_confidence": avg_confidence,
                "max_confidence": max_confidence,
                "has_results": bool(context and context.strip())
            }
            
            self.logger.info(
                "Knowledge base search completed",
                has_results=search_result["has_results"],
                results_count=search_result["results_count"],
                avg_confidence=avg_confidence
            )
            
            return search_result
            
        except Exception as e:
            self.logger.error("Knowledge base search failed", error=str(e))
            return {
                "context": "",
                "sources": [],
                "results_count": 0,
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "has_results": False,
                "error": str(e)
            }
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction (can be enhanced with NLP libraries)
        # Remove special characters and split
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = cleaned_text.split()
        
        # Filter out common Vietnamese stop words
        stop_words = {
            'tôi', 'bạn', 'anh', 'chị', 'em', 'của', 'và', 'có', 'là', 'trong',
            'với', 'cho', 'để', 'được', 'không', 'này', 'đó', 'về', 'như',
            'khi', 'nào', 'sao', 'ai', 'gì', 'đâu', 'thế', 'hỏi', 'biết'
        }
        
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords[:10]  # Return top 10 keywords
    
    def format_response(
        self,
        context: str,
        query: str,
        sources: List[Dict[str, str]] = None
    ) -> str:
        """
        Format a response with context and sources.
        
        Args:
            context: Retrieved context
            query: Original query
            sources: List of source documents
            
        Returns:
            Formatted response string
        """
        if not context or not context.strip():
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu."
        
        response_parts = [
            f"Dựa trên thông tin từ tài liệu, đây là câu trả lời cho câu hỏi của bạn:\n",
            context.strip()
        ]
        
        # Add sources if available
        if sources:
            response_parts.append("\n\nNguồn tài liệu:")
            for i, source in enumerate(sources, 1):
                filename = source.get('filename', 'Unknown')
                chunk_count = source.get('chunk_count', 0)
                response_parts.append(f"{i}. {filename} ({chunk_count} đoạn văn)")
        
        return "\n".join(response_parts)
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate and analyze a user query.
        
        Args:
            query: User query to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "query_length": len(query.strip()),
            "has_question_words": False,
            "keywords": []
        }
        
        cleaned_query = query.strip()
        
        # Check if query is empty
        if not cleaned_query:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Câu hỏi không được để trống")
            return validation_result
        
        # Check minimum length
        if len(cleaned_query) < 3:
            validation_result["is_valid"] = False
            validation_result["errors"].append("Câu hỏi quá ngắn")
            return validation_result
        
        # Check maximum length
        if len(cleaned_query) > 500:
            validation_result["warnings"].append("Câu hỏi khá dài, có thể ảnh hưởng đến chất lượng tìm kiếm")
        
        # Check for question words
        question_words = ['gì', 'sao', 'như thế nào', 'tại sao', 'khi nào', 'ở đâu', 'ai', 'làm sao']
        validation_result["has_question_words"] = any(
            word in cleaned_query.lower() for word in question_words
        )
        
        # Extract keywords
        validation_result["keywords"] = self.extract_keywords(cleaned_query)
        
        if not validation_result["keywords"]:
            validation_result["warnings"].append("Không tìm thấy từ khóa rõ ràng trong câu hỏi")
        
        return validation_result