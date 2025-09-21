"""Intent Analysis Agent for understanding user queries."""

import re
from typing import Dict, Any, List

from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    get_logger, LoggerMixin, IntentType, 
    HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD, settings
)
from tools import SearchTools


class IntentAnalyzer(LoggerMixin):
    """Agent responsible for analyzing user intent and extracting query information."""
    
    def __init__(self):
        """Initialize intent analyzer."""
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,  # Use configurable model
            temperature=0.1,  # Keep low for intent analysis
            max_output_tokens=2048,  # Sufficient for intent analysis
            google_api_key=settings.google_api_key
        )
        self.search_tools = SearchTools()
        
    def analyze_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze user intent from query.
        
        Args:
            user_query: The user's input query
            
        Returns:
            Dictionary containing intent analysis results
        """
        self.logger.info("Analyzing user intent", query=user_query[:100])
        
        try:
            # Validate query first
            validation = self.search_tools.validate_query(user_query)
            if not validation["is_valid"]:
                return self._create_intent_result(
                    intent=IntentType.UNKNOWN,
                    confidence=0.0,
                    query=user_query,
                    errors=validation["errors"]
                )
            
            # Extract basic patterns
            intent_result = self._extract_intent_patterns(user_query)
            
            # If confidence is low, use LLM for better analysis
            if intent_result["confidence"] < MEDIUM_CONFIDENCE_THRESHOLD:
                llm_result = self._analyze_with_llm(user_query)
                if llm_result["confidence"] > intent_result["confidence"]:
                    intent_result = llm_result
            
            # Add extracted keywords
            intent_result["keywords"] = self.search_tools.extract_keywords(user_query)
            intent_result["validation"] = validation
            
            self.logger.info(
                "Intent analysis completed",
                intent=intent_result["intent"],
                confidence=intent_result["confidence"]
            )
            
            return intent_result
            
        except Exception as e:
            self.logger.error("Intent analysis failed", error=str(e))
            return self._create_intent_result(
                intent=IntentType.UNKNOWN,
                confidence=0.0,
                query=user_query,
                errors=[f"Analysis error: {str(e)}"]
            )
    
    def _extract_intent_patterns(self, query: str) -> Dict[str, Any]:
        """Extract intent using pattern matching."""
        query_lower = query.lower().strip()
        
        # Pure Weather Query patterns - Chỉ hỏi thời tiết đơn thuần
        pure_weather_patterns = [
            r'^(thời tiết|weather|dự báo)\b.*\b(hôm nay|ngày|hiện tại|bây giờ)\b',
            r'\b(thời tiết|weather)\b.*\b(ở|tại|trong)\b.*\b(thành phố|tỉnh|khu vực)\b',
            r'^(hôm nay|ngày hôm nay)\b.*\b(thời tiết|weather|trời)\b.*\b(ra sao|như thế nào|thế nào)\b',
            r'^(trời|thời tiết)\b.*\b(ra sao|như thế nào|thế nào)\b',
            r'\b(nhiệt độ|độ ẩm|mưa|nắng|gió)\b.*\b(hôm nay|hiện tại|bây giờ)\b',
            r'^(có mưa|có nắng|có gió)\b',
            r'\b(dự báo thời tiết|weather forecast)\b'
        ]
        
        for pattern in pure_weather_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                # Kiểm tra xem có từ khóa nông nghiệp không
                agriculture_keywords = ['trồng', 'cà phê', 'lúa', 'khoai', 'hồ tiêu', 'nông nghiệp', 'cây trồng', 'thu hoạch', 'gieo', 'phun thuốc', 'bón phân']
                has_agriculture = any(keyword in query_lower for keyword in agriculture_keywords)
                
                if not has_agriculture:
                    return self._create_intent_result(
                        intent=IntentType.WEATHER_QUERY,
                        confidence=0.9,
                        query=query,
                        reasoning="Detected pure weather query pattern"
                    )
        
        # Weather + Agriculture patterns - Tư vấn nông nghiệp dựa trên thời tiết
        weather_agriculture_patterns = [
            r'\b(thời tiết|weather|dự báo)\b.*\b(trồng|cà phê|lúa|khoai|hồ tiêu|nông nghiệp)\b',
            r'\b(trồng|cà phê|lúa|khoai|hồ tiêu)\b.*\b(thời tiết|weather|dự báo|mưa|nắng|gió)\b',
            r'\b(nên trồng|có nên)\b.*\b(thời tiết|mưa|nắng)\b',
            r'\b(thời tiết.*có.*phù hợp|phù hợp.*thời tiết)\b',
            r'\b(dự báo.*tác động|ảnh hưởng.*thời tiết)\b',
            r'\b(nhiệt độ|độ ẩm|lượng mưa)\b.*\b(cà phê|lúa|trồng trọt)\b',
            r'\b(mưa|nắng|gió)\b.*\b(có nên|nên)\b.*\b(trồng|phun thuốc|bón phân|thu hoạch)\b'
        ]
        
        for pattern in weather_agriculture_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return self._create_intent_result(
                    intent=IntentType.WEATHER_AGRICULTURE,
                    confidence=0.9,
                    query=query,
                    reasoning="Detected weather-agriculture consultation pattern"
                )
        
        # Agricultural consultation patterns - Tư vấn nông nghiệp
        agriculture_patterns = [
            r'\b(cà phê|cafe|coffee)\b',
            r'\b(trồng|gieo|trồng trọt|canh tác)\b',
            r'\b(sâu bệnh|côn trùng|bệnh|sâu|mọt)\b',
            r'\b(phân bón|thuốc|thuốc trừ sâu|dinh dưỡng)\b',
            r'\b(nông nghiệp|nông dân|làm ruộng)\b',
            r'\b(cây trồng|lúa|ngô|khoai)\b',
            r'\b(tưới|tưới nước|chăm sóc)\b',
            r'\b(thu hoạch|mùa màng|năng suất)\b',
            r'\b(đất|đất trồng|thổ nhưỡng)\b'
        ]
        
        agriculture_score = 0
        for pattern in agriculture_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                agriculture_score += 0.3
        
        if agriculture_score >= 0.3:
            return self._create_intent_result(
                intent=IntentType.SEARCH_DOCUMENT,
                confidence=min(agriculture_score, 0.95),
                query=query,
                reasoning="Detected agricultural consultation pattern"
            )
        
        # Search document patterns - General
        search_patterns = [
            r'\b(tìm|search|tra cứu|tìm kiếm|cho tôi biết|hỏi về)\b',
            r'\b(thông tin|tài liệu|document|file|pdf)\b',
            r'\b(là gì|như thế nào|tại sao|khi nào|ở đâu|làm sao)\b',
            r'\?$'  # Ends with question mark
        ]
        
        search_score = 0
        matched_patterns = []
        
        for pattern in search_patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                search_score += 0.3
                matched_patterns.append(pattern)
        
        # Check for question words
        question_words = ['gì', 'sao', 'như thế nào', 'tại sao', 'khi nào', 'ở đâu', 'ai', 'làm sao', 'bao nhiêu', 'ra sao']
        if any(word in query_lower for word in question_words):
            search_score += 0.5
        
        # Enhanced domain-specific keywords for agriculture
        domain_keywords = [
            'cà phê', 'nông nghiệp', 'trồng trọt', 'sản xuất', 'quy trình',
            'sâu bệnh', 'côn trùng', 'phân bón', 'thuốc', 'tưới', 'chăm sóc',
            'thu hoạch', 'năng suất', 'đất', 'khí hậu', 'cây trồng'
        ]
        if any(keyword in query_lower for keyword in domain_keywords):
            search_score += 0.4
        
        if search_score >= 0.3:
            return self._create_intent_result(
                intent=IntentType.SEARCH_DOCUMENT,
                confidence=min(search_score, 0.95),
                query=query,
                reasoning=f"Matched {len(matched_patterns)} search patterns"
            )
        
        # General question patterns
        if len(query.strip()) > 10 and ('?' in query or any(
            word in query_lower for word in ['như thế nào', 'là gì', 'tại sao']
        )):
            return self._create_intent_result(
                intent=IntentType.GENERAL_QUESTION,
                confidence=0.6,
                query=query,
                reasoning="Detected general question pattern"
            )
        
        # Default to unknown with low confidence
        return self._create_intent_result(
            intent=IntentType.UNKNOWN,
            confidence=0.1,
            query=query,
            reasoning="No clear patterns detected"
        )
    
    def _analyze_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM for advanced intent analysis."""
        prompt = f"""
        Bạn là một trợ lý AI chuyên về nông nghiệp và tư vấn sâu bệnh. 
        Hãy phân tích ý định của người dùng từ câu hỏi sau và trả lời theo format JSON:

        Câu hỏi: "{query}"

        Các loại ý định có thể:
        1. "search_document" - Tìm kiếm thông tin về nông nghiệp, cà phê, sâu bệnh, phương pháp trồng trọt, chăm sóc cây trồng
        2. "general_question" - Câu hỏi chung về nông nghiệp không cần tìm tài liệu cụ thể
        3. "weather_agriculture" - Tư vấn nông nghiệp dựa trên thời tiết, điều kiện khí hậu cho cây trồng
        4. "unknown" - Không liên quan đến nông nghiệp hoặc không rõ ý định

        Ưu tiên phân loại "search_document" cho các câu hỏi về:
        - Cà phê và quy trình sản xuất
        - Sâu bệnh và cách phòng trị
        - Phân bón và dinh dưỡng cây trồng
        - Kỹ thuật trồng trọt
        - Chăm sóc và thu hoạch

        Trả về JSON với format:
        {{
            "intent": "<intent_type>",
            "confidence": <float từ 0.0 đến 1.0>,
            "reasoning": "<lý do phân tích>"
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Try to extract JSON from response
            import json
            if content.startswith('{') and content.endswith('}'):
                result = json.loads(content)
                
                # Validate result
                if all(key in result for key in ['intent', 'confidence', 'reasoning']):
                    intent = result['intent']
                    confidence = float(result['confidence'])
                    reasoning = result['reasoning']
                    
                    # Validate intent type
                    if intent in [e.value for e in IntentType]:
                        return self._create_intent_result(
                            intent=IntentType(intent),
                            confidence=confidence,
                            query=query,
                            reasoning=f"LLM: {reasoning}"
                        )
            
            # If parsing fails, fall back to pattern-based result
            self.logger.warning("Failed to parse LLM response", response=content)
            return self._extract_intent_patterns(query)
            
        except Exception as e:
            self.logger.error("LLM intent analysis failed", error=str(e))
            return self._extract_intent_patterns(query)
    
    def _create_intent_result(
        self,
        intent: IntentType,
        confidence: float,
        query: str,
        reasoning: str = "",
        errors: List[str] = None
    ) -> Dict[str, Any]:
        """Create standardized intent result."""
        return {
            "intent": intent,
            "confidence": confidence,
            "query": query,
            "reasoning": reasoning,
            "errors": errors or [],
            "timestamp": self._get_timestamp(),
            "confidence_level": self._get_confidence_level(confidence)
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level as string."""
        if confidence >= HIGH_CONFIDENCE_THRESHOLD:
            return "high"
        elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
            return "medium"
        else:
            return "low"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()