"""Action Execution Agent for performing actions based on intent."""

import re
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    get_logger, LoggerMixin, IntentType, 
    ERROR_RESPONSES,
    HIGH_CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD, settings
)
from tools import SearchTools
from tools.agriculture_weather_advisor import AgricultureWeatherAdvisor, AgricultureAdvice, WeatherCondition


class ActionExecutor(LoggerMixin):
    """Agent responsible for executing actions based on analyzed intent."""
    
    def __init__(self):
        """Initialize action executor."""
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,  # Use configurable model
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,  # Increased for longer responses
            google_api_key=settings.google_api_key
        )
        self.search_tools = SearchTools()
        self.weather_advisor = AgricultureWeatherAdvisor()
        
    async def execute_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute action based on intent analysis results.
        
        Args:
            state: Current state containing intent analysis results
            
        Returns:
            Updated state with response and results
        """
        intent = state.get("intent")
        confidence = state.get("confidence", 0.0)
        user_query = state.get("user_query", "")
        
        self.logger.info(
            "Executing action",
            intent=intent,
            confidence=confidence,
            query=user_query[:100]
        )
        
        try:
            # Route to appropriate action based on intent
            if intent == IntentType.SEARCH_DOCUMENT:
                result = self._handle_document_search(state)
            elif intent == IntentType.GENERAL_QUESTION:
                result = self._handle_general_question(state)
            elif intent == IntentType.WEATHER_QUERY:
                result = await self._handle_weather_query(state)
            elif intent == IntentType.WEATHER_AGRICULTURE:
                result = await self._handle_weather_agriculture(state)
            else:
                result = self._handle_unknown_intent(state)
            
            # Add execution metadata
            result.update({
                "action_completed": True,
                "execution_timestamp": self._get_timestamp()
            })
            
            self.logger.info(
                "Action execution completed",
                intent=intent,
                has_response=bool(result.get("response"))
            )
            
            return result
            
        except Exception as e:
            self.logger.error("Action execution failed", error=str(e))
            return self._create_error_response(state, str(e))
    
    def _handle_document_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document search intent."""
        user_query = state.get("user_query", "")
        confidence = state.get("confidence", 0.0)
        
        # Perform knowledge base search
        search_result = self.search_tools.search_knowledge_base(
            query=user_query,
            limit=15,  # Further increased for maximum comprehensiveness
            min_score=0.3 if confidence >= HIGH_CONFIDENCE_THRESHOLD else 0.2  # Even lower threshold
        )
        
        if search_result["has_results"]:
            # Generate response using LLM with context
            response = self._generate_contextual_response(
                query=user_query,
                context=search_result["context"],
                sources=search_result["sources"]
            )
        else:
            response = ERROR_RESPONSES["no_results"]
        
        return {
            **state,
            "response": response,
            "response_type": "document_search",
            "search_results": search_result,
            "sources": search_result["sources"],
            "context_used": search_result.get("context", "")
        }
    
    def _handle_general_question(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general question intent."""
        user_query = state.get("user_query", "")
        confidence = state.get("confidence", 0.0)
        
        # Try document search first
        search_result = self.search_tools.search_knowledge_base(
            query=user_query,
            limit=3,
            min_score=0.5
        )
        
        if search_result["has_results"] and search_result["max_confidence"] > 0.6:
            # Use document-based response
            response = self._generate_contextual_response(
                query=user_query,
                context=search_result["context"],
                sources=search_result["sources"]
            )
            response_type = "general_with_context"
        else:
            # Generate general response
            response = self._generate_general_response(user_query)
            response_type = "general_without_context"
        
        return {
            **state,
            "response": response,
            "response_type": response_type,
            "search_results": search_result,
            "sources": search_result.get("sources", []),
            "context_used": search_result.get("context", "")
        }
    
    async def _handle_weather_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pure weather query with conversation context."""
        try:
            query = state.get("user_query", "")
            conversation_history = state.get("conversation_history", [])
            last_weather_data = state.get("last_weather_data")
            last_location = state.get("last_location")
            
            self.logger.info(f"Processing weather query: {query}")
            
            # Check if this is a follow-up question
            is_followup = self._is_weather_followup_question(query, conversation_history)
            
            # Extract location - use last location for follow-up questions
            if is_followup and last_location:
                location = last_location
                self.logger.info(f"Using previous location for follow-up: {location}")
            else:
                location = self._extract_location_from_query(query)
            
            if not location:
                return {
                    **state,
                    "response": (
                        "🗺️ **Cần thông tin địa điểm**\n\n"
                        "Để xem thông tin thời tiết, vui lòng cho biết bạn muốn biết thời tiết ở đâu:\n"
                        "• Tỉnh/thành phố (ví dụ: Hồ Chí Minh, Hà Nội)\n"
                        "• Huyện/quận cụ thể (ví dụ: Buôn Ma Thuột, Quận 1)\n"
                        "• Xã/phường chi tiết (ví dụ: xã Ea Kao)\n\n"
                        "💡 *Ví dụ: 'thời tiết hôm nay ở Đắk Lắk'*"
                    ),
                    "response_type": "weather_query",
                    "search_results": None
                }
            
            # Get weather data - use cached data for recent follow-ups
            if (is_followup and last_weather_data and 
                last_weather_data.get('location') == location and
                datetime.now().timestamp() - last_weather_data.get('timestamp', 0) < 1800):  # 30 minutes
                weather = last_weather_data['weather']
                self.logger.info("Using cached weather data for follow-up")
            else:
                weather = await self.weather_advisor.get_current_weather(location)
            
            if not weather:
                return {
                    **state,
                    "response": (
                        f"❌ **Không thể lấy dữ liệu thời tiết cho '{location}'**\n\n"
                        "Vui lòng kiểm tra lại tên địa điểm hoặc thử với:\n"
                        "• Tên tỉnh/thành phố chính xác\n"
                        "• Tên huyện/quận lớn trong khu vực\n"
                        "• Tên tiếng Việt không dấu\n\n"
                        "💡 *Ví dụ: thay vì 'Krông Năng' hãy thử 'Dak Lak'*"
                    ),
                    "response_type": "weather_query",
                    "search_results": None
                }
            
            # Format pure weather response (without agriculture advice)
            response = self._format_pure_weather_response(weather)
            
            # Update state with weather data
            weather_data_update = {
                'weather': weather,
                'location': location,
                'timestamp': datetime.now().timestamp()
            }
            
            return {
                **state,
                "response": response,
                "response_type": "weather_query",
                "last_weather_data": weather_data_update,
                "last_location": location,
                "search_results": {
                    "has_results": True,
                    "context": response,
                    "sources": [{
                        "type": "weather_api",
                        "location": weather.location_name,
                        "weather_summary": f"{weather.temperature}°C, {weather.description}",
                        "confidence": "95%"
                    }],
                    "max_confidence": 0.95
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in weather query handling: {e}")
            return {
                **state,
                "response": (
                    "⚠️ **Lỗi lấy thông tin thời tiết**\n\n"
                    "Xin lỗi, tôi gặp sự cố khi lấy thông tin thời tiết. "
                    "Vui lòng thử lại sau ít phút.\n\n"
                    "💡 *Có thể thử với tên địa điểm khác hoặc liên hệ hỗ trợ.*"
                ),
                "response_type": "weather_error",
                "search_results": None
            }
    
    async def _handle_weather_agriculture(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle weather-agriculture consultation requests with conversation context."""
        try:
            query = state.get("user_query", "")
            conversation_history = state.get("conversation_history", [])
            last_weather_data = state.get("last_weather_data")
            last_location = state.get("last_location")
            
            self.logger.info(f"Processing weather-agriculture query: {query}")
            self.logger.info(f"State keys: {list(state.keys())}")
            self.logger.info(f"Conversation history: {len(conversation_history) if conversation_history else 0} items")
            self.logger.info(f"Last weather data: {bool(last_weather_data)}")
            self.logger.info(f"Last location: {last_location}")
            
            # Check if this is a follow-up question
            is_followup = self._is_weather_followup_question(query, conversation_history)
            self.logger.info(f"Follow-up detected: {is_followup}")
            
            # Extract location - use last location for follow-up questions
            if is_followup and last_location:
                location = last_location
                self.logger.info(f"Using previous location for follow-up: {location}")
            else:
                location = self._extract_location_from_query(query)
                self.logger.info(f"Extracted location from query: {location}")
            
            if not location:
                return {
                    **state,
                    "response": (
                        "🗺️ **Cần thông tin địa điểm**\n\n"
                        "Để đưa ra tư vấn chính xác, vui lòng cho biết bạn đang ở:\n"
                        "• Tỉnh/thành phố (ví dụ: Đắk Lắk, Lâm Đồng)\n"
                        "• Hoặc huyện/quận cụ thể (ví dụ: Buôn Ma Thuột)\n"
                        "• Hoặc xã/phường chi tiết (ví dụ: xã Ea Kao)\n\n"
                        "💡 *Bạn có thể hỏi: 'thời tiết ở Đắk Lắk như thế nào cho cà phê?'*"
                    ),
                    "response_type": "weather_agriculture",
                    "search_results": None
                }
            
            # Get weather data - use cached data for recent follow-ups
            if (is_followup and last_weather_data and 
                last_weather_data.get('location') == location and
                datetime.now().timestamp() - last_weather_data.get('timestamp', 0) < 1800):  # 30 minutes
                weather = last_weather_data['weather']
                self.logger.info("Using cached weather data for follow-up")
            else:
                self.logger.info(f"Fetching new weather data for location: {location}")
                weather = await self.weather_advisor.get_current_weather(location)
                if not weather:
                    self.logger.warning(f"Weather API failed for {location}, using demo data")
                    weather = self.weather_advisor._get_demo_weather_data(location)
            
            if not weather:
                self.logger.error(f"No weather data available for {location}")
                return {
                    **state,
                    "response": (
                        f"❌ **Không thể lấy dữ liệu thời tiết cho '{location}'**\n\n"
                        "Vui lòng kiểm tra lại tên địa điểm hoặc thử với:\n"
                        "• Tên tỉnh/thành phố chính xác\n"
                        "• Tên huyện/quận lớn trong khu vực\n"
                        "• Tên tiếng Việt không dấu\n\n"
                        "💡 *Ví dụ: thay vì 'Krông Năng' hãy thử 'Dak Lak'*"
                    ),
                    "response_type": "weather_agriculture",
                    "search_results": None
                }
            
            # Extract crop type
            crop_type = self._extract_crop_from_query(query)
            
            # Get agriculture advice
            advice = await self.weather_advisor.generate_agriculture_advice(
                location=location, 
                crop_type=crop_type
            )
            
            # Get detailed context from knowledge base
            detailed_context = self._get_detailed_agriculture_context(crop_type, weather, query)
            
            # Generate comprehensive response
            response = self._generate_comprehensive_weather_agriculture_response(
                weather=weather,
                advice=advice,
                detailed_context=detailed_context,
                user_query=query,
                conversation_history=conversation_history
            )
            
            # Update state with new weather data
            weather_data_update = {
                'weather': weather,
                'location': location,
                'timestamp': datetime.now().timestamp()
            }
            
            # Create sources info
            sources = [{
                "type": "weather_api",
                "location": weather.location_name,
                "weather_summary": f"{weather.temperature}°C, {weather.description}",
                "confidence": "95%"
            }]
            
            # Add knowledge base sources
            if detailed_context.get("sources"):
                sources.extend(detailed_context["sources"])
            
            return {
                **state,
                "response": response,
                "response_type": "weather_agriculture_detailed",
                "last_weather_data": weather_data_update,
                "last_location": location,
                "search_results": {
                    "has_results": True,
                    "context": response,
                    "sources": sources,
                    "max_confidence": 0.95
                },
                "sources": sources
            }
            
        except Exception as e:
            self.logger.error(f"Error in weather-agriculture handling: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                **state,
                "response": (
                    f"⚠️ **Debug: Lỗi xử lý yêu cầu**\n\n"
                    f"Error: {str(e)}\n\n"
                    f"Query: {state.get('user_query', 'N/A')}\n"
                    f"Conversation history length: {len(state.get('conversation_history', []))}\n"
                    f"Last weather data: {bool(state.get('last_weather_data'))}\n"
                    f"Last location: {state.get('last_location', 'N/A')}\n\n"
                    "Vui lòng thử lại hoặc liên hệ hỗ trợ kỹ thuật."
                ),
                "response_type": "weather_agriculture_error",
                "search_results": None
            }
    
    def _extract_location_and_crop(self, query: str) -> tuple[str, str]:
        """Extract location and crop type from query."""
        # Default location (Vietnam)
        location = "Hà Nội"
        crop = "cà phê"
        
        # Extract Vietnamese locations
        location_patterns = [
            r'\b(Hà Nội|hà nội|hanoi)\b',
            r'\b(Hồ Chí Minh|hồ chí minh|sài gòn|saigon|tp hcm)\b',
            r'\b(Đà Nẵng|đà nẵng|da nang)\b',
            r'\b(Đăk Lăk|đăk lăk|dak lak|buôn ma thuột)\b',
            r'\b(Gia Lai|gia lai|pleiku)\b',
            r'\b(Lâm Đồng|lâm đồng|đà lạt)\b',
            r'\b(Kon Tum|kon tum)\b',
            r'\b(Nghệ An|nghệ an|vinh)\b'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location_text = match.group(1).lower()
                if "hà nội" in location_text or "hanoi" in location_text:
                    location = "Hà Nội"
                elif "hồ chí minh" in location_text or "sài gòn" in location_text or "saigon" in location_text:
                    location = "Hồ Chí Minh"
                elif "đà nẵng" in location_text:
                    location = "Đà Nẵng"
                elif "đăk lăk" in location_text or "buôn ma thuột" in location_text:
                    location = "Buôn Ma Thuột"
                elif "gia lai" in location_text or "pleiku" in location_text:
                    location = "Pleiku"
                elif "lâm đồng" in location_text or "đà lạt" in location_text:
                    location = "Đà Lạt"
                break
        
        # Extract crop type
        crop_patterns = [
            r'\b(cà phê|cafe|coffee)\b',
            r'\b(lúa|rice|gạo)\b',
            r'\b(khoai tây|potato)\b',
            r'\b(hồ tiêu|pepper|tiêu)\b',
            r'\b(ngô|corn|bắp)\b',
            r'\b(đậu|bean)\b'
        ]
        
        for pattern in crop_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                crop_text = match.group(1).lower()
                if "cà phê" in crop_text or "cafe" in crop_text or "coffee" in crop_text:
                    crop = "cà phê"
                elif "lúa" in crop_text or "rice" in crop_text:
                    crop = "lúa"
                elif "khoai" in crop_text or "potato" in crop_text:
                    crop = "khoai tây"
                elif "hồ tiêu" in crop_text or "pepper" in crop_text or "tiêu" in crop_text:
                    crop = "hồ tiêu"
                break
        
        return location, crop
    
    def _format_weather_advice_response(self, advice: 'AgricultureAdvice', weather: 'WeatherCondition' = None) -> str:
        """Format weather advice into readable response with real API data."""
        response_parts = []
        
        # Header with real location and time  
        current_time = weather.timestamp if weather else datetime.now()
        location_name = weather.location_name if weather else advice.location
        response_parts.append(f"THỜI TIẾT (lúc {current_time.strftime('%H:%M')})")
        response_parts.append(f"{location_name}")
        response_parts.append(f"CN, {current_time.strftime('%d/%m/%Y')}, {current_time.strftime('%H:%M')}")
        response_parts.append("")
        
        # Main temperature display
        main_temp = weather.temperature if weather else 25.0
        main_desc = weather.description if weather else "Trời quang"
        response_parts.append(f"{main_temp:.1f}°C")
        response_parts.append(f"{main_desc}")
        response_parts.append("")
        
        # Weather details in grid format using REAL API data
        response_parts.append("CHI TIẾT | NHIỆT ĐỘ | TỐC ĐỘ GIÓ (km/h)")
        response_parts.append("")
        
        if weather:
            # Row 1: Temperature, Feels like, Humidity
            response_parts.append(f"Nhiệt độ        | Cảm giác ...     | Độ ẩm")
            response_parts.append(f"{weather.temperature:.1f}°C            | {weather.feels_like:.1f}°C          | {weather.humidity}%")
            response_parts.append("")
            
            # Row 2: Wind, Rain probability, Pressure  
            response_parts.append(f"Gió             | Khả năng ...     | Áp suất kh...")
            response_parts.append(f"{weather.wind_speed:.1f} km/h       | {weather.rain_probability:.0f}%              | {weather.pressure:.0f} hPa")
            response_parts.append("")
            
            # Row 3: Visibility, UV Index, Cloud cover
            response_parts.append(f"Tầm nhìn        | Chỉ số UV       | Độ che mây")
            response_parts.append(f"{weather.visibility:.1f} km         | {weather.uv_index:.2f}            | {weather.clouds}%")
            response_parts.append("")
            
            # Row 4: Dew point, Sunrise, Sunset  
            sunrise_time = weather.sunrise.strftime('%H:%M') if weather.sunrise else "06:15"
            sunset_time = weather.sunset.strftime('%H:%M') if weather.sunset else "18:30"
            response_parts.append(f"Điểm sương      | Bình minh       | Hoàng hôn")
            response_parts.append(f"{weather.dew_point:.1f}°C          | {sunrise_time}           | {sunset_time}")
            response_parts.append("")
        else:
            # Fallback if no weather data
            response_parts.append(f"Nhiệt độ        | Cảm giác như    | Độ ẩm")
            response_parts.append(f"25.0°C            | 27.0°C          | 65%")
            response_parts.append("")
        
        # Weather description
        response_parts.append(f"Tình trạng: {main_desc}")
        response_parts.append("")
        
        # Agriculture advice section
        response_parts.append("=== TƯ VẤN NÔNG NGHIỆP ===")
        response_parts.append(f"Cây trồng: {advice.crop_type}")
        response_parts.append("")
        
        # Recommendations
        if advice.recommendations:
            response_parts.append("KHUYẾN NGHỊ:")
            for i, rec in enumerate(advice.recommendations, 1):
                response_parts.append(f"{i}. {rec}")
            response_parts.append("")
        
        # Optimal activities
        if advice.optimal_activities:
            response_parts.append("NÊN THỰC HIỆN:")
            for i, activity in enumerate(advice.optimal_activities, 1):
                response_parts.append(f"{i}. {activity}")
            response_parts.append("")
        
        # Activities to avoid
        if advice.avoid_activities:
            response_parts.append("NÊN TRÁNH:")
            for i, activity in enumerate(advice.avoid_activities, 1):
                response_parts.append(f"{i}. {activity}")
            response_parts.append("")
        
        # Warnings
        if advice.warnings:
            response_parts.append("CẢNH BÁO:")
            for i, warning in enumerate(advice.warnings, 1):
                response_parts.append(f"{i}. {warning}")
            response_parts.append("")
        
        # Confidence score
        response_parts.append(f"Độ tin cậy: {advice.confidence:.1%}")
        
        return "\n".join(response_parts)
    
    def _handle_unknown_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unknown intent."""
        confidence = state.get("confidence", 0.0)
        
        if confidence < 0.3:
            response = ERROR_RESPONSES["unknown_intent"]
        else:
            response = ERROR_RESPONSES["low_confidence"]
        
        return {
            **state,
            "response": response,
            "response_type": "unknown",
            "search_results": None,
            "sources": []
        }
    
    def _generate_contextual_response(
        self,
        query: str,
        context: str,
        sources: list
    ) -> str:
        """Generate response using LLM with retrieved context."""
        prompt = f"""
        Bạn là một chuyên gia nông nghiệp giàu kinh nghiệm, chuyên tư vấn về canh tác cà phê, lúa, hồ tiêu, ngô, khoai tây và các cây trồng khác. Hãy trả lời câu hỏi một cách tự nhiên, thân thiện và chi tiết như đang tư vấn trực tiếp cho nông dân.

        Câu hỏi: "{query}"

        Kiến thức tham khảo:
        {context}

        Hãy trả lời một cách:
        • **Tự nhiên**: Như đang nói chuyện với nông dân, không cần nhắc đến "tài liệu" hay "theo thông tin"
        • **Thực tế**: Đưa ra lời khuyên cụ thể, có thể áp dụng được ngay
        • **Toàn diện**: Bao gồm nguyên nhân, triệu chứng, cách phòng trừ, thời điểm thích hợp
        • **Có cấu trúc**: Sử dụng tiêu đề và bullet points để dễ đọc
        • **Chi tiết**: Viết đầy đủ 400-600 từ với thông tin hữu ích

        Viết như một chuyên gia đang chia sẻ kinh nghiệm thực tế, không phải đang trích dẫn tài liệu.
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Add sources in a natural way
            if sources:
                content += f"\n\n---\n*Thông tin tham khảo từ: {', '.join([s.get('filename', 'tài liệu chuyên ngành').replace('.pdf', '') for s in sources])}"
            
            return content
            
        except Exception as e:
            self.logger.error("Failed to generate contextual response", error=str(e))
            return self.search_tools.format_response(context, query, sources)
    
    def _generate_general_response(self, query: str) -> str:
        """Generate general response for questions without context."""
        prompt = f"""
        Người dùng hỏi: "{query}"

        Bạn là một chuyên gia nông nghiệp thân thiện. Hãy trả lời một cách lịch sự rằng bạn chuyên 
        tư vấn về canh tác các loại cây trồng như cà phê, lúa, hồ tiêu, ngô, khoai tây dựa trên 
        kiến thức chuyên môn có sẵn.

        Gợi ý họ hỏi những câu hỏi cụ thể hơn về:
        - Kỹ thuật canh tác
        - Phòng trừ sâu bệnh  
        - Chăm sóc cây trồng
        - Biện pháp tăng năng suất

        Trả lời ngắn gọn, thân thiện và hướng dẫn cụ thể.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            self.logger.error("Failed to generate general response", error=str(e))
            return ERROR_RESPONSES["no_results"]
    
    def _create_error_response(self, state: Dict[str, Any], error: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            **state,
            "response": ERROR_RESPONSES["processing_error"],
            "response_type": "error",
            "search_results": None,
            "sources": [],
            "error": error,
            "action_completed": False
        }
    
    def _extract_location_from_query(self, query: str) -> str:
        """Extract location from weather query using smart detection."""
        try:
            # First try LLM-based extraction for better accuracy
            location = self._extract_location_with_llm(query)
            if location and location != "Unknown":
                return location
        except Exception as e:
            self.logger.warning(f"LLM location extraction failed: {e}")
        
        # Fallback to pattern-based extraction
        return self._extract_location_with_patterns(query)
    
    def _extract_location_with_llm(self, query: str) -> str:
        """Use LLM to extract location from query with detailed administrative levels."""
        prompt = f"""
        Hãy trích xuất địa điểm chính xác nhất từ câu hỏi sau.
        Ưu tiên trích xuất đến cấp xã/phường/thị trấn nếu có, sau đó đến huyện/quận, rồi tỉnh/thành phố.
        Trả về tên địa điểm bằng tiếng Anh để sử dụng với API thời tiết.

        Câu hỏi: "{query}"

        Quy tắc trích xuất theo thứ tự ưu tiên:
        1. **Xã/Phường/Thị trấn**: Nếu có đề cập xã/phường cụ thể
        2. **Huyện/Quận**: Nếu có đề cập huyện/quận cụ thể  
        3. **Tỉnh/Thành phố**: Nếu chỉ có tỉnh/thành phố
        4. **Thành phố lớn**: Ưu tiên thành phố chính của tỉnh

        Ví dụ chi tiết:
        - "xã Tân Phú, huyện Châu Thành, An Giang" → "Tan Phu, Chau Thanh, An Giang"
        - "phường 1, quận 1, TP HCM" → "Ward 1, District 1, Ho Chi Minh City"  
        - "huyện Đắk Pơ, Gia Lai" → "Dak Po, Gia Lai"
        - "thị trấn Pleiku, Gia Lai" → "Pleiku, Gia Lai"
        - "Gia Lai" → "Pleiku" (thành phố chính)
        - "Lâm Đồng" → "Da Lat" (thành phố chính)
        - "Đắk Lắk" → "Buon Ma Thuot" (thành phố chính)
        - "Khánh Hòa" → "Nha Trang" (thành phố chính)

        Đặc biệt lưu ý:
        - Với các tỉnh miền núi/nông thôn: trả về tên huyện hoặc thị trấn chính
        - Với thành phố lớn: có thể trả về quận/huyện cụ thể
        - Nếu không rõ địa điểm: trả về "Ho Chi Minh City"

        Format trả về:
        - Nếu có đầy đủ thông tin: "Tên cụ thể, Huyện/Quận, Tỉnh/Thành"
        - Nếu chỉ có huyện: "Tên huyện, Tỉnh" 
        - Nếu chỉ có tỉnh: "Thành phố chính của tỉnh"

        Chỉ trả về tên địa điểm, không giải thích:
        """
        
        try:
            response = self.llm.invoke(prompt)
            location = response.content.strip().strip('"\'')
            
            # Validate the response
            if location and len(location) > 2 and location != "Unknown":
                self.logger.info(f"LLM extracted detailed location: {location} from query: {query}")
                return location
            
        except Exception as e:
            self.logger.error(f"LLM detailed location extraction error: {e}")
        
        return None
    
    def _extract_location_with_patterns(self, query: str) -> str:
        """Extract location using regex patterns with detailed administrative levels."""
        query_lower = query.lower()
        
        # Pattern for detailed administrative structure
        # Look for: xã/phường + huyện/quận + tỉnh/thành phố
        detailed_patterns = [
            # Full structure: xã/phường + huyện + tỉnh
            r'(?:xã|phường|thị trấn)\s+([^,]+),?\s*(?:huyện|quận|thành phố|tp)\s+([^,]+),?\s*(?:tỉnh|thành phố|tp)?\s*([^,\.]+)',
            # Huyện + tỉnh
            r'(?:huyện|quận|thành phố|tp)\s+([^,]+),?\s*(?:tỉnh|thành phố|tp)?\s*([^,\.]+)',
            # Just tỉnh/thành phố
            r'(?:tỉnh|thành phố|tp)\s+([^,\.]+)',
        ]
        
        for pattern in detailed_patterns:
            import re
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                if len(groups) == 3:  # xã + huyện + tỉnh
                    xa, huyen, tinh = groups
                    location = f"{xa.strip().title()}, {huyen.strip().title()}, {tinh.strip().title()}"
                elif len(groups) == 2:  # huyện + tỉnh
                    huyen, tinh = groups
                    location = f"{huyen.strip().title()}, {tinh.strip().title()}"
                else:  # chỉ tỉnh
                    tinh = groups[0]
                    location = self._get_main_city_of_province(tinh.strip().title())
                
                self.logger.info(f"Detailed pattern matched: {location} from query: {query}")
                return location
        
        # Extended Vietnamese locations with district/commune level
        location_mappings = {
            # Ho Chi Minh City districts
            'quận 1|district 1': 'District 1, Ho Chi Minh City',
            'quận 3|district 3': 'District 3, Ho Chi Minh City', 
            'quận 7|district 7': 'District 7, Ho Chi Minh City',
            'quận bình thạnh|binh thanh': 'Binh Thanh District, Ho Chi Minh City',
            'quận thủ đức|thu duc': 'Thu Duc District, Ho Chi Minh City',
            'quận gò vấp|go vap': 'Go Vap District, Ho Chi Minh City',
            
            # Hanoi districts
            'quận ba đình|ba dinh': 'Ba Dinh District, Hanoi',
            'quận hoàn kiếm|hoan kiem': 'Hoan Kiem District, Hanoi',
            'quận đống đa|dong da': 'Dong Da District, Hanoi',
            'quận cầu giấy|cau giay': 'Cau Giay District, Hanoi',
            
            # Gia Lai detailed
            'huyện đắk pơ|dak po': 'Dak Po, Gia Lai',
            'huyện chư prông|chu prong': 'Chu Prong, Gia Lai',
            'huyện ia grai': 'Ia Grai, Gia Lai',
            'thị xã an khê|an khe': 'An Khe, Gia Lai',
            'huyện kong chro': 'Kong Chro, Gia Lai',
            
            # Đắk Lắk detailed
            'huyện ea hleo': 'Ea H Leo, Dak Lak',
            'huyện krông búk|krong buk': 'Krong Buk, Dak Lak',
            'huyện m đrắk|m drak': 'M Drak, Dak Lak',
            'thị xã buôn hồ|buon ho': 'Buon Ho, Dak Lak',
            
            # Lâm Đồng detailed
            'huyện đức trọng|duc trong': 'Duc Trong, Lam Dong',
            'huyện lạc dương|lac duong': 'Lac Duong, Lam Dong',
            'huyện đơn dương|don duong': 'Don Duong, Lam Dong',
            'thị xã bảo lộc|bao loc': 'Bao Loc, Lam Dong',
            
            # Major provinces with main cities
            'hồ chí minh|sài gòn|tp hcm|thành phố hồ chí minh': 'Ho Chi Minh City',
            'hà nội|hanoi|thủ đô': 'Hanoi',
            'đà nẵng|da nang': 'Da Nang',
            'cần thơ|can tho': 'Can Tho',
            'hải phòng|hai phong': 'Hai Phong',
            
            # Central Vietnam
            'đà lạt|da lat|lâm đồng|lam dong': 'Da Lat',
            'gia lai|pleiku': 'Pleiku',
            'đắk lắk|dak lak|buôn ma thuột|buon ma thuot': 'Buon Ma Thuot',
            'khánh hòa|khanh hoa|nha trang': 'Nha Trang',
            'huế|hue|thừa thiên huế': 'Hue',
            'quảng nam|quang nam|hội an|hoi an': 'Hoi An',
            'bình định|binh dinh|quy nhon': 'Quy Nhon',
            'phú yên|phu yen|tuy hoa': 'Tuy Hoa',
            
            # Northern Vietnam  
            'nghệ an|nghe an|vinh': 'Vinh',
            'thái nguyên|thai nguyen': 'Thai Nguyen',
            'lạng sơn|lang son': 'Lang Son',
            'hạ long|ha long|quảng ninh': 'Ha Long',
            'sapa|sa pa|lào cai': 'Sapa',
            'cao bằng|cao bang': 'Cao Bang',
            'hà giang|ha giang': 'Ha Giang',
            'điện biên|dien bien': 'Dien Bien Phu',
            
            # Southern Vietnam
            'vũng tàu|vung tau|bà rịa|ba ria': 'Vung Tau',
            'biên hòa|bien hoa|đồng nai|dong nai': 'Bien Hoa',
            'bình dương|binh duong|thủ dầu một': 'Thu Dau Mot',
            'tây ninh|tay ninh': 'Tay Ninh',
            'phú quốc|phu quoc': 'Phu Quoc',
            'cà mau|ca mau': 'Ca Mau',
            
            # Mekong Delta
            'an giang|long xuyên|long xuyen': 'Long Xuyen',
            'sóc trăng|soc trang': 'Soc Trang',
            'vĩnh long|vinh long': 'Vinh Long',
            'bến tre|ben tre': 'Ben Tre',
            'trà vinh|tra vinh': 'Tra Vinh',
            'hậu giang|hau giang|vị thanh': 'Vi Thanh',
            'kiên giang|kien giang|rạch giá': 'Rach Gia',
            
            # Central Highlands
            'kon tum': 'Kon Tum',
            'đắk nông|dak nong': 'Gia Nghia',
        }
        
        # Try to match patterns with priority for more specific locations
        matched_locations = []
        for pattern, location in location_mappings.items():
            if any(keyword in query_lower for keyword in pattern.split('|')):
                matched_locations.append((location, len(pattern.split('|'))))
        
        # Sort by specificity (more keywords = more specific)
        if matched_locations:
            # Return the most specific match
            best_match = sorted(matched_locations, key=lambda x: x[1], reverse=True)[0]
            location = best_match[0]
            self.logger.info(f"Specific pattern matched: {location} from query: {query}")
            return location
        
        # Final fallback: look for any administrative indicators
        admin_indicators = [
            (r'xã\s+(\w+(?:\s+\w+)?)', 'commune'),
            (r'phường\s+(\w+(?:\s+\w+)?)', 'ward'),
            (r'huyện\s+(\w+(?:\s+\w+)?)', 'district'), 
            (r'quận\s+(\w+(?:\s+\w+)?)', 'district'),
            (r'thị xã\s+(\w+(?:\s+\w+)?)', 'town'),
            (r'thành phố\s+(\w+(?:\s+\w+)?)', 'city')
        ]
        
        for pattern, admin_type in admin_indicators:
            import re
            match = re.search(pattern, query_lower)
            if match:
                place_name = match.group(1).title()
                self.logger.info(f"Administrative unit detected: {place_name} ({admin_type})")
                return place_name
        
        # Ultimate fallback
        self.logger.info("No detailed location detected, using default: Ho Chi Minh City")
        return 'Ho Chi Minh City'
    
    def _get_main_city_of_province(self, province: str) -> str:
        """Get main city/capital of a province."""
        province_capitals = {
            'Gia Lai': 'Pleiku',
            'Lam Dong': 'Da Lat', 
            'Dak Lak': 'Buon Ma Thuot',
            'Khanh Hoa': 'Nha Trang',
            'Binh Dinh': 'Quy Nhon',
            'Phu Yen': 'Tuy Hoa',
            'Quang Nam': 'Hoi An',
            'Nghe An': 'Vinh',
            'Thai Nguyen': 'Thai Nguyen',
            'Cao Bang': 'Cao Bang',
            'Ha Giang': 'Ha Giang',
            'An Giang': 'Long Xuyen',
            'Ca Mau': 'Ca Mau',
            'Soc Trang': 'Soc Trang',
            'Vinh Long': 'Vinh Long',
            'Ben Tre': 'Ben Tre',
            'Kon Tum': 'Kon Tum'
        }
        
        return province_capitals.get(province, province)
    
    def _format_pure_weather_response(self, weather: WeatherCondition) -> str:
        """Format pure weather response without agriculture advice."""
        from datetime import datetime
        
        current_time = datetime.now().strftime("%H:%M, %A")
        
        response = f"""📍 **{weather.location_name}**
🕐 {current_time}
{"-" * 40}

🌡️ **Nhiệt độ hiện tại**
     {weather.temperature}°C
     Cảm giác như {weather.feels_like}°C
     {weather.description}

💧 **Độ ẩm**: {weather.humidity}%
💨 **Gió**: {weather.wind_speed:.1f} km/h {weather.wind_direction_text}
🔆 **Áp suất**: {weather.pressure} hPa
👁️ **Tầm nhìn**: {weather.visibility} km
☀️ **Chỉ số UV**: {weather.uv_index} ({self.weather_advisor._get_uv_description(weather.uv_index)})
☁️ **Mây**: {weather.clouds}%
💧 **Điểm sương**: {weather.dew_point:.1f}°C"""

        if weather.sunrise and weather.sunset:
            response += f"""
🌅 **Bình minh**: {weather.sunrise.strftime('%H:%M')}
🌇 **Hoàng hôn**: {weather.sunset.strftime('%H:%M')}"""

        if weather.rain_probability:
            response += f"""
🌧️ **Xác suất mưa**: {weather.rain_probability}%"""
        
        return response
    
    def _is_weather_followup_question(self, query: str, conversation_history: List[Dict]) -> bool:
        """Check if current query is a follow-up to previous weather question."""
        if not conversation_history:
            return False
        
        # Check last conversation turn
        last_turn = conversation_history[-1] if conversation_history else {}
        last_intent = last_turn.get("intent")
        
        # Check if last question was about weather
        if last_intent not in ["weather_query", "weather_agriculture"]:
            return False
        
        # Check if current query is contextual (no explicit location)
        query_lower = query.lower()
        followup_indicators = [
            "với thời tiết này", "trong điều kiện này", "theo thông tin trên",
            "dựa vào thời tiết", "nên làm gì", "có phù hợp", "thì sao",
            "làm gì tiếp", "có nên", "tương tự", "như vậy"
        ]
        
        return any(indicator in query_lower for indicator in followup_indicators)
    
    def _get_detailed_agriculture_context(self, crop: str, weather: 'WeatherCondition', query: str) -> Dict[str, Any]:
        """Get detailed agriculture context from knowledge base."""
        try:
            # Enhanced search query combining crop and weather conditions
            search_query = f"""
            {crop} {query} 
            nhiệt độ {weather.temperature}°C 
            độ ẩm {weather.humidity}% 
            {weather.description}
            tưới nước phân bón chăm sóc
            """
            
            # Search knowledge base with expanded context
            search_result = self.search_tools.search_knowledge_base(
                query=search_query,
                limit=10,  # Get more results for comprehensive advice
                min_score=0.2  # Lower threshold for broader context
            )
            
            return {
                "context": search_result.get("context", ""),
                "sources": search_result.get("sources", []),
                "has_results": search_result.get("has_results", False)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get detailed agriculture context: {e}")
            return {"context": "", "sources": [], "has_results": False}
    
    def _generate_comprehensive_weather_agriculture_response(
        self, 
        weather: 'WeatherCondition',
        advice: 'AgricultureAdvice', 
        detailed_context: Dict[str, Any],
        user_query: str,
        conversation_history: List[Dict]
    ) -> str:
        """Generate comprehensive response combining weather + detailed knowledge."""
        
        # Build context from conversation history
        conversation_context = ""
        if conversation_history:
            recent_context = conversation_history[-2:]  # Last 2 turns
            for turn in recent_context:
                conversation_context += f"Q: {turn.get('user_query', '')}\nA: {turn.get('response', '')[:200]}...\n\n"
        
        # Enhanced prompt for comprehensive response
        prompt = f"""
        Bạn là chuyên gia nông nghiệp hàng đầu với 20+ năm kinh nghiệm tư vấn cà phê và cây trồng.
        Hãy tạo ra lời tư vấn chi tiết, toàn diện dựa trên thông tin thời tiết thực tế và kiến thức chuyên môn.

        THÔNG TIN THỜI TIẾT HIỆN TẠI:
        📍 Địa điểm: {weather.location_name}
        🌡️ Nhiệt độ: {weather.temperature}°C (cảm giác {weather.feels_like}°C)
        💧 Độ ẩm: {weather.humidity}%
        💨 Gió: {weather.wind_speed:.1f} km/h {weather.wind_direction_text}
        ☀️ UV: {weather.uv_index} 
        🔆 Áp suất: {weather.pressure} hPa
        ☁️ Mây: {weather.clouds}%
        📝 Tình trạng: {weather.description}

        CÂU HỎI HIỆN TẠI: "{user_query}"

        NGỮ CẢNH HỘI THOẠI TRƯỚC:
        {conversation_context}

        KIẾN THỨC CHUYÊN MÔN THAM KHẢO:
        {detailed_context.get('context', 'Không có thông tin bổ sung')}

        NHIỆM VỤ:
        1. Phân tích chi tiết tác động của thời tiết lên {advice.crop_type}
        2. Đưa ra khuyến nghị cụ thể, có thể thực hiện ngay
        3. Giải thích lý do khoa học đằng sau mỗi khuyến nghị
        4. Bao gồm lưu ý về thời điểm, cách thức thực hiện
        5. Cảnh báo rủi ro và cách phòng tránh

        YÊU CẦU TRÌNH BÀY:
        • **Tự nhiên**: Như đang tư vấn trực tiếp cho nông dân
        • **Chi tiết**: 500-800 từ với thông tin hữu ích
        • **Có cấu trúc**: Sử dụng heading và bullet points rõ ràng
        • **Thực tế**: Khuyến nghị có thể áp dụng ngay với điều kiện hiện tại
        • **Khoa học**: Giải thích cơ sở khoa học khi cần thiết

        FORMAT RESPONSE:
        
        ## 🌤️ PHÂN TÍCH THỜI TIẾT & TÁC ĐỘNG

        ## 💡 KHUYẾN NGHỊ CHI TIẾT

        ## ⚠️ LƯU Ý QUAN TRỌNG

        ## 📅 KẾ HOẠCH THỰC HIỆN

        Hãy viết như một chuyên gia đang chia sẻ kinh nghiệm thực tế, không phải trích dẫn tài liệu.
        """
        
        try:
            response = self.llm.invoke(prompt)
            comprehensive_response = response.content.strip()
            
            # Add weather display header
            weather_header = self.weather_advisor.format_detailed_weather_response(weather, advice)
            
            # Combine weather display + comprehensive advice
            final_response = f"{weather_header}\n\n{'='*60}\n🧑‍🌾 **TƯ VẤN CHUYÊN SÂU TỪ CHUYÊN GIA**\n{'='*60}\n\n{comprehensive_response}"
            
            # Add sources if available
            if detailed_context.get("sources"):
                source_names = [s.get('filename', 'tài liệu chuyên ngành').replace('.pdf', '') for s in detailed_context["sources"][:3]]
                final_response += f"\n\n---\n*📚 Tham khảo từ: {', '.join(source_names)}*"
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"Failed to generate comprehensive response: {e}")
            # Fallback to basic weather response
            return self.weather_advisor.format_detailed_weather_response(weather, advice)
    
    def _extract_crop_from_query(self, query: str) -> str:
        """Extract crop type from user query."""
        query_lower = query.lower()
        
        # Common Vietnamese crop names
        crop_patterns = {
            'cà phê': ['cà phê', 'cafe', 'coffee'],
            'lúa': ['lúa', 'lúa gạo', 'gạo', 'rice'],
            'tiêu': ['tiêu', 'hạt tiêu', 'pepper'],
            'cao su': ['cao su', 'rubber'],
            'điều': ['điều', 'hạt điều', 'cashew'],
            'dừa': ['dừa', 'coconut'],
            'chuối': ['chuối', 'banana'],
            'xoài': ['xoài', 'mango'],
            'bưởi': ['bưởi', 'pomelo'],
            'cam': ['cam', 'orange'],
            'chanh': ['chanh', 'lemon'],
            'khoai lang': ['khoai lang', 'sweet potato'],
            'khoai tây': ['khoai tây', 'potato'],
            'ngô': ['ngô', 'bắp', 'corn', 'maize'],
            'đậu': ['đậu', 'bean'],
            'rau': ['rau', 'vegetable'],
            'hoa': ['hoa', 'flower']
        }
        
        # Check for crop mentions in query
        for crop, patterns in crop_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    self.logger.info(f"Detected crop: {crop}")
                    return crop
        
        # Default to coffee if no specific crop mentioned
        self.logger.info("No specific crop detected, defaulting to 'cà phê'")
        return 'cà phê'
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()