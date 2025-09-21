"""Agricultural weather advisory system."""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

from config import get_logger, LoggerMixin, settings


@dataclass
class WeatherCondition:
    """Weather condition data."""
    temperature: float
    humidity: float
    rainfall: float
    wind_speed: float
    description: str
    timestamp: datetime
    # Additional detailed weather info
    feels_like: float = 0
    pressure: float = 0
    visibility: float = 0
    uv_index: float = 0
    dew_point: float = 0
    wind_direction: float = 0
    wind_direction_text: str = ""
    clouds: float = 0
    sunrise: Optional[datetime] = None
    sunset: Optional[datetime] = None
    rain_probability: float = 0
    location_name: str = ""


@dataclass
class AgricultureAdvice:
    """Agriculture advice based on weather."""
    crop_type: str
    location: str
    weather_summary: str
    recommendations: List[str]
    warnings: List[str]
    optimal_activities: List[str]
    avoid_activities: List[str]
    confidence: float


class AgricultureWeatherAdvisor(LoggerMixin):
    """Provides agriculture advice based on weather conditions."""
    
    def __init__(self):
        """Initialize weather advisor."""
        self.api_key = settings.weather_api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
        # Agriculture-specific weather thresholds
        self.thresholds = {
            "coffee": {
                "temp_min": 15, "temp_max": 30,
                "humidity_min": 60, "humidity_max": 80,
                "rainfall_max": 200,  # mm/month
                "wind_max": 15  # km/h
            },
            "rice": {
                "temp_min": 20, "temp_max": 35,
                "humidity_min": 70, "humidity_max": 90,
                "rainfall_min": 100, "rainfall_max": 400,
                "wind_max": 20
            },
            "potato": {
                "temp_min": 15, "temp_max": 25,
                "humidity_min": 65, "humidity_max": 85,
                "rainfall_max": 150,
                "wind_max": 25
            },
            "pepper": {
                "temp_min": 20, "temp_max": 32,
                "humidity_min": 60, "humidity_max": 85,
                "rainfall_max": 250,
                "wind_max": 15
            }
        }
        
    async def get_current_weather(self, location: str) -> Optional[WeatherCondition]:
        """
        Get current weather for location with fallback to demo data.
        
        Args:
            location: City name or coordinates
            
        Returns:
            Current weather condition or demo data if API fails
        """
        try:
            url = f"{self.base_url}/weather"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric",
                "lang": "vi"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Calculate dew point
                        temp = data["main"]["temp"]
                        humidity = data["main"]["humidity"]
                        dew_point = self._calculate_dew_point(temp, humidity)
                        
                        # Get wind direction text
                        wind_deg = data["wind"].get("deg", 0)
                        wind_dir_text = self._get_wind_direction_text(wind_deg)
                        
                        # Calculate rain probability from clouds
                        clouds_percent = data["clouds"]["all"]
                        rain_prob = min(clouds_percent * 0.8, 100) if clouds_percent > 50 else 0
                        
                        return WeatherCondition(
                            temperature=data["main"]["temp"],
                            humidity=data["main"]["humidity"],
                            rainfall=data.get("rain", {}).get("1h", 0),
                            wind_speed=data["wind"]["speed"] * 3.6,  # Convert m/s to km/h
                            description=data["weather"][0]["description"],
                            timestamp=datetime.now(),
                            # Additional detailed info from API
                            feels_like=data["main"]["feels_like"],
                            pressure=data["main"]["pressure"],
                            visibility=data.get("visibility", 10000) / 1000,  # Convert m to km
                            dew_point=dew_point,
                            wind_direction=wind_deg,
                            wind_direction_text=wind_dir_text,
                            clouds=clouds_percent,
                            sunrise=datetime.fromtimestamp(data["sys"]["sunrise"]),
                            sunset=datetime.fromtimestamp(data["sys"]["sunset"]),
                            rain_probability=rain_prob,
                            location_name=data["name"],
                            uv_index=0  # Will be fetched separately
                        )
                    else:
                        self.logger.warning("Weather API request failed, using demo data", 
                                          status=response.status, location=location)
                        return self._get_demo_weather_data(location)
                        
        except Exception as e:
            self.logger.warning("Weather API failed, using demo data", error=str(e), location=location)
            return self._get_demo_weather_data(location)
    
    async def get_weather_forecast(self, location: str, days: int = 5) -> List[WeatherCondition]:
        """
        Get weather forecast for location.
        
        Args:
            location: City name
            days: Number of forecast days
            
        Returns:
            List of weather conditions
        """
        try:
            url = f"{self.base_url}/forecast"
            params = {
                "q": location,
                "appid": self.api_key,
                "units": "metric",
                "lang": "vi"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        forecasts = []
                        
                        for item in data["list"][:days * 8]:  # 8 forecasts per day (3-hour intervals)
                            forecasts.append(WeatherCondition(
                                temperature=item["main"]["temp"],
                                humidity=item["main"]["humidity"],
                                rainfall=item.get("rain", {}).get("3h", 0),
                                wind_speed=item["wind"]["speed"] * 3.6,
                                description=item["weather"][0]["description"],
                                timestamp=datetime.fromtimestamp(item["dt"])
                            ))
                        
                        return forecasts
                    else:
                        self.logger.warning("Forecast API request failed", 
                                          status=response.status, location=location)
                        return []
                        
        except Exception as e:
            self.logger.error("Failed to get forecast", error=str(e), location=location)
            return []
    
    def analyze_weather_for_crop(self, weather: WeatherCondition, crop_type: str) -> Dict[str, Any]:
        """
        Analyze weather conditions for specific crop.
        
        Args:
            weather: Current weather condition
            crop_type: Type of crop (coffee, rice, potato, pepper)
            
        Returns:
            Analysis results
        """
        crop_type = crop_type.lower()
        if crop_type not in self.thresholds:
            crop_type = "coffee"  # Default to coffee
        
        thresholds = self.thresholds[crop_type]
        analysis = {
            "crop": crop_type,
            "suitable": True,
            "issues": [],
            "recommendations": [],
            "score": 100
        }
        
        # Temperature analysis
        if weather.temperature < thresholds["temp_min"]:
            analysis["suitable"] = False
            analysis["issues"].append(f"Nhiệt độ quá thấp ({weather.temperature}°C)")
            analysis["recommendations"].append("Che chắn cây trồng, tưới nước ấm")
            analysis["score"] -= 30
        elif weather.temperature > thresholds["temp_max"]:
            analysis["suitable"] = False
            analysis["issues"].append(f"Nhiệt độ quá cao ({weather.temperature}°C)")
            analysis["recommendations"].append("Tăng tưới nước, che bóng mát")
            analysis["score"] -= 25
        
        # Humidity analysis
        if weather.humidity < thresholds["humidity_min"]:
            analysis["issues"].append(f"Độ ẩm thấp ({weather.humidity}%)")
            analysis["recommendations"].append("Tăng tưới phun sương")
            analysis["score"] -= 15
        elif weather.humidity > thresholds["humidity_max"]:
            analysis["issues"].append(f"Độ ẩm cao ({weather.humidity}%)")
            analysis["recommendations"].append("Tăng thông gió, cẩn thận bệnh nấm")
            analysis["score"] -= 10
        
        # Rainfall analysis
        if "rainfall_min" in thresholds and weather.rainfall < thresholds["rainfall_min"]:
            analysis["issues"].append("Lượng mưa không đủ")
            analysis["recommendations"].append("Bổ sung tưới nước")
            analysis["score"] -= 20
        elif weather.rainfall > thresholds["rainfall_max"]:
            analysis["issues"].append("Mưa quá nhiều")
            analysis["recommendations"].append("Thoát nước, phòng ngừa úng")
            analysis["score"] -= 20
        
        # Wind analysis
        if weather.wind_speed > thresholds["wind_max"]:
            analysis["issues"].append(f"Gió mạnh ({weather.wind_speed:.1f} km/h)")
            analysis["recommendations"].append("Chằng chống cây trồng")
            analysis["score"] -= 15
        
        analysis["score"] = max(0, analysis["score"])
        return analysis
    
    def _calculate_dew_point(self, temp: float, humidity: float) -> float:
        """Calculate dew point using Magnus formula."""
        import math
        a = 17.27
        b = 237.7
        alpha = ((a * temp) / (b + temp)) + math.log(humidity / 100.0)
        return (b * alpha) / (a - alpha)
    
    def _get_wind_direction_text(self, degrees: float) -> str:
        """Convert wind direction degrees to compass text."""
        directions = [
            "Bắc", "Bắc Đông Bắc", "Đông Bắc", "Đông Đông Bắc",
            "Đông", "Đông Đông Nam", "Đông Nam", "Nam Đông Nam",
            "Nam", "Nam Tây Nam", "Tây Nam", "Tây Tây Nam",
            "Tây", "Tây Tây Bắc", "Tây Bắc", "Bắc Tây Bắc"
        ]
        index = int((degrees + 11.25) / 22.5) % 16
        return directions[index]
    
    async def get_uv_index(self, location: str) -> float:
        """Get UV index for location."""
        try:
            # Get coordinates first
            url = f"{self.base_url}/weather"
            params = {
                "q": location,
                "appid": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        lat = data["coord"]["lat"]
                        lon = data["coord"]["lon"]
                        
                        # Get UV index
                        uv_url = f"http://api.openweathermap.org/data/2.5/uvi"
                        uv_params = {
                            "lat": lat,
                            "lon": lon,
                            "appid": self.api_key
                        }
                        
                        async with session.get(uv_url, params=uv_params) as uv_response:
                            if uv_response.status == 200:
                                uv_data = await uv_response.json()
                                return uv_data.get("value", 0)
                                
        except Exception as e:
            self.logger.warning("Failed to get UV index", error=str(e))
            
        return 0
    
    async def generate_agriculture_advice(
        self, 
        location: str, 
        crop_type: str = "cà phê",
        include_forecast: bool = True
    ) -> Optional[AgricultureAdvice]:
        """
        Generate comprehensive agriculture advice.
        
        Args:
            location: Location name
            crop_type: Type of crop
            include_forecast: Include forecast analysis
            
        Returns:
            Agriculture advice or None if failed
        """
        try:
            # Normalize crop type
            crop_mapping = {
                "cà phê": "coffee", "cafe": "coffee", "coffee": "coffee",
                "lúa": "rice", "rice": "rice",
                "khoai tây": "potato", "potato": "potato",
                "hồ tiêu": "pepper", "pepper": "pepper"
            }
            normalized_crop = crop_mapping.get(crop_type.lower(), "coffee")
            
            # Get current weather
            current_weather = await self.get_current_weather(location)
            if not current_weather:
                return None
            
            # Get UV index
            uv_index = await self.get_uv_index(location)
            current_weather.uv_index = uv_index
            
            # Analyze current conditions
            current_analysis = self.analyze_weather_for_crop(current_weather, normalized_crop)
            
            # Get forecast if requested
            forecast_analysis = []
            if include_forecast:
                forecast = await self.get_weather_forecast(location, days=3)
                for weather in forecast[:8]:  # Next 24 hours (8 x 3-hour intervals)
                    forecast_analysis.append(
                        self.analyze_weather_for_crop(weather, normalized_crop)
                    )
            
            # Generate comprehensive advice
            advice = self._compile_advice(
                location, crop_type, current_weather, 
                current_analysis, forecast_analysis
            )
            
            self.logger.info("Agriculture advice generated", 
                           location=location, crop=crop_type, 
                           score=current_analysis["score"])
            
            return advice
            
        except Exception as e:
            self.logger.error("Failed to generate advice", error=str(e))
            return None
    
    def _compile_advice(
        self, 
        location: str, 
        crop_type: str,
        weather: WeatherCondition,
        current_analysis: Dict[str, Any],
        forecast_analysis: List[Dict[str, Any]]
    ) -> AgricultureAdvice:
        """Compile comprehensive agriculture advice."""
        
        recommendations = []
        warnings = []
        optimal_activities = []
        avoid_activities = []
        
        # Current weather recommendations
        recommendations.extend(current_analysis["recommendations"])
        
        if current_analysis["issues"]:
            warnings.extend(current_analysis["issues"])
        
        # Weather-specific activities
        if weather.temperature >= 20 and weather.temperature <= 28:
            optimal_activities.append("Thời điểm tốt để trồng cây")
            optimal_activities.append("Phun thuốc trừ sâu hiệu quả")
        
        if weather.rainfall < 5:  # Low rain
            optimal_activities.append("Tưới nước cho cây")
            optimal_activities.append("Bón phân")
        else:
            avoid_activities.append("Tránh phun thuốc khi mưa")
            avoid_activities.append("Tránh bón phân hóa học")
        
        if weather.wind_speed > 20:
            avoid_activities.append("Tránh phun thuốc")
            avoid_activities.append("Tránh cắt tỉa cành")
        
        # Forecast-based recommendations
        if forecast_analysis:
            avg_score = sum(f["score"] for f in forecast_analysis) / len(forecast_analysis)
            if avg_score < 60:
                warnings.append("Dự báo thời tiết không thuận lợi trong 24h tới")
                recommendations.append("Chuẩn bị biện pháp bảo vệ cây trồng")
        
        # Crop-specific advice
        crop_advice = self._get_crop_specific_advice(crop_type, weather)
        recommendations.extend(crop_advice["recommendations"])
        optimal_activities.extend(crop_advice["activities"])
        
        weather_summary = (
            f"Nhiệt độ: {weather.temperature}°C, "
            f"Độ ẩm: {weather.humidity}%, "
            f"Mưa: {weather.rainfall:.1f}mm, "
            f"Gió: {weather.wind_speed:.1f}km/h - {weather.description}"
        )
        
        return AgricultureAdvice(
            crop_type=crop_type,
            location=location,
            weather_summary=weather_summary,
            recommendations=list(set(recommendations)),
            warnings=list(set(warnings)),
            optimal_activities=list(set(optimal_activities)),
            avoid_activities=list(set(avoid_activities)),
            confidence=current_analysis["score"] / 100
        )
    
    def _get_crop_specific_advice(self, crop_type: str, weather: WeatherCondition) -> Dict[str, List[str]]:
        """Get crop-specific advice based on weather."""
        crop_lower = crop_type.lower()
        
        if "cà phê" in crop_lower or "coffee" in crop_lower:
            return {
                "recommendations": [
                    "Kiểm tra độ ẩm đất thường xuyên",
                    "Theo dõi sâu bệnh do thời tiết",
                    "Điều chỉnh tán cây phù hợp"
                ],
                "activities": [
                    "Thu hoạch cà phê chín" if weather.temperature < 30 else "Chờ thời tiết mát hơn để thu hoạch",
                    "Tỉa cành cà phê" if weather.wind_speed < 15 else "Hoãn việc tỉa cành"
                ]
            }
        
        elif "lúa" in crop_lower or "rice" in crop_lower:
            return {
                "recommendations": [
                    "Kiểm tra mực nước ruộng",
                    "Theo dõi sâu bệnh lúa",
                    "Đảm bảo thoát nước tốt"
                ],
                "activities": [
                    "Gieo sạ lúa" if 20 <= weather.temperature <= 30 else "Chờ nhiệt độ phù hợp",
                    "Bón phân cho lúa" if weather.rainfall < 10 else "Hoãn bón phân"
                ]
            }
        
        elif "khoai" in crop_lower or "potato" in crop_lower:
            return {
                "recommendations": [
                    "Kiểm tra độ ẩm đất",
                    "Đắp luống cao tránh úng",
                    "Theo dõi bệnh nấm"
                ],
                "activities": [
                    "Trồng khoai tây" if 15 <= weather.temperature <= 25 else "Chờ nhiệt độ phù hợp",
                    "Thu hoạch khoai" if weather.rainfall < 5 else "Hoãn thu hoạch"
                ]
            }
        
        else:  # Default general advice
            return {
                "recommendations": [
                    "Theo dõi thời tiết định kỳ",
                    "Điều chỉnh tưới nước phù hợp",
                    "Phòng trừ sâu bệnh"
                ],
                "activities": [
                    "Chăm sóc cây trồng thường xuyên"
                ]
            }
    
    def _get_uv_description(self, uv_index: float) -> str:
        """Get UV index description."""
        if uv_index <= 2:
            return "Thấp"
        elif uv_index <= 5:
            return "Trung bình"
        elif uv_index <= 7:
            return "Cao"
        elif uv_index <= 10:
            return "Rất cao"
        else:
            return "Cực cao"
    
    def _get_demo_weather_data(self, location: str) -> WeatherCondition:
        """Get demo weather data when API is not available."""
        self.logger.warning(f"Using demo weather data for {location}")
        
        return WeatherCondition(
            temperature=28.0,
            feels_like=32.0,
            humidity=78,
            rainfall=0.0,
            wind_speed=12.5,
            wind_direction=225,  # SW
            wind_direction_text="SW",
            pressure=1013.0,
            visibility=8.5,
            uv_index=6.0,
            clouds=45,
            dew_point=24.2,
            description="Partly cloudy",
            timestamp=datetime.now(),
            sunrise=datetime.now().replace(hour=6, minute=12, second=0, microsecond=0),
            sunset=datetime.now().replace(hour=18, minute=8, second=0, microsecond=0),
            rain_probability=15,
            location_name=f"{location}, VN"
        )
    
    def format_detailed_weather_response(
        self, 
        weather: WeatherCondition, 
        agriculture_advice: AgricultureAdvice
    ) -> str:
        """Format comprehensive weather and agriculture advice response like weather apps."""
        
        from datetime import datetime
        
        # Header with location and time
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
☀️ **Chỉ số UV**: {weather.uv_index} ({self._get_uv_description(weather.uv_index)})
☁️ **Mây**: {weather.clouds}%
💧 **Điểm sương**: {weather.dew_point:.1f}°C"""

        if weather.sunrise and weather.sunset:
            response += f"""
🌅 **Bình minh**: {weather.sunrise.strftime('%H:%M')}
🌇 **Hoàng hôn**: {weather.sunset.strftime('%H:%M')}"""

        if weather.rain_probability:
            response += f"""
🌧️ **Xác suất mưa**: {weather.rain_probability}%"""
        
        # Agriculture recommendations section
        response += f"""

{"=" * 60}
🌾 **TƯ VẤN NÔNG NGHIỆP CHO {agriculture_advice.crop_type.upper()}**
{"=" * 60}

📊 **Đánh giá**: {agriculture_advice.weather_summary}
🎯 **Độ tin cậy**: {agriculture_advice.confidence:.1%}

💡 **Khuyến nghị hôm nay**:"""
        
        for rec in agriculture_advice.recommendations:
            response += f"\n   🌱 {rec}"
        
        if agriculture_advice.warnings:
            response += f"\n\n⚠️ **Lưu ý quan trọng**:"
            for warning in agriculture_advice.warnings:
                response += f"\n   • {warning}"
        
        if agriculture_advice.optimal_activities:
            response += f"\n\n✅ **Hoạt động phù hợp**:"
            for activity in agriculture_advice.optimal_activities:
                response += f"\n   • {activity}"
        
        return response
    
    async def _analyze_weather_for_agriculture(
        self,
        weather: WeatherCondition,
        crop_type: str,
        location: str,
        forecast: Optional[List[WeatherCondition]] = None
    ) -> AgricultureAdvice:
        """Analyze weather conditions for agriculture advice."""
        
        recommendations = []
        warnings = []
        optimal_activities = []
        avoid_activities = []
        
        # Temperature analysis
        if weather.temperature > 35:
            warnings.append("Nhiệt độ quá cao - Cần che chắn cho cây trồng")
            avoid_activities.append("Tránh làm việc ngoài trời vào giữa ngày")
            recommendations.append("Tưới nước nhiều hơn để làm mát")
        elif weather.temperature < 15:
            warnings.append("Nhiệt độ thấp - Có thể ảnh hưởng đến sinh trưởng")
            recommendations.append("Che chắn cây trồng khỏi gió lạnh")
        else:
            optimal_activities.append("Nhiệt độ phù hợp cho các hoạt động nông nghiệp")
        
        # Humidity analysis
        if weather.humidity > 80:
            warnings.append("Độ ẩm cao - Nguy cơ bệnh nấm")
            recommendations.append("Tăng cường thông gió")
            avoid_activities.append("Tránh tưới nước vào buổi tối")
        elif weather.humidity < 30:
            warnings.append("Độ ẩm thấp - Cây có thể bị khô")
            recommendations.append("Tăng tần suất tưới nước")
        
        # Wind analysis
        if weather.wind_speed > 25:
            warnings.append("Gió mạnh - Có thể gãy cành cây")
            avoid_activities.append("Tránh phun thuốc khi gió mạnh")
        elif weather.wind_speed < 5:
            recommendations.append("Gió nhẹ - Thích hợp cho phun thuốc")
        
        # UV index analysis
        if weather.uv_index > 7:
            warnings.append("Chỉ số UV cao - Bảo vệ công nhân")
            recommendations.append("Làm việc vào sáng sớm hoặc chiều muộn")
        
        # Pressure analysis
        if weather.pressure < 1000:
            warnings.append("Áp suất thấp - Có thể có mưa")
            recommendations.append("Chuẩn bị che chắn cây trồng")
        
        # Crop-specific advice
        crop_advice = self._get_crop_specific_advice(crop_type, weather)
        recommendations.extend(crop_advice["recommendations"])
        optimal_activities.extend(crop_advice["activities"])
        
        # Calculate confidence based on weather conditions
        confidence = 0.8
        if weather.temperature > 35 or weather.temperature < 10:
            confidence -= 0.2
        if weather.humidity > 90 or weather.humidity < 20:
            confidence -= 0.1
        if weather.wind_speed > 30:
            confidence -= 0.1
        
        # Weather summary
        weather_summary = (
            f"Nhiệt độ: {weather.temperature}°C (cảm giác {weather.feels_like}°C), "
            f"Độ ẩm: {weather.humidity}%, "
            f"Gió: {weather.wind_speed:.1f}km/h, "
            f"UV: {weather.uv_index} - {weather.description}"
        )
        
        return AgricultureAdvice(
            crop_type=crop_type,
            location=location,
            weather_summary=weather_summary,
            recommendations=list(set(recommendations)),
            warnings=list(set(warnings)),
            optimal_activities=list(set(optimal_activities)),
            avoid_activities=list(set(avoid_activities)),
            confidence=confidence
        )
    
    async def generate_agriculture_advice(
        self,
        location: str,
        crop_type: str = "cây trồng tổng quát",
        include_forecast: bool = False
    ) -> Optional[AgricultureAdvice]:
        """Generate agriculture advice based on weather conditions."""
        try:
            # Get current weather
            weather = await self.get_current_weather(location)
            if not weather:
                # Use demo data if API fails
                weather = self._get_demo_weather_data(location)
            
            # Get weather forecast if requested
            forecast = None
            if include_forecast:
                forecast = await self.get_weather_forecast(location, days=3)
            
            # Generate advice
            advice = await self._analyze_weather_for_agriculture(
                weather, crop_type, location, forecast
            )
            
            return advice
            
        except Exception as e:
            self.logger.error(f"Failed to generate agriculture advice: {e}")
            return None