# 🤖 Chatbot Nông Nghiệp Thông Minh

### 1. Cài đặt dependencies
pip install -r requirements.txt

### 2. Cấu hình API Key
Tạo file `.env`:
```env
GOOGLE_API_KEY=your_google_api_key_here
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

**Lấy API Keys:**
- **Google Gemini**: [AI Studio](https://aistudio.google.com/) (miễn phí)
- **OpenWeather**: [OpenWeatherMap](https://openweathermap.org/api) 

### 3. Tải dữ liệu vào vector database
python ingest_data.py

### 4. Chạy chatbot
cd chatbot_project
streamlit run ui/app.py

## 🎯 Tính năng

- **🌤️ Thông tin thời tiết thực tế** - OpenWeather API với dữ liệu chi tiết
- **🌾 Tư vấn nông nghiệp** - Cà phê, lúa, tiêu, v.v. dựa trên nguồn dữ liệu chính xác, được chọn lọc

## 💬 Cách sử dụng

### Hỏi thời tiết:
```
"thời tiết hôm nay ở Đà Lạt như thế nào?"
```

### Tư vấn nông nghiệp:
```
"với thời tiết này thì nên làm gì với cây cà phê?"
"cây tiêu cần chăm sóc như thế nào trong mùa mưa?"
```

### Câu hỏi liên tiếp:
```
Câu 1: "thời tiết ở Buôn Ma Thuột ra sao?"
Câu 2: "với điều kiện này thì trồng lúa có được không?"
```

```
chatbot_project/
├── agents/           # LangGraph agents (intent + action)
├── tools/           # Weather + agriculture advisor
├── graph/           # State manager + workflow
├── ingest/          # PDF processing + vector store
├── data/            # PDF documents (file PDF)
├── vectordb/        # FAISS database (tự tạo)
```

## 🔧 Nâng cao

### Thêm tài liệu mới:
1. Copy PDF vào `data/`
2. Chạy `python ingest_documents.py`

### Thay đổi model:
Sửa trong `config/settings.py`:
```python
LLM_MODEL = "gemini-1.5-flash"  # hoặc "gemini-pro"
```

### Tùy chỉnh weather format:
Sửa `tools/agriculture_weather_advisor.py`
