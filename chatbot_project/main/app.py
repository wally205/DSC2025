"""Main application entry point."""

import argparse
import sys
from pathlib import Path

from config import configure_logging, get_logger, ensure_directories
from graph import ChatbotGraphBuilder


def main():
    """Main application function."""
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    # Ensure directories exist
    ensure_directories()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LangGraph Chatbot Application")
    parser.add_argument(
        "--mode",
        choices=["chat", "api", "ui"],
        default="chat",
        help="Application mode: chat (CLI), api (FastAPI server), ui (Streamlit)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (chat mode only)"
    )
    
    args = parser.parse_args()
    import asyncio
    
    logger.info("Starting LangGraph Chatbot", mode=args.mode)
    
    if args.mode == "chat":
        asyncio.run(run_chat_mode(args.query))
    elif args.mode == "api":
        run_api_mode()
    elif args.mode == "ui":
        run_ui_mode()


async def run_chat_mode(single_query: str = None):
    """Run in interactive chat mode."""
    logger = get_logger(__name__)
    logger.info("Starting chat mode")
    
    # Initialize chatbot
    chatbot = ChatbotGraphBuilder()
    
    print("🤖 LangGraph Chatbot - Chat Mode")
    print("Nhập 'quit', 'exit', hoặc 'q' để thoát")
    print("-" * 50)
    
    if single_query:
        # Process single query
        await process_single_query(chatbot, single_query)
        return
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("\n👤 Bạn: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Tạm biệt!")
                break
            
            if not user_input:
                continue
            
            await process_single_query(chatbot, user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            logger.error("Chat mode error", error=str(e))
            print(f"❌ Lỗi: {str(e)}")


async def process_single_query(chatbot: ChatbotGraphBuilder, query: str):
    """Process a single query and display results."""
    logger = get_logger(__name__)
    
    try:
        print(f"\n🤔 Đang xử lý: {query}")
        print("-" * 50)
        
        # Process query with conversation history
        result = await chatbot.process_query_with_history(query)
        
        # Display results
        print(f"🤖 Bot: {result.get('response', 'Không có phản hồi')}")
        
        # Display metadata
        print(f"\n📊 Chi tiết:")
        print(f"   Ý định: {result.get('intent', 'N/A')}")
        print(f"   Độ tin cậy: {result.get('confidence', 0):.2f}")
        print(f"   Loại phản hồi: {result.get('response_type', 'N/A')}")
        
        sources = result.get('sources', [])
        if sources:
            print(f"   Nguồn tài liệu: {len(sources)} file")
            for i, source in enumerate(sources[:3], 1):  # Show max 3 sources
                print(f"     {i}. {source.get('filename', 'Unknown')}")
        
    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        print(f"❌ Lỗi khi xử lý: {str(e)}")


def run_api_mode():
    """Run FastAPI server."""
    logger = get_logger(__name__)
    logger.info("Starting API mode")
    
    try:
        from api import run_server
        print("🚀 Starting FastAPI server...")
        print("📚 API Documentation: http://localhost:8000/docs")
        run_server()
    except ImportError as e:
        logger.error("Failed to import API module", error=str(e))
        print("❌ Lỗi: Không thể khởi động API server")
        sys.exit(1)


def run_ui_mode():
    """Run Streamlit UI."""
    logger = get_logger(__name__)
    logger.info("Starting UI mode")
    
    try:
        import subprocess
        import os
        
        # Get the streamlit app path
        ui_path = Path(__file__).parent.parent / "ui" / "streamlit_app.py"
        
        print("🎨 Starting Streamlit UI...")
        print("🌐 UI available at: http://localhost:8501")
        
        # Run streamlit
        subprocess.run([
            "streamlit", "run", str(ui_path),
            "--server.address", "localhost",
            "--server.port", "8501"
        ])
        
    except Exception as e:
        logger.error("Failed to start Streamlit UI", error=str(e))
        print(f"❌ Lỗi: Không thể khởi động UI - {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()