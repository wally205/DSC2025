"""Data ingestion runner script."""

import argparse
import sys
from pathlib import Path

from config import configure_logging, get_logger, ensure_directories
from ingest import DataIngester


def main():
    """Main ingestion function."""
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    # Ensure directories exist
    ensure_directories()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data Ingestion for LangGraph Chatbot")
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to directory containing PDF files"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing data before ingestion"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Ingest single PDF file"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show ingestion status"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting data ingestion", args=vars(args))
    
    # Initialize data ingester
    data_ingester = DataIngester()
    
    print("📚 LangGraph Chatbot - Data Ingestion")
    print("-" * 50)
    
    try:
        if args.status:
            show_status(data_ingester)
        elif args.file:
            ingest_single_file(data_ingester, args.file)
        else:
            ingest_directory(data_ingester, args.data_path, args.clear)
            
    except Exception as e:
        logger.error("Ingestion failed", error=str(e))
        print(f"❌ Lỗi: {str(e)}")
        sys.exit(1)


def show_status(data_ingester: DataIngester):
    """Show current ingestion status."""
    logger = get_logger(__name__)
    
    print("📊 Trạng thái ingestion hiện tại:")
    print("-" * 30)
    
    try:
        status = data_ingester.get_ingestion_status()
        
        print(f"Collection: {status.get('collection_name', 'N/A')}")
        print(f"Số tài liệu: {status.get('document_count', 0)}")
        print(f"Đường dẫn DB: {status.get('db_path', 'N/A')}")
        
        if status.get('error'):
            print(f"⚠️  Lỗi: {status['error']}")
        else:
            print("✅ Vector database hoạt động bình thường")
            
    except Exception as e:
        logger.error("Failed to get status", error=str(e))
        print(f"❌ Không thể lấy trạng thái: {str(e)}")


def ingest_single_file(data_ingester: DataIngester, file_path: str):
    """Ingest a single PDF file."""
    logger = get_logger(__name__)
    
    file_path = Path(file_path)
    
    print(f"📄 Ingesting file: {file_path}")
    print("-" * 30)
    
    if not file_path.exists():
        print(f"❌ File không tồn tại: {file_path}")
        sys.exit(1)
    
    if not file_path.name.lower().endswith('.pdf'):
        print(f"❌ File không phải PDF: {file_path}")
        sys.exit(1)
    
    try:
        print("⏳ Đang xử lý file...")
        
        documents_count = data_ingester.ingest_single_file(file_path)
        
        print(f"✅ Ingestion hoàn thành!")
        print(f"📊 Số documents được tạo: {documents_count}")
        
        # Show updated status
        print("\n📊 Trạng thái sau ingestion:")
        show_status(data_ingester)
        
    except Exception as e:
        logger.error("Single file ingestion failed", error=str(e))
        print(f"❌ Lỗi khi ingest file: {str(e)}")
        raise


def ingest_directory(data_ingester: DataIngester, data_path: str, clear_existing: bool):
    """Ingest all PDF files from a directory."""
    logger = get_logger(__name__)
    
    data_path = Path(data_path)
    
    print(f"📁 Ingesting directory: {data_path}")
    print(f"🗑️  Clear existing: {clear_existing}")
    print("-" * 30)
    
    if not data_path.exists():
        print(f"❌ Thư mục không tồn tại: {data_path}")
        sys.exit(1)
    
    if not data_path.is_dir():
        print(f"❌ Đường dẫn không phải thư mục: {data_path}")
        sys.exit(1)
    
    # Check for PDF files
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  Không tìm thấy file PDF trong: {data_path}")
        sys.exit(1)
    
    print(f"📄 Tìm thấy {len(pdf_files)} file PDF")
    
    # Show files to be processed
    print("\nFile sẽ được xử lý:")
    for pdf_file in pdf_files:
        print(f"  - {pdf_file.name}")
    
    # Confirm if clearing existing data
    if clear_existing:
        print("\n⚠️  Sẽ xóa toàn bộ dữ liệu hiện có!")
        confirm = input("Bạn có chắc chắn? (y/N): ").strip().lower()
        if confirm != 'y':
            print("❌ Đã hủy ingestion")
            return
    
    try:
        print("\n⏳ Bắt đầu ingestion...")
        
        documents_count = data_ingester.ingest_from_directory(
            directory_path=data_path,
            clear_existing=clear_existing
        )
        
        print(f"\n✅ Ingestion hoàn thành!")
        print(f"📊 Tổng số documents được tạo: {documents_count}")
        print(f"📁 Từ {len(pdf_files)} file PDF")
        
        # Show updated status
        print("\n📊 Trạng thái sau ingestion:")
        show_status(data_ingester)
        
    except Exception as e:
        logger.error("Directory ingestion failed", error=str(e))
        print(f"❌ Lỗi khi ingest thư mục: {str(e)}")
        raise


if __name__ == "__main__":
    main()