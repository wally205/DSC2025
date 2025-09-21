#!/usr/bin/env python3
"""Script to ingest PDF data into FAISS vector store with Vietnamese embedding."""

from pathlib import Path
from ingest.data_ingester import DataIngester

def main():
    """Main function to ingest data."""
    print("🚀 Starting data ingestion with Vietnamese embedding...")
    
    # Initialize ingester
    ingester = DataIngester()
    
    # Clear existing and ingest new data
    total_docs = ingester.ingest_from_directory(clear_existing=True)
    
    print(f"✅ Ingestion completed! Processed {total_docs} documents")
    
    # Show stats
    stats = ingester.get_ingestion_status()
    print(f"📊 Vector store stats: {stats}")

if __name__ == "__main__":
    main()