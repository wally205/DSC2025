"""Data ingestion orchestrator."""

from pathlib import Path
from typing import List, Optional

from langchain.schema import Document

from config import get_logger, settings, LoggerMixin
from .pdf_processor import PDFProcessor
from .vector_store import VectorStore


class DataIngester(LoggerMixin):
    """Orchestrates data ingestion from PDFs to vector store."""
    
    def __init__(self):
        """Initialize data ingester."""
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        
    def ingest_from_directory(
        self,
        directory_path: Optional[Path] = None,
        clear_existing: bool = False
    ) -> int:
        """
        Ingest all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            clear_existing: Whether to clear existing data first
            
        Returns:
            Number of documents processed
        """
        if directory_path is None:
            directory_path = Path(settings.pdf_data_path)
            
        self.logger.info(
            "Starting data ingestion",
            directory=str(directory_path),
            clear_existing=clear_existing
        )
        
        # Validate directory
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # Clear existing data if requested
        if clear_existing:
            self.logger.info("Clearing existing vector store data")
            self.vector_store.clear_collection()
        
        # Process PDFs
        documents = self.pdf_processor.process_directory(directory_path)
        
        if not documents:
            self.logger.warning("No documents processed from directory")
            return 0
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        self.logger.info(
            "Data ingestion completed successfully",
            total_documents=len(documents)
        )
        
        return len(documents)
    
    def ingest_single_file(self, file_path: Path) -> int:
        """
        Ingest a single PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Number of documents processed
        """
        self.logger.info("Ingesting single file", file_path=str(file_path))
        
        # Validate file 
        if not self.pdf_processor.validate_pdf_file(file_path):
            raise ValueError(f"Invalid PDF file: {file_path}")
        
        # Process PDF
        documents = self.pdf_processor.process_pdf(file_path)
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        self.logger.info(
            "Single file ingestion completed",
            file_path=str(file_path),
            documents_created=len(documents)
        )
        
        return len(documents)
    
    def get_ingestion_status(self) -> dict:
        """
        Get current ingestion status.
        
        Returns:
            Dictionary with ingestion statistics
        """
        return self.vector_store.get_collection_stats()