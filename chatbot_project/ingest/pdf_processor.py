"""PDF document processing utilities."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import get_logger, settings, LoggerMixin


class PDFProcessor(LoggerMixin):
    """Processes PDF documents for vector storage."""
    
    def __init__(self):
        """Initialize PDF processor."""
        # Improved text splitter for Vietnamese content
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            # Vietnamese-specific separators
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence endings with space
                "。",    # Vietnamese period
                "? ",    # Question marks with space
                "! ",    # Exclamation marks with space
                "; ",    # Semicolons with space
                ", ",    # Commas with space
                " ",     # Spaces
                ""       # Character level
            ],
            keep_separator=True,  # Keep separators for better context
            add_start_index=True  # Add start index for tracking
        )
        
    def process_pdf(self, file_path: Path) -> List[Document]:
        """
        Process a single PDF file into document chunks.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of processed documents
        """
        try:
            # Ensure file_path is a Path object
            if isinstance(file_path, str):
                file_path = Path(file_path)
                
            self.logger.info("Processing PDF file", file_path=str(file_path))
            
            # Load PDF
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            # Preprocess text for Vietnamese content
            for page in pages:
                page.page_content = self._preprocess_vietnamese_text(page.page_content)
            
            # Split into chunks
            documents = self.text_splitter.split_documents(pages)
            
            # Filter out empty or very short chunks
            documents = [doc for doc in documents if len(doc.page_content.strip()) > 50]
            
            # Add metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "source": str(file_path),
                    "filename": file_path.name,
                    "chunk_id": i,
                    "total_chunks": len(documents),
                    "file_hash": self._get_file_hash(file_path),
                    "processed_timestamp": self._get_timestamp(),
                    "chunk_length": len(doc.page_content)
                })
            
            self.logger.info(
                "PDF processing completed",
                file_path=str(file_path),
                chunks_created=len(documents)
            )
            
            return documents
            
        except Exception as e:
            self.logger.error(
                "Error processing PDF",
                file_path=str(file_path),
                error=str(e)
            )
            raise
    
    def process_directory(self, directory_path: Path) -> List[Document]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all processed documents
        """
        all_documents = []
        pdf_files = list(directory_path.glob("*.pdf"))
        
        self.logger.info(
            "Processing PDF directory",
            directory=str(directory_path),
            pdf_count=len(pdf_files)
        )
        
        for pdf_file in pdf_files:
            try:
                documents = self.process_pdf(pdf_file)
                all_documents.extend(documents)
            except Exception as e:
                self.logger.warning(
                    "Failed to process PDF file",
                    file_path=str(pdf_file),
                    error=str(e)
                )
                continue
        
        self.logger.info(
            "Directory processing completed",
            total_documents=len(all_documents),
            processed_files=len([f for f in pdf_files])
        )
        
        return all_documents
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get SHA-256 hash of file content."""
        # Ensure file_path is a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _preprocess_vietnamese_text(self, text: str) -> str:
        """
        Preprocess Vietnamese text for better chunking.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned and preprocessed text
        """
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and common PDF artifacts
        text = re.sub(r'\n\d+\n', '\n', text)  # Remove standalone page numbers
        text = re.sub(r'\f', '\n\n', text)     # Replace form feeds with paragraph breaks
        
        # Fix common Vietnamese text issues from PDF extraction
        text = re.sub(r'([a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ])([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])', 
                  r'\1. \2', text)  # Add periods between sentences that got joined
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'([.!?])([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])', r'\1 \2', text)
        
        # Remove extra spaces
        text = re.sub(r' +', ' ', text)
        
        # Clean up line breaks
        text = re.sub(r'\n +', '\n', text)
        text = re.sub(r' +\n', '\n', text)
        
        return text.strip()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def validate_pdf_file(self, file_path: Path) -> bool:
        """
        Validate if a file is a valid PDF.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if valid PDF, False otherwise
        """
        if not file_path.exists():
            return False
            
        if file_path.suffix.lower() != '.pdf':
            return False
            
        # Check file size (max 10MB by default)
        max_size = getattr(settings, 'max_file_size_mb', 10) * 1024 * 1024
        if file_path.stat().st_size > max_size:
            self.logger.warning(
                "PDF file too large",
                file_path=str(file_path),
                size_mb=file_path.stat().st_size / (1024 * 1024)
            )
            return False
            
        return True