"""Utility functions for the chatbot project."""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any

from config import get_logger, LoggerMixin


class FileUtils(LoggerMixin):
    """File utility functions."""
    
    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def copy_files_to_data_folder(source_files: List[str], data_folder: str = "./data") -> List[str]:
        """
        Copy files to data folder for processing.
        
        Args:
            source_files: List of source file paths
            data_folder: Destination data folder
            
        Returns:
            List of copied file paths
        """
        logger = get_logger(__name__)
        data_path = Path(data_folder)
        data_path.mkdir(parents=True, exist_ok=True)
        
        copied_files = []
        
        for source_file in source_files:
            source_path = Path(source_file)
            
            if not source_path.exists():
                logger.warning("Source file not found", file=source_file)
                continue
            
            if not source_path.name.lower().endswith('.pdf'):
                logger.warning("File is not a PDF", file=source_file)
                continue
            
            # Copy to data folder
            dest_path = data_path / source_path.name
            shutil.copy2(source_path, dest_path)
            copied_files.append(str(dest_path))
            
            logger.info("File copied", source=source_file, destination=str(dest_path))
        
        return copied_files
    
    @staticmethod
    def get_file_info(file_path: Path) -> Dict[str, Any]:
        """Get file information."""
        if not file_path.exists():
            return {"exists": False}
        
        stat = file_path.stat()
        
        return {
            "exists": True,
            "name": file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified": stat.st_mtime,
            "extension": file_path.suffix.lower()
        }


class ValidationUtils(LoggerMixin):
    """Validation utility functions."""
    
    @staticmethod
    def validate_pdf_files(file_paths: List[str]) -> Dict[str, List[str]]:
        """
        Validate PDF files.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary with valid and invalid files
        """
        logger = get_logger(__name__)
        
        valid_files = []
        invalid_files = []
        
        for file_path in file_paths:
            path = Path(file_path)
            
            if not path.exists():
                invalid_files.append(f"{file_path} (file not found)")
                continue
            
            if not path.name.lower().endswith('.pdf'):
                invalid_files.append(f"{file_path} (not a PDF)")
                continue
            
            # Check file size (max 50MB)
            if path.stat().st_size > 50 * 1024 * 1024:
                invalid_files.append(f"{file_path} (file too large)")
                continue
            
            valid_files.append(file_path)
        
        logger.info(
            "File validation completed",
            valid_count=len(valid_files),
            invalid_count=len(invalid_files)
        )
        
        return {
            "valid": valid_files,
            "invalid": invalid_files
        }


class TextUtils:
    """Text processing utility functions."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = " ".join(text.split())
        
        # Remove special characters but keep Vietnamese characters
        import re
        cleaned = re.sub(r'[^\w\s\u00C0-\u024F\u1EA0-\u1EF9]', ' ', cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 100) -> str:
        """Truncate text to specified length."""
        if len(text) <= max_length:
            return text
        
        return text[:max_length-3] + "..."
    
    @staticmethod
    def extract_vietnamese_words(text: str) -> List[str]:
        """Extract Vietnamese words from text."""
        import re
        
        # Pattern for Vietnamese words
        vietnamese_pattern = r'[a-zA-ZÀ-ỹĂăÂâÊêÔôƠơƯưĐđ]+'
        
        words = re.findall(vietnamese_pattern, text.lower())
        
        # Filter out single characters and common stop words
        stop_words = {
            'và', 'của', 'có', 'là', 'trong', 'với', 'cho', 'để', 'được',
            'không', 'này', 'đó', 'về', 'như', 'khi', 'nào', 'sao', 'ai', 'gì'
        }
        
        filtered_words = [
            word for word in words 
            if len(word) > 2 and word not in stop_words
        ]
        
        return list(set(filtered_words))  # Remove duplicates