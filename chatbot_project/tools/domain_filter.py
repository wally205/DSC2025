"""Domain-specific document filtering for agricultural queries."""

import re
from typing import List, Dict, Set, Tuple, Optional
from langchain.schema import Document

from config import get_logger, LoggerMixin


class AgricultureDomainFilter(LoggerMixin):
    """Filter documents based on agricultural domain entities."""
    
    def __init__(self):
        """Initialize domain filter with agricultural entities."""
        
        # Define crop-specific keywords and their variations
        self.crop_entities = {
            'cà_phê': {
                'keywords': [
                    'cà phê', 'ca phe', 'coffee', 'arabica', 'robusta',
                    'café', 'cây cà phê', 'cà-phê', 'ca-phe'
                ],
                'related_terms': [
                    'cherry', 'nhân', 'vỏ quả', 'xử lý ướt', 'xử lý khô',
                    'rang', 'pha chế', 'espresso', 'phin'
                ]
            },
            'lúa': {
                'keywords': [
                    'lúa', 'lua', 'gạo', 'rice', 'ruộng lúa', 'cây lúa',
                    'lúa nước', 'lúa tẻ', 'lúa nàng hương'
                ],
                'related_terms': [
                    'thóc', 'cối xay', 'máy gặt', 'cấy', 'gieo sạ',
                    'ruộng', 'đồng ruộng', 'mùa vụ', 'mùa chiêm', 'mùa mùa'
                ]
            },
            'hồ_tiêu': {
                'keywords': [
                    'hồ tiêu', 'ho tieu', 'pepper', 'tiêu', 'tieu',
                    'cây hồ tiêu', 'hạt tiêu', 'tiêu đen', 'tiêu trắng'
                ],
                'related_terms': [
                    'giàn leo', 'trụ đỡ', 'dây leo', 'hạt', 'quả',
                    'thu hoạch tiêu', 'sấy khô', 'phơi'
                ]
            },
            'ngô': {
                'keywords': [
                    'ngô', 'ngo', 'corn', 'maize', 'bắp', 'cây ngô',
                    'ngô sinh khối', 'ngô lai', 'ngô đường'
                ],
                'related_terms': [
                    'bắp non', 'hạt ngô', 'lõi ngô', 'thân ngô', 'lá ngô',
                    'tua ngô', 'râu ngô', 'silage', 'thức ăn chăn nuôi'
                ]
            },
            'khoai_tây': {
                'keywords': [
                    'khoai tây', 'khoai tay', 'potato', 'khoai', 'tây',
                    'cây khoai tây', 'củ khoai tây', 'khoai-tây'
                ],
                'related_terms': [
                    'củ', 'giống khoai', 'trồng khoai', 'thu hoạch khoai',
                    'bảo quản khoai', 'kho chứa', 'mầm', 'nảy mầm'
                ]
            }
        }
        
        # Disease and pest terms
        self.disease_pest_terms = {
            'diseases': [
                'bệnh', 'benh', 'disease', 'nhiễm bệnh', 'dịch bệnh',
                'đạo ôn', 'phỏng lá', 'khô lá', 'héo', 'thối',
                'nấm', 'vi khuẩn', 'virus', 'bệnh lý'
            ],
            'pests': [
                'sâu', 'sau', 'pest', 'côn trùng', 'sâu bệnh', 'hại',
                'rầy', 'ray', 'nhện', 'bọ', 'ruồi', 'muỗi',
                'ăn lá', 'cắn phá', 'hút nhựa'
            ]
        }
        
        # Agricultural techniques
        self.technique_terms = [
            'kỹ thuật', 'ky thuat', 'technique', 'phương pháp', 'cách',
            'trồng', 'trong', 'canh tác', 'chăm sóc', 'cham soc',
            'bón phân', 'bon phan', 'tưới', 'tuoi', 'cắt tỉa',
            'thu hoạch', 'thu hoach', 'harvest', 'bảo quản', 'bao quan'
        ]
        
        self.logger.info("Agriculture domain filter initialized")
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove diacritics for better matching (simplified)
        replacements = {
            'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
            'ă': 'a', 'ắ': 'a', 'ằ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
            'â': 'a', 'ấ': 'a', 'ầ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
            'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
            'ê': 'e', 'ế': 'e', 'ề': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
            'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
            'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
            'ô': 'o', 'ố': 'o', 'ồ': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
            'ơ': 'o', 'ớ': 'o', 'ờ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
            'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
            'ư': 'u', 'ứ': 'u', 'ừ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
            'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
            'đ': 'd'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def extract_crop_entities(self, query: str) -> List[str]:
        """
        Extract crop entities from query.
        
        Args:
            query: User query
            
        Returns:
            List of detected crop entities
        """
        query_normalized = self.normalize_text(query)
        detected_crops = []
        
        for crop, data in self.crop_entities.items():
            # Check main keywords
            for keyword in data['keywords']:
                keyword_normalized = self.normalize_text(keyword)
                if keyword_normalized in query_normalized:
                    detected_crops.append(crop)
                    break
        
        self.logger.info("Extracted crop entities", 
                        query=query[:50], detected_crops=detected_crops)
        
        return detected_crops
    
    def calculate_document_relevance(self, document: Document, query_crops: List[str]) -> float:
        """
        Calculate relevance score for document based on detected crops.
        
        Args:
            document: Document to score
            query_crops: List of crops detected in query
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not query_crops:
            return 1.0  # No filtering needed
        
        content_normalized = self.normalize_text(document.page_content)
        filename_normalized = self.normalize_text(document.metadata.get('filename', ''))
        
        total_score = 0.0
        max_possible_score = len(query_crops)
        
        for crop in query_crops:
            crop_score = 0.0
            crop_data = self.crop_entities.get(crop, {})
            
            # Score based on main keywords (higher weight)
            main_keywords = crop_data.get('keywords', [])
            for keyword in main_keywords:
                keyword_normalized = self.normalize_text(keyword)
                
                # Check in content
                content_matches = content_normalized.count(keyword_normalized)
                if content_matches > 0:
                    crop_score += 0.6 * min(content_matches / 5.0, 1.0)  # Cap at 1.0
                
                # Check in filename (very high weight)
                if keyword_normalized in filename_normalized:
                    crop_score += 0.8
            
            # Score based on related terms (lower weight)
            related_terms = crop_data.get('related_terms', [])
            for term in related_terms:
                term_normalized = self.normalize_text(term)
                if term_normalized in content_normalized:
                    crop_score += 0.2
            
            # Normalize crop score to max 1.0
            crop_score = min(crop_score, 1.0)
            total_score += crop_score
        
        # Calculate final relevance score
        relevance = total_score / max_possible_score if max_possible_score > 0 else 1.0
        
        return relevance
    
    def filter_documents_by_domain(
        self, 
        documents: List[Document], 
        query: str,
        min_relevance: float = 0.1
    ) -> List[Tuple[Document, float]]:
        """
        Filter documents based on domain relevance.
        
        Args:
            documents: List of documents to filter
            query: Original user query
            min_relevance: Minimum relevance threshold
            
        Returns:
            List of (document, relevance_score) tuples
        """
        # Extract crop entities from query
        query_crops = self.extract_crop_entities(query)
        
        if not query_crops:
            # No specific crops detected, return all documents with score 1.0
            return [(doc, 1.0) for doc in documents]
        
        # Calculate relevance for each document
        scored_documents = []
        for doc in documents:
            relevance = self.calculate_document_relevance(doc, query_crops)
            if relevance >= min_relevance:
                scored_documents.append((doc, relevance))
        
        # Sort by relevance score (descending)
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info("Documents filtered by domain",
                        input_count=len(documents),
                        output_count=len(scored_documents),
                        query_crops=query_crops,
                        min_relevance=min_relevance)
        
        return scored_documents
    
    def get_crop_specific_context(self, query: str) -> Dict[str, List[str]]:
        """
        Get crop-specific context keywords for enhanced search.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with crop-specific keywords
        """
        detected_crops = self.extract_crop_entities(query)
        context = {}
        
        for crop in detected_crops:
            crop_data = self.crop_entities.get(crop, {})
            all_terms = crop_data.get('keywords', []) + crop_data.get('related_terms', [])
            context[crop] = all_terms
        
        return context