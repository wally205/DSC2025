"""Hybrid retriever combining BM25 and dense vector search."""

import re
from typing import List, Dict, Any, Tuple
from pathlib import Path

from langchain.schema import Document
from rank_bm25 import BM25Okapi
import numpy as np

from config import get_logger, LoggerMixin


class HybridRetriever(LoggerMixin):
    """Hybrid retriever using BM25 + Dense vector search."""
    
    def __init__(self, vector_store, bm25_weight: float = 0.5, dense_weight: float = 0.5):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_store: Dense vector store (FAISS)
            bm25_weight: Weight for BM25 scores (default 0.5)
            dense_weight: Weight for dense scores (default 0.5)
        """
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        
        # BM25 components
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        
        # Vietnamese text processing
        self.vietnamese_stopwords = {
            'và', 'của', 'trong', 'trên', 'với', 'từ', 'đến', 'cho', 'về', 'tại',
            'là', 'có', 'được', 'sẽ', 'đã', 'đang', 'bị', 'bằng', 'theo', 'như',
            'khi', 'nếu', 'mà', 'để', 'này', 'đó', 'các', 'những', 'một', 'hai',
            'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười', 'thì', 'cũng'
        }
        
        self.logger.info("Hybrid retriever initialized", 
                        bm25_weight=bm25_weight, dense_weight=dense_weight)
    
    def vietnamese_tokenize(self, text: str) -> List[str]:
        """
        Vietnamese-specific tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Lowercase and clean
        text = text.lower().strip()
        
        # Remove special characters but keep Vietnamese diacritics
        text = re.sub(r'[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', ' ', text)
        
        # Split by whitespace
        tokens = text.split()
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens 
                 if len(token) > 2 and token not in self.vietnamese_stopwords]
        
        return tokens
    
    def build_bm25_index(self, documents: List[Document]) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of documents to index
        """
        self.logger.info("Building BM25 index", doc_count=len(documents))
        
        self.documents = documents
        self.tokenized_docs = []
        
        for doc in documents:
            tokens = self.vietnamese_tokenize(doc.page_content)
            self.tokenized_docs.append(tokens)
        
        # Create BM25 index
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
            self.logger.info("BM25 index built successfully")
        else:
            self.logger.warning("No tokenized documents for BM25 index")
    
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Perform BM25 search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.bm25 is None or not self.documents:
            self.logger.warning("BM25 index not available")
            return []
        
        query_tokens = self.vietnamese_tokenize(query)
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include relevant results
                results.append((self.documents[idx], float(scores[idx])))
        
        self.logger.info("BM25 search completed", 
                        query_tokens=len(query_tokens), results=len(results))
        
        return results
    
    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def hybrid_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining BM25 and dense vector search.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples sorted by hybrid score
        """
        self.logger.info("Performing hybrid search", query=query[:50], k=k)
        
        # Get BM25 results
        bm25_results = self.bm25_search(query, k=k*2)  # Get more for fusion
        
        # Get dense vector results
        try:
            dense_docs = self.vector_store.similarity_search_with_scores(query, k=k*2)
            # Convert FAISS distance to similarity (lower distance = higher similarity)
            dense_results = [(doc, 1.0 / (1.0 + score)) for doc, score in dense_docs]
        except Exception as e:
            self.logger.warning("Dense search failed", error=str(e))
            dense_results = []
        
        # Create document score maps
        bm25_scores = {}
        dense_scores = {}
        
        # Normalize BM25 scores
        if bm25_results:
            bm25_raw_scores = [score for _, score in bm25_results]
            bm25_norm_scores = self.normalize_scores(bm25_raw_scores)
            
            for (doc, _), norm_score in zip(bm25_results, bm25_norm_scores):
                doc_id = f"{doc.metadata.get('filename', 'unknown')}_{hash(doc.page_content[:100])}"
                bm25_scores[doc_id] = norm_score
        
        # Normalize dense scores
        if dense_results:
            dense_raw_scores = [score for _, score in dense_results]
            dense_norm_scores = self.normalize_scores(dense_raw_scores)
            
            for (doc, _), norm_score in zip(dense_results, dense_norm_scores):
                doc_id = f"{doc.metadata.get('filename', 'unknown')}_{hash(doc.page_content[:100])}"
                dense_scores[doc_id] = norm_score
        
        # Combine scores
        all_docs = {}
        hybrid_scores = {}
        
        # Add BM25 documents
        for doc, _ in bm25_results:
            doc_id = f"{doc.metadata.get('filename', 'unknown')}_{hash(doc.page_content[:100])}"
            all_docs[doc_id] = doc
            hybrid_scores[doc_id] = self.bm25_weight * bm25_scores.get(doc_id, 0)
        
        # Add dense documents
        for doc, _ in dense_results:
            doc_id = f"{doc.metadata.get('filename', 'unknown')}_{hash(doc.page_content[:100])}"
            all_docs[doc_id] = doc
            if doc_id in hybrid_scores:
                hybrid_scores[doc_id] += self.dense_weight * dense_scores.get(doc_id, 0)
            else:
                hybrid_scores[doc_id] = self.dense_weight * dense_scores.get(doc_id, 0)
        
        # Sort by hybrid score
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k results
        final_results = []
        for doc_id, score in sorted_results[:k]:
            if score > 0:
                final_results.append((all_docs[doc_id], score))
        
        self.logger.info("Hybrid search completed", 
                        bm25_results=len(bm25_results),
                        dense_results=len(dense_results),
                        final_results=len(final_results))
        
        return final_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "bm25_ready": self.bm25 is not None,
            "document_count": len(self.documents),
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "tokenized_docs": len(self.tokenized_docs)
        }