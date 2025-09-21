"""Vector store management using FAISS."""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import get_logger, settings, LoggerMixin


class VectorStore(LoggerMixin):
    """Manages vector storage operations with FAISS."""
    
    def __init__(self):
        """Initialize vector store."""
        self.use_vietnamese_model = True
        
        # Use Vietnamese embedding model for better Vietnamese text understanding
        print("🇻🇳 Using Vietnamese embedding model for better Vietnamese text understanding")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="dangvantuan/vietnamese-embedding",
                model_kwargs={'device': 'cpu', 'trust_remote_code': False},
                encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
            )
            print("✅ Using Vietnamese model: dangvantuan/vietnamese-embedding")
        except Exception as e:
            print(f"❌ Vietnamese model failed ({e}), falling back to multilingual model")
            try:
                # Fallback to multilingual model that supports Vietnamese
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={'device': 'cpu', 'trust_remote_code': False},
                    encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
                )
                print("✅ Using multilingual model: paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as e2:
                print(f"❌ Multilingual model failed ({e2}), using English model as last resort")
                # Last resort: English model
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    encode_kwargs={'normalize_embeddings': True}
                )
                print("✅ Using English model as fallback: all-MiniLM-L6-v2")
        
        self.db_path = Path(settings.chroma_db_path)
        self.index_file = self.db_path / "faiss_index.pkl"
        self.metadata_file = self.db_path / "metadata.pkl"
        self._vector_store: Optional[FAISS] = None
        
        # Set text length limits based on model type
        if self.use_vietnamese_model:
            self.max_text_length = 400  # Conservative for Vietnamese models
        else:
            self.max_text_length = 800  # Higher for English models
        
        # Ensure database directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Auto-load existing index if available
        if self.index_file.exists():
            try:
                self._vector_store = self._create_or_load_vector_store()
                self.logger.info("Auto-loaded existing FAISS index during initialization")
            except Exception as e:
                self.logger.warning(f"Failed to auto-load index during init: {e}")
                self._vector_store = None
        
    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit Vietnamese model token limits."""
        if len(text) <= self.max_text_length:
            return text
        # Truncate và thêm ellipsis
        return text[:self.max_text_length-3] + "..."
        
    @property
    def vector_store(self) -> FAISS:
        """Get or create vector store instance."""
        if self._vector_store is None:
            self._vector_store = self._create_or_load_vector_store()
        return self._vector_store
    
    def _create_or_load_vector_store(self) -> FAISS:
        """Create or load FAISS vector store."""
        try:
            # Try to load existing index
            if self.index_file.exists():
                self.logger.info("Loading existing FAISS index")
                self.logger.info(f"Index file path: {self.index_file}")
                self.logger.info(f"DB path: {self.db_path}")
                
                vector_store = FAISS.load_local(
                    str(self.db_path), 
                    self.embeddings,
                    index_name="faiss_index",
                    allow_dangerous_deserialization=True  # We trust our own files
                )
                self.logger.info("FAISS index loaded successfully")
                return vector_store
            else:
                self.logger.info("No existing FAISS index found")
                self.logger.info(f"Looking for: {self.index_file}")
                # Return None - will be created when first documents are added
                return None
                
        except Exception as e:
            self.logger.error("Failed to create/load vector store", error=str(e))
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        try:
            self.logger.info("Adding documents to FAISS vector store", count=len(documents))
            
            # Filter out empty documents and truncate text
            valid_docs = []
            for doc in documents:
                if doc.page_content.strip():
                    # Truncate text to fit model limits
                    truncated_content = self._truncate_text(doc.page_content)
                    truncated_doc = Document(
                        page_content=truncated_content,
                        metadata=doc.metadata
                    )
                    valid_docs.append(truncated_doc)
                    
            self.logger.info(f"Filtered to {len(valid_docs)} valid documents (truncated to {self.max_text_length} chars)")
            
            if not valid_docs:
                self.logger.warning("No valid documents to add")
                return []
            
            # Process in smaller batches to avoid memory issues
            batch_size = 50
            all_ids = []
            
            for i in range(0, len(valid_docs), batch_size):
                batch = valid_docs[i:i + batch_size]
                batch_num = i//batch_size + 1
                total_batches = (len(valid_docs) + batch_size - 1)//batch_size
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} docs)")
                
                try:
                    if self._vector_store is None:
                        # Create new FAISS index with first batch
                        self._vector_store = FAISS.from_documents(batch, self.embeddings)
                        self.logger.info(f"Created new FAISS index with batch {batch_num}")
                    else:
                        # Add to existing index
                        self._vector_store.add_documents(batch)
                        self.logger.info(f"Added batch {batch_num} to existing index")
                    
                    # Generate IDs for this batch - fix indexing
                    start_idx = len(all_ids)
                    batch_ids = [f"doc_{start_idx + j}" for j in range(len(batch))]
                    all_ids.extend(batch_ids)
                    
                    # Save after each batch
                    self._save_index()
                    self.logger.info(f"Batch {batch_num} saved successfully")
                    
                except Exception as batch_error:
                    self.logger.error(f"Error in batch {batch_num}: {batch_error}")
                    # If Vietnamese model fails with token length, try fallback
                    if "index out of range" in str(batch_error) and self.use_vietnamese_model:
                        self.logger.warning("Vietnamese model token limit exceeded, switching to multilingual model")
                        self.use_vietnamese_model = False
                        try:
                            self.embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                                model_kwargs={'device': 'cpu'},
                                encode_kwargs={'normalize_embeddings': True}
                            )
                            print("✅ Switched to multilingual model for better compatibility")
                        except Exception:
                            self.embeddings = HuggingFaceEmbeddings(
                                model_name="sentence-transformers/all-MiniLM-L6-v2",
                                model_kwargs={'device': 'cpu'},
                                encode_kwargs={'normalize_embeddings': True}
                            )
                            print("✅ Switched to English model as last resort")
                        # Reset vector store and try again
                        self._vector_store = None
                        continue
                    else:
                        continue
            
            self.logger.info(
                "Documents added successfully",
                document_count=len(valid_docs),
                ids_count=len(all_ids)
            )
            
            return all_ids
            
        except Exception as e:
            self.logger.error("Failed to add documents", error=str(e))
            raise
    
    def _save_index(self):
        """Save FAISS index to disk."""
        if self._vector_store is not None:
            self._vector_store.save_local(str(self.db_path), index_name="faiss_index")
            
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (not used in FAISS)
            
        Returns:
            List of similar documents
        """
        try:
            self.logger.info(
                "Performing similarity search",
                query=query[:100],
                k=k
            )
            
            if self._vector_store is None:
                self.logger.warning("No vector store available for search")
                return []
            
            results = self._vector_store.similarity_search(query, k=k)
            
            self.logger.info(
                "Similarity search completed",
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_msg = traceback.format_exc()
            self.logger.error("Similarity search failed", error=error_msg, traceback=traceback_msg)
            print(f"❌ Search error: {error_msg}")
            print(f"📜 Traceback: {traceback_msg}")
            return []
    
    def similarity_search_with_scores(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if self._vector_store is None:
                self.logger.warning("No vector store available for search")
                return []
                
            results = self._vector_store.similarity_search_with_score(query, k=k)
            
            self.logger.info(
                "Similarity search with scores completed",
                query=query[:100],
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            traceback_msg = traceback.format_exc()
            self.logger.error("Similarity search with scores failed", error=error_msg, traceback=traceback_msg)
            print(f"❌ Search with scores error: {error_msg}")
            print(f"📜 Traceback: {traceback_msg}")
            return []
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by IDs.
        Note: FAISS doesn't support deletion, so this will log a warning.
        
        Args:
            ids: List of document IDs to delete
        """
        self.logger.warning("FAISS doesn't support document deletion", ids_count=len(ids))
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        try:
            self.logger.warning("Clearing FAISS vector store")
            
            # Remove index files
            if self.index_file.exists():
                self.index_file.unlink()
            if self.metadata_file.exists():
                self.metadata_file.unlink()
                
            # Reset vector store
            self._vector_store = None
            
            self.logger.info("Vector store cleared successfully")
            
        except Exception as e:
            self.logger.error("Failed to clear vector store", error=str(e))
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        try:
            if self._vector_store is None:
                document_count = 0
            else:
                # FAISS doesn't have a direct count method, estimate from index
                document_count = self._vector_store.index.ntotal if hasattr(self._vector_store, 'index') else 0
            
            stats = {
                "collection_name": "faiss_index",
                "document_count": document_count,
                "db_path": str(self.db_path)
            }
            
            self.logger.info("Retrieved collection stats", stats=stats)
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get collection stats", error=str(e))
            return {
                "collection_name": "faiss_index",
                "document_count": 0,
                "db_path": str(self.db_path),
                "error": str(e)
            }