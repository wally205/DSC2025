"""Document retrieval tools with hybrid search and domain filtering."""

from typing import Dict, List, Optional, Tuple

from langchain.schema import Document

from config import get_logger, LoggerMixin, DEFAULT_SEARCH_LIMIT, SIMILARITY_THRESHOLD
from ingest import VectorStore
from tools.domain_filter import AgricultureDomainFilter
from tools.hybrid_retriever import HybridRetriever


class DocumentRetriever(LoggerMixin):
    """Handles document retrieval with domain-specific filtering."""
    
    def __init__(self):
        """Initialize document retriever."""
        self.vector_store = VectorStore()
        self.domain_filter = AgricultureDomainFilter()
        self.hybrid_retriever = None
        self.use_hybrid = False
        
        # Try to initialize hybrid retriever
        self._init_hybrid_retriever()
        
    def _init_hybrid_retriever(self):
        """Initialize hybrid retriever with existing documents."""
        try:
            # Get existing documents from vector store for BM25 indexing
            stats = self.vector_store.get_collection_stats()
            
            if stats.get('document_count', 0) > 0:
                self.logger.info("Initializing hybrid retriever")
                self.hybrid_retriever = HybridRetriever(
                    vector_store=self.vector_store,
                    bm25_weight=0.5,
                    dense_weight=0.5
                )
                
                # Get sample documents to build BM25 index
                # Note: This is a simplified approach - in production you'd want to
                # store and reload the BM25 index properly
                sample_docs = self.vector_store.similarity_search("", k=1000)  # Get many docs
                if sample_docs:
                    self.hybrid_retriever.build_bm25_index(sample_docs)
                    self.use_hybrid = True
                    self.logger.info("Hybrid retriever initialized successfully")
                else:
                    self.logger.warning("No documents found for BM25 indexing")
            else:
                self.logger.info("No documents in vector store, using dense search only")
                
        except Exception as e:
            self.logger.warning("Failed to initialize hybrid retriever", error=str(e))
            self.use_hybrid = False
        
    def search_documents(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Search for relevant documents with domain-specific filtering.
        
        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents filtered by domain
        """
        self.logger.info(
            "Searching documents with domain filtering",
            query=query[:100],
            limit=limit,
            score_threshold=score_threshold
        )
        
        try:
            # Use hybrid search if available, otherwise fall back to dense search
            if self.use_hybrid and self.hybrid_retriever:
                self.logger.info("Using hybrid search (BM25 + Dense)")
                # Get hybrid results
                hybrid_results = self.hybrid_retriever.hybrid_search(query, k=limit * 3)
                initial_results = [doc for doc, _ in hybrid_results]
                
                if not initial_results:
                    self.logger.warning("No hybrid results found, falling back to dense search")
                    initial_results = self.vector_store.similarity_search(
                        query=query,
                        k=limit * 3,
                        score_threshold=score_threshold or SIMILARITY_THRESHOLD
                    )
            else:
                self.logger.info("Using dense search only")
                # Get initial results from vector store
                initial_results = self.vector_store.similarity_search(
                    query=query,
                    k=limit * 3,  # Get more results for better filtering
                    score_threshold=score_threshold or SIMILARITY_THRESHOLD
                )
            
            if not initial_results:
                self.logger.warning("No initial results found")
                return []
            
            # Apply domain filtering
            filtered_results = self.domain_filter.filter_documents_by_domain(
                documents=initial_results,
                query=query,
                min_relevance=0.1  # Minimum domain relevance
            )
            
            # Extract documents and sort by combined score
            final_results = [doc for doc, relevance in filtered_results[:limit]]
            
            self.logger.info("Domain-filtered search completed", 
                           initial_count=len(initial_results),
                           filtered_count=len(final_results))
            
            return final_results
            
        except Exception as e:
            self.logger.error("Document search failed", error=str(e))
            # Fall back to basic vector search
            try:
                return self.vector_store.similarity_search(query, k=limit)
            except:
                return []
    
    def search_with_scores(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents with similarity scores using hybrid search when available.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of (document, score) tuples
        """
        self.logger.info(
            "Searching documents with scores",
            query=query[:100],
            limit=limit,
            use_hybrid=self.use_hybrid
        )
        
        try:
            # Use hybrid search if available
            if self.use_hybrid and self.hybrid_retriever:
                self.logger.info("Using hybrid search with scores")
                hybrid_results = self.hybrid_retriever.hybrid_search(query, k=limit * 3)
                
                if hybrid_results:
                    initial_results = hybrid_results
                else:
                    self.logger.warning("No hybrid results, falling back to dense search")
                    initial_results = self.vector_store.similarity_search_with_scores(
                        query, k=limit * 3
                    )
            else:
                self.logger.info("Using dense search with scores")
                # Get initial results with scores
                initial_results = self.vector_store.similarity_search_with_scores(
                    query, k=limit * 3  # Get more for better filtering
                )
            
            if not initial_results:
                return []
            
            # Extract documents for domain filtering
            documents = [doc for doc, score in initial_results]
            
            # Apply domain filtering
            filtered_results = self.domain_filter.filter_documents_by_domain(
                documents=documents,
                query=query,
                min_relevance=0.1
            )
            
            # Combine vector scores with domain relevance
            final_results = []
            for doc, domain_score in filtered_results[:limit]:
                # Find original vector score
                vector_score = 0.0
                for orig_doc, orig_score in initial_results:
                    if orig_doc.page_content == doc.page_content:
                        vector_score = orig_score
                        break
                
                # Combine scores (70% vector, 30% domain)
                combined_score = 0.7 * vector_score + 0.3 * domain_score
                final_results.append((doc, combined_score))
            
            self.logger.info("Domain-filtered search with scores completed", 
                           results_count=len(final_results))
            
            return final_results
            
        except Exception as e:
            self.logger.error("Document search with scores failed", error=str(e))
            # Fall back to vector store search
            try:
                return self.vector_store.similarity_search_with_scores(query, k=limit)
            except:
                return []
    
    def get_relevant_context(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        min_score: float = SIMILARITY_THRESHOLD
    ) -> str:
        """
        Get relevant context as formatted string.
        
        Args:
            query: Search query
            limit: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            Formatted context string
        """
        self.logger.info(
            "Getting relevant context",
            query=query[:100],
            limit=limit,
            min_score=min_score
        )
        
        try:
            results_with_scores = self.search_with_scores(query, limit)
            
            # Filter by minimum score
            filtered_results = [
                (doc, score) for doc, score in results_with_scores
                if score >= min_score
            ]
            
            if not filtered_results:
                self.logger.warning("No relevant context found")
                return ""
            
            # Format context
            context_parts = []
            for i, (doc, score) in enumerate(filtered_results, 1):
                source = doc.metadata.get('filename', 'Unknown')
                content = doc.page_content.strip()
                
                context_parts.append(
                    f"[Nguồn {i}: {source} (Độ tương đồng: {score:.2f})]\n{content}\n"
                )
            
            context = "\n".join(context_parts)
            
            self.logger.info(
                "Relevant context retrieved",
                context_length=len(context),
                sources_count=len(filtered_results)
            )
            
            return context
            
        except Exception as e:
            self.logger.error("Failed to get relevant context", error=str(e))
            return ""
    
    def get_document_sources(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Dict[str, str]]:
        """
        Get list of document sources for a query.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of source information dictionaries
        """
        try:
            results = self.search_documents(query, limit)
            
            sources = []
            seen_sources = set()
            
            for doc in results:
                filename = doc.metadata.get('filename', 'Unknown')
                source_path = doc.metadata.get('source', 'Unknown')
                
                if filename not in seen_sources:
                    sources.append({
                        'filename': filename,
                        'source_path': source_path,
                        'chunk_count': sum(
                            1 for d in results 
                            if d.metadata.get('filename') == filename
                        )
                    })
                    seen_sources.add(filename)
            
            self.logger.info(
                "Retrieved document sources",
                query=query[:100],
                sources_count=len(sources)
            )
            
            return sources
            
        except Exception as e:
            self.logger.error("Failed to get document sources", error=str(e))
            return []
    
    def rebuild_hybrid_index(self):
        """Rebuild BM25 index for hybrid search."""
        # Temporarily disabled due to BM25 package issues
        self.logger.info("Hybrid search temporarily disabled")
        # try:
        #     self.logger.info("Rebuilding hybrid search index")
        #     
        #     if not self.hybrid_retriever:
        #         self.hybrid_retriever = HybridRetriever(
        #             vector_store=self.vector_store,
        #             bm25_weight=0.5,
        #             dense_weight=0.5
        #         )
        #     
        #     # Get all documents for BM25 indexing
        #     # This is a simplified approach - in production, you'd implement proper indexing
        #     sample_docs = self.vector_store.similarity_search("", k=2000)  # Get many docs
        #     
        #     if sample_docs:
        #         self.hybrid_retriever.build_bm25_index(sample_docs)
        #         self.use_hybrid = True
        #         self.logger.info("Hybrid index rebuilt successfully", doc_count=len(sample_docs))
        #     else:
        #         self.logger.warning("No documents found for BM25 indexing")
        #         self.use_hybrid = False
        #         
        # except Exception as e:
        #     self.logger.error("Failed to rebuild hybrid index", error=str(e))
        #     self.use_hybrid = False
    
    def get_search_stats(self) -> Dict[str, any]:
        """Get search statistics."""
        stats = {
            "use_hybrid": self.use_hybrid,
            "vector_store_stats": self.vector_store.get_collection_stats()
        }
        
        if self.hybrid_retriever:
            stats["hybrid_stats"] = self.hybrid_retriever.get_stats()
            
        return stats