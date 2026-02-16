from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from neo4j_conn import get_neo4j_session
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# Import the contextual BM25 system
try:
    from contextual_bm25 import ContextualBM25, DocumentContext, BM25Result
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ Contextual BM25 not available. Using standard keyword search.")

# Import the two-stage filtering system
try:
    from two_stage_filtering import TwoStageFilter
    FILTERING_AVAILABLE = True
except ImportError:
    FILTERING_AVAILABLE = False
    print("Two-stage filtering not available. Using standard retrieval.")

# Import the contextual enhancer
try:
    from contextual_enhancer import ContextualEnhancer
    CONTEXTUAL_ENHANCEMENT_AVAILABLE = True
except ImportError:
    CONTEXTUAL_ENHANCEMENT_AVAILABLE = False
    print("Contextual enhancement not available. Using standard chunk processing.")

# Import the new contextual embedder
try:
    from contextual_embeddings import ContextualEmbedder
    CONTEXTUAL_EMBEDDINGS_AVAILABLE = True
except ImportError:
    CONTEXTUAL_EMBEDDINGS_AVAILABLE = False
    print("Contextual embeddings not available. Using standard embeddings.")

@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    content: str
    source: str
    score: float
    result_type: str  # "vector", "graph", "keyword"
    metadata: Dict[str, Any] | None = None

@dataclass
class QueryAnalysis:
    """Analysis of a query for search optimization."""
    intent: str  # "factual", "analytical", "comparative"
    entities: List[str]
    keywords: List[str]
    reasoning_path: List[str] | None = None

class HybridRetriever:
    """Hybrid search system combining vector, graph, and keyword search with advanced filtering."""
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "documents"):
        """Initialize the hybrid retriever."""
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize contextual embedder if available
        if CONTEXTUAL_EMBEDDINGS_AVAILABLE:
            self.contextual_embedder = ContextualEmbedder(
                model_name="all-MiniLM-L6-v2",
                context_window_size=200,
                max_context_length=512
            )
            print("✅ Contextual embedder initialized")
        else:
            self.contextual_embedder = None
            print("⚠️ Contextual embedder not available, using standard embeddings")
        
        # Initialize two-stage filtering system
        try:
            from two_stage_filtering import TwoStageFilter
            self.filter = TwoStageFilter(
                relevance_threshold=0.3,
                quality_threshold=0.5,
                confidence_threshold=0.6,
                max_chunks=10
            )
            print("✅ Two-stage filtering initialized")
        except ImportError as e:
            print(f"⚠️ Two-stage filtering not available: {e}")
            self.filter = None
        except Exception as e:
            print(f"⚠️ Two-stage filtering initialization failed: {e}")
            self.filter = None
        
        # Initialize contextual BM25 system
        if BM25_AVAILABLE:
            self.bm25 = ContextualBM25(k1=1.2, b=0.75)
            print("✅ Contextual BM25 initialized")
        else:
            self.bm25 = None
            print("⚠️ Contextual BM25 not available, using fallback keyword search")
        
        # Initialize contextual enhancer
        if CONTEXTUAL_ENHANCEMENT_AVAILABLE:
            self.contextual_enhancer = ContextualEnhancer(context_window_size=200)
            print("✅ Contextual enhancement initialized")
        else:
            self.contextual_enhancer = None
            print("⚠️ Contextual enhancement not available, using standard chunk processing")
        
        # Initialize collection if it doesn't exist
        self._init_collection()
    
    def _init_collection(self):
        """Initialize Qdrant collection for document embeddings."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_model.get_sentence_embedding_dimension(),
                        distance=Distance.COSINE
                    )
                )
                print(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"Warning: Could not initialize Qdrant collection: {e}")
            print("Qdrant collection will be created when first needed.")
    
    def add_document_chunks(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store and BM25 index with contextual enhancement."""
        if not chunks:
            return
        
        # Ensure collection exists
        try:
            self._init_collection()
        except Exception as e:
            print(f"Could not initialize collection for adding chunks: {e}")
            return
        
        # Use contextual embeddings if available (preferred over basic contextual enhancement)
        if self.contextual_embedder and CONTEXTUAL_EMBEDDINGS_AVAILABLE:
            try:
                # Group chunks by document for contextual embedding
                document_groups = self._group_chunks_by_document(chunks)
                
                all_points = []
                for doc_name, doc_chunks in document_groups.items():
                    # Generate contextual embeddings for this document
                    contextual_embeddings = self.contextual_embedder.embed_document_chunks(
                        doc_chunks, doc_name
                    )
                    
                    print(f"✅ Generated {len(contextual_embeddings)} contextual embeddings for {doc_name}")
                    
                    # Convert to Qdrant points
                    for i, contextual_embedding in enumerate(contextual_embeddings):
                        point = PointStruct(
                            id=len(all_points) + i,
                            vector=contextual_embedding.embedding.tolist(),
                            payload={
                                "text": contextual_embedding.original_text,
                                "enhanced_text": contextual_embedding.enhanced_text,
                                "chunk_id": contextual_embedding.chunk_context.chunk_id,
                                "source_file": doc_name,
                                "metadata": {
                                    **contextual_embedding.chunk_context.metadata,
                                    "document_context": {
                                        "document_type": contextual_embedding.document_context.document_type,
                                        "domain": contextual_embedding.document_context.domain,
                                        "technical_level": contextual_embedding.document_context.technical_level,
                                        "main_topics": contextual_embedding.document_context.main_topics
                                    },
                                    "chunk_context": {
                                        "content_type": contextual_embedding.chunk_context.content_type,
                                        "chunk_position": contextual_embedding.chunk_context.chunk_position,
                                        "importance_score": contextual_embedding.chunk_context.importance_score,
                                        "key_entities": contextual_embedding.chunk_context.key_entities
                                    },
                                    "embedding_metadata": contextual_embedding.embedding_metadata
                                }
                            }
                        )
                        all_points.append(point)
                
                points = all_points
                
            except Exception as e:
                print(f"⚠️ Contextual embeddings failed: {e}")
                print("Falling back to standard contextual enhancement...")
                # Fall back to standard contextual enhancement
                points = self._prepare_contextual_enhanced_points(chunks)
        
        # Fall back to basic contextual enhancement if contextual embeddings not available
        elif self.contextual_enhancer and CONTEXTUAL_ENHANCEMENT_AVAILABLE:
            points = self._prepare_contextual_enhanced_points(chunks)
        else:
            # Standard processing without any contextual enhancement
            points = self._prepare_standard_points(chunks)
        
        # Upload to Qdrant
        try:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Added {len(points)} chunks to vector store")
            
            # Add to BM25 index if available
            if self.bm25 and BM25_AVAILABLE:
                self._add_to_bm25_index(chunks)
                
        except Exception as e:
            print(f"Error adding chunks to vector store: {e}")
    
    def _group_chunks_by_document(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by their source document."""
        document_groups = {}
        
        for chunk in chunks:
            doc_name = chunk.get('source_file', 'unknown')
            if doc_name not in document_groups:
                document_groups[doc_name] = []
            document_groups[doc_name].append(chunk)
            
        return document_groups
    
    def _prepare_contextual_enhanced_points(self, chunks: List[Dict[str, Any]]) -> List[PointStruct]:
        """Prepare points using basic contextual enhancement."""
        try:
            enhanced_chunks = self.contextual_enhancer.enhance_chunks_for_embedding(chunks)
            print(f"✅ Enhanced {len(enhanced_chunks)} chunks with contextual information")
            
            points = []
            for i, enhanced_chunk in enumerate(enhanced_chunks):
                # Generate embedding from enhanced text
                embedding = self.embedding_model.encode(enhanced_chunk.enhanced_text).tolist()
                
                # Create point with enhanced metadata
                point = PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "text": enhanced_chunk.original_text,  # Store original text for display
                        "enhanced_text": enhanced_chunk.enhanced_text,  # Store enhanced text
                        "chunk_id": chunks[i].get("chunk_id", f"chunk_{i}"),
                        "source_file": chunks[i].get("source_file", "unknown"),
                        "metadata": {
                            **chunks[i].get("metadata", {}),
                            "context_type": enhanced_chunk.context_type,
                            "enhancement_metadata": enhanced_chunk.enhancement_metadata
                        }
                    }
                )
                points.append(point)
            
            return points
            
        except Exception as e:
            print(f"⚠️ Contextual enhancement failed: {e}")
            print("Falling back to standard processing...")
            return self._prepare_standard_points(chunks)
    
    def _prepare_standard_points(self, chunks: List[Dict[str, Any]]) -> List[PointStruct]:
        """Prepare standard points for Qdrant without contextual enhancement."""
        points = []
        for i, chunk in enumerate(chunks):
            # Generate embedding from original text
            embedding = self.embedding_model.encode(chunk["text"]).tolist()
            
            # Create point
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "source_file": chunk.get("source_file", "unknown"),
                    "metadata": chunk.get("metadata", {})
                }
            )
            points.append(point)
        return points
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query to determine intent and extract entities/keywords."""
        # Simple intent classification
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "how", "why", "when", "where"]):
            intent = "factual"
        elif any(word in query_lower for word in ["compare", "difference", "versus", "vs"]):
            intent = "comparative"
        elif any(word in query_lower for word in ["analyze", "explain", "describe", "show"]):
            intent = "analytical"
        else:
            intent = "factual"
        
        # Extract potential entities (simple heuristic)
        entities = []
        # Look for capitalized words that might be entities
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
        
        # Extract keywords
        keywords = [word.lower() for word in re.findall(r'\b\w+\b', query.lower()) 
                   if len(word) > 3 and word not in ["what", "how", "why", "when", "where", "the", "and", "for", "with"]]
        
        return QueryAnalysis(
            intent=intent,
            entities=entities,
            keywords=keywords,
            reasoning_path=None
        )
    
    def vector_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform vector similarity search."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_result:
                search_result_obj = SearchResult(
                    content=result.payload["text"],
                    source=result.payload["source_file"],
                    score=result.score,
                    result_type="vector",
                    metadata=result.payload.get("metadata", {})
                )
                results.append(search_result_obj)
            
            return results
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []
    
    def graph_search(self, query: str, entities: List[str], depth: int = 2) -> List[SearchResult]:
        """Perform graph traversal search from extracted entities."""
        try:
            with get_neo4j_session() as session:
                results = []
                
                for entity in entities:
                    # Find entity and expand from it
                    query_cypher = f"""
                    MATCH (e:Entity {{name: $entity_name}})
                    OPTIONAL MATCH path = (e)-[*1..{depth}]-(related)
                    RETURN e, related, path
                    LIMIT 10
                    """
                    
                    result = session.run(query_cypher, entity_name=entity)
                    
                    for record in result:
                        if record["e"]:
                            # Add the entity itself
                            entity_node = record["e"]
                            results.append(SearchResult(
                                content=f"Entity: {entity_node['name']} ({entity_node.get('type', 'UNKNOWN')})",
                                source="graph",
                                score=1.0,
                                result_type="graph",
                                metadata={"entity_type": entity_node.get("type"), "entity_name": entity_node["name"]}
                            ))
                        
                        if record["related"]:
                            # Add related entities
                            related_node = record["related"]
                            results.append(SearchResult(
                                content=f"Related: {related_node['name']} ({related_node.get('type', 'UNKNOWN')})",
                                source="graph",
                                score=0.8,
                                result_type="graph",
                                metadata={"entity_type": related_node.get("type"), "entity_name": related_node["name"]}
                            ))
                
                return results
                
        except Exception as e:
            print(f"Error in graph search: {e}")
            return []
    
    def keyword_search(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Perform keyword-based search using contextual BM25 or fallback method."""
        try:
            # Try contextual BM25 first
            if self.bm25 and BM25_AVAILABLE:
                return self._contextual_bm25_search(query, keywords)
            else:
                # Fallback to original keyword search
                return self._fallback_keyword_search(query, keywords)
                
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []

    def _contextual_bm25_search(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Perform contextual BM25 search."""
        try:
            # Use BM25 to search documents
            if not self.bm25:
                return self._fallback_keyword_search(query, keywords)
            
            bm25_results = self.bm25.search(query, top_k=10, context_boost=1.3)
            
            # Convert BM25 results to SearchResult format
            results = []
            for bm25_result in bm25_results:
                search_result = SearchResult(
                    content=bm25_result.content,
                    source=bm25_result.source,
                    score=bm25_result.score,
                    result_type="bm25",
                    metadata={
                        **bm25_result.metadata,
                        "highlighted_terms": bm25_result.highlighted_terms,
                        "entities": bm25_result.context.entities,
                        "section": bm25_result.context.section
                    }
                )
                results.append(search_result)
            
            return results
            
        except Exception as e:
            print(f"Error in contextual BM25 search: {e}")
            return self._fallback_keyword_search(query, keywords)

    def _fallback_keyword_search(self, query: str, keywords: List[str]) -> List[SearchResult]:
        """Fallback keyword search method."""
        try:
            # Simple keyword matching in vector store payload
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=[0] * self.embedding_model.get_sentence_embedding_dimension(),  # Dummy vector
                limit=50,  # Get more results for keyword filtering
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchValue(value=keyword)
                        ) for keyword in keywords[:3]  # Limit to first 3 keywords
                    ]
                ) if keywords else None
            )
            
            results = []
            for result in search_result:
                # Calculate keyword match score
                text_lower = result.payload["text"].lower()
                keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
                score = keyword_matches / len(keywords) if keywords else 0.5
                
                search_result_obj = SearchResult(
                    content=result.payload["text"],
                    source=result.payload["source_file"],
                    score=score,
                    result_type="keyword",
                    metadata=result.payload.get("metadata", {})
                )
                results.append(search_result_obj)
            
            return results[:10]  # Return top 10
            
        except Exception as e:
            print(f"Error in fallback keyword search: {e}")
            return []

    def _add_to_bm25_index(self, chunks: List[Dict[str, Any]]):
        """Add chunks to the BM25 index with contextual information."""
        try:
            if not self.bm25:
                return
            
            # Extract document texts and create contexts
            documents = []
            contexts = []
            
            for chunk in chunks:
                # Extract basic information
                content = chunk["text"]
                source_file = chunk.get("source_file", "unknown")
                metadata = chunk.get("metadata", {})
                
                # Create document context
                context = DocumentContext(
                    source_file=source_file,
                    section=metadata.get("section_header"),
                    preceding_context=metadata.get("preceding_context"),
                    following_context=metadata.get("following_context"),
                    entities=metadata.get("entities", []),
                    metadata=metadata
                )
                
                documents.append(content)
                contexts.append(context)
            
            # Add to BM25 index
            self.bm25.add_documents(documents, contexts)
            print(f"Added {len(documents)} documents to BM25 index")
            
        except Exception as e:
            print(f"Error adding documents to BM25 index: {e}")
    
    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Main hybrid retrieval method with advanced two-stage filtering."""
        # Step 1: Analyze query
        analysis = self.analyze_query(query)
        
        # Step 2: Perform different types of search with higher initial top_k for filtering
        initial_top_k = max(top_k * 3, 20)  # Get more results for filtering
        vector_results = self.vector_search(query, top_k=initial_top_k)
        graph_results = self.graph_search(query, analysis.entities, depth=2)
        keyword_results = self.keyword_search(query, analysis.keywords)
        
        # Step 3: Combine and rerank results
        all_results = vector_results + graph_results + keyword_results
        
        # Enhanced reranking: prefer vector results, then BM25, then graph, then keyword
        for result in all_results:
            if result.result_type == "vector":
                result.score *= 1.2
            elif result.result_type == "bm25":
                result.score *= 1.15  # BM25 gets higher weight than regular keyword search
            elif result.result_type == "graph":
                result.score *= 1.1
            # keyword results keep original score
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        # Step 4: Apply two-stage filtering if available
        if self.filter is not None and all_results:
            try:
                # Import the RetrievedChunk type
                from two_stage_filtering import RetrievedChunk
                
                # Convert SearchResult objects to RetrievedChunk objects
                retrieved_chunks = []
                for result in all_results:
                    chunk = RetrievedChunk(
                        content=result.content,
                        score=result.score,
                        source=result.source,
                        entity_matches=[],  # Will be populated by entity extraction if needed
                        graph_distance=None,
                        vector_similarity=result.score if result.result_type == "vector" else None,
                        chunk_id=result.metadata.get("chunk_id") if result.metadata else None
                    )
                    retrieved_chunks.append(chunk)
                
                # Apply two-stage filtering
                filtering_result = self.filter.filter_chunks(query, retrieved_chunks)
                
                # Convert filtered chunks back to SearchResult objects
                final_results = []
                for i, chunk in enumerate(filtering_result.filtered_chunks):
                    confidence_score = filtering_result.confidence_scores[i] if i < len(filtering_result.confidence_scores) else 0.5
                    search_result = SearchResult(
                        content=chunk.content,
                        source=chunk.source,
                        score=confidence_score,  # Use confidence score as final score
                        result_type="filtered",
                        metadata={"original_score": chunk.score, "confidence": confidence_score}
                    )
                    final_results.append(search_result)
                
                print(f"🔍 Two-stage filtering: {len(all_results)} → {len(final_results)} results")
                return final_results[:top_k]
                
            except Exception as e:
                print(f"⚠️ Two-stage filtering failed: {e}")
                print("Falling back to standard retrieval...")
                return all_results[:top_k]
        
        # Return unfiltered results if filtering is not available
        return all_results[:top_k]
    
    def multi_hop_reasoning(self, query: str) -> List[SearchResult]:
        """Perform multi-hop reasoning for complex queries."""
        analysis = self.analyze_query(query)
        
        if analysis.intent == "analytical":
            # For analytical queries, use graph traversal with multiple hops
            return self.graph_search(query, analysis.entities, depth=3)
        else:
            # For other queries, use standard hybrid retrieval
            return self.retrieve(query)
    
    def clear_vector_store(self):
        """Clear all data from the vector store."""
        try:
            # Get all point IDs first
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=10000  # Get all points
            )
            
            if search_result[0]:  # If there are points to delete
                point_ids = [point.id for point in search_result[0]]
                
                # Delete the points using the correct selector format
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector={"points": point_ids}
                )
                print(f"Cleared {len(point_ids)} points from vector store collection: {self.collection_name}")
            else:
                print(f"No points to clear in vector store collection: {self.collection_name}")
                
        except Exception as e:
            print(f"Error clearing vector store: {e}")
            # Try alternative method - delete collection and recreate
            try:
                print("Attempting to delete and recreate collection...")
                self.qdrant_client.delete_collection(self.collection_name)
                self._init_collection()
                print(f"Successfully deleted and recreated collection: {self.collection_name}")
            except Exception as e2:
                print(f"Failed to delete/recreate collection: {e2}")
                raise
    
    def remove_document_from_vector_store(self, document_name: str) -> int:
        """Remove all chunks for a specific document from the vector store."""
        try:
            # Find all points for this document
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=document_name)
                        )
                    ]
                ),
                limit=1000  # Adjust based on expected chunk count
            )
            
            if not search_result[0]:  # No points found
                return 0
            
            # Extract point IDs
            point_ids = [point.id for point in search_result[0]]
            
            # Delete the points
            if point_ids:
                self.qdrant_client.delete(
                    collection_name=self.collection_name,
                    points_selector={"points": point_ids}
                )
                print(f"Removed {len(point_ids)} chunks for document: {document_name}")
                return len(point_ids)
            
            return 0
            
        except Exception as e:
            print(f"Error removing document from vector store: {e}")
            return 0
    
    def list_documents_in_vector_store(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store."""
        try:
            # Get all points to extract unique document names
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000
            )
            
            documents = {}
            for point in search_result[0]:
                source_file = point.payload.get("source_file", "unknown")
                if source_file not in documents:
                    documents[source_file] = {
                        "name": source_file,
                        "chunks": 0,
                        "last_updated": None
                    }
                documents[source_file]["chunks"] += 1
            
            return list(documents.values())
            
        except Exception as e:
            print(f"Error listing documents in vector store: {e}")
            return []
    
    def add_documents(self, chunks):
        """Add documents to the vector store."""
        # Convert DocumentChunk objects to dictionaries
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "text": chunk.text,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "metadata": chunk.metadata or {}
            }
            chunk_dicts.append(chunk_dict)
        
        self.add_document_chunks(chunk_dicts)
    
    def clear_all(self):
        """Clear all data from the vector store."""
        self.clear_vector_store()
    
    def remove_document(self, document_name: str) -> bool:
        """Remove a document from the vector store."""
        try:
            removed_count = self.remove_document_from_vector_store(document_name)
            return removed_count > 0
        except Exception:
            return False
    
    def list_documents(self) -> List[str]:
        """List document names in the vector store."""
        try:
            documents = self.list_documents_in_vector_store()
            return [doc["name"] for doc in documents]
        except Exception:
            return []
    
    def get_document_chunks(self, document_name: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document from the vector store."""
        try:
            # Find all points for this document
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=document_name)
                        )
                    ]
                ),
                limit=1000  # Adjust based on expected chunk count
            )
            
            if not search_result[0]:  # No points found
                return []
            
            # Convert points to chunk dictionaries
            chunks = []
            for point in search_result[0]:
                chunk = {
                    "text": point.payload.get("text", ""),
                    "chunk_id": point.payload.get("chunk_id", ""),
                    "source_file": point.payload.get("source_file", ""),
                    "metadata": point.payload.get("metadata", {})
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"Error getting document chunks for {document_name}: {e}")
            return [] 