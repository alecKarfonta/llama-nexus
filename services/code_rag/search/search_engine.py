"""
Code search engine for Code RAG system.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.entities import AnyEntity, FunctionEntity, ClassEntity, VariableEntity, ModuleEntity
from ..vectorstore.embeddings import CodeEmbedder


class QueryIntent(Enum):
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    PATTERN = "pattern"
    EXACT = "exact"


@dataclass
class SearchContext:
    """Context information for search queries."""
    language: Optional[str] = None
    file_path: Optional[str] = None
    project_name: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Individual search result."""
    entity: AnyEntity
    score: float
    explanation: str
    match_type: str  # "semantic", "exact", "partial"
    metadata: Dict[str, Any]


@dataclass
class SearchResponse:
    """Complete search response."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    suggestions: List[str]
    query_intent: QueryIntent


class CodeSearchEngine:
    """Main search engine for code entities."""
    
    def __init__(self):
        self.embedder = CodeEmbedder()
        self.entities: List[AnyEntity] = []
        self.entity_embeddings: List = []
        self._indexed = False
    
    def add_entities(self, entities: List[AnyEntity]):
        """Add entities to the search index."""
        self.entities.extend(entities)
        
        # Generate embeddings for new entities
        new_embeddings = []
        for entity in entities:
            embedding = self.embedder.embed_entity(entity)
            new_embeddings.append(embedding)
        
        self.entity_embeddings.extend(new_embeddings)
        self._indexed = True

        # Index structural information
        self._index_structural_info()
    
    def _index_structural_info(self):
        """Index structural information for faster search."""
        # Create a map of entity names to entities for quick lookup
        self.entity_map = {entity.name: entity for entity in self.entities}
        
        # Create a map of functions that are called by other functions
        self.call_map = {}
        for entity in self.entities:
            if isinstance(entity, FunctionEntity) and entity.calls:
                for called_function in entity.calls:
                    if called_function not in self.call_map:
                        self.call_map[called_function] = []
                    self.call_map[called_function].append(entity.name)
    
    def search(self, query: str, context: Optional[SearchContext] = None, 
              top_k: int = 10, threshold: float = 0.0) -> SearchResponse:
        """Main search entry point."""
        start_time = time.time()
        
        if not self._indexed:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time_ms=0.0,
                suggestions=[],
                query_intent=QueryIntent.SEMANTIC
            )
        
        # Classify query intent
        intent = self._classify_query_intent(query)
        print(f"Query: '{query}', Intent: {intent}") # DEBUG
        
        # Execute search based on intent
        if intent == QueryIntent.SEMANTIC:
            results = self._semantic_search(query, top_k, threshold)
        elif intent == QueryIntent.EXACT:
            results = self._exact_search(query, top_k)
        elif intent == QueryIntent.STRUCTURAL:
            results = self._structural_search(query, top_k)
        elif intent == QueryIntent.FUNCTIONAL:
            results = self._functional_search(query, top_k, threshold)
        else:
            # Default to semantic search
            results = self._semantic_search(query, top_k, threshold)
        
        # Apply context filtering if provided
        if context:
            results = self._apply_context_filter(results, context)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(query, results)
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time,
            suggestions=suggestions,
            query_intent=intent
        )
    
    def _semantic_search(self, query: str, top_k: int, threshold: float) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Find similar entities
        similar_entities = self.embedder.find_similar_entities(
            query_embedding, 
            self.entity_embeddings, 
            self.entities, 
            top_k, 
            threshold
        )
        
        # Convert to search results
        results = []
        for similarity, entity in similar_entities:
            result = SearchResult(
                entity=entity,
                score=similarity,
                explanation=f"Semantic similarity: {similarity:.3f}",
                match_type="semantic",
                metadata={"similarity": similarity}
            )
            results.append(result)
        
        return results
    
    def _exact_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Perform exact name matching."""
        results = []
        query_lower = query.lower()
        
        for entity in self.entities:
            score = 0.0
            match_type = "none"
            
            # Exact name match
            if entity.name.lower() == query_lower:
                score = 1.0
                match_type = "exact"
            # Partial name match
            elif query_lower in entity.name.lower():
                score = len(query) / len(entity.name)
                match_type = "partial"
            
            if score > 0:
                result = SearchResult(
                    entity=entity,
                    score=score,
                    explanation=f"{match_type.title()} name match",
                    match_type=match_type,
                    metadata={"match_type": match_type}
                )
                results.append(result)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _structural_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Search based on code structure (inheritance, calls, etc.)."""
        results = []
        query_parts = query.lower().split()
        
        # Simple structural query parsing (e.g., "calls <function_name>")
        if len(query_parts) == 2 and query_parts[0] == "calls":
            called_function = query_parts[1]
            if called_function in self.call_map:
                calling_functions = self.call_map[called_function]
                for func_name in calling_functions:
                    if func_name in self.entity_map:
                        entity = self.entity_map[func_name]
                        result = SearchResult(
                            entity=entity,
                            score=0.9,  # High score for direct structural match
                            explanation=f"Calls {called_function}",
                            match_type="structural",
                            metadata={"type": "function_call"}
                        )
                        results.append(result)

        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
    
    def _functional_search(self, query: str, top_k: int, threshold: float) -> List[SearchResult]:
        """Search based on functional purpose (docstrings)."""
        # This is essentially a semantic search on docstrings
        docstrings = [entity.docstring for entity in self.entities if hasattr(entity, 'docstring') and entity.docstring]
        entities_with_docstrings = [entity for entity in self.entities if hasattr(entity, 'docstring') and entity.docstring]
        
        if not docstrings:
            return []
            
        # Embed the query and docstrings
        print(f"Functional search query: {query}") # DEBUG
        print(f"Entities with docstrings: {[entity.name for entity in entities_with_docstrings]}") # DEBUG
        query_embedding = self.embedder.embed_query(query)
        docstring_embeddings = [self.embedder.embed_query(doc) for doc in docstrings]
        
        # Find similar docstrings
        similar_entities = self.embedder.find_similar_entities(
            query_embedding,
            docstring_embeddings,
            entities_with_docstrings,
            top_k,
            threshold
        )
        
        # Convert to search results
        results = []
        for similarity, entity in similar_entities:
            result = SearchResult(
                entity=entity,
                score=similarity,
                explanation=f"Functional match on docstring (similarity: {similarity:.3f})",
                match_type="functional",
                metadata={"similarity": similarity}
            )
            results.append(result)
            
        return results
    
    def _classify_query_intent(self, query: str) -> QueryIntent:
        """Classify the user's query intent."""
        query_lower = query.lower()
        
        # Structural patterns
        structural_patterns = [
            "inherit", "extend", "implement", "subclass", "parent", "child",
            "calls", "uses", "imports", "depends"
        ]
        if any(pattern in query_lower for pattern in structural_patterns):
            return QueryIntent.STRUCTURAL
        
        # Functional patterns
        functional_patterns = [
            "authentication", "logging", "database", "api", "validation",
            "parsing", "error handling", "security", "encryption"
        ]
        if any(pattern in query_lower for pattern in functional_patterns):
            return QueryIntent.FUNCTIONAL
        
        # Pattern search
        pattern_keywords = [
            "pattern", "singleton", "factory", "observer", "decorator",
            "strategy", "adapter", "builder"
        ]
        if any(keyword in query_lower for keyword in pattern_keywords):
            return QueryIntent.PATTERN
        
        # Exact match patterns
        if len(query.split()) == 1 and query.isidentifier():
            return QueryIntent.EXACT
        
        # Default to semantic
        return QueryIntent.SEMANTIC
    
    def _apply_context_filter(self, results: List[SearchResult], 
                            context: SearchContext) -> List[SearchResult]:
        """Apply context-based filtering to results."""
        filtered_results = []
        
        for result in results:
            entity = result.entity
            include = True
            
            # Language filter
            if context.language and entity.language != context.language:
                include = False
            
            # File path filter
            if context.file_path and context.file_path not in entity.file_path:
                include = False
            
            # User preferences
            if context.user_preferences:
                # Example: filter by complexity
                max_complexity = context.user_preferences.get("max_complexity")
                if (max_complexity and isinstance(entity, FunctionEntity) 
                    and entity.complexity > max_complexity):
                    include = False
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _generate_suggestions(self, query: str, results: List[SearchResult]) -> List[str]:
        """Generate search suggestions based on query and results."""
        suggestions = []
        
        if not results:
            # Suggest related terms
            suggestions.extend([
                f"{query} function",
                f"{query} class",
                f"{query} method",
                f"implement {query}",
                f"{query} pattern"
            ])
        else:
            # Suggest based on found entities
            entity_types = set()
            for result in results[:5]:  # Look at top 5 results
                entity_types.add(result.entity.entity_type.value)
            
            for entity_type in entity_types:
                suggestions.append(f"{query} {entity_type}")
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        entity_counts = {}
        for entity in self.entities:
            entity_type = entity.entity_type.value
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        return {
            "total_entities": len(self.entities),
            "entity_counts": entity_counts,
            "indexed": self._indexed,
            "cache_size": self.embedder.get_cache_size()
        }
    
    def clear_index(self):
        """Clear the search index."""
        self.entities.clear()
        self.entity_embeddings.clear()
        self.embedder.clear_cache()
        self._indexed = False 