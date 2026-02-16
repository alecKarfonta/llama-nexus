#!/usr/bin/env python3

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
import os
from langchain_anthropic import ChatAnthropic
from entity_extractor import Entity, Relationship
from entity_linker import EntityLinker
from advanced_reasoning_engine import AdvancedReasoningEngine, ReasoningPath
from entity_extractor import EntityExtractor
from hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

@dataclass
class QueryExpansion:
    """Represents query expansion results."""
    original_query: str
    expanded_terms: List[str]
    entity_links: List[str]
    relationship_terms: List[str]
    reasoning_paths: List[ReasoningPath]
    confidence: float

@dataclass
class EnhancedSearchResult:
    """Enhanced search result with graph-based reasoning."""
    query: str
    results: List[Dict[str, Any]]
    reasoning_paths: List[ReasoningPath]
    inferred_relationships: List[Relationship]
    entity_clusters: List[List[str]]
    explanation: List[str]
    confidence: float

@dataclass
class QueryIntent:
    """Represents the intent of a user query."""
    intent_type: str
    confidence: float
    entities: List[str]
    reasoning_required: bool
    search_strategy: str
    complexity_level: str
    follow_up_questions: List[str]

@dataclass
class SearchStrategy:
    """Represents a search strategy for a query."""
    components: List[Dict[str, Any]]
    reasoning_paths: List[ReasoningPath]
    confidence: float
    explanation: str

@dataclass
class EnhancedQueryResult:
    """Enhanced query result with reasoning and explanations."""
    answer: str
    sources: List[Dict[str, Any]]
    reasoning_paths: List[ReasoningPath]
    confidence: float
    search_strategy: SearchStrategy
    follow_up_suggestions: List[str]
    explanation: str

class EnhancedQueryProcessor:
    """
    Enhanced query processor with advanced reasoning capabilities.
    
    Integrates query understanding, reasoning, and multi-strategy search
    to provide comprehensive answers to complex queries.
    """
    
    def __init__(self):
        """Initialize the enhanced query processor."""
        self.entity_linker = EntityLinker()
        self.reasoning_engine = AdvancedReasoningEngine()
        self.entity_extractor = EntityExtractor()
        self.hybrid_retriever = HybridRetriever()
        try:
            from graph_reasoner import GraphReasoner
            self.graph_reasoner = GraphReasoner()
        except ImportError:
            self.graph_reasoner = None
        
        # Initialize LLM client
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                self.llm = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=0.1,
                    max_tokens=2048,
                    anthropic_api_key=api_key
                )
            except Exception as e:
                logger.warning(f"Could not initialize LLM: {e}")
                self.llm = None
        else:
            logger.warning("No ANTHROPIC_API_KEY found. LLM features will be disabled.")
            self.llm = None
        
        # Query patterns for different reasoning types
        self.reasoning_patterns = {
            "relationship_query": [
                r"how (is|are) (.+) (related|connected|linked) to (.+)",
                r"what (is|are) the (relationship|connection) between (.+) and (.+)",
                r"explain (the )?(relationship|connection) between (.+) and (.+)"
            ],
            "multi_hop_query": [
                r"what (connects|links|relates) (.+) to (.+)",
                r"find (the )?(path|route) from (.+) to (.+)",
                r"how (does|do) (.+) (connect|link|relate) to (.+)"
            ],
            "entity_exploration": [
                r"what (is|are) (.+) (related|connected|linked) to",
                r"find (all )?(entities|components) (related|connected|linked) to (.+)",
                r"explore (.+) (relationships|connections)"
            ]
        }
        
        # Intent classification patterns
        self.intent_patterns = {
            'factual': {
                'keywords': ['what is', 'who is', 'when', 'where', 'how many'],
                'confidence_boost': 0.1
            },
            'comparative': {
                'keywords': ['compare', 'difference', 'similar', 'versus', 'vs', 'better', 'worse'],
                'confidence_boost': 0.2
            },
            'causal': {
                'keywords': ['cause', 'effect', 'because', 'leads to', 'results in', 'due to'],
                'confidence_boost': 0.3
            },
            'procedural': {
                'keywords': ['how to', 'steps', 'process', 'procedure', 'method'],
                'confidence_boost': 0.2
            },
            'analytical': {
                'keywords': ['analyze', 'explain', 'why', 'reason', 'analysis'],
                'confidence_boost': 0.3
            },
            'temporal': {
                'keywords': ['timeline', 'history', 'evolution', 'before', 'after', 'during'],
                'confidence_boost': 0.2
            }
        }
    
    def process_enhanced_query(self, query: str, context: Optional[Dict] = None) -> EnhancedQueryResult:
        """
        Process a query with enhanced capabilities.
        
        Args:
            query: The user query
            context: Optional context information
            
        Returns:
            Enhanced query result with reasoning and explanations
        """
        # 1. Analyze query intent
        query_intent = self.analyze_query_intent(query)
        
        # 2. Extract entities
        extraction_result = self.entity_extractor.extract_entities_and_relations(query)
        entities = [entity.name for entity in extraction_result.entities]
        
        # 3. Execute reasoning if required
        reasoning_paths = []
        if query_intent.reasoning_required and self.reasoning_engine:
            try:
                reasoning_paths = self.reasoning_engine.execute_reasoning(query, entities)
            except Exception as e:
                logger.warning(f"Reasoning execution failed: {e}")
                reasoning_paths = []
        
        # 4. Plan search strategy
        search_strategy = self.plan_search_strategy(query_intent, reasoning_paths)
        
        # 5. Execute search
        search_results = self.execute_search_strategy(search_strategy, query)
        
        # 6. Generate comprehensive answer
        answer = self.generate_enhanced_answer(query, search_results, reasoning_paths, query_intent)
        
        # 7. Generate follow-up suggestions
        follow_up_suggestions = self.generate_follow_up_suggestions(query_intent, reasoning_paths)
        
        # 8. Create explanation
        explanation = self.create_explanation(query_intent, reasoning_paths, search_strategy)
        
        return EnhancedQueryResult(
            answer=answer,
            sources=search_results,
            reasoning_paths=reasoning_paths,
            confidence=query_intent.confidence,
            search_strategy=search_strategy,
            follow_up_suggestions=follow_up_suggestions,
            explanation=explanation
        )
    
    def analyze_query_intent(self, query: str) -> QueryIntent:
        """
        Analyze query intent using pattern matching and LLM.
        
        Args:
            query: The user query
            
        Returns:
            QueryIntent object with intent classification
        """
        # Pattern-based intent detection
        base_intent = self._pattern_based_intent_detection(query)
        
        # LLM-based intent refinement
        refined_intent = self._llm_intent_refinement(query, base_intent)
        
        # Determine reasoning requirements
        reasoning_required = self._determine_reasoning_requirement(refined_intent)
        
        # Plan search strategy
        search_strategy = self._plan_search_strategy(refined_intent)
        
        return QueryIntent(
            intent_type=refined_intent['intent'],
            confidence=refined_intent['confidence'],
            entities=refined_intent.get('entities', []),
            reasoning_required=reasoning_required,
            search_strategy=search_strategy,
            complexity_level=refined_intent.get('complexity', 'medium'),
            follow_up_questions=refined_intent.get('follow_up_questions', [])
        )
    
    def _pattern_based_intent_detection(self, query: str) -> Dict[str, Any]:
        """Detect intent using pattern matching."""
        query_lower = query.lower()
        detected_intents = []
        
        for intent_type, pattern_info in self.intent_patterns.items():
            for keyword in pattern_info['keywords']:
                if keyword in query_lower:
                    detected_intents.append({
                        'intent': intent_type,
                        'confidence': 0.6 + pattern_info['confidence_boost'],
                        'keyword': keyword
                    })
        
        if detected_intents:
            # Return the highest confidence intent
            best_intent = max(detected_intents, key=lambda x: x['confidence'])
            return {
                'intent': best_intent['intent'],
                'confidence': best_intent['confidence'],
                'detection_method': 'pattern'
            }
        
        return {
            'intent': 'factual',
            'confidence': 0.5,
            'detection_method': 'default'
        }
    
    def _llm_intent_refinement(self, query: str, base_intent: Dict[str, Any]) -> Dict[str, Any]:
        """Refine intent classification using LLM."""
        prompt = f"""
        Analyze the following query and provide detailed intent classification:
        
        Query: "{query}"
        Base Intent: {base_intent['intent']} (confidence: {base_intent['confidence']})
        
        Provide a JSON response with:
        {{
            "intent": "refined_intent_type",
            "confidence": 0.95,
            "entities": ["entity1", "entity2"],
            "complexity": "low|medium|high",
            "reasoning_required": true/false,
            "search_strategy": "vector|graph|hybrid|multi_hop",
            "follow_up_questions": ["question1", "question2"]
        }}
        
        Intent types:
        - FACTUAL: Seeking specific facts or information
        - COMPARATIVE: Comparing entities or concepts
        - CAUSAL: Cause-effect relationships
        - PROCEDURAL: How-to or step-by-step instructions
        - ANALYTICAL: Deep analysis or explanation
        - TEMPORAL: Time-based or historical information
        """
        
        try:
            if self.llm:
                response = self.llm.invoke(prompt)
                refined_intent = json.loads(response.content)
            else:
                refined_intent = base_intent
            
            # Merge with base intent
            refined_intent['base_intent'] = base_intent['intent']
            refined_intent['detection_method'] = 'llm_refined'
            
            return refined_intent
        except Exception as e:
            logger.warning(f"LLM intent refinement failed: {e}")
            return base_intent
    
    def _determine_reasoning_requirement(self, intent: Dict[str, Any]) -> bool:
        """Determine if reasoning is required for this intent."""
        reasoning_intents = ['causal', 'comparative', 'analytical', 'temporal']
        return intent.get('intent', '').lower() in reasoning_intents
    
    def _plan_search_strategy(self, intent: Dict[str, Any]) -> str:
        """Plan the search strategy based on intent."""
        intent_type = intent.get('intent', '').lower()
        
        if intent_type == 'causal':
            return 'graph'
        elif intent_type == 'comparative':
            return 'hybrid'
        elif intent_type == 'analytical':
            return 'multi_hop'
        elif intent_type == 'temporal':
            return 'graph'
        elif intent_type == 'procedural':
            return 'hybrid'
        else:
            return 'vector'
    
    def plan_search_strategy(self, query_intent: QueryIntent, reasoning_paths: List[ReasoningPath]) -> SearchStrategy:
        """
        Plan a comprehensive search strategy.
        
        Args:
            query_intent: The analyzed query intent
            reasoning_paths: Reasoning paths from the reasoning engine
            
        Returns:
            SearchStrategy object
        """
        components = []
        
        # Add reasoning component if reasoning paths exist
        if reasoning_paths:
            components.append({
                'type': 'reasoning',
                'priority': 1,
                'paths': reasoning_paths,
                'confidence': max(path.confidence for path in reasoning_paths) if reasoning_paths else 0.0
            })
        
        # Add search components based on intent
        if query_intent.search_strategy == 'vector':
            components.append({
                'type': 'vector_search',
                'priority': 2,
                'query': query_intent.intent_type,
                'confidence': 0.7
            })
        elif query_intent.search_strategy == 'graph':
            components.append({
                'type': 'graph_traversal',
                'priority': 2,
                'entities': query_intent.entities,
                'confidence': 0.8
            })
        elif query_intent.search_strategy == 'hybrid':
            components.append({
                'type': 'vector_search',
                'priority': 2,
                'query': query_intent.intent_type,
                'confidence': 0.7
            })
            components.append({
                'type': 'graph_traversal',
                'priority': 3,
                'entities': query_intent.entities,
                'confidence': 0.8
            })
        elif query_intent.search_strategy == 'multi_hop':
            components.append({
                'type': 'multi_hop_reasoning',
                'priority': 1,
                'max_hops': 3,
                'confidence': 0.6
            })
            components.append({
                'type': 'vector_search',
                'priority': 2,
                'query': query_intent.intent_type,
                'confidence': 0.7
            })
        
        # Sort components by priority
        components.sort(key=lambda x: x['priority'])
        
        # Calculate overall confidence
        confidence = sum(comp['confidence'] for comp in components) / len(components) if components else 0.0
        
        # Generate explanation
        explanation = self._generate_strategy_explanation(components, query_intent)
        
        return SearchStrategy(
            components=components,
            reasoning_paths=reasoning_paths,
            confidence=confidence,
            explanation=explanation
        )
    
    def _generate_strategy_explanation(self, components: List[Dict], query_intent: QueryIntent) -> str:
        """Generate explanation for the search strategy."""
        explanations = []
        
        for component in components:
            if component['type'] == 'reasoning':
                explanations.append("Using advanced reasoning to find relationships")
            elif component['type'] == 'vector_search':
                explanations.append("Searching for semantically similar content")
            elif component['type'] == 'graph_traversal':
                explanations.append("Traversing knowledge graph for direct relationships")
            elif component['type'] == 'multi_hop_reasoning':
                explanations.append("Using multi-hop reasoning for complex relationships")
        
        return f"Query intent: {query_intent.intent_type}. Strategy: {' + '.join(explanations)}"
    
    def execute_search_strategy(self, search_strategy: SearchStrategy, query: str) -> List[Dict[str, Any]]:
        """
        Execute the planned search strategy.
        
        Args:
            search_strategy: The planned search strategy
            query: The original query
            
        Returns:
            List of search results
        """
        all_results = []
        
        # Analyze query to get entities for graph search
        query_analysis = self.hybrid_retriever.analyze_query(query)
        
        for component in search_strategy.components:
            if component['type'] == 'vector_search':
                results = self.hybrid_retriever.vector_search(query, top_k=5)
                all_results.extend(results)
            elif component['type'] == 'graph_traversal':
                results = self.hybrid_retriever.graph_search(query, query_analysis.entities, depth=2)
                all_results.extend(results)
            elif component['type'] == 'multi_hop_reasoning':
                # Multi-hop reasoning results are already in reasoning_paths
                continue
        
        # Convert SearchResult objects to dictionaries and remove duplicates
        result_dicts = []
        for result in all_results:
            result_dicts.append({
                'content': result.content,
                'source': result.source,
                'score': result.score,
                'result_type': result.result_type,
                'metadata': result.metadata or {}
            })
        
        # Remove duplicates and sort by relevance
        unique_results = self._deduplicate_results(result_dicts)
        return sorted(unique_results, key=lambda x: x.get('score', 0), reverse=True)
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate search results."""
        seen = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result.get('content', ''))
            if content_hash not in seen:
                seen.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def generate_enhanced_answer(self, query: str, search_results: List[Dict], 
                                reasoning_paths: List[ReasoningPath], query_intent: QueryIntent) -> str:
        """
        Generate an enhanced answer using search results and reasoning.
        
        Args:
            query: The original query
            search_results: Search results from various strategies
            reasoning_paths: Reasoning paths from the reasoning engine
            query_intent: The analyzed query intent
            
        Returns:
            Enhanced answer string
        """
        # Prepare context for LLM
        context = self._prepare_answer_context(search_results, reasoning_paths, query_intent)
        
        # Generate answer using LLM
        prompt = self._create_answer_prompt(query, context, query_intent)
        
        try:
            if self.llm:
                response = self.llm.invoke(prompt)
                return response.content
            else:
                return self._generate_fallback_answer(query, search_results)
        except Exception as e:
            logger.error(f"Failed to generate enhanced answer: {e}")
            return self._generate_fallback_answer(query, search_results)
    
    def _prepare_answer_context(self, search_results: List[Dict], 
                               reasoning_paths: List[ReasoningPath], 
                               query_intent: QueryIntent) -> str:
        """Prepare context for answer generation."""
        context_parts = []
        
        # Add search results
        if search_results:
            context_parts.append("Relevant information:")
            for i, result in enumerate(search_results[:3], 1):
                context_parts.append(f"{i}. {result.get('content', '')}")
        
        # Add reasoning paths
        if reasoning_paths:
            context_parts.append("\nReasoning paths:")
            for path in reasoning_paths:
                explanation = self.reasoning_engine.generate_reasoning_explanation([path])
                context_parts.append(f"- {explanation}")
        
        # Add intent information
        context_parts.append(f"\nQuery intent: {query_intent.intent_type}")
        context_parts.append(f"Complexity level: {query_intent.complexity_level}")
        
        return "\n".join(context_parts)
    
    def _create_answer_prompt(self, query: str, context: str, query_intent: QueryIntent) -> str:
        """Create a prompt for answer generation."""
        intent_instructions = {
            'factual': "Provide a clear, factual answer with specific details.",
            'comparative': "Compare the entities or concepts, highlighting similarities and differences.",
            'causal': "Explain the cause-effect relationships and their implications.",
            'procedural': "Provide step-by-step instructions or procedures.",
            'analytical': "Provide a detailed analysis with insights and implications.",
            'temporal': "Explain the timeline, sequence, or historical context."
        }
        
        instruction = intent_instructions.get(query_intent.intent_type.lower(), 
                                           "Provide a comprehensive answer.")
        
        prompt = f"""
        Answer the following query using the provided context:
        
        Query: {query}
        
        Context:
        {context}
        
        Instructions: {instruction}
        
        Provide a comprehensive, well-structured answer that directly addresses the query.
        Include relevant details from the context and explain any reasoning used.
        """
        
        return prompt
    
    def _generate_fallback_answer(self, query: str, search_results: List[Dict]) -> str:
        """Generate a fallback answer if LLM generation fails."""
        if not search_results:
            return "I couldn't find relevant information to answer your query."
        
        # Use the most relevant search result
        best_result = search_results[0]
        return f"Based on the available information: {best_result.get('content', '')}"
    
    def generate_follow_up_suggestions(self, query_intent: QueryIntent, 
                                     reasoning_paths: List[ReasoningPath]) -> List[str]:
        """
        Generate follow-up question suggestions.
        
        Args:
            query_intent: The analyzed query intent
            reasoning_paths: Reasoning paths from the reasoning engine
            
        Returns:
            List of follow-up question suggestions
        """
        suggestions = []
        
        # Add intent-specific suggestions
        if query_intent.intent_type.lower() == 'comparative':
            suggestions.extend([
                "What are the key differences between these entities?",
                "How do these entities compare in terms of performance?",
                "What are the advantages and disadvantages of each?"
            ])
        elif query_intent.intent_type.lower() == 'causal':
            suggestions.extend([
                "What are the underlying causes?",
                "What are the potential effects or consequences?",
                "Are there any intermediate factors involved?"
            ])
        elif query_intent.intent_type.lower() == 'procedural':
            suggestions.extend([
                "What are the prerequisites for this process?",
                "What are the common pitfalls to avoid?",
                "Are there alternative approaches?"
            ])
        
        # Add reasoning-based suggestions
        for path in reasoning_paths:
            if path.path_type == 'causal':
                suggestions.append(f"What causes {path.entities[0]} to affect {path.entities[-1]}?")
            elif path.path_type == 'comparative':
                suggestions.append(f"How do {path.entities[0]} and {path.entities[1]} differ in practice?")
        
        # Add general suggestions
        suggestions.extend([
            "Can you provide more specific details?",
            "What are the implications of this information?",
            "How does this relate to other concepts in the knowledge base?"
        ])
        
        return suggestions[:5]  # Limit to 5 suggestions
    
    def create_explanation(self, query_intent: QueryIntent, reasoning_paths: List[ReasoningPath], 
                          search_strategy: SearchStrategy) -> str:
        """
        Create a comprehensive explanation of the query processing.
        
        Args:
            query_intent: The analyzed query intent
            reasoning_paths: Reasoning paths from the reasoning engine
            search_strategy: The search strategy used
            
        Returns:
            Explanation string
        """
        explanation_parts = []
        
        # Intent explanation
        explanation_parts.append(f"Query Intent: {query_intent.intent_type} (confidence: {query_intent.confidence:.2f})")
        explanation_parts.append(f"Complexity Level: {query_intent.complexity_level}")
        
        # Reasoning explanation
        if reasoning_paths:
            explanation_parts.append(f"\nReasoning Applied:")
            for path in reasoning_paths:
                explanation_parts.append(f"- {path.path_type.title()} reasoning: {path.reasoning_chain[0] if path.reasoning_chain else 'N/A'}")
        
        # Search strategy explanation
        explanation_parts.append(f"\nSearch Strategy: {search_strategy.explanation}")
        explanation_parts.append(f"Strategy Confidence: {search_strategy.confidence:.2f}")
        
        return "\n".join(explanation_parts)
    
    def process_query(self, query: str, entities: List[Entity], relationships: List[Relationship]) -> EnhancedSearchResult:
        """Process a query with graph-based reasoning."""
        # Build the knowledge graph
        self.graph_reasoner.build_graph(entities, relationships)
        
        # Analyze query type
        query_type = self._classify_query(query)
        
        # Extract entities from query
        query_entities = self._extract_entities_from_query(query, entities)
        
        # Perform query expansion
        expansion = self._expand_query(query, query_entities, relationships)
        
        # Execute search based on query type
        if query_type == "relationship_query":
            result = self._handle_relationship_query(query, query_entities)
        elif query_type == "multi_hop_query":
            result = self._handle_multi_hop_query(query, query_entities)
        elif query_type == "entity_exploration":
            result = self._handle_entity_exploration_query(query, query_entities)
        else:
            result = self._handle_general_query(query, expansion)
        
        return result
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        for query_type, patterns in self.reasoning_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return "general_query"
    
    def _extract_entities_from_query(self, query: str, available_entities: List[Entity]) -> List[Entity]:
        """Extract entities mentioned in the query."""
        query_lower = query.lower()
        found_entities = []
        
        for entity in available_entities:
            entity_name_lower = entity.name.lower()
            
            # Check for exact match
            if entity_name_lower in query_lower:
                found_entities.append(entity)
            # Check for partial match
            elif any(word in entity_name_lower for word in entity_name_lower.split()):
                if any(word in query_lower for word in entity_name_lower.split()):
                    found_entities.append(entity)
        
        return found_entities
    
    def _expand_query(self, query: str, query_entities: List[Entity], relationships: List[Relationship]) -> QueryExpansion:
        """Expand query using graph-based reasoning."""
        expanded_terms = []
        entity_links = []
        relationship_terms = []
        reasoning_paths = []
        
        # Find related entities for each query entity
        for entity in query_entities:
            related = self.graph_reasoner.find_related_entities(entity.name, max_hops=2)
            
            for hop_level, related_list in related.items():
                for related_entity, confidence in related_list:
                    if confidence > 0.5:  # Only include high-confidence relationships
                        expanded_terms.append(related_entity)
                        entity_links.append(f"{entity.name} -> {related_entity}")
        
        # Extract relationship terms from the query
        relationship_keywords = ["related", "connected", "linked", "part of", "contains", "works for", "located in"]
        for keyword in relationship_keywords:
            if keyword in query.lower():
                relationship_terms.append(keyword)
        
        # Find reasoning paths if we have multiple entities
        if len(query_entities) >= 2:
            for i in range(len(query_entities)):
                for j in range(i + 1, len(query_entities)):
                    paths = self.graph_reasoner.find_paths(
                        query_entities[i].name, 
                        query_entities[j].name, 
                        max_hops=3
                    )
                    reasoning_paths.extend(paths)
        
        return QueryExpansion(
            original_query=query,
            expanded_terms=expanded_terms,
            entity_links=entity_links,
            relationship_terms=relationship_terms,
            reasoning_paths=reasoning_paths,
            confidence=0.8
        )
    
    def _handle_relationship_query(self, query: str, query_entities: List[Entity]) -> EnhancedSearchResult:
        """Handle queries asking about relationships between entities."""
        if len(query_entities) < 2:
            return self._create_empty_result(query, "Need at least two entities to analyze relationships")
        
        # Get the first two entities for relationship analysis
        entity1, entity2 = query_entities[0], query_entities[1]
        
        # Explain the relationship
        reasoning_result = self.graph_reasoner.explain_relationship(entity1.name, entity2.name)
        
        # Find related entities
        related_entities = self.graph_reasoner.find_related_entities(entity1.name, max_hops=2)
        
        # Create results
        results = []
        for hop_level, entities in related_entities.items():
            for entity_name, confidence in entities:
                if entity_name != entity1.name:
                    results.append({
                        "entity": entity_name,
                        "relationship": hop_level,
                        "confidence": confidence,
                        "source_entity": entity1.name
                    })
        
        return EnhancedSearchResult(
            query=query,
            results=results,
            reasoning_paths=reasoning_result.paths,
            inferred_relationships=reasoning_result.inferred_relationships,
            entity_clusters=self.graph_reasoner.find_entity_clusters(),
            explanation=reasoning_result.reasoning_steps,
            confidence=reasoning_result.confidence
        )
    
    def _handle_multi_hop_query(self, query: str, query_entities: List[Entity]) -> EnhancedSearchResult:
        """Handle multi-hop reasoning queries."""
        if len(query_entities) < 2:
            return self._create_empty_result(query, "Need at least two entities for multi-hop reasoning")
        
        entity1, entity2 = query_entities[0], query_entities[1]
        
        # Find all paths between entities
        paths = self.graph_reasoner.find_paths(entity1.name, entity2.name, max_hops=4)
        
        # Infer relationships from paths
        inferred_rels = self.graph_reasoner.infer_relationships(entity1.name, entity2.name, max_hops=4)
        
        # Create results showing the paths
        results = []
        for path in paths[:5]:  # Top 5 paths
            results.append({
                "path": path.path,
                "relationships": path.relationships,
                "confidence": path.confidence,
                "length": path.path_length
            })
        
        explanation = [
            f"Found {len(paths)} paths between {entity1.name} and {entity2.name}",
            f"Path lengths range from 1 to {max([p.path_length for p in paths]) if paths else 0} hops",
            f"Top path confidence: {max([p.confidence for p in paths]) if paths else 0:.2f}"
        ]
        
        return EnhancedSearchResult(
            query=query,
            results=results,
            reasoning_paths=paths,
            inferred_relationships=inferred_rels,
            entity_clusters=self.graph_reasoner.find_entity_clusters(),
            explanation=explanation,
            confidence=max([p.confidence for p in paths]) if paths else 0.0
        )
    
    def _handle_entity_exploration_query(self, query: str, query_entities: List[Entity]) -> EnhancedSearchResult:
        """Handle queries exploring entities and their relationships."""
        if not query_entities:
            return self._create_empty_result(query, "No entities found in query")
        
        entity = query_entities[0]
        
        # Find related entities
        related_entities = self.graph_reasoner.find_related_entities(entity.name, max_hops=3)
        
        # Get entity centrality
        centrality = self.graph_reasoner.get_entity_centrality()
        
        # Create results
        results = []
        for hop_level, entities in related_entities.items():
            for entity_name, confidence in entities:
                centrality_info = centrality.get(entity_name, {})
                results.append({
                    "entity": entity_name,
                    "relationship_level": hop_level,
                    "confidence": confidence,
                    "centrality": centrality_info.get("overall", 0),
                    "source_entity": entity.name
                })
        
        # Sort by confidence
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        explanation = [
            f"Found {len(results)} entities related to {entity.name}",
            f"Relationship levels: {list(related_entities.keys())}",
            f"Average confidence: {sum(r['confidence'] for r in results) / len(results) if results else 0:.2f}"
        ]
        
        return EnhancedSearchResult(
            query=query,
            results=results,
            reasoning_paths=[],
            inferred_relationships=[],
            entity_clusters=self.graph_reasoner.find_entity_clusters(),
            explanation=explanation,
            confidence=0.8
        )
    
    def _handle_general_query(self, query: str, expansion: QueryExpansion) -> EnhancedSearchResult:
        """Handle general queries with graph-based expansion."""
        # Use expanded terms to enhance search
        enhanced_query = query + " " + " ".join(expansion.expanded_terms[:5])
        
        # Find relevant entities based on expansion
        results = []
        for entity_link in expansion.entity_links[:10]:
            source, target = entity_link.split(" -> ")
            results.append({
                "entity": target,
                "relationship": "related to",
                "confidence": 0.7,
                "source_entity": source
            })
        
        explanation = [
            f"Expanded query with {len(expansion.expanded_terms)} terms",
            f"Found {len(expansion.entity_links)} entity relationships",
            f"Query expansion confidence: {expansion.confidence:.2f}"
        ]
        
        return EnhancedSearchResult(
            query=query,
            results=results,
            reasoning_paths=expansion.reasoning_paths,
            inferred_relationships=[],
            entity_clusters=self.graph_reasoner.find_entity_clusters(),
            explanation=explanation,
            confidence=expansion.confidence
        )
    
    def _create_empty_result(self, query: str, message: str) -> EnhancedSearchResult:
        """Create an empty search result."""
        return EnhancedSearchResult(
            query=query,
            results=[],
            reasoning_paths=[],
            inferred_relationships=[],
            entity_clusters=[],
            explanation=[message],
            confidence=0.0
        )
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get statistics about query processing."""
        graph_stats = self.graph_reasoner.get_graph_statistics()
        entity_stats = self.entity_linker.get_entity_statistics()
        
        return {
            "graph_statistics": graph_stats,
            "entity_statistics": entity_stats,
            "total_entities": len(self.graph_reasoner.entity_cache),
            "total_relationships": self.graph_reasoner.graph.number_of_edges()
        } 