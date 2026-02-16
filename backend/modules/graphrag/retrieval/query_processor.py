from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
import re
import json
import os

@dataclass
class QueryIntent:
    """Represents the intent of a query."""
    primary_intent: str  # "factual", "analytical", "comparative", "temporal"
    confidence: float
    reasoning_type: str | None = None  # "multi_hop", "single_hop", "direct"

@dataclass
class QueryEntity:
    """Represents an entity extracted from a query."""
    name: str
    entity_type: str | None = None
    confidence: float = 1.0
    context: str | None = None

@dataclass
class QueryExpansion:
    """Represents query expansion terms."""
    original_query: str
    expanded_terms: List[str]
    reasoning: str | None = None

@dataclass
class ReasoningPath:
    """Represents a reasoning path for complex queries."""
    steps: List[str]
    entities_involved: List[str]
    expected_outcome: str
    confidence: float

class QueryProcessor:
    """Advanced query understanding and processing."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: str | None = None):
        """Initialize the query processor."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            try:
                self.llm = ChatAnthropic(
                    model=model_name,
                    temperature=0.1,
                    anthropic_api_key=self.api_key
                )
            except Exception as e:
                print(f"Warning: Could not initialize Claude: {e}")
                self.llm = None
        else:
            print("Warning: No ANTHROPIC_API_KEY found. LLM features will be disabled.")
            self.llm = None
        
        # Query patterns for different intents
        self.intent_patterns = {
            "factual": [
                r"what (is|are|does|do)",
                r"how (does|do|is|are)",
                r"when (does|do|is|are)",
                r"where (does|do|is|are)",
                r"which (one|system|component)"
            ],
            "analytical": [
                r"analyze",
                r"explain",
                r"describe",
                r"show me",
                r"tell me about",
                r"what are the (reasons|causes|effects)"
            ],
            "comparative": [
                r"compare",
                r"difference",
                r"versus",
                r"vs",
                r"better than",
                r"worse than",
                r"similar to"
            ],
            "temporal": [
                r"over time",
                r"evolution",
                r"history",
                r"timeline",
                r"when did",
                r"how has"
            ]
        }
    
    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a query."""
        query_lower = query.lower()
        
        # Check patterns for each intent
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return QueryIntent(
                        primary_intent=intent,
                        confidence=0.8,
                        reasoning_type="direct" if intent == "factual" else "multi_hop"
                    )
        
        # Default to factual if no pattern matches
        return QueryIntent(
            primary_intent="factual",
            confidence=0.5,
            reasoning_type="direct"
        )
    
    def extract_entities(self, query: str) -> List[QueryEntity]:
        """Extract entities from the query."""
        entities = []
        
        # Simple entity extraction (can be enhanced with LLM)
        words = query.split()
        for i, word in enumerate(words):
            # Look for capitalized words that might be entities
            if word[0].isupper() and len(word) > 2:
                # Get context (words around the entity)
                start = max(0, i - 2)
                end = min(len(words), i + 3)
                context = " ".join(words[start:end])
                
                entity = QueryEntity(
                    name=word,
                    entity_type=None,  # Could be determined by LLM
                    confidence=0.7,
                    context=context
                )
                entities.append(entity)
        
        return entities
    
    def expand_query(self, query: str, intent: QueryIntent) -> QueryExpansion:
        """Expand query with related terms based on intent."""
        expanded_terms = []
        reasoning = None
        
        if intent.primary_intent == "factual":
            # Add synonyms and related terms
            expanded_terms = self._expand_factual_query(query)
            reasoning = "Added synonyms and related terms for factual query"
        
        elif intent.primary_intent == "analytical":
            # Add analytical terms
            expanded_terms = self._expand_analytical_query(query)
            reasoning = "Added analytical and explanatory terms"
        
        elif intent.primary_intent == "comparative":
            # Add comparison terms
            expanded_terms = self._expand_comparative_query(query)
            reasoning = "Added comparison and contrast terms"
        
        return QueryExpansion(
            original_query=query,
            expanded_terms=expanded_terms,
            reasoning=reasoning
        )
    
    def _expand_factual_query(self, query: str) -> List[str]:
        """Expand factual queries with synonyms."""
        # Simple synonym expansion
        synonyms = {
            "brake": ["braking", "brake system", "brake components"],
            "engine": ["motor", "powerplant", "engine block"],
            "maintenance": ["service", "repair", "upkeep", "care"],
            "component": ["part", "element", "piece", "unit"],
            "system": ["mechanism", "apparatus", "setup"]
        }
        
        expanded = []
        query_lower = query.lower()
        
        for term, syns in synonyms.items():
            if term in query_lower:
                expanded.extend(syns)
        
        return expanded
    
    def _expand_analytical_query(self, query: str) -> List[str]:
        """Expand analytical queries with explanatory terms."""
        analytical_terms = [
            "analysis", "explanation", "details", "information",
            "how it works", "mechanism", "process", "function"
        ]
        return analytical_terms
    
    def _expand_comparative_query(self, query: str) -> List[str]:
        """Expand comparative queries with comparison terms."""
        comparison_terms = [
            "difference", "similarity", "comparison", "versus",
            "pros and cons", "advantages", "disadvantages"
        ]
        return comparison_terms
    
    def plan_reasoning_path(self, query: str, entities: List[QueryEntity], intent: QueryIntent) -> ReasoningPath:
        """Plan a reasoning path for complex queries."""
        if intent.reasoning_type == "multi_hop":
            # Plan multi-hop reasoning
            steps = [
                f"1. Identify relevant entities: {', '.join([e.name for e in entities])}",
                "2. Find direct relationships between entities",
                "3. Explore indirect connections (2-3 hops)",
                "4. Synthesize information from multiple paths"
            ]
            
            return ReasoningPath(
                steps=steps,
                entities_involved=[e.name for e in entities],
                expected_outcome="Comprehensive analysis with multiple perspectives",
                confidence=0.8
            )
        
        else:
            # Single-hop reasoning
            steps = [
                f"1. Find entity: {entities[0].name if entities else 'unknown'}",
                "2. Retrieve direct relationships",
                "3. Return relevant information"
            ]
            
            return ReasoningPath(
                steps=steps,
                entities_involved=[e.name for e in entities],
                expected_outcome="Direct answer to factual query",
                confidence=0.9
            )
    
    def process_query_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM for advanced query processing if available."""
        if not self.llm:
            return {"error": "LLM not available"}
        
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following query and extract:
        1. Intent (factual, analytical, comparative, temporal)
        2. Entities mentioned
        3. Key concepts
        4. Suggested reasoning approach
        
        Query: {query}
        
        Return as JSON:
        {{
            "intent": "intent_type",
            "confidence": 0.9,
            "entities": [
                {{"name": "entity_name", "type": "entity_type", "confidence": 0.8}}
            ],
            "concepts": ["concept1", "concept2"],
            "reasoning_approach": "single_hop|multi_hop|comparative"
        }}
        """)
        
        try:
            response = self.llm.invoke(prompt.format(query=query))
            return json.loads(response.content)
        except Exception as e:
            return {"error": f"LLM processing failed: {e}"}
    
    def get_query_analysis(self, query: str) -> Dict[str, Any]:
        """Get comprehensive query analysis."""
        # Basic analysis
        intent = self.classify_intent(query)
        entities = self.extract_entities(query)
        expansion = self.expand_query(query, intent)
        reasoning_path = self.plan_reasoning_path(query, entities, intent)
        
        # LLM analysis if available
        llm_analysis = self.process_query_llm(query)
        
        return {
            "query": query,
            "intent": intent,
            "entities": entities,
            "expansion": expansion,
            "reasoning_path": reasoning_path,
            "llm_analysis": llm_analysis
        }
    
    def process_query(self, query: str) -> str:
        """Process a query and return a response (placeholder for RAG integration)."""
        try:
            # Get query analysis
            analysis = self.get_query_analysis(query)
            
            # For now, return a simple response based on intent
            intent = analysis["intent"].primary_intent
            entities = [e.name for e in analysis["entities"]]
            
            if intent == "factual":
                if entities:
                    return f"Looking for factual information about {', '.join(entities)}"
                else:
                    return f"Searching for factual information about: {query}"
            
            elif intent == "analytical":
                return f"Analyzing the query: {query}"
            
            elif intent == "comparative":
                return f"Comparing entities mentioned in: {query}"
            
            else:
                return f"Processing query: {query}"
                
        except Exception as e:
            return f"Error processing query: {str(e)}" 