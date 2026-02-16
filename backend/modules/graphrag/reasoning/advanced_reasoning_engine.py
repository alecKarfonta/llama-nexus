"""
Advanced Reasoning Engine for Graph RAG System

This module implements advanced reasoning capabilities including:
- Multi-hop reasoning across knowledge graph
- Causal reasoning for cause-effect relationships
- Comparative reasoning between entities
- Temporal reasoning for time-based queries
- Hierarchical reasoning for concept hierarchies
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from neo4j import Session
import re
from datetime import datetime, timedelta

from neo4j_conn import get_neo4j_session
from entity_extractor import EntityExtractor
from query_processor import QueryProcessor

logger = logging.getLogger(__name__)

@dataclass
class ReasoningPath:
    """Represents a reasoning path in the knowledge graph."""
    path_type: str
    entities: List[str]
    relationships: List[str]
    confidence: float
    evidence: List[str]
    reasoning_chain: List[str]

@dataclass
class ComparativeAnalysis:
    """Represents a comparative analysis between entities."""
    entity1: str
    entity2: str
    comparison_criteria: List[str]
    similarities: List[str]
    differences: List[str]
    conclusion: str
    confidence: float

@dataclass
class CausalChain:
    """Represents a causal chain in the knowledge graph."""
    cause: str
    effect: str
    intermediate_steps: List[str]
    confidence: float
    evidence: List[str]

class AdvancedReasoningEngine:
    """
    Advanced reasoning engine for complex query processing.
    
    Implements multi-hop reasoning, causal analysis, comparative reasoning,
    and temporal reasoning capabilities.
    """
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.query_processor = QueryProcessor()
        self.reasoning_patterns = {
            'causal': self.causal_reasoning,
            'comparative': self.comparative_reasoning,
            'temporal': self.temporal_reasoning,
            'hierarchical': self.hierarchical_reasoning,
            'multi_hop': self.multi_hop_reasoning
        }
        
    def analyze_query_complexity(self, query: str) -> Dict[str, Any]:
        """
        Analyze query complexity to determine reasoning strategy.
        
        Args:
            query: The user query to analyze
            
        Returns:
            Dict containing complexity analysis
        """
        complexity_indicators = {
            'causal_keywords': ['cause', 'effect', 'because', 'leads to', 'results in', 'due to'],
            'comparative_keywords': ['compare', 'difference', 'similar', 'versus', 'vs', 'better', 'worse'],
            'temporal_keywords': ['when', 'before', 'after', 'during', 'timeline', 'history'],
            'hierarchical_keywords': ['type of', 'category', 'classification', 'subtype'],
            'multi_hop_keywords': ['how', 'why', 'explain', 'process', 'steps']
        }
        
        query_lower = query.lower()
        detected_patterns = []
        
        for pattern, keywords in complexity_indicators.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_patterns.append(pattern.replace('_keywords', ''))
        
        # Determine primary reasoning type
        primary_reasoning = 'factual'
        if detected_patterns:
            # Prioritize more complex reasoning types
            if 'causal' in detected_patterns:
                primary_reasoning = 'causal'
            elif 'comparative' in detected_patterns:
                primary_reasoning = 'comparative'
            elif 'temporal' in detected_patterns:
                primary_reasoning = 'temporal'
            elif 'hierarchical' in detected_patterns:
                primary_reasoning = 'hierarchical'
            elif 'multi_hop' in detected_patterns:
                primary_reasoning = 'multi_hop'
        
        return {
            'primary_reasoning': primary_reasoning,
            'detected_patterns': detected_patterns,
            'complexity_level': 'high' if len(detected_patterns) > 1 else 'medium',
            'requires_multi_hop': 'multi_hop' in detected_patterns or 'causal' in detected_patterns
        }
    
    def execute_reasoning(self, query: str, entities: List[str]) -> List[ReasoningPath]:
        """
        Execute appropriate reasoning based on query analysis.
        
        Args:
            query: The user query
            entities: Extracted entities from the query
            
        Returns:
            List of reasoning paths
        """
        complexity = self.analyze_query_complexity(query)
        reasoning_type = complexity['primary_reasoning']
        
        if reasoning_type in self.reasoning_patterns:
            return self.reasoning_patterns[reasoning_type](query, entities)
        else:
            return self.multi_hop_reasoning(query, entities)
    
    def causal_reasoning(self, query: str, entities: List[str]) -> List[ReasoningPath]:
        """
        Perform causal reasoning to find cause-effect relationships.
        
        Args:
            query: The user query
            entities: Extracted entities from the query
            
        Returns:
            List of causal reasoning paths
        """
        reasoning_paths = []
        
        with get_neo4j_session() as session:
            for entity in entities:
                # Find causal chains in the knowledge graph
                causal_chains = self._find_causal_chains(session, entity)
                
                for chain in causal_chains:
                    reasoning_paths.append(ReasoningPath(
                        path_type='causal',
                        entities=[chain.cause, chain.effect],
                        relationships=['CAUSES'],
                        confidence=chain.confidence,
                        evidence=chain.evidence,
                        reasoning_chain=[f"{chain.cause} causes {chain.effect}"]
                    ))
        
        return reasoning_paths
    
    def _find_causal_chains(self, session: Session, entity: str) -> List[CausalChain]:
        """Find causal chains for a given entity."""
        chains = []
        
        # Query for direct causal relationships
        cypher_query = """
        MATCH path = (cause:Entity)-[:CAUSES*1..3]->(effect:Entity)
        WHERE cause.name = $entity_name OR effect.name = $entity_name
        RETURN path, cause, effect
        ORDER BY length(path) ASC
        LIMIT 10
        """
        
        results = session.run(cypher_query, entity_name=entity)
        
        for record in results:
            path = record['path']
            cause = record['cause']['name']
            effect = record['effect']['name']
            
            # Calculate confidence based on path length and relationship strength
            confidence = max(0.5, 1.0 - (len(path) - 1) * 0.1)
            
            chains.append(CausalChain(
                cause=cause,
                effect=effect,
                intermediate_steps=[node['name'] for node in path.nodes[1:-1]],
                confidence=confidence,
                evidence=[f"Direct causal relationship found in knowledge graph"]
            ))
        
        return chains
    
    def comparative_reasoning(self, query: str, entities: List[str]) -> List[ReasoningPath]:
        """
        Perform comparative reasoning between entities.
        
        Args:
            query: The user query
            entities: Extracted entities from the query
            
        Returns:
            List of comparative reasoning paths
        """
        if len(entities) < 2:
            return []
        
        reasoning_paths = []
        
        # Extract comparison criteria from query
        comparison_criteria = self._extract_comparison_criteria(query)
        
        with get_neo4j_session() as session:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    entity1, entity2 = entities[i], entities[j]
                    
                    comparison = self._compare_entities(session, entity1, entity2, comparison_criteria)
                    
                    if comparison:
                        reasoning_paths.append(ReasoningPath(
                            path_type='comparative',
                            entities=[entity1, entity2],
                            relationships=['COMPARED_TO'],
                            confidence=comparison.confidence,
                            evidence=comparison.similarities + comparison.differences,
                            reasoning_chain=[comparison.conclusion]
                        ))
        
        return reasoning_paths
    
    def _extract_comparison_criteria(self, query: str) -> List[str]:
        """Extract comparison criteria from the query."""
        criteria_keywords = {
            'performance': ['performance', 'speed', 'efficiency', 'power'],
            'cost': ['cost', 'price', 'expensive', 'cheap', 'affordable'],
            'quality': ['quality', 'reliability', 'durability', 'robust'],
            'features': ['features', 'capabilities', 'functions', 'specifications'],
            'size': ['size', 'dimensions', 'weight', 'volume']
        }
        
        query_lower = query.lower()
        criteria = []
        
        for criterion, keywords in criteria_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                criteria.append(criterion)
        
        return criteria if criteria else ['general']
    
    def _compare_entities(self, session: Session, entity1: str, entity2: str, criteria: List[str]) -> Optional[ComparativeAnalysis]:
        """Compare two entities based on specified criteria."""
        # Get entity properties from knowledge graph
        entity1_props = self._get_entity_properties(session, entity1)
        entity2_props = self._get_entity_properties(session, entity2)
        
        if not entity1_props or not entity2_props:
            return None
        
        similarities = []
        differences = []
        
        # Compare based on criteria
        for criterion in criteria:
            if criterion in entity1_props and criterion in entity2_props:
                if entity1_props[criterion] == entity2_props[criterion]:
                    similarities.append(f"Both have same {criterion}: {entity1_props[criterion]}")
                else:
                    differences.append(f"{criterion}: {entity1_props[criterion]} vs {entity2_props[criterion]}")
        
        if not similarities and not differences:
            return None
        
        # Generate conclusion
        if similarities and not differences:
            conclusion = f"{entity1} and {entity2} are very similar"
        elif differences and not similarities:
            conclusion = f"{entity1} and {entity2} are quite different"
        else:
            conclusion = f"{entity1} and {entity2} have both similarities and differences"
        
        confidence = min(0.9, 0.5 + len(similarities + differences) * 0.1)
        
        return ComparativeAnalysis(
            entity1=entity1,
            entity2=entity2,
            comparison_criteria=criteria,
            similarities=similarities,
            differences=differences,
            conclusion=conclusion,
            confidence=confidence
        )
    
    def _get_entity_properties(self, session: Session, entity: str) -> Dict[str, Any]:
        """Get properties of an entity from the knowledge graph."""
        cypher_query = """
        MATCH (e:Entity {name: $entity_name})
        RETURN e
        """
        
        result = session.run(cypher_query, entity_name=entity)
        record = result.single()
        
        if record:
            return dict(record['e'])
        return {}
    
    def temporal_reasoning(self, query: str, entities: List[str]) -> List[ReasoningPath]:
        """
        Perform temporal reasoning for time-based queries.
        
        Args:
            query: The user query
            entities: Extracted entities from the query
            
        Returns:
            List of temporal reasoning paths
        """
        reasoning_paths = []
        
        # Extract temporal information from query
        temporal_info = self._extract_temporal_info(query)
        
        with get_neo4j_session() as session:
            for entity in entities:
                temporal_paths = self._find_temporal_paths(session, entity, temporal_info)
                reasoning_paths.extend(temporal_paths)
        
        return reasoning_paths
    
    def _extract_temporal_info(self, query: str) -> Dict[str, Any]:
        """Extract temporal information from the query."""
        temporal_patterns = {
            'before': r'before\s+(\w+)',
            'after': r'after\s+(\w+)',
            'during': r'during\s+(\w+)',
            'when': r'when\s+(\w+)',
            'timeline': r'timeline|history|evolution'
        }
        
        temporal_info = {}
        query_lower = query.lower()
        
        for temporal_type, pattern in temporal_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                temporal_info[temporal_type] = match.group(1) if match.groups() else True
        
        return temporal_info
    
    def _find_temporal_paths(self, session: Session, entity: str, temporal_info: Dict[str, Any]) -> List[ReasoningPath]:
        """Find temporal paths for an entity."""
        paths = []
        
        # Query for temporal relationships
        cypher_query = """
        MATCH (e:Entity {name: $entity_name})-[r:TEMPORAL|BEFORE|AFTER|DURING]->(related:Entity)
        RETURN e, r, related
        ORDER BY r.timestamp DESC
        LIMIT 5
        """
        
        results = session.run(cypher_query, entity_name=entity)
        
        for record in results:
            relationship = record['r']
            related_entity = record['related']['name']
            
            paths.append(ReasoningPath(
                path_type='temporal',
                entities=[entity, related_entity],
                relationships=[relationship.type],
                confidence=0.7,
                evidence=[f"Temporal relationship: {relationship.type}"],
                reasoning_chain=[f"{entity} {relationship.type} {related_entity}"]
            ))
        
        return paths
    
    def hierarchical_reasoning(self, query: str, entities: List[str]) -> List[ReasoningPath]:
        """
        Perform hierarchical reasoning for concept hierarchies.
        
        Args:
            query: The user query
            entities: Extracted entities from the query
            
        Returns:
            List of hierarchical reasoning paths
        """
        reasoning_paths = []
        
        with get_neo4j_session() as session:
            for entity in entities:
                hierarchy_paths = self._find_hierarchy_paths(session, entity)
                reasoning_paths.extend(hierarchy_paths)
        
        return reasoning_paths
    
    def _find_hierarchy_paths(self, session: Session, entity: str) -> List[ReasoningPath]:
        """Find hierarchical relationships for an entity."""
        paths = []
        
        # Query for hierarchical relationships
        cypher_query = """
        MATCH path = (e:Entity {name: $entity_name})-[:IS_A|PART_OF|CONTAINS*1..3]->(related:Entity)
        RETURN path, e, related
        UNION
        MATCH path = (related:Entity)-[:IS_A|PART_OF|CONTAINS*1..3]->(e:Entity {name: $entity_name})
        RETURN path, e, related
        LIMIT 10
        """
        
        results = session.run(cypher_query, entity_name=entity)
        
        for record in results:
            path = record['path']
            related_entity = record['related']['name']
            
            # Determine hierarchy direction
            if path.start_node['name'] == entity:
                direction = "subordinate"
                relationship_type = "IS_A"
            else:
                direction = "superordinate"
                relationship_type = "CONTAINS"
            
            paths.append(ReasoningPath(
                path_type='hierarchical',
                entities=[entity, related_entity],
                relationships=[relationship_type],
                confidence=0.8,
                evidence=[f"Hierarchical relationship: {entity} {direction} to {related_entity}"],
                reasoning_chain=[f"{entity} is {direction} to {related_entity}"]
            ))
        
        return paths
    
    def multi_hop_reasoning(self, query: str, entities: List[str]) -> List[ReasoningPath]:
        """
        Perform multi-hop reasoning across the knowledge graph.
        
        Args:
            query: The user query
            entities: Extracted entities from the query
            
        Returns:
            List of multi-hop reasoning paths
        """
        reasoning_paths = []
        
        with get_neo4j_session() as session:
            for entity in entities:
                # Find multi-hop paths
                multi_hop_paths = self._find_multi_hop_paths(session, entity, max_hops=3)
                reasoning_paths.extend(multi_hop_paths)
        
        return reasoning_paths
    
    def _find_multi_hop_paths(self, session: Session, entity: str, max_hops: int = 3) -> List[ReasoningPath]:
        """Find multi-hop paths for an entity."""
        paths = []
        
        # Query for multi-hop relationships
        cypher_query = f"""
        MATCH path = (e:Entity {{name: $entity_name}})-[*1..{max_hops}]-(related:Entity)
        WHERE related.name <> $entity_name
        RETURN path, e, related
        ORDER BY length(path) ASC
        LIMIT 10
        """
        
        results = session.run(cypher_query, entity_name=entity)
        
        for record in results:
            path = record['path']
            related_entity = record['related']['name']
            
            # Extract path information
            path_entities = [node['name'] for node in path.nodes]
            path_relationships = [rel.type for rel in path.relationships]
            
            # Calculate confidence based on path length
            confidence = max(0.3, 1.0 - (len(path) - 1) * 0.2)
            
            # Generate reasoning chain
            reasoning_chain = []
            for i, rel in enumerate(path.relationships):
                if i < len(path.nodes) - 1:
                    source = path.nodes[i]['name']
                    target = path.nodes[i + 1]['name']
                    reasoning_chain.append(f"{source} {rel.type} {target}")
            
            paths.append(ReasoningPath(
                path_type='multi_hop',
                entities=path_entities,
                relationships=path_relationships,
                confidence=confidence,
                evidence=[f"Multi-hop path with {len(path)} steps"],
                reasoning_chain=reasoning_chain
            ))
        
        return paths
    
    def generate_reasoning_explanation(self, reasoning_paths: List[ReasoningPath]) -> str:
        """
        Generate a human-readable explanation of reasoning paths.
        
        Args:
            reasoning_paths: List of reasoning paths
            
        Returns:
            Human-readable explanation
        """
        if not reasoning_paths:
            return "No reasoning paths found for this query."
        
        explanations = []
        
        for path in reasoning_paths:
            if path.path_type == 'causal':
                explanations.append(f"Causal reasoning: {' → '.join(path.reasoning_chain)}")
            elif path.path_type == 'comparative':
                explanations.append(f"Comparative analysis: {path.reasoning_chain[0]}")
            elif path.path_type == 'temporal':
                explanations.append(f"Temporal reasoning: {' → '.join(path.reasoning_chain)}")
            elif path.path_type == 'hierarchical':
                explanations.append(f"Hierarchical reasoning: {' → '.join(path.reasoning_chain)}")
            elif path.path_type == 'multi_hop':
                explanations.append(f"Multi-hop reasoning: {' → '.join(path.reasoning_chain)}")
        
        return "\n".join(explanations) 