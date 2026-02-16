#!/usr/bin/env python3

import networkx as nx
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import json
from entity_extractor import Entity, Relationship

@dataclass
class ReasoningPath:
    """Represents a reasoning path through the knowledge graph."""
    source: str
    target: str
    path: List[str]
    relationships: List[str]
    confidence: float
    path_length: int
    metadata: Dict[str, Any]

@dataclass
class ReasoningResult:
    """Result of graph-based reasoning."""
    query: str
    paths: List[ReasoningPath]
    inferred_relationships: List[Relationship]
    confidence: float
    reasoning_steps: List[str]

class GraphReasoner:
    """Performs graph-based reasoning on the knowledge graph."""
    
    def __init__(self):
        """Initialize the graph reasoner."""
        self.graph = nx.MultiDiGraph()
        self.entity_cache = {}
        self.relationship_patterns = {
            "transitive": ["part of", "contains", "includes"],
            "symmetric": ["related to", "connected to", "associated with"],
            "inverse": {
                "part of": "contains",
                "contains": "part of",
                "works for": "employs",
                "employs": "works for"
            }
        }
    
    def build_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """Build a NetworkX graph from entities and relationships."""
        # Clear existing graph
        self.graph.clear()
        
        # Add entities as nodes
        for entity in entities:
            self.graph.add_node(entity.name, 
                              entity_type=entity.entity_type,
                              description=entity.description,
                              confidence=entity.confidence)
            self.entity_cache[entity.name] = entity
        
        # Add relationships as edges
        for rel in relationships:
            self.graph.add_edge(
                rel.source, 
                rel.target,
                relation_type=rel.relation_type,
                confidence=rel.confidence,
                context=rel.context,
                weight=rel.confidence
            )
    
    def find_paths(self, source: str, target: str, max_hops: int = 3) -> List[ReasoningPath]:
        """Find all paths between two entities up to max_hops."""
        if source not in self.graph or target not in self.graph:
            return []
        
        paths = []
        
        # Use NetworkX to find all simple paths
        try:
            all_paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=max_hops))
        except nx.NetworkXNoPath:
            return []
        
        for path in all_paths:
            if len(path) <= max_hops + 1:  # +1 because path includes both endpoints
                reasoning_path = self._create_reasoning_path(path)
                if reasoning_path:
                    paths.append(reasoning_path)
        
        return paths
    
    def _create_reasoning_path(self, path: List[str]) -> Optional[ReasoningPath]:
        """Create a reasoning path from a node path."""
        if len(path) < 2:
            return None
        
        relationships = []
        total_confidence = 1.0
        
        # Extract relationships along the path
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            # Get edge data
            edge_data = self.graph.get_edge_data(source, target)
            if edge_data:
                # Get the first edge (in case of multiple edges)
                edge_key = list(edge_data.keys())[0]
                rel_type = edge_data[edge_key].get('relation_type', 'unknown')
                confidence = edge_data[edge_key].get('confidence', 0.5)
                
                relationships.append(rel_type)
                total_confidence *= confidence
        
        return ReasoningPath(
            source=path[0],
            target=path[-1],
            path=path,
            relationships=relationships,
            confidence=total_confidence,
            path_length=len(path) - 1,
            metadata={"path_type": "direct"}
        )
    
    def infer_relationships(self, source: str, target: str, max_hops: int = 2) -> List[Relationship]:
        """Infer relationships between entities using graph reasoning."""
        paths = self.find_paths(source, target, max_hops)
        inferred_rels = []
        
        for path in paths:
            if path.confidence > 0.3:  # Minimum confidence threshold
                inferred_rel = self._infer_relationship_from_path(path)
                if inferred_rel:
                    inferred_rels.append(inferred_rel)
        
        return inferred_rels
    
    def _infer_relationship_from_path(self, path: ReasoningPath) -> Optional[Relationship]:
        """Infer a relationship from a reasoning path."""
        if not path.relationships:
            return None
        
        # Simple inference rules
        if len(path.relationships) == 1:
            # Direct relationship
            rel_type = path.relationships[0]
        elif len(path.relationships) == 2:
            # Two-hop relationship
            rel_type = self._infer_two_hop_relationship(path.relationships)
        else:
            # Multi-hop relationship
            rel_type = self._infer_multi_hop_relationship(path.relationships)
        
        if rel_type:
            return Relationship(
                source=path.source,
                target=path.target,
                relation_type=rel_type,
                context=f"Inferred via path: {' -> '.join(path.path)}",
                confidence=path.confidence * 0.8,  # Reduce confidence for inferred relationships
                metadata={
                    "inference_method": "graph_reasoning",
                    "path_length": path.path_length,
                    "original_path": path.path,
                    "original_relationships": path.relationships
                }
            )
        
        return None
    
    def _infer_two_hop_relationship(self, relationships: List[str]) -> Optional[str]:
        """Infer relationship from two-hop path."""
        if len(relationships) != 2:
            return None
        
        rel1, rel2 = relationships
        
        # Transitive relationships
        if rel1 in self.relationship_patterns["transitive"] and rel2 in self.relationship_patterns["transitive"]:
            return "part of"  # Default transitive relationship
        
        # Inverse relationships
        if rel1 in self.relationship_patterns["inverse"] and self.relationship_patterns["inverse"][rel1] == rel2:
            return "related to"  # Neutral relationship for inverse pairs
        
        return "related to"  # Default for unknown patterns
    
    def _infer_multi_hop_relationship(self, relationships: List[str]) -> Optional[str]:
        """Infer relationship from multi-hop path."""
        # For longer paths, use a more generic relationship
        return "indirectly related to"
    
    def find_related_entities(self, entity: str, max_hops: int = 2) -> Dict[str, List[Tuple[str, float]]]:
        """Find entities related to a given entity within max_hops."""
        if entity not in self.graph:
            return {}
        
        related = defaultdict(list)
        
        # Use BFS to find related entities
        visited = set()
        queue = deque([(entity, 0, 1.0)])  # (node, hops, confidence)
        
        while queue:
            current, hops, confidence = queue.popleft()
            
            if current in visited or hops > max_hops:
                continue
            
            visited.add(current)
            
            # Add to results if not the original entity
            if current != entity:
                related[f"{hops}-hop"].append((current, confidence))
            
            # Explore neighbors
            for neighbor in self.graph.neighbors(current):
                edge_data = self.graph.get_edge_data(current, neighbor)
                if edge_data:
                    edge_key = list(edge_data.keys())[0]
                    edge_confidence = edge_data[edge_key].get('confidence', 0.5)
                    new_confidence = confidence * edge_confidence
                    
                    queue.append((neighbor, hops + 1, new_confidence))
        
        return dict(related)
    
    def explain_relationship(self, source: str, target: str) -> ReasoningResult:
        """Explain the relationship between two entities."""
        paths = self.find_paths(source, target, max_hops=3)
        
        # Sort paths by confidence
        paths.sort(key=lambda x: x.confidence, reverse=True)
        
        # Generate explanation
        explanation = self._generate_explanation(source, target, paths)
        
        # Infer relationships from paths
        inferred_rels = []
        for path in paths[:3]:  # Top 3 paths
            inferred_rel = self._infer_relationship_from_path(path)
            if inferred_rel:
                inferred_rels.append(inferred_rel)
        
        return ReasoningResult(
            query=f"Explain relationship between {source} and {target}",
            paths=paths,
            inferred_relationships=inferred_rels,
            confidence=max([p.confidence for p in paths]) if paths else 0.0,
            reasoning_steps=explanation
        )
    
    def _generate_explanation(self, source: str, target: str, paths: List[ReasoningPath]) -> List[str]:
        """Generate human-readable explanation of relationships."""
        explanations = []
        
        if not paths:
            explanations.append(f"No direct relationship found between {source} and {target}")
            return explanations
        
        # Explain the best path
        best_path = paths[0]
        if best_path.path_length == 1:
            explanations.append(f"{source} is directly {best_path.relationships[0]} {target}")
        else:
            path_explanation = f"{source}"
            for i, rel in enumerate(best_path.relationships):
                path_explanation += f" is {rel} {best_path.path[i+1]}"
            explanations.append(path_explanation)
        
        # Add confidence information
        explanations.append(f"Confidence: {best_path.confidence:.2f}")
        
        # Add alternative paths if available
        if len(paths) > 1:
            explanations.append(f"Found {len(paths)} different paths between these entities")
        
        return explanations
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        if not self.graph.nodes():
            return {"nodes": 0, "edges": 0, "density": 0}
        
        try:
            # Convert to simple graph for statistics
            simple_graph = nx.DiGraph()
            for u, v, data in self.graph.edges(data=True):
                simple_graph.add_edge(u, v, **data)
            
            return {
                "nodes": simple_graph.number_of_nodes(),
                "edges": simple_graph.number_of_edges(),
                "density": nx.density(simple_graph),
                "connected_components": nx.number_strongly_connected_components(simple_graph),
                "average_clustering": nx.average_clustering(simple_graph),
                "average_shortest_path": nx.average_shortest_path_length(simple_graph) if nx.is_strongly_connected(simple_graph) else None
            }
        except Exception as e:
            # Fallback to basic statistics
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": 0,
                "error": str(e)
            }
    
    def find_entity_clusters(self) -> List[List[str]]:
        """Find clusters of related entities."""
        try:
            # Convert to simple graph for clustering
            simple_graph = nx.DiGraph()
            for u, v, data in self.graph.edges(data=True):
                simple_graph.add_edge(u, v, **data)
            
            # Use strongly connected components as clusters
            components = list(nx.strongly_connected_components(simple_graph))
            
            # Also find weakly connected components for broader clusters
            weak_components = list(nx.weakly_connected_components(simple_graph))
            
            # Combine and deduplicate
            all_clusters = []
            seen_nodes = set()
            
            for component in components + weak_components:
                component_list = list(component)
                if len(component_list) > 1:  # Only include clusters with multiple entities
                    # Check if this cluster is already included
                    if not any(node in seen_nodes for node in component_list):
                        all_clusters.append(component_list)
                        seen_nodes.update(component_list)
            
            return all_clusters
        except Exception as e:
            # Fallback to empty clusters
            return []
    
    def get_entity_centrality(self) -> Dict[str, float]:
        """Calculate centrality measures for entities."""
        if not self.graph.nodes():
            return {}
        
        # Calculate different centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        closeness_centrality = nx.closeness_centrality(self.graph)
        
        # Combine centrality measures
        centrality = {}
        for node in self.graph.nodes():
            centrality[node] = {
                "degree": degree_centrality.get(node, 0),
                "betweenness": betweenness_centrality.get(node, 0),
                "closeness": closeness_centrality.get(node, 0),
                "overall": (degree_centrality.get(node, 0) + 
                           betweenness_centrality.get(node, 0) + 
                           closeness_centrality.get(node, 0)) / 3
            }
        
        return centrality 