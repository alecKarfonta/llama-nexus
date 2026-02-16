from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
import numpy as np
from neo4j import GraphDatabase
import json
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

@dataclass
class Community:
    """Represents a community of related entities."""
    id: str
    name: str
    entities: List[str]
    summary: str
    central_entities: List[str]
    metadata: Dict[str, Any]

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any]
    community_id: str | None = None

@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    source: str
    target: str
    relation_type: str
    properties: Dict[str, Any]

class KnowledgeGraphBuilder:
    """Builds and manages knowledge graphs using Neo4j and NetworkX."""
    
    def __init__(self, neo4j_uri: str | None = None, neo4j_user: str | None = None, 
                 neo4j_password: str | None = None, claude_api_key: str | None = None):
        """Initialize the knowledge graph builder."""
        # Neo4j connection
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        
        # Initialize Neo4j driver
        try:
            self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Connected to Neo4j successfully")
        except Exception as e:
            print(f"⚠️  Neo4j connection failed: {e}")
            self.driver = None
        
        # Initialize Claude for community summaries and enrichment
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        if self.claude_api_key:
            try:
                self.claude = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=0.1,
                    anthropic_api_key=self.claude_api_key
                )
                print("✅ Claude initialized successfully")
            except Exception as e:
                print(f"⚠️  Claude initialization failed: {e}")
                self.claude = None
        else:
            print("⚠️  No Claude API key provided, using fallback methods")
            self.claude = None
        
        # In-memory graph for analysis
        self.graph = nx.Graph()
        
    def add_entities_and_relationships(self, entities: List[Any], relationships: List[Dict], domain: str = "general") -> None:
        """Add entities and relationships to the knowledge graph with domain tracking."""
        
        # Add to Neo4j
        if self.driver:
            with self.driver.session() as session:
                # Add entities with occurrence tracking and domain
                for entity in entities:
                    # Handle both Entity objects and dictionaries
                    if isinstance(entity, dict):
                        entity_name = entity["name"]
                        entity_type = entity["type"]
                        entity_description = entity.get("description", "")
                    else:
                        entity_name = entity.name
                        entity_type = entity.entity_type
                        entity_description = entity.description
                    
                    session.run("""
                        MERGE (e:Entity {id: $id})
                        ON CREATE SET e.name = $name, e.type = $type, e.description = $description, 
                                     e.occurrence = 1, e.domain = $domain
                        ON MATCH SET e.occurrence = e.occurrence + 1, e.domain = $domain
                        """, 
                        id=entity_name,
                        name=entity_name,
                        type=entity_type,
                        description=entity_description,
                        domain=domain
                    )
                
                # Add relationships with occurrence tracking and domain
                for rel in relationships:
                    session.run("""
                        MATCH (source:Entity {id: $source})
                        MATCH (target:Entity {id: $target})
                        MERGE (source)-[r:RELATES_TO {type: $type}]->(target)
                        ON CREATE SET r.context = $context, r.weight = 1, r.domain = $domain
                        ON MATCH SET r.weight = r.weight + 1, r.domain = $domain
                        """,
                        source=rel["source"],
                        target=rel["target"],
                        type=rel["relation"],
                        context=rel.get("context", ""),
                        domain=domain
                    )
        
        # Add to NetworkX for analysis with occurrence tracking and domain
        for entity in entities:
            # Handle both Entity objects and dictionaries
            if isinstance(entity, dict):
                entity_name = entity["name"]
                entity_type = entity["type"]
                entity_description = entity.get("description", "")
            else:
                entity_name = entity.name
                entity_type = entity.entity_type
                entity_description = entity.description
            
            if entity_name in self.graph:
                # Increment occurrence count
                current_occurrence = self.graph.nodes[entity_name].get("occurrence", 1)
                self.graph.nodes[entity_name]["occurrence"] = current_occurrence + 1
                # Update domain
                self.graph.nodes[entity_name]["domain"] = domain
            else:
                # Add new node with occurrence count and domain
                self.graph.add_node(entity_name, 
                                  type=entity_type,
                                  description=entity_description,
                                  occurrence=1,
                                  domain=domain)
        
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            rel_type = rel["relation"]
            
            if self.graph.has_edge(source, target):
                # Increment weight for existing edge
                current_weight = self.graph.edges[source, target].get("weight", 1)
                self.graph.edges[source, target]["weight"] = current_weight + 1
                # Update domain
                self.graph.edges[source, target]["domain"] = domain
            else:
                # Add new edge with weight and domain
                self.graph.add_edge(source, target,
                                  type=rel_type,
                                  context=rel.get("context", ""),
                                  weight=1,
                                  domain=domain)
    
    def detect_communities(self, algorithm: str = "leiden") -> List[Community]:
        """Detect communities in the knowledge graph."""
        
        if len(self.graph.nodes()) == 0:
            return []
        
        communities = []
        
        try:
            if algorithm == "leiden":
                # Try Leiden algorithm
                try:
                    import leidenalg as la
                    import igraph as ig
                    
                    # Convert NetworkX to igraph
                    edges = list(self.graph.edges())
                    g_ig = ig.Graph(edges)
                    
                    # Detect communities using Leiden
                    partition = la.find_partition(g_ig, la.ModularityVertexPartition)
                    
                    # Convert back to NetworkX communities
                    for i, community_nodes in enumerate(partition):
                        node_names = [self.graph.nodes[node]["name"] for node in community_nodes]
                        communities.append(self._create_community(f"community_{i}", node_names))
                        
                except ImportError:
                    print("⚠️  leidenalg not available, falling back to Louvain")
                    algorithm = "louvain"
            
            if algorithm == "louvain":
                # Use Louvain algorithm
                try:
                    import community
                    partition = community.best_partition(self.graph)
                    
                    # Group nodes by community
                    community_groups = {}
                    for node, comm_id in partition.items():
                        if comm_id not in community_groups:
                            community_groups[comm_id] = []
                        community_groups[comm_id].append(node)
                    
                    # Create Community objects
                    for comm_id, nodes in community_groups.items():
                        communities.append(self._create_community(f"community_{comm_id}", nodes))
                        
                except ImportError:
                    print("⚠️  python-louvain not available, using simple clustering")
                    algorithm = "simple"
            
            if algorithm == "simple":
                # Simple connected components approach
                components = list(nx.connected_components(self.graph))
                for i, component in enumerate(components):
                    communities.append(self._create_community(f"community_{i}", list(component)))
                    
        except Exception as e:
            print(f"⚠️  Community detection failed: {e}")
            # Fallback: treat each node as its own community
            for i, node in enumerate(self.graph.nodes()):
                communities.append(self._create_community(f"community_{i}", [node]))
        
        return communities
    
    def _create_community(self, community_id: str, entity_names: List[str]) -> Community:
        """Create a Community object with summary."""
        
        # Find central entities (nodes with highest degree)
        central_entities = []
        if entity_names:
            degrees = [(name, self.graph.degree(name)) for name in entity_names if name in self.graph]
            central_entities = [name for name, _ in sorted(degrees, key=lambda x: x[1], reverse=True)[:3]]
        
        # Generate summary using Claude or fallback
        summary = self._generate_community_summary(entity_names, central_entities)
        
        return Community(
            id=community_id,
            name=f"Community {community_id}",
            entities=entity_names,
            summary=summary,
            central_entities=central_entities,
            metadata={"size": len(entity_names)}
        )
    
    def _generate_community_summary(self, entity_names: List[str], central_entities: List[str]) -> str:
        """Generate a summary for a community using Claude or fallback."""
        
        if not entity_names:
            return "Empty community"
        
        if self.claude and len(entity_names) > 1:
            try:
                prompt = ChatPromptTemplate.from_template("""
                Analyze the following group of entities and provide a brief summary of what they represent:
                
                Entities: {entities}
                Central entities: {central_entities}
                
                Provide a 1-2 sentence summary of what this group of entities represents or what domain they belong to.
                """)
                
                response = self.claude.invoke(prompt.format(
                    entities=", ".join(entity_names),
                    central_entities=", ".join(central_entities) if central_entities else "None"
                ))
                
                return response.content.strip()
                
            except Exception as e:
                print(f"⚠️  Claude summary generation failed: {e}")
        
        # Fallback: simple summary
        if len(entity_names) == 1:
            return f"Single entity: {entity_names[0]}"
        else:
            return f"Community with {len(entity_names)} entities, central: {', '.join(central_entities[:2])}"
    
    def create_hierarchical_structure(self, communities: List[Community]) -> Dict[str, Any]:
        """Create a hierarchical structure from communities."""
        # TODO: Implement hierarchical clustering
        return {
            "root": {
                "communities": [comm.id for comm in communities],
                "total_entities": sum(len(comm.entities) for comm in communities)
            }
        }
    
    def enrich_graph(self, enrichment_type: str = "inferred_relationships") -> List[Dict]:
        """Enrich the knowledge graph with additional information."""
        
        if enrichment_type == "inferred_relationships":
            return self._infer_relationships()
        elif enrichment_type == "entity_properties":
            return self._enrich_entity_properties()
        else:
            return []
    
    def _infer_relationships(self) -> List[Dict]:
        """Infer additional relationships based on graph structure."""
        inferred_relationships = []
        
        # Find entities that are 2 hops apart but not directly connected
        for node1 in self.graph.nodes():
            for node2 in self.graph.nodes():
                if node1 != node2:
                    # Check if they're 2 hops apart
                    try:
                        paths = list(nx.all_simple_paths(self.graph, node1, node2, cutoff=2))
                        if len(paths) > 0 and not self.graph.has_edge(node1, node2):
                            # They're connected via 2 hops but not directly
                            inferred_relationships.append({
                                "source": node1,
                                "target": node2,
                                "relation": "INFERRED_RELATED",
                                "context": f"Connected via {len(paths)} paths",
                                "confidence": 0.5
                            })
                    except nx.NetworkXNoPath:
                        continue
        
        return inferred_relationships
    
    def _enrich_entity_properties(self) -> List[Dict]:
        """Enrich entity properties using Claude."""
        enriched_properties = []
        
        if not self.claude:
            return enriched_properties
        
        # Get entities with descriptions
        entities_with_descriptions = [
            (node, data.get("description", ""))
            for node, data in self.graph.nodes(data=True)
            if data.get("description")
        ]
        
        for entity_name, description in entities_with_descriptions[:5]:  # Limit to avoid API costs
            try:
                prompt = ChatPromptTemplate.from_template("""
                Analyze the following entity and suggest additional properties that could be extracted:
                
                Entity: {entity_name}
                Description: {description}
                
                Return as JSON:
                {{
                    "entity": "{entity_name}",
                    "suggested_properties": {{
                        "category": "string",
                        "complexity": "low|medium|high",
                        "importance": "low|medium|high"
                    }}
                }}
                """)
                
                response = self.claude.invoke(prompt.format(
                    entity_name=entity_name,
                    description=description
                ))
                
                result = json.loads(response.content)
                enriched_properties.append(result)
                
            except Exception as e:
                print(f"⚠️  Entity enrichment failed for {entity_name}: {e}")
        
        return enriched_properties
    
    def get_graph_statistics(self, domain: str | None = None) -> Dict[str, Any]:
        """Get statistics about the knowledge graph, optionally filtered by domain."""
        if len(self.graph.nodes()) == 0:
            return {"nodes": 0, "edges": 0, "communities": 0, "domain": domain}
        
        # Filter by domain if specified
        if domain:
            filtered_nodes = [n for n, data in self.graph.nodes(data=True) 
                            if data.get("domain") == domain]
            filtered_edges = [(u, v) for u, v, data in self.graph.edges(data=True) 
                             if data.get("domain") == domain]
            
            # Create subgraph for statistics
            subgraph = self.graph.subgraph(filtered_nodes)
            
            return {
                "nodes": subgraph.number_of_nodes(),
                "edges": subgraph.number_of_edges(),
                "density": nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0,
                "average_clustering": nx.average_clustering(subgraph) if subgraph.number_of_nodes() > 2 else 0,
                "connected_components": nx.number_connected_components(subgraph),
                "average_shortest_path": nx.average_shortest_path_length(subgraph) if nx.is_connected(subgraph) else None,
                "domain": domain,
                "filtered": True
            }
        else:
            return {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "density": nx.density(self.graph),
                "average_clustering": nx.average_clustering(self.graph),
                "connected_components": nx.number_connected_components(self.graph),
                "average_shortest_path": nx.average_shortest_path_length(self.graph) if nx.is_connected(self.graph) else None,
                "domain": "all",
                "filtered": False
            }
    
    def export_graph(self, format: str = "json") -> str:
        """Export the knowledge graph in various formats."""
        if format == "json":
            return json.dumps({
                "nodes": [{"id": node, **data} for node, data in self.graph.nodes(data=True)],
                "edges": [{"source": u, "target": v, **data} for u, v, data in self.graph.edges(data=True)]
            }, indent=2)
        elif format == "gexf":
            nx.write_gexf(self.graph, "knowledge_graph.gexf")
            return "knowledge_graph.gexf"
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_graph_json(self, domain: str | None = None) -> Dict[str, Any]:
        """Export the knowledge graph as JSON dictionary, optionally filtered by domain."""
        if domain:
            # Filter nodes and edges by domain
            filtered_nodes = [(node, data) for node, data in self.graph.nodes(data=True) 
                             if data.get("domain") == domain]
            filtered_edges = [(u, v, data) for u, v, data in self.graph.edges(data=True) 
                             if data.get("domain") == domain]
            
            return {
                "nodes": [{"id": node, "label": node, "type": data.get("type", "UNKNOWN"), 
                          "occurrence": data.get("occurrence", 1), "domain": data.get("domain", domain), **data} 
                         for node, data in filtered_nodes],
                "edges": [{"source": u, "target": v, "type": data.get("type", "RELATES_TO"), 
                          "weight": data.get("weight", 1), "domain": data.get("domain", domain), **data} 
                         for u, v, data in filtered_edges],
                "domain": domain,
                "filtered": True
            }
        else:
            return {
                "nodes": [{"id": node, "label": node, "type": data.get("type", "UNKNOWN"), 
                          "occurrence": data.get("occurrence", 1), "domain": data.get("domain", "general"), **data} 
                         for node, data in self.graph.nodes(data=True)],
                "edges": [{"source": u, "target": v, "type": data.get("type", "RELATES_TO"), 
                          "weight": data.get("weight", 1), "domain": data.get("domain", "general"), **data} 
                         for u, v, data in self.graph.edges(data=True)],
                "domain": "all",
                "filtered": False
            }
    
    def get_graph_stats(self, domain: str | None = None) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return self.get_graph_statistics(domain)
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domains in the knowledge graph."""
        domains = set()
        
        # Get domains from nodes
        for node, data in self.graph.nodes(data=True):
            if "domain" in data:
                domains.add(data["domain"])
        
        # Get domains from edges
        for u, v, data in self.graph.edges(data=True):
            if "domain" in data:
                domains.add(data["domain"])
        
        return sorted(list(domains))
    
    def get_filtered_graph_data(self, domain: str | None = None, max_entities: int = 100, 
                               max_relationships: int = 200, min_occurrence: int = 1, 
                               min_confidence: float = 0.0, entity_types: List[str] | None = None,
                               relationship_types: List[str] | None = None, sort_by: str = "occurrence",
                               sort_order: str = "desc") -> Dict[str, Any]:
        """
        Get filtered knowledge graph data with support for various filtering options.
        """
        if not self.driver:
            raise RuntimeError("Neo4j driver not available. Check Neo4j connection configuration.")
        
        with self.driver.session() as session:
            # Base query to get all nodes and relationships
            base_query = """
            MATCH (n:Entity)
            OPTIONAL MATCH (n)-[r:RELATES_TO]->(m:Entity)
            RETURN n, r, m
            """
            
            # Initial data fetch
            results = session.run(base_query)
            
            # Process nodes and relationships from results
            all_nodes = {}
            all_relationships = []
            
            for record in results:
                node_n = record.get('n')
                if node_n:
                    node_id = node_n.get('id', node_n.element_id)
                    if node_id not in all_nodes:
                        all_nodes[node_id] = {
                            "id": node_id,
                            "label": node_n.get('name', 'Unknown'),
                            "type": list(node_n.labels)[0] if node_n.labels else "Unknown",
                            "properties": dict(node_n),
                            "occurrence": node_n.get('occurrence', 1)
                        }
                
                rel = record.get('r')
                if rel:
                    source_node = rel.start_node
                    target_node = rel.end_node
                    source_id = source_node.get('id', source_node.element_id)
                    target_id = target_node.get('id', target_node.element_id)
                    all_relationships.append({
                        "id": rel.element_id,
                        "source": source_id,
                        "target": target_id,
                        "label": rel.get('type', rel.type),
                        "type": rel.get('type', rel.type),
                        "weight": rel.get('weight', 1),
                        "properties": dict(rel)
                    })

            # Convert nodes dict to list
            nodes = list(all_nodes.values())
            relationships = all_relationships

            total_entities_before_filter = len(nodes)
            total_relationships_before_filter = len(relationships)

            # Apply filters
            if domain:
                nodes = [n for n in nodes if n['properties'].get('domain') == domain]
                # Filter relationships based on nodes in the domain
                node_ids_in_domain = {n['id'] for n in nodes}
                relationships = [r for r in relationships if r['source'] in node_ids_in_domain and r['target'] in node_ids_in_domain]

            if entity_types:
                nodes = [n for n in nodes if n['type'] in entity_types]
                node_ids_after_type_filter = {n['id'] for n in nodes}
                relationships = [r for r in relationships if r['source'] in node_ids_after_type_filter and r['target'] in node_ids_after_type_filter]

            if relationship_types:
                relationships = [r for r in relationships if r['type'] in relationship_types]

            nodes = [n for n in nodes if n.get('occurrence', 1) >= min_occurrence]
            
            # Note: confidence filter might require a 'confidence' property on relationships
            relationships = [r for r in relationships if r['properties'].get('confidence', 1.0) >= min_confidence]

            # Sorting logic
            if sort_by == 'occurrence':
                nodes = sorted(nodes, key=lambda x: x.get('occurrence', 1), reverse=(sort_order == 'desc'))
            # Add other sorting options as needed

            # Apply limits
            nodes = nodes[:max_entities]
            relationships = relationships[:max_relationships]
            
            # Ensure relationships only connect nodes that are in the final node list
            final_node_ids = {n['id'] for n in nodes}
            relationships = [r for r in relationships if r['source'] in final_node_ids and r['target'] in final_node_ids]

            return {
                "nodes": nodes,
                "edges": relationships,
                "stats": {
                    "nodes": len(nodes),
                    "edges": len(relationships),
                    "total_entities_before_filter": total_entities_before_filter,
                    "total_relationships_before_filter": total_relationships_before_filter,
                },
                "filters_applied": {
                    "domain": domain,
                    "max_entities": max_entities,
                    "max_relationships": max_relationships,
                    "min_occurrence": min_occurrence,
                    "min_confidence": min_confidence,
                    "entity_types": entity_types,
                    "relationship_types": relationship_types,
                    "sort_by": sort_by,
                    "sort_order": sort_order,
                }
            }
    
    def get_top_entities(self, domain: str | None = None, limit: int = 50, 
                        min_occurrence: int = 1, entity_type: str | None = None) -> List[Dict[str, Any]]:
        """Get top entities by occurrence count."""
        
        # Get all nodes
        all_nodes = list(self.graph.nodes(data=True))
        
        # Filter by domain
        if domain:
            all_nodes = [(node, data) for node, data in all_nodes if data.get("domain") == domain]
        
        # Filter by occurrence threshold
        all_nodes = [(node, data) for node, data in all_nodes 
                     if data.get("occurrence", 1) >= min_occurrence]
        
        # Filter by entity type
        if entity_type:
            all_nodes = [(node, data) for node, data in all_nodes 
                        if data.get("type") == entity_type]
        
        # Sort by occurrence (descending)
        all_nodes.sort(key=lambda x: x[1].get("occurrence", 1), reverse=True)
        
        # Apply limit
        top_nodes = all_nodes[:limit]
        
        # Convert to list format
        top_entities = []
        for node, data in top_nodes:
            top_entities.append({
                "name": node,
                "type": data.get("type", "UNKNOWN"),
                "occurrence": data.get("occurrence", 1),
                "confidence": data.get("confidence", 1.0),
                "domain": data.get("domain", "general"),
                "description": data.get("description", "")
            })
        
        return top_entities
    
    def get_top_relationships(self, domain: str | None = None, limit: int = 50, 
                             min_weight: int = 1, relationship_type: str | None = None) -> List[Dict[str, Any]]:
        """Get top relationships by weight/occurrence count."""
        
        # Get all edges
        all_edges = list(self.graph.edges(data=True))
        
        # Filter by domain
        if domain:
            all_edges = [(u, v, data) for u, v, data in all_edges if data.get("domain") == domain]
        
        # Filter by weight threshold
        all_edges = [(u, v, data) for u, v, data in all_edges 
                     if data.get("weight", 1) >= min_weight]
        
        # Filter by relationship type
        if relationship_type:
            all_edges = [(u, v, data) for u, v, data in all_edges 
                        if data.get("type") == relationship_type]
        
        # Sort by weight (descending)
        all_edges.sort(key=lambda x: x[2].get("weight", 1), reverse=True)
        
        # Apply limit
        top_edges = all_edges[:limit]
        
        # Convert to list format
        top_relationships = []
        for u, v, data in top_edges:
            top_relationships.append({
                "source": u,
                "target": v,
                "type": data.get("type", "RELATES_TO"),
                "weight": data.get("weight", 1),
                "domain": data.get("domain", "general"),
                "context": data.get("context", "")
            })
        
        return top_relationships
    
    def get_domain_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each domain in the knowledge graph."""
        domain_stats = {}
        available_domains = self.get_available_domains()
        
        for domain in available_domains:
            domain_stats[domain] = self.get_graph_statistics(domain)
        
        return domain_stats
    
    def build_graph(self, entities, relationships, domain: str = "general"):
        """Build the knowledge graph from entities and relationships."""
        # Convert Entity and Relationship objects to dictionaries
        entity_dicts = []
        for entity in entities:
            entity_dicts.append({
                "name": entity.name,
                "type": entity.entity_type,
                "description": entity.description or "",
                "source_file": getattr(entity, 'source_chunk', None)
            })
        
        relationship_dicts = []
        for rel in relationships:
            relationship_dicts.append({
                "source": rel.source,
                "target": rel.target,
                "relation": rel.relation_type,
                "context": rel.context or ""
            })
        
        self.add_entities_and_relationships(entity_dicts, relationship_dicts, domain)
    
    def clear_graph(self):
        """Clear the knowledge graph."""
        self.clear_knowledge_graph()
    
    def remove_document_entities(self, document_name: str) -> bool:
        """Remove entities for a specific document."""
        try:
            removed_count = self.remove_document_from_knowledge_graph(document_name)
            return removed_count > 0
        except Exception:
            return False
    
    def list_documents(self) -> List[str]:
        """List document names in the knowledge graph."""
        try:
            documents = self.list_documents_in_knowledge_graph()
            return [doc["name"] for doc in documents]
        except Exception:
            return []
    
    def close(self):
        """Close the Neo4j connection."""
        if self.driver:
            self.driver.close()
    
    def clear_knowledge_graph(self):
        """Clear all data from the knowledge graph."""
        try:
            # Clear Neo4j database
            if self.driver:
                with self.driver.session() as session:
                    # Delete all nodes and relationships
                    session.run("MATCH (n) DETACH DELETE n")
                    print("Cleared all data from Neo4j knowledge graph")
            
            # Clear NetworkX graph
            self.graph.clear()
            print("Cleared in-memory knowledge graph")
            
        except Exception as e:
            print(f"Error clearing knowledge graph: {e}")
            raise
    
    def remove_document_from_knowledge_graph(self, document_name: str) -> int:
        """Remove all entities and relationships for a specific document from the knowledge graph."""
        try:
            removed_count = 0
            
            if self.driver:
                with self.driver.session() as session:
                    # Find and delete entities from this document
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE e.source_file = $document_name
                        RETURN e.name as name
                    """, document_name=document_name)
                    
                    entity_names = [record["name"] for record in result]
                    
                    if entity_names:
                        # Delete relationships involving these entities
                        session.run("""
                            MATCH (e:Entity)
                            WHERE e.name IN $entity_names
                            OPTIONAL MATCH (e)-[r]-()
                            DELETE r
                        """, entity_names=entity_names)
                        
                        # Delete the entities
                        session.run("""
                            MATCH (e:Entity)
                            WHERE e.name IN $entity_names
                            DELETE e
                        """, entity_names=entity_names)
                        
                        removed_count = len(entity_names)
                        print(f"Removed {removed_count} entities for document: {document_name}")
            
            # Update NetworkX graph
            nodes_to_remove = []
            for node in self.graph.nodes():
                if self.graph.nodes[node].get("source_file") == document_name:
                    nodes_to_remove.append(node)
            
            for node in nodes_to_remove:
                self.graph.remove_node(node)
            
            return removed_count
            
        except Exception as e:
            print(f"Error removing document from knowledge graph: {e}")
            return 0
    
    def list_documents_in_knowledge_graph(self) -> List[Dict[str, Any]]:
        """List all documents in the knowledge graph."""
        try:
            documents = {}
            
            if self.driver:
                with self.driver.session() as session:
                    # Get all entities grouped by source file
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE e.source_file IS NOT NULL
                        RETURN e.source_file as source_file, count(e) as entity_count
                    """)
                    
                    for record in result:
                        source_file = record["source_file"]
                        entity_count = record["entity_count"]
                        
                        documents[source_file] = {
                            "name": source_file,
                            "entities": entity_count,
                            "relationships": 0  # Would need separate query for relationship count
                        }
            
            return list(documents.values())
            
        except Exception as e:
            print(f"Error listing documents in knowledge graph: {e}")
            return []
    
    def get_direct_relationships(self, entity_name: str) -> list:
        """Return a list of (target, relation_type) for all direct relationships from the given entity."""
        if entity_name not in self.graph:
            return []
        relationships = []
        for neighbor in self.graph.neighbors(entity_name):
            rel_type = self.graph.edges[entity_name, neighbor].get("type", "RELATED")
            relationships.append((neighbor, rel_type))
        return relationships 