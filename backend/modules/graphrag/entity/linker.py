#!/usr/bin/env python3

import re
import string
from typing import List, Dict, Any, Tuple, Optional
from difflib import SequenceMatcher
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from entity_extractor import Entity
import requests
import logging
import json
import time
from urllib.parse import quote

logger = logging.getLogger(__name__)

@dataclass
class EntityLink:
    """Represents a link between similar entities."""
    source_entity: Entity
    target_entity: Entity
    similarity_score: float
    link_type: str  # "exact", "fuzzy", "semantic", "disambiguation"
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class LinkedEntity:
    """Represents an entity linked to a knowledge base."""
    entity_text: str
    entity_type: str
    confidence: float
    knowledge_base_id: str
    knowledge_base_url: str
    description: Optional[str] = None
    aliases: List[str] = None
    properties: Dict[str, Any] = None

class EntityLinker:
    """Entity linker for knowledge base integration."""
    
    def __init__(self):
        """Initialize the entity linker."""
        self.entity_clusters = defaultdict(list)  # cluster_id -> [entities]
        self.entity_to_cluster = {}  # entity_id -> cluster_id
        self.next_cluster_id = 0
        
        # Load spaCy for better text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("⚠️ spaCy model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize TF-IDF vectorizer for semantic similarity
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        self.entity_vectors = {}
        self.vectorizer_fitted = False
        
        self.wikidata_base_url = "https://www.wikidata.org/w/api.php"
        self.dbpedia_base_url = "https://dbpedia.org/sparql"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GraphRAG-EntityLinker/1.0'
        })
    
    def link_entities(self, entities: List[Entity], existing_entities: List[Entity] = None) -> List[EntityLink]:
        """Link new entities with existing ones and create clusters."""
        if existing_entities is None:
            existing_entities = []
        
        links = []
        
        # Process new entities
        for entity in entities:
            # Find best match among existing entities
            best_match = self._find_best_match(entity, existing_entities)
            
            if best_match:
                # Create link
                link = EntityLink(
                    source_entity=entity,
                    target_entity=best_match,
                    similarity_score=best_match[1],
                    link_type=best_match[2],
                    confidence=best_match[1],
                    metadata={"method": "entity_linking"}
                )
                links.append(link)
                
                # Add to cluster
                self._add_to_cluster(entity, best_match[0])
            else:
                # Create new cluster
                self._create_new_cluster(entity)
        
        return links
    
    def _find_best_match(self, entity: Entity, existing_entities: List[Entity]) -> Optional[Tuple[Entity, float, str]]:
        """Find the best matching entity among existing ones."""
        best_match = None
        best_score = 0.0
        best_type = ""
        
        for existing in existing_entities:
            # Exact match
            if self._exact_match(entity, existing):
                return (existing, 1.0, "exact")
            
            # Fuzzy match
            fuzzy_score = self._fuzzy_similarity(entity.name, existing.name)
            if fuzzy_score > 0.8 and fuzzy_score > best_score:
                best_match = existing
                best_score = fuzzy_score
                best_type = "fuzzy"
            
            # Semantic similarity
            semantic_score = self._semantic_similarity(entity, existing)
            if semantic_score > 0.7 and semantic_score > best_score:
                best_match = existing
                best_score = semantic_score
                best_type = "semantic"
        
        if best_score > 0.6:
            return (best_match, best_score, best_type)
        
        return None
    
    def _exact_match(self, entity1: Entity, entity2: Entity) -> bool:
        """Check for exact match (case-insensitive)."""
        return (entity1.name.lower().strip() == entity2.name.lower().strip() and
                entity1.entity_type == entity2.entity_type)
    
    def _fuzzy_similarity(self, name1: str, name2: str) -> float:
        """Calculate fuzzy string similarity."""
        # Normalize names
        name1 = self._normalize_entity_name(name1)
        name2 = self._normalize_entity_name(name2)
        
        # Use sequence matcher
        similarity = SequenceMatcher(None, name1, name2).ratio()
        
        # Additional checks for abbreviations and variations
        if self._is_abbreviation(name1, name2) or self._is_abbreviation(name2, name1):
            similarity = max(similarity, 0.8)
        
        return similarity
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for comparison."""
        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        name = ' '.join(name.split())
        return name.lower()
    
    def _is_abbreviation(self, short: str, long: str) -> bool:
        """Check if one name is an abbreviation of the other."""
        short = short.replace('.', '').upper()
        long_words = long.upper().split()
        
        if len(short) <= 2:
            return False
        
        # Check if short is initials of long
        if len(short) == len(long_words):
            return all(short[i] == long_words[i][0] for i in range(len(short)))
        
        return False
    
    def _semantic_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate semantic similarity between entities."""
        # Prepare text for vectorization
        text1 = f"{entity1.name} {entity1.entity_type}"
        text2 = f"{entity2.name} {entity2.entity_type}"
        
        # Add context if available
        if entity1.description:
            text1 += f" {entity1.description}"
        if entity2.description:
            text2 += f" {entity2.description}"
        
        # Vectorize
        if not self.vectorizer_fitted:
            self.vectorizer.fit([text1, text2])
            self.vectorizer_fitted = True
        
        vectors = self.vectorizer.transform([text1, text2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        return similarity
    
    def _add_to_cluster(self, entity: Entity, cluster_entity: Entity):
        """Add entity to existing cluster."""
        cluster_id = self.entity_to_cluster.get(cluster_entity.name, None)
        if cluster_id is None:
            cluster_id = self._create_new_cluster(cluster_entity)
        
        self.entity_clusters[cluster_id].append(entity)
        self.entity_to_cluster[entity.name] = cluster_id
    
    def _create_new_cluster(self, entity: Entity) -> int:
        """Create a new cluster for an entity."""
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        
        self.entity_clusters[cluster_id].append(entity)
        self.entity_to_cluster[entity.name] = cluster_id
        
        return cluster_id
    
    def disambiguate_entity(self, entity: Entity, context: str) -> Entity:
        """Disambiguate entity based on context."""
        # Find all entities with similar names
        candidates = []
        for cluster_id, cluster_entities in self.entity_clusters.items():
            for cluster_entity in cluster_entities:
                if self._fuzzy_similarity(entity.name, cluster_entity.name) > 0.6:
                    candidates.append(cluster_entity)
        
        if not candidates:
            return entity
        
        # Score candidates based on context similarity
        best_candidate = entity
        best_score = 0.0
        
        for candidate in candidates:
            # Create context vectors
            context_text = f"{candidate.name} {candidate.entity_type}"
            if candidate.description:
                context_text += f" {candidate.description}"
            
            # Compare with input context
            similarity = self._context_similarity(context_text, context)
            
            if similarity > best_score:
                best_score = similarity
                best_candidate = candidate
        
        return best_candidate
    
    def _context_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text contexts."""
        # Use TF-IDF for context similarity
        if not self.vectorizer_fitted:
            self.vectorizer.fit([text1, text2])
            self.vectorizer_fitted = True
        
        vectors = self.vectorizer.transform([text1, text2])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        return similarity
    
    def get_entity_clusters(self) -> Dict[int, List[Entity]]:
        """Get all entity clusters."""
        return dict(self.entity_clusters)
    
    def get_entity_links(self) -> List[EntityLink]:
        """Get all entity links."""
        links = []
        for cluster_id, entities in self.entity_clusters.items():
            if len(entities) > 1:
                # Create links between all entities in cluster
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        link = EntityLink(
                            source_entity=entities[i],
                            target_entity=entities[j],
                            similarity_score=0.8,  # Default for cluster members
                            link_type="cluster",
                            confidence=0.8,
                            metadata={"cluster_id": cluster_id}
                        )
                        links.append(link)
        
        return links
    
    def merge_clusters(self, cluster_id1: int, cluster_id2: int):
        """Merge two entity clusters."""
        if cluster_id1 not in self.entity_clusters or cluster_id2 not in self.entity_clusters:
            return
        
        # Move all entities from cluster2 to cluster1
        entities_to_move = self.entity_clusters[cluster_id2]
        self.entity_clusters[cluster_id1].extend(entities_to_move)
        
        # Update entity_to_cluster mapping
        for entity in entities_to_move:
            self.entity_to_cluster[entity.name] = cluster_id1
        
        # Remove cluster2
        del self.entity_clusters[cluster_id2]
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get statistics about entity linking."""
        total_entities = sum(len(entities) for entities in self.entity_clusters.values())
        total_clusters = len(self.entity_clusters)
        
        return {
            "total_entities": total_entities,
            "total_clusters": total_clusters,
            "average_cluster_size": total_entities / total_clusters if total_clusters > 0 else 0,
            "largest_cluster_size": max(len(entities) for entities in self.entity_clusters.values()) if self.entity_clusters else 0
        }
    
    def link_entities_to_knowledge_bases(self, entities: List[Dict[str, Any]], text: str = None) -> List[LinkedEntity]:
        """
        Link entities to knowledge bases.
        
        Args:
            entities: List of extracted entities
            text: Original text for context
            
        Returns:
            List of linked entities
        """
        linked_entities = []
        
        for entity in entities:
            entity_text = entity.get('name', entity.get('text', ''))
            entity_type = entity.get('entity_type', '')
            confidence = entity.get('confidence', 0.5)
            
            if not entity_text:
                continue
            
            # Try to link to Wikidata first
            wikidata_link = self._link_to_wikidata(entity_text, entity_type)
            
            if wikidata_link:
                linked_entity = LinkedEntity(
                    entity_text=entity_text,
                    entity_type=entity_type,
                    confidence=confidence,
                    knowledge_base_id=wikidata_link['id'],
                    knowledge_base_url=wikidata_link['url'],
                    description=wikidata_link.get('description'),
                    aliases=wikidata_link.get('aliases', []),
                    properties=wikidata_link.get('properties', {})
                )
                linked_entities.append(linked_entity)
            else:
                # Try DBpedia as fallback
                dbpedia_link = self._link_to_dbpedia(entity_text, entity_type)
                
                if dbpedia_link:
                    linked_entity = LinkedEntity(
                        entity_text=entity_text,
                        entity_type=entity_type,
                        confidence=confidence,
                        knowledge_base_id=dbpedia_link['id'],
                        knowledge_base_url=dbpedia_link['url'],
                        description=dbpedia_link.get('description'),
                        aliases=dbpedia_link.get('aliases', []),
                        properties=dbpedia_link.get('properties', {})
                    )
                    linked_entities.append(linked_entity)
        
        logger.info(f"Linked {len(linked_entities)} entities to knowledge bases")
        return linked_entities
    
    def _link_to_wikidata(self, entity_text: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Link entity to Wikidata."""
        try:
            # Search for entity in Wikidata
            search_params = {
                'action': 'wbsearchentities',
                'search': entity_text,
                'language': 'en',
                'format': 'json',
                'limit': 5
            }
            
            response = self.session.get(self.wikidata_base_url, params=search_params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'search' in data and data['search']:
                # Find the best match
                best_match = self._find_best_wikidata_match(data['search'], entity_text, entity_type)
                
                if best_match:
                    # Get detailed information
                    entity_info = self._get_wikidata_entity_info(best_match['id'])
                    
                    if entity_info:
                        return {
                            'id': best_match['id'],
                            'url': f"https://www.wikidata.org/wiki/{best_match['id']}",
                            'description': entity_info.get('description'),
                            'aliases': entity_info.get('aliases', []),
                            'properties': entity_info.get('claims', {})
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error linking to Wikidata: {e}")
            return None
    
    def _find_best_wikidata_match(self, search_results: List[Dict], entity_text: str, entity_type: str) -> Optional[Dict]:
        """Find the best Wikidata match for an entity."""
        if not search_results:
            return None
        
        # Score each result
        scored_results = []
        for result in search_results:
            score = 0
            
            # Exact match gets high score
            if result['label'].lower() == entity_text.lower():
                score += 10
            elif entity_text.lower() in result['label'].lower():
                score += 5
            
            # Check aliases
            if 'aliases' in result:
                for alias in result['aliases']:
                    if alias.lower() == entity_text.lower():
                        score += 8
                    elif entity_text.lower() in alias.lower():
                        score += 3
            
            # Check description relevance
            if 'description' in result:
                desc_lower = result['description'].lower()
                entity_lower = entity_type.lower()
                if entity_lower in desc_lower:
                    score += 2
            
            scored_results.append((result, score))
        
        # Return the highest scoring result
        if scored_results:
            best_result = max(scored_results, key=lambda x: x[1])
            if best_result[1] > 0:  # Only return if we have some confidence
                return best_result[0]
        
        return None
    
    def _get_wikidata_entity_info(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a Wikidata entity."""
        try:
            params = {
                'action': 'wbgetentities',
                'ids': entity_id,
                'languages': 'en',
                'format': 'json',
                'props': 'labels|descriptions|aliases|claims'
            }
            
            response = self.session.get(self.wikidata_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'entities' in data and entity_id in data['entities']:
                entity = data['entities'][entity_id]
                
                return {
                    'description': entity.get('descriptions', {}).get('en', {}).get('value'),
                    'aliases': [alias['value'] for alias in entity.get('aliases', {}).get('en', [])],
                    'claims': entity.get('claims', {})
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Wikidata entity info: {e}")
            return None
    
    def _link_to_dbpedia(self, entity_text: str, entity_type: str) -> Optional[Dict[str, Any]]:
        """Link entity to DBpedia."""
        try:
            # Search for entity in DBpedia
            search_query = f"""
            SELECT ?entity ?label ?abstract ?type
            WHERE {{
                ?entity rdfs:label ?label .
                FILTER(LANG(?label) = "en")
                FILTER(CONTAINS(LCASE(?label), "{entity_text.lower()}"))
                OPTIONAL {{ ?entity rdfs:type ?type }}
                OPTIONAL {{ ?entity dbo:abstract ?abstract }}
                FILTER(LANG(?abstract) = "en")
            }}
            LIMIT 5
            """
            
            params = {
                'query': search_query,
                'format': 'json'
            }
            
            response = self.session.get(self.dbpedia_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'results' in data and 'bindings' in data['results']:
                bindings = data['results']['bindings']
                
                if bindings:
                    # Find the best match
                    best_match = self._find_best_dbpedia_match(bindings, entity_text, entity_type)
                    
                    if best_match:
                        return {
                            'id': best_match['entity']['value'],
                            'url': best_match['entity']['value'],
                            'description': best_match.get('abstract', {}).get('value'),
                            'aliases': [best_match['label']['value']],
                            'properties': {'type': best_match.get('type', {}).get('value')}
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error linking to DBpedia: {e}")
            return None
    
    def _find_best_dbpedia_match(self, bindings: List[Dict], entity_text: str, entity_type: str) -> Optional[Dict]:
        """Find the best DBpedia match for an entity."""
        if not bindings:
            return None
        
        # Score each result
        scored_results = []
        for binding in bindings:
            score = 0
            label = binding.get('label', {}).get('value', '')
            
            # Exact match gets high score
            if label.lower() == entity_text.lower():
                score += 10
            elif entity_text.lower() in label.lower():
                score += 5
            
            # Check type relevance
            entity_type_lower = entity_type.lower()
            if 'type' in binding:
                type_value = binding['type']['value'].lower()
                if entity_type_lower in type_value:
                    score += 3
            
            scored_results.append((binding, score))
        
        # Return the highest scoring result
        if scored_results:
            best_result = max(scored_results, key=lambda x: x[1])
            if best_result[1] > 0:  # Only return if we have some confidence
                return best_result[0]
        
        return None
    
    def disambiguate_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """
        Disambiguate entities using knowledge base information.
        
        Args:
            entities: List of extracted entities
            text: Original text for context
            
        Returns:
            List of disambiguated entities
        """
        disambiguated_entities = []
        
        for entity in entities:
            entity_text = entity.get('name', entity.get('text', ''))
            
            if not entity_text:
                continue
            
            # Try to link the entity
            linked_entity = self._link_to_wikidata(entity_text, entity.get('entity_type', ''))
            
            if linked_entity:
                # Use knowledge base information to improve entity
                improved_entity = entity.copy()
                improved_entity['knowledge_base_id'] = linked_entity['id']
                improved_entity['knowledge_base_url'] = linked_entity['url']
                improved_entity['description'] = linked_entity.get('description')
                improved_entity['aliases'] = linked_entity.get('aliases', [])
                
                # Improve entity type if available
                if linked_entity.get('properties'):
                    # Extract type information from properties
                    entity_type = self._extract_entity_type_from_properties(linked_entity['properties'])
                    if entity_type:
                        improved_entity['entity_type'] = entity_type
                
                disambiguated_entities.append(improved_entity)
            else:
                disambiguated_entities.append(entity)
        
        return disambiguated_entities
    
    def _extract_entity_type_from_properties(self, properties: Dict[str, Any]) -> Optional[str]:
        """Extract entity type from Wikidata properties."""
        # Common Wikidata property mappings
        type_properties = {
            'P31': 'instance_of',  # instance of
            'P106': 'occupation',   # occupation
            'P27': 'country',       # country of citizenship
            'P17': 'country',       # country
            'P159': 'location',     # headquarters location
        }
        
        for prop_id, prop_type in type_properties.items():
            if prop_id in properties:
                # Extract the type from the property value
                # This is a simplified implementation
                return prop_type
        
        return None
    
    def enrich_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich entities with additional information from knowledge bases.
        
        Args:
            entities: List of entities to enrich
            
        Returns:
            List of enriched entities
        """
        enriched_entities = []
        
        for entity in entities:
            entity_text = entity.get('name', entity.get('text', ''))
            
            if not entity_text:
                enriched_entities.append(entity)
                continue
            
            # Try to get additional information
            wikidata_info = self._link_to_wikidata(entity_text, entity.get('entity_type', ''))
            
            if wikidata_info:
                enriched_entity = entity.copy()
                enriched_entity['enriched'] = True
                enriched_entity['knowledge_base_info'] = wikidata_info
                
                # Add additional properties
                if wikidata_info.get('properties'):
                    enriched_entity['additional_properties'] = self._extract_additional_properties(
                        wikidata_info['properties']
                    )
                
                enriched_entities.append(enriched_entity)
            else:
                enriched_entities.append(entity)
        
        return enriched_entities
    
    def _extract_additional_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Extract additional properties from Wikidata claims."""
        additional_props = {}
        
        # Common properties to extract
        property_mappings = {
            'P569': 'birth_date',
            'P570': 'death_date',
            'P19': 'birth_place',
            'P20': 'death_place',
            'P166': 'awards',
            'P69': 'educated_at',
            'P937': 'work_location',
        }
        
        for prop_id, prop_name in property_mappings.items():
            if prop_id in properties:
                # Extract the property value
                # This is a simplified implementation
                additional_props[prop_name] = f"Property {prop_id}"
        
        return additional_props
    
    def is_available(self) -> bool:
        """Check if the entity linker is available."""
        try:
            # Test connectivity to Wikidata
            response = self.session.get(self.wikidata_base_url, params={'action': 'help'}, timeout=5)
            return response.status_code == 200
        except:
            return False

def get_entity_linker() -> EntityLinker:
    """Get a singleton instance of the entity linker."""
    if not hasattr(get_entity_linker, '_instance'):
        get_entity_linker._instance = EntityLinker()
    return get_entity_linker._instance 