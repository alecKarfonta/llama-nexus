from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import re
import os
from rel_extractor import get_relationship_extractor

@dataclass
class Entity:
    """Represents an extracted entity."""
    name: str
    entity_type: str
    description: str | None = None
    confidence: float = 1.0
    source_chunk: str | None = None
    metadata: Dict[str, Any] | None = None

@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source: str
    target: str
    relation_type: str
    context: str | None = None
    confidence: float = 1.0
    metadata: Dict[str, Any] | None = None

@dataclass
class ExtractionResult:
    """Result of entity and relationship extraction."""
    entities: List[Entity]
    relationships: List[Relationship]
    claims: List[str]
    source_chunk: str

class EntityExtractor:
    """GLiNER-based entity and relationship extractor with LLM fallback."""
    
    def __init__(self, model_name: str = "claude-3-sonnet-20240229", api_key: str | None = None):
        """Initialize the entity extractor with GLiNER only."""
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # No LLM fallback - only use GLiNER for entity/relationship extraction
        self.llm = None
        print("ℹ️ Using GLiNER only for entity/relationship extraction")
        
        # Define entity types for different domains
        self.entity_types = {
            "general": ["PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "PROCESS"],
            "technical": ["COMPONENT", "SPECIFICATION", "PROCEDURE", "SYSTEM", "INTERFACE"],
            "automotive": ["COMPONENT", "SPECIFICATION", "PROCEDURE", "MAINTENANCE_ITEM", "SYMPTOM", "SOLUTION"],
            "medical": ["SYMPTOM", "DIAGNOSIS", "TREATMENT", "MEDICATION", "PROCEDURE"],
            "legal": ["LAW", "REGULATION", "CASE", "PRECEDENT", "JURISDICTION"]
        }
        
        # Define relationship types
        self.relationship_types = {
            "general": ["RELATES_TO", "PART_OF", "CONTAINS", "CAUSES", "REQUIRES"],
            "technical": ["CONNECTS_TO", "DEPENDS_ON", "IMPLEMENTS", "CONFIGURES", "MONITORS"],
            "automotive": ["PART_OF", "CONNECTS_TO", "REQUIRES", "CAUSES", "FIXES", "SCHEDULED_AT"],
            "medical": ["TREATS", "CAUSES", "SYMPTOM_OF", "PRESCRIBED_FOR", "INTERACTS_WITH"],
            "legal": ["AMENDS", "CITES", "OVERRULES", "APPLIES_TO", "DEFINES"]
        }
    
    def extract_entities_and_relations(self, text_chunk: str, domain: str = "general") -> ExtractionResult:
        """Extract entities and relationships from text using GLiNER only."""
        # First, try to use GLiNER for both entity and relationship extraction
        rel_extractor = get_relationship_extractor()
        try:
            print(f"🔍 Using GLiNER for entity extraction...")
            # Use GLiNER to extract entities
            gliner_entities = rel_extractor.extract_entities(
                text=text_chunk,
                labels=["person", "organisation", "location", "date", "component", "system", "symptom", "solution", "maintenance", "specification", "requirement", "safety", "time", "founder", "position"],
                threshold=0.5
            )
            
            if gliner_entities.get("entities"):
                print(f"✅ GLiNER found {len(gliner_entities['entities'])} entities")
                entities = []
                for gliner_entity in gliner_entities["entities"]:
                    entity = Entity(
                        name=gliner_entity["text"],
                        entity_type=gliner_entity["label"].upper(),
                        description=f"Extracted by GLiNER model (confidence: {gliner_entity['score']:.3f})",
                        confidence=gliner_entity["score"],
                        source_chunk=text_chunk,
                        metadata={
                            "domain": domain,
                            "ner_source": "gliner_model",
                            "original_ner_type": gliner_entity["label"]
                        }
                    )
                    entities.append(entity)
                
                # Now extract relationships with GLiNER
                print(f"🔗 Using GLiNER for relationship extraction...")
                default_relations = [
                    # Business/Organizational relationships
                    {"relation": "works for", "pairs_filter": [("person", "organisation")]},
                    {"relation": "founded", "pairs_filter": [("person", "organisation")]},
                    {"relation": "located in", "pairs_filter": [("organisation", "location"), ("person", "location")]},
                    {"relation": "acquired", "pairs_filter": [("organisation", "organisation")]},
                    {"relation": "subsidiary of", "pairs_filter": [("organisation", "organisation")]},
                    {"relation": "produces", "pairs_filter": [("organisation", "product")]},
                    {"relation": "part of", "pairs_filter": [("organisation", "organisation")]},
                    
                    # Technical/Component relationships
                    {"relation": "part of", "pairs_filter": [("component", "component"), ("system", "component")]},
                    {"relation": "connects to", "pairs_filter": [("component", "component")]},
                    {"relation": "contains", "pairs_filter": [("component", "component"), ("system", "component")]},
                    {"relation": "controls", "pairs_filter": [("component", "component")]},
                    {"relation": "supplies", "pairs_filter": [("component", "component")]},
                    {"relation": "replaces", "pairs_filter": [("component", "component")]},
                    {"relation": "maintains", "pairs_filter": [("procedure", "component"), ("maintenance", "component")]},
                    {"relation": "requires", "pairs_filter": [("component", "component"), ("procedure", "component")]},
                    {"relation": "causes", "pairs_filter": [("component", "symptom"), ("condition", "symptom")]},
                    {"relation": "fixes", "pairs_filter": [("solution", "symptom"), ("procedure", "symptom")]},
                    
                    # Temporal relationships
                    {"relation": "scheduled for", "pairs_filter": [("maintenance", "time"), ("procedure", "time")]},
                    {"relation": "created on", "pairs_filter": [("organisation", "date"), ("product", "date")]},
                    
                    # Specification relationships
                    {"relation": "specifies", "pairs_filter": [("specification", "component"), ("requirement", "component")]},
                    {"relation": "applies to", "pairs_filter": [("specification", "component"), ("requirement", "component")]},
                    
                    # General relationships
                    {"relation": "related to", "pairs_filter": [("component", "component"), ("system", "component"), ("organisation", "organisation"), ("person", "organisation")]},
                    {"relation": "associated with", "pairs_filter": [("component", "component"), ("system", "component"), ("organisation", "organisation"), ("person", "organisation")]},
                    {"relation": "depends on", "pairs_filter": [("component", "component"), ("system", "component"), ("procedure", "component")]},
                    {"relation": "affects", "pairs_filter": [("component", "component"), ("condition", "symptom"), ("procedure", "component")]},
                    {"relation": "influences", "pairs_filter": [("component", "component"), ("condition", "symptom"), ("procedure", "component")]},
                    {"relation": "supports", "pairs_filter": [("component", "component"), ("system", "component")]},
                    {"relation": "enables", "pairs_filter": [("component", "component"), ("system", "component")]},
                    {"relation": "prevents", "pairs_filter": [("component", "component"), ("safety", "component")]},
                    {"relation": "protects", "pairs_filter": [("component", "component"), ("safety", "component")]},
                    {"relation": "monitors", "pairs_filter": [("component", "component"), ("system", "component")]},
                    {"relation": "regulates", "pairs_filter": [("component", "component"), ("system", "component")]},
                    {"relation": "operates", "pairs_filter": [("component", "component"), ("system", "component")]},
                    {"relation": "manages", "pairs_filter": [("component", "component"), ("system", "component"), ("person", "organisation")]},
                    {"relation": "directs", "pairs_filter": [("person", "organisation"), ("organisation", "organisation")]},
                    {"relation": "leads", "pairs_filter": [("person", "organisation"), ("organisation", "organisation")]},
                    {"relation": "owns", "pairs_filter": [("person", "organisation"), ("organisation", "organisation")]},
                    {"relation": "invests in", "pairs_filter": [("person", "organisation"), ("organisation", "organisation")]},
                    {"relation": "collaborates with", "pairs_filter": [("organisation", "organisation"), ("person", "organisation")]},
                    {"relation": "competes with", "pairs_filter": [("organisation", "organisation")]},
                    {"relation": "partners with", "pairs_filter": [("organisation", "organisation")]},
                ]
                entity_labels = list({e.entity_type.lower() for e in entities})
                gliner_result = rel_extractor.extract_relations(
                    text=text_chunk,
                    relations=default_relations,
                    entity_labels=entity_labels,
                    threshold=0.5
                )
                relationships = []
                for rel in gliner_result.get("relations", []):
                    relationships.append(Relationship(
                        source=rel.get("source") or rel.get("text", ""),
                        target=rel.get("target", ""),
                        relation_type=rel.get("label", rel.get("relation", "")),
                        context=rel.get("context", None),
                        confidence=rel.get("score", 1.0),
                        metadata={"extraction_method": "gliner"}
                    ))
                
                # Apply validation and filtering
                entities = self.validate_entities(entities)
                relationships = self.validate_relationships(relationships, entities)
                
                print(f"✅ After validation: {len(entities)} entities, {len(relationships)} relationships")
                
                return ExtractionResult(
                    entities=entities,
                    relationships=relationships,
                    claims=[],
                    source_chunk=text_chunk
                )
            else:
                print(f"⚠️ GLiNER found no entities, returning empty result")
                return ExtractionResult(
                    entities=[],
                    relationships=[],
                    claims=[],
                    source_chunk=text_chunk
                )
        except Exception as e:
            print(f"⚠️ GLiNER extraction failed: {e}, returning empty result")
            return ExtractionResult(
                entities=[],
                relationships=[],
                claims=[],
                source_chunk=text_chunk
            )
    
    def extract_from_chunks(self, chunks: List[str], domain: str = "general") -> List[ExtractionResult]:
        """Extract entities and relationships from multiple text chunks."""
        results = []
        
        for chunk in chunks:
            result = self.extract_entities_and_relations(chunk, domain)
            results.append(result)
        
        return results
    
    def extract_with_context(self, text_chunk: str, context: str, domain: str = "general") -> ExtractionResult:
        """Extract entities with additional context."""
        enhanced_text = f"Context: {context}\n\nText to analyze: {text_chunk}"
        return self.extract_entities_and_relations(enhanced_text, domain)
    
    def validate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Validate and clean extracted entities with proper deduplication."""
        validated_entities = []
        seen_entities = set()  # Track normalized entity names
        
        for entity in entities:
            # Basic validation
            if not entity.name or len(entity.name.strip()) < 2:
                continue
            
            # Clean entity name
            entity.name = entity.name.strip()
            
            # Normalize for deduplication (lowercase, remove extra spaces)
            normalized_name = " ".join(entity.name.lower().split())
            
            # Skip if we've seen this entity before (case-insensitive)
            if normalized_name in seen_entities:
                continue
            
            # Additional quality checks
            if self._is_low_quality_entity(entity):
                continue
            
            seen_entities.add(normalized_name)
            validated_entities.append(entity)
        
        return validated_entities
    
    def _is_low_quality_entity(self, entity: Entity) -> bool:
        """Check if an entity is low quality and should be filtered out."""
        # Filter out very short entities (likely noise)
        if len(entity.name) < 2:
            return True
        
        # Filter out common stop words or low-quality patterns
        low_quality_patterns = [
            r'^\d+$',  # Just numbers
            r'^[a-z]$',  # Single letter
            r'^(the|a|an|and|or|but|in|on|at|to|for|of|with|by)$',  # Common stop words
        ]
        
        for pattern in low_quality_patterns:
            if re.match(pattern, entity.name.lower()):
                return True
        
        return False
    
    def validate_relationships(self, relationships: List[Relationship], entities: List[Entity]) -> List[Relationship]:
        """Validate relationships and ensure they reference valid entities."""
        validated_relationships = []
        entity_names = {e.name.lower() for e in entities}  # Case-insensitive lookup
        
        for relationship in relationships:
            # Basic validation
            if not relationship.source or not relationship.target:
                continue
            
            # Check if both entities exist in our entity list (case-insensitive)
            source_exists = any(relationship.source.lower() == entity_name for entity_name in entity_names)
            target_exists = any(relationship.target.lower() == entity_name for entity_name in entity_names)
            
            if not source_exists or not target_exists:
                continue
            
            # Additional quality checks
            if self._is_low_quality_relationship(relationship):
                continue
            
            validated_relationships.append(relationship)
        
        return validated_relationships
    
    def _is_low_quality_relationship(self, relationship: Relationship) -> bool:
        """Check if a relationship is low quality and should be filtered out."""
        # Filter out self-referential relationships
        if relationship.source.lower() == relationship.target.lower():
            return True
        
        # Filter out relationships with very short entities
        if len(relationship.source) < 2 or len(relationship.target) < 2:
            return True
        
        # Filter out relationships with generic relation types
        generic_relations = {"related", "associated", "connected", "linked"}
        if relationship.relation_type.lower() in generic_relations:
            return True
        
        return False
    
    def _parse_claude_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse Claude's response and extract JSON data."""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
            
            # If no JSON found, try to extract arrays
            entities = self._extract_entities_array(raw_response)
            relationships = self._extract_relationships_array(raw_response)
            claims = self._extract_claims_array(raw_response)
            
            return {
                "entities": entities,
                "relationships": relationships,
                "claims": claims
            }
            
        except Exception as e:
            print(f"Error parsing Claude response: {e}")
            return {"entities": [], "relationships": [], "claims": []}
    
    def _extract_entities_array(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using regex patterns."""
        entities = []
        
        # Look for entity patterns in the text
        entity_patterns = [
            r'"([^"]+)"\s*:\s*"([^"]+)"',  # "name": "type"
            r'name["\s]*:["\s]*"([^"]+)"',  # name: "entity"
            r'type["\s]*:["\s]*"([^"]+)"',  # type: "PERSON"
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        entities.append({
                            "name": match[0],
                            "type": match[1],
                            "description": "",
                            "confidence": 0.8
                        })
                else:
                    entities.append({
                        "name": match,
                        "type": "ENTITY",
                        "description": "",
                        "confidence": 0.8
                    })
        
        return entities
    
    def _extract_relationships_array(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships from text using regex patterns."""
        relationships = []
        
        # Look for relationship patterns
        rel_patterns = [
            r'source["\s]*:["\s]*"([^"]+)".*?target["\s]*:["\s]*"([^"]+)".*?relation["\s]*:["\s]*"([^"]+)"',
            r'"([^"]+)"\s*->\s*"([^"]+)"\s*:\s*"([^"]+)"',  # "A" -> "B": "relation"
        ]
        
        for pattern in rel_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match) >= 3:
                    relationships.append({
                        "source": match[0],
                        "target": match[1],
                        "relation": match[2],
                        "context": "",
                        "confidence": 0.8
                    })
        
        return relationships
    
    def _extract_claims_array(self, text: str) -> List[str]:
        """Extract claims from text."""
        claims = []
        
        # Look for claim patterns
        claim_patterns = [
            r'"([^"]+)"',  # Quoted strings
            r'•\s*([^\n]+)',  # Bullet points
            r'-\s*([^\n]+)',  # Dash points
        ]
        
        for pattern in claim_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 10:  # Only include substantial claims
                    claims.append(match.strip())
        
        return claims[:5]  # Limit to 5 claims 