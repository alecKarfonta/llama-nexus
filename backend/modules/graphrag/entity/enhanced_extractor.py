"""
Enhanced Entity and Relationship Extractor
Integrates SpanBERT, dependency parsing, and entity linking for improved extraction
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from entity_extractor import Entity, Relationship, ExtractionResult
from span_extractor import SpanBERTExtractor, get_span_extractor
from dependency_extractor import DependencyExtractor, get_dependency_extractor
from entity_linker import EntityLinker, get_entity_linker

logger = logging.getLogger(__name__)

@dataclass
class EnhancedExtractionResult:
    """Enhanced extraction result with additional metadata."""
    entities: List[Entity]
    relationships: List[Relationship]
    claims: List[str]
    source_chunk: str
    span_extractions: List[Dict[str, Any]] = None
    dependency_extractions: List[Dict[str, Any]] = None
    linked_entities: List[Dict[str, Any]] = None
    extraction_metadata: Dict[str, Any] = None

class EnhancedEntityExtractor:
    """Enhanced entity and relationship extractor with multiple extraction methods."""
    
    def __init__(self, use_spanbert: bool = True, use_dependency: bool = True, use_entity_linking: bool = True):
        """
        Initialize the enhanced extractor.
        
        Args:
            use_spanbert: Whether to use SpanBERT extraction
            use_dependency: Whether to use dependency parsing
            use_entity_linking: Whether to use entity linking
        """
        self.use_spanbert = use_spanbert
        self.use_dependency = use_dependency
        self.use_entity_linking = use_entity_linking
        
        # Initialize extractors
        self.span_extractor = get_span_extractor() if use_spanbert else None
        self.dependency_extractor = get_dependency_extractor() if use_dependency else None
        self.entity_linker = get_entity_linker() if use_entity_linking else None
        
        logger.info(f"Enhanced extractor initialized - SpanBERT: {use_spanbert}, Dependency: {use_dependency}, Entity Linking: {use_entity_linking}")
    
    def extract_entities_and_relations(self, text_chunk: str, domain: str = "general") -> EnhancedExtractionResult:
        """
        Extract entities and relationships using multiple methods.
        
        Args:
            text_chunk: Text to process
            domain: Domain context
            
        Returns:
            Enhanced extraction result
        """
        logger.info(f"🔍 Starting enhanced extraction for domain: {domain}")
        
        # Initialize result containers
        all_entities = []
        all_relationships = []
        span_extractions = []
        dependency_extractions = []
        linked_entities = []
        
        # 1. SpanBERT extraction
        if self.use_spanbert and self.span_extractor and self.span_extractor.is_available():
            try:
                logger.info("🔍 Using SpanBERT for span-based extraction...")
                spans = self.span_extractor.extract_spans(text_chunk)
                span_relations = self.span_extractor.extract_span_relations(text_chunk, spans)
                
                # Convert spans to entities
                for span in spans:
                    entity = Entity(
                        name=span.text,
                        entity_type=span.entity_type.upper(),
                        description=f"Extracted by SpanBERT (confidence: {span.confidence:.3f})",
                        confidence=span.confidence,
                        source_chunk=text_chunk,
                        metadata={
                            "domain": domain,
                            "extraction_method": "spanbert",
                            "span_id": span.span_id,
                            "start": span.start,
                            "end": span.end
                        }
                    )
                    all_entities.append(entity)
                
                # Convert span relations to relationships
                for span_rel in span_relations:
                    relationship = Relationship(
                        source=span_rel.subject_span.text,
                        target=span_rel.object_span.text,
                        relation_type=span_rel.relation_type,
                        context=span_rel.context,
                        confidence=span_rel.confidence,
                        metadata={
                            "extraction_method": "spanbert",
                            "source_span_id": span_rel.subject_span.span_id,
                            "target_span_id": span_rel.object_span.span_id
                        }
                    )
                    all_relationships.append(relationship)
                
                span_extractions = [{"spans": len(spans), "relations": len(span_relations)}]
                logger.info(f"✅ SpanBERT extracted {len(spans)} spans and {len(span_relations)} relations")
                
            except Exception as e:
                logger.error(f"❌ SpanBERT extraction failed: {e}")
        elif self.use_spanbert:
            logger.warning("SpanBERT is enabled but not available. Skipping...")
        
        # 2. Dependency parsing extraction
        if self.use_dependency and self.dependency_extractor and self.dependency_extractor.is_available():
            try:
                logger.info("🔍 Using dependency parsing for syntactic extraction...")
                dep_relations = self.dependency_extractor.extract_relations(text_chunk)
                
                # Convert dependency relations to relationships
                for dep_rel in dep_relations:
                    relationship = Relationship(
                        source=dep_rel.subject,
                        target=dep_rel.object,
                        relation_type=dep_rel.relation_type,
                        context=dep_rel.context,
                        confidence=dep_rel.confidence,
                        metadata={
                            "extraction_method": "dependency_parsing",
                            "dependency_path": dep_rel.dependency_path
                        }
                    )
                    all_relationships.append(relationship)
                
                dependency_extractions = [{"relations": len(dep_relations)}]
                logger.info(f"✅ Dependency parsing extracted {len(dep_relations)} relations")
                
            except Exception as e:
                logger.error(f"❌ Dependency parsing extraction failed: {e}")
        
        # 3. Entity linking
        if self.use_entity_linking and self.entity_linker and self.entity_linker.is_available():
            try:
                logger.info("🔍 Using entity linking for disambiguation...")
                linked_entities = self.entity_linker.link_entities(all_entities)
                
                # Enrich entities with knowledge base information
                enriched_entities = self.entity_linker.enrich_entities(all_entities)
                
                # Update entities with knowledge base information
                for i, enriched_entity in enumerate(enriched_entities):
                    if i < len(all_entities):
                        if 'knowledge_base_id' in enriched_entity:
                            all_entities[i].metadata['knowledge_base_id'] = enriched_entity['knowledge_base_id']
                        if 'knowledge_base_url' in enriched_entity:
                            all_entities[i].metadata['knowledge_base_url'] = enriched_entity['knowledge_base_url']
                        if 'description' in enriched_entity:
                            all_entities[i].metadata['description'] = enriched_entity['description']
                        if 'enriched' in enriched_entity:
                            all_entities[i].metadata['enriched'] = enriched_entity['enriched']
                
                logger.info(f"✅ Entity linking processed {len(linked_entities)} entities")
                
            except Exception as e:
                logger.error(f"❌ Entity linking failed: {e}")
        
        # 4. Deduplicate and merge results
        final_entities = self._deduplicate_entities(all_entities)
        final_relationships = self._deduplicate_relationships(all_relationships)
        
        # 5. Create extraction metadata
        extraction_metadata = {
            "total_entities": len(final_entities),
            "total_relationships": len(final_relationships),
            "span_extractions": span_extractions,
            "dependency_extractions": dependency_extractions,
            "linked_entities_count": len(linked_entities),
            "domain": domain,
            "extraction_methods": []
        }
        
        if self.use_spanbert and self.span_extractor and self.span_extractor.is_available():
            extraction_metadata["extraction_methods"].append("spanbert")
        if self.use_dependency and self.dependency_extractor and self.dependency_extractor.is_available():
            extraction_metadata["extraction_methods"].append("dependency_parsing")
        if self.use_entity_linking and self.entity_linker and self.entity_linker.is_available():
            extraction_metadata["extraction_methods"].append("entity_linking")
        
        logger.info(f"✅ Enhanced extraction complete: {len(final_entities)} entities, {len(final_relationships)} relationships")
        
        return EnhancedExtractionResult(
            entities=final_entities,
            relationships=final_relationships,
            claims=[],  # Claims not implemented in enhanced version
            source_chunk=text_chunk,
            span_extractions=span_extractions,
            dependency_extractions=dependency_extractions,
            linked_entities=linked_entities,
            extraction_metadata=extraction_metadata
        )
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities based on name and type."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create a key for deduplication
            key = (entity.name.lower(), entity.entity_type.lower())
            
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # Merge metadata if entity already exists
                existing_entity = next(e for e in unique_entities if (e.name.lower(), e.entity_type.lower()) == key)
                existing_entity.metadata.update(entity.metadata)
                # Keep the higher confidence
                if entity.confidence > existing_entity.confidence:
                    existing_entity.confidence = entity.confidence
        
        return unique_entities
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Deduplicate relationships based on source, target, and type."""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # Create a key for deduplication
            key = (rel.source.lower(), rel.target.lower(), rel.relation_type.lower())
            
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
            else:
                # Merge metadata if relationship already exists
                existing_rel = next(r for r in unique_relationships if (r.source.lower(), r.target.lower(), r.relation_type.lower()) == key)
                existing_rel.metadata.update(rel.metadata)
                # Keep the higher confidence
                if rel.confidence > existing_rel.confidence:
                    existing_rel.confidence = rel.confidence
        
        return unique_relationships
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the extraction capabilities."""
        stats = {
            "spanbert_available": self.span_extractor.is_available() if self.span_extractor else False,
            "dependency_available": self.dependency_extractor.is_available() if self.dependency_extractor else False,
            "entity_linking_available": self.entity_linker.is_available() if self.entity_linker else False,
            "extraction_methods": []
        }
        
        if self.use_spanbert and self.span_extractor and self.span_extractor.is_available():
            stats["extraction_methods"].append("spanbert")
        if self.use_dependency and self.dependency_extractor and self.dependency_extractor.is_available():
            stats["extraction_methods"].append("dependency_parsing")
        if self.use_entity_linking and self.entity_linker and self.entity_linker.is_available():
            stats["extraction_methods"].append("entity_linking")
        
        return stats

def get_enhanced_entity_extractor() -> EnhancedEntityExtractor:
    """Get a singleton instance of the enhanced entity extractor."""
    if not hasattr(get_enhanced_entity_extractor, '_instance'):
        get_enhanced_entity_extractor._instance = EnhancedEntityExtractor()
    return get_enhanced_entity_extractor._instance 