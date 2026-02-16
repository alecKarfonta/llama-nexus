"""
Dependency Parsing-based Relationship Extractor
Uses syntactic dependencies to extract relationships between entities
"""

import spacy
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)

@dataclass
class DependencyRelation:
    """Represents a relationship extracted using dependency parsing."""
    subject: str
    object: str
    relation_type: str
    confidence: float
    context: str
    dependency_path: List[str]

class DependencyExtractor:
    """Dependency parsing-based relationship extractor."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the dependency extractor.
        
        Args:
            model_name: spaCy model name to use
        """
        self.model_name = model_name
        
        try:
            logger.info(f"Loading spaCy model: {model_name}")
            self.nlp = spacy.load(model_name)
            logger.info("✅ spaCy model loaded successfully")
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Installing...")
            try:
                import subprocess
                subprocess.run([self.nlp.python_executable, "-m", "spacy", "download", model_name], check=True)
                self.nlp = spacy.load(model_name)
                logger.info("✅ spaCy model installed and loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to install spaCy model: {e}")
                self.nlp = None
        except Exception as e:
            logger.error(f"❌ Failed to load spaCy model: {e}")
            self.nlp = None
    
    def extract_relations(self, text: str, entities: List[Dict[str, Any]] = None) -> List[DependencyRelation]:
        """
        Extract relationships using dependency parsing.
        
        Args:
            text: Input text to process
            entities: Optional list of pre-extracted entities
            
        Returns:
            List of dependency-based relations
        """
        if not self.nlp:
            logger.warning("spaCy model not loaded, skipping dependency extraction")
            return []
        
        try:
            doc = self.nlp(text)
            relations = []
            
            # Extract relations using different dependency patterns
            relations.extend(self._extract_subject_verb_object_relations(doc))
            relations.extend(self._extract_prepositional_relations(doc))
            relations.extend(self._extract_nominal_relations(doc))
            relations.extend(self._extract_compound_relations(doc))
            
            # If entities are provided, focus on entity pairs
            if entities:
                entity_relations = self._extract_entity_based_relations(doc, entities)
                relations.extend(entity_relations)
            
            logger.info(f"Extracted {len(relations)} dependency-based relations")
            return relations
            
        except Exception as e:
            logger.error(f"Error extracting dependency relations: {e}")
            return []
    
    def _extract_subject_verb_object_relations(self, doc) -> List[DependencyRelation]:
        """Extract subject-verb-object relationships."""
        relations = []
        
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Find subject and object
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child
                    elif child.dep_ in ["dobj", "pobj"]:
                        obj = child
                
                if subject and obj:
                    relation = DependencyRelation(
                        subject=subject.text,
                        object=obj.text,
                        relation_type=token.lemma_,
                        confidence=0.8,
                        context=f"{subject.text} {token.text} {obj.text}",
                        dependency_path=[subject.dep_, token.dep_, obj.dep_]
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_prepositional_relations(self, doc) -> List[DependencyRelation]:
        """Extract prepositional relationships."""
        relations = []
        
        for token in doc:
            if token.pos_ == "ADP":  # Preposition
                # Find the head noun and the object of the preposition
                head = token.head
                obj = None
                
                for child in token.children:
                    if child.dep_ == "pobj":
                        obj = child
                        break
                
                if head and obj:
                    relation = DependencyRelation(
                        subject=head.text,
                        object=obj.text,
                        relation_type=f"{token.text}_of",
                        confidence=0.7,
                        context=f"{head.text} {token.text} {obj.text}",
                        dependency_path=[head.dep_, token.dep_, obj.dep_]
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_nominal_relations(self, doc) -> List[DependencyRelation]:
        """Extract nominal relationships (noun-noun relationships)."""
        relations = []
        
        for token in doc:
            if token.pos_ == "NOUN":
                # Look for compound nouns and appositions
                for child in token.children:
                    if child.dep_ == "compound":
                        relation = DependencyRelation(
                            subject=child.text,
                            object=token.text,
                            relation_type="part_of",
                            confidence=0.6,
                            context=f"{child.text} {token.text}",
                            dependency_path=[child.dep_, token.dep_]
                        )
                        relations.append(relation)
                    
                    elif child.dep_ == "appos":
                        relation = DependencyRelation(
                            subject=token.text,
                            object=child.text,
                            relation_type="is_a",
                            confidence=0.7,
                            context=f"{token.text} {child.text}",
                            dependency_path=[token.dep_, child.dep_]
                        )
                        relations.append(relation)
        
        return relations
    
    def _extract_compound_relations(self, doc) -> List[DependencyRelation]:
        """Extract compound relationships."""
        relations = []
        
        for token in doc:
            if token.dep_ == "compound":
                head = token.head
                relation = DependencyRelation(
                    subject=token.text,
                    object=head.text,
                    relation_type="modifies",
                    confidence=0.6,
                    context=f"{token.text} {head.text}",
                    dependency_path=[token.dep_, head.dep_]
                )
                relations.append(relation)
        
        return relations
    
    def _extract_entity_based_relations(self, doc, entities: List[Dict[str, Any]]) -> List[DependencyRelation]:
        """Extract relationships between specific entities."""
        relations = []
        
        # Create entity spans
        entity_spans = []
        for entity in entities:
            entity_text = entity.get('name', entity.get('text', ''))
            if entity_text:
                # Find entity in the document
                for token in doc:
                    if entity_text.lower() in token.text.lower():
                        entity_spans.append((token, entity))
                        break
        
        # Find relationships between entity pairs
        for i, (token1, entity1) in enumerate(entity_spans):
            for j, (token2, entity2) in enumerate(entity_spans):
                if i == j:
                    continue
                
                # Check if tokens are in the same sentence
                if token1.sent == token2.sent:
                    relation = self._find_entity_relation(token1, token2, entity1, entity2)
                    if relation:
                        relations.append(relation)
        
        return relations
    
    def _find_entity_relation(self, token1, token2, entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Optional[DependencyRelation]:
        """Find relationship between two entity tokens."""
        # Check dependency path between tokens
        path = self._get_dependency_path(token1, token2)
        
        if not path:
            return None
        
        # Determine relation type based on dependency path and entity types
        relation_type = self._classify_entity_relation(path, entity1, entity2)
        
        if relation_type:
            return DependencyRelation(
                subject=entity1.get('name', entity1.get('text', '')),
                object=entity2.get('name', entity2.get('text', '')),
                relation_type=relation_type,
                confidence=0.7,
                context=f"{token1.text} {path} {token2.text}",
                dependency_path=path
            )
        
        return None
    
    def _get_dependency_path(self, token1, token2) -> List[str]:
        """Get the dependency path between two tokens."""
        try:
            # Find the lowest common ancestor
            ancestor = token1.sent.root
            
            # Get path from token1 to ancestor
            path1 = []
            current = token1
            while current != ancestor:
                path1.append(current.dep_)
                current = current.head
            
            # Get path from ancestor to token2
            path2 = []
            current = token2
            while current != ancestor:
                path2.insert(0, current.dep_)
                current = current.head
            
            return path1 + path2
        except:
            return []
    
    def _classify_entity_relation(self, path: List[str], entity1: Dict[str, Any], entity2: Dict[str, Any]) -> Optional[str]:
        """Classify the relationship type based on dependency path and entity types."""
        entity1_type = entity1.get('entity_type', '').lower()
        entity2_type = entity2.get('entity_type', '').lower()
        
        # Check for specific patterns
        if 'nsubj' in path and 'dobj' in path:
            return 'action_on'
        elif 'compound' in path:
            return 'part_of'
        elif 'appos' in path:
            return 'is_a'
        elif 'prep' in path:
            return 'located_in'
        
        # Entity type specific patterns
        if entity1_type == 'person' and entity2_type == 'organization':
            if 'nsubj' in path:
                return 'works_for'
            elif 'dobj' in path:
                return 'employs'
        
        elif entity1_type == 'organization' and entity2_type == 'organization':
            if 'compound' in path:
                return 'subsidiary_of'
            elif 'prep' in path:
                return 'part_of'
        
        elif entity1_type == 'person' and entity2_type == 'person':
            if 'compound' in path:
                return 'related_to'
        
        return None
    
    def get_syntactic_features(self, text: str) -> Dict[str, Any]:
        """Extract syntactic features from text."""
        if not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            
            features = {
                'sentences': len(list(doc.sents)),
                'tokens': len(doc),
                'noun_chunks': len(list(doc.noun_chunks)),
                'verbs': len([token for token in doc if token.pos_ == 'VERB']),
                'entities': len(doc.ents),
                'dependency_relations': []
            }
            
            # Extract dependency relations
            for token in doc:
                if token.dep_ != 'punct':
                    features['dependency_relations'].append({
                        'token': token.text,
                        'dep': token.dep_,
                        'head': token.head.text if token.head else None,
                        'pos': token.pos_
                    })
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting syntactic features: {e}")
            return {}
    
    def is_available(self) -> bool:
        """Check if the dependency extractor is available."""
        return self.nlp is not None

def get_dependency_extractor() -> DependencyExtractor:
    """Get a singleton instance of the dependency extractor."""
    if not hasattr(get_dependency_extractor, '_instance'):
        get_dependency_extractor._instance = DependencyExtractor()
    return get_dependency_extractor._instance 