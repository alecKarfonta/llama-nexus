"""
SpanBERT-based Entity and Relationship Extractor
Provides enhanced span-based extraction using SpanBERT model
"""

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Span:
    """Represents a text span with entity information."""
    start: int
    end: int
    text: str
    entity_type: str
    confidence: float
    span_id: Optional[str] = None

@dataclass
class SpanRelation:
    """Represents a relationship between spans."""
    subject_span: Span
    object_span: Span
    relation_type: str
    confidence: float
    context: str

class SpanBERTExtractor:
    """SpanBERT-based entity and relationship extractor."""
    
    def __init__(self, model_name: str = "mrm8488/spanbert-finetuned-squadv2", device: str = "cpu"):
        """
        Initialize the SpanBERT extractor.
        
        Args:
            model_name: HuggingFace model name for SpanBERT
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.device = device
        self.model_name = model_name
        
        try:
            logger.info(f"Loading SpanBERT model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info("✅ SpanBERT model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load SpanBERT model: {e}")
            self.model = None
            self.tokenizer = None
    
    def extract_spans(self, text: str, entity_types: List[str] = None) -> List[Span]:
        """
        Extract entity spans from text using SpanBERT.
        
        Args:
            text: Input text to process
            entity_types: List of entity types to extract
            
        Returns:
            List of extracted spans
        """
        if not self.model:
            logger.warning("SpanBERT model not loaded, skipping extraction")
            return []
        
        try:
            # Tokenize the text
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(**tokens)
                hidden_states = outputs.last_hidden_state
            
            # Extract spans using attention weights and span representations
            spans = self._extract_spans_from_hidden_states(hidden_states, text, entity_types)
            
            logger.info(f"Extracted {len(spans)} spans from text")
            return spans
            
        except Exception as e:
            logger.error(f"Error extracting spans: {e}")
            return []
    
    def extract_span_relations(self, text: str, spans: List[Span], relation_types: List[str] = None) -> List[SpanRelation]:
        """
        Extract relationships between spans using SpanBERT.
        
        Args:
            text: Input text
            spans: List of extracted spans
            relation_types: List of relation types to extract
            
        Returns:
            List of span relations
        """
        if not self.model or len(spans) < 2:
            return []
        
        try:
            relations = []
            
            # Create span pairs for relation extraction
            for i, span1 in enumerate(spans):
                for j, span2 in enumerate(spans):
                    if i == j:
                        continue
                    
                    # Check if spans are close enough for a relation
                    if self._spans_are_related(span1, span2, text):
                        relation = self._extract_relation_between_spans(span1, span2, text, relation_types)
                        if relation:
                            relations.append(relation)
            
            logger.info(f"Extracted {len(relations)} span relations")
            return relations
            
        except Exception as e:
            logger.error(f"Error extracting span relations: {e}")
            return []
    
    def _extract_spans_from_hidden_states(self, hidden_states: torch.Tensor, text: str, entity_types: List[str]) -> List[Span]:
        """Extract spans from model hidden states."""
        spans = []
        
        # Convert hidden states to span representations
        # This is a simplified implementation - in practice, you'd need more sophisticated span extraction
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Use attention weights to identify potential entity spans
        attention_weights = torch.softmax(hidden_states @ hidden_states.transpose(-2, -1), dim=-1)
        
        # Find spans with high attention scores
        for i in range(1, seq_len - 1):  # Skip special tokens
            if attention_weights[0, i, i] > 0.5:  # High self-attention
                # Find span boundaries
                start, end = self._find_span_boundaries(attention_weights[0, i, :], i)
                
                if end > start:
                    span_text = self.tokenizer.decode(
                        self.tokenizer.encode(text)[start:end],
                        skip_special_tokens=True
                    )
                    
                    if span_text.strip():
                        span = Span(
                            start=start,
                            end=end,
                            text=span_text.strip(),
                            entity_type=self._classify_span(span_text, entity_types),
                            confidence=float(attention_weights[0, i, i])
                        )
                        spans.append(span)
        
        return spans
    
    def _find_span_boundaries(self, attention_row: torch.Tensor, center: int) -> Tuple[int, int]:
        """Find span boundaries based on attention weights."""
        # Simple boundary detection - in practice, use more sophisticated methods
        threshold = 0.3
        start = center
        end = center
        
        # Expand left
        for i in range(center - 1, 0, -1):
            if attention_row[i] > threshold:
                start = i
            else:
                break
        
        # Expand right
        for i in range(center + 1, len(attention_row)):
            if attention_row[i] > threshold:
                end = i
            else:
                break
        
        return start, end
    
    def _classify_span(self, span_text: str, entity_types: List[str]) -> str:
        """Classify a span into an entity type."""
        # Simple classification based on text patterns
        span_lower = span_text.lower()
        
        if entity_types:
            # Use provided entity types
            for entity_type in entity_types:
                if self._matches_entity_pattern(span_text, entity_type):
                    return entity_type
        
        # Default classification based on patterns
        if any(word in span_lower for word in ['inc', 'corp', 'company', 'ltd']):
            return 'ORGANIZATION'
        elif any(word in span_lower for word in ['mr', 'ms', 'dr', 'professor']):
            return 'PERSON'
        elif any(word in span_lower for word in ['street', 'avenue', 'road', 'city']):
            return 'LOCATION'
        elif any(word in span_lower for word in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']):
            return 'DATE'
        else:
            return 'MISC'
    
    def _matches_entity_pattern(self, text: str, entity_type: str) -> bool:
        """Check if text matches an entity type pattern."""
        text_lower = text.lower()
        entity_lower = entity_type.lower()
        
        # Simple pattern matching
        if entity_lower in ['person', 'people']:
            return any(word in text_lower for word in ['mr', 'ms', 'dr', 'professor', 'president', 'ceo'])
        elif entity_lower in ['organization', 'company']:
            return any(word in text_lower for word in ['inc', 'corp', 'company', 'ltd', 'llc'])
        elif entity_lower in ['location', 'place']:
            return any(word in text_lower for word in ['street', 'avenue', 'road', 'city', 'state', 'country'])
        elif entity_lower in ['date', 'time']:
            return any(word in text_lower for word in ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'])
        
        return False
    
    def _spans_are_related(self, span1: Span, span2: Span, text: str) -> bool:
        """Check if two spans are likely to have a relationship."""
        # Check distance between spans
        distance = abs(span1.start - span2.start)
        if distance > 100:  # Too far apart
            return False
        
        # Check if they appear in the same sentence
        span1_in_text = span1.text in text
        span2_in_text = span2.text in text
        
        return span1_in_text and span2_in_text
    
    def _extract_relation_between_spans(self, span1: Span, span2: Span, text: str, relation_types: List[str]) -> Optional[SpanRelation]:
        """Extract relationship between two spans."""
        # Simple relation extraction based on text patterns
        # In practice, use more sophisticated methods
        
        # Check for common relation patterns
        relation_patterns = {
            'founder': ['founded', 'founder', 'co-founder'],
            'works_for': ['works for', 'employed by', 'employee of'],
            'located_in': ['located in', 'based in', 'headquartered in'],
            'acquired': ['acquired', 'bought', 'purchased'],
            'part_of': ['part of', 'subsidiary of', 'division of']
        }
        
        # Find the text between spans
        start_pos = min(span1.start, span2.start)
        end_pos = max(span1.end, span2.end)
        context_text = text[start_pos:end_pos].lower()
        
        for relation_type, patterns in relation_patterns.items():
            if relation_types and relation_type not in relation_types:
                continue
                
            for pattern in patterns:
                if pattern in context_text:
                    return SpanRelation(
                        subject_span=span1,
                        object_span=span2,
                        relation_type=relation_type,
                        confidence=min(span1.confidence, span2.confidence),
                        context=text[start_pos:end_pos]
                    )
        
        return None
    
    def is_available(self) -> bool:
        """Check if the SpanBERT model is available."""
        return self.model is not None

def get_span_extractor() -> SpanBERTExtractor:
    """Get a singleton instance of the SpanBERT extractor."""
    if not hasattr(get_span_extractor, '_instance'):
        get_span_extractor._instance = SpanBERTExtractor()
    return get_span_extractor._instance 