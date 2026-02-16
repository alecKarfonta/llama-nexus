from typing import List, Dict, Any, Tuple
from entity_extractor import Entity
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import numpy as np

class EntityResolver:
    """Resolves and deduplicates entities using semantic similarity, fuzzy matching, and LLM-based disambiguation."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", fuzzy_threshold: int = 85, semantic_threshold: float = 0.85):
        self.model = SentenceTransformer(model_name)
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold

    def resolve_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate and merge entities."""
        if not entities:
            return []
        
        # Step 1: Group by entity type
        type_groups: Dict[str, List[Entity]] = {}
        for entity in entities:
            type_groups.setdefault(entity.entity_type, []).append(entity)
        
        resolved_entities = []
        for entity_type, group in type_groups.items():
            merged = self._deduplicate_group(group)
            resolved_entities.extend(merged)
        return resolved_entities

    def _deduplicate_group(self, group: List[Entity]) -> List[Entity]:
        """Deduplicate a group of entities of the same type."""
        if len(group) == 1:
            return group
        
        # Step 2: Semantic similarity clustering
        names = [e.name for e in group]
        embeddings = self.model.encode(names)
        used = set()
        merged_entities = []
        
        for i, entity in enumerate(group):
            if i in used:
                continue
            cluster = [entity]
            used.add(i)
            for j in range(i+1, len(group)):
                if j in used:
                    continue
                # Semantic similarity
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                # Fuzzy string match
                fuzzy = fuzz.ratio(entity.name, group[j].name)
                if sim >= self.semantic_threshold or fuzzy >= self.fuzzy_threshold:
                    cluster.append(group[j])
                    used.add(j)
            # Merge cluster
            merged_entity = self._merge_cluster(cluster)
            merged_entities.append(merged_entity)
        return merged_entities

    def _cosine_similarity(self, a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _merge_cluster(self, cluster: List[Entity]) -> Entity:
        """Merge a cluster of similar entities into one."""
        if len(cluster) == 1:
            return cluster[0]
        # Choose the most frequent or longest name
        names = [e.name for e in cluster]
        name = max(set(names), key=names.count)
        # Merge descriptions
        descriptions = [e.description for e in cluster if e.description]
        description = descriptions[0] if descriptions else None
        # Average confidence
        confidence = float(np.mean([e.confidence for e in cluster]))
        # Merge metadata
        metadata = {}
        for e in cluster:
            if e.metadata:
                metadata.update(e.metadata)
        return Entity(
            name=name,
            entity_type=cluster[0].entity_type,
            description=description,
            confidence=confidence,
            source_chunk=None,
            metadata=metadata
        )

    def disambiguate_entities_llm(self, entities: List[Entity], context: str = "") -> List[Entity]:
        """Stub for LLM-based disambiguation (to be implemented with LLM integration)."""
        # For now, just return the input list
        # In production, use an LLM to resolve ambiguous cases based on context
        return entities

    def integrate_user_feedback(self, entities: List[Entity], feedback: Dict[str, Any]) -> List[Entity]:
        """Stub for user feedback integration."""
        # For now, just return the input list
        # In production, use feedback to merge/split entities as needed
        return entities 