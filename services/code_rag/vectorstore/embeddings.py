"""
Code embeddings system for Code RAG.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel
import hashlib

from ..models.entities import FunctionEntity, ClassEntity, VariableEntity, ModuleEntity, AnyEntity


class CodeEmbedder:
    """Generate semantic embeddings for code entities."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """Initialize the code embedder with a pre-trained model."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Cache for embeddings
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
    def embed_entity(self, entity: AnyEntity) -> np.ndarray:
        """Generate embedding for any code entity."""
        # Create cache key
        cache_key = self._create_cache_key(entity)
        
        # Check cache first
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # Generate text representation
        text = entity.to_search_text()
        
        # Generate embedding
        embedding = self._encode_text(text)
        
        # Cache the result
        self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def embed_function(self, func: FunctionEntity, include_body: bool = False) -> np.ndarray:
        """Generate specialized embedding for a function."""
        # Build comprehensive text representation
        text_parts = []
        
        # Function signature
        params_str = ", ".join([f"{p['name']}: {p.get('type', 'Any')}" for p in func.parameters])
        signature = f"def {func.name}({params_str})"
        if func.return_type:
            signature += f" -> {func.return_type}"
        text_parts.append(signature)
        
        # Docstring
        if func.docstring:
            text_parts.append(func.docstring)
        
        # Decorators
        if func.decorators:
            text_parts.append(f"decorators: {', '.join(func.decorators)}")
        
        # Context information
        if func.is_async:
            text_parts.append("async function")
        
        if func.is_method and func.class_name:
            text_parts.append(f"method of class {func.class_name}")
        
        # Function calls (what this function calls)
        if func.calls:
            text_parts.append(f"calls: {', '.join(func.calls[:5])}")  # Limit to first 5
        
        text = " ".join(text_parts)
        return self._encode_text(text)
    
    def embed_class(self, cls: ClassEntity) -> np.ndarray:
        """Generate specialized embedding for a class."""
        text_parts = []
        
        # Class declaration
        class_decl = f"class {cls.name}"
        if cls.base_classes:
            class_decl += f"({', '.join(cls.base_classes)})"
        text_parts.append(class_decl)
        
        # Docstring
        if cls.docstring:
            text_parts.append(cls.docstring)
        
        # Methods
        if cls.methods:
            text_parts.append(f"methods: {', '.join(cls.methods[:10])}")  # Limit to first 10
        
        # Properties
        if cls.properties:
            text_parts.append(f"properties: {', '.join(cls.properties[:10])}")
        
        # Interfaces
        if cls.interfaces:
            text_parts.append(f"implements: {', '.join(cls.interfaces)}")
        
        # Design patterns
        if cls.design_patterns:
            text_parts.append(f"patterns: {', '.join(cls.design_patterns)}")
        
        # Type information
        if cls.is_abstract:
            text_parts.append("abstract class")
        if cls.is_interface:
            text_parts.append("interface")
        
        text = " ".join(text_parts)
        return self._encode_text(text)
    
    def embed_variable(self, var: VariableEntity) -> np.ndarray:
        """Generate embedding for a variable."""
        text_parts = []
        
        # Variable declaration
        var_decl = f"variable {var.name}"
        if var.variable_type:
            var_decl += f": {var.variable_type}"
        text_parts.append(var_decl)
        
        # Scope and type information
        text_parts.append(f"scope: {var.scope}")
        
        if var.is_constant:
            text_parts.append("constant")
        
        if var.initial_value:
            text_parts.append(f"value: {var.initial_value}")
        
        text = " ".join(text_parts)
        return self._encode_text(text)
    
    def embed_module(self, module: ModuleEntity) -> np.ndarray:
        """Generate embedding for a module."""
        text_parts = []
        
        # Module name
        text_parts.append(f"module {module.name}")
        
        # Docstring
        if module.docstring:
            text_parts.append(module.docstring)
        
        # Imports
        if module.imports:
            text_parts.append(f"imports: {', '.join(module.imports[:10])}")
        
        # Functions
        if module.functions:
            text_parts.append(f"functions: {', '.join(module.functions[:10])}")
        
        # Classes
        if module.classes:
            text_parts.append(f"classes: {', '.join(module.classes[:10])}")
        
        text = " ".join(text_parts)
        return self._encode_text(text)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query."""
        return self._encode_text(query)
    
    def embed_batch(self, entities: List[AnyEntity]) -> List[np.ndarray]:
        """Generate embeddings for a batch of entities efficiently."""
        embeddings = []
        
        # Group entities by type for optimized processing
        functions = [e for e in entities if isinstance(e, FunctionEntity)]
        classes = [e for e in entities if isinstance(e, ClassEntity)]
        variables = [e for e in entities if isinstance(e, VariableEntity)]
        modules = [e for e in entities if isinstance(e, ModuleEntity)]
        
        # Process each type in batches
        if functions:
            func_texts = [func.to_search_text() for func in functions]
            func_embeddings = self._encode_texts_batch(func_texts)
            embeddings.extend(func_embeddings)
        
        if classes:
            class_texts = [cls.to_search_text() for cls in classes]
            class_embeddings = self._encode_texts_batch(class_texts)
            embeddings.extend(class_embeddings)
        
        if variables:
            var_texts = [var.to_search_text() for var in variables]
            var_embeddings = self._encode_texts_batch(var_texts)
            embeddings.extend(var_embeddings)
        
        if modules:
            module_texts = [mod.to_search_text() for mod in modules]
            module_embeddings = self._encode_texts_batch(module_texts)
            embeddings.extend(module_embeddings)
        
        return embeddings
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into embedding."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding (first token)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.flatten()
    
    def _encode_texts_batch(self, texts: List[str], batch_size: int = 8) -> List[np.ndarray]:
        """Encode multiple texts in batches for efficiency."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Add individual embeddings
            for embedding in batch_embeddings:
                embeddings.append(embedding)
        
        return embeddings
    
    def _create_cache_key(self, entity: AnyEntity) -> str:
        """Create a cache key for an entity."""
        content = f"{entity.id}:{entity.name}:{entity.entity_type.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def find_similar_entities(self, query_embedding: np.ndarray, 
                            entity_embeddings: List[np.ndarray],
                            entities: List[AnyEntity],
                            top_k: int = 10,
                            threshold: float = 0.0) -> List[tuple]:
        """Find most similar entities to a query embedding."""
        similarities = []
        
        for i, entity_embedding in enumerate(entity_embeddings):
            similarity = self.calculate_similarity(query_embedding, entity_embedding)
            if similarity >= threshold:
                similarities.append((similarity, entities[i]))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self._embedding_cache)


class CodeContextEmbedder(CodeEmbedder):
    """Enhanced embedder that includes contextual information."""
    
    def embed_function_with_context(self, func: FunctionEntity, 
                                   context: Dict[str, Any]) -> np.ndarray:
        """Embed function with additional context."""
        text_parts = []
        
        # Base function text
        base_text = self.embed_function(func)
        text_parts.append(func.to_search_text())
        
        # Add context
        if "containing_class" in context:
            text_parts.append(f"in class {context['containing_class']}")
        
        if "module_name" in context:
            text_parts.append(f"in module {context['module_name']}")
        
        if "framework" in context:
            text_parts.append(f"framework: {context['framework']}")
        
        if "related_functions" in context:
            related = context["related_functions"][:3]  # Limit to 3
            text_parts.append(f"related to: {', '.join(related)}")
        
        text = " ".join(text_parts)
        return self._encode_text(text) 