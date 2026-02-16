"""
Base parser interface for Code RAG system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time
from pathlib import Path

from ..models.entities import (
    ParseResult, FunctionEntity, ClassEntity, VariableEntity, 
    ModuleEntity, RelationshipEntity, AnyEntity, RelationshipType
)


class BaseCodeParser(ABC):
    """Abstract base class for all code parsers."""
    
    def __init__(self):
        self.supported_extensions: List[str] = []
        self.language_name: str = ""
    
    @abstractmethod
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a code file and extract entities and relationships."""
        pass
    
    @abstractmethod
    def extract_functions(self, ast_tree: Any, source_code: str, file_path: str) -> List[FunctionEntity]:
        """Extract function entities from AST."""
        pass
    
    @abstractmethod
    def extract_classes(self, ast_tree: Any, source_code: str, file_path: str) -> List[ClassEntity]:
        """Extract class entities from AST."""
        pass
    
    @abstractmethod
    def extract_variables(self, ast_tree: Any, source_code: str, file_path: str) -> List[VariableEntity]:
        """Extract variable entities from AST."""
        pass
    
    @abstractmethod
    def extract_imports(self, ast_tree: Any, source_code: str, file_path: str) -> List[str]:
        """Extract import statements."""
        pass
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        file_extension = Path(file_path).suffix.lower()
        return file_extension in self.supported_extensions
    
    def extract_relationships(self, entities: List[AnyEntity], ast_tree: Any, source_code: str) -> List[RelationshipEntity]:
        """Extract relationships between entities. Base implementation."""
        relationships = []
        
        # Extract function call relationships
        for entity in entities:
            if isinstance(entity, FunctionEntity):
                for called_function in entity.calls:
                    # Find the target entity
                    target_entity = self._find_entity_by_name(entities, called_function)
                    if target_entity:
                        rel = RelationshipEntity(
                            source_entity_id=entity.id,
                            target_entity_id=target_entity.id,
                            relationship_type=RelationshipType.CALLS,
                            context=f"{entity.name} calls {called_function}",
                            confidence=0.9
                        )
                        relationships.append(rel)
        
        return relationships
    
    def _find_entity_by_name(self, entities: List[AnyEntity], name: str) -> Optional[AnyEntity]:
        """Find an entity by name in the list."""
        for entity in entities:
            if entity.name == name:
                return entity
        return None
    
    def _calculate_complexity(self, ast_node: Any) -> int:
        """Calculate cyclomatic complexity. Base implementation returns 1."""
        return 1
    
    def _determine_visibility(self, name: str) -> str:
        """Determine visibility based on naming conventions."""
        if name.startswith('__') and name.endswith('__'):
            return "special"
        elif name.startswith('_'):
            return "private"
        else:
            return "public"
    
    def _extract_docstring(self, ast_node: Any) -> Optional[str]:
        """Extract docstring from AST node. Base implementation."""
        return None
    
    def _get_source_segment(self, source_code: str, line_start: int, line_end: int) -> str:
        """Extract source code segment between line numbers."""
        lines = source_code.split('\n')
        if line_start <= 0 or line_end > len(lines):
            return ""
        
        # Convert to 0-based indexing
        start_idx = max(0, line_start - 1)
        end_idx = min(len(lines), line_end)
        
        return '\n'.join(lines[start_idx:end_idx])


class ParseError(Exception):
    """Exception raised when parsing fails."""
    
    def __init__(self, message: str, file_path: str, line_number: Optional[int] = None):
        self.message = message
        self.file_path = file_path
        self.line_number = line_number
        super().__init__(f"Parse error in {file_path}: {message}")


class UnsupportedLanguageError(Exception):
    """Exception raised when trying to parse an unsupported language."""
    pass 