"""
Integration module for Code RAG and GraphRAG systems.
"""

import requests
import json
import tempfile
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import time

from ..models.entities import AnyEntity, FunctionEntity, ClassEntity, VariableEntity, ModuleEntity, RelationshipEntity
from ..parsers.python_parser import PythonParser


class GraphRAGIntegration:
    """Integrates Code RAG artifacts with GraphRAG system."""
    
    def __init__(self, graphrag_api_url: str = "http://localhost:8000"):
        """Initialize the integration with GraphRAG API."""
        self.graphrag_api_url = graphrag_api_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CodeRAG-Integration/1.0'
        })
    
    def send_code_entities_to_graphrag(self, 
                                     entities: List[AnyEntity], 
                                     relationships: List[RelationshipEntity],
                                     domain: str = "code",
                                     project_name: str = "code_project") -> Dict[str, Any]:
        """
        Send Code RAG entities and relationships to GraphRAG.
        
        Args:
            entities: List of code entities (functions, classes, etc.)
            relationships: List of relationships between entities
            domain: Domain for the entities (default: "code")
            project_name: Name of the code project
            
        Returns:
            Integration result with success status and metadata
        """
        try:
            # Convert Code RAG entities to GraphRAG format
            graphrag_entities = self._convert_entities_to_graphrag_format(entities)
            graphrag_relationships = self._convert_relationships_to_graphrag_format(relationships)
            
            # Create a temporary document with code information
            code_document = self._create_code_document(entities, relationships, project_name)
            
            # Send to GraphRAG via document ingestion
            result = self._send_to_graphrag_via_document(code_document, domain)
            
            return {
                "success": True,
                "entities_sent": len(graphrag_entities),
                "relationships_sent": len(graphrag_relationships),
                "domain": domain,
                "project_name": project_name,
                "graphrag_response": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "entities_sent": 0,
                "relationships_sent": 0
            }
    
    def send_code_file_to_graphrag(self, 
                                  file_path: str, 
                                  domain: str = "code",
                                  project_name: str = None) -> Dict[str, Any]:
        """
        Parse a code file and send its entities to GraphRAG.
        
        Args:
            file_path: Path to the code file
            domain: Domain for the entities
            project_name: Name of the project (defaults to file name)
            
        Returns:
            Integration result
        """
        try:
            # Parse the code file
            parser = PythonParser()
            parse_result = parser.parse_file(file_path)
            
            if not parse_result.success:
                return {
                    "success": False,
                    "error": f"Failed to parse {file_path}: {parse_result.error}",
                    "entities_sent": 0,
                    "relationships_sent": 0
                }
            
            # Use file name as project name if not provided
            if project_name is None:
                project_name = Path(file_path).stem
            
            # Send entities to GraphRAG
            return self.send_code_entities_to_graphrag(
                parse_result.entities,
                parse_result.relationships,
                domain,
                project_name
            )
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "entities_sent": 0,
                "relationships_sent": 0
            }
    
    def send_code_directory_to_graphrag(self, 
                                      directory_path: str, 
                                      domain: str = "code",
                                      project_name: str = None,
                                      recursive: bool = True) -> Dict[str, Any]:
        """
        Parse all code files in a directory and send to GraphRAG.
        
        Args:
            directory_path: Path to the directory containing code files
            domain: Domain for the entities
            project_name: Name of the project (defaults to directory name)
            recursive: Whether to search recursively
            
        Returns:
            Integration result with summary
        """
        try:
            # Find all code files
            code_files = self._find_code_files(directory_path, recursive)
            
            if not code_files:
                return {
                    "success": False,
                    "error": f"No code files found in {directory_path}",
                    "files_processed": 0,
                    "entities_sent": 0,
                    "relationships_sent": 0
                }
            
            # Use directory name as project name if not provided
            if project_name is None:
                project_name = Path(directory_path).name
            
            # Process all files
            all_entities = []
            all_relationships = []
            successful_files = 0
            failed_files = 0
            
            for file_path in code_files:
                try:
                    parser = PythonParser()
                    parse_result = parser.parse_file(file_path)
                    
                    if parse_result.success:
                        all_entities.extend(parse_result.entities)
                        all_relationships.extend(parse_result.relationships)
                        successful_files += 1
                    else:
                        failed_files += 1
                        
                except Exception as e:
                    failed_files += 1
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            # Send all entities to GraphRAG
            if all_entities:
                result = self.send_code_entities_to_graphrag(
                    all_entities, all_relationships, domain, project_name
                )
                result.update({
                    "files_processed": successful_files,
                    "files_failed": failed_files,
                    "total_files": len(code_files)
                })
                return result
            else:
                return {
                    "success": False,
                    "error": "No entities extracted from any files",
                    "files_processed": successful_files,
                    "files_failed": failed_files,
                    "total_files": len(code_files),
                    "entities_sent": 0,
                    "relationships_sent": 0
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "files_processed": 0,
                "files_failed": 0,
                "total_files": 0,
                "entities_sent": 0,
                "relationships_sent": 0
            }
    
    def _convert_entities_to_graphrag_format(self, entities: List[AnyEntity]) -> List[Dict[str, Any]]:
        """Convert Code RAG entities to GraphRAG format."""
        graphrag_entities = []
        
        for entity in entities:
            # Create GraphRAG entity format
            graphrag_entity = {
                "name": entity.name,
                "type": entity.entity_type.value.upper(),
                "description": self._create_entity_description(entity),
                "source_file": entity.file_path,
                "metadata": {
                    "language": entity.language,
                    "line_start": entity.line_start,
                    "line_end": entity.line_end,
                    "visibility": entity.visibility.value,
                    "code_rag_id": entity.id
                }
            }
            
            # Add entity-specific metadata
            if isinstance(entity, FunctionEntity):
                graphrag_entity["metadata"].update({
                    "parameters": entity.parameters,
                    "return_type": entity.return_type,
                    "is_async": entity.is_async,
                    "is_method": entity.is_method,
                    "complexity": entity.complexity,
                    "calls": entity.calls
                })
            elif isinstance(entity, ClassEntity):
                graphrag_entity["metadata"].update({
                    "base_classes": entity.base_classes,
                    "methods": entity.methods,
                    "properties": entity.properties,
                    "is_abstract": entity.is_abstract,
                    "is_interface": entity.is_interface
                })
            elif isinstance(entity, VariableEntity):
                graphrag_entity["metadata"].update({
                    "variable_type": entity.variable_type,
                    "scope": entity.scope,
                    "is_constant": entity.is_constant
                })
            elif isinstance(entity, ModuleEntity):
                graphrag_entity["metadata"].update({
                    "imports": entity.imports,
                    "functions": entity.functions,
                    "classes": entity.classes
                })
            
            graphrag_entities.append(graphrag_entity)
        
        return graphrag_entities
    
    def _convert_relationships_to_graphrag_format(self, relationships: List[RelationshipEntity]) -> List[Dict[str, Any]]:
        """Convert Code RAG relationships to GraphRAG format."""
        graphrag_relationships = []
        
        for rel in relationships:
            # Create GraphRAG relationship format
            graphrag_relationship = {
                "source": rel.source_entity_id,
                "target": rel.target_entity_id,
                "relation": rel.relationship_type.value.upper(),
                "context": rel.context,
                "metadata": {
                    "confidence": rel.confidence,
                    "code_rag_id": rel.id
                }
            }
            
            graphrag_relationships.append(graphrag_relationship)
        
        return graphrag_relationships
    
    def _create_entity_description(self, entity: AnyEntity) -> str:
        """Create a human-readable description for the entity."""
        if isinstance(entity, FunctionEntity):
            params_str = ", ".join([f"{p.get('name', '')}: {p.get('type', 'Any')}" for p in entity.parameters])
            desc = f"Function {entity.name}({params_str})"
            if entity.return_type:
                desc += f" -> {entity.return_type}"
            if entity.docstring:
                desc += f" - {entity.docstring[:100]}"
            return desc
            
        elif isinstance(entity, ClassEntity):
            desc = f"Class {entity.name}"
            if entity.base_classes:
                desc += f" inherits from {', '.join(entity.base_classes)}"
            if entity.docstring:
                desc += f" - {entity.docstring[:100]}"
            return desc
            
        elif isinstance(entity, VariableEntity):
            desc = f"Variable {entity.name}"
            if entity.variable_type:
                desc += f": {entity.variable_type}"
            if entity.is_constant:
                desc += " (constant)"
            return desc
            
        elif isinstance(entity, ModuleEntity):
            desc = f"Module {entity.name}"
            if entity.docstring:
                desc += f" - {entity.docstring[:100]}"
            return desc
            
        else:
            return f"{entity.entity_type.value.title()} {entity.name}"
    
    def _create_code_document(self, entities: List[AnyEntity], relationships: List[RelationshipEntity], project_name: str) -> str:
        """Create a text document representing the code structure for GraphRAG ingestion."""
        lines = []
        lines.append(f"# Code Project: {project_name}")
        lines.append("")
        
        # Group entities by type
        functions = [e for e in entities if isinstance(e, FunctionEntity)]
        classes = [e for e in entities if isinstance(e, ClassEntity)]
        variables = [e for e in entities if isinstance(e, VariableEntity)]
        modules = [e for e in entities if isinstance(e, ModuleEntity)]
        
        # Document modules
        if modules:
            lines.append("## Modules")
            for module in modules:
                lines.append(f"- {module.name}")
                if module.imports:
                    lines.append(f"  Imports: {', '.join(module.imports[:5])}")
                if module.functions:
                    lines.append(f"  Functions: {', '.join(module.functions[:5])}")
                if module.classes:
                    lines.append(f"  Classes: {', '.join(module.classes[:5])}")
                lines.append("")
        
        # Document classes
        if classes:
            lines.append("## Classes")
            for cls in classes:
                lines.append(f"- {cls.name}")
                if cls.base_classes:
                    lines.append(f"  Inherits from: {', '.join(cls.base_classes)}")
                if cls.methods:
                    lines.append(f"  Methods: {', '.join(cls.methods[:5])}")
                if cls.docstring:
                    lines.append(f"  Description: {cls.docstring}")
                lines.append("")
        
        # Document functions
        if functions:
            lines.append("## Functions")
            for func in functions:
                params_str = ", ".join([f"{p.get('name', '')}: {p.get('type', 'Any')}" for p in func.parameters])
                lines.append(f"- {func.name}({params_str})")
                if func.return_type:
                    lines.append(f"  Returns: {func.return_type}")
                if func.docstring:
                    lines.append(f"  Description: {func.docstring}")
                if func.calls:
                    lines.append(f"  Calls: {', '.join(func.calls[:3])}")
                lines.append("")
        
        # Document variables
        if variables:
            lines.append("## Variables")
            for var in variables:
                lines.append(f"- {var.name}")
                if var.variable_type:
                    lines.append(f"  Type: {var.variable_type}")
                if var.is_constant:
                    lines.append("  Constant: true")
                lines.append("")
        
        # Document relationships
        if relationships:
            lines.append("## Relationships")
            for rel in relationships:
                lines.append(f"- {rel.source_entity_id} {rel.relationship_type.value} {rel.target_entity_id}")
                if rel.context:
                    lines.append(f"  Context: {rel.context}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _send_to_graphrag_via_document(self, document_content: str, domain: str) -> Dict[str, Any]:
        """Send code document to GraphRAG via document ingestion endpoint."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(document_content)
                temp_file_path = f.name
            
            try:
                # Send file to GraphRAG
                with open(temp_file_path, 'rb') as f:
                    files = {'files': ('code_document.txt', f, 'text/plain')}
                    data = {
                        'domain': domain,
                        'build_knowledge_graph': 'true'
                    }
                    
                    response = self.session.post(
                        f"{self.graphrag_api_url}/ingest-documents",
                        files=files,
                        data=data
                    )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    raise Exception(f"GraphRAG API error: {response.status_code} - {response.text}")
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            raise Exception(f"Failed to send to GraphRAG: {str(e)}")
    
    def _find_code_files(self, directory_path: str, recursive: bool = True) -> List[str]:
        """Find all code files in the directory."""
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}
        code_files = []
        
        if recursive:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    if Path(file).suffix in code_extensions:
                        code_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory_path):
                if Path(file).suffix in code_extensions:
                    code_files.append(os.path.join(directory_path, file))
        
        return code_files
    
    def search_graphrag_for_code(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search GraphRAG for code-related information.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results from GraphRAG
        """
        try:
            # Use GraphRAG's search endpoint
            response = self.session.post(
                f"{self.graphrag_api_url}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "domain": "code"  # Filter for code domain
                }
            )
            
            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                raise Exception(f"GraphRAG search error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error searching GraphRAG: {e}")
            return []
    
    def get_graphrag_statistics(self) -> Dict[str, Any]:
        """Get statistics from GraphRAG about code entities."""
        try:
            response = self.session.get(f"{self.graphrag_api_url}/statistics")
            
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"GraphRAG statistics error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Error getting GraphRAG statistics: {e}")
            return {}


class CodeRAGToGraphRAGBridge:
    """Bridge class for seamless integration between Code RAG and GraphRAG."""
    
    def __init__(self, graphrag_api_url: str = "http://localhost:8000"):
        """Initialize the bridge."""
        self.integration = GraphRAGIntegration(graphrag_api_url)
        self.parser = PythonParser()
    
    def auto_detect_and_integrate(self, path: str, domain: str = "code") -> Dict[str, Any]:
        """
        Automatically detect if path is a file or directory and integrate with GraphRAG.
        
        Args:
            path: Path to file or directory
            domain: Domain for the entities
            
        Returns:
            Integration result
        """
        path_obj = Path(path)
        
        if path_obj.is_file():
            return self.integration.send_code_file_to_graphrag(path, domain)
        elif path_obj.is_dir():
            return self.integration.send_code_directory_to_graphrag(path, domain)
        else:
            return {
                "success": False,
                "error": f"Path does not exist: {path}",
                "entities_sent": 0,
                "relationships_sent": 0
            }
    
    def search_code_in_graphrag(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for code-related information in GraphRAG."""
        return self.integration.search_graphrag_for_code(query, top_k)
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get status of the integration and GraphRAG system."""
        try:
            # Check GraphRAG health
            health_response = self.integration.session.get(f"{self.integration.graphrag_api_url}/health")
            health_status = health_response.json() if health_response.status_code == 200 else None
            
            # Get statistics
            stats = self.integration.get_graphrag_statistics()
            
            return {
                "graphrag_healthy": health_status is not None,
                "graphrag_status": health_status,
                "statistics": stats,
                "integration_ready": health_status is not None
            }
            
        except Exception as e:
            return {
                "graphrag_healthy": False,
                "graphrag_status": None,
                "statistics": {},
                "integration_ready": False,
                "error": str(e)
            } 