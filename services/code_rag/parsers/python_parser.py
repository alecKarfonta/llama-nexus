"""
Python code parser for Code RAG system.
"""

import ast
import time
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from .base_parser import BaseCodeParser, ParseError
from ..models.entities import (
    ParseResult, FunctionEntity, ClassEntity, VariableEntity, 
    ModuleEntity, RelationshipEntity, EntityType, Visibility, RelationshipType
)


class PythonParser(BaseCodeParser):
    """Parser for Python code files."""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.py', '.pyw']
        self.language_name = "python"
    
    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a Python file and extract all entities and relationships."""
        start_time = time.time()
        
        try:
            # Read source code
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST
            try:
                ast_tree = ast.parse(source_code, filename=file_path)
            except SyntaxError as e:
                return ParseResult(
                    file_path=file_path,
                    language=self.language_name,
                    success=False,
                    error=f"Syntax error: {str(e)}",
                    parse_time_ms=(time.time() - start_time) * 1000
                )
            
            # Extract entities
            entities = []
            
            # Extract module entity
            module_entity = self._extract_module_entity(file_path, source_code, ast_tree)
            entities.append(module_entity)
            
            # Extract functions
            functions = self.extract_functions(ast_tree, source_code, file_path)
            entities.extend(functions)
            
            # Extract classes
            classes = self.extract_classes(ast_tree, source_code, file_path)
            entities.extend(classes)
            
            # Extract global variables
            variables = self.extract_variables(ast_tree, source_code, file_path)
            entities.extend(variables)
            
            # Extract relationships
            relationships = self.extract_relationships(entities, ast_tree, source_code)
            
            parse_time = (time.time() - start_time) * 1000
            
            return ParseResult(
                file_path=file_path,
                language=self.language_name,
                success=True,
                entities=entities,
                relationships=relationships,
                ast_tree=ast_tree,
                source_code=source_code,
                parse_time_ms=parse_time
            )
            
        except Exception as e:
            return ParseResult(
                file_path=file_path,
                language=self.language_name,
                success=False,
                error=str(e),
                parse_time_ms=(time.time() - start_time) * 1000
            )
    
    def extract_functions(self, ast_tree: ast.AST, source_code: str, file_path: str) -> List[FunctionEntity]:
        """Extract function entities from Python AST."""
        functions = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_entity = self._create_function_entity(node, source_code, file_path)
                functions.append(func_entity)
        
        return functions
    
    def extract_classes(self, ast_tree: ast.AST, source_code: str, file_path: str) -> List[ClassEntity]:
        """Extract class entities from Python AST."""
        classes = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.ClassDef):
                class_entity = self._create_class_entity(node, source_code, file_path)
                classes.append(class_entity)
        
        return classes
    
    def extract_variables(self, ast_tree: ast.AST, source_code: str, file_path: str) -> List[VariableEntity]:
        """Extract global variable entities from Python AST."""
        variables = []
        
        # Only extract module-level assignments
        for node in ast_tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_entity = self._create_variable_entity(target, node, source_code, file_path)
                        variables.append(var_entity)
        
        return variables
    
    def extract_imports(self, ast_tree: ast.AST, source_code: str, file_path: str) -> List[str]:
        """Extract import statements from Python AST."""
        imports = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == "*":
                        imports.append(f"{module}.*")
                    else:
                        imports.append(f"{module}.{alias.name}")
        
        return imports
    
    def _create_function_entity(self, node: ast.FunctionDef, source_code: str, file_path: str) -> FunctionEntity:
        """Create a FunctionEntity from AST node."""
        # Extract parameters
        parameters = self._extract_parameters(node.args)
        
        # Extract return type
        return_type = self._extract_return_type(node)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract decorators
        decorators = [self._get_decorator_name(d) for d in node.decorator_list]
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        
        # Determine visibility
        visibility = getattr(Visibility, self._determine_visibility(node.name).upper())
        
        # Extract function calls
        calls = self._extract_function_calls(node)
        
        # Check if it's a method (inside a class)
        is_method = self._is_method(node)
        class_name = self._get_containing_class_name(node) if is_method else None
        
        return FunctionEntity(
            name=node.name,
            entity_type=EntityType.FUNCTION,
            file_path=file_path,
            line_start=node.lineno,
            line_end=getattr(node, 'end_lineno', node.lineno),
            language=self.language_name,
            visibility=visibility,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            decorators=decorators,
            complexity=complexity,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            class_name=class_name,
            calls=calls
        )
    
    def _create_class_entity(self, node: ast.ClassDef, source_code: str, file_path: str) -> ClassEntity:
        """Create a ClassEntity from AST node."""
        # Extract base classes
        base_classes = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_classes.append(self._get_attribute_name(base))
        
        # Extract methods and properties
        methods = []
        properties = []
        
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(child.name)
            elif isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        properties.append(target.id)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Check if abstract
        is_abstract = any(
            decorator.id == 'abstractmethod' if isinstance(decorator, ast.Name) else False
            for decorator in getattr(node, 'decorator_list', [])
        )
        
        # Determine visibility
        visibility = getattr(Visibility, self._determine_visibility(node.name).upper())
        
        return ClassEntity(
            name=node.name,
            entity_type=EntityType.CLASS,
            file_path=file_path,
            line_start=node.lineno,
            line_end=getattr(node, 'end_lineno', node.lineno),
            language=self.language_name,
            visibility=visibility,
            base_classes=base_classes,
            methods=methods,
            properties=properties,
            is_abstract=is_abstract,
            docstring=docstring
        )
    
    def _create_variable_entity(self, target: ast.Name, assign_node: ast.Assign, 
                              source_code: str, file_path: str) -> VariableEntity:
        """Create a VariableEntity from AST nodes."""
        # Determine if it's a constant (all uppercase)
        is_constant = target.id.isupper()
        
        # Extract initial value (simplified)
        initial_value = None
        if isinstance(assign_node.value, ast.Constant):
            initial_value = str(assign_node.value.value)
        elif isinstance(assign_node.value, ast.Name):
            initial_value = assign_node.value.id
        
        # Determine visibility
        visibility = getattr(Visibility, self._determine_visibility(target.id).upper())
        
        return VariableEntity(
            name=target.id,
            entity_type=EntityType.VARIABLE,
            file_path=file_path,
            line_start=assign_node.lineno,
            line_end=getattr(assign_node, 'end_lineno', assign_node.lineno),
            language=self.language_name,
            visibility=visibility,
            scope="global",
            is_constant=is_constant,
            initial_value=initial_value
        )
    
    def _extract_module_entity(self, file_path: str, source_code: str, ast_tree: ast.AST) -> ModuleEntity:
        """Create a ModuleEntity for the file."""
        module_name = Path(file_path).stem
        
        # Extract imports
        imports = self.extract_imports(ast_tree, source_code, file_path)
        
        # Extract top-level functions and classes
        functions = []
        classes = []
        
        for node in ast_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
        
        # Extract module docstring
        docstring = ast.get_docstring(ast_tree)
        
        return ModuleEntity(
            name=module_name,
            entity_type=EntityType.MODULE,
            file_path=file_path,
            line_start=1,
            line_end=len(source_code.split('\n')),
            language=self.language_name,
            imports=imports,
            functions=functions,
            classes=classes,
            docstring=docstring
        )
    
    def _extract_parameters(self, args: ast.arguments) -> List[Dict[str, str]]:
        """Extract function parameters from AST arguments."""
        parameters = []
        
        # Regular arguments
        for i, arg in enumerate(args.args):
            param_info = {
                "name": arg.arg,
                "type": self._get_annotation_string(arg.annotation) if arg.annotation else None,
                "default": None,
                "kind": "positional"
            }
            
            # Check for default values
            defaults_offset = len(args.args) - len(args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                param_info["default"] = self._get_default_value_string(args.defaults[default_idx])
            
            parameters.append(param_info)
        
        # Keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            param_info = {
                "name": arg.arg,
                "type": self._get_annotation_string(arg.annotation) if arg.annotation else None,
                "default": self._get_default_value_string(args.kw_defaults[i]) if args.kw_defaults[i] else None,
                "kind": "keyword_only"
            }
            parameters.append(param_info)
        
        # *args
        if args.vararg:
            parameters.append({
                "name": args.vararg.arg,
                "type": self._get_annotation_string(args.vararg.annotation) if args.vararg.annotation else None,
                "default": None,
                "kind": "var_positional"
            })
        
        # **kwargs
        if args.kwarg:
            parameters.append({
                "name": args.kwarg.arg,
                "type": self._get_annotation_string(args.kwarg.annotation) if args.kwarg.annotation else None,
                "default": None,
                "kind": "var_keyword"
            })
        
        return parameters
    
    def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation from function."""
        if node.returns:
            return self._get_annotation_string(node.returns)
        return None
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Get decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return self._get_attribute_name(decorator)
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return self._get_attribute_name(decorator.func)
        return "unknown"
    
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'module.Class')."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return '.'.join(reversed(parts))
    
    def _get_annotation_string(self, annotation: ast.expr) -> str:
        """Convert type annotation to string."""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            return self._get_attribute_name(annotation)
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        else:
            return "Any"
    
    def _get_default_value_string(self, default: ast.expr) -> str:
        """Convert default value to string."""
        if isinstance(default, ast.Constant):
            return repr(default.value)
        elif isinstance(default, ast.Name):
            return default.id
        else:
            return "..."
    
    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls within a function."""
        calls = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(self._get_attribute_name(child.func))
        
        return list(calls)
    
    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if function is a method (inside a class)."""
        # This is a simplified check - in practice, we'd need to track the AST hierarchy
        return len(node.args.args) > 0 and node.args.args[0].arg in ('self', 'cls')
    
    def _get_containing_class_name(self, node: ast.FunctionDef) -> Optional[str]:
        """Get the name of the containing class (simplified implementation)."""
        # This would need proper AST hierarchy tracking in a full implementation
        return None
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity 