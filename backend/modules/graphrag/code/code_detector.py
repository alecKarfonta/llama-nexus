"""
Code detection and routing module for GraphRAG.
"""

import os
import tempfile
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import mimetypes
import magic


class CodeLanguage(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    CSHARP = "csharp"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    R = "r"
    MATLAB = "matlab"
    UNKNOWN = "unknown"


class CodeDetector:
    """Detects code files and determines their language."""
    
    def __init__(self):
        """Initialize the code detector."""
        self.code_extensions = {
            '.py': CodeLanguage.PYTHON,
            '.pyw': CodeLanguage.PYTHON,
            '.js': CodeLanguage.JAVASCRIPT,
            '.jsx': CodeLanguage.JAVASCRIPT,
            '.ts': CodeLanguage.TYPESCRIPT,
            '.tsx': CodeLanguage.TYPESCRIPT,
            '.java': CodeLanguage.JAVA,
            '.cpp': CodeLanguage.CPP,
            '.cc': CodeLanguage.CPP,
            '.cxx': CodeLanguage.CPP,
            '.c': CodeLanguage.C,
            '.go': CodeLanguage.GO,
            '.rs': CodeLanguage.RUST,
            '.php': CodeLanguage.PHP,
            '.rb': CodeLanguage.RUBY,
            '.cs': CodeLanguage.CSHARP,
            '.swift': CodeLanguage.SWIFT,
            '.kt': CodeLanguage.KOTLIN,
            '.scala': CodeLanguage.SCALA,
            '.r': CodeLanguage.R,
            '.m': CodeLanguage.MATLAB
        }
        
        # Initialize python-magic for MIME type detection
        try:
            self.magic = magic.Magic(mime=True)
        except ImportError:
            print("Warning: python-magic not available, using fallback detection")
            self.magic = None
    
    def detect_code_file(self, file_path: str) -> Tuple[bool, Optional[CodeLanguage]]:
        """
        Detect if a file is a code file and determine its language.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (is_code_file, language)
        """
        try:
            # Check file extension first
            file_ext = Path(file_path).suffix.lower()
            if file_ext in self.code_extensions:
                return True, self.code_extensions[file_ext]
            
            # If no extension match, try MIME type detection
            if self.magic:
                try:
                    mime_type = self.magic.from_file(file_path)
                    return self._detect_from_mime_type(mime_type)
                except Exception:
                    pass
            
            # Fallback: check file content for code patterns
            return self._detect_from_content(file_path)
            
        except Exception as e:
            print(f"Error detecting code file {file_path}: {e}")
            return False, None
    
    def _detect_from_mime_type(self, mime_type: str) -> Tuple[bool, Optional[CodeLanguage]]:
        """Detect code language from MIME type."""
        mime_to_language = {
            'text/x-python': CodeLanguage.PYTHON,
            'text/javascript': CodeLanguage.JAVASCRIPT,
            'application/javascript': CodeLanguage.JAVASCRIPT,
            'text/x-java-source': CodeLanguage.JAVA,
            'text/x-c++src': CodeLanguage.CPP,
            'text/x-csrc': CodeLanguage.C,
            'text/x-go': CodeLanguage.GO,
            'text/x-rust': CodeLanguage.RUST,
            'text/x-php': CodeLanguage.PHP,
            'text/x-ruby': CodeLanguage.RUBY,
            'text/x-csharp': CodeLanguage.CSHARP,
            'text/x-swift': CodeLanguage.SWIFT,
            'text/x-kotlin': CodeLanguage.KOTLIN,
            'text/x-scala': CodeLanguage.SCALA,
            'text/x-r': CodeLanguage.R,
            'text/x-matlab': CodeLanguage.MATLAB
        }
        
        language = mime_to_language.get(mime_type)
        return language is not None, language
    
    def _detect_from_content(self, file_path: str) -> Tuple[bool, Optional[CodeLanguage]]:
        """Detect code language from file content patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1024)  # Read first 1KB
                
            # Python patterns
            if any(pattern in content for pattern in ['import ', 'def ', 'class ', 'from ', 'if __name__']):
                return True, CodeLanguage.PYTHON
            
            # JavaScript patterns
            if any(pattern in content for pattern in ['function ', 'var ', 'let ', 'const ', 'console.log', 'export ']):
                return True, CodeLanguage.JAVASCRIPT
            
            # Java patterns
            if any(pattern in content for pattern in ['public class', 'import java', 'package ', 'public static void main']):
                return True, CodeLanguage.JAVA
            
            # C++ patterns
            if any(pattern in content for pattern in ['#include <iostream>', '#include <vector>', 'using namespace std', 'std::']):
                return True, CodeLanguage.CPP
            
            # C patterns
            if any(pattern in content for pattern in ['#include <stdio.h>', '#include <stdlib.h>', 'int main(', 'printf(']):
                return True, CodeLanguage.C
            
            # Go patterns
            if any(pattern in content for pattern in ['package main', 'import "fmt"', 'func main(', 'fmt.Println']):
                return True, CodeLanguage.GO
            
            # Rust patterns
            if any(pattern in content for pattern in ['fn main(', 'use std::', 'println!', 'let mut ']):
                return True, CodeLanguage.RUST
            
            # PHP patterns
            if any(pattern in content for pattern in ['<?php', 'function ', '$', 'echo ']):
                return True, CodeLanguage.PHP
            
            # Ruby patterns
            if any(pattern in content for pattern in ['def ', 'puts ', 'require ', 'class ']):
                return True, CodeLanguage.RUBY
            
            return False, None
            
        except Exception:
            return False, None
    
    def get_code_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a code file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        is_code, language = self.detect_code_file(file_path)
        
        if not is_code:
            return {
                "is_code": False,
                "language": None,
                "file_path": file_path,
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        
        # Get additional file information
        file_info = {
            "is_code": True,
            "language": language.value if language else None,
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "file_extension": Path(file_path).suffix.lower(),
            "file_name": Path(file_path).name
        }
        
        # Try to get line count
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for _ in f)
                file_info["line_count"] = line_count
        except Exception:
            file_info["line_count"] = 0
        
        return file_info


class CodeRAGRouter:
    """Routes code files to Code RAG for specialized processing."""
    
    def __init__(self, code_rag_api_url: str = "http://localhost:8003"):
        """Initialize the router with Code RAG API URL."""
        self.code_rag_api_url = code_rag_api_url
        self.detector = CodeDetector()
        self.session = requests.Session()
    
    def route_file_to_code_rag(self, file_path: str, project_name: str = None) -> Dict[str, Any]:
        """
        Route a file to Code RAG if it's a code file.
        
        Args:
            file_path: Path to the file
            project_name: Name of the project (defaults to file name)
            
        Returns:
            Routing result
        """
        # Detect if it's a code file
        file_info = self.detector.get_code_file_info(file_path)
        
        if not file_info["is_code"]:
            return {
                "routed": False,
                "reason": "Not a code file",
                "file_info": file_info
            }
        
        # Route to Code RAG
        try:
            if project_name is None:
                project_name = Path(file_path).stem
            
            # Send to Code RAG API
            response = self.session.post(
                f"{self.code_rag_api_url}/analyze",
                json={
                    "file_path": file_path,
                    "project_name": project_name
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "routed": True,
                    "reason": f"Successfully routed to Code RAG ({file_info['language']})",
                    "file_info": file_info,
                    "code_rag_result": result
                }
            else:
                return {
                    "routed": False,
                    "reason": f"Code RAG API error: {response.status_code}",
                    "file_info": file_info,
                    "error": response.text
                }
                
        except Exception as e:
            return {
                "routed": False,
                "reason": f"Failed to route to Code RAG: {str(e)}",
                "file_info": file_info,
                "error": str(e)
            }
    
    def route_directory_to_code_rag(self, directory_path: str, project_name: str = None) -> Dict[str, Any]:
        """
        Route all code files in a directory to Code RAG.
        
        Args:
            directory_path: Path to the directory
            project_name: Name of the project (defaults to directory name)
            
        Returns:
            Routing result with summary
        """
        if project_name is None:
            project_name = Path(directory_path).name
        
        # Find all files in directory
        all_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                all_files.append(os.path.join(root, file))
        
        # Detect code files
        code_files = []
        non_code_files = []
        
        for file_path in all_files:
            file_info = self.detector.get_code_file_info(file_path)
            if file_info["is_code"]:
                code_files.append(file_path)
            else:
                non_code_files.append(file_path)
        
        if not code_files:
            return {
                "routed": False,
                "reason": "No code files found in directory",
                "total_files": len(all_files),
                "code_files": 0,
                "non_code_files": len(non_code_files)
            }
        
        # Route code files to Code RAG
        successful_routes = 0
        failed_routes = 0
        route_results = []
        
        for code_file in code_files:
            result = self.route_file_to_code_rag(code_file, project_name)
            route_results.append(result)
            
            if result["routed"]:
                successful_routes += 1
            else:
                failed_routes += 1
        
        return {
            "routed": successful_routes > 0,
            "reason": f"Routed {successful_routes}/{len(code_files)} code files to Code RAG",
            "total_files": len(all_files),
            "code_files": len(code_files),
            "non_code_files": len(non_code_files),
            "successful_routes": successful_routes,
            "failed_routes": failed_routes,
            "route_results": route_results
        }
    
    def check_code_rag_health(self) -> Dict[str, Any]:
        """Check if Code RAG is available and healthy."""
        try:
            response = self.session.get(f"{self.code_rag_api_url}/health", timeout=5)
            
            if response.status_code == 200:
                return {
                    "available": True,
                    "status": "healthy",
                    "response": response.json()
                }
            else:
                return {
                    "available": False,
                    "status": f"HTTP {response.status_code}",
                    "response": response.text
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "available": False,
                "status": "connection_error",
                "response": "Code RAG service not reachable"
            }
        except Exception as e:
            return {
                "available": False,
                "status": "error",
                "response": str(e)
            }


class HybridDocumentProcessor:
    """Processes documents with hybrid approach: text to GraphRAG, code to Code RAG."""
    
    def __init__(self, graphrag_api_url: str = "http://localhost:8000", 
                 code_rag_api_url: str = "http://localhost:8003"):
        """Initialize the hybrid processor."""
        self.graphrag_api_url = graphrag_api_url
        self.code_rag_api_url = code_rag_api_url
        self.detector = CodeDetector()
        self.router = CodeRAGRouter(code_rag_api_url)
        self.session = requests.Session()
    
    def process_file_hybrid(self, file_path: str, domain: str = "general") -> Dict[str, Any]:
        """
        Process a file using hybrid approach: route code files to Code RAG, others to GraphRAG.
        
        Args:
            file_path: Path to the file
            domain: Domain for processing
            
        Returns:
            Processing result
        """
        file_info = self.detector.get_code_file_info(file_path)
        
        if file_info["is_code"]:
            # Route to Code RAG
            code_rag_result = self.router.route_file_to_code_rag(file_path)
            
            # Also send to GraphRAG for unified search
            graphrag_result = self._send_to_graphrag(file_path, domain)
            
            return {
                "file_type": "code",
                "language": file_info["language"],
                "code_rag_processing": code_rag_result,
                "graphrag_processing": graphrag_result,
                "hybrid_processing": True
            }
        else:
            # Send to GraphRAG only
            graphrag_result = self._send_to_graphrag(file_path, domain)
            
            return {
                "file_type": "document",
                "language": None,
                "code_rag_processing": None,
                "graphrag_processing": graphrag_result,
                "hybrid_processing": False
            }
    
    def _send_to_graphrag(self, file_path: str, domain: str) -> Dict[str, Any]:
        """Send file to GraphRAG for processing."""
        try:
            with open(file_path, 'rb') as f:
                files = {'files': (Path(file_path).name, f, 'application/octet-stream')}
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
                return {
                    "success": True,
                    "response": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"GraphRAG API error: {response.status_code}",
                    "response": response.text
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of both GraphRAG and Code RAG systems."""
        # Check GraphRAG health
        try:
            graphrag_response = self.session.get(f"{self.graphrag_api_url}/health", timeout=5)
            graphrag_healthy = graphrag_response.status_code == 200
        except:
            graphrag_healthy = False
        
        # Check Code RAG health
        code_rag_health = self.router.check_code_rag_health()
        
        return {
            "graphrag_healthy": graphrag_healthy,
            "code_rag_healthy": code_rag_health["available"],
            "hybrid_available": graphrag_healthy and code_rag_health["available"],
            "code_rag_status": code_rag_health
        } 