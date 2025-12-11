"""
RAG Node Executors - Document and retrieval operations
"""

from typing import Dict, Any, List
import httpx
import os
from pathlib import Path
from .base import NodeExecutor, ExecutionContext


class DocumentLoaderExecutor(NodeExecutor):
    """Load documents from various sources"""
    
    node_type = "document_loader"
    display_name = "Document Loader"
    category = "rag"
    description = "Load documents from files, URLs, or directories"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        source = inputs.get("source", "")
        source_type = self.get_config_value("sourceType", "file")
        
        context.log(f"Loading documents from {source_type}: {source}")
        
        documents = []
        
        if source_type == "file":
            # Load single file
            path = Path(source)
            if path.exists():
                content = path.read_text(errors="ignore")
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": str(path),
                        "filename": path.name,
                        "type": path.suffix,
                    }
                })
        
        elif source_type == "directory":
            # Load all files from directory
            path = Path(source)
            recursive = self.get_config_value("recursive", True)
            file_types = self.get_config_value("fileTypes", [".txt", ".md", ".pdf"])
            
            if path.exists() and path.is_dir():
                pattern = "**/*" if recursive else "*"
                for file_path in path.glob(pattern):
                    if file_path.is_file() and file_path.suffix in file_types:
                        try:
                            content = file_path.read_text(errors="ignore")
                            documents.append({
                                "content": content,
                                "metadata": {
                                    "source": str(file_path),
                                    "filename": file_path.name,
                                    "type": file_path.suffix,
                                }
                            })
                        except Exception as e:
                            context.log(f"Error loading {file_path}: {e}", level="warning")
        
        elif source_type == "url":
            # Fetch from URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(source)
                response.raise_for_status()
                content = response.text
                
                documents.append({
                    "content": content,
                    "metadata": {
                        "source": source,
                        "type": "url",
                    }
                })
        
        context.log(f"Loaded {len(documents)} documents")
        return {"documents": documents}


class ChunkerExecutor(NodeExecutor):
    """Split documents into chunks"""
    
    node_type = "chunker"
    display_name = "Text Chunker"
    category = "rag"
    description = "Split documents into smaller chunks"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        documents = inputs.get("documents", [])
        
        strategy = self.get_config_value("strategy", "fixed")
        chunk_size = self.get_config_value("chunkSize", 512)
        overlap = self.get_config_value("overlap", 50)
        
        context.log(f"Chunking {len(documents)} documents with {strategy} strategy")
        
        chunks = []
        
        for doc in documents:
            content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
            metadata = doc.get("metadata", {}) if isinstance(doc, dict) else {}
            
            if strategy == "fixed":
                # Fixed-size chunking
                doc_chunks = self._fixed_chunk(content, chunk_size, overlap)
            elif strategy == "recursive":
                # Recursive character text splitting
                doc_chunks = self._recursive_chunk(content, chunk_size, overlap)
            else:
                # Default to fixed
                doc_chunks = self._fixed_chunk(content, chunk_size, overlap)
            
            for i, chunk_text in enumerate(doc_chunks):
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "chunk_count": len(doc_chunks),
                    }
                })
        
        context.log(f"Created {len(chunks)} chunks")
        return {"chunks": chunks}
    
    def _fixed_chunk(self, text: str, size: int, overlap: int) -> List[str]:
        """Split text into fixed-size chunks with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - overlap
            if start >= len(text):
                break
        return chunks
    
    def _recursive_chunk(self, text: str, size: int, overlap: int) -> List[str]:
        """Recursively split text on separators"""
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._split_recursive(text, separators, size, overlap)
    
    def _split_recursive(self, text: str, separators: List[str], size: int, overlap: int) -> List[str]:
        """Helper for recursive splitting"""
        if len(text) <= size:
            return [text] if text.strip() else []
        
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current = ""
                
                for part in parts:
                    if len(current) + len(part) + len(sep) <= size:
                        current += (sep if current else "") + part
                    else:
                        if current:
                            chunks.append(current)
                        current = part
                
                if current:
                    chunks.append(current)
                
                return chunks
        
        # No separator worked, use fixed chunking
        return self._fixed_chunk(text, size, overlap)


class RetrieverExecutor(NodeExecutor):
    """Semantic search / retrieval"""
    
    node_type = "retriever"
    display_name = "Semantic Search"
    category = "rag"
    description = "Retrieve relevant documents via semantic search"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        query = inputs.get("query", "")
        
        if not query:
            return {"documents": [], "scores": []}
        
        collection = self.get_config_value("collection", "default")
        k = self.get_config_value("k", 5)
        threshold = self.get_config_value("threshold")
        
        context.log(f"Searching for: {query[:100]}...")
        
        # Call RAG API
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{backend_url}/api/v1/rag/search",
                json={
                    "query": query,
                    "collection": collection,
                    "limit": k,
                    "threshold": threshold,
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                documents = [r.get("content", "") for r in results]
                scores = [r.get("score", 0) for r in results]
                
                context.log(f"Found {len(documents)} relevant documents")
                return {"documents": documents, "scores": scores}
            else:
                context.log(f"Search failed: {response.status_code}", level="warning")
                return {"documents": [], "scores": []}


class VectorStoreExecutor(NodeExecutor):
    """Store or query vector database"""
    
    node_type = "vector_store"
    display_name = "Vector Store"
    category = "rag"
    description = "Store or query vectors in database"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        data = inputs.get("data")
        operation = self.get_config_value("operation", "query")
        collection = self.get_config_value("collection", "default")
        
        context.log(f"Vector store operation: {operation}")
        
        backend_url = os.environ.get("BACKEND_API_URL", "http://localhost:8700")
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            if operation == "upsert":
                # Insert or update vectors
                response = await client.post(
                    f"{backend_url}/api/v1/rag/vectors/upsert",
                    json={
                        "collection": collection,
                        "documents": data if isinstance(data, list) else [data],
                    }
                )
                response.raise_for_status()
                result = response.json()
                context.log(f"Upserted {result.get('count', 0)} vectors")
                return {"result": result}
            
            elif operation == "query":
                # Query vectors
                response = await client.post(
                    f"{backend_url}/api/v1/rag/vectors/query",
                    json={
                        "collection": collection,
                        "vector": data,
                        "limit": self.get_config_value("limit", 5),
                    }
                )
                response.raise_for_status()
                result = response.json()
                return {"result": result.get("results", [])}
            
            elif operation == "delete":
                # Delete vectors
                response = await client.post(
                    f"{backend_url}/api/v1/rag/vectors/delete",
                    json={
                        "collection": collection,
                        "ids": data if isinstance(data, list) else [data],
                    }
                )
                response.raise_for_status()
                result = response.json()
                context.log(f"Deleted {result.get('count', 0)} vectors")
                return {"result": result}
        
        return {"result": None}
