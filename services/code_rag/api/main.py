"""
Main FastAPI application for Code RAG system.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import time
from pathlib import Path
import uuid

from ..parsers.python_parser import PythonParser
from ..search.search_engine import CodeSearchEngine, SearchContext, QueryIntent
from ..models.entities import AnyEntity


# Pydantic models for API
class SearchRequest(BaseModel):
    query: str
    language: Optional[str] = None
    top_k: int = 10
    threshold: float = 0.0
    context: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    search_time_ms: float
    suggestions: List[str]
    query_intent: str


class IngestionRequest(BaseModel):
    file_paths: List[str]
    project_name: str
    language: Optional[str] = None


class IngestionResponse(BaseModel):
    task_id: str
    status: str
    message: str


class AnalysisResponse(BaseModel):
    file_path: str
    language: str
    success: bool
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    parse_time_ms: float
    error: Optional[str] = None


# Initialize FastAPI app
app = FastAPI(
    title="Code RAG API",
    description="Intelligent code search and retrieval system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
search_engine = CodeSearchEngine()
python_parser = PythonParser()

# In-memory storage for tasks (in production, use Redis or database)
ingestion_tasks: Dict[str, Dict[str, Any]] = {}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Code RAG API is running",
        "version": "1.0.0",
        "endpoints": [
            "/docs",
            "/search",
            "/ingest",
            "/analyze",
            "/statistics"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "statistics": search_engine.get_statistics()
    }


@app.post("/search", response_model=SearchResponse)
async def search_code(request: SearchRequest):
    """Search for code entities."""
    try:
        # Build search context
        context = None
        if request.context:
            context = SearchContext(
                language=request.context.get("language"),
                file_path=request.context.get("file_path"),
                project_name=request.context.get("project_name"),
                user_preferences=request.context.get("user_preferences")
            )
        
        # Execute search
        search_response = search_engine.search(
            query=request.query,
            context=context,
            top_k=request.top_k,
            threshold=request.threshold
        )
        
        # Convert results to API format
        results = []
        for result in search_response.results:
            result_dict = {
                "entity": result.entity.to_dict(),
                "score": result.score,
                "explanation": result.explanation,
                "match_type": result.match_type,
                "metadata": result.metadata
            }
            results.append(result_dict)
        
        return SearchResponse(
            query=search_response.query,
            results=results,
            total_results=search_response.total_results,
            search_time_ms=search_response.search_time_ms,
            suggestions=search_response.suggestions,
            query_intent=search_response.query_intent.value
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_code(
    request: IngestionRequest,
    background_tasks: BackgroundTasks
):
    """Ingest code files for indexing."""
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        ingestion_tasks[task_id] = {
            "status": "started",
            "progress": 0,
            "total_files": len(request.file_paths),
            "processed_files": 0,
            "entities_found": 0,
            "error": None,
            "start_time": time.time()
        }
        
        # Start background processing
        background_tasks.add_task(
            process_ingestion,
            task_id,
            request.file_paths,
            request.project_name,
            request.language
        )
        
        return IngestionResponse(
            task_id=task_id,
            status="started",
            message=f"Started processing {len(request.file_paths)} files"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/upload", response_model=IngestionResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    project_name: str = "uploaded_project",
    background_tasks: BackgroundTasks = None
):
    """Upload and ingest code files."""
    try:
        # Save uploaded files to temporary directory
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for file in files:
            if file.filename:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                file_paths.append(file_path)
        
        # Create ingestion request
        request = IngestionRequest(
            file_paths=file_paths,
            project_name=project_name,
            language=None  # Auto-detect
        )
        
        # Process ingestion
        return await ingest_code(request, background_tasks)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/ingest/{task_id}")
async def get_ingestion_status(task_id: str):
    """Get the status of an ingestion task."""
    if task_id not in ingestion_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = ingestion_tasks[task_id]
    
    # Calculate progress
    if task["total_files"] > 0:
        progress = (task["processed_files"] / task["total_files"]) * 100
    else:
        progress = 0
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": progress,
        "processed_files": task["processed_files"],
        "total_files": task["total_files"],
        "entities_found": task["entities_found"],
        "error": task["error"],
        "elapsed_time": time.time() - task["start_time"]
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_file(file_path: str):
    """Analyze a single code file."""
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine parser based on file extension
        if python_parser.can_parse(file_path):
            parser = python_parser
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Parse the file
        result = parser.parse_file(file_path)
        
        return AnalysisResponse(
            file_path=result.file_path,
            language=result.language,
            success=result.success,
            entities=[entity.to_dict() for entity in result.entities],
            relationships=[rel.to_dict() for rel in result.relationships],
            parse_time_ms=result.parse_time_ms,
            error=result.error
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/statistics")
async def get_statistics():
    """Get system statistics."""
    stats = search_engine.get_statistics()
    
    return {
        "search_engine": stats,
        "active_tasks": len([t for t in ingestion_tasks.values() if t["status"] == "processing"]),
        "completed_tasks": len([t for t in ingestion_tasks.values() if t["status"] == "completed"]),
        "failed_tasks": len([t for t in ingestion_tasks.values() if t["status"] == "failed"])
    }


@app.delete("/index")
async def clear_index():
    """Clear the search index."""
    try:
        search_engine.clear_index()
        return {"message": "Index cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear index: {str(e)}")


# Background task functions
async def process_ingestion(task_id: str, file_paths: List[str], 
                          project_name: str, language: Optional[str]):
    """Background task to process file ingestion."""
    task = ingestion_tasks[task_id]
    task["status"] = "processing"
    
    try:
        all_entities = []
        
        for i, file_path in enumerate(file_paths):
            try:
                # Determine parser
                if language == "python" or python_parser.can_parse(file_path):
                    parser = python_parser
                else:
                    # Skip unsupported files for now
                    continue
                
                # Parse file
                result = parser.parse_file(file_path)
                
                if result.success:
                    all_entities.extend(result.entities)
                    task["entities_found"] += len(result.entities)
                
                task["processed_files"] += 1
                
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
        
        # Add entities to search engine
        if all_entities:
            search_engine.add_entities(all_entities)
        
        task["status"] = "completed"
        
    except Exception as e:
        task["status"] = "failed"
        task["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 