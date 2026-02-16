from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging
from datetime import datetime
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NER API",
    description="Named Entity Recognition API using distilbert-NER model",
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

# Pydantic models
class NERRequest(BaseModel):
    text: str
    return_offsets: bool = True
    return_scores: bool = True

class NERResponse(BaseModel):
    entities: List[Dict[str, Any]]
    text: str
    processing_time: float
    model_info: Dict[str, Any]

class BatchNERRequest(BaseModel):
    texts: List[str]
    return_offsets: bool = True
    return_scores: bool = True

class BatchNERResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float
    model_info: Dict[str, Any]

# Global variables for model and pipeline
ner_pipeline = None
model_info = {}

def load_ner_model():
    """Load the distilbert-NER model and create pipeline."""
    global ner_pipeline, model_info
    
    try:
        logger.info("Loading distilbert-NER model...")
        
        # Use pipeline approach for simplicity and reliability
        model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
        ner_pipeline = pipeline("token-classification", model=model_name)
        
        # Store model info
        model_info = {
            "model_name": model_name,
            "model_type": "bert-large",
            "task": "token_classification",
            "entity_types": ["LOC", "MISC", "ORG", "PER"],
            "model_size": "~340M parameters (estimated)"
        }
        
        logger.info(f"✅ NER model loaded successfully: {model_info['model_size']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load NER model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the NER model on startup."""
    success = load_ner_model()
    if not success:
        logger.error("Failed to initialize NER model. Service may not function properly.")

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "NER API is running",
        "model": "dslim/distilbert-NER",
        "endpoints": {
            "/ner": "Single text NER",
            "/ner/batch": "Batch NER processing",
            "/health": "Health check",
            "/model-info": "Model information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = "healthy" if ner_pipeline is not None else "unhealthy"
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_loaded": ner_pipeline is not None
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_info": model_info,
        "pipeline_ready": ner_pipeline is not None
    }

@app.post("/ner", response_model=NERResponse)
async def extract_entities(request: NERRequest):
    """Extract named entities from a single text."""
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="NER model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Run NER pipeline
        entities = ner_pipeline(
            request.text
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Format entities for response
        formatted_entities = []
        for entity in entities:
            # Convert numpy types to native Python types
            score = entity.get("score", 0.0)
            if isinstance(score, np.floating):
                score = float(score)
            
            formatted_entity = {
                "entity": entity["entity"],
                "word": entity["word"],
                "score": score,
                "start": entity.get("start", 0),
                "end": entity.get("end", 0)
            }
            formatted_entities.append(formatted_entity)
        
        return NERResponse(
            entities=formatted_entities,
            text=request.text,
            processing_time=processing_time,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Error processing NER request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.post("/ner/batch", response_model=BatchNERResponse)
async def extract_entities_batch(request: BatchNERRequest):
    """Extract named entities from multiple texts in batch."""
    if ner_pipeline is None:
        raise HTTPException(status_code=503, detail="NER model not loaded")
    
    try:
        start_time = datetime.now()
        results = []
        
        for i, text in enumerate(request.texts):
            try:
                # Run NER pipeline for each text
                entities = ner_pipeline(
                    text
                )
                
                # Format entities
                formatted_entities = []
                for entity in entities:
                    # Convert numpy types to native Python types
                    score = entity.get("score", 0.0)
                    if isinstance(score, np.floating):
                        score = float(score)
                    
                    formatted_entity = {
                        "entity": entity["entity"],
                        "word": entity["word"],
                        "score": score,
                        "start": entity.get("start", 0),
                        "end": entity.get("end", 0)
                    }
                    formatted_entities.append(formatted_entity)
                
                results.append({
                    "text": text,
                    "entities": formatted_entities,
                    "entity_count": len(formatted_entities)
                })
                
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                results.append({
                    "text": text,
                    "entities": [],
                    "entity_count": 0,
                    "error": str(e)
                })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchNERResponse(
            results=results,
            processing_time=processing_time,
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Error processing batch NER request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/ner/entities")
async def get_entity_types():
    """Get the list of entity types that the model can recognize."""
    return {
        "entity_types": ["LOC", "MISC", "ORG", "PER"],
        "descriptions": {
            "LOC": "Location (cities, countries, etc.)",
            "MISC": "Miscellaneous entities",
            "ORG": "Organization",
            "PER": "Person"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 