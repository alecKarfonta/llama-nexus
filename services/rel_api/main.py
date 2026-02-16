from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import os
import numpy as np
import traceback
import torch

# GLiNER imports
from gliner import GLiNER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Relationship Extraction API",
    description="Relationship extraction API using GLiNER multitask model with UTCA framework",
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
class RelationRequest(BaseModel):
    text: str
    relations: List[Dict[str, Any]]
    entity_labels: Optional[List[str]] = None
    threshold: float = 0.5

class RelationResponse(BaseModel):
    text: str
    relations: List[Dict[str, Any]]
    processing_time: float
    model_info: Dict[str, Any]

class BatchRelationRequest(BaseModel):
    texts: List[str]
    relations: List[Dict[str, Any]]
    entity_labels: Optional[List[str]] = None
    threshold: float = 0.5

class BatchRelationResponse(BaseModel):
    results: List[Dict[str, Any]]
    processing_time: float
    model_info: Dict[str, Any]

class EntityRequest(BaseModel):
    text: str
    labels: List[str]
    threshold: float = 0.5

# Global variables for model
DEFAULT_ENTITY_LABELS = [
    "person", "organisation", "location", "date", "component", "system", "symptom", "solution", "maintenance", "specification", "requirement", "safety", "time", "founder", "position"
]

DEFAULT_RELATION_TYPES = [
    {
        "relation": "works for",
        "pairs_filter": [("person", "organisation")],
        "distance_threshold": 100
    },
    {
        "relation": "located in", 
        "pairs_filter": [("person", "location"), ("organisation", "location")],
        "distance_threshold": 100
    },
    {
        "relation": "founded",
        "pairs_filter": [("person", "organisation")],
        "distance_threshold": 100
    },
    {
        "relation": "part of",
        "pairs_filter": [("organisation", "organisation"), ("component", "system")],
        "distance_threshold": 100
    },
    {
        "relation": "requires",
        "pairs_filter": [("component", "component"), ("procedure", "component")],
        "distance_threshold": 100
    },
    {
        "relation": "causes",
        "pairs_filter": [("component", "symptom"), ("system", "symptom")],
        "distance_threshold": 100
    },
    {
        "relation": "fixes",
        "pairs_filter": [("solution", "symptom"), ("procedure", "symptom")],
        "distance_threshold": 100
    },
    {
        "relation": "scheduled for",
        "pairs_filter": [("maintenance", "time"), ("procedure", "time")],
        "distance_threshold": 100
    },
    {
        "relation": "specifies",
        "pairs_filter": [("specification", "component"), ("requirement", "component")],
        "distance_threshold": 100
    }
]

gliner_model = None
model_info = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gliner_model():
    """Load the GLiNER model with GPU support."""
    global gliner_model, model_info
    
    try:
        logger.info("Loading GLiNER model...")
        logger.info(f"Device detected: {device}")
        
        # Initialize GLiNER model
        gliner_model = GLiNER.from_pretrained("knowledgator/gliner-multitask-large-v0.5")
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            logger.info(f"Moving GLiNER model to GPU: {torch.cuda.get_device_name(0)}")
            gliner_model = gliner_model.to(device)
            # Set model to evaluation mode for inference
            gliner_model.eval()
        else:
            logger.info("CUDA not available, using CPU")
        
        # Store model info
        model_info = {
            "model_name": "knowledgator/gliner-multitask-large-v0.5",
            "model_type": "gliner-multitask",
            "framework": "GLiNER",
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "capabilities": [
                "Named Entity Recognition (NER)",
                "Relation Extraction",
                "Multi-task Information Extraction"
            ],
            "supported_tasks": "Entity and Relationship Extraction",
            "pipeline_available": True,
            "default_entity_labels": DEFAULT_ENTITY_LABELS,
            "default_relation_types": DEFAULT_RELATION_TYPES
        }
        
        logger.info("✅ GLiNER model loaded successfully")
        if torch.cuda.is_available():
            logger.info(f"✅ Model running on GPU: {torch.cuda.get_device_name(0)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load GLiNER model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the GLiNER model on startup."""
    success = load_gliner_model()
    if not success:
        logger.error("Failed to initialize GLiNER model. Service may not function properly.")

@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Relationship Extraction API is running",
        "model": "knowledgator/gliner-multitask-large-v0.5",
        "framework": "GLiNER",
        "endpoints": {
            "/extract-relations": "Single text relation extraction",
            "/extract-relations/batch": "Batch relation extraction",
            "/extract-entities": "Entity extraction",
            "/health": "Health check",
            "/model-info": "Model information",
            "/capabilities": "API capabilities"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    status = "healthy" if gliner_model is not None else "unhealthy"
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_loaded": gliner_model is not None
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model."""
    if gliner_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_info": model_info,
        "model_ready": gliner_model is not None,
        "default_entity_labels": DEFAULT_ENTITY_LABELS,
        "default_relation_types": DEFAULT_RELATION_TYPES
    }

@app.get("/capabilities")
async def get_capabilities():
    """Get API capabilities and default entity labels."""
    return {
        "capabilities": [
            "Entity Extraction",
            "Relationship Extraction", 
            "Custom Entity Labels",
            "Custom Relation Types",
            "Batch Processing",
            "UTCA Framework"
        ],
        "default_entity_labels": DEFAULT_ENTITY_LABELS,
        "default_relation_types": DEFAULT_RELATION_TYPES
    }

def clean_entity_text(text: str) -> str:
    """Clean entity text by removing punctuation and normalizing."""
    import re
    
    # Remove leading/trailing punctuation
    text = text.strip()
    text = re.sub(r'^[^\w\s]+', '', text)  # Remove leading punctuation
    text = re.sub(r'[^\w\s]+$', '', text)  # Remove trailing punctuation
    
    # Remove common unwanted patterns
    text = re.sub(r'^\d+\.?\s*', '', text)  # Remove numbered lists
    text = re.sub(r'^[•\-*]\s*', '', text)  # Remove bullet points
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_valid_entity(text: str, entity_type: str, score: float) -> bool:
    """Check if an entity is valid based on various criteria."""
    
    # Clean the text
    cleaned_text = clean_entity_text(text)
    
    # Basic validation
    if not cleaned_text or len(cleaned_text) < 2:
        return False
    
    # Confidence threshold (higher for shorter entities)
    min_confidence = 0.7 if len(cleaned_text) <= 3 else 0.6
    if score < min_confidence:
        return False
    
    # Filter out common noise
    noise_patterns = [
        r'^\d+$',  # Just numbers
        r'^[a-z]$',  # Single letters
        r'^[A-Z]$',  # Single capital letters
        r'^\d+[a-zA-Z]$',  # Number + single letter
        r'^[a-zA-Z]\d+$',  # Letter + numbers
        r'^[^\w\s]+$',  # Only punctuation
        r'^(the|a|an|and|or|but|in|on|at|to|for|of|with|by|from|up|down|out|off|over|under|above|below|before|after|during|while|since|until|unless|if|then|else|when|where|why|how|what|which|who|whom|whose|that|this|these|those|it|its|they|them|their|we|us|our|you|your|he|him|his|she|her|hers|i|me|my|mine)$',  # Common words
    ]
    
    import re
    for pattern in noise_patterns:
        if re.match(pattern, cleaned_text.lower()):
            return False
    
    # Entity type specific validation
    if entity_type.lower() in ['person', 'organisation']:
        # Names should be at least 2 characters and not just numbers
        if len(cleaned_text) < 2 or cleaned_text.isdigit():
            return False
    
    return True

def _matches_pairs_filter(entity1, entity2, pairs_filter):
    """Check if entity pair matches the filter."""
    if not pairs_filter:
        return True
    
    entity1_type = entity1.get("label", "").lower()
    entity2_type = entity2.get("label", "").lower()
    
    for pair in pairs_filter:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            type1, type2 = pair
            # Convert both to lowercase for comparison
            type1_lower = type1.lower()
            type2_lower = type2.lower()
            
            if (entity1_type == type1_lower and entity2_type == type2_lower) or \
               (entity1_type == type2_lower and entity2_type == type1_lower):
                return True
    
    return False

@app.post("/extract-relations", response_model=RelationResponse)
async def extract_relations(request: RelationRequest):
    """Extract relationships from text using GLiNER."""
    if gliner_model is None:
        raise HTTPException(status_code=503, detail="GLiNER model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Use provided entity_labels or default
        entity_labels = request.entity_labels if request.entity_labels else DEFAULT_ENTITY_LABELS
        
        # Use provided relations or default
        relations = request.relations if request.relations else DEFAULT_RELATION_TYPES
        
        logger.info("Using GLiNER for entity and relationship extraction")
        
        # Extract entities first using GLiNER
        entities = gliner_model.predict_entities(request.text, entity_labels, threshold=request.threshold)
        
        # For now, we'll implement a simple relationship extraction based on entity proximity
        # In a full implementation, you would use a dedicated relationship extraction model
        processed_relations = []
        
        # Create relationships based on entity pairs and relation definitions
        for relation_def in relations:
            relation_type = relation_def.get("relation", "")
            pairs_filter = relation_def.get("pairs_filter", [])
            distance_threshold = relation_def.get("distance_threshold", 100)
            
            # Find entity pairs that match the filter
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    # Check if this pair matches the filter
                    if _matches_pairs_filter(entity1, entity2, pairs_filter):
                        # Calculate distance between entities
                        distance = abs(entity1.get("start", 0) - entity2.get("start", 0))
                        
                        if distance <= distance_threshold:
                            # Extract context around both entities
                            context_start = max(0, min(entity1.get("start", 0), entity2.get("start", 0)) - 50)
                            context_end = min(len(request.text), max(entity1.get("end", 0), entity2.get("end", 0)) + 50)
                            context = request.text[context_start:context_end]
                            
                            relation = {
                                "source": entity1.get("text", ""),
                                "target": entity2.get("text", ""),
                                "label": relation_type,
                                "score": min(entity1.get("score", 0.5), entity2.get("score", 0.5)),
                                "context": context,
                                "source_type": entity1.get("label", "entity"),
                                "target_type": entity2.get("label", "entity")
                            }
                            processed_relations.append(relation)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "text": request.text,
            "relations": processed_relations,
            "processing_time": processing_time,
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Error extracting relations: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error extracting relations: {str(e)}")

@app.post("/extract-relations/batch", response_model=BatchRelationResponse)
async def extract_relations_batch(request: BatchRelationRequest):
    """Extract relationships from multiple texts in batch."""
    if gliner_model is None:
        raise HTTPException(status_code=503, detail="GLiNER model not loaded")
    
    try:
        start_time = datetime.now()
        results = []
        
        # Use provided entity_labels or default
        entity_labels = request.entity_labels if request.entity_labels else DEFAULT_ENTITY_LABELS
        
        # Use provided relations or default
        relations = request.relations if request.relations else DEFAULT_RELATION_TYPES
        
        for text in request.texts:
            try:
                # Extract entities first using GLiNER
                entities = gliner_model.predict_entities(text, entity_labels, threshold=request.threshold)
                
                # Create relationships based on entity pairs and relation definitions
                processed_relations = []
                for relation_def in relations:
                    relation_type = relation_def.get("relation", "")
                    pairs_filter = relation_def.get("pairs_filter", [])
                    distance_threshold = relation_def.get("distance_threshold", 100)
                    
                    # Find entity pairs that match the filter
                    for i, entity1 in enumerate(entities):
                        for j, entity2 in enumerate(entities[i+1:], i+1):
                            # Check if this pair matches the filter
                            if _matches_pairs_filter(entity1, entity2, pairs_filter):
                                # Calculate distance between entities
                                distance = abs(entity1.get("start", 0) - entity2.get("start", 0))
                                
                                if distance <= distance_threshold:
                                    # Extract context around both entities
                                    context_start = max(0, min(entity1.get("start", 0), entity2.get("start", 0)) - 50)
                                    context_end = min(len(text), max(entity1.get("end", 0), entity2.get("end", 0)) + 50)
                                    context = text[context_start:context_end]
                                    
                                    relation = {
                                        "source": entity1.get("text", ""),
                                        "target": entity2.get("text", ""),
                                        "label": relation_type,
                                        "score": min(entity1.get("score", 0.5), entity2.get("score", 0.5)),
                                        "context": context,
                                        "source_type": entity1.get("label", "entity"),
                                        "target_type": entity2.get("label", "entity")
                                    }
                                    processed_relations.append(relation)
                
                results.append({
                    "text": text,
                    "relations": processed_relations,
                    "relation_count": len(processed_relations)
                })
                
            except Exception as e:
                logger.error(f"Error processing text in batch: {e}")
                results.append({
                    "text": text,
                    "relations": [],
                    "relation_count": 0,
                    "error": str(e)
                })

        end_time = datetime.now()
        
        return {
            "results": results,
            "processing_time": (end_time - start_time).total_seconds(),
            "model_info": model_info
        }

    except Exception as e:
        logger.error(f"Error during batch relation extraction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/extract-entities")
async def extract_entities(request: EntityRequest):
    """Extract entities from text using GLiNER."""
    if gliner_model is None:
        raise HTTPException(status_code=503, detail="GLiNER model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Use provided labels or default
        labels = request.labels if request.labels else DEFAULT_ENTITY_LABELS
        
        # Extract entities using GLiNER
        entities = gliner_model.predict_entities(request.text, labels, threshold=request.threshold)
        
        # Process the results
        processed_entities = []
        for entity in entities:
            cleaned_text = clean_entity_text(entity["text"])
            if is_valid_entity(cleaned_text, entity["label"], entity["score"]):
                processed_entity = {
                    "text": cleaned_text,
                    "label": entity["label"],
                    "score": float(entity["score"]) if isinstance(entity["score"], np.floating) else entity["score"],
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0)
                }
                processed_entities.append(processed_entity)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "text": request.text,
            "entities": processed_entities,
            "entity_count": len(processed_entities),
            "processing_time": processing_time,
            "model_info": model_info
        }
        
    except Exception as e:
        logger.error(f"Error extracting entities: {e}")
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 