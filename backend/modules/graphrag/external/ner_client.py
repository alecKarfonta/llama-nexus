"""
NER API Client for integration with the main GraphRAG API
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class NERClient:
    """Client for communicating with the NER API service."""
    
    def __init__(self, base_url: str = "http://ner-api:8001"):
        """
        Initialize the NER client.
        
        Args:
            base_url: Base URL of the NER API service
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.timeout = 30  # 30 second timeout
        
    def health_check(self) -> bool:
        """
        Check if the NER API is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"NER API health check failed: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the loaded NER model.
        
        Returns:
            Model information dictionary or None if failed
        """
        try:
            response = self.session.get(f"{self.base_url}/model-info", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get model info: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return None
    
    def extract_entities(self, text: str, return_offsets: bool = True, 
                        return_scores: bool = True) -> Optional[Dict[str, Any]]:
        """
        Extract entities from a single text.
        
        Args:
            text: Text to process
            return_offsets: Whether to return character offsets
            return_scores: Whether to return confidence scores
            
        Returns:
            NER response dictionary or None if failed
        """
        try:
            payload = {
                "text": text,
                "return_offsets": return_offsets,
                "return_scores": return_scores
            }
            
            response = self.session.post(f"{self.base_url}/ner", json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"NER API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return None
    
    def extract_entities_batch(self, texts: List[str], return_offsets: bool = True,
                              return_scores: bool = True) -> Optional[Dict[str, Any]]:
        """
        Extract entities from multiple texts in batch.
        
        Args:
            texts: List of texts to process
            return_offsets: Whether to return character offsets
            return_scores: Whether to return confidence scores
            
        Returns:
            Batch NER response dictionary or None if failed
        """
        try:
            payload = {
                "texts": texts,
                "return_offsets": return_offsets,
                "return_scores": return_scores
            }
            
            response = self.session.post(f"{self.base_url}/ner/batch", json=payload, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Batch NER API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting entities in batch: {e}")
            return None
    
    def get_entity_types(self) -> Optional[Dict[str, Any]]:
        """
        Get the list of entity types that the model can recognize.
        
        Returns:
            Entity types dictionary or None if failed
        """
        try:
            response = self.session.get(f"{self.base_url}/ner/entities", timeout=self.timeout)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get entity types: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error getting entity types: {e}")
            return None

# Global NER client instance
ner_client = None

def get_ner_client() -> Optional[NERClient]:
    """
    Get the global NER client instance, creating it if necessary.
    
    Returns:
        NERClient instance or None if initialization failed
    """
    global ner_client
    
    if ner_client is None:
        try:
            ner_client = NERClient()
            # Test the connection
            if ner_client.health_check():
                logger.info("✅ NER API client initialized successfully")
            else:
                logger.warning("⚠️  NER API is not available")
                ner_client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize NER client: {e}")
            ner_client = None
    
    return ner_client

def extract_entities_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract entities from text using the NER API.
    
    Args:
        text: Text to process
        
    Returns:
        List of extracted entities
    """
    client = get_ner_client()
    if client is None:
        logger.warning("NER API not available, returning empty entities")
        return []
    
    result = client.extract_entities(text)
    if result and "entities" in result:
        return result["entities"]
    else:
        logger.warning("Failed to extract entities from text")
        return []

def extract_entities_from_chunks(chunks: List[str]) -> List[List[Dict[str, Any]]]:
    """
    Extract entities from multiple text chunks.
    
    Args:
        chunks: List of text chunks to process
        
    Returns:
        List of entity lists (one per chunk)
    """
    client = get_ner_client()
    if client is None:
        logger.warning("NER API not available, returning empty entities")
        return [[] for _ in chunks]
    
    result = client.extract_entities_batch(chunks)
    if result and "results" in result:
        return [item.get("entities", []) for item in result["results"]]
    else:
        logger.warning("Failed to extract entities from chunks")
        return [[] for _ in chunks]

def get_available_entity_types() -> List[str]:
    """
    Get the list of available entity types.
    
    Returns:
        List of entity type strings
    """
    client = get_ner_client()
    if client is None:
        return []
    
    result = client.get_entity_types()
    if result and "entity_types" in result:
        return result["entity_types"]
    else:
        return []

def is_ner_available() -> bool:
    """
    Check if the NER API is available.
    
    Returns:
        True if NER API is available, False otherwise
    """
    client = get_ner_client()
    return client is not None and client.health_check() 