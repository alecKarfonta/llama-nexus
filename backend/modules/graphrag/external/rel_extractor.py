"""
Relationship Extraction Client using GLiNER API
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)

class RelationshipExtractor:
    """Client for the GLiNER relationship extraction API."""
    
    def __init__(self, base_url: str = "http://rel-api:8002"):
        """
        Initialize the relationship extractor.
        
        Args:
            base_url: Base URL of the GLiNER API service
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.timeout = 30  # 30 second timeout
        
    def _make_request(self, endpoint: str, method: str = "GET", **kwargs) -> Dict[str, Any]:
        """
        Make a request to the GLiNER API.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            **kwargs: Request parameters
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.RequestException: If the request fails
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "POST":
                response = self.session.post(url, timeout=self.timeout, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check if the GLiNER API is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = self._make_request("/health")
            return response.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded GLiNER model.
        
        Returns:
            Model information dictionary
        """
        return self._make_request("/model-info")
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of the GLiNER model.
        
        Returns:
            Capabilities information
        """
        return self._make_request("/capabilities")
    
    def extract_entities(
        self, 
        text: str, 
        labels: List[str], 
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Extract entities from text using GLiNER.
        
        Args:
            text: Input text to process
            labels: Entity labels to extract
            threshold: Confidence threshold for extraction
            
        Returns:
            Dictionary containing extracted entities and metadata
        """
        params = {
            "text": text,
            "labels": labels,
            "threshold": threshold
        }
        
        return self._make_request("/extract-entities", method="POST", json=params)
    
    def extract_relations(
        self,
        text: str,
        relations: List[Dict[str, Any]],
        entity_labels: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Extract relationships from text using GLiNER.
        
        Args:
            text: Input text to process
            relations: List of relation definitions
            entity_labels: Optional entity labels to extract
            threshold: Confidence threshold for extraction
            
        Returns:
            Dictionary containing extracted relations and metadata
        """
        data = {
            "text": text,
            "relations": relations,
            "threshold": threshold
        }
        
        if entity_labels:
            data["entity_labels"] = entity_labels
        
        return self._make_request("/extract-relations", method="POST", json=data)
    
    def extract_relations_batch(
        self,
        texts: List[str],
        relations: List[Dict[str, Any]],
        entity_labels: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Extract relationships from multiple texts in batch.
        
        Args:
            texts: List of input texts to process
            relations: List of relation definitions
            entity_labels: Optional entity labels to extract
            threshold: Confidence threshold for extraction
            
        Returns:
            Dictionary containing extracted relations for all texts
        """
        data = {
            "texts": texts,
            "relations": relations,
            "threshold": threshold
        }
        
        if entity_labels:
            data["entity_labels"] = entity_labels
        
        return self._make_request("/extract-relations/batch", method="POST", json=data)
    
    def extract_company_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract company-related relationships from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary containing extracted company relations
        """
        # Define common company-related relations
        relations = [
            {
                "relation": "founder",
                "pairs_filter": [("organisation", "founder")]
            },
            {
                "relation": "inception date", 
                "pairs_filter": [("organisation", "date")]
            },
            {
                "relation": "CEO",
                "pairs_filter": [("organisation", "position")]
            },
            {
                "relation": "headquarters",
                "pairs_filter": [("organisation", "location")]
            },
            {
                "relation": "industry",
                "pairs_filter": [("organisation", "industry")]
            }
        ]
        
        entity_labels = ["organisation", "founder", "date", "position", "location", "industry"]
        
        return self.extract_relations(text, relations, entity_labels)
    
    def extract_person_relations(self, text: str) -> Dict[str, Any]:
        """
        Extract person-related relationships from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary containing extracted person relations
        """
        # Define common person-related relations
        relations = [
            {
                "relation": "works for",
                "pairs_filter": [("person", "organisation")]
            },
            {
                "relation": "position",
                "pairs_filter": [("person", "position")]
            },
            {
                "relation": "location",
                "pairs_filter": [("person", "location")]
            },
            {
                "relation": "education",
                "pairs_filter": [("person", "institution")]
            }
        ]
        
        entity_labels = ["person", "organisation", "position", "location", "institution"]
        
        return self.extract_relations(text, relations, entity_labels)
    
    def extract_custom_relations(
        self, 
        text: str, 
        relation_definitions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract custom relationships based on user-defined relation definitions.
        
        Args:
            text: Input text to process
            relation_definitions: List of custom relation definitions
            
        Returns:
            Dictionary containing extracted custom relations
        """
        return self.extract_relations(text, relation_definitions)
    
    def is_available(self) -> bool:
        """
        Check if the GLiNER API is available and ready.
        
        Returns:
            True if available, False otherwise
        """
        try:
            # Try to get model info - if it works, the service is available
            self.get_model_info()
            return True
        except Exception as e:
            logger.debug(f"GLiNER API not available: {e}")
            return False

# Global instance for easy access
rel_extractor = RelationshipExtractor()

def get_relationship_extractor() -> RelationshipExtractor:
    """
    Get the global relationship extractor instance.
    
    Returns:
        RelationshipExtractor instance
    """
    return rel_extractor 