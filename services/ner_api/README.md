# NER API

A FastAPI-based Named Entity Recognition (NER) service using the [distilbert-NER](https://huggingface.co/dslim/distilbert-NER) model from Hugging Face.

## Features

- **Single Text NER**: Extract entities from individual text inputs
- **Batch Processing**: Process multiple texts efficiently
- **Entity Types**: LOC (Location), MISC (Miscellaneous), ORG (Organization), PER (Person)
- **Confidence Scores**: Get confidence scores for each detected entity
- **Character Offsets**: Get precise character positions of entities
- **Health Monitoring**: Built-in health checks and model status

## Model Information

- **Model**: `dslim/distilbert-NER`
- **Type**: DistilBERT (distilled version of BERT)
- **Parameters**: ~66M parameters
- **Performance**: F1 Score of 0.9217 on CoNLL-2003 dataset
- **Entity Types**: LOC, MISC, ORG, PER

## API Endpoints

### Health Check
```http
GET /health
```

### Model Information
```http
GET /model-info
```

### Entity Types
```http
GET /ner/entities
```

### Single Text NER
```http
POST /ner
```

**Request Body:**
```json
{
  "text": "My name is Wolfgang and I live in Berlin.",
  "return_offsets": true,
  "return_scores": true
}
```

**Response:**
```json
{
  "entities": [
    {
      "entity": "PER",
      "word": "Wolfgang",
      "score": 0.9987,
      "start": 11,
      "end": 19
    },
    {
      "entity": "LOC",
      "word": "Berlin",
      "score": 0.9992,
      "start": 30,
      "end": 36
    }
  ],
  "text": "My name is Wolfgang and I live in Berlin.",
  "processing_time": 0.045,
  "model_info": {
    "model_name": "dslim/bert-base-NER",
    "model_type": "bert-base",
    "task": "token_classification",
    "entity_types": ["LOC", "MISC", "ORG", "PER"],
    "model_size": "110,000,000 parameters"
  }
}
```

### Batch NER Processing
```http
POST /ner/batch
```

**Request Body:**
```json
{
  "texts": [
    "Apple Inc. is headquartered in Cupertino, California.",
    "John Smith works at Google in Mountain View.",
    "The Eiffel Tower is located in Paris, France."
  ],
  "return_offsets": true,
  "return_scores": true
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "Apple Inc. is headquartered in Cupertino, California.",
      "entities": [
        {
          "entity": "ORG",
          "word": "Apple Inc.",
          "score": 0.9989,
          "start": 0,
          "end": 9
        },
        {
          "entity": "LOC",
          "word": "Cupertino",
          "score": 0.9991,
          "start": 25,
          "end": 34
        },
        {
          "entity": "LOC",
          "word": "California",
          "score": 0.9995,
          "start": 36,
          "end": 46
        }
      ],
      "entity_count": 3
    }
  ],
  "processing_time": 0.123,
  "model_info": {
    "model_name": "dslim/bert-base-NER",
    "model_type": "bert-base",
    "task": "token_classification",
    "entity_types": ["LOC", "MISC", "ORG", "PER"],
    "model_size": "110,000,000 parameters"
  }
}
```

## Usage Examples

### Python Client

```python
import requests
import json

# Single text NER
def extract_entities(text):
    url = "http://localhost:8001/ner"
    payload = {
        "text": text,
        "return_offsets": True,
        "return_scores": True
    }
    
    response = requests.post(url, json=payload)
    return response.json()

# Example usage
text = "Steve Jobs founded Apple in Cupertino, California."
result = extract_entities(text)
print(json.dumps(result, indent=2))
```

### cURL Examples

```bash
# Health check
curl http://localhost:8001/health

# Single text NER
curl -X POST http://localhost:8001/ner \
  -H "Content-Type: application/json" \
  -d '{
    "text": "My name is Wolfgang and I live in Berlin.",
    "return_offsets": true,
    "return_scores": true
  }'

# Batch NER
curl -X POST http://localhost:8001/ner/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Apple Inc. is headquartered in Cupertino, California.",
      "John Smith works at Google in Mountain View."
    ],
    "return_offsets": true,
    "return_scores": true
  }'
```

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t ner-api .

# Run the container
docker run -p 8001:8001 ner-api
```

### Docker Compose

Add to your `docker-compose.yml`:

```yaml
services:
  ner-api:
    build:
      context: ./ner_api
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - HF_HOME=/app/.cache/huggingface
      - TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
    volumes:
      - ner_cache:/app/.cache
    networks:
      - app-network
    restart: unless-stopped

volumes:
  ner_cache:

networks:
  app-network:
    driver: bridge
```

## Testing

Run the test script to verify the API functionality:

```bash
python test_ner_api.py
```

## Performance

- **Model Loading**: ~30-60 seconds on first startup
- **Inference Time**: ~50-100ms per text (depending on length)
- **Memory Usage**: ~500MB RAM
- **CPU Usage**: Moderate (can be optimized with GPU)

## Configuration

### Environment Variables

- `HF_HOME`: Hugging Face cache directory
- `TRANSFORMERS_CACHE`: Transformers cache directory
- `CUDA_VISIBLE_DEVICES`: GPU device selection (if available)

### Model Caching

The model is automatically cached in `/app/.cache/huggingface/` on first download. Subsequent starts will be faster.

## Error Handling

The API includes comprehensive error handling:

- **503 Service Unavailable**: Model not loaded
- **500 Internal Server Error**: Processing errors
- **422 Validation Error**: Invalid request format

## Monitoring

- Health check endpoint for Docker health checks
- Processing time tracking
- Model status monitoring
- Detailed error logging

## Integration with Main API

The NER API can be integrated with the main GraphRAG API by adding a service call to the NER endpoints. This allows for enhanced entity extraction capabilities in the document processing pipeline. 