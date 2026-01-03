"""
Route modules for the LlamaCPP Management API.
Each module contains a FastAPI APIRouter for a specific feature area.
"""

from .workflows import router as workflows_router
from .rag import router as rag_router
from .graphrag import router as graphrag_router
from .conversations import router as conversations_router
from .registry import router as registry_router
from .prompts import router as prompts_router
from .benchmark import router as benchmark_router
from .batch import router as batch_router
from .models import router as models_router
from .templates import router as templates_router
from .tokens import router as tokens_router
from .service import router as service_router
from .stt import router as stt_router
from .tts import router as tts_router
from .tools import router as tools_router
from .finetuning import router as finetuning_router
from .quantization import router as quantization_router

__all__ = [
    'workflows_router',
    'rag_router',
    'graphrag_router',
    'conversations_router',
    'registry_router',
    'prompts_router',
    'benchmark_router',
    'batch_router',
    'models_router',
    'templates_router',
    'tokens_router',
    'service_router',
    'stt_router',
    'tts_router',
    'tools_router',
    'finetuning_router',
    'quantization_router',
]
