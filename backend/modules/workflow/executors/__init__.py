"""
Node Executors - Implementations for each workflow node type
"""

from .base import NodeExecutor, ExecutionContext
from .trigger_executors import ManualTriggerExecutor, HttpWebhookExecutor
from .llm_executors import LLMChatExecutor, OpenAIChatExecutor, OpenAIAPILLMExecutor, EmbeddingExecutor
from .rag_executors import DocumentLoaderExecutor, ChunkerExecutor, RetrieverExecutor, VectorStoreExecutor
from .graphrag_executors import (
    GraphRAGSearchExecutor,
    EntityExtractionExecutor,
    MultiHopReasoningExecutor,
    CausalReasoningExecutor,
    ComparativeReasoningExecutor,
    EntityLinkingExecutor,
    CodeDetectionExecutor,
    CodeSearchExecutor,
)
from .data_executors import (
    TemplateExecutor, 
    JsonParseExecutor, 
    JsonStringifyExecutor,
    MapperExecutor,
    FilterExecutor,
)
from .control_executors import (
    ConditionExecutor,
    SwitchExecutor,
    LoopExecutor,
    MergeExecutor,
    DelayExecutor,
)
from .api_executors import HttpRequestExecutor, GraphQLExecutor
from .output_executors import OutputExecutor, LogExecutor
from .mcp_executors import (
    MCPToolExecutor,
    MCPResourceExecutor,
    MCPPromptExecutor,
    MCPServerListExecutor,
)
from .database_executors import (
    SQLQueryExecutor,
    CacheGetExecutor,
    CacheSetExecutor,
    CacheDeleteExecutor,
    QdrantSearchExecutor,
)
from .code_executors import (
    CodeExecutorExecutor,
    FunctionCallExecutor,
    ShellCommandExecutor,
)

# Registry of all node executors
NODE_EXECUTORS = {
    # Triggers
    'manual_trigger': ManualTriggerExecutor,
    'http_webhook': HttpWebhookExecutor,
    
    # LLM
    'llm_chat': LLMChatExecutor,
    'openai_chat': OpenAIChatExecutor,
    'openai_api_llm': OpenAIAPILLMExecutor,
    'embedding': EmbeddingExecutor,
    
    # RAG
    'document_loader': DocumentLoaderExecutor,
    'chunker': ChunkerExecutor,
    'retriever': RetrieverExecutor,
    'vector_store': VectorStoreExecutor,
    
    # GraphRAG
    'graphrag_search': GraphRAGSearchExecutor,
    'entity_extraction': EntityExtractionExecutor,
    'multi_hop_reasoning': MultiHopReasoningExecutor,
    'causal_reasoning': CausalReasoningExecutor,
    'comparative_reasoning': ComparativeReasoningExecutor,
    'entity_linking': EntityLinkingExecutor,
    
    # Data
    'template': TemplateExecutor,
    'json_parse': JsonParseExecutor,
    'json_stringify': JsonStringifyExecutor,
    'mapper': MapperExecutor,
    'filter': FilterExecutor,
    
    # Control
    'condition': ConditionExecutor,
    'switch': SwitchExecutor,
    'loop': LoopExecutor,
    'merge': MergeExecutor,
    'delay': DelayExecutor,
    
    # API
    'http_request': HttpRequestExecutor,
    'graphql_query': GraphQLExecutor,
    
    # MCP
    'mcp_tool': MCPToolExecutor,
    'mcp_resource': MCPResourceExecutor,
    'mcp_prompt': MCPPromptExecutor,
    'mcp_servers': MCPServerListExecutor,
    
    # Database
    'sql_query': SQLQueryExecutor,
    'cache_get': CacheGetExecutor,
    'cache_set': CacheSetExecutor,
    'cache_delete': CacheDeleteExecutor,
    'qdrant_search': QdrantSearchExecutor,
    
    # Code/Tools
    'code_executor': CodeExecutorExecutor,
    'function_call': FunctionCallExecutor,
    'shell_command': ShellCommandExecutor,
    'code_detection': CodeDetectionExecutor,
    'code_search': CodeSearchExecutor,
    
    # Output
    'output': OutputExecutor,
    'log': LogExecutor,
}

def get_executor(node_type: str) -> type:
    """Get executor class for a node type"""
    return NODE_EXECUTORS.get(node_type)

__all__ = [
    'NodeExecutor',
    'ExecutionContext',
    'NODE_EXECUTORS',
    'get_executor',
]
