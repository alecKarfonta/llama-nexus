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
    GraphRAGDocumentIngestExecutor,
    GraphRAGDocumentListExecutor,
    GraphRAGDocumentDeleteExecutor,
    GraphRAGKnowledgeGraphRebuildExecutor,
    CodeDetectionExecutor,
    CodeSearchExecutor,
    GraphRAGCodeIngestExecutor,
    GraphRAGCodeIngestionStatusExecutor,
    GraphRAGCodeAnalysisExecutor,
    GraphRAGCodeStatsExecutor,
    GraphRAGCapabilitiesExecutor,
    GraphRAGQueryComplexityExecutor,
    EnhancedEntityExtractionExecutor,
    GraphRAGNERExtractionExecutor,
    GraphRAGRelationExtractionExecutor,
    GraphRAGEnhancedQueryExecutor,
    GraphRAGChatContextExecutor,
    GraphRAGInstrumentationExecutor,
    GraphRAGEvaluationRunExecutor,
    GraphRAGTestSuiteExecutor,
    GraphRAGDomainStatsExecutor,
    GraphRAGGraphExplorerExecutor,
    GraphRAGEnhancedExtractionDiagnosticExecutor,
    GraphRAGEvaluationStatusExecutor,
    GraphRAGFeedbackSubmitExecutor,
    GraphRAGFeedbackAnalyticsExecutor,
    GraphRAGSocialMemoryIngestExecutor,
    GraphRAGSocialMemoryContextExecutor,
    GraphRAGSocialMemoryFactsExecutor,
    GraphRAGSocialMemoryAdminExecutor,
    GraphRAGRedditCrawlExecutor,
    GraphRAGRedditCrawlStatusExecutor,
    GraphRAGRedditCrawlResultsExecutor,
    GraphRAGUpstreamExecutor,
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
    'graphrag_document_ingest': GraphRAGDocumentIngestExecutor,
    'graphrag_document_list': GraphRAGDocumentListExecutor,
    'graphrag_document_delete': GraphRAGDocumentDeleteExecutor,
    'graphrag_kg_rebuild': GraphRAGKnowledgeGraphRebuildExecutor,
    'graphrag_code_ingest': GraphRAGCodeIngestExecutor,
    'graphrag_code_ingestion_status': GraphRAGCodeIngestionStatusExecutor,
    'graphrag_code_analysis': GraphRAGCodeAnalysisExecutor,
    'graphrag_code_stats': GraphRAGCodeStatsExecutor,
    'graphrag_capabilities': GraphRAGCapabilitiesExecutor,
    'graphrag_query_complexity': GraphRAGQueryComplexityExecutor,
    'enhanced_entity_extraction': EnhancedEntityExtractionExecutor,
    'graphrag_ner_extraction': GraphRAGNERExtractionExecutor,
    'graphrag_relation_extraction': GraphRAGRelationExtractionExecutor,
    'graphrag_enhanced_query': GraphRAGEnhancedQueryExecutor,
    'graphrag_chat_context': GraphRAGChatContextExecutor,
    'graphrag_instrumentation': GraphRAGInstrumentationExecutor,
    'graphrag_evaluation_run': GraphRAGEvaluationRunExecutor,
    'graphrag_test_suite': GraphRAGTestSuiteExecutor,
    'graphrag_domain_stats': GraphRAGDomainStatsExecutor,
    'graphrag_graph_explorer': GraphRAGGraphExplorerExecutor,
    'graphrag_enhanced_extraction_test': GraphRAGEnhancedExtractionDiagnosticExecutor,
    'graphrag_evaluation_status': GraphRAGEvaluationStatusExecutor,
    'graphrag_feedback_submit': GraphRAGFeedbackSubmitExecutor,
    'graphrag_feedback_analytics': GraphRAGFeedbackAnalyticsExecutor,
    'graphrag_social_memory_ingest': GraphRAGSocialMemoryIngestExecutor,
    'graphrag_social_memory_context': GraphRAGSocialMemoryContextExecutor,
    'graphrag_social_memory_facts': GraphRAGSocialMemoryFactsExecutor,
    'graphrag_social_memory_admin': GraphRAGSocialMemoryAdminExecutor,
    'graphrag_reddit_crawl': GraphRAGRedditCrawlExecutor,
    'graphrag_reddit_crawl_status': GraphRAGRedditCrawlStatusExecutor,
    'graphrag_reddit_crawl_results': GraphRAGRedditCrawlResultsExecutor,
    'graphrag_upstream': GraphRAGUpstreamExecutor,
    
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
