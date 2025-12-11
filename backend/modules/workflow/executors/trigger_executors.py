"""
Trigger Node Executors - Entry points for workflow execution
"""

from typing import Dict, Any
from .base import NodeExecutor, ExecutionContext


class ManualTriggerExecutor(NodeExecutor):
    """Manual trigger - starts workflow manually"""
    
    node_type = "manual_trigger"
    display_name = "Manual Trigger"
    category = "trigger"
    description = "Manually trigger workflow execution"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        context.log("Manual trigger activated")
        
        # Pass through any trigger data
        trigger_data = context.variables.get("_trigger_data", {})
        
        return {
            "trigger": trigger_data,
        }


class HttpWebhookExecutor(NodeExecutor):
    """HTTP Webhook trigger - receives HTTP requests"""
    
    node_type = "http_webhook"
    display_name = "HTTP Webhook"
    category = "trigger"
    description = "Trigger workflow via HTTP request"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        trigger_data = context.variables.get("_trigger_data", {})
        
        context.log(f"Webhook received: {trigger_data.get('method', 'POST')} request")
        
        return {
            "body": trigger_data.get("body", {}),
            "headers": trigger_data.get("headers", {}),
            "query": trigger_data.get("query", {}),
        }
