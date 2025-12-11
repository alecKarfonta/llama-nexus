"""
Output Node Executors - Workflow outputs and responses
"""

from typing import Dict, Any
import json
from .base import NodeExecutor, ExecutionContext


class OutputExecutor(NodeExecutor):
    """Workflow output"""
    
    node_type = "output"
    display_name = "Output"
    category = "output"
    description = "Final output of the workflow"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        value = inputs.get("value")
        output_name = self.get_config_value("name", "result")
        
        context.log(f"Workflow output '{output_name}': {type(value).__name__}")
        
        # Store in context for final result
        context.set_variable(f"_output_{output_name}", value)
        
        return {"_final_output": value}


class WebhookResponseExecutor(NodeExecutor):
    """HTTP response for webhook trigger"""
    
    node_type = "webhook_response"
    display_name = "Webhook Response"
    category = "output"
    description = "Send HTTP response for webhook-triggered workflow"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        body = inputs.get("body")
        headers = inputs.get("headers", {})
        
        status_code = self.get_config_value("statusCode", 200)
        content_type = self.get_config_value("contentType", "application/json")
        
        context.log(f"Webhook response: {status_code}")
        
        # Prepare response
        if content_type == "application/json" and not isinstance(body, str):
            response_body = json.dumps(body, default=str)
        else:
            response_body = str(body) if body is not None else ""
        
        response = {
            "statusCode": status_code,
            "body": response_body,
            "headers": {
                "Content-Type": content_type,
                **headers,
            }
        }
        
        # Store for webhook handler
        context.set_variable("_webhook_response", response)
        
        return {"_response": response}


class LogExecutor(NodeExecutor):
    """Log message"""
    
    node_type = "log"
    display_name = "Log"
    category = "output"
    description = "Log a message during workflow execution"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        message = inputs.get("message")
        level = self.get_config_value("level", "info")
        
        # Format message
        if isinstance(message, (dict, list)):
            formatted = json.dumps(message, indent=2, default=str)
        else:
            formatted = str(message)
        
        context.log(f"[LOG] {formatted}", level=level)
        
        # Passthrough the message
        return {"passthrough": message}


class NotificationExecutor(NodeExecutor):
    """Send notification"""
    
    node_type = "notification"
    display_name = "Notification"
    category = "output"
    description = "Send a notification via configured channel"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        message = inputs.get("message", "")
        channel = self.get_config_value("channel", "log")
        
        context.log(f"Notification [{channel}]: {message}")
        
        # For now, just log the notification
        # Future: integrate with Slack, email, webhooks, etc.
        
        return {"sent": True}


class FileWriteExecutor(NodeExecutor):
    """Write to file"""
    
    node_type = "file_write"
    display_name = "File Write"
    category = "output"
    description = "Write content to a file"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        path = inputs.get("path", "")
        content = inputs.get("content", "")
        
        if not path:
            path = self.get_config_value("path", "")
        
        if not path:
            raise ValueError("File path is required")
        
        context.log(f"Writing to file: {path}")
        
        # Safety check - only allow writing to specific directories
        from pathlib import Path
        file_path = Path(path)
        
        # Ensure path is within allowed directories
        allowed_dirs = ["/app/data", "/tmp", "data"]
        is_allowed = False
        for allowed in allowed_dirs:
            try:
                if file_path.resolve().is_relative_to(Path(allowed).resolve()):
                    is_allowed = True
                    break
            except:
                pass
        
        if not is_allowed:
            raise ValueError(f"Writing to {path} is not allowed")
        
        # Write content
        if isinstance(content, (dict, list)):
            content = json.dumps(content, indent=2, default=str)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(str(content))
        
        context.log(f"Wrote {len(str(content))} chars to {path}")
        
        return {"written": True}
