"""
Tools Bridge for MCP

Converts between MCP tool definitions and OpenAI-compatible function calling format.
This enables MCP tools to be used seamlessly with the existing chat infrastructure.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ToolsBridge:
    """Converts between MCP tools and OpenAI function definitions."""
    
    @staticmethod
    def mcp_to_openai(server_name: str, mcp_tools: List[Any]) -> List[Dict]:
        """Convert MCP tool definitions to OpenAI function format.
        
        Args:
            server_name: Name of the MCP server (used for namespacing)
            mcp_tools: List of MCP tool definitions
            
        Returns:
            List of OpenAI-compatible tool definitions
        """
        openai_tools = []
        
        for tool in mcp_tools:
            try:
                # Handle both dict and object representations
                if hasattr(tool, 'name'):
                    tool_name = tool.name
                    tool_description = getattr(tool, 'description', '')
                    input_schema = getattr(tool, 'inputSchema', {})
                else:
                    tool_name = tool.get('name', '')
                    tool_description = tool.get('description', '')
                    input_schema = tool.get('inputSchema', {})
                
                # Create namespaced tool name: mcp_<server>_<tool>
                namespaced_name = f"mcp_{server_name}_{tool_name}"
                
                # Build OpenAI-compatible tool definition
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": namespaced_name,
                        "description": f"[MCP:{server_name}] {tool_description}",
                        "parameters": input_schema if input_schema else {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    }
                }
                
                openai_tools.append(openai_tool)
                
            except Exception as e:
                logger.warning(f"Failed to convert MCP tool to OpenAI format: {e}")
                continue
        
        return openai_tools
    
    @staticmethod
    def parse_mcp_tool_call(tool_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract server and tool name from a namespaced tool call.
        
        Args:
            tool_name: Namespaced tool name (e.g., "mcp_filesystem_read_file")
            
        Returns:
            Tuple of (server_name, tool_name) or (None, None) if not an MCP tool
        """
        if not tool_name.startswith("mcp_"):
            return None, None
        
        parts = tool_name.split("_", 2)
        if len(parts) < 3:
            logger.warning(f"Invalid MCP tool name format: {tool_name}")
            return None, None
        
        # parts[0] = "mcp", parts[1] = server_name, parts[2] = tool_name
        return parts[1], parts[2]
    
    @staticmethod
    def is_mcp_tool(tool_name: str) -> bool:
        """Check if a tool name is an MCP tool (namespaced).
        
        Args:
            tool_name: Tool name to check
            
        Returns:
            True if this is an MCP-namespaced tool
        """
        return tool_name.startswith("mcp_")
    
    @staticmethod
    def format_tool_result(result: Any) -> str:
        """Format an MCP tool result for inclusion in chat messages.
        
        Args:
            result: Raw result from MCP tool execution
            
        Returns:
            Formatted string representation
        """
        if result is None:
            return "Tool executed successfully (no output)"
        
        if isinstance(result, str):
            return result
        
        if isinstance(result, (dict, list)):
            try:
                return json.dumps(result, indent=2, default=str)
            except Exception:
                return str(result)
        
        return str(result)
    
    @staticmethod
    def create_tool_summary(server_name: str, tools: List[Any]) -> str:
        """Create a human-readable summary of available tools.
        
        Args:
            server_name: Name of the MCP server
            tools: List of MCP tool definitions
            
        Returns:
            Formatted summary string
        """
        if not tools:
            return f"No tools available from {server_name}"
        
        lines = [f"**{server_name}** ({len(tools)} tools):"]
        
        for tool in tools:
            if hasattr(tool, 'name'):
                name = tool.name
                desc = getattr(tool, 'description', 'No description')
            else:
                name = tool.get('name', 'unknown')
                desc = tool.get('description', 'No description')
            
            # Truncate long descriptions
            if len(desc) > 80:
                desc = desc[:77] + "..."
            
            lines.append(f"  • `{name}`: {desc}")
        
        return "\n".join(lines)
    
    @staticmethod
    def merge_tool_lists(*tool_lists: List[Dict]) -> List[Dict]:
        """Merge multiple tool lists, handling duplicates.
        
        Args:
            *tool_lists: Variable number of tool lists to merge
            
        Returns:
            Merged list with duplicates removed (by function name)
        """
        seen_names = set()
        merged = []
        
        for tools in tool_lists:
            for tool in tools:
                func_name = tool.get("function", {}).get("name", "")
                if func_name and func_name not in seen_names:
                    seen_names.add(func_name)
                    merged.append(tool)
        
        return merged
