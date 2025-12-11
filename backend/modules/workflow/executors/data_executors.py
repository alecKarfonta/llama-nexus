"""
Data Transform Node Executors - Data manipulation and transformation
"""

from typing import Dict, Any, List
import json
import re
from .base import NodeExecutor, ExecutionContext


class TemplateExecutor(NodeExecutor):
    """Render template with variables"""
    
    node_type = "template"
    display_name = "Template"
    category = "data"
    description = "Render template with variables"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        variables = inputs.get("vars", {})
        template = self.get_config_value("template", "")
        
        context.log(f"Rendering template with {len(variables)} variables")
        
        # Simple mustache-style templating
        output = template
        
        # Replace {{variable}} patterns
        for key, value in variables.items():
            pattern = r'\{\{\s*' + re.escape(key) + r'\s*\}\}'
            output = re.sub(pattern, str(value), output)
        
        # Also support workflow variables
        for key, value in context.variables.items():
            if not key.startswith("_"):
                pattern = r'\{\{\s*' + re.escape(key) + r'\s*\}\}'
                output = re.sub(pattern, str(value), output)
        
        return {"output": output}


class JsonParseExecutor(NodeExecutor):
    """Parse JSON string to object"""
    
    node_type = "json_parse"
    display_name = "JSON Parse"
    category = "data"
    description = "Parse JSON string to object"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        input_str = inputs.get("input", "")
        
        if not input_str:
            return {"output": None}
        
        if isinstance(input_str, (dict, list)):
            # Already parsed
            return {"output": input_str}
        
        try:
            # Try to extract JSON from text (handles markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', str(input_str))
            if json_match:
                input_str = json_match.group(1)
            
            output = json.loads(input_str)
            context.log("Successfully parsed JSON")
            return {"output": output}
        except json.JSONDecodeError as e:
            context.log(f"JSON parse error: {e}", level="error")
            raise ValueError(f"Invalid JSON: {e}")


class JsonStringifyExecutor(NodeExecutor):
    """Convert object to JSON string"""
    
    node_type = "json_stringify"
    display_name = "JSON Stringify"
    category = "data"
    description = "Convert object to JSON string"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        input_obj = inputs.get("input")
        pretty = self.get_config_value("pretty", False)
        
        if input_obj is None:
            return {"output": "null"}
        
        if pretty:
            output = json.dumps(input_obj, indent=2, default=str)
        else:
            output = json.dumps(input_obj, default=str)
        
        context.log(f"Stringified to {len(output)} chars")
        return {"output": output}


class MapperExecutor(NodeExecutor):
    """Map over array items"""
    
    node_type = "mapper"
    display_name = "Array Map"
    category = "data"
    description = "Transform each item in an array"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        items = inputs.get("items", [])
        expression = self.get_config_value("expression", "item")
        
        if not items:
            return {"results": []}
        
        context.log(f"Mapping {len(items)} items")
        
        results = []
        for i, item in enumerate(items):
            try:
                # Create evaluation context
                local_vars = {
                    "item": item,
                    "index": i,
                    "items": items,
                }
                
                # Evaluate expression
                result = eval(expression, {"__builtins__": {}}, local_vars)
                results.append(result)
            except Exception as e:
                context.log(f"Map error at index {i}: {e}", level="warning")
                results.append(None)
        
        return {"results": results}


class FilterExecutor(NodeExecutor):
    """Filter array items"""
    
    node_type = "filter"
    display_name = "Array Filter"
    category = "data"
    description = "Filter items based on condition"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        items = inputs.get("items", [])
        condition = self.get_config_value("condition", "True")
        
        if not items:
            return {"results": []}
        
        context.log(f"Filtering {len(items)} items")
        
        results = []
        for i, item in enumerate(items):
            try:
                local_vars = {
                    "item": item,
                    "index": i,
                    "items": items,
                }
                
                if eval(condition, {"__builtins__": {}}, local_vars):
                    results.append(item)
            except Exception as e:
                context.log(f"Filter error at index {i}: {e}", level="warning")
        
        context.log(f"Filtered to {len(results)} items")
        return {"results": results}


class ExtractJsonExecutor(NodeExecutor):
    """Extract JSON from text"""
    
    node_type = "extract_json"
    display_name = "Extract JSON"
    category = "data"
    description = "Extract JSON objects from text"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text", "")
        
        if not text:
            return {"json": None}
        
        # Try to find JSON in the text
        # First, try markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return {"json": json.loads(json_match.group(1))}
            except:
                pass
        
        # Try to find JSON object or array
        for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
            match = re.search(pattern, text)
            if match:
                try:
                    return {"json": json.loads(match.group(0))}
                except:
                    continue
        
        context.log("No valid JSON found in text", level="warning")
        return {"json": None}


class RegexExtractExecutor(NodeExecutor):
    """Extract with regex"""
    
    node_type = "regex_extract"
    display_name = "Regex Extract"
    category = "data"
    description = "Extract text using regular expressions"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        text = inputs.get("text", "")
        pattern = self.get_config_value("pattern", "")
        
        if not text or not pattern:
            return {"matches": []}
        
        try:
            matches = re.findall(pattern, text)
            context.log(f"Found {len(matches)} matches")
            return {"matches": matches}
        except re.error as e:
            context.log(f"Regex error: {e}", level="error")
            raise ValueError(f"Invalid regex pattern: {e}")


class AggregatorExecutor(NodeExecutor):
    """Aggregate multiple inputs"""
    
    node_type = "aggregator"
    display_name = "Aggregator"
    category = "data"
    description = "Combine multiple inputs into one output"
    
    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext) -> Dict[str, Any]:
        # Collect all inputs into a list
        result = []
        
        for key, value in inputs.items():
            if value is not None:
                result.append(value)
        
        context.log(f"Aggregated {len(result)} inputs")
        return {"output": result}
