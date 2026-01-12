"""
Workflow Templates - Pre-built workflow templates for common use cases
"""

from typing import List, Dict, Any

# Pre-built workflow templates
WORKFLOW_TEMPLATES = [
    {
        "id": "rag-qa",
        "name": "RAG Q&A Pipeline",
        "description": "Query documents with semantic search and LLM response generation",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "retriever-1",
                "type": "retriever",
                "position": {"x": 350, "y": 200},
                "data": {
                    "label": "Semantic Search",
                    "nodeType": "retriever",
                    "config": {"collection": "default", "k": 5},
                    "inputs": [{"id": "query", "name": "Query", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "documents", "name": "Documents", "type": "array"},
                        {"id": "scores", "name": "Scores", "type": "array"},
                    ],
                }
            },
            {
                "id": "template-1",
                "type": "template",
                "position": {"x": 600, "y": 200},
                "data": {
                    "label": "Build Prompt",
                    "nodeType": "template",
                    "config": {
                        "template": "Based on the following context, answer the question.\n\nContext:\n{{documents}}\n\nQuestion: {{query}}\n\nAnswer:"
                    },
                    "inputs": [{"id": "vars", "name": "Variables", "type": "object"}],
                    "outputs": [{"id": "output", "name": "Output", "type": "string"}],
                }
            },
            {
                "id": "llm-1",
                "type": "llm_chat",
                "position": {"x": 850, "y": 200},
                "data": {
                    "label": "Generate Answer",
                    "nodeType": "llm_chat",
                    "config": {"temperature": 0.7, "maxTokens": 1024},
                    "inputs": [
                        {"id": "messages", "name": "Messages", "type": "array", "required": True},
                        {"id": "system", "name": "System Prompt", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "response", "name": "Response", "type": "string"},
                        {"id": "usage", "name": "Token Usage", "type": "object"},
                    ],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1100, "y": 200},
                "data": {
                    "label": "Result",
                    "nodeType": "output",
                    "config": {"name": "answer"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "retriever-1", "targetHandle": "query"},
            {"id": "c2", "source": "retriever-1", "sourceHandle": "documents", "target": "template-1", "targetHandle": "vars"},
            {"id": "c3", "source": "template-1", "sourceHandle": "output", "target": "llm-1", "targetHandle": "messages"},
            {"id": "c4", "source": "llm-1", "sourceHandle": "response", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "api-enrichment",
        "name": "API Data Enrichment",
        "description": "Fetch data from an API, transform it, and generate insights with LLM",
        "category": "api",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "http-1",
                "type": "http_request",
                "position": {"x": 350, "y": 200},
                "data": {
                    "label": "Fetch API Data",
                    "nodeType": "http_request",
                    "config": {"method": "GET", "timeout": 30},
                    "inputs": [
                        {"id": "url", "name": "URL", "type": "string"},
                        {"id": "body", "name": "Body", "type": "any"},
                        {"id": "headers", "name": "Headers", "type": "object"},
                    ],
                    "outputs": [
                        {"id": "response", "name": "Response", "type": "any"},
                        {"id": "status", "name": "Status", "type": "number"},
                        {"id": "headers", "name": "Headers", "type": "object"},
                    ],
                }
            },
            {
                "id": "json-1",
                "type": "json_stringify",
                "position": {"x": 600, "y": 200},
                "data": {
                    "label": "Format Data",
                    "nodeType": "json_stringify",
                    "config": {"pretty": True},
                    "inputs": [{"id": "input", "name": "Input", "type": "object", "required": True}],
                    "outputs": [{"id": "output", "name": "Output", "type": "string"}],
                }
            },
            {
                "id": "llm-1",
                "type": "llm_chat",
                "position": {"x": 850, "y": 200},
                "data": {
                    "label": "Analyze Data",
                    "nodeType": "llm_chat",
                    "config": {"temperature": 0.3, "maxTokens": 2048},
                    "inputs": [
                        {"id": "messages", "name": "Messages", "type": "array", "required": True},
                        {"id": "system", "name": "System Prompt", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "response", "name": "Response", "type": "string"},
                        {"id": "usage", "name": "Token Usage", "type": "object"},
                    ],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1100, "y": 200},
                "data": {
                    "label": "Analysis",
                    "nodeType": "output",
                    "config": {"name": "analysis"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "http-1", "targetHandle": "url"},
            {"id": "c2", "source": "http-1", "sourceHandle": "response", "target": "json-1", "targetHandle": "input"},
            {"id": "c3", "source": "json-1", "sourceHandle": "output", "target": "llm-1", "targetHandle": "messages"},
            {"id": "c4", "source": "llm-1", "sourceHandle": "response", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "data-transform",
        "name": "Data Transformation Pipeline",
        "description": "Load, filter, map, and output transformed data",
        "category": "data",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "filter-1",
                "type": "filter",
                "position": {"x": 350, "y": 200},
                "data": {
                    "label": "Filter Items",
                    "nodeType": "filter",
                    "config": {"condition": "item.active == True"},
                    "inputs": [{"id": "items", "name": "Items", "type": "array", "required": True}],
                    "outputs": [{"id": "results", "name": "Results", "type": "array"}],
                }
            },
            {
                "id": "mapper-1",
                "type": "mapper",
                "position": {"x": 600, "y": 200},
                "data": {
                    "label": "Transform Items",
                    "nodeType": "mapper",
                    "config": {"expression": "{'name': item.name, 'value': item.value * 2}"},
                    "inputs": [{"id": "items", "name": "Items", "type": "array", "required": True}],
                    "outputs": [{"id": "results", "name": "Results", "type": "array"}],
                }
            },
            {
                "id": "log-1",
                "type": "log",
                "position": {"x": 850, "y": 200},
                "data": {
                    "label": "Log Results",
                    "nodeType": "log",
                    "config": {"level": "info"},
                    "inputs": [{"id": "message", "name": "Message", "type": "any", "required": True}],
                    "outputs": [{"id": "passthrough", "name": "Passthrough", "type": "any"}],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1100, "y": 200},
                "data": {
                    "label": "Result",
                    "nodeType": "output",
                    "config": {"name": "transformed_data"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "filter-1", "targetHandle": "items"},
            {"id": "c2", "source": "filter-1", "sourceHandle": "results", "target": "mapper-1", "targetHandle": "items"},
            {"id": "c3", "source": "mapper-1", "sourceHandle": "results", "target": "log-1", "targetHandle": "message"},
            {"id": "c4", "source": "log-1", "sourceHandle": "passthrough", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "conditional-flow",
        "name": "Conditional Branching",
        "description": "Branch workflow based on conditions",
        "category": "control",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "condition-1",
                "type": "condition",
                "position": {"x": 350, "y": 200},
                "data": {
                    "label": "Check Value",
                    "nodeType": "condition",
                    "config": {"condition": "input > 50"},
                    "inputs": [{"id": "input", "name": "Input", "type": "any", "required": True}],
                    "outputs": [
                        {"id": "true", "name": "True", "type": "any"},
                        {"id": "false", "name": "False", "type": "any"},
                    ],
                }
            },
            {
                "id": "log-true",
                "type": "log",
                "position": {"x": 600, "y": 100},
                "data": {
                    "label": "High Value",
                    "nodeType": "log",
                    "config": {"level": "info"},
                    "inputs": [{"id": "message", "name": "Message", "type": "any", "required": True}],
                    "outputs": [{"id": "passthrough", "name": "Passthrough", "type": "any"}],
                }
            },
            {
                "id": "log-false",
                "type": "log",
                "position": {"x": 600, "y": 300},
                "data": {
                    "label": "Low Value",
                    "nodeType": "log",
                    "config": {"level": "info"},
                    "inputs": [{"id": "message", "name": "Message", "type": "any", "required": True}],
                    "outputs": [{"id": "passthrough", "name": "Passthrough", "type": "any"}],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 850, "y": 200},
                "data": {
                    "label": "Merge Results",
                    "nodeType": "merge",
                    "config": {},
                    "inputs": [
                        {"id": "input1", "name": "Input 1", "type": "any"},
                        {"id": "input2", "name": "Input 2", "type": "any"},
                        {"id": "input3", "name": "Input 3", "type": "any"},
                    ],
                    "outputs": [{"id": "output", "name": "Output", "type": "array"}],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1100, "y": 200},
                "data": {
                    "label": "Result",
                    "nodeType": "output",
                    "config": {"name": "result"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "condition-1", "targetHandle": "input"},
            {"id": "c2", "source": "condition-1", "sourceHandle": "true", "target": "log-true", "targetHandle": "message"},
            {"id": "c3", "source": "condition-1", "sourceHandle": "false", "target": "log-false", "targetHandle": "message"},
            {"id": "c4", "source": "log-true", "sourceHandle": "passthrough", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c5", "source": "log-false", "sourceHandle": "passthrough", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c6", "source": "merge-1", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "code-transform",
        "name": "Code-Based Transform",
        "description": "Transform data using custom Python code",
        "category": "tools",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "code-1",
                "type": "code_executor",
                "position": {"x": 400, "y": 200},
                "data": {
                    "label": "Transform Data",
                    "nodeType": "code_executor",
                    "config": {
                        "language": "python",
                        "code": "# Input data is available as 'input'\ndata = input or []\n\n# Process each item\nprocessed = []\nfor item in data:\n    processed.append({\n        'value': item * 2,\n        'original': item\n    })\n\n# Set the result\nresult = processed",
                        "timeout": 30
                    },
                    "inputs": [{"id": "input", "name": "Input", "type": "any"}],
                    "outputs": [
                        {"id": "result", "name": "Result", "type": "any"},
                        {"id": "stdout", "name": "Stdout", "type": "string"},
                    ],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 700, "y": 200},
                "data": {
                    "label": "Result",
                    "nodeType": "output",
                    "config": {"name": "transformed"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "code-1", "targetHandle": "input"},
            {"id": "c2", "source": "code-1", "sourceHandle": "result", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "simple-chat",
        "name": "Chat with Your Model",
        "description": "Simple chat interface with your deployed LLM - perfect for getting started",
        "category": "llm",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start Chat",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "template-1",
                "type": "template",
                "position": {"x": 350, "y": 200},
                "data": {
                    "label": "Format Message",
                    "nodeType": "template",
                    "config": {
                        "template": "[{\"role\": \"user\", \"content\": \"{{query}}\"}]"
                    },
                    "inputs": [{"id": "vars", "name": "Variables", "type": "object"}],
                    "outputs": [{"id": "output", "name": "Output", "type": "string"}],
                }
            },
            {
                "id": "json-parse-1",
                "type": "json_parse",
                "position": {"x": 600, "y": 200},
                "data": {
                    "label": "Parse Messages",
                    "nodeType": "json_parse",
                    "config": {},
                    "inputs": [{"id": "input", "name": "Input", "type": "string", "required": True}],
                    "outputs": [{"id": "output", "name": "Output", "type": "object"}],
                }
            },
            {
                "id": "llm-1",
                "type": "llm_chat",
                "position": {"x": 850, "y": 200},
                "data": {
                    "label": "Chat Completion",
                    "nodeType": "llm_chat",
                    "config": {"temperature": 0.7, "maxTokens": 2048},
                    "inputs": [
                        {"id": "messages", "name": "Messages", "type": "array", "required": True},
                        {"id": "system", "name": "System Prompt", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "response", "name": "Response", "type": "string"},
                        {"id": "usage", "name": "Token Usage", "type": "object"},
                    ],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1100, "y": 200},
                "data": {
                    "label": "Chat Response",
                    "nodeType": "output",
                    "config": {"name": "response"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "template-1", "targetHandle": "vars"},
            {"id": "c2", "source": "template-1", "sourceHandle": "output", "target": "json-parse-1", "targetHandle": "input"},
            {"id": "c3", "source": "json-parse-1", "sourceHandle": "output", "target": "llm-1", "targetHandle": "messages"},
            {"id": "c4", "source": "llm-1", "sourceHandle": "response", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "rag-local-llm",
        "name": "Document Q&A with Local LLM",
        "description": "Query documents using semantic search and generate answers with your deployed model",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 250},
                "data": {
                    "label": "Ask Question",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "retriever-1",
                "type": "retriever",
                "position": {"x": 350, "y": 250},
                "data": {
                    "label": "Search Documents",
                    "nodeType": "retriever",
                    "config": {"collection": "default", "k": 3},
                    "inputs": [{"id": "query", "name": "Query", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "documents", "name": "Documents", "type": "array"},
                        {"id": "scores", "name": "Scores", "type": "array"},
                    ],
                }
            },
            {
                "id": "template-1",
                "type": "template",
                "position": {"x": 600, "y": 250},
                "data": {
                    "label": "Build Context",
                    "nodeType": "template",
                    "config": {
                        "template": "[{\"role\": \"system\", \"content\": \"You are a helpful assistant. Answer questions based on the provided context.\"}, {\"role\": \"user\", \"content\": \"Context:\\n{{documents}}\\n\\nQuestion: {{query}}\\n\\nAnswer:\"}]"
                    },
                    "inputs": [{"id": "vars", "name": "Variables", "type": "object"}],
                    "outputs": [{"id": "output", "name": "Output", "type": "string"}],
                }
            },
            {
                "id": "json-parse-1",
                "type": "json_parse",
                "position": {"x": 850, "y": 250},
                "data": {
                    "label": "Parse Messages",
                    "nodeType": "json_parse",
                    "config": {},
                    "inputs": [{"id": "input", "name": "Input", "type": "string", "required": True}],
                    "outputs": [{"id": "output", "name": "Output", "type": "object"}],
                }
            },
            {
                "id": "llm-1",
                "type": "llm_chat",
                "position": {"x": 1100, "y": 250},
                "data": {
                    "label": "Generate Answer",
                    "nodeType": "llm_chat",
                    "config": {"temperature": 0.3, "maxTokens": 1024},
                    "inputs": [
                        {"id": "messages", "name": "Messages", "type": "array", "required": True},
                    ],
                    "outputs": [
                        {"id": "response", "name": "Response", "type": "string"},
                        {"id": "usage", "name": "Token Usage", "type": "object"},
                    ],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1350, "y": 250},
                "data": {
                    "label": "Answer",
                    "nodeType": "output",
                    "config": {"name": "answer"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "retriever-1", "targetHandle": "query"},
            {"id": "c2", "source": "retriever-1", "sourceHandle": "documents", "target": "template-1", "targetHandle": "vars"},
            {"id": "c3", "source": "template-1", "sourceHandle": "output", "target": "json-parse-1", "targetHandle": "input"},
            {"id": "c4", "source": "json-parse-1", "sourceHandle": "output", "target": "llm-1", "targetHandle": "messages"},
            {"id": "c5", "source": "llm-1", "sourceHandle": "response", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "multi-model-compare",
        "name": "Compare Multiple Models",
        "description": "Send the same prompt to multiple models and compare their responses",
        "category": "llm",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 250},
                "data": {
                    "label": "Start",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "template-1",
                "type": "template",
                "position": {"x": 300, "y": 250},
                "data": {
                    "label": "Format Prompt",
                    "nodeType": "template",
                    "config": {
                        "template": "[{\"role\": \"user\", \"content\": \"{{prompt}}\"}]"
                    },
                    "inputs": [{"id": "vars", "name": "Variables", "type": "object"}],
                    "outputs": [{"id": "output", "name": "Output", "type": "string"}],
                }
            },
            {
                "id": "json-parse-1",
                "type": "json_parse",
                "position": {"x": 500, "y": 250},
                "data": {
                    "label": "Parse Messages",
                    "nodeType": "json_parse",
                    "config": {},
                    "inputs": [{"id": "input", "name": "Input", "type": "string", "required": True}],
                    "outputs": [{"id": "output", "name": "Output", "type": "object"}],
                }
            },
            {
                "id": "llm-1",
                "type": "llm_chat",
                "position": {"x": 750, "y": 150},
                "data": {
                    "label": "Model 1",
                    "nodeType": "llm_chat",
                    "config": {"temperature": 0.7, "maxTokens": 512},
                    "inputs": [
                        {"id": "messages", "name": "Messages", "type": "array", "required": True},
                    ],
                    "outputs": [
                        {"id": "response", "name": "Response", "type": "string"},
                        {"id": "usage", "name": "Token Usage", "type": "object"},
                    ],
                }
            },
            {
                "id": "llm-2",
                "type": "llm_chat",
                "position": {"x": 750, "y": 350},
                "data": {
                    "label": "Model 2",
                    "nodeType": "llm_chat",
                    "config": {"temperature": 0.7, "maxTokens": 512},
                    "inputs": [
                        {"id": "messages", "name": "Messages", "type": "array", "required": True},
                    ],
                    "outputs": [
                        {"id": "response", "name": "Response", "type": "string"},
                        {"id": "usage", "name": "Token Usage", "type": "object"},
                    ],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 1000, "y": 250},
                "data": {
                    "label": "Combine Results",
                    "nodeType": "merge",
                    "config": {},
                    "inputs": [
                        {"id": "input1", "name": "Input 1", "type": "any"},
                        {"id": "input2", "name": "Input 2", "type": "any"},
                    ],
                    "outputs": [{"id": "output", "name": "Output", "type": "array"}],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1250, "y": 250},
                "data": {
                    "label": "Comparison",
                    "nodeType": "output",
                    "config": {"name": "comparison"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "template-1", "targetHandle": "vars"},
            {"id": "c2", "source": "template-1", "sourceHandle": "output", "target": "json-parse-1", "targetHandle": "input"},
            {"id": "c3", "source": "json-parse-1", "sourceHandle": "output", "target": "llm-1", "targetHandle": "messages"},
            {"id": "c4", "source": "json-parse-1", "sourceHandle": "output", "target": "llm-2", "targetHandle": "messages"},
            {"id": "c5", "source": "llm-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c6", "source": "llm-2", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c7", "source": "merge-1", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "intelligent-qa",
        "name": "Intelligent Q&A with GraphRAG",
        "description": "Ask questions and get LLM answers with knowledge graph context using hybrid search",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Ask Question",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "graphrag-search-1",
                "type": "graphrag_search",
                "position": {"x": 400, "y": 200},
                "data": {
                    "label": "GraphRAG Hybrid Search",
                    "nodeType": "graphrag_search",
                    "config": {"searchType": "auto", "topK": 10},
                    "inputs": [
                        {"id": "query", "name": "Query", "type": "string", "required": True},
                        {"id": "domain", "name": "Domain", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "answer", "name": "Answer", "type": "string"},
                        {"id": "documents", "name": "Documents", "type": "array"},
                        {"id": "sources", "name": "Sources", "type": "array"},
                        {"id": "entities", "name": "Entities", "type": "array"},
                    ],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 700, "y": 200},
                "data": {
                    "label": "Answer",
                    "nodeType": "output",
                    "config": {"name": "answer"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "graphrag-search-1", "targetHandle": "query"},
            {"id": "c2", "source": "graphrag-search-1", "sourceHandle": "answer", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "entity-discovery",
        "name": "Entity Extraction Pipeline",
        "description": "Extract and link entities from documents automatically using GraphRAG",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "doc-loader-1",
                "type": "document_loader",
                "position": {"x": 300, "y": 200},
                "data": {
                    "label": "Load Document",
                    "nodeType": "document_loader",
                    "config": {"sourceType": "file"},
                    "inputs": [{"id": "source", "name": "Source", "type": "string", "required": True}],
                    "outputs": [{"id": "documents", "name": "Documents", "type": "array"}],
                }
            },
            {
                "id": "template-1",
                "type": "template",
                "position": {"x": 550, "y": 200},
                "data": {
                    "label": "Extract Text",
                    "nodeType": "template",
                    "config": {"template": "{{documents[0].content}}"},
                    "inputs": [{"id": "vars", "name": "Variables", "type": "object"}],
                    "outputs": [{"id": "output", "name": "Output", "type": "string"}],
                }
            },
            {
                "id": "entity-extract-1",
                "type": "entity_extraction",
                "position": {"x": 800, "y": 200},
                "data": {
                    "label": "Extract Entities",
                    "nodeType": "entity_extraction",
                    "config": {},
                    "inputs": [
                        {"id": "text", "name": "Text", "type": "string", "required": True},
                        {"id": "domain", "name": "Domain", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "entity_count", "name": "Entity Count", "type": "number"},
                    ],
                }
            },
            {
                "id": "entity-link-1",
                "type": "entity_linking",
                "position": {"x": 1050, "y": 200},
                "data": {
                    "label": "Link Entities",
                    "nodeType": "entity_linking",
                    "config": {},
                    "inputs": [
                        {"id": "entities", "name": "Entities", "type": "array", "required": True},
                        {"id": "context", "name": "Context", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "linked_entities", "name": "Linked Entities", "type": "array"},
                        {"id": "disambiguated", "name": "Disambiguated", "type": "array"},
                        {"id": "link_count", "name": "Link Count", "type": "number"},
                    ],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1300, "y": 200},
                "data": {
                    "label": "Results",
                    "nodeType": "output",
                    "config": {"name": "extracted_entities"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "doc-loader-1", "targetHandle": "source"},
            {"id": "c2", "source": "doc-loader-1", "sourceHandle": "documents", "target": "template-1", "targetHandle": "vars"},
            {"id": "c3", "source": "template-1", "sourceHandle": "output", "target": "entity-extract-1", "targetHandle": "text"},
            {"id": "c4", "source": "entity-extract-1", "sourceHandle": "entities", "target": "entity-link-1", "targetHandle": "entities"},
            {"id": "c5", "source": "entity-link-1", "sourceHandle": "linked_entities", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "multi-hop-investigation",
        "name": "Multi-Hop Knowledge Investigation",
        "description": "Find connections between concepts through knowledge graph reasoning",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start Investigation",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "multi-hop-1",
                "type": "multi_hop_reasoning",
                "position": {"x": 400, "y": 200},
                "data": {
                    "label": "Find Path",
                    "nodeType": "multi_hop_reasoning",
                    "config": {"maxHops": 3},
                    "inputs": [
                        {"id": "source", "name": "Source Entity", "type": "string", "required": True},
                        {"id": "target", "name": "Target Entity", "type": "string", "required": True},
                    ],
                    "outputs": [
                        {"id": "reasoning_path", "name": "Reasoning Path", "type": "array"},
                        {"id": "answer", "name": "Answer", "type": "string"},
                        {"id": "sources", "name": "Sources", "type": "array"},
                        {"id": "hop_count", "name": "Hop Count", "type": "number"},
                    ],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 700, "y": 200},
                "data": {
                    "label": "Investigation Results",
                    "nodeType": "output",
                    "config": {"name": "reasoning_result"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "multi-hop-1", "targetHandle": "source"},
            {"id": "c2", "source": "multi-hop-1", "sourceHandle": "answer", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "causal-analysis",
        "name": "Causal Analysis Workflow",
        "description": "Analyze cause-effect relationships in your knowledge base",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "causal-1",
                "type": "causal_reasoning",
                "position": {"x": 400, "y": 200},
                "data": {
                    "label": "Causal Analysis",
                    "nodeType": "causal_reasoning",
                    "config": {},
                    "inputs": [{"id": "query", "name": "Query", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "causes", "name": "Causes", "type": "array"},
                        {"id": "effects", "name": "Effects", "type": "array"},
                        {"id": "reasoning", "name": "Reasoning", "type": "string"},
                        {"id": "answer", "name": "Answer", "type": "string"},
                    ],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 700, "y": 200},
                "data": {
                    "label": "Causal Analysis",
                    "nodeType": "output",
                    "config": {"name": "causal_analysis"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "causal-1", "targetHandle": "query"},
            {"id": "c2", "source": "causal-1", "sourceHandle": "answer", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "graphrag-batch-processing",
        "name": "GraphRAG Batch Document Processing",
        "description": "Complete pipeline for batch processing documents through GraphRAG with entity extraction, knowledge graph building, and intelligent indexing",
        "category": "graphrag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 200},
                "data": {
                    "label": "Start Batch Processing",
                    "nodeType": "manual_trigger",
                    "config": {"name": "batch_start", "description": "Trigger batch document processing"},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "trigger"}],
                }
            },
            {
                "id": "file-reader-1",
                "type": "file_reader",
                "position": {"x": 300, "y": 100},
                "data": {
                    "label": "Load Documents",
                    "nodeType": "file_reader",
                    "config": {
                        "path": "/documents/*.pdf",
                        "format": "auto",
                        "batch_mode": True
                    },
                    "inputs": [{"id": "trigger", "name": "Trigger", "type": "trigger"}],
                    "outputs": [{"id": "documents", "name": "Documents", "type": "array"}],
                }
            },
            {
                "id": "entity-extract-1",
                "type": "entity_extraction",
                "position": {"x": 500, "y": 100},
                "data": {
                    "label": "Extract Entities (GLiNER)",
                    "nodeType": "entity_extraction",
                    "config": {
                        "labels": ["person", "organization", "location", "product", "technology", "concept"],
                        "threshold": 0.5,
                        "extract_relationships": True
                    },
                    "inputs": [{"id": "text", "name": "Text", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"}
                    ],
                }
            },
            {
                "id": "code-detect-1",
                "type": "code_detection",
                "position": {"x": 500, "y": 250},
                "data": {
                    "label": "Detect Code Blocks",
                    "nodeType": "code_detection",
                    "config": {
                        "extract_metadata": True,
                        "classify_language": True
                    },
                    "inputs": [{"id": "text", "name": "Text", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "has_code", "name": "Has Code", "type": "boolean"},
                        {"id": "code_blocks", "name": "Code Blocks", "type": "array"}
                    ],
                }
            },
            {
                "id": "graphrag-search-1",
                "type": "graphrag_search",
                "position": {"x": 700, "y": 175},
                "data": {
                    "label": "Index in Knowledge Graph",
                    "nodeType": "graphrag_search",
                    "config": {
                        "searchType": "hybrid",
                        "buildGraph": True,
                        "updateVectors": True
                    },
                    "inputs": [
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "documents", "name": "Documents", "type": "array"}
                    ],
                    "outputs": [
                        {"id": "graph_stats", "name": "Graph Statistics", "type": "object"},
                        {"id": "indexed_count", "name": "Indexed Count", "type": "number"}
                    ],
                }
            },
            {
                "id": "condition-1",
                "type": "conditional",
                "position": {"x": 900, "y": 250},
                "data": {
                    "label": "Check for Code",
                    "nodeType": "conditional",
                    "config": {
                        "condition": "has_code === true",
                        "outputTrue": "process_code",
                        "outputFalse": "skip_code"
                    },
                    "inputs": [{"id": "has_code", "name": "Has Code", "type": "boolean"}],
                    "outputs": [
                        {"id": "process_code", "name": "Process Code", "type": "trigger"},
                        {"id": "skip_code", "name": "Skip Code", "type": "trigger"}
                    ],
                }
            },
            {
                "id": "code-search-1",
                "type": "code_search",
                "position": {"x": 1100, "y": 200},
                "data": {
                    "label": "Index Code for Search",
                    "nodeType": "code_search",
                    "config": {
                        "index_functions": True,
                        "index_classes": True,
                        "extract_dependencies": True
                    },
                    "inputs": [
                        {"id": "trigger", "name": "Trigger", "type": "trigger"},
                        {"id": "code_blocks", "name": "Code Blocks", "type": "array"}
                    ],
                    "outputs": [{"id": "indexed_code", "name": "Indexed Code", "type": "object"}],
                }
            },
            {
                "id": "aggregator-1",
                "type": "data_aggregator",
                "position": {"x": 1300, "y": 175},
                "data": {
                    "label": "Aggregate Results",
                    "nodeType": "data_aggregator",
                    "config": {"merge_strategy": "combine"},
                    "inputs": [
                        {"id": "graph_stats", "name": "Graph Stats", "type": "object"},
                        {"id": "indexed_code", "name": "Indexed Code", "type": "object"},
                        {"id": "indexed_count", "name": "Document Count", "type": "number"}
                    ],
                    "outputs": [{"id": "summary", "name": "Processing Summary", "type": "object"}],
                }
            },
            {
                "id": "notification-1",
                "type": "webhook",
                "position": {"x": 1500, "y": 175},
                "data": {
                    "label": "Send Completion Notification",
                    "nodeType": "webhook",
                    "config": {
                        "url": "${WEBHOOK_URL}",
                        "method": "POST",
                        "headers": {"Content-Type": "application/json"}
                    },
                    "inputs": [{"id": "data", "name": "Data", "type": "object"}],
                    "outputs": [{"id": "response", "name": "Response", "type": "object"}],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1700, "y": 175},
                "data": {
                    "label": "Batch Processing Complete",
                    "nodeType": "output",
                    "config": {"name": "batch_results"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "file-reader-1", "targetHandle": "trigger"},
            {"id": "c2", "source": "file-reader-1", "sourceHandle": "documents", "target": "entity-extract-1", "targetHandle": "text"},
            {"id": "c3", "source": "file-reader-1", "sourceHandle": "documents", "target": "code-detect-1", "targetHandle": "text"},
            {"id": "c4", "source": "entity-extract-1", "sourceHandle": "entities", "target": "graphrag-search-1", "targetHandle": "entities"},
            {"id": "c5", "source": "entity-extract-1", "sourceHandle": "relationships", "target": "graphrag-search-1", "targetHandle": "relationships"},
            {"id": "c6", "source": "file-reader-1", "sourceHandle": "documents", "target": "graphrag-search-1", "targetHandle": "documents"},
            {"id": "c7", "source": "code-detect-1", "sourceHandle": "has_code", "target": "condition-1", "targetHandle": "has_code"},
            {"id": "c8", "source": "condition-1", "sourceHandle": "process_code", "target": "code-search-1", "targetHandle": "trigger"},
            {"id": "c9", "source": "code-detect-1", "sourceHandle": "code_blocks", "target": "code-search-1", "targetHandle": "code_blocks"},
            {"id": "c10", "source": "graphrag-search-1", "sourceHandle": "graph_stats", "target": "aggregator-1", "targetHandle": "graph_stats"},
            {"id": "c11", "source": "graphrag-search-1", "sourceHandle": "indexed_count", "target": "aggregator-1", "targetHandle": "indexed_count"},
            {"id": "c12", "source": "code-search-1", "sourceHandle": "indexed_code", "target": "aggregator-1", "targetHandle": "indexed_code"},
            {"id": "c13", "source": "aggregator-1", "sourceHandle": "summary", "target": "notification-1", "targetHandle": "data"},
            {"id": "c14", "source": "notification-1", "sourceHandle": "response", "target": "output-1", "targetHandle": "value"},
        ],
    },
]


def get_workflow_templates() -> List[Dict[str, Any]]:
    """Get all available workflow templates."""
    return [
        {
            "id": t["id"],
            "name": t["name"],
            "description": t["description"],
            "category": t["category"],
            "nodeCount": len(t["nodes"]),
        }
        for t in WORKFLOW_TEMPLATES
    ]


def get_workflow_template(template_id: str) -> Dict[str, Any]:
    """Get a specific workflow template by ID."""
    for t in WORKFLOW_TEMPLATES:
        if t["id"] == template_id:
            return t
    return None










