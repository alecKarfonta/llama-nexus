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
        "id": "graphrag-capability-aware-query",
        "name": "Capability-Aware GraphRAG Query",
        "description": "Inspect GraphRAG service coverage and classify a query before choosing a reasoning path",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 220},
                "data": {
                    "label": "Question",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "capabilities-1",
                "type": "graphrag_capabilities",
                "position": {"x": 350, "y": 100},
                "data": {
                    "label": "GraphRAG Capabilities",
                    "nodeType": "graphrag_capabilities",
                    "config": {},
                    "inputs": [],
                    "outputs": [
                        {"id": "enabled", "name": "Enabled", "type": "boolean"},
                        {"id": "services", "name": "Services", "type": "object"},
                        {"id": "capabilities", "name": "Capabilities", "type": "array"},
                        {"id": "capability_ids", "name": "Capability IDs", "type": "array"},
                        {"id": "capability_count", "name": "Capability Count", "type": "number"},
                        {"id": "bridge_available", "name": "Bridge Available", "type": "boolean"},
                        {"id": "allowed_prefixes", "name": "Allowed Prefixes", "type": "array"},
                    ],
                }
            },
            {
                "id": "complexity-1",
                "type": "graphrag_query_complexity",
                "position": {"x": 350, "y": 300},
                "data": {
                    "label": "Analyze Query",
                    "nodeType": "graphrag_query_complexity",
                    "config": {},
                    "inputs": [{"id": "query", "name": "Query", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "query", "name": "Query", "type": "string"},
                        {"id": "primary_reasoning", "name": "Primary Reasoning", "type": "string"},
                        {"id": "detected_patterns", "name": "Detected Patterns", "type": "array"},
                        {"id": "complexity_level", "name": "Complexity Level", "type": "string"},
                        {"id": "requires_multi_hop", "name": "Requires Multi-Hop", "type": "boolean"},
                        {"id": "analysis", "name": "Analysis", "type": "object"},
                    ],
                }
            },
            {
                "id": "context-1",
                "type": "graphrag_chat_context",
                "position": {"x": 350, "y": 500},
                "data": {
                    "label": "Build Chat Context",
                    "nodeType": "graphrag_chat_context",
                    "config": {"mode": "v2", "topK": 5},
                    "inputs": [
                        {"id": "query", "name": "Query", "type": "string", "required": True},
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "top_k", "name": "Top K", "type": "number"},
                        {"id": "domain", "name": "Domain", "type": "string"},
                        {"id": "rag_enabled", "name": "RAG Enabled", "type": "boolean"},
                        {"id": "rag_domains", "name": "RAG Domains", "type": "array"},
                        {"id": "rag_search_mode", "name": "RAG Search Mode", "type": "string"},
                        {"id": "pipeline_run_ids", "name": "Pipeline Run IDs", "type": "array"},
                    ],
                    "outputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "context_text", "name": "Context Text", "type": "string"},
                        {"id": "chunks", "name": "Chunks", "type": "array"},
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "graph_connections", "name": "Graph Connections", "type": "array"},
                        {"id": "cross_references", "name": "Cross References", "type": "array"},
                        {"id": "quality_signals", "name": "Quality Signals", "type": "object"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 650, "y": 300},
                "data": {
                    "label": "Combine Context",
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
                "position": {"x": 920, "y": 220},
                "data": {
                    "label": "Routing Context",
                    "nodeType": "output",
                    "config": {"name": "graphrag_routing_context"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "complexity-1", "targetHandle": "query"},
            {"id": "c2", "source": "trigger-1", "sourceHandle": "trigger", "target": "context-1", "targetHandle": "query"},
            {"id": "c3", "source": "capabilities-1", "sourceHandle": "capability_ids", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c4", "source": "complexity-1", "sourceHandle": "analysis", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c5", "source": "context-1", "sourceHandle": "context_text", "target": "merge-1", "targetHandle": "input3"},
            {"id": "c6", "source": "merge-1", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "graphrag-observability-quality",
        "name": "GraphRAG Observability & Quality",
        "description": "Collect capability, domain, graph, feedback, evaluation, and quality-test signals from GraphRAG",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 220},
                "data": {
                    "label": "Start Quality Check",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "capabilities-1",
                "type": "graphrag_capabilities",
                "position": {"x": 340, "y": 60},
                "data": {
                    "label": "Capabilities",
                    "nodeType": "graphrag_capabilities",
                    "config": {},
                    "inputs": [],
                    "outputs": [
                        {"id": "enabled", "name": "Enabled", "type": "boolean"},
                        {"id": "services", "name": "Services", "type": "object"},
                        {"id": "capabilities", "name": "Capabilities", "type": "array"},
                        {"id": "capability_ids", "name": "Capability IDs", "type": "array"},
                        {"id": "capability_count", "name": "Capability Count", "type": "number"},
                        {"id": "bridge_available", "name": "Bridge Available", "type": "boolean"},
                        {"id": "allowed_prefixes", "name": "Allowed Prefixes", "type": "array"},
                    ],
                }
            },
            {
                "id": "domain-stats-1",
                "type": "graphrag_domain_stats",
                "position": {"x": 340, "y": 190},
                "data": {
                    "label": "Domain Stats",
                    "nodeType": "graphrag_domain_stats",
                    "config": {},
                    "inputs": [],
                    "outputs": [
                        {"id": "domain_stats", "name": "Domain Stats", "type": "object"},
                        {"id": "domains", "name": "Domains", "type": "array"},
                        {"id": "domain_count", "name": "Domain Count", "type": "number"},
                    ],
                }
            },
            {
                "id": "instrumentation-1",
                "type": "graphrag_instrumentation",
                "position": {"x": 340, "y": 320},
                "data": {
                    "label": "Instrumentation",
                    "nodeType": "graphrag_instrumentation",
                    "config": {},
                    "inputs": [],
                    "outputs": [
                        {"id": "query_statistics", "name": "Query Statistics", "type": "object"},
                        {"id": "graph_statistics", "name": "Graph Statistics", "type": "object"},
                        {"id": "entity_centrality", "name": "Entity Centrality", "type": "object"},
                        {"id": "entity_clusters", "name": "Entity Clusters", "type": "array"},
                        {"id": "cluster_count", "name": "Cluster Count", "type": "number"},
                        {"id": "query_response", "name": "Query Response", "type": "object"},
                        {"id": "graph_response", "name": "Graph Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "evaluation-status-1",
                "type": "graphrag_evaluation_status",
                "position": {"x": 640, "y": 130},
                "data": {
                    "label": "Evaluation Readiness",
                    "nodeType": "graphrag_evaluation_status",
                    "config": {},
                    "inputs": [],
                    "outputs": [
                        {"id": "retrieval", "name": "Retrieval", "type": "object"},
                        {"id": "chunking", "name": "Chunking", "type": "object"},
                        {"id": "available", "name": "Available", "type": "boolean"},
                    ],
                }
            },
            {
                "id": "quality-suite-1",
                "type": "graphrag_test_suite",
                "position": {"x": 640, "y": 300},
                "data": {
                    "label": "Quality Suite",
                    "nodeType": "graphrag_test_suite",
                    "config": {"suite": "quality"},
                    "inputs": [{"id": "suite", "name": "Suite", "type": "string"}],
                    "outputs": [
                        {"id": "suite", "name": "Suite", "type": "string"},
                        {"id": "result", "name": "Result", "type": "object"},
                        {"id": "success", "name": "Success", "type": "boolean"},
                    ],
                }
            },
            {
                "id": "feedback-analytics-1",
                "type": "graphrag_feedback_analytics",
                "position": {"x": 640, "y": 470},
                "data": {
                    "label": "Feedback Analytics",
                    "nodeType": "graphrag_feedback_analytics",
                    "config": {"days": 30},
                    "inputs": [
                        {"id": "days", "name": "Days", "type": "number"},
                        {"id": "domain", "name": "Domain", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "period_days", "name": "Period Days", "type": "number"},
                        {"id": "domain_filter", "name": "Domain Filter", "type": "string"},
                        {"id": "overall", "name": "Overall", "type": "object"},
                        {"id": "daily_trends", "name": "Daily Trends", "type": "array"},
                        {"id": "domain_stats", "name": "Domain Stats", "type": "array"},
                        {"id": "problem_queries", "name": "Problem Queries", "type": "array"},
                        {"id": "total_feedback", "name": "Total Feedback", "type": "number"},
                        {"id": "satisfaction_rate", "name": "Satisfaction Rate", "type": "number"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 930, "y": 220},
                "data": {
                    "label": "Merge Signals",
                    "nodeType": "merge",
                    "config": {},
                    "inputs": [
                        {"id": "input1", "name": "Input 1", "type": "any"},
                        {"id": "input2", "name": "Input 2", "type": "any"},
                        {"id": "input3", "name": "Input 3", "type": "any"},
                        {"id": "input4", "name": "Input 4", "type": "any"},
                    ],
                    "outputs": [{"id": "output", "name": "Output", "type": "array"}],
                }
            },
            {
                "id": "merge-2",
                "type": "merge",
                "position": {"x": 1180, "y": 220},
                "data": {
                    "label": "Quality Summary",
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
                "position": {"x": 1450, "y": 220},
                "data": {
                    "label": "GraphRAG Quality Context",
                    "nodeType": "output",
                    "config": {"name": "graphrag_quality_context"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "domain-stats-1", "sourceHandle": "domain_stats", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c2", "source": "instrumentation-1", "sourceHandle": "graph_response", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c3", "source": "evaluation-status-1", "sourceHandle": "available", "target": "merge-1", "targetHandle": "input3"},
            {"id": "c4", "source": "feedback-analytics-1", "sourceHandle": "overall", "target": "merge-1", "targetHandle": "input4"},
            {"id": "c5", "source": "capabilities-1", "sourceHandle": "capabilities", "target": "merge-2", "targetHandle": "input1"},
            {"id": "c6", "source": "quality-suite-1", "sourceHandle": "result", "target": "merge-2", "targetHandle": "input2"},
            {"id": "c7", "source": "merge-1", "sourceHandle": "output", "target": "merge-2", "targetHandle": "input3"},
            {"id": "c8", "source": "merge-2", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "graphrag-extraction-diagnostics",
        "name": "GraphRAG Extraction Diagnostics",
        "description": "Compare NER, relationship extraction, enhanced extraction, and diagnostic extraction on sample text",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 220},
                "data": {
                    "label": "Sample Text",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "ner-1",
                "type": "graphrag_ner_extraction",
                "position": {"x": 360, "y": 70},
                "data": {
                    "label": "NER Extraction",
                    "nodeType": "graphrag_ner_extraction",
                    "config": {},
                    "inputs": [
                        {"id": "text", "name": "Text", "type": "string"},
                        {"id": "texts", "name": "Texts", "type": "array"},
                    ],
                    "outputs": [
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "entity_count", "name": "Entity Count", "type": "number"},
                        {"id": "results", "name": "Results", "type": "array"},
                        {"id": "total_texts", "name": "Total Texts", "type": "number"},
                        {"id": "extraction_method", "name": "Extraction Method", "type": "string"},
                    ],
                }
            },
            {
                "id": "rel-1",
                "type": "graphrag_relation_extraction",
                "position": {"x": 360, "y": 220},
                "data": {
                    "label": "Relationship Extraction",
                    "nodeType": "graphrag_relation_extraction",
                    "config": {"threshold": 0.5},
                    "inputs": [
                        {"id": "text", "name": "Text", "type": "string", "required": True},
                        {"id": "relations", "name": "Relations", "type": "array"},
                        {"id": "entity_labels", "name": "Entity Labels", "type": "array"},
                        {"id": "threshold", "name": "Threshold", "type": "number"},
                    ],
                    "outputs": [
                        {"id": "relations", "name": "Relations", "type": "array"},
                        {"id": "relation_count", "name": "Relation Count", "type": "number"},
                        {"id": "processing_time", "name": "Processing Time", "type": "number"},
                        {"id": "model_info", "name": "Model Info", "type": "object"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "enhanced-1",
                "type": "enhanced_entity_extraction",
                "position": {"x": 360, "y": 380},
                "data": {
                    "label": "Enhanced Extraction",
                    "nodeType": "enhanced_entity_extraction",
                    "config": {"useSpanbert": True, "useDependency": True, "useEntityLinking": True},
                    "inputs": [
                        {"id": "text", "name": "Text", "type": "string", "required": True},
                        {"id": "domain", "name": "Domain", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "entity_count", "name": "Entity Count", "type": "number"},
                        {"id": "relationship_count", "name": "Relationship Count", "type": "number"},
                        {"id": "claims", "name": "Claims", "type": "array"},
                        {"id": "extraction_metadata", "name": "Extraction Metadata", "type": "object"},
                        {"id": "linked_entities", "name": "Linked Entities", "type": "array"},
                    ],
                }
            },
            {
                "id": "diagnostic-1",
                "type": "graphrag_enhanced_extraction_test",
                "position": {"x": 690, "y": 220},
                "data": {
                    "label": "Extraction Diagnostic",
                    "nodeType": "graphrag_enhanced_extraction_test",
                    "config": {},
                    "inputs": [
                        {"id": "text", "name": "Text", "type": "string", "required": True},
                        {"id": "domain", "name": "Domain", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "success", "name": "Success", "type": "boolean"},
                        {"id": "message", "name": "Message", "type": "string"},
                        {"id": "entities_found", "name": "Entities Found", "type": "number"},
                        {"id": "relationships_found", "name": "Relationships Found", "type": "number"},
                        {"id": "extraction_methods_used", "name": "Methods Used", "type": "array"},
                        {"id": "sample_entities", "name": "Sample Entities", "type": "array"},
                        {"id": "sample_relationships", "name": "Sample Relationships", "type": "array"},
                        {"id": "result", "name": "Result", "type": "object"},
                    ],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 990, "y": 220},
                "data": {
                    "label": "Compare Extraction Outputs",
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
                "id": "merge-2",
                "type": "merge",
                "position": {"x": 1240, "y": 220},
                "data": {
                    "label": "Diagnostic Summary",
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
                "position": {"x": 1510, "y": 220},
                "data": {
                    "label": "Extraction Diagnostics",
                    "nodeType": "output",
                    "config": {"name": "graphrag_extraction_diagnostics"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "ner-1", "targetHandle": "text"},
            {"id": "c2", "source": "trigger-1", "sourceHandle": "trigger", "target": "rel-1", "targetHandle": "text"},
            {"id": "c3", "source": "trigger-1", "sourceHandle": "trigger", "target": "enhanced-1", "targetHandle": "text"},
            {"id": "c4", "source": "trigger-1", "sourceHandle": "trigger", "target": "diagnostic-1", "targetHandle": "text"},
            {"id": "c5", "source": "ner-1", "sourceHandle": "entities", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c6", "source": "rel-1", "sourceHandle": "relations", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c7", "source": "enhanced-1", "sourceHandle": "extraction_metadata", "target": "merge-1", "targetHandle": "input3"},
            {"id": "c8", "source": "diagnostic-1", "sourceHandle": "result", "target": "merge-2", "targetHandle": "input1"},
            {"id": "c9", "source": "merge-1", "sourceHandle": "output", "target": "merge-2", "targetHandle": "input2"},
            {"id": "c10", "source": "merge-2", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "graphrag-social-memory-context",
        "name": "GraphRAG Social Memory Context",
        "description": "Ingest a conversation turn and retrieve personalized social-memory context for downstream answers",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 220},
                "data": {
                    "label": "Conversation Turn",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "memory-ingest-1",
                "type": "graphrag_social_memory_ingest",
                "position": {"x": 380, "y": 120},
                "data": {
                    "label": "Store Message",
                    "nodeType": "graphrag_social_memory_ingest",
                    "config": {"platform": "twitch", "sync": True, "priority": 3},
                    "inputs": [
                        {"id": "platform", "name": "Platform", "type": "string"},
                        {"id": "platform_user_id", "name": "Platform User ID", "type": "string", "required": True},
                        {"id": "display_name", "name": "Display Name", "type": "string"},
                        {"id": "text", "name": "Text", "type": "string", "required": True},
                        {"id": "timestamp", "name": "Timestamp", "type": "string"},
                        {"id": "message_id", "name": "Message ID", "type": "string"},
                        {"id": "stream_id", "name": "Stream ID", "type": "string"},
                        {"id": "session_id", "name": "Session ID", "type": "string"},
                        {"id": "previous_messages", "name": "Previous Messages", "type": "array"},
                        {"id": "priority", "name": "Priority", "type": "number"},
                        {"id": "sync", "name": "Sync", "type": "boolean"},
                    ],
                    "outputs": [
                        {"id": "person_id", "name": "Person ID", "type": "string"},
                        {"id": "episode_id", "name": "Episode ID", "type": "string"},
                        {"id": "facts_written", "name": "Facts Written", "type": "number"},
                        {"id": "queued", "name": "Queued", "type": "boolean"},
                        {"id": "skipped_duplicate", "name": "Skipped Duplicate", "type": "boolean"},
                        {"id": "queue_message_id", "name": "Queue Message ID", "type": "string"},
                        {"id": "message", "name": "Message", "type": "object"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "memory-context-1",
                "type": "graphrag_social_memory_context",
                "position": {"x": 380, "y": 360},
                "data": {
                    "label": "Retrieve Context",
                    "nodeType": "graphrag_social_memory_context",
                    "config": {"platform": "twitch", "mode": "context", "maxFacts": 25, "includeRecentEpisodes": 3},
                    "inputs": [
                        {"id": "platform", "name": "Platform", "type": "string"},
                        {"id": "platform_user_id", "name": "Platform User ID", "type": "string", "required": True},
                        {"id": "query", "name": "Query", "type": "string"},
                        {"id": "max_facts", "name": "Max Facts", "type": "number"},
                        {"id": "include_recent_episodes", "name": "Recent Episodes", "type": "number"},
                        {"id": "as_of", "name": "As Of", "type": "string"},
                        {"id": "mode", "name": "Mode", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "person_id", "name": "Person ID", "type": "string"},
                        {"id": "formatted", "name": "Formatted", "type": "string"},
                        {"id": "facts", "name": "Facts", "type": "array"},
                        {"id": "recent_episodes", "name": "Recent Episodes", "type": "array"},
                        {"id": "retrieval_path", "name": "Retrieval Path", "type": "string"},
                        {"id": "cache_hit", "name": "Cache Hit", "type": "boolean"},
                        {"id": "fact_ids", "name": "Fact IDs", "type": "array"},
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "trace", "name": "Trace", "type": "object"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "memory-facts-1",
                "type": "graphrag_social_memory_facts",
                "position": {"x": 720, "y": 120},
                "data": {
                    "label": "Person Facts",
                    "nodeType": "graphrag_social_memory_facts",
                    "config": {},
                    "inputs": [
                        {"id": "person_id", "name": "Person ID", "type": "string", "required": True},
                        {"id": "as_of", "name": "As Of", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "person_id", "name": "Person ID", "type": "string"},
                        {"id": "facts", "name": "Facts", "type": "array"},
                        {"id": "fact_count", "name": "Fact Count", "type": "number"},
                        {"id": "as_of", "name": "As Of", "type": "string"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 980, "y": 240},
                "data": {
                    "label": "Memory Bundle",
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
                "position": {"x": 1270, "y": 240},
                "data": {
                    "label": "Personal Context",
                    "nodeType": "output",
                    "config": {"name": "graphrag_social_memory_context"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "memory-ingest-1", "targetHandle": "text"},
            {"id": "c2", "source": "trigger-1", "sourceHandle": "trigger", "target": "memory-context-1", "targetHandle": "query"},
            {"id": "c3", "source": "memory-ingest-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c4", "source": "memory-context-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c5", "source": "memory-ingest-1", "sourceHandle": "person_id", "target": "memory-facts-1", "targetHandle": "person_id"},
            {"id": "c6", "source": "memory-facts-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input3"},
            {"id": "c7", "source": "merge-1", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "graphrag-knowledge-graph-explorer",
        "name": "GraphRAG Knowledge Graph Explorer",
        "description": "Collect filtered graph data, entity neighborhoods, communities, top entities, and top relationships with upstream graph filters",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 240},
                "data": {
                    "label": "Graph Filters",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "filtered-graph-1",
                "type": "graphrag_graph_explorer",
                "position": {"x": 380, "y": 80},
                "data": {
                    "label": "Filtered Graph",
                    "nodeType": "graphrag_graph_explorer",
                    "config": {"mode": "filtered_graph"},
                    "inputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "domain", "name": "Domain", "type": "string"},
                        {"id": "entity_name", "name": "Entity Name", "type": "string"},
                        {"id": "entity", "name": "Entity", "type": "string"},
                        {"id": "max_entities", "name": "Max Entities", "type": "number"},
                        {"id": "max_relationships", "name": "Max Relationships", "type": "number"},
                        {"id": "min_occurrence", "name": "Min Occurrence", "type": "number"},
                        {"id": "min_confidence", "name": "Min Confidence", "type": "number"},
                        {"id": "entity_types", "name": "Entity Types", "type": "array"},
                        {"id": "relationship_types", "name": "Relationship Types", "type": "array"},
                        {"id": "entity_type", "name": "Entity Type", "type": "string"},
                        {"id": "relationship_type", "name": "Relationship Type", "type": "string"},
                        {"id": "limit", "name": "Limit", "type": "number"},
                        {"id": "min_weight", "name": "Min Weight", "type": "number"},
                    ],
                    "outputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "nodes", "name": "Nodes", "type": "array"},
                        {"id": "edges", "name": "Edges", "type": "array"},
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "communities", "name": "Communities", "type": "array"},
                        {"id": "community_count", "name": "Community Count", "type": "number"},
                        {"id": "neighborhood", "name": "Neighborhood", "type": "object"},
                        {"id": "stats", "name": "Stats", "type": "object"},
                        {"id": "result", "name": "Result", "type": "object"},
                    ],
                }
            },
            {
                "id": "top-entities-1",
                "type": "graphrag_graph_explorer",
                "position": {"x": 380, "y": 260},
                "data": {
                    "label": "Top Entities",
                    "nodeType": "graphrag_graph_explorer",
                    "config": {"mode": "top_entities"},
                    "inputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "domain", "name": "Domain", "type": "string"},
                        {"id": "entity_name", "name": "Entity Name", "type": "string"},
                        {"id": "entity", "name": "Entity", "type": "string"},
                        {"id": "max_entities", "name": "Max Entities", "type": "number"},
                        {"id": "max_relationships", "name": "Max Relationships", "type": "number"},
                        {"id": "min_occurrence", "name": "Min Occurrence", "type": "number"},
                        {"id": "min_confidence", "name": "Min Confidence", "type": "number"},
                        {"id": "entity_types", "name": "Entity Types", "type": "array"},
                        {"id": "relationship_types", "name": "Relationship Types", "type": "array"},
                        {"id": "entity_type", "name": "Entity Type", "type": "string"},
                        {"id": "relationship_type", "name": "Relationship Type", "type": "string"},
                        {"id": "limit", "name": "Limit", "type": "number"},
                        {"id": "min_weight", "name": "Min Weight", "type": "number"},
                    ],
                    "outputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "nodes", "name": "Nodes", "type": "array"},
                        {"id": "edges", "name": "Edges", "type": "array"},
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "communities", "name": "Communities", "type": "array"},
                        {"id": "community_count", "name": "Community Count", "type": "number"},
                        {"id": "neighborhood", "name": "Neighborhood", "type": "object"},
                        {"id": "stats", "name": "Stats", "type": "object"},
                        {"id": "result", "name": "Result", "type": "object"},
                    ],
                }
            },
            {
                "id": "top-relationships-1",
                "type": "graphrag_graph_explorer",
                "position": {"x": 380, "y": 440},
                "data": {
                    "label": "Top Relationships",
                    "nodeType": "graphrag_graph_explorer",
                    "config": {"mode": "top_relationships"},
                    "inputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "domain", "name": "Domain", "type": "string"},
                        {"id": "entity_name", "name": "Entity Name", "type": "string"},
                        {"id": "entity", "name": "Entity", "type": "string"},
                        {"id": "max_entities", "name": "Max Entities", "type": "number"},
                        {"id": "max_relationships", "name": "Max Relationships", "type": "number"},
                        {"id": "min_occurrence", "name": "Min Occurrence", "type": "number"},
                        {"id": "min_confidence", "name": "Min Confidence", "type": "number"},
                        {"id": "entity_types", "name": "Entity Types", "type": "array"},
                        {"id": "relationship_types", "name": "Relationship Types", "type": "array"},
                        {"id": "entity_type", "name": "Entity Type", "type": "string"},
                        {"id": "relationship_type", "name": "Relationship Type", "type": "string"},
                        {"id": "limit", "name": "Limit", "type": "number"},
                        {"id": "min_weight", "name": "Min Weight", "type": "number"},
                    ],
                    "outputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "nodes", "name": "Nodes", "type": "array"},
                        {"id": "edges", "name": "Edges", "type": "array"},
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "communities", "name": "Communities", "type": "array"},
                        {"id": "community_count", "name": "Community Count", "type": "number"},
                        {"id": "neighborhood", "name": "Neighborhood", "type": "object"},
                        {"id": "stats", "name": "Stats", "type": "object"},
                        {"id": "result", "name": "Result", "type": "object"},
                    ],
                }
            },
            {
                "id": "entity-neighborhood-1",
                "type": "graphrag_graph_explorer",
                "position": {"x": 380, "y": 620},
                "data": {
                    "label": "Entity Neighborhood",
                    "nodeType": "graphrag_graph_explorer",
                    "config": {"mode": "entity_neighborhood", "entityName": "GraphRAG", "limit": 20},
                    "inputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "domain", "name": "Domain", "type": "string"},
                        {"id": "entity_name", "name": "Entity Name", "type": "string"},
                        {"id": "entity", "name": "Entity", "type": "string"},
                        {"id": "max_entities", "name": "Max Entities", "type": "number"},
                        {"id": "max_relationships", "name": "Max Relationships", "type": "number"},
                        {"id": "min_occurrence", "name": "Min Occurrence", "type": "number"},
                        {"id": "min_confidence", "name": "Min Confidence", "type": "number"},
                        {"id": "entity_types", "name": "Entity Types", "type": "array"},
                        {"id": "relationship_types", "name": "Relationship Types", "type": "array"},
                        {"id": "entity_type", "name": "Entity Type", "type": "string"},
                        {"id": "relationship_type", "name": "Relationship Type", "type": "string"},
                        {"id": "limit", "name": "Limit", "type": "number"},
                        {"id": "min_weight", "name": "Min Weight", "type": "number"},
                    ],
                    "outputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "nodes", "name": "Nodes", "type": "array"},
                        {"id": "edges", "name": "Edges", "type": "array"},
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "communities", "name": "Communities", "type": "array"},
                        {"id": "community_count", "name": "Community Count", "type": "number"},
                        {"id": "neighborhood", "name": "Neighborhood", "type": "object"},
                        {"id": "stats", "name": "Stats", "type": "object"},
                        {"id": "result", "name": "Result", "type": "object"},
                    ],
                }
            },
            {
                "id": "communities-1",
                "type": "graphrag_graph_explorer",
                "position": {"x": 380, "y": 800},
                "data": {
                    "label": "Communities",
                    "nodeType": "graphrag_graph_explorer",
                    "config": {"mode": "communities"},
                    "inputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "domain", "name": "Domain", "type": "string"},
                        {"id": "entity_name", "name": "Entity Name", "type": "string"},
                        {"id": "entity", "name": "Entity", "type": "string"},
                        {"id": "max_entities", "name": "Max Entities", "type": "number"},
                        {"id": "max_relationships", "name": "Max Relationships", "type": "number"},
                        {"id": "min_occurrence", "name": "Min Occurrence", "type": "number"},
                        {"id": "min_confidence", "name": "Min Confidence", "type": "number"},
                        {"id": "entity_types", "name": "Entity Types", "type": "array"},
                        {"id": "relationship_types", "name": "Relationship Types", "type": "array"},
                        {"id": "entity_type", "name": "Entity Type", "type": "string"},
                        {"id": "relationship_type", "name": "Relationship Type", "type": "string"},
                        {"id": "limit", "name": "Limit", "type": "number"},
                        {"id": "min_weight", "name": "Min Weight", "type": "number"},
                    ],
                    "outputs": [
                        {"id": "mode", "name": "Mode", "type": "string"},
                        {"id": "filters", "name": "Filters", "type": "object"},
                        {"id": "nodes", "name": "Nodes", "type": "array"},
                        {"id": "edges", "name": "Edges", "type": "array"},
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "communities", "name": "Communities", "type": "array"},
                        {"id": "community_count", "name": "Community Count", "type": "number"},
                        {"id": "neighborhood", "name": "Neighborhood", "type": "object"},
                        {"id": "stats", "name": "Stats", "type": "object"},
                        {"id": "result", "name": "Result", "type": "object"},
                    ],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 760, "y": 440},
                "data": {
                    "label": "Graph Signals",
                    "nodeType": "merge",
                    "config": {},
                    "inputs": [
                        {"id": "input1", "name": "Input 1", "type": "any"},
                        {"id": "input2", "name": "Input 2", "type": "any"},
                        {"id": "input3", "name": "Input 3", "type": "any"},
                        {"id": "input4", "name": "Input 4", "type": "any"},
                        {"id": "input5", "name": "Input 5", "type": "any"},
                    ],
                    "outputs": [{"id": "output", "name": "Output", "type": "array"}],
                }
            },
            {
                "id": "output-1",
                "type": "output",
                "position": {"x": 1040, "y": 440},
                "data": {
                    "label": "Graph Explorer",
                    "nodeType": "output",
                    "config": {"name": "graphrag_knowledge_graph_explorer"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "filtered-graph-1", "targetHandle": "filters"},
            {"id": "c2", "source": "trigger-1", "sourceHandle": "trigger", "target": "top-entities-1", "targetHandle": "filters"},
            {"id": "c3", "source": "trigger-1", "sourceHandle": "trigger", "target": "top-relationships-1", "targetHandle": "filters"},
            {"id": "c4", "source": "trigger-1", "sourceHandle": "trigger", "target": "entity-neighborhood-1", "targetHandle": "filters"},
            {"id": "c5", "source": "trigger-1", "sourceHandle": "trigger", "target": "communities-1", "targetHandle": "filters"},
            {"id": "c6", "source": "filtered-graph-1", "sourceHandle": "result", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c7", "source": "top-entities-1", "sourceHandle": "entities", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c8", "source": "top-relationships-1", "sourceHandle": "relationships", "target": "merge-1", "targetHandle": "input3"},
            {"id": "c9", "source": "entity-neighborhood-1", "sourceHandle": "neighborhood", "target": "merge-1", "targetHandle": "input4"},
            {"id": "c10", "source": "communities-1", "sourceHandle": "communities", "target": "merge-1", "targetHandle": "input5"},
            {"id": "c11", "source": "merge-1", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "graphrag-code-rag-indexing",
        "name": "GraphRAG Code-RAG Indexing",
        "description": "Index code paths, inspect ingestion progress, analyze a file, and collect Code-RAG statistics",
        "category": "tools",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 240},
                "data": {
                    "label": "Code Paths",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "code-ingest-1",
                "type": "graphrag_code_ingest",
                "position": {"x": 360, "y": 90},
                "data": {
                    "label": "Index Code Paths",
                    "nodeType": "graphrag_code_ingest",
                    "config": {"projectName": "default_project"},
                    "inputs": [
                        {"id": "file_paths", "name": "File Paths", "type": "array", "required": True},
                        {"id": "project_name", "name": "Project Name", "type": "string"},
                        {"id": "language", "name": "Language", "type": "string"},
                    ],
                    "outputs": [
                        {"id": "task_id", "name": "Task ID", "type": "string"},
                        {"id": "status", "name": "Status", "type": "string"},
                        {"id": "message", "name": "Message", "type": "string"},
                        {"id": "project_name", "name": "Project Name", "type": "string"},
                        {"id": "file_count", "name": "File Count", "type": "number"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "ingest-status-1",
                "type": "graphrag_code_ingestion_status",
                "position": {"x": 660, "y": 90},
                "data": {
                    "label": "Ingestion Status",
                    "nodeType": "graphrag_code_ingestion_status",
                    "config": {},
                    "inputs": [{"id": "task_id", "name": "Task ID", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "task_id", "name": "Task ID", "type": "string"},
                        {"id": "status", "name": "Status", "type": "string"},
                        {"id": "progress", "name": "Progress", "type": "number"},
                        {"id": "processed_files", "name": "Processed Files", "type": "number"},
                        {"id": "total_files", "name": "Total Files", "type": "number"},
                        {"id": "entities_found", "name": "Entities Found", "type": "number"},
                        {"id": "error", "name": "Error", "type": "string"},
                        {"id": "elapsed_time", "name": "Elapsed Time", "type": "number"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "code-analysis-1",
                "type": "graphrag_code_analysis",
                "position": {"x": 360, "y": 300},
                "data": {
                    "label": "Analyze File",
                    "nodeType": "graphrag_code_analysis",
                    "config": {},
                    "inputs": [{"id": "file_path", "name": "File Path", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "file_path", "name": "File Path", "type": "string"},
                        {"id": "language", "name": "Language", "type": "string"},
                        {"id": "success", "name": "Success", "type": "boolean"},
                        {"id": "entities", "name": "Entities", "type": "array"},
                        {"id": "relationships", "name": "Relationships", "type": "array"},
                        {"id": "entity_count", "name": "Entity Count", "type": "number"},
                        {"id": "relationship_count", "name": "Relationship Count", "type": "number"},
                        {"id": "parse_time_ms", "name": "Parse Time", "type": "number"},
                        {"id": "error", "name": "Error", "type": "string"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "code-stats-1",
                "type": "graphrag_code_stats",
                "position": {"x": 660, "y": 300},
                "data": {
                    "label": "Code-RAG Stats",
                    "nodeType": "graphrag_code_stats",
                    "config": {},
                    "inputs": [],
                    "outputs": [
                        {"id": "search_engine", "name": "Search Engine", "type": "object"},
                        {"id": "active_tasks", "name": "Active Tasks", "type": "number"},
                        {"id": "completed_tasks", "name": "Completed Tasks", "type": "number"},
                        {"id": "failed_tasks", "name": "Failed Tasks", "type": "number"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 980, "y": 210},
                "data": {
                    "label": "Code-RAG Bundle",
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
                "position": {"x": 1250, "y": 210},
                "data": {
                    "label": "Code-RAG Indexing",
                    "nodeType": "output",
                    "config": {"name": "graphrag_code_rag_indexing"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "code-ingest-1", "targetHandle": "file_paths"},
            {"id": "c2", "source": "code-ingest-1", "sourceHandle": "task_id", "target": "ingest-status-1", "targetHandle": "task_id"},
            {"id": "c3", "source": "trigger-1", "sourceHandle": "trigger", "target": "code-analysis-1", "targetHandle": "file_path"},
            {"id": "c4", "source": "ingest-status-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c5", "source": "code-analysis-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c6", "source": "code-stats-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input3"},
            {"id": "c7", "source": "merge-1", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "graphrag-reddit-research-ingestion",
        "name": "GraphRAG Reddit Research Ingestion",
        "description": "Crawl Reddit discussions, auto-ingest them into GraphRAG, and collect crawl status and results",
        "category": "rag",
        "nodes": [
            {
                "id": "trigger-1",
                "type": "manual_trigger",
                "position": {"x": 100, "y": 220},
                "data": {
                    "label": "Research Query",
                    "nodeType": "manual_trigger",
                    "config": {},
                    "inputs": [],
                    "outputs": [{"id": "trigger", "name": "Trigger", "type": "any"}],
                }
            },
            {
                "id": "reddit-crawl-1",
                "type": "graphrag_reddit_crawl",
                "position": {"x": 360, "y": 120},
                "data": {
                    "label": "Start Reddit Crawl",
                    "nodeType": "graphrag_reddit_crawl",
                    "config": {
                        "subreddits": ["LocalLLaMA", "MachineLearning"],
                        "sortBy": "relevance",
                        "timeFilter": "month",
                        "maxPosts": 50,
                        "maxCommentsPerPost": 20,
                        "commentDepth": 2,
                        "includeReplies": True,
                        "autoIngest": True,
                        "filterNsfw": True,
                    },
                    "inputs": [
                        {"id": "query", "name": "Query", "type": "string", "required": True},
                        {"id": "subreddits", "name": "Subreddits", "type": "array"},
                        {"id": "sort_by", "name": "Sort By", "type": "string"},
                        {"id": "time_filter", "name": "Time Filter", "type": "string"},
                        {"id": "max_posts", "name": "Max Posts", "type": "number"},
                        {"id": "max_comments_per_post", "name": "Max Comments", "type": "number"},
                        {"id": "comment_depth", "name": "Comment Depth", "type": "number"},
                        {"id": "include_replies", "name": "Include Replies", "type": "boolean"},
                        {"id": "auto_ingest", "name": "Auto Ingest", "type": "boolean"},
                        {"id": "filter_nsfw", "name": "Filter NSFW", "type": "boolean"},
                    ],
                    "outputs": [
                        {"id": "crawl_id", "name": "Crawl ID", "type": "string"},
                        {"id": "status", "name": "Status", "type": "string"},
                        {"id": "posts_found", "name": "Posts Found", "type": "number"},
                        {"id": "comments_found", "name": "Comments Found", "type": "number"},
                        {"id": "total_content_length", "name": "Content Length", "type": "number"},
                        {"id": "estimated_processing_time", "name": "Estimated Time", "type": "string"},
                        {"id": "ingestion_status", "name": "Ingestion Status", "type": "string"},
                        {"id": "query", "name": "Query", "type": "string"},
                        {"id": "subreddits", "name": "Subreddits", "type": "array"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "crawl-status-1",
                "type": "graphrag_reddit_crawl_status",
                "position": {"x": 660, "y": 70},
                "data": {
                    "label": "Crawl Status",
                    "nodeType": "graphrag_reddit_crawl_status",
                    "config": {},
                    "inputs": [{"id": "crawl_id", "name": "Crawl ID", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "crawl_id", "name": "Crawl ID", "type": "string"},
                        {"id": "status", "name": "Status", "type": "string"},
                        {"id": "progress", "name": "Progress", "type": "number"},
                        {"id": "posts_processed", "name": "Posts Processed", "type": "number"},
                        {"id": "comments_processed", "name": "Comments Processed", "type": "number"},
                        {"id": "current_post", "name": "Current Post", "type": "string"},
                        {"id": "error", "name": "Error", "type": "string"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "crawl-results-1",
                "type": "graphrag_reddit_crawl_results",
                "position": {"x": 660, "y": 300},
                "data": {
                    "label": "Crawl Results",
                    "nodeType": "graphrag_reddit_crawl_results",
                    "config": {},
                    "inputs": [{"id": "crawl_id", "name": "Crawl ID", "type": "string", "required": True}],
                    "outputs": [
                        {"id": "crawl_id", "name": "Crawl ID", "type": "string"},
                        {"id": "status", "name": "Status", "type": "string"},
                        {"id": "results", "name": "Results", "type": "array"},
                        {"id": "result_count", "name": "Result Count", "type": "number"},
                        {"id": "statistics", "name": "Statistics", "type": "object"},
                        {"id": "ingestion_status", "name": "Ingestion Status", "type": "string"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 980, "y": 200},
                "data": {
                    "label": "Reddit Crawl Bundle",
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
                "position": {"x": 1250, "y": 200},
                "data": {
                    "label": "Reddit GraphRAG Crawl",
                    "nodeType": "output",
                    "config": {"name": "graphrag_reddit_research_ingestion"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "trigger-1", "sourceHandle": "trigger", "target": "reddit-crawl-1", "targetHandle": "query"},
            {"id": "c2", "source": "reddit-crawl-1", "sourceHandle": "crawl_id", "target": "crawl-status-1", "targetHandle": "crawl_id"},
            {"id": "c3", "source": "reddit-crawl-1", "sourceHandle": "crawl_id", "target": "crawl-results-1", "targetHandle": "crawl_id"},
            {"id": "c4", "source": "reddit-crawl-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c5", "source": "crawl-status-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c6", "source": "crawl-results-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input3"},
            {"id": "c7", "source": "merge-1", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
        ],
    },
    {
        "id": "graphrag-corpus-maintenance",
        "name": "GraphRAG Corpus Maintenance",
        "description": "List indexed GraphRAG documents and rebuild the knowledge graph for a corpus",
        "category": "rag",
        "nodes": [
            {
                "id": "list-documents-1",
                "type": "graphrag_document_list",
                "position": {"x": 180, "y": 100},
                "data": {
                    "label": "List Documents",
                    "nodeType": "graphrag_document_list",
                    "config": {},
                    "inputs": [],
                    "outputs": [
                        {"id": "documents", "name": "Documents", "type": "array"},
                        {"id": "total_documents", "name": "Total Documents", "type": "number"},
                        {"id": "vector_store_documents", "name": "Vector Documents", "type": "number"},
                        {"id": "knowledge_graph_documents", "name": "Graph Documents", "type": "number"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "rebuild-kg-1",
                "type": "graphrag_kg_rebuild",
                "position": {"x": 180, "y": 330},
                "data": {
                    "label": "Rebuild Knowledge Graph",
                    "nodeType": "graphrag_kg_rebuild",
                    "config": {"domain": "general"},
                    "inputs": [{"id": "domain", "name": "Domain", "type": "string"}],
                    "outputs": [
                        {"id": "domain", "name": "Domain", "type": "string"},
                        {"id": "message", "name": "Message", "type": "string"},
                        {"id": "documents_processed", "name": "Documents Processed", "type": "number"},
                        {"id": "entities_extracted", "name": "Entities Extracted", "type": "number"},
                        {"id": "relationships_extracted", "name": "Relationships Extracted", "type": "number"},
                        {"id": "response", "name": "Response", "type": "object"},
                    ],
                }
            },
            {
                "id": "domain-stats-1",
                "type": "graphrag_domain_stats",
                "position": {"x": 500, "y": 220},
                "data": {
                    "label": "Domain Stats",
                    "nodeType": "graphrag_domain_stats",
                    "config": {},
                    "inputs": [],
                    "outputs": [
                        {"id": "domain_stats", "name": "Domain Stats", "type": "object"},
                        {"id": "domains", "name": "Domains", "type": "array"},
                        {"id": "domain_count", "name": "Domain Count", "type": "number"},
                    ],
                }
            },
            {
                "id": "merge-1",
                "type": "merge",
                "position": {"x": 820, "y": 220},
                "data": {
                    "label": "Corpus Maintenance Bundle",
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
                "position": {"x": 1110, "y": 220},
                "data": {
                    "label": "Corpus Maintenance",
                    "nodeType": "output",
                    "config": {"name": "graphrag_corpus_maintenance"},
                    "inputs": [{"id": "value", "name": "Value", "type": "any", "required": True}],
                    "outputs": [],
                }
            },
        ],
        "connections": [
            {"id": "c1", "source": "list-documents-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input1"},
            {"id": "c2", "source": "rebuild-kg-1", "sourceHandle": "response", "target": "merge-1", "targetHandle": "input2"},
            {"id": "c3", "source": "domain-stats-1", "sourceHandle": "domain_stats", "target": "merge-1", "targetHandle": "input3"},
            {"id": "c4", "source": "merge-1", "sourceHandle": "output", "target": "output-1", "targetHandle": "value"},
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
