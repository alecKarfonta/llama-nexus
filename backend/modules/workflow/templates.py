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








