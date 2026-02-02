/**
 * MCP (Model Context Protocol) Types
 * 
 * Type definitions for MCP servers, tools, and resources used throughout the frontend.
 */

// Transport types
export type TransportType = 'stdio' | 'http';

// MCP Server configuration
export interface MCPServerConfig {
    name: string;
    transport: TransportType;
    // Stdio transport
    command?: string;
    args?: string[];
    env?: Record<string, string>;
    // HTTP transport
    url?: string;
    auth_token?: string;
    // Common
    description?: string;
    icon?: string;
    enabled: boolean;
    auto_connect: boolean;
    created_at?: string;
    updated_at?: string;
}

// MCP Server with status
export interface MCPServer extends MCPServerConfig {
    connected: boolean;
    tools_count: number;
    resources_count: number;
    error?: string | null;
    latency_ms?: number | null;
    last_connected?: string | null;
}

// MCP Tool definition
export interface MCPTool {
    name: string;
    description?: string;
    inputSchema?: {
        type: string;
        properties?: Record<string, {
            type: string;
            description?: string;
            enum?: string[];
            default?: any;
        }>;
        required?: string[];
    };
}

// MCP Tool in OpenAI format
export interface MCPToolOpenAI {
    type: 'function';
    function: {
        name: string;
        description: string;
        parameters: {
            type: string;
            properties: Record<string, any>;
            required?: string[];
        };
    };
}

// MCP Resource
export interface MCPResource {
    uri: string;
    name?: string;
    description?: string;
    mimeType?: string;
}

// Popular server definition
export interface PopularMCPServer {
    id: string;
    name: string;
    description: string;
    icon: string;
    transport: TransportType;
    requires_path: boolean;
    requires_env: string[];
    requires_oauth: boolean;
    tools: string[];
    docs_url?: string;
    npm_package?: string;
}

// API Request types
export interface AddServerRequest {
    name: string;
    transport: TransportType;
    command?: string;
    args?: string[];
    env?: Record<string, string>;
    url?: string;
    auth_token?: string;
    description?: string;
    icon?: string;
    enabled?: boolean;
    auto_connect?: boolean;
}

export interface AddPopularServerRequest {
    server_id: string;
    path?: string;
    env_vars?: Record<string, string>;
}

export interface ExecuteToolRequest {
    arguments?: Record<string, any>;
}

// API Response types
export interface MCPStatusResponse {
    available: boolean;
    connected_servers: number;
    total_tools: number;
}

export interface ListServersResponse {
    servers: MCPServer[];
}

export interface AddServerResponse {
    success: boolean;
    server: MCPServerConfig;
    connection?: {
        connected: boolean;
        error?: string | null;
        tools_count?: number;
    };
}

export interface PopularServersResponse {
    servers: PopularMCPServer[];
}

export interface ServerToolsResponse {
    server: string;
    tools: MCPTool[];
    tools_openai_format: MCPToolOpenAI[];
}

export interface ServerResourcesResponse {
    server: string;
    resources: MCPResource[];
}

export interface ExecuteToolResponse {
    success: boolean;
    result: any;
    isError?: boolean;
}

export interface AllToolsResponse {
    tools: MCPToolOpenAI[];
    count: number;
}
