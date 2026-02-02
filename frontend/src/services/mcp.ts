/**
 * MCP Service
 * 
 * API service for managing MCP (Model Context Protocol) server connections,
 * listing tools, and executing tool calls.
 */

import { apiService } from './api';
import type {
    MCPStatusResponse,
    ListServersResponse,
    AddServerRequest,
    AddServerResponse,
    AddPopularServerRequest,
    PopularServersResponse,
    ServerToolsResponse,
    ServerResourcesResponse,
    ExecuteToolRequest,
    ExecuteToolResponse,
    AllToolsResponse,
    MCPServer,
} from '@/types/mcp';

class MCPService {
    private baseUrl = '/api/v1/mcp';

    /**
     * Get overall MCP module status
     */
    async getStatus(): Promise<MCPStatusResponse> {
        const response = await apiService.get(`${this.baseUrl}/status`);
        return response;
    }

    /**
     * List all configured MCP servers with their status
     */
    async listServers(): Promise<MCPServer[]> {
        const response: ListServersResponse = await apiService.get(`${this.baseUrl}/servers`);
        return response.servers;
    }

    /**
     * Add a new MCP server configuration
     */
    async addServer(config: AddServerRequest): Promise<AddServerResponse> {
        return apiService.post(`${this.baseUrl}/servers`, config);
    }

    /**
     * Add a popular pre-configured MCP server
     */
    async addPopularServer(request: AddPopularServerRequest): Promise<AddServerResponse> {
        return apiService.post(`${this.baseUrl}/servers/popular`, request);
    }

    /**
     * List available popular MCP servers
     */
    async listPopularServers(): Promise<PopularServersResponse> {
        return apiService.get(`${this.baseUrl}/servers/popular`);
    }

    /**
     * Get a specific server's configuration and status
     */
    async getServer(name: string): Promise<{ config: any; status: any }> {
        return apiService.get(`${this.baseUrl}/servers/${name}`);
    }

    /**
     * Update a server's configuration
     */
    async updateServer(name: string, updates: Partial<AddServerRequest>): Promise<{ success: boolean; server: any }> {
        return apiService.patch(`${this.baseUrl}/servers/${name}`, updates);
    }

    /**
     * Delete a server configuration
     */
    async deleteServer(name: string): Promise<{ success: boolean }> {
        return apiService.delete(`${this.baseUrl}/servers/${name}`);
    }

    /**
     * Connect to an MCP server
     */
    async connectServer(name: string): Promise<{
        success: boolean;
        connected: boolean;
        error?: string;
        tools_count?: number;
        resources_count?: number;
        latency_ms?: number;
    }> {
        return apiService.post(`${this.baseUrl}/servers/${name}/connect`);
    }

    /**
     * Disconnect from an MCP server
     */
    async disconnectServer(name: string): Promise<{ success: boolean }> {
        return apiService.post(`${this.baseUrl}/servers/${name}/disconnect`);
    }

    /**
     * List tools available from a connected MCP server
     */
    async listServerTools(name: string): Promise<ServerToolsResponse> {
        return apiService.get(`${this.baseUrl}/servers/${name}/tools`);
    }

    /**
     * List resources available from a connected MCP server
     */
    async listServerResources(name: string): Promise<ServerResourcesResponse> {
        return apiService.get(`${this.baseUrl}/servers/${name}/resources`);
    }

    /**
     * Execute a tool on a connected MCP server
     */
    async executeTool(serverName: string, toolName: string, args: Record<string, any> = {}): Promise<ExecuteToolResponse> {
        return apiService.post(`${this.baseUrl}/servers/${serverName}/tools/${toolName}`, { arguments: args });
    }

    /**
     * List all tools from all connected MCP servers (OpenAI format)
     */
    async listAllTools(): Promise<AllToolsResponse> {
        return apiService.get(`${this.baseUrl}/tools`);
    }

    /**
     * Execute an MCP tool by its namespaced name (mcp_server_tool)
     */
    async executeToolByName(toolName: string, args: Record<string, any> = {}): Promise<{
        success: boolean;
        server: string;
        tool: string;
        result: string;
    }> {
        return apiService.post(`${this.baseUrl}/tools/execute?tool_name=${toolName}`, { arguments: args });
    }

    /**
     * Get a human-readable summary of all MCP tools
     */
    async getToolsSummary(): Promise<{ summary: string }> {
        return apiService.get(`${this.baseUrl}/tools/summary`);
    }
}

export const mcpService = new MCPService();
export default mcpService;
