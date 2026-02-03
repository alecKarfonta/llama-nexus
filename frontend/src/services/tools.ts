/**
 * Tools Service for Function Calling
 * Handles execution of built-in tools and function definitions
 * Supports research agents with web search, RAG, and data gathering
 * Extended to support MCP (Model Context Protocol) tools
 */

import type { Tool, ToolCall, WeatherQuery, CalculatorQuery, CodeExecutionQuery } from '@/types/api';
import { apiService } from './api';
import { mcpService } from './mcp';

// Extended tool type with source tracking
export interface ExtendedTool extends Tool {
  _source?: 'builtin' | 'mcp';
  _serverName?: string;  // For MCP tools, the server name
}


// Web search result type
interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  source?: string;
}

// RAG search result type
interface RAGSearchResult {
  content: string;
  source: string;
  score: number;
  metadata?: Record<string, any>;
}

export class ToolsService {
  // Built-in tool definitions for testing
  static getAvailableTools(): Tool[] {
    return [
      // === RESEARCH TOOLS ===
      {
        type: 'function',
        function: {
          name: 'web_search',
          description: 'Search the web for current information on any topic. Use this to find recent news, facts, documentation, or any information not in your training data.',
          parameters: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'The search query - be specific and include relevant keywords'
              },
              num_results: {
                type: 'number',
                description: 'Number of results to return (default: 5, max: 10)'
              }
            },
            required: ['query']
          }
        }
      },
      {
        type: 'function',
        function: {
          name: 'knowledge_search',
          description: 'Search the local knowledge base and documents. Use this to find information from uploaded documents, research papers, and indexed content.',
          parameters: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'The semantic search query'
              },
              domain: {
                type: 'string',
                description: 'Optional: specific knowledge domain to search in'
              },
              top_k: {
                type: 'number',
                description: 'Number of results to return (default: 5)'
              }
            },
            required: ['query']
          }
        }
      },
      {
        type: 'function',
        function: {
          name: 'fetch_url',
          description: 'Fetch and extract content from a specific URL. Use this to read articles, documentation, or web pages.',
          parameters: {
            type: 'object',
            properties: {
              url: {
                type: 'string',
                description: 'The URL to fetch content from'
              },
              extract_type: {
                type: 'string',
                enum: ['text', 'markdown', 'summary'],
                description: 'How to extract content: text (raw), markdown (formatted), or summary (brief)'
              }
            },
            required: ['url']
          }
        }
      },
      {
        type: 'function',
        function: {
          name: 'save_research_note',
          description: 'Save a research note or finding for later reference. Use this to build up research during a session.',
          parameters: {
            type: 'object',
            properties: {
              title: {
                type: 'string',
                description: 'Title or topic of the note'
              },
              content: {
                type: 'string',
                description: 'The research note content'
              },
              sources: {
                type: 'array',
                items: { type: 'string' },
                description: 'List of sources for this information'
              },
              tags: {
                type: 'array',
                items: { type: 'string' },
                description: 'Tags to categorize this note'
              }
            },
            required: ['title', 'content']
          }
        }
      },
      // === UTILITY TOOLS ===
      {
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get current weather information for a specific location',
          parameters: {
            type: 'object',
            properties: {
              location: {
                type: 'string',
                description: 'The city and state/country, e.g., "London, UK" or "New York, NY"'
              },
              unit: {
                type: 'string',
                enum: ['celsius', 'fahrenheit'],
                description: 'Temperature unit preference'
              }
            },
            required: ['location']
          }
        }
      },
      {
        type: 'function',
        function: {
          name: 'calculate',
          description: 'Perform mathematical calculations and evaluate expressions',
          parameters: {
            type: 'object',
            properties: {
              expression: {
                type: 'string',
                description: 'Mathematical expression to evaluate, e.g., "2 + 2 * 3" or "sqrt(16)"'
              }
            },
            required: ['expression']
          }
        }
      },
      {
        type: 'function',
        function: {
          name: 'execute_code',
          description: 'Execute code in various programming languages (simulated for demo)',
          parameters: {
            type: 'object',
            properties: {
              language: {
                type: 'string',
                enum: ['python', 'javascript', 'bash'],
                description: 'Programming language to execute'
              },
              code: {
                type: 'string',
                description: 'Code to execute'
              }
            },
            required: ['language', 'code']
          }
        }
      },
      {
        type: 'function',
        function: {
          name: 'get_system_info',
          description: 'Get current system information and metrics',
          parameters: {
            type: 'object',
            properties: {
              metric: {
                type: 'string',
                enum: ['cpu', 'memory', 'disk', 'gpu', 'all'],
                description: 'Type of system metric to retrieve'
              }
            },
            required: ['metric']
          }
        }
      },
      {
        type: 'function',
        function: {
          name: 'generate_uuid',
          description: 'Generate a universally unique identifier (UUID)',
          parameters: {
            type: 'object',
            properties: {
              version: {
                type: 'string',
                enum: ['v4', 'v1'],
                description: 'UUID version to generate'
              }
            },
            required: []
          }
        }
      },
      // === DATE/TIME TOOLS ===
      {
        type: 'function',
        function: {
          name: 'get_current_time',
          description: 'Get the current date and time in various formats and timezones',
          parameters: {
            type: 'object',
            properties: {
              timezone: {
                type: 'string',
                description: 'Timezone (e.g., "America/New_York", "Europe/London", "UTC")'
              },
              format: {
                type: 'string',
                enum: ['iso', 'human', 'unix'],
                description: 'Output format'
              }
            },
            required: []
          }
        }
      }
    ];
  }

  // Get only research-focused tools
  static getResearchTools(): Tool[] {
    const researchToolNames = ['web_search', 'knowledge_search', 'fetch_url', 'save_research_note', 'calculate'];
    return this.getAvailableTools().filter(t => researchToolNames.includes(t.function.name));
  }

  // Get built-in tools with source marking
  static getBuiltInToolsExtended(): ExtendedTool[] {
    return this.getAvailableTools().map(tool => ({
      ...tool,
      _source: 'builtin' as const,
    }));
  }

  // Fetch MCP tools from all connected servers
  static async getMCPTools(): Promise<ExtendedTool[]> {
    try {
      const response = await mcpService.listAllTools();
      // Handle potential response structure variations
      const tools = response?.tools ?? [];
      if (!Array.isArray(tools)) {
        console.warn('getMCPTools: response.tools is not an array', response);
        return [];
      }
      return tools.map(tool => ({
        type: 'function' as const,
        function: tool.function,
        _source: 'mcp' as const,
        _serverName: tool.function.name.split('_')[1], // Extract server name from mcp_servername_toolname
      }));
    } catch (error) {
      console.warn('Failed to fetch MCP tools:', error);
      return [];
    }
  }

  // Get all tools (built-in + MCP) - async because MCP tools need to be fetched
  static async getAllTools(): Promise<ExtendedTool[]> {
    const builtIn = this.getBuiltInToolsExtended();
    const mcpTools = await this.getMCPTools();
    return [...builtIn, ...mcpTools];
  }


  // Research notes storage (in-memory for session)
  private static researchNotes: Array<{
    id: string;
    title: string;
    content: string;
    sources: string[];
    tags: string[];
    timestamp: string;
  }> = [];

  // Get all research notes from current session
  static getResearchNotes() {
    return this.researchNotes;
  }

  // Clear research notes (new session)
  static clearResearchNotes() {
    this.researchNotes = [];
  }

  // Execute a tool call and return the result
  static async executeToolCall(toolCall: ToolCall): Promise<string> {
    const { name, arguments: args } = toolCall.function;

    try {
      const parsedArgs = JSON.parse(args);

      // Check if this is an MCP tool (namespaced with mcp_)
      if (name.startsWith('mcp_')) {
        return await this.executeMCPTool(name, parsedArgs);
      }

      switch (name) {
        // Research tools
        case 'web_search':
          return await this.executeWebSearchTool(parsedArgs);

        case 'knowledge_search':
          return await this.executeKnowledgeSearchTool(parsedArgs);

        case 'fetch_url':
          return await this.executeFetchUrlTool(parsedArgs);

        case 'save_research_note':
          return this.executeSaveNoteTool(parsedArgs);

        // Utility tools
        case 'get_weather':
          return this.executeWeatherTool(parsedArgs as WeatherQuery);

        case 'calculate':
          return this.executeCalculatorTool(parsedArgs as CalculatorQuery);

        case 'execute_code':
          return this.executeCodeTool(parsedArgs as CodeExecutionQuery);

        case 'get_system_info':
          return await this.executeSystemInfoTool(parsedArgs);

        case 'generate_uuid':
          return this.executeUUIDTool(parsedArgs);

        case 'get_current_time':
          return this.executeTimeTool(parsedArgs);

        default:
          return `Error: Unknown tool "${name}"`;
      }
    } catch (error) {
      return `Error executing tool ${name}: ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  }

  // Execute an MCP tool via the MCP service
  private static async executeMCPTool(toolName: string, args: Record<string, unknown>): Promise<string> {
    try {
      const result = await mcpService.executeToolByName(toolName, args);
      if (result.success) {
        // Format the result nicely
        if (typeof result.result === 'string') {
          return result.result;
        }
        return JSON.stringify(result.result, null, 2);
      } else {
        return `MCP Tool Error: ${result.result}`;
      }
    } catch (error) {
      return `Error executing MCP tool ${toolName}: ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
  }


  // === RESEARCH TOOL IMPLEMENTATIONS ===

  private static async executeWebSearchTool(query: { query: string; num_results?: number }): Promise<string> {
    const { query: searchQuery, num_results = 5 } = query;

    try {
      // Try to use backend web search endpoint if available
      const response = await apiService.post('/api/v1/tools/web-search', {
        query: searchQuery,
        num_results: Math.min(num_results, 10)
      });

      if (response.data?.results) {
        return JSON.stringify({
          query: searchQuery,
          results: response.data.results,
          source: 'live_search',
          timestamp: new Date().toISOString()
        }, null, 2);
      }
    } catch {
      // Fallback to simulated results if backend unavailable
    }

    // Simulated web search results for demo
    const simulatedResults: WebSearchResult[] = [
      {
        title: `${searchQuery} - Latest Information`,
        url: `https://example.com/search?q=${encodeURIComponent(searchQuery)}`,
        snippet: `Comprehensive information about ${searchQuery}. This simulated result would contain relevant content from web search.`,
        source: 'example.com'
      },
      {
        title: `Understanding ${searchQuery} - A Guide`,
        url: `https://docs.example.com/${searchQuery.toLowerCase().replace(/\s+/g, '-')}`,
        snippet: `An in-depth guide covering ${searchQuery}. Learn about key concepts, best practices, and recent developments.`,
        source: 'docs.example.com'
      },
      {
        title: `${searchQuery} News and Updates`,
        url: `https://news.example.com/topic/${searchQuery.toLowerCase().replace(/\s+/g, '-')}`,
        snippet: `Latest news and updates about ${searchQuery}. Stay informed with recent developments and announcements.`,
        source: 'news.example.com'
      }
    ];

    return JSON.stringify({
      query: searchQuery,
      results: simulatedResults.slice(0, num_results),
      source: 'simulated',
      note: 'Configure SEARCH_API_KEY for live results',
      timestamp: new Date().toISOString()
    }, null, 2);
  }

  private static async executeKnowledgeSearchTool(query: { query: string; domain?: string; top_k?: number }): Promise<string> {
    const { query: searchQuery, domain, top_k = 5 } = query;

    try {
      // Use RAG search endpoint
      const response = await apiService.post('/api/v1/rag/search', {
        query: searchQuery,
        domain_id: domain,
        top_k
      });

      if (response.data?.results) {
        return JSON.stringify({
          query: searchQuery,
          domain: domain || 'all',
          results: response.data.results.map((r: any) => ({
            content: r.content,
            source: r.metadata?.source || r.document_name || 'Unknown',
            score: r.score,
            metadata: r.metadata
          })),
          source: 'knowledge_base',
          timestamp: new Date().toISOString()
        }, null, 2);
      }
    } catch {
      // Fallback
    }

    return JSON.stringify({
      query: searchQuery,
      domain: domain || 'all',
      results: [],
      message: 'No documents found. Upload documents to the knowledge base first.',
      source: 'knowledge_base',
      timestamp: new Date().toISOString()
    }, null, 2);
  }

  private static async executeFetchUrlTool(query: { url: string; extract_type?: string }): Promise<string> {
    const { url, extract_type = 'text' } = query;

    try {
      // Try backend URL fetch endpoint
      const response = await apiService.post('/api/v1/tools/fetch-url', {
        url,
        extract_type
      });

      if (response.data?.content) {
        return JSON.stringify({
          url,
          content: response.data.content,
          title: response.data.title,
          extract_type,
          timestamp: new Date().toISOString()
        }, null, 2);
      }
    } catch {
      // Fallback
    }

    // Simulated fetch for demo
    return JSON.stringify({
      url,
      content: `[Simulated content from ${url}]\n\nThis is a placeholder for fetched content. Configure the backend URL fetcher for live content extraction.`,
      extract_type,
      simulated: true,
      timestamp: new Date().toISOString()
    }, null, 2);
  }

  private static executeSaveNoteTool(query: { title: string; content: string; sources?: string[]; tags?: string[] }): string {
    const { title, content, sources = [], tags = [] } = query;

    const note = {
      id: crypto.randomUUID(),
      title,
      content,
      sources,
      tags,
      timestamp: new Date().toISOString()
    };

    this.researchNotes.push(note);

    return JSON.stringify({
      status: 'saved',
      note_id: note.id,
      title,
      total_notes: this.researchNotes.length,
      message: `Research note "${title}" saved. You now have ${this.researchNotes.length} notes in this session.`,
      timestamp: note.timestamp
    }, null, 2);
  }

  private static executeTimeTool(query: { timezone?: string; format?: string }): string {
    const { timezone = 'UTC', format = 'human' } = query;

    const now = new Date();
    let timeString: string;

    try {
      switch (format) {
        case 'iso':
          timeString = now.toISOString();
          break;
        case 'unix':
          timeString = Math.floor(now.getTime() / 1000).toString();
          break;
        case 'human':
        default:
          timeString = now.toLocaleString('en-US', {
            timeZone: timezone,
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            timeZoneName: 'short'
          });
      }
    } catch {
      timeString = now.toISOString();
    }

    return JSON.stringify({
      timezone,
      format,
      time: timeString,
      utc_offset: now.getTimezoneOffset(),
      timestamp: now.toISOString()
    }, null, 2);
  }

  private static executeWeatherTool(query: WeatherQuery): string {
    // Simulate weather API call
    const { location, unit = 'celsius' } = query;
    const temp = unit === 'celsius' ? '22°C' : '72°F';
    const conditions = ['Sunny', 'Cloudy', 'Rainy', 'Partly Cloudy'][Math.floor(Math.random() * 4)];

    return JSON.stringify({
      location,
      temperature: temp,
      conditions,
      humidity: '65%',
      wind: '10 km/h',
      timestamp: new Date().toISOString()
    }, null, 2);
  }

  private static executeCalculatorTool(query: CalculatorQuery): string {
    const { expression } = query;

    try {
      // Basic math evaluation (in real implementation, use a safe evaluator)
      // This is a simplified demo - in production, use a proper math library
      const safeExpression = expression.replace(/[^0-9+\-*/().\s]/g, '');
      const result = eval(safeExpression);

      return JSON.stringify({
        expression,
        result,
        timestamp: new Date().toISOString()
      }, null, 2);
    } catch (error) {
      return JSON.stringify({
        expression,
        error: 'Invalid mathematical expression',
        timestamp: new Date().toISOString()
      }, null, 2);
    }
  }

  private static executeCodeTool(query: CodeExecutionQuery): string {
    const { language, code } = query;

    // Simulate code execution (in real implementation, use sandboxed execution)
    const outputs = {
      python: `# Simulated Python execution\n# Code: ${code}\n# Output: This would execute in a Python interpreter`,
      javascript: `// Simulated JavaScript execution\n// Code: ${code}\n// Output: This would execute in a JS runtime`,
      bash: `# Simulated Bash execution\n# Command: ${code}\n# Output: This would execute in a shell`
    };

    return JSON.stringify({
      language,
      code,
      output: outputs[language] || 'Unsupported language',
      timestamp: new Date().toISOString(),
      simulated: true
    }, null, 2);
  }

  private static async executeSystemInfoTool(query: { metric: string }): Promise<string> {
    const { metric } = query;

    try {
      // Fetch real system info from backend
      const response = await fetch('/api/v1/resources');
      if (!response.ok) {
        throw new Error(`Failed to fetch resources: ${response.status}`);
      }
      const data = await response.json();

      // Format the response based on requested metric
      const systemInfo: Record<string, unknown> = {
        cpu: data.cpu ? {
          usage_percent: data.cpu.percent,
          cores: data.cpu.count,
          frequency_mhz: data.cpu.freq?.current
        } : { error: 'CPU info unavailable' },
        memory: data.memory ? {
          used_mb: Math.round(data.memory.used_mb),
          total_mb: Math.round(data.memory.total_mb),
          available_mb: Math.round(data.memory.available_mb),
          usage_percent: data.memory.percent
        } : { error: 'Memory info unavailable' },
        gpu: data.gpu ? {
          vram_used_mb: data.gpu.vram_used_mb,
          vram_total_mb: data.gpu.vram_total_mb,
          usage_percent: data.gpu.usage_percent,
          temperature_c: data.gpu.temperature_c,
          power_watts: data.gpu.power_watts
        } : { status: data.gpu?.status || 'unavailable', reason: data.gpu?.reason },
        disk: { usage: '450GB', total: '1TB', percentage: '45%' }, // Disk not in backend yet
        all: {
          cpu: data.cpu,
          memory: data.memory,
          gpu: data.gpu,
          timestamp: data.timestamp
        }
      };

      return JSON.stringify(systemInfo[metric] || systemInfo.all, null, 2);
    } catch (error) {
      // Fallback to simulated data on error
      console.error('Failed to fetch system info:', error);
      const fallback = {
        cpu: { usage: '45%', cores: 8, model: 'Intel Core i7' },
        memory: { usage: '8.2GB', total: '16GB', percentage: '51%' },
        gpu: { error: 'Failed to fetch GPU info' },
        disk: { usage: '450GB', total: '1TB', percentage: '45%' },
        all: {
          error: 'Failed to fetch system info',
          timestamp: new Date().toISOString()
        }
      };
      return JSON.stringify(fallback[metric as keyof typeof fallback] || fallback.all, null, 2);
    }
  }

  private static executeUUIDTool(query: { version?: string }): string {
    const { version = 'v4' } = query;

    // Generate a simple UUID (in real implementation, use a proper UUID library)
    const uuid = version === 'v4'
      ? 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      })
      : 'xxxxxxxx-xxxx-1xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });

    return JSON.stringify({
      uuid,
      version,
      timestamp: new Date().toISOString()
    }, null, 2);
  }

  // Check if a tool call is valid
  static validateToolCall(toolCall: ToolCall): boolean {
    const availableTools = this.getAvailableTools();
    const tool = availableTools.find(t => t.function.name === toolCall.function.name);

    if (!tool) {
      return false;
    }

    try {
      const args = JSON.parse(toolCall.function.arguments);
      // Basic validation - in production, implement proper schema validation
      return typeof args === 'object' && args !== null;
    } catch {
      return false;
    }
  }
}

// Create singleton instance
export const toolsService = new ToolsService();

// Export default for convenience
export default toolsService;
