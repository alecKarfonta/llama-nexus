/**
 * Tools Service for Function Calling
 * Handles execution of built-in tools and function definitions
 * Supports research agents with web search, RAG, and data gathering
 * Extended to support MCP (Model Context Protocol) tools
 */

import type { Tool, ToolCall, WeatherQuery, CalculatorQuery, CodeExecutionQuery } from '@/types/api';
import { apiService } from './api';
import { mcpService } from './mcp';
import { debugLog } from '@/utils/debug';

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

type CalculatorToken =
  | { type: 'number'; value: number }
  | { type: 'identifier'; value: string }
  | { type: 'operator'; value: '+' | '-' | '*' | '/' | '^' }
  | { type: 'leftParen' | 'rightParen' | 'comma' };

interface CalculatorFunction {
  minArgs: number;
  maxArgs?: number;
  execute: (args: number[]) => number;
}

const calculatorFunctions: Record<string, CalculatorFunction> = {
  abs: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.abs(value) },
  acos: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.acos(value) },
  asin: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.asin(value) },
  atan: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.atan(value) },
  ceil: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.ceil(value) },
  cos: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.cos(value) },
  floor: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.floor(value) },
  log: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.log(value) },
  max: { minArgs: 1, execute: (args) => Math.max(...args) },
  min: { minArgs: 1, execute: (args) => Math.min(...args) },
  pow: { minArgs: 2, maxArgs: 2, execute: ([base, exponent]) => Math.pow(base, exponent) },
  round: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.round(value) },
  sin: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.sin(value) },
  sqrt: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.sqrt(value) },
  tan: { minArgs: 1, maxArgs: 1, execute: ([value]) => Math.tan(value) },
};

const calculatorConstants: Record<string, number> = {
  e: Math.E,
  pi: Math.PI,
};

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
        debugLog('tools', 'MCP tools response did not include an array', response);
        return [];
      }
      return tools.map(tool => ({
        type: 'function' as const,
        function: tool.function,
        _source: 'mcp' as const,
        _serverName: tool.function.name.split('_')[1], // Extract server name from mcp_servername_toolname
      }));
    } catch (error) {
      debugLog('tools', 'MCP tools unavailable', error);
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

  private static tokenizeCalculatorExpression(expression: string): CalculatorToken[] {
    const tokens: CalculatorToken[] = [];
    let index = 0;

    const isDigit = (value: string) => value >= '0' && value <= '9';
    const isIdentifierStart = (value: string) => /[a-z]/i.test(value);
    const isIdentifierPart = (value: string) => /[a-z0-9_]/i.test(value);

    while (index < expression.length) {
      const char = expression[index];

      if (/\s/.test(char)) {
        index += 1;
        continue;
      }

      if (isDigit(char) || char === '.') {
        const start = index;
        let dotCount = char === '.' ? 1 : 0;
        index += 1;

        while (index < expression.length) {
          const next = expression[index];
          if (next === '.') {
            dotCount += 1;
            if (dotCount > 1) {
              throw new Error('Invalid number');
            }
            index += 1;
            continue;
          }
          if (!isDigit(next)) break;
          index += 1;
        }

        const value = Number(expression.slice(start, index));
        if (!Number.isFinite(value)) {
          throw new Error('Invalid number');
        }
        tokens.push({ type: 'number', value });
        continue;
      }

      if (isIdentifierStart(char)) {
        const start = index;
        index += 1;
        while (index < expression.length && isIdentifierPart(expression[index])) {
          index += 1;
        }
        tokens.push({ type: 'identifier', value: expression.slice(start, index).toLowerCase() });
        continue;
      }

      if (char === '+' || char === '-' || char === '*' || char === '/' || char === '^') {
        tokens.push({ type: 'operator', value: char });
        index += 1;
        continue;
      }

      if (char === '(') {
        tokens.push({ type: 'leftParen' });
        index += 1;
        continue;
      }

      if (char === ')') {
        tokens.push({ type: 'rightParen' });
        index += 1;
        continue;
      }

      if (char === ',') {
        tokens.push({ type: 'comma' });
        index += 1;
        continue;
      }

      throw new Error('Unsupported character');
    }

    return tokens;
  }

  private static evaluateCalculatorExpression(expression: string): number {
    const tokens = this.tokenizeCalculatorExpression(expression);
    let position = 0;

    if (tokens.length === 0) {
      throw new Error('Expression is empty');
    }

    const peek = () => tokens[position];
    const consume = () => tokens[position++];
    const peekOperator = () => {
      const token = peek();
      return token?.type === 'operator' ? token : undefined;
    };

    const consumeType = (type: CalculatorToken['type']) => {
      if (peek()?.type === type) {
        position += 1;
        return true;
      }
      return false;
    };

    const parseExpression = (): number => parseAdditive();

    const parseAdditive = (): number => {
      let value = parseMultiplicative();
      let operator = peekOperator();

      while (operator && (operator.value === '+' || operator.value === '-')) {
        consume();
        const right = parseMultiplicative();
        value = operator.value === '+' ? value + right : value - right;
        operator = peekOperator();
      }

      return value;
    };

    const parseMultiplicative = (): number => {
      let value = parsePower();
      let operator = peekOperator();

      while (operator && (operator.value === '*' || operator.value === '/')) {
        consume();
        const right = parsePower();
        value = operator.value === '*' ? value * right : value / right;
        operator = peekOperator();
      }

      return value;
    };

    const parsePower = (): number => {
      const value = parseUnary();

      if (peekOperator()?.value === '^') {
        consume();
        return Math.pow(value, parsePower());
      }

      return value;
    };

    const parseUnary = (): number => {
      const operator = peekOperator();

      if (operator && (operator.value === '+' || operator.value === '-')) {
        consume();
        const value = parseUnary();
        return operator.value === '-' ? -value : value;
      }

      return parsePrimary();
    };

    const parsePrimary = (): number => {
      const token = consume();

      if (!token) {
        throw new Error('Unexpected end of expression');
      }

      if (token.type === 'number') {
        return token.value;
      }

      if (token.type === 'identifier') {
        if (consumeType('leftParen')) {
          const args: number[] = [];

          if (!consumeType('rightParen')) {
            do {
              args.push(parseExpression());
            } while (consumeType('comma'));

            if (!consumeType('rightParen')) {
              throw new Error('Expected closing parenthesis');
            }
          }

          const fn = calculatorFunctions[token.value];
          if (!fn) {
            throw new Error('Unknown function');
          }
          if (args.length < fn.minArgs || (fn.maxArgs !== undefined && args.length > fn.maxArgs)) {
            throw new Error('Invalid argument count');
          }

          const result = fn.execute(args);
          if (!Number.isFinite(result)) {
            throw new Error('Invalid function result');
          }
          return result;
        }

        const constant = calculatorConstants[token.value];
        if (constant === undefined) {
          throw new Error('Unknown identifier');
        }
        return constant;
      }

      if (token.type === 'leftParen') {
        const value = parseExpression();
        if (!consumeType('rightParen')) {
          throw new Error('Expected closing parenthesis');
        }
        return value;
      }

      throw new Error('Unexpected token');
    };

    const result = parseExpression();

    if (position !== tokens.length || !Number.isFinite(result)) {
      throw new Error('Invalid mathematical expression');
    }

    return result;
  }

  private static executeCalculatorTool(query: CalculatorQuery): string {
    const { expression } = query;

    try {
      const result = this.evaluateCalculatorExpression(expression);

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
