/**
 * Tools Service for Function Calling
 * Handles execution of built-in tools and function definitions
 */

import type { Tool, ToolCall, WeatherQuery, CalculatorQuery, CodeExecutionQuery } from '@/types/api';

export class ToolsService {
  // Built-in tool definitions for testing
  static getAvailableTools(): Tool[] {
    return [
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
                enum: ['cpu', 'memory', 'disk', 'all'],
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
      }
    ];
  }

  // Execute a tool call and return the result
  static async executeToolCall(toolCall: ToolCall): Promise<string> {
    const { name, arguments: args } = toolCall.function;
    
    try {
      const parsedArgs = JSON.parse(args);
      
      switch (name) {
        case 'get_weather':
          return this.executeWeatherTool(parsedArgs as WeatherQuery);
        
        case 'calculate':
          return this.executeCalculatorTool(parsedArgs as CalculatorQuery);
        
        case 'execute_code':
          return this.executeCodeTool(parsedArgs as CodeExecutionQuery);
        
        case 'get_system_info':
          return this.executeSystemInfoTool(parsedArgs);
        
        case 'generate_uuid':
          return this.executeUUIDTool(parsedArgs);
        
        default:
          return `Error: Unknown tool "${name}"`;
      }
    } catch (error) {
      return `Error executing tool ${name}: ${error instanceof Error ? error.message : 'Unknown error'}`;
    }
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

  private static executeSystemInfoTool(query: { metric: string }): string {
    const { metric } = query;
    
    const systemInfo = {
      cpu: { usage: '45%', cores: 8, model: 'Intel Core i7' },
      memory: { usage: '8.2GB', total: '16GB', percentage: '51%' },
      disk: { usage: '450GB', total: '1TB', percentage: '45%' },
      all: {
        cpu: { usage: '45%', cores: 8, model: 'Intel Core i7' },
        memory: { usage: '8.2GB', total: '16GB', percentage: '51%' },
        disk: { usage: '450GB', total: '1TB', percentage: '45%' },
        uptime: '5 days, 3 hours',
        timestamp: new Date().toISOString()
      }
    };
    
    return JSON.stringify(systemInfo[metric as keyof typeof systemInfo] || systemInfo.all, null, 2);
  }

  private static executeUUIDTool(query: { version?: string }): string {
    const { version = 'v4' } = query;
    
    // Generate a simple UUID (in real implementation, use a proper UUID library)
    const uuid = version === 'v4' 
      ? 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
          const r = Math.random() * 16 | 0;
          const v = c === 'x' ? r : (r & 0x3 | 0x8);
          return v.toString(16);
        })
      : 'xxxxxxxx-xxxx-1xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
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
