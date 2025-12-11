/**
 * Workflow API Service
 * Handles all workflow-related API calls
 */

import { apiService } from './api';
import type {
  Workflow,
  WorkflowNode,
  WorkflowConnection,
  WorkflowExecution,
  NodeExecution,
  NodeTypeDefinition,
  WorkflowSettings,
} from '@/types/workflow';

// Request/Response types
interface WorkflowCreate {
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  connections: WorkflowConnection[];
  variables?: Record<string, any>;
  settings?: WorkflowSettings;
}

interface WorkflowUpdate {
  name?: string;
  description?: string;
  nodes?: WorkflowNode[];
  connections?: WorkflowConnection[];
  variables?: Record<string, any>;
  settings?: WorkflowSettings;
  isActive?: boolean;
}

interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  nodeCount?: number;
  nodes?: WorkflowNode[];
  connections?: WorkflowConnection[];
}

interface WorkflowListResponse {
  workflows: Workflow[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

interface ExecutionListResponse {
  executions: WorkflowExecution[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

interface ValidationResponse {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

interface VersionInfo {
  version: number;
  summary: string;
  createdAt: string;
}

class WorkflowApiService {
  /**
   * List all workflows
   */
  async listWorkflows(params?: {
    limit?: number;
    offset?: number;
    search?: string;
  }): Promise<WorkflowListResponse> {
    const response = await apiService.get('/api/v1/workflows', { params });
    return response.data;
  }

  /**
   * Create a new workflow
   */
  async createWorkflow(workflow: WorkflowCreate): Promise<Workflow> {
    const response = await apiService.post('/api/v1/workflows', workflow);
    return response.data;
  }

  /**
   * Get a workflow by ID
   */
  async getWorkflow(workflowId: string): Promise<Workflow> {
    const response = await apiService.get(`/api/v1/workflows/${workflowId}`);
    return response.data;
  }

  /**
   * Update a workflow
   */
  async updateWorkflow(workflowId: string, update: WorkflowUpdate): Promise<Workflow> {
    const response = await apiService.put(`/api/v1/workflows/${workflowId}`, update);
    return response.data;
  }

  /**
   * Delete a workflow
   */
  async deleteWorkflow(workflowId: string): Promise<{ status: string; workflow_id: string }> {
    const response = await apiService.delete(`/api/v1/workflows/${workflowId}`);
    return response.data;
  }

  /**
   * Get workflow version history
   */
  async getWorkflowVersions(workflowId: string): Promise<{ versions: VersionInfo[] }> {
    const response = await apiService.get(`/api/v1/workflows/${workflowId}/versions`);
    return response.data;
  }

  /**
   * Restore workflow to a specific version
   */
  async restoreWorkflowVersion(workflowId: string, version: number): Promise<Workflow> {
    const response = await apiService.post(`/api/v1/workflows/${workflowId}/restore/${version}`);
    return response.data;
  }

  /**
   * Execute a workflow
   */
  async executeWorkflow(
    workflowId: string,
    triggerData?: Record<string, any>
  ): Promise<{ execution_id: string; workflow_id: string; status: string }> {
    const response = await apiService.post(
      `/api/v1/workflows/${workflowId}/execute`,
      triggerData || {}
    );
    return response.data;
  }

  /**
   * List executions for a workflow
   */
  async listExecutions(
    workflowId: string,
    params?: { limit?: number; offset?: number }
  ): Promise<ExecutionListResponse> {
    const response = await apiService.get(`/api/v1/workflows/${workflowId}/executions`, { params });
    return response.data;
  }

  /**
   * Get execution details
   */
  async getExecution(executionId: string): Promise<WorkflowExecution> {
    const response = await apiService.get(`/api/v1/workflows/executions/${executionId}`);
    return response.data;
  }

  /**
   * Cancel a running execution
   */
  async cancelExecution(
    executionId: string
  ): Promise<{ status: string; execution_id: string }> {
    const response = await apiService.post(`/api/v1/workflows/executions/${executionId}/cancel`);
    return response.data;
  }

  /**
   * List available node types
   */
  async listNodeTypes(): Promise<{ node_types: NodeTypeDefinition[] }> {
    const response = await apiService.get('/api/v1/workflows/nodes/types');
    return response.data;
  }

  /**
   * Validate a workflow definition
   */
  async validateWorkflow(workflow: WorkflowCreate): Promise<ValidationResponse> {
    const response = await apiService.post('/api/v1/workflows/validate', workflow);
    return response.data;
  }

  /**
   * List workflow templates
   */
  async listTemplates(): Promise<{ templates: WorkflowTemplate[] }> {
    const response = await apiService.get('/api/v1/workflows/templates');
    return response.data;
  }

  /**
   * Get a specific template
   */
  async getTemplate(templateId: string): Promise<WorkflowTemplate> {
    const response = await apiService.get(`/api/v1/workflows/templates/${templateId}`);
    return response.data;
  }

  /**
   * Create workflow from template
   */
  async instantiateTemplate(
    templateId: string,
    params?: { name?: string; description?: string; variables?: Record<string, any> }
  ): Promise<Workflow> {
    const response = await apiService.post(
      `/api/v1/workflows/templates/${templateId}/instantiate`,
      params || {}
    );
    return response.data;
  }

  /**
   * Poll execution status until complete
   */
  async pollExecution(
    executionId: string,
    onUpdate: (execution: WorkflowExecution) => void,
    intervalMs: number = 1000,
    maxAttempts: number = 300
  ): Promise<WorkflowExecution> {
    let attempts = 0;
    
    while (attempts < maxAttempts) {
      const execution = await this.getExecution(executionId);
      onUpdate(execution);
      
      if (['success', 'error', 'cancelled'].includes(execution.status)) {
        return execution;
      }
      
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
      attempts++;
    }
    
    throw new Error('Execution polling timed out');
  }
}

export const workflowApi = new WorkflowApiService();
export default workflowApi;
