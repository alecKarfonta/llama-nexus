import type { ModelInfo } from './api'

export type RuntimeBackend = 'llamacpp' | 'vllm' | 'ninfer'
export type RuntimeAction = 'start' | 'restart'

export interface DeploymentRouteContext {
  action?: RuntimeAction
  backend?: RuntimeBackend
  modelName?: string
  endpoint?: string
}

export interface DeploymentLaunchContext extends DeploymentRouteContext {
  action: RuntimeAction
  backend: RuntimeBackend
  modelName: string
  endpoint: string
}

export interface RuntimeRouteState {
  deployment?: DeploymentRouteContext
}

export interface BenchmarkRouteState extends RuntimeRouteState {
  model?: ModelInfo
}

export interface DeployRouteState {
  model?: ModelInfo
  backend?: RuntimeBackend
}

export const runtimeBackendLabel = (backend?: RuntimeBackend) => (
  backend === 'vllm' ? 'vLLM' : backend === 'ninfer' ? 'NInfer' : backend === 'llamacpp' ? 'llama.cpp' : 'Runtime'
)

export const modelInfoFromDeployment = (deployment: DeploymentRouteContext): ModelInfo => ({
  id: `${deployment.backend || 'runtime'}:${deployment.modelName || 'active-model'}`,
  name: deployment.modelName || 'Active deployment',
  variant: '',
  size: 0,
  status: 'running',
  framework: deployment.backend === 'vllm' ? 'vllm' : deployment.backend === 'ninfer' ? 'ninfer' : 'llama.cpp',
  port: Number(deployment.endpoint?.match(/:(\d+)$/)?.[1]) || undefined,
})

export const deployStateFromDeployment = (
  deployment: DeploymentRouteContext | null | undefined,
  model?: ModelInfo | null,
): DeployRouteState | undefined => {
  if (model) {
    return {
      backend: deployment?.backend || 'llamacpp',
      model,
    }
  }

  return deployment?.backend ? { backend: deployment.backend } : undefined
}
