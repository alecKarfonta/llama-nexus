import type { ModelInfo } from './api'

export type QuantizationJobSource = 'adapter-export' | 'quantization'

export interface QuantizationJobHandoff {
  id?: string
  name?: string
  source?: QuantizationJobSource
  adapterName?: string
  quantType?: string
}

export type QuantizationModelSourceType = 'local' | 'huggingface'

export interface QuantizationModelHandoff {
  name: string
  sourceModel: string
  sourceType: QuantizationModelSourceType
  repositoryId?: string
  localPath?: string
  parameters?: string
}

export interface QuantizationRouteState {
  job?: QuantizationJobHandoff
  quantizationJob?: QuantizationJobHandoff
  model?: QuantizationModelHandoff
}

export interface QuantizedOutputInfo {
  id: string
  format: string
  quant_type: string
  file_path: string
  file_size: number
  status: string
}

export interface QuantizationJobInfo {
  name: string
  source_model: string
  outputs: QuantizedOutputInfo[]
}

const MODELS_DIR_PREFIX = '/home/llamacpp/models/'

const outputFileName = (filePath: string) => filePath.split('/').pop() || filePath

const parseDeployableGgufName = (filePath: string, fallbackName: string, fallbackVariant: string) => {
  const filename = outputFileName(filePath)
  const stem = filename.toLowerCase().endsWith('.gguf') ? filename.slice(0, -5) : filename.replace(/\.[^.]+$/, '')
  const normalizedVariant = fallbackVariant.toUpperCase()
  const separators = ['-', '_', '.']

  for (const separator of separators) {
    const suffix = `${separator}${normalizedVariant}`
    if (stem.toUpperCase().endsWith(suffix)) {
      return {
        name: stem.slice(0, -suffix.length) || fallbackName,
        variant: normalizedVariant,
      }
    }
  }

  if (stem.toUpperCase().endsWith(normalizedVariant)) {
    return {
      name: stem.slice(0, -normalizedVariant.length).replace(/[-_.]+$/, '') || fallbackName,
      variant: normalizedVariant,
    }
  }

  return {
    name: fallbackName || stem,
    variant: normalizedVariant,
  }
}

export const quantizationJobHandoffKey = (handoff: QuantizationJobHandoff) => (
  [
    handoff.id || '',
    handoff.name || '',
    handoff.source || '',
    handoff.adapterName || '',
    handoff.quantType || '',
  ].join(':')
)

export const quantizationModelHandoffKey = (handoff: QuantizationModelHandoff) => (
  [
    handoff.name,
    handoff.sourceModel,
    handoff.sourceType,
    handoff.repositoryId || '',
    handoff.localPath || '',
  ].join(':')
)

export const quantizationJobNameFromModel = (modelName: string) =>
  `${modelName.replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '') || 'model'}-quantized`

export const quantizationModelHandoffFromModelInfo = (model: ModelInfo): QuantizationModelHandoff => {
  const sourceModel = model.localPath || model.path || model.repositoryId || model.name
  const isLocal =
    Boolean(model.localPath || model.path) ||
    model.source === 'local' ||
    model.framework === 'llama.cpp' ||
    model.framework === 'ggml'

  return {
    name: model.name,
    sourceModel,
    sourceType: isLocal ? 'local' : 'huggingface',
    repositoryId: model.repositoryId,
    localPath: model.localPath || model.path,
    parameters: model.parameters,
  }
}

export const quantizationHandoffFromAdapterExport = (
  response: {
    quantization_job_id?: string
    quantization_job_name?: string
    quant_type?: string
  },
  adapterName: string,
  fallbackQuantType: string,
): QuantizationJobHandoff | null => {
  if (!response.quantization_job_id) return null

  return {
    id: response.quantization_job_id,
    name: response.quantization_job_name || `lora-export-${adapterName}`,
    source: 'adapter-export',
    adapterName,
    quantType: String(response.quant_type || fallbackQuantType).toUpperCase(),
  }
}

export const modelInfoFromQuantizedOutput = (
  job: QuantizationJobInfo,
  output: QuantizedOutputInfo,
): ModelInfo => {
  const parsed = parseDeployableGgufName(output.file_path, job.name, output.quant_type)
  const filename = outputFileName(output.file_path)
  const localPath = output.file_path.startsWith(MODELS_DIR_PREFIX)
    ? output.file_path.slice(MODELS_DIR_PREFIX.length)
    : filename

  return {
    id: output.id,
    name: parsed.name,
    variant: parsed.variant,
    size: output.file_size,
    status: 'available',
    quantization: output.quant_type,
    framework: 'llama.cpp',
    source: 'local',
    localPath,
    filename,
    path: output.file_path,
    description: `Quantized from ${job.source_model}`,
  }
}

export const deployableOutputForJob = <TOutput extends QuantizedOutputInfo>(job: { outputs: TOutput[] }) =>
  job.outputs.find((output) => output.format === 'gguf' && output.status === 'completed' && output.file_path)
