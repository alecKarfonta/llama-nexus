import type { ModelInfo } from './api'

export type TrainingDatasetSource = 'dataset-library' | 'distillation' | 'book-datasets'

export interface TrainingDatasetHandoff {
  id: string
  name?: string
  source?: TrainingDatasetSource
  exampleCount?: number
}

export interface TrainingModelHandoff {
  name: string
  baseModel: string
  source?: 'model-library' | string
  repositoryId?: string
  localPath?: string
  parameters?: string
}

export interface FineTuningRouteState {
  dataset?: TrainingDatasetHandoff
  model?: TrainingModelHandoff
}

export type AdapterHandoffSource = 'training-job' | 'adapter-registry'

export interface AdapterHandoff {
  id?: string
  name?: string
  baseModel?: string
  adapterPath?: string
  trainingJobId?: string
  source?: AdapterHandoffSource | string
}

export interface EvaluationRouteState {
  adapter?: AdapterHandoff
}

export interface AdapterComparisonRouteState {
  adapter?: AdapterHandoff
  adapters?: AdapterHandoff[]
}

export interface BenchmarkComparisonRouteState {
  jobId?: string
  jobIds?: string[]
}

export interface AdapterHandoffMatchCandidate {
  id?: string
  name?: string
  training_job_id?: string
  job_id?: string
  adapter_path?: string
  path?: string
}

export const normalizeHandoffMatchValue = (value?: string | number | null) =>
  value == null ? undefined : String(value).trim().toLowerCase() || undefined

export const compactHandoffMatchValues = (values: Array<string | number | null | undefined>) =>
  values
    .map(normalizeHandoffMatchValue)
    .filter((value): value is string => Boolean(value))

export const adapterHandoffKey = (handoff: AdapterHandoff) => (
  [
    handoff.id || '',
    handoff.trainingJobId || '',
    handoff.name || '',
    handoff.baseModel || '',
    handoff.adapterPath || '',
    handoff.source || '',
  ].join(':')
)

export const adapterHandoffsKey = (handoffs: AdapterHandoff[]) =>
  handoffs.map(adapterHandoffKey).join('|')

export const trainingDatasetHandoffKey = (handoff: TrainingDatasetHandoff) => (
  [handoff.id, handoff.name || '', handoff.source || '', handoff.exampleCount ?? ''].join(':')
)

export const trainingModelHandoffKey = (handoff: TrainingModelHandoff) => (
  [
    handoff.name,
    handoff.baseModel,
    handoff.source || '',
    handoff.repositoryId || '',
    handoff.localPath || '',
  ].join(':')
)

export const trainingJobNameFromModel = (modelName: string) =>
  `${modelName.replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '') || 'model'}-adapter`

export const trainingModelHandoffFromModelInfo = (model: ModelInfo): TrainingModelHandoff => {
  const baseModel = model.repositoryId || model.path || model.localPath || model.name

  return {
    name: model.name,
    baseModel,
    source: 'model-library',
    repositoryId: model.repositoryId,
    localPath: model.localPath || model.path,
    parameters: model.parameters,
  }
}

export const trainingDatasetHandoffFromLibraryDataset = (dataset: {
  id: string
  name?: string
  num_examples?: number
}): TrainingDatasetHandoff => ({
  id: dataset.id,
  name: dataset.name,
  source: 'dataset-library',
  exampleCount: dataset.num_examples,
})

export const trainingDatasetHandoffFromDistillationJob = (job: {
  output_dataset_id: string
  name?: string
  generated_count?: number
}): TrainingDatasetHandoff => ({
  id: job.output_dataset_id,
  name: `${job.name || 'Distilled'} dataset`,
  source: 'distillation',
  exampleCount: job.generated_count,
})

export const adapterHandoffFromTrainingJob = (job: {
  id?: string | number
  name?: string
  base_model?: string
  adapter_path?: string
}): AdapterHandoff => ({
  id: job.id == null ? undefined : String(job.id),
  name: job.name,
  baseModel: job.base_model,
  adapterPath: job.adapter_path,
  source: 'training-job',
})

export const adapterHandoffFromRegistryAdapter = (adapter: {
  id?: string | number
  job_id?: string | number
  training_job_id?: string | number
  name?: string
  base_model?: string
  adapter_path?: string
  path?: string
}): AdapterHandoff => ({
  id: String(adapter.job_id || adapter.training_job_id || adapter.id || ''),
  trainingJobId: adapter.training_job_id == null ? undefined : String(adapter.training_job_id),
  name: adapter.name,
  baseModel: adapter.base_model,
  adapterPath: adapter.adapter_path || adapter.path,
  source: 'adapter-registry',
})

export const findAdapterForHandoff = <T extends AdapterHandoffMatchCandidate>(
  handoff: AdapterHandoff,
  adapters: T[],
  isSelectable: (adapter: T) => boolean = () => true,
) => {
  const requestedIds = compactHandoffMatchValues([handoff.id, handoff.trainingJobId])
  const requestedPath = normalizeHandoffMatchValue(handoff.adapterPath)
  const requestedName = normalizeHandoffMatchValue(handoff.name)

  return adapters.find((adapter) => {
    if (!isSelectable(adapter)) return false

    const adapterIds = compactHandoffMatchValues([adapter.id, adapter.training_job_id, adapter.job_id])
    const adapterPaths = compactHandoffMatchValues([adapter.adapter_path, adapter.path])

    return (
      requestedIds.some((id) => adapterIds.includes(id)) ||
      Boolean(requestedPath && adapterPaths.includes(requestedPath)) ||
      Boolean(requestedName && normalizeHandoffMatchValue(adapter.name) === requestedName)
    )
  })
}
