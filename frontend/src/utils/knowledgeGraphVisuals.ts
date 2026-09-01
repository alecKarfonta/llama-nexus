export type KnowledgeGraphDetailMode = 'auto' | 'performance' | 'detail';
export type KnowledgeGraphLabelDensity = 'focus' | 'priority' | 'rich';
export type KnowledgeGraphRelationVisualKind =
  'connects' |
  'contains' |
  'controls' |
  'conflicts' |
  'creates' |
  'evaluates' |
  'flows' |
  'guards' |
  'influences' |
  'references' |
  'sequence' |
  'supports' |
  'transforms';
export type KnowledgeGraphScaleTier = 'compact' | 'standard' | 'large' | 'massive';
export type KnowledgeGraphZoomTier = 'far' | 'middle' | 'near';

const KNOWN_ENTITY_VISUAL_TYPES = new Set([
  'api',
  'code',
  'concept',
  'config',
  'custom',
  'data',
  'date',
  'evaluation',
  'event',
  'hardware',
  'location',
  'memory',
  'metric',
  'model',
  'organization',
  'person',
  'process',
  'product',
  'prompt',
  'runtime',
  'security',
  'service',
  'technology',
  'vector',
]);

const ENTITY_VISUAL_TYPE_ALIASES: Record<string, string> = {
  agent: 'technology',
  ai_model: 'model',
  algorithm: 'technology',
  application: 'technology',
  app: 'technology',
  api: 'api',
  api_client: 'api',
  auth: 'security',
  authentication: 'security',
  authorization: 'security',
  benchmark: 'evaluation',
  benchmark_metric: 'metric',
  checkpoint: 'model',
  code: 'code',
  codebase: 'code',
  company: 'organization',
  component: 'technology',
  config: 'config',
  configuration: 'config',
  container: 'technology',
  cuda: 'hardware',
  database: 'data',
  dataset: 'data',
  deployment: 'runtime',
  document: 'data',
  embedding: 'vector',
  embeddings: 'vector',
  endpoint: 'service',
  environment: 'config',
  env_var: 'config',
  eval: 'evaluation',
  evaluation: 'evaluation',
  evaluation_metric: 'metric',
  file: 'code',
  format: 'concept',
  framework: 'technology',
  gpu: 'hardware',
  hardware: 'hardware',
  inference_model: 'model',
  key: 'security',
  library: 'code',
  llm: 'model',
  machine_learning_model: 'model',
  maintenance: 'process',
  method: 'process',
  metric: 'metric',
  ml_model: 'model',
  model: 'model',
  module: 'code',
  named_entity_recognition: 'model',
  ner: 'model',
  ner_model: 'model',
  org: 'organization',
  organisation: 'organization',
  package: 'code',
  parameter: 'config',
  pipeline: 'process',
  policy: 'security',
  prompt: 'prompt',
  prompt_template: 'prompt',
  project: 'organization',
  quantization: 'runtime',
  rag: 'vector',
  relation_extractor: 'model',
  relation_model: 'model',
  relationship_extractor: 'model',
  relationship_model: 'model',
  repository: 'code',
  requirement: 'config',
  retrieval: 'vector',
  route: 'service',
  runner: 'runtime',
  runtime: 'runtime',
  safety: 'security',
  secret: 'security',
  secrets: 'security',
  service: 'service',
  serving: 'runtime',
  session: 'memory',
  setting: 'config',
  settings: 'config',
  software: 'code',
  solution: 'product',
  specification: 'concept',
  state: 'memory',
  system: 'service',
  task: 'process',
  template: 'prompt',
  token: 'security',
  tool: 'technology',
  vector: 'vector',
  vector_store: 'vector',
  workflow: 'process',
};

const GENERIC_ENTITY_VISUAL_TYPES = new Set(['custom', 'technology', 'concept', 'product']);

export const normalizeKnowledgeGraphType = (value?: string | null) =>
  (value || 'custom').trim().toLowerCase();

export const getKnowledgeGraphRelationVisualKind = (type?: string | null): KnowledgeGraphRelationVisualKind => {
  const normalized = normalizeKnowledgeGraphType(type).replace(/[_-]+/g, ' ');
  if (/\b(conflict\w*|contradict\w*|oppos\w*|incompatib\w*|fail\w*|error\w*|break\w*|regress\w*)\b/.test(normalized)) return 'conflicts';
  if (/\b(preced\w*|follow\w*|before|after|next|previous|then|sequence\w*|step|stage|phase|order\w*|created\s+on|created\s+at)\b/.test(normalized)) return 'sequence';
  if (/\b(protect\w*|prevent\w*|guard\w*|secur\w*|shield\w*|block\w*|den(?:y|ies|ied)|restrict\w*)\b/.test(normalized)) return 'guards';
  if (/\b(support\w*|requir\w*|need\w*|depend\w*|prerequisite\w*|back\w*|assist\w*|provid\w*|collaborat\w*|partner\w*|invest\w*|enabl\w*)\b/.test(normalized)) return 'supports';
  if (/\b(replac\w*|transform\w*|convert\w*|adapt\w*|migrat\w*|swap\w*|substitut\w*|supersed\w*)\b/.test(normalized)) return 'transforms';
  if (/\b(contain\w*|includ\w*|part|member\w*|inside|within|component\w*|compris\w*|subsidiar\w*|employ\w*|works?\s+for)\b/.test(normalized)) return 'contains';
  if (/\b(control\w*|manag\w*|own\w*|govern\w*|configur\w*|orchestrat\w*|rout\w*|deploy\w*|serv\w*|run\w*|operat\w*|regulat\w*|direct\w*|lead\w*|acquir\w*)\b/.test(normalized)) return 'controls';
  if (/\b(creat\w*|generat\w*|produc\w*|emit\w*|writ\w*|extract\w*|build\w*|deriv\w*|found\w*|suppl\w*)\b/.test(normalized)) return 'creates';
  if (/\b(evaluat\w*|measur\w*|monitor\w*|observ\w*|track\w*|inspect\w*|audit\w*|scor\w*|benchmark\w*|rank\w*|compar\w*|test\w*|validat\w*|grad\w*|specif\w*|appl\w*)\b/.test(normalized)) return 'evaluates';
  if (/\b(affect\w*|caus\w*|trigger\w*|driv\w*|influenc\w*|impact\w*|chang\w*|mutat\w*|update\w*|alter\w*)\b/.test(normalized)) return 'influences';
  if (/\b(reference\w*|refer\w*|mention\w*|cite\w*|quote\w*|map\w*|link\w* to|point\w* to)\b/.test(normalized)) return 'references';
  if (/\b(us(?:e|es|ed|ing)|call\w*|send\w*|flow\w*|input\w*|output\w*|connect\w*|pass\w*|handoff\w*)\b/.test(normalized)) return 'flows';
  return 'connects';
};

export const getKnowledgeGraphEntityVisualAlias = (type?: string | null) =>
  ENTITY_VISUAL_TYPE_ALIASES[normalizeKnowledgeGraphType(type)] || normalizeKnowledgeGraphType(type);

export const isKnowledgeGraphGenericEntityVisualType = (type?: string | null) =>
  GENERIC_ENTITY_VISUAL_TYPES.has(normalizeKnowledgeGraphType(type));

export const getKnowledgeGraphEntityVisualType = (type?: string | null, label?: string | null) => {
  const normalized = normalizeKnowledgeGraphType(type);
  const aliased = getKnowledgeGraphEntityVisualAlias(normalized);
  if (aliased !== 'custom' && KNOWN_ENTITY_VISUAL_TYPES.has(aliased) && !GENERIC_ENTITY_VISUAL_TYPES.has(aliased)) {
    return aliased;
  }

  const rawLabel = (label || '').trim();
  const text = rawLabel
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[._/\\-]+/g, ' ')
    .toLowerCase();
  if (!text) return KNOWN_ENTITY_VISUAL_TYPES.has(aliased) ? aliased : normalized;

  if (/\b(seconds?|minutes?|hours?|days?|weeks?|months?|years?|datetime|date)\b|\b\d{4}\b/.test(text)) {
    return 'date';
  }
  if (/\b(model|llm|qwen|llama|mistral|adapter|transformer|tokenizer|checkpoint|gguf|gliner|reranker|classifier|ner|named entity recognition|relation extractor|relationship extractor|relationship model|triplet extractor)\b|[/\\.]gguf\b/.test(text)) {
    return 'model';
  }
  if (/\b(auth|authentication|authorization|secret|secrets|api key|token|jwt|oauth|password|permission|policy|guardrail|credential)\b/.test(text)) {
    return 'security';
  }
  if (/\b(config|configuration|setting|settings|preference|environment variable|env var|parameter|flag|option|yaml|toml)\b/.test(text)) {
    return 'config';
  }
  if (/\b(api|rest api|openapi|sdk|client|endpoint|route|webhook|http interface)\b/.test(text)) {
    return 'api';
  }
  if (/\b(graph\s*rag|graphrag|rag|embedding|embeddings|vector|vectors|vector store|qdrant|faiss|milvus|ann|nearest neighbor|similarity search|retrieval|retrieval index)\b/.test(text)) {
    return 'vector';
  }
  if (/\b(dataset|document|corpus|chunk|embedding store|vector store|qdrant|faiss|milvus|neo4j|redis|database|index|table|schema)\b/.test(text)) {
    return 'data';
  }
  if (/\b(memory|cache|session|conversation history|state store|context window|kv cache|history)\b/.test(text)) {
    return 'memory';
  }
  if (/\b(prompt|template|instruction|system prompt|chat template|few shot|few-shot)\b/.test(text)) {
    return 'prompt';
  }
  if (/\b(api|endpoint|route|service|server|backend|frontend|websocket|proxy|worker|gateway|router)\b/.test(text)) {
    return 'service';
  }
  if (/\b(vllm|llama\\.cpp|llama-cpp|runtime|runner|deployment|deploy|serving|inference server|quantization|quantized|container|docker compose|orchestrator)\b/.test(text)) {
    return 'runtime';
  }
  if (/\b(graph\s*rag|graphrag|rag|embedding|embeddings|vector|vectors|vector store|qdrant|faiss|milvus|ann|nearest neighbor|similarity search|retrieval|retrieval index)\b/.test(text)) {
    return 'vector';
  }
  if (/\b(workflow|pipeline|process|processor|manager|engine|automation|orchestration|extraction|upload|search|training|fine tuning|fine-tuning|quantization|testing|deployment)\b/.test(text)) {
    return 'process';
  }
  if (/\b(eval|evaluation|evaluate|grader|benchmark|leaderboard|scorecard|comparison|compare|test suite|quality gate)\b/.test(text)) {
    return 'evaluation';
  }
  if (/\b(metric|score|accuracy|recall|precision|f1|latency|throughput|tokens?\/s|perplexity)\b/.test(text)) {
    return 'metric';
  }
  if (/\b(dataset|document|corpus|chunk|embedding store|vector store|qdrant|neo4j|redis|database|cache|index|table|schema)\b/.test(text)) {
    return 'data';
  }
  if (/\b(gpu|cuda|nvidia|vram|kernel|tensor core|hardware|device)\b/.test(text)) {
    return 'hardware';
  }
  if (/\b(dockerfile|docker|react|vite|tsx|typescript|python|\\.py|module|package|repository|repo|codebase|library|script|file|class|function)\b|[/\\.](py|tsx?|jsx?|json|ya?ml|md)\b/.test(text)) {
    return 'code';
  }
  if (/\b(technology|component|module|node|tool)\b/.test(text)) {
    return 'technology';
  }
  if (/\b(pipeline|process|workflow|automation|orchestration|extraction|upload|search|evaluation|benchmark|training|fine tuning|fine-tuning|quantization|testing|deployment)\b/.test(text)) {
    return 'process';
  }
  if (/\b(concept|strategy|accuracy|divergence|frontier|specification|format|inference|service|services)\b/.test(text)) {
    return 'concept';
  }
  if (/\b(nvidia|google|openai|anthropic|meta|microsoft|qwen|alibaba|huggingface|research|team|company|organization|foundation|institute|university|labs?|corp|inc)\b/.test(text)) {
    return 'organization';
  }

  return KNOWN_ENTITY_VISUAL_TYPES.has(aliased) ? aliased : normalized;
};

export const getKnowledgeGraphAmbientLabelLimit = (
  detailMode: KnowledgeGraphDetailMode,
  labelDensity: KnowledgeGraphLabelDensity,
  tier: KnowledgeGraphScaleTier,
  zoomTier: KnowledgeGraphZoomTier,
  viewportWidth: number,
  baseLimit: number,
  nodeCount: number
) => {
  if (baseLimit <= 0) return 0;
  if (labelDensity === 'focus') return 0;

  const compactViewport = viewportWidth < 700;
  if (detailMode === 'performance') {
    const performanceLimits: Record<KnowledgeGraphScaleTier, Record<KnowledgeGraphZoomTier, number>> = compactViewport
      ? {
        massive: { far: 2, middle: 3, near: 5 },
        large: { far: 3, middle: 5, near: 8 },
        standard: { far: 5, middle: 9, near: 14 },
        compact: { far: 7, middle: 12, near: 18 },
      }
      : {
        massive: { far: 4, middle: 8, near: 14 },
        large: { far: 8, middle: 16, near: 28 },
        standard: { far: 16, middle: 30, near: 48 },
        compact: { far: 22, middle: 40, near: 64 },
      };
    return Math.min(baseLimit, nodeCount, performanceLimits[tier][zoomTier]);
  }

  const priorityLimits: Record<KnowledgeGraphScaleTier, Record<KnowledgeGraphZoomTier, number>> = {
    massive: compactViewport ? { far: 2, middle: 4, near: 7 } : { far: 5, middle: 10, near: 16 },
    large: compactViewport ? { far: 4, middle: 8, near: 14 } : { far: 12, middle: 22, near: 34 },
    standard: compactViewport ? { far: 10, middle: 18, near: 28 } : { far: 22, middle: 38, near: 58 },
    compact: compactViewport ? { far: 12, middle: 22, near: 34 } : { far: 24, middle: 42, near: 64 },
  };
  const richLimits: Record<KnowledgeGraphScaleTier, Record<KnowledgeGraphZoomTier, number>> = {
    massive: compactViewport ? { far: 3, middle: 6, near: 10 } : { far: 7, middle: 14, near: 24 },
    large: compactViewport ? { far: 5, middle: 10, near: 18 } : { far: 18, middle: 32, near: 54 },
    standard: compactViewport ? { far: 12, middle: 24, near: 38 } : { far: 32, middle: 58, near: 86 },
    compact: compactViewport ? { far: 14, middle: 28, near: 44 } : { far: 36, middle: 66, near: 98 },
  };
  const limitByTier = labelDensity === 'rich' ? richLimits : priorityLimits;
  const smallGraphLabelTarget = tier === 'compact' && nodeCount <= 90
    ? Math.min(nodeCount, compactViewport ? labelDensity === 'rich' ? 28 : 12 : labelDensity === 'rich' ? 42 : 18)
    : tier === 'standard' && nodeCount <= 260 && !compactViewport
      ? Math.min(nodeCount, labelDensity === 'rich' ? 68 : 32)
      : 0;

  const detailBoost = detailMode === 'detail' && (tier === 'compact' || tier === 'standard') && zoomTier === 'near' && !compactViewport ? 4 : 0;
  return Math.min(baseLimit, Math.max(limitByTier[tier][zoomTier] + detailBoost, smallGraphLabelTarget));
};

export const getKnowledgeGraphAnchorLabelLimit = (
  detailMode: KnowledgeGraphDetailMode,
  labelDensity: KnowledgeGraphLabelDensity,
  tier: KnowledgeGraphScaleTier,
  zoomTier: KnowledgeGraphZoomTier,
  viewportWidth: number,
  nodeCount: number
) => {
  if (labelDensity !== 'focus') return 0;
  if (nodeCount <= 0) return 0;

  const compactViewport = viewportWidth < 700;
  const limitByTier: Record<KnowledgeGraphScaleTier, Record<KnowledgeGraphZoomTier, number>> = compactViewport
    ? {
      massive: { far: 2, middle: 3, near: 5 },
      large: { far: 3, middle: 5, near: 8 },
      standard: { far: 6, middle: 10, near: 14 },
      compact: { far: 8, middle: 13, near: 18 },
    }
    : {
      massive: { far: 5, middle: 8, near: 12 },
      large: { far: 9, middle: 15, near: 24 },
      standard: { far: 16, middle: 28, near: 40 },
      compact: { far: 20, middle: 34, near: 48 },
    };

  const limit = limitByTier[tier][zoomTier];
  return Math.min(nodeCount, detailMode === 'performance' ? Math.max(1, Math.floor(limit * 0.42)) : limit);
};

export const getKnowledgeGraphPriorityLabelQualityFloor = (tier: KnowledgeGraphScaleTier) => {
  if (tier === 'massive') return 150;
  if (tier === 'large') return 112;
  if (tier === 'standard') return 82;
  return 64;
};
