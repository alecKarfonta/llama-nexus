type JsonRecord = Record<string, unknown>;

export interface GraphRAGDocumentOCRStatusCard {
  id: 'ocr';
  label: 'OCR';
  status: 'ready' | 'disabled' | 'blocked' | 'unavailable';
  ready: boolean;
  enabled: boolean;
  present: boolean;
  detail: string;
  meta: string;
}

export interface GraphRAGServiceStatusEntry {
  name: string;
  data: JsonRecord;
  ready: boolean;
  configured: boolean;
  available: boolean;
  optional: boolean;
  status: string;
  url: string;
  healthUrl: string;
  compatibleHealthUrls: string[];
  attemptedHealthUrls: string[];
  healthAttempts: GraphRAGServiceHealthAttempt[];
  deploymentHint: string;
  requiredEnv: string[];
  description: string;
  urlSource: string;
  urlSourceLabel: string;
  autoDiscovered: boolean;
  explicit: boolean;
  message: string;
}

export interface GraphRAGServiceHealthAttempt {
  url: string;
  status: string;
  statusCode?: number;
  error: string;
}

export interface GraphRAGServiceContractItem {
  id: string;
  label: string;
  status: string;
  detail: string;
  meta: string;
}

export interface GraphRAGServiceEvidenceCapability {
  id: string;
  label: string;
  status: string;
  ready: boolean;
  formats: string[];
  elementTypes: string[];
  relationshipTypes: string[];
  interactionOutputs: string[];
  detail: string;
}

export interface GraphRAGServiceSmokeTest {
  id: string;
  label: string;
  method: string;
  scope: string;
  status: string;
  ready: boolean;
  endpoint: string;
  candidates: string[];
  command: string;
  expected: string;
  traceFields: string[];
}

export interface GraphRAGServiceWorkflowEntry {
  id: string;
  label: string;
  status: string;
  ready: boolean;
  endpoint: string;
  message: string;
  nextStep: string;
  tooltip: string;
}

export interface GraphRAGServiceChecklistItem {
  id: string;
  index: number;
  label: string;
  status: string;
  ready: boolean;
  detail: string;
  action: string;
  endpoint: string;
  mode: string;
  checks: string[];
}

export interface GraphRAGServiceQuickstartItem {
  id: string;
  index: number;
  label: string;
  status: string;
  ready: boolean;
  summary: string;
  command: string;
  envVars: string[];
  profiles: string[];
  checks: string[];
  endpoint: string;
  nextStep: string;
}

export interface GraphRAGServiceInteractionPathway {
  id: string;
  index: number;
  label: string;
  status: string;
  ready: boolean;
  recommended: boolean;
  summary: string;
  action: string;
  mode: string;
  endpoint: string;
  secondaryEndpoint: string;
  checks: string[];
  outputs: string[];
  usesLocalPipeline: boolean;
  usesRemoteGraph: boolean;
}

export interface NormalizedGraphRAGServiceStatus {
  meta: JsonRecord;
  services: JsonRecord;
  entries: GraphRAGServiceStatusEntry[];
  readyCount: number;
  configuredCount: number;
  requiredServicesReady: boolean;
  graphRAGCoreService?: GraphRAGServiceStatusEntry;
  loaded: boolean;
  remoteGraphRAGReady: boolean;
  smokeStatus: string;
  summary: string;
  sourceSummary: string;
  nextSteps: string[];
  operatingModeLabel: string;
  operatingModeSummary: string;
  operatingModeAction: string;
  documentOCRStatus: GraphRAGDocumentOCRStatusCard;
  contractItems: GraphRAGServiceContractItem[];
  evidenceCapabilities: GraphRAGServiceEvidenceCapability[];
  smokeTests: GraphRAGServiceSmokeTest[];
  workflowEntries: GraphRAGServiceWorkflowEntry[];
  operatorChecklist: GraphRAGServiceChecklistItem[];
  deploymentQuickstart: GraphRAGServiceQuickstartItem[];
  interactionPathways: GraphRAGServiceInteractionPathway[];
  recommendedPathwayId: string;
  workflowReadyCount: number;
  workflowCount: number;
  severity: 'success' | 'warning' | 'info' | 'error';
}

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const asString = (value: unknown, fallback = ''): string =>
  typeof value === 'string' && value.trim()
    ? value.trim()
    : typeof value === 'number' && Number.isFinite(value)
      ? String(value)
      : fallback;

const asNumber = (value: unknown, fallback = 0): number => {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : fallback;
};

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asStringArray = (value: unknown): string[] =>
  Array.isArray(value)
    ? value.map(item => asString(item)).filter(Boolean)
    : [];

const asOptionalNumber = (value: unknown): number | undefined => {
  const parsed = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN;
  return Number.isFinite(parsed) ? parsed : undefined;
};

const uniqueStrings = (values: string[]): string[] => Array.from(new Set(values.filter(Boolean)));

const normalizeHealthAttempts = (value: unknown): GraphRAGServiceHealthAttempt[] =>
  asArray(value).map((item, index) => {
    const data = isRecord(item) ? item : {};
    const url = asString(data.url);
    const status = asString(data.status, url ? 'attempted' : `attempt-${index + 1}`);
    const statusCode = asOptionalNumber(data.status_code);
    const error = asString(data.error);

    return {
      url,
      status,
      ...(statusCode !== undefined ? { statusCode } : {}),
      error,
    };
  }).filter(attempt => attempt.url || attempt.status || attempt.error);

const formatKey = (value: string): string =>
  value
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, letter => letter.toUpperCase());

export const normalizeGraphRAGDocumentOCRStatus = (value: unknown): GraphRAGDocumentOCRStatusCard => {
  if (!isRecord(value)) {
    return {
      id: 'ocr',
      label: 'OCR',
      status: 'unavailable',
      ready: false,
      enabled: false,
      present: false,
      detail: 'OCR runtime status has not been reported by the backend yet.',
      meta: '',
    };
  }

  const ready = value.ready === true || value.available === true;
  const enabled = value.enabled === true || value.configured === true;
  const rawStatus = asString(value.status).toLowerCase();
  const runtime = isRecord(value.runtime) ? value.runtime : {};
  const missing = asStringArray(value.missing);
  const engine = asString(value.engine, 'pytesseract');
  const tesseractPath = asString(runtime.tesseract_path);
  const requiredEnv = asStringArray(value.required_env);
  const message = asString(value.message);

  const status: GraphRAGDocumentOCRStatusCard['status'] = ready
    ? 'ready'
    : !enabled || rawStatus === 'disabled'
      ? 'disabled'
      : rawStatus === 'unavailable'
        ? 'unavailable'
        : 'blocked';

  const detail = message || (
    ready
      ? 'Image OCR is enabled for scanned pages and packaged document images.'
      : enabled
        ? 'Image OCR is enabled, but the runtime needs attention.'
        : 'Enable image OCR to extract text from scanned pages and packaged document images.'
  );

  const meta = ready
    ? [engine, tesseractPath].filter(Boolean).join(' · ')
    : missing.length > 0
      ? `Missing ${missing.map(formatKey).join(', ')}`
      : requiredEnv.length > 0
        ? requiredEnv.join(' / ')
        : asString(value.deployment_hint);

  return {
    id: 'ocr',
    label: 'OCR',
    status,
    ready,
    enabled,
    present: true,
    detail,
    meta,
  };
};

export const normalizeGraphRAGServiceStatus = (
  value: unknown,
  health: unknown = undefined,
): NormalizedGraphRAGServiceStatus => {
  const raw = isRecord(value) ? value : {};
  const healthRecord = isRecord(health) ? health : {};
  const meta = isRecord(raw._meta) ? raw._meta : {};
  const nestedServices = isRecord(raw.services) ? raw.services : {};
  const services = Object.keys(nestedServices).length > 0
    ? nestedServices
    : Object.fromEntries(Object.entries(raw).filter(([name]) => !name.startsWith('_')));

  const entries = Object.entries(services).map(([name, serviceValue]) => {
    const data = isRecord(serviceValue) ? serviceValue : {};
    const service = isRecord(data.service) ? data.service : {};
    const ready = data.ready === true;
    const configured = data.configured === true || data.enabled === true;
    const available = data.available === true;
    const optional = data.optional === true;
    const requiredEnv = asStringArray(data.required_env);
    const status = asString(data.status, ready ? 'ready' : configured ? 'unavailable' : 'not_configured');
    const url = asString(data.url, asString(service.url));
    const healthUrl = asString(service.health_url);
    const compatibleHealthUrls = uniqueStrings([
      ...asStringArray(service.compatible_health_urls),
      ...asStringArray(data.compatible_health_urls),
      healthUrl,
    ]);
    const rawHealthAttempts = [
      ...normalizeHealthAttempts(service.health_attempts),
      ...normalizeHealthAttempts(data.health_attempts),
    ];
    const healthAttempts = rawHealthAttempts.filter((attempt, index) => {
      const key = `${attempt.url}|${attempt.status}|${attempt.statusCode ?? ''}|${attempt.error}`;
      return rawHealthAttempts.findIndex(candidate =>
        `${candidate.url}|${candidate.status}|${candidate.statusCode ?? ''}|${candidate.error}` === key,
      ) === index;
    });
    const attemptedHealthUrls = uniqueStrings([
      ...asStringArray(service.attempted_health_urls),
      ...asStringArray(data.attempted_health_urls),
      ...healthAttempts.map(attempt => attempt.url),
    ]);
    const deploymentHint = asString(data.deployment_hint, asString(data.deploy_command, asString(service.deployment_hint)));
    const urlSource = asString(data.url_source, asString(service.url_source, asString(data.source, asString(service.source))));
    const urlSourceLabel = asString(
      data.url_source_label,
      asString(service.url_source_label, urlSource ? formatKey(urlSource) : ''),
    );
    const autoDiscovered = data.auto_discovered === true || service.auto_discovered === true;
    const explicit = data.explicit === true || service.explicit === true;
    const message = asString(
      service.message,
      asString(
        data.message,
        ready
          ? 'Health probe passed.'
          : configured
            ? 'Configured service did not pass the health probe.'
            : `Set ${requiredEnv[0] || 'service URL'} to enable.`,
      ),
    );

    return {
      name,
      data,
      ready,
      configured,
      available,
      optional,
      status,
      url,
      healthUrl,
      compatibleHealthUrls,
      attemptedHealthUrls,
      healthAttempts,
      deploymentHint,
      requiredEnv,
      description: asString(data.description),
      urlSource,
      urlSourceLabel,
      autoDiscovered,
      explicit,
      message,
    };
  });

  const readyCount = entries.filter(service => service.ready).length;
  const configuredCount = entries.filter(service => service.configured).length;
  const requiredServiceEntries = entries.filter(service => !service.optional);
  const requiredServicesReady = requiredServiceEntries.length > 0 && requiredServiceEntries.every(service => service.ready);
  const graphRAGCoreService = entries.find(service => service.name === 'graphrag');
  const metaStatus = asString(meta.status);
  const smokeStatus = ['ready', 'partial', 'local', 'disabled', 'offline'].includes(metaStatus)
    ? metaStatus
    : entries.length === 0
      ? 'offline'
      : requiredServicesReady
        ? 'ready'
        : readyCount > 0
          ? 'partial'
          : healthRecord.graphrag_enabled === false
            ? 'disabled'
            : 'offline';
  const operatingMode = isRecord(meta.operating_mode) ? meta.operating_mode : {};
  const interactionContract = isRecord(meta.interaction_contract) ? meta.interaction_contract : {};
  const uploadContract = isRecord(interactionContract.upload) ? interactionContract.upload : {};
  const chatContract = isRecord(interactionContract.chat) ? interactionContract.chat : {};
  const searchContract = isRecord(interactionContract.search) ? interactionContract.search : {};
  const documentPipelineContract = isRecord(interactionContract.document_pipeline)
    ? interactionContract.document_pipeline
    : {};
  const deploymentContract = isRecord(interactionContract.deployment) ? interactionContract.deployment : {};
  const documentOCRStatus = normalizeGraphRAGDocumentOCRStatus(
    meta.document_ocr ?? deploymentContract.document_ocr ?? healthRecord.document_ocr,
  );
  const pipelineFormatCount = asNumber(documentPipelineContract.format_count, asStringArray(documentPipelineContract.supported_formats).length);
  const pipelineMeta = pipelineFormatCount > 0
    ? `${pipelineFormatCount} formats`
    : asStringArray(documentPipelineContract.supported_formats).slice(0, 4).map(formatKey).join(', ');
  const evidenceCapabilities = asArray(documentPipelineContract.evidence_capabilities).map((item, index) => {
    const data = isRecord(item) ? item : {};
    const id = asString(data.id, `evidence-${index + 1}`);
    const label = asString(data.label, formatKey(id));
    const status = asString(data.status, documentPipelineContract.ready === false ? 'blocked' : asString(documentPipelineContract.status, 'local'));

    return {
      id,
      label,
      status,
      ready: data.ready === true || status === 'ready' || status === 'local',
      formats: asStringArray(data.formats),
      elementTypes: asStringArray(data.element_types),
      relationshipTypes: asStringArray(data.relationship_types),
      interactionOutputs: asStringArray(data.interaction_outputs),
      detail: asString(data.detail),
    };
  }).filter(capability => capability.label);
  const evidenceLabels = evidenceCapabilities.map(capability => capability.label);
  const evidenceStatus = asString(
    documentPipelineContract.status,
    documentPipelineContract.ready === false ? 'blocked' : 'local',
  );
  const contractItems: GraphRAGServiceContractItem[] = [
    {
      id: 'pipeline',
      label: 'Pipeline',
      status: asString(
        documentPipelineContract.status,
        documentPipelineContract.available === true ? 'local' : 'blocked',
      ),
      detail: formatKey(asString(documentPipelineContract.mode, 'local_document_evidence_preview')),
      meta: pipelineMeta || asString(documentPipelineContract.endpoint),
    },
    {
      id: 'upload',
      label: 'Upload',
      status: uploadContract.remote_submission === true
        ? 'ready'
        : uploadContract.available === true
          ? 'local'
          : 'blocked',
      detail: formatKey(asString(uploadContract.behavior, 'unavailable')),
      meta: asString(uploadContract.preferred_endpoint, asString(uploadContract.endpoint_strategy)),
    },
    {
      id: 'chat',
      label: 'Chat',
      status: chatContract.available === true ? 'ready' : 'blocked',
      detail: formatKey(asString(chatContract.mode, 'standard_chat')),
      meta: asStringArray(chatContract.context_sources).map(formatKey).join(' + '),
    },
    {
      id: 'search',
      label: 'Search',
      status: searchContract.available === true ? 'ready' : searchContract.fallback ? 'local' : 'blocked',
      detail: searchContract.available === true
        ? 'Remote Graph'
        : formatKey(asString(searchContract.fallback, 'blocked')),
      meta: asString(searchContract.endpoint),
    },
    ...(evidenceCapabilities.length > 0 ? [{
      id: 'evidence',
      label: 'Evidence',
      status: evidenceStatus,
      detail: evidenceLabels.slice(0, 6).join(' + '),
      meta: `${evidenceCapabilities.length} families`,
    }] : []),
    ...(documentOCRStatus.present ? [documentOCRStatus] : []),
  ].filter(item => item.detail || item.meta);

  const smokeTests = asArray(deploymentContract.smoke_tests).map((item, index) => {
    const data = isRecord(item) ? item : {};
    const endpoint = asString(data.url, asString(data.path, asString(data.preferred_url)));
    const candidates = asStringArray(data.urls);
    const traceFields = asStringArray(data.trace_fields);
    const status = asString(
      data.status,
      data.ready === true
        ? 'ready'
        : endpoint || candidates.length > 0
          ? 'blocked'
          : 'not_configured',
    );

    return {
      id: asString(data.id, `smoke-${index + 1}`),
      label: asString(data.label, `Smoke test ${index + 1}`),
      method: asString(data.method, 'GET'),
      scope: asString(data.scope),
      status,
      ready: data.ready === true,
      endpoint,
      candidates,
      command: asString(data.command),
      expected: asString(data.expected),
      traceFields,
    };
  });

  const workflowEntries = Object.entries(isRecord(meta.workflows) ? meta.workflows : {}).map(([id, workflow]) => {
    const data = isRecord(workflow) ? workflow : {};
    const status = asString(data.status, data.ready === true ? 'ready' : 'blocked');
    const label = asString(data.label, formatKey(id));
    const endpoint = asString(data.endpoint);
    const message = asString(data.message);
    const nextStep = asString(data.next_step);
    const tooltip = [message, nextStep || endpoint].filter(Boolean).join(' ');

    return {
      id,
      label,
      status,
      ready: data.ready === true,
      endpoint,
      message,
      nextStep,
      tooltip,
    };
  });

  const operatorChecklist = asArray(meta.operator_checklist).map((item, index) => {
    const data = isRecord(item) ? item : {};
    const status = asString(data.status, data.ready === true ? 'ready' : 'blocked');

    return {
      id: asString(data.id, `step-${index + 1}`),
      index,
      label: asString(data.label, `Step ${index + 1}`),
      status,
      ready: data.ready === true,
      detail: asString(data.detail),
      action: asString(data.action),
      endpoint: asString(data.endpoint),
      mode: asString(data.mode),
      checks: asStringArray(data.checks),
    };
  });

  const deploymentQuickstart = asArray(meta.deployment_quickstart).map((item, index) => {
    const data = isRecord(item) ? item : {};
    const status = asString(data.status, data.ready === true ? 'ready' : 'blocked');

    return {
      id: asString(data.id, `quickstart-${index + 1}`),
      index,
      label: asString(data.label, `Quickstart ${index + 1}`),
      status,
      ready: data.ready === true,
      summary: asString(data.summary),
      command: asString(data.command),
      envVars: asStringArray(data.env_vars),
      profiles: asStringArray(data.profiles),
      checks: asStringArray(data.checks),
      endpoint: asString(data.endpoint),
      nextStep: asString(data.next_step),
    };
  });
  const interactionPathways = asArray(meta.interaction_pathways).map((item, index) => {
    const data = isRecord(item) ? item : {};
    const status = asString(data.status, data.ready === true ? 'ready' : 'blocked');

    return {
      id: asString(data.id, `pathway-${index + 1}`),
      index,
      label: asString(data.label, `Pathway ${index + 1}`),
      status,
      ready: data.ready === true || ['ready', 'local', 'partial'].includes(status),
      recommended: data.recommended === true,
      summary: asString(data.summary),
      action: asString(data.action),
      mode: asString(data.mode),
      endpoint: asString(data.endpoint),
      secondaryEndpoint: asString(data.secondary_endpoint),
      checks: asStringArray(data.checks),
      outputs: asStringArray(data.outputs),
      usesLocalPipeline: data.uses_local_pipeline === true,
      usesRemoteGraph: data.uses_remote_graph === true,
    };
  });
  const recommendedPathwayId = asString(
    meta.recommended_pathway_id,
    interactionPathways.find(pathway => pathway.recommended)?.id || '',
  );

  const workflowReadyCount = asNumber(
    meta.workflow_ready_count,
    workflowEntries.filter(workflow => workflow.ready).length,
  );
  const workflowCount = asNumber(meta.workflow_count, workflowEntries.length);
  const severity: NormalizedGraphRAGServiceStatus['severity'] = smokeStatus === 'ready'
    ? 'success'
    : smokeStatus === 'partial'
      ? 'warning'
      : smokeStatus === 'disabled' || smokeStatus === 'local'
        ? 'info'
        : 'error';

  return {
    meta,
    services,
    entries,
    readyCount,
    configuredCount,
    requiredServicesReady,
    graphRAGCoreService,
    loaded: entries.length > 0,
    remoteGraphRAGReady: graphRAGCoreService?.ready === true,
    smokeStatus,
    summary: asString(meta.summary),
    sourceSummary: asString(meta.source_summary),
    nextSteps: asStringArray(meta.next_steps),
    operatingModeLabel: asString(operatingMode.label),
    operatingModeSummary: asString(operatingMode.summary),
    operatingModeAction: asString(operatingMode.recommended_action),
    documentOCRStatus,
    contractItems,
    evidenceCapabilities,
    smokeTests,
    workflowEntries,
    operatorChecklist,
    deploymentQuickstart,
    interactionPathways,
    recommendedPathwayId,
    workflowReadyCount,
    workflowCount,
    severity,
  };
};
