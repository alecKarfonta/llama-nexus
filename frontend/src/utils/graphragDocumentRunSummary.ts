type JsonRecord = Record<string, unknown>;

export interface GraphRAGDocumentRunSummary {
  status: string;
  readyForChat: boolean;
  chunkCount: number;
  entityCount: number;
  relationshipCount: number;
  stageCount: number;
  completedStageCount: number;
  sampleCount: number;
}

export interface GraphRAGDocumentRunGroup {
  run: JsonRecord;
  runs: JsonRecord[];
  runCount: number;
  domains: string[];
  chunks: number;
  entities: number;
  relationships: number;
  chatReady: boolean;
}

export interface GraphRAGDocumentLibrarySummary {
  chunks: number;
  entities: number;
  relationships: number;
  chatReadyCount: number;
  hasHistory: boolean;
}

export interface GraphRAGWorkbenchPipelineResultInput {
  corpusResult?: unknown;
  selectedPipelineRunId?: string | null;
  selectedBatchRun?: unknown;
  selectedRecentRun?: unknown;
}

export interface GraphRAGRichEvidenceOutputSummary {
  count: number;
  title: string;
  body: string;
  sampleGroups: string[];
}

interface GraphRAGDocumentStageSummary {
  completed: boolean;
  sampleCount: number;
}

const isRecord = (value: unknown): value is JsonRecord =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const asRecord = (value: unknown): JsonRecord => (isRecord(value) ? value : {});

const asArray = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);

const asString = (value: unknown, fallback = ""): string =>
  typeof value === "string" && value.trim() ? value.trim() : fallback;

const asNumber = (value: unknown): number => {
  if (typeof value === "number" && Number.isFinite(value)) return Math.max(0, value);
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.max(0, parsed);
  }
  return 0;
};

const nonEmptyRecord = (value: unknown): JsonRecord | undefined => {
  if (!isRecord(value)) return undefined;
  return Object.keys(value).length > 0 ? value : undefined;
};

const isCompleteStatus = (status: string): boolean =>
  ["complete", "completed", "ready", "success"].includes(status);

const pipelineRunTimestamp = (run: JsonRecord): number => {
  const timestamps = [
    run.completed_at,
    run.updated_at,
    run.created_at,
    run.started_at,
  ]
    .map((value) => Date.parse(asString(value)))
    .filter(Number.isFinite);
  return timestamps.length > 0 ? Math.max(...timestamps) : 0;
};

const sampleOutputCount = (value: unknown): number => {
  if (Array.isArray(value)) return value.length;
  if (isRecord(value)) return Object.keys(value).length > 0 ? 1 : 0;
  if (typeof value === "string") return value.trim() ? 1 : 0;
  if (typeof value === "number" || typeof value === "boolean") return 1;
  return 0;
};

const pluralize = (value: number, singular: string, plural = `${singular}s`): string =>
  `${value.toLocaleString()} ${value === 1 ? singular : plural}`;

export const buildGraphRAGRichEvidenceOutputSummary = (
  evidenceProfile: unknown,
): GraphRAGRichEvidenceOutputSummary | null => {
  const profile = asRecord(evidenceProfile);
  const quickCounts = asRecord(profile.quick_counts);
  const graphEvidence = asRecord(profile.graph_evidence);

  const families = [
    {
      key: "charts",
      priority: 120,
      count: asNumber(quickCounts.charts),
      label: "chart",
      plural: "charts",
      relationCount: asNumber(graphEvidence.chart_relationship_count),
      relationLabel: "chart link",
    },
    {
      key: "tables",
      priority: 115,
      count: asNumber(quickCounts.tables) || asNumber(quickCounts.table_elements),
      label: "table",
      plural: "tables",
      relationCount: asNumber(graphEvidence.table_fact_count),
      relationLabel: "table fact",
    },
    {
      key: "forms",
      priority: 110,
      count: asNumber(quickCounts.forms) || asNumber(quickCounts.form_fields),
      label: "form",
      plural: "forms",
      relationCount:
        asNumber(graphEvidence.form_field_relationship_count) ||
        asNumber(graphEvidence.form_relationship_count),
      relationLabel: "form link",
    },
    {
      key: "styled_text",
      priority: 105,
      count: asNumber(quickCounts.styled_text_blocks) || asNumber(quickCounts.styled_text),
      label: "styled text",
      plural: "styled text",
      relationCount: asNumber(graphEvidence.styled_text_relationship_count),
      relationLabel: "style link",
    },
    {
      key: "connectors",
      priority: 100,
      count: asNumber(quickCounts.connectors),
      label: "connector",
      plural: "connectors",
      relationCount: asNumber(graphEvidence.connector_relationship_count),
      relationLabel: "connector link",
    },
    {
      key: "diagrams",
      priority: 80,
      count: asNumber(quickCounts.diagrams),
      label: "diagram",
      plural: "diagrams",
      relationCount: asNumber(graphEvidence.diagram_relationship_count),
      relationLabel: "diagram link",
    },
    {
      key: "notes",
      priority: 76,
      count: asNumber(quickCounts.speaker_notes),
      label: "speaker note",
      plural: "speaker notes",
      relationCount: asNumber(graphEvidence.speaker_notes_relationship_count),
      relationLabel: "note link",
    },
    {
      key: "links",
      priority: 72,
      count: asNumber(quickCounts.links),
      label: "link",
      plural: "links",
      relationCount: asNumber(graphEvidence.link_relationship_count),
      relationLabel: "URL link",
    },
    {
      key: "attachments",
      priority: 68,
      count: asNumber(quickCounts.attachments),
      label: "attachment",
      plural: "attachments",
      relationCount: asNumber(graphEvidence.attachment_relationship_count),
      relationLabel: "attachment link",
    },
  ]
    .filter((family) => family.count > 0 || family.relationCount > 0)
    .sort((left, right) =>
      right.priority - left.priority ||
      right.count - left.count ||
      right.relationCount - left.relationCount ||
      left.key.localeCompare(right.key),
    );

  if (families.length === 0) return null;

  const primaryFamilies = families.slice(0, 5);
  const primaryFamilyKeys = new Set(primaryFamilies.map((family) => family.key));
  const sampleGroups = primaryFamilies
    .map((family) => pluralize(family.count || family.relationCount, family.label, family.plural));
  const primaryRelationGroups = primaryFamilies
    .filter((family) => family.relationCount > 0)
    .map((family) => pluralize(family.relationCount, family.relationLabel));
  const fallbackRelationGroups = families
    .filter((family) => family.relationCount > 0 && !primaryFamilyKeys.has(family.key))
    .sort((left, right) => right.relationCount - left.relationCount || left.key.localeCompare(right.key))
    .map((family) => pluralize(family.relationCount, family.relationLabel));
  const relationGroups = [...primaryRelationGroups, ...fallbackRelationGroups].slice(0, 4);
  const sourceEvidenceTotal = families.reduce((total, family) => total + family.count, 0);
  const graphEvidenceTotal = families.reduce((total, family) => total + family.relationCount, 0);

  return {
    count: Math.max(sourceEvidenceTotal, graphEvidenceTotal),
    title: sampleGroups.join(" · "),
    body: relationGroups.length > 0
      ? relationGroups.join(" · ")
      : "Structured evidence is available for citation-aware chat.",
    sampleGroups,
  };
};

export const summarizeGraphRAGDocumentPipelineRun = (
  run: unknown,
): GraphRAGDocumentRunSummary => {
  const record = asRecord(run);
  const status = asString(record.status, "unknown").toLowerCase();
  const summary = asRecord(record.summary);
  const stages = asArray(record.stages).filter(isRecord);
  const resumeHandoff = asRecord(record.resume_handoff);
  const interactionHandoff = asRecord(record.interaction_handoff);

  const stageSummaries: GraphRAGDocumentStageSummary[] = stages.map((stage) => {
    const stageStatus = asString(stage.status, "unknown").toLowerCase();
    const sampleOutputs = asRecord(stage.sample_outputs);
    return {
      completed: isCompleteStatus(stageStatus),
      sampleCount: Object.values(sampleOutputs).reduce<number>(
        (total, output) => total + sampleOutputCount(output),
        0,
      ),
    };
  });

  return {
    status,
    readyForChat:
      resumeHandoff.ready === true ||
      interactionHandoff.ready === true ||
      isCompleteStatus(status),
    chunkCount: asNumber(summary.chunk_count),
    entityCount: asNumber(summary.entity_candidate_count),
    relationshipCount: asNumber(summary.relationship_candidate_count),
    stageCount: stages.length,
    completedStageCount: stageSummaries.filter((stage) => stage.completed).length,
    sampleCount: stageSummaries.reduce<number>(
      (total, stage) => total + stage.sampleCount,
      0,
    ),
  };
};

export const graphRAGDocumentPipelineRunKey = (
  run: unknown,
  fallbackDomain = "general",
): string => {
  const record = asRecord(run);
  const document = asRecord(record.document);
  const filename = asString(document.filename) || asString(record.run_id);
  const domain = asString(document.domain) || fallbackDomain;
  return `${domain.toLowerCase()}::${filename.toLowerCase()}`;
};

export const graphRAGPipelineRunsShareDocument = (runs: unknown[]): boolean => {
  const keys = new Set(
    runs
      .map((run) => graphRAGDocumentPipelineRunKey(run))
      .filter(Boolean),
  );
  return keys.size === 1;
};

export const graphRAGPipelineRunReadyForChat = (run: unknown): boolean =>
  summarizeGraphRAGDocumentPipelineRun(run).readyForChat;

export const preferredGraphRAGPipelineRunForDocumentChat = (
  runs: unknown[],
): JsonRecord | null => {
  const usableRuns = runs
    .map(asRecord)
    .filter((run) => asString(run.run_id));
  if (usableRuns.length === 0) return null;

  return [...usableRuns].sort((left, right) => {
    const readyDelta =
      Number(graphRAGPipelineRunReadyForChat(right)) -
      Number(graphRAGPipelineRunReadyForChat(left));
    if (readyDelta !== 0) return readyDelta;
    return pipelineRunTimestamp(right) - pipelineRunTimestamp(left);
  })[0];
};

export const selectGraphRAGWorkbenchPipelineResult = ({
  corpusResult,
  selectedPipelineRunId,
  selectedBatchRun,
  selectedRecentRun,
}: GraphRAGWorkbenchPipelineResultInput): JsonRecord => {
  const resultRecord = asRecord(corpusResult);
  const selectedRunId = asString(selectedPipelineRunId || "");
  const loadedRun = nonEmptyRecord(resultRecord.pipeline_run);
  const loadedRunId = asString(loadedRun?.run_id);
  const selectedLoadedRun =
    selectedRunId && loadedRunId === selectedRunId ? loadedRun : undefined;
  const defaultLoadedRun =
    !selectedRunId && loadedRunId ? loadedRun : undefined;

  return (
    nonEmptyRecord(resultRecord.pipeline_upload) ||
    nonEmptyRecord(resultRecord.pipeline_preview) ||
    selectedLoadedRun ||
    nonEmptyRecord(selectedRecentRun) ||
    nonEmptyRecord(selectedBatchRun) ||
    defaultLoadedRun ||
    {}
  );
};

export const groupGraphRAGDocumentPipelineRuns = (
  runs: unknown[],
  options: { fallbackDomain?: string; groupAcrossDomains?: boolean } = {},
): GraphRAGDocumentRunGroup[] => {
  const groups = runs.reduce<Map<string, GraphRAGDocumentRunGroup>>(
    (groupMap, run, index) => {
      const record = asRecord(run);
      const document = asRecord(record.document);
      const filename = asString(document.filename, `Run ${index + 1}`);
      const domain = asString(document.domain, options.fallbackDomain || "general");
      const key = options.groupAcrossDomains
        ? filename.toLowerCase()
        : `${domain.toLowerCase()}::${filename.toLowerCase()}`;
      const summary = summarizeGraphRAGDocumentPipelineRun(record);
      const current = groupMap.get(key);

      if (current) {
        current.runs.push(record);
        current.runCount += 1;
        if (!current.domains.includes(domain)) {
          current.domains.push(domain);
        }
        current.chunks += summary.chunkCount;
        current.entities += summary.entityCount;
        current.relationships += summary.relationshipCount;
        current.chatReady = current.chatReady || summary.readyForChat;
        return groupMap;
      }

      groupMap.set(key, {
        run: record,
        runs: [record],
        runCount: 1,
        domains: [domain],
        chunks: summary.chunkCount,
        entities: summary.entityCount,
        relationships: summary.relationshipCount,
        chatReady: summary.readyForChat,
      });
      return groupMap;
    },
    new Map(),
  );

  return Array.from(groups.values())
    .map((group) => {
      const preferredRun =
        preferredGraphRAGPipelineRunForDocumentChat(group.runs) || group.run;
      const preferredDomain = asString(
        asRecord(preferredRun.document).domain,
        options.fallbackDomain || "general",
      );
      const domains = [
        preferredDomain,
        ...group.domains.filter((domain) => domain !== preferredDomain),
      ];

      return {
        ...group,
        run: preferredRun,
        domains,
      };
    })
    .sort(
      (left, right) =>
        pipelineRunTimestamp(right.run) - pipelineRunTimestamp(left.run),
    );
};

export const summarizeGraphRAGDocumentLibrary = (
  groups: GraphRAGDocumentRunGroup[],
): GraphRAGDocumentLibrarySummary =>
  groups.reduce<GraphRAGDocumentLibrarySummary>(
    (totals, group) => {
      const preferredRun =
        preferredGraphRAGPipelineRunForDocumentChat(group.runs) ||
        group.run;
      const summary = summarizeGraphRAGDocumentPipelineRun(preferredRun);
      return {
        chunks: totals.chunks + summary.chunkCount,
        entities: totals.entities + summary.entityCount,
        relationships: totals.relationships + summary.relationshipCount,
        chatReadyCount: totals.chatReadyCount + (group.chatReady ? 1 : 0),
        hasHistory: totals.hasHistory || group.runCount > 1,
      };
    },
    {
      chunks: 0,
      entities: 0,
      relationships: 0,
      chatReadyCount: 0,
      hasHistory: false,
    },
  );
