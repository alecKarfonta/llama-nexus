import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  Grid,
  IconButton,
  InputAdornment,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
  useMediaQuery,
} from "@mui/material";
import { alpha, useTheme } from "@mui/material/styles";
import {
  AccountTree as GraphIcon,
  Analytics as AnalyticsIcon,
  Article as ArticleIcon,
  AutoFixHigh as ExtractionIcon,
  CloudUpload as UploadIcon,
  DataObject as DataIcon,
  Description as DocumentIcon,
  OpenInNew as OpenIcon,
  FactCheck as QualityIcon,
  PlayArrow as RunIcon,
  Psychology as ReasoningIcon,
  QuestionAnswer as ChatIcon,
  Refresh as RefreshIcon,
  Search as SearchIcon,
  Send as SendIcon,
  Stop as StopIcon,
  Storage as StorageIcon,
} from "@mui/icons-material";
import {
  CodeHighlighter,
  MetricTile,
  PageShell,
  SectionCard,
  StatusChip,
  type StatusChipConfigMap,
} from "@/components/common";
import { GraphRAGPipelineTrace } from "@/components/graphrag/GraphRAGPipelineTrace";
import { getRoutePath } from "@/config/navigation";
import { apiService as api } from "@/services/api";
import {
  buildGraphRAGBatchPrompt,
  buildGraphRAGBatchQuestions,
  type GraphRAGBatchQuestion,
} from "@/utils/graphragBatchPrompts";
import { cleanDocumentTextForReading } from "@/utils/chatReferenceText";
import { buildGraphRAGDocumentQuestions } from "@/utils/graphragDocumentPrompts";
import {
  buildGraphRAGTraceContextQuery,
  buildGraphRAGTraceDisplayPrompt,
  buildGraphRAGTracePrompt,
  type GraphRAGTraceChatContextRequest,
} from "@/utils/graphragPipelinePrompts";
import {
  buildGraphRAGRichEvidenceOutputSummary,
  groupGraphRAGDocumentPipelineRuns,
  graphRAGPipelineRunsShareDocument,
  preferredGraphRAGPipelineRunForDocumentChat,
  selectGraphRAGWorkbenchPipelineResult,
  summarizeGraphRAGDocumentLibrary,
  summarizeGraphRAGDocumentPipelineRun,
} from "@/utils/graphragDocumentRunSummary";
import { normalizeGraphRAGServiceStatus } from "@/utils/graphragServiceStatus";
import {
  cleanGraphRAGTraceSampleText,
  graphRAGTraceSampleEvidenceSummary,
  graphRAGTraceSampleGroupDisplayLabel,
  graphRAGTraceSampleLinkTarget,
  isLowValueGraphRAGTraceSampleText,
  pickBestGraphRAGTraceSample,
  rankGraphRAGTraceSampleGroups,
  type GraphRAGTraceSampleRole,
} from "@/utils/graphragTraceSamples";
import { displayPathNodes } from "@/utils/graphragPathDisplay";
import {
  cleanGraphReferenceEndpointLabel,
  cleanGraphReferenceEvidenceText,
  cleanGraphReferenceLabel,
  formatGraphReferenceRelation,
} from "@/utils/graphragReferenceDisplay";
import {
  buildGraphRAGUploadManifest,
  formatGraphRAGUploadFileKind,
  formatGraphRAGUploadFileSize,
} from "@/utils/graphragUploadManifest";
import {
  buildGraphRAGPipelineArtifactManifest,
  normalizeGraphRAGPipelineArtifactDownloads,
  type GraphRAGPipelineArtifactDownload,
} from "@/utils/graphragPipelineArtifacts";
import {
  buildGraphRAGWorkbenchPanelState,
  graphRAGDocumentWorkspaceActiveStepTarget,
  graphRAGDocumentWorkspaceMobileGridAreas,
  graphRAGDocumentWorkspaceStepTargets,
  type GraphRAGDocumentWorkspaceSection,
} from "@/utils/graphragWorkbenchPanelState";
import {
  buildGraphRAGPipelineEvidenceSamplePrompt,
  buildSelectedGraphRAGContextOverride,
  buildSelectedGraphRAGPrompt,
} from "@/utils/graphragWorkbenchChat";
import { buildGraphRAGDocumentReadiness } from "@/utils/graphragDocumentReadiness";
import { accentColors } from "@/utils/theme";

type SearchType = "hybrid" | "vector" | "graph" | "keyword";
type TestSuite =
  "unit" | "integration" | "performance" | "quality" | "comprehensive";
type FeedbackRating = -1 | 0 | 1;
type SocialPlatform = "twitch" | "youtube" | "kick" | "rumble";
type SocialMemoryEvalSuite = "all" | "locomo" | "vtuber";
type UpstreamMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
type DocumentInsightView = "overview" | "evidence" | "processing";

const STATUS_CONFIG: StatusChipConfigMap = {
  available: { color: accentColors.success, label: "Available" },
  blocked: { color: accentColors.rose, label: "Blocked" },
  disabled: { color: accentColors.slate, label: "Disabled" },
  error: { color: accentColors.rose, label: "Error" },
  evaluation: { color: accentColors.purple, label: "Evaluation" },
  extraction: { color: accentColors.warning, label: "Extraction" },
  graph: { color: accentColors.primary, label: "Graph" },
  memory: { color: accentColors.info, label: "Memory" },
  local: { color: accentColors.info, label: "Local" },
  online: { color: accentColors.success, label: "Online" },
  offline: { color: accentColors.rose, label: "Offline" },
  partial: { color: accentColors.warning, label: "Partial" },
  ready: { color: accentColors.success, label: "Ready" },
  reason: { color: accentColors.rose, label: "Reasoning" },
  running: { color: accentColors.info, label: "Running" },
  healthy: { color: accentColors.success, label: "Healthy" },
  not_configured: { color: accentColors.slate, label: "Not Configured" },
  unavailable: { color: accentColors.rose, label: "Unavailable" },
  unhealthy: { color: accentColors.rose, label: "Unhealthy" },
};

const SEARCH_TYPES: Array<{ value: SearchType; label: string }> = [
  { value: "hybrid", label: "Hybrid" },
  { value: "graph", label: "Graph" },
  { value: "vector", label: "Vector" },
  { value: "keyword", label: "Keyword" },
];

const TEST_SUITES: Array<{ value: TestSuite; label: string }> = [
  { value: "quality", label: "Quality" },
  { value: "integration", label: "Integration" },
  { value: "unit", label: "Unit" },
  { value: "performance", label: "Performance" },
  { value: "comprehensive", label: "Comprehensive" },
];

const PIPELINE_SUCCESS_STATUSES = new Set([
  "complete",
  "completed",
  "ready",
  "success",
]);
const PIPELINE_RUNNING_STATUSES = new Set([
  "running",
  "processing",
  "pending",
  "submitted",
]);
const PIPELINE_WARNING_STATUSES = new Set([
  "warning",
  "partial",
  "skipped",
  "preview_only",
  "not_configured",
]);
const PIPELINE_FAILURE_STATUSES = new Set([
  "failed",
  "error",
  "unavailable",
  "unhealthy",
  "misconfigured",
]);

const pipelineRunIsTerminal = (run: Record<string, unknown>): boolean => {
  const status = asString(run.status).toLowerCase();
  const resumeHandoff = asRecord(run.resume_handoff);
  const interactionHandoff = asRecord(run.interaction_handoff);
  const summary = asRecord(run.summary);
  const localContext = asRecord(summary.local_context);
  const localGraph = asRecord(summary.local_graph);
  const chatReady =
    resumeHandoff.ready === true ||
    interactionHandoff.ready === true ||
    asNumber(summary.interaction_question_count) > 0;
  const localEvidenceReady =
    asNumber(summary.chunk_count) > 0 &&
    (
      asNumber(summary.relationship_candidate_count) > 0 ||
      asNumber(localGraph.relationship_count) > 0 ||
      asNumber(localGraph.edge_count) > 0
    ) &&
    (
      asNumber(localContext.chunk_count) > 0 ||
      asNumber(localContext.cached_chunk_count) > 0
    );
  return (
    chatReady ||
    localEvidenceReady ||
    PIPELINE_SUCCESS_STATUSES.has(status) ||
    PIPELINE_WARNING_STATUSES.has(status) ||
    PIPELINE_FAILURE_STATUSES.has(status)
  );
};

const preferredDocumentInsightViewForRuns = (
  runs: Record<string, unknown>[],
): DocumentInsightView =>
  runs.length > 0 && runs.every(pipelineRunIsTerminal)
    ? "overview"
    : "processing";

const BATCH_UPLOAD_RECOVERY_TIMEOUT_MS = 45000;
const BATCH_UPLOAD_RECOVERY_POLL_MS = 2500;
const BATCH_UPLOAD_RECOVERY_CLOCK_SKEW_MS = 1000;

const FEEDBACK_RATINGS: Array<{ value: FeedbackRating; label: string }> = [
  { value: 1, label: "Positive" },
  { value: 0, label: "Neutral" },
  { value: -1, label: "Negative" },
];

const SOCIAL_PLATFORMS: Array<{ value: SocialPlatform; label: string }> = [
  { value: "twitch", label: "Twitch" },
  { value: "youtube", label: "YouTube" },
  { value: "kick", label: "Kick" },
  { value: "rumble", label: "Rumble" },
];

const SOCIAL_MEMORY_EVAL_SUITES: Array<{
  value: SocialMemoryEvalSuite;
  label: string;
}> = [
  { value: "all", label: "All" },
  { value: "locomo", label: "LoCoMo" },
  { value: "vtuber", label: "vTuber" },
];

const UPSTREAM_METHODS: UpstreamMethod[] = [
  "GET",
  "POST",
  "PUT",
  "PATCH",
  "DELETE",
];
const GRAPHRAG_DOCUMENT_ACCEPT =
  ".txt,.md,.pdf,.json,.csv,.html,.xml,.eml,.doc,.docx,.docm,.dotx,.dotm,.ppt,.pptx,.pptm,.potx,.potm,.ppsx,.ppsm,.xls,.xlsx,.xlsm,.xltx,.xltm,.rtf,.odt,.ods,.odp,.epub,.zip,.png,.jpg,.jpeg,.gif,.bmp,.tif,.tiff,.webp,.svg,message/rfc822,image/*";
const MAX_CHAT_CONTEXT_PIPELINE_RUNS = 16;

const fieldGridSx = {
  display: "grid",
  gridTemplateColumns: { xs: "1fr", md: "repeat(3, minmax(0, 1fr))" },
  gap: 1,
  alignItems: "start",
} as const;

const actionButtonSx = (color: string) => ({
  borderRadius: "8px",
  bgcolor: alpha(color, 0.88),
  color: "primary.contrastText",
  "&:hover": { bgcolor: color },
  "&.Mui-disabled": {
    color: alpha("#fff", 0.78),
    bgcolor: alpha(color, 0.18),
  },
});

const asRecord = (value: unknown): Record<string, unknown> => {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }

  return {};
};

const asArray = (value: unknown): unknown[] =>
  Array.isArray(value) ? value : [];

const asString = (value: unknown, fallback = ""): string =>
  typeof value === "string" && value.trim() ? value : fallback;

const asNumber = (value: unknown, fallback = 0): number => {
  const parsed =
    typeof value === "number"
      ? value
      : typeof value === "string"
        ? Number(value)
        : Number.NaN;
  return Number.isFinite(parsed) ? parsed : fallback;
};

const asStringArray = (value: unknown): string[] =>
  asArray(value)
    .map((item) => asString(item))
    .filter(Boolean);

const delay = (ms: number): Promise<void> =>
  new Promise((resolve) => globalThis.setTimeout(resolve, ms));

const normalizeDocumentFilename = (value: string): string =>
  value.trim().toLowerCase();

const pipelineRunDocumentFilename = (run: Record<string, unknown>): string =>
  asString(asRecord(run.document).filename);

const pipelineRunTimestampMs = (run: Record<string, unknown>): number => {
  const value = asString(run.updated_at) || asString(run.created_at);
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
};

const pipelineRunStatus = (run: Record<string, unknown>): string =>
  asString(run.status).toLowerCase();

const completedRecentPipelineRunsForFiles = (
  runs: Record<string, unknown>[],
  files: File[],
  uploadStartedAtMs: number,
): Record<string, unknown>[] => {
  const wantedNames = Array.from(
    new Set(
      files
        .map((file) => normalizeDocumentFilename(file.name))
        .filter(Boolean),
    ),
  );
  if (wantedNames.length === 0) return [];

  const cutoffMs = uploadStartedAtMs - BATCH_UPLOAD_RECOVERY_CLOCK_SKEW_MS;
  const candidates = runs
    .filter((run) => {
      const runId = asString(run.run_id);
      const filename = normalizeDocumentFilename(pipelineRunDocumentFilename(run));
      const status = pipelineRunStatus(run);
      const timestampMs = pipelineRunTimestampMs(run);
      return (
        Boolean(runId) &&
        Boolean(filename) &&
        wantedNames.includes(filename) &&
        PIPELINE_SUCCESS_STATUSES.has(status) &&
        !PIPELINE_FAILURE_STATUSES.has(status) &&
        timestampMs >= cutoffMs
      );
    })
    .sort((left, right) => pipelineRunTimestampMs(right) - pipelineRunTimestampMs(left));

  const matched = wantedNames
    .map((filename) =>
      candidates.find(
        (run) =>
          normalizeDocumentFilename(pipelineRunDocumentFilename(run)) === filename,
      ),
    )
    .filter((run): run is Record<string, unknown> => Boolean(run));

  return matched.length === wantedNames.length ? matched : [];
};

const buildRecoveredBatchUploadResult = ({
  files,
  runs,
  domain,
  uploadStartedAtMs,
}: {
  files: File[];
  runs: Record<string, unknown>[];
  domain: string;
  uploadStartedAtMs: number;
}): Record<string, unknown> => {
  const runIds = runs.map((run) => asString(run.run_id)).filter(Boolean);
  const filenames = files.map((file) => file.name).filter(Boolean);
  const uploadManifestBase = buildGraphRAGUploadManifest(
    files.map((file) => ({ name: file.name, size: file.size })),
    { buildKnowledgeGraph: true, useSemanticChunking: true },
  );
  const fileRows = files.map((file) => {
    const run = runs.find(
      (candidate) =>
        normalizeDocumentFilename(pipelineRunDocumentFilename(candidate)) ===
        normalizeDocumentFilename(file.name),
    );

    return {
      filename: file.name,
      name: file.name,
      kind: formatGraphRAGUploadFileKind(file.name),
      size_bytes: file.size,
      status: run ? asString(run.status, "complete") : "missing",
      run_id: run ? asString(run.run_id) : "",
    };
  });
  const chatContextRequest = {
    query: `Use these recovered GraphRAG pipeline traces to answer across ${filenames.join(", ")}. Cite document, page, slide, table, chart, and graph-path evidence when available.`,
    domain,
    pipeline_run_ids: runIds,
    include_graph: true,
    use_local_pipeline_context: true,
  };
  const batchResumeHandoff = {
    ready: runIds.length > 0,
    run_ids: runIds,
    filenames,
    recommended_mode: "graph_path_reasoning",
    chat_context_request: chatContextRequest,
    route_state: {
      graphragHandoff: {
        runIds,
        filenames,
        domain,
        chatContextRequest,
        recommendedMode: "graph_path_reasoning",
      },
    },
  };
  const uploadManifest = {
    ...uploadManifestBase,
    file_count: files.length,
    total_bytes: uploadManifestBase.totalBytes,
    accepted: runs.length,
    failed: Math.max(0, files.length - runs.length),
    domains: [domain],
    files: fileRows,
    next_step: "Open chat to ask over these documents with graph context.",
    batch_resume_handoff: batchResumeHandoff,
  };

  return {
    status: "complete",
    recovered_from_recent_runs: true,
    recovery_reason: "upload_response_dropped",
    total_files: files.length,
    processed: runs.length,
    failed: Math.max(0, files.length - runs.length),
    duration_ms: Math.max(0, Date.now() - uploadStartedAtMs),
    runs,
    results: fileRows,
    upload_manifest: uploadManifest,
    batch_resume_handoff: batchResumeHandoff,
    summary: {
      upload_manifest: uploadManifest,
      batch_resume_handoff: batchResumeHandoff,
      operational_handoff: {
        status: "recovered_from_recent_runs",
        label: "Documents processed",
        next_step: "Open chat to ask over these documents with graph context.",
      },
    },
  };
};

const uniqueStringList = (values: string[]): string[] => {
  const seen = new Set<string>();
  return values
    .map((value) => value.trim())
    .filter((value) => {
      if (!value || seen.has(value)) return false;
      seen.add(value);
      return true;
    });
};

const getErrorMessage = (error: unknown) => {
  if (error instanceof Error) return error.message;

  const errorRecord = asRecord(error);
  const response = asRecord(errorRecord.response);
  const data = asRecord(response.data);
  const detail = data.detail ?? data.message ?? errorRecord.message;

  return typeof detail === "string" ? detail : "GraphRAG request failed";
};

const formatNumber = (value: unknown) => {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toLocaleString();
  }

  if (typeof value === "string" && value.trim()) {
    return value;
  }

  return "0";
};

const positiveCount = (value: unknown): number | null => {
  const parsed =
    typeof value === "number"
      ? value
      : typeof value === "string"
        ? Number(value)
        : 0;
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
};

const formatFileSize = formatGraphRAGUploadFileSize;
const formatFileKind = formatGraphRAGUploadFileKind;

const graphEdgeEvidenceFamily = (edge: Record<string, unknown>): string => {
  const metadata = asRecord(edge.metadata);
  const properties = asRecord(edge.properties);
  const relation = asString(
    edge.label,
    asString(edge.relation, asString(edge.type, "related_to")),
  ).toLowerCase();
  const searchable = [
    relation,
    edge.evidence,
    ...Object.entries(metadata).flatMap(([key, value]) => [key, value]),
    ...Object.entries(properties).flatMap(([key, value]) => [key, value]),
  ]
    .map((value) => String(value || "").toLowerCase())
    .join(" ");

  if (searchable.includes("chart")) return "chart";
  if (searchable.includes("table")) return "table";
  if (searchable.includes("form_field") || searchable.includes("form field")) {
    return "form";
  }
  if (searchable.includes("styled_text") || searchable.includes("styled text") || searchable.includes("style_summary")) {
    return "styled text";
  }
  if (searchable.includes("diagram") || searchable.includes("smartart")) {
    return "diagram";
  }
  if (searchable.includes("connector") || searchable.includes("arrow")) {
    return "connector";
  }
  if (searchable.includes("link") || searchable.includes("uri")) return "link";
  if (searchable.includes("code")) return "code";
  if (searchable.includes("image") || searchable.includes("ocr")) {
    return "image";
  }
  if (searchable.includes("speaker") || searchable.includes("notes")) {
    return "notes";
  }
  if (!relation.startsWith("contains") && !relation.startsWith("has_")) {
    return "semantic";
  }
  return "structural";
};

const graphEvidenceFamilyPriority = (family: string): number => {
  const normalized = family.toLowerCase();
  if (normalized === "semantic") return 120;
  if (normalized === "connector") return 115;
  if (normalized === "form") return 112;
  if (normalized === "styled text") return 108;
  if (normalized === "diagram") return 96;
  if (normalized === "link") return 54;
  if (normalized === "chart") return 88;
  if (normalized === "table") return 82;
  if (normalized === "image") return 72;
  if (normalized === "notes") return 70;
  if (normalized === "code") return 68;
  if (normalized === "structural") return 20;
  return 50;
};

const graphEvidenceFamilyPreviewPriority = (family: string): number => {
  const normalized = family.replace(/\s+edges$/i, "").toLowerCase();
  if (normalized === "chart") return 140;
  if (normalized === "table") return 136;
  if (normalized === "connector") return 132;
  if (normalized === "form") return 128;
  if (normalized === "styled text") return 124;
  if (normalized === "diagram") return 116;
  if (normalized === "image") return 108;
  if (normalized === "notes") return 104;
  if (normalized === "semantic") return 60;
  if (normalized === "structural") return 20;
  return 80;
};

const formatDuration = (durationMs: number): string => {
  if (durationMs < 1000) return `${durationMs}ms`;
  if (durationMs < 60_000) return `${(durationMs / 1000).toFixed(1)}s`;
  const minutes = Math.floor(durationMs / 60_000);
  const seconds = Math.round((durationMs % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
};

const formatThroughput = (value: unknown): string => {
  const parsed = asNumber(value);
  if (parsed <= 0) return "0 files/s";
  return `${parsed >= 10 ? parsed.toFixed(1) : parsed.toFixed(2)} files/s`;
};

const FRIENDLY_GRAPHRAG_LABELS: Record<string, string> = {
  capability_cards: "Chat capabilities",
  chunk_evidence_cards: "Evidence cards",
  compatible_health_urls: "Health routes",
  compatible_ingest_urls: "Ingest routes",
  detected_capabilities: "Detected abilities",
  embedding_evidence_cards: "Embedding evidence",
  embedding_evidence_family_counts: "Embedding evidence mix",
  embedding_inputs: "Embedding text",
  embedding_sample_categories: "Embedding categories",
  graph_interaction_plan: "Chat context",
  graph_candidates: "Relationships found",
  graph_index: "Knowledge graph",
  graph_paths: "Graph paths",
  graph_rehearsals: "Search rehearsals",
  chunking: "Reference chunks",
  document_profile: "Document profile",
  embedding_plan: "Search embeddings",
  local_context_cache: "Retrieval cache",
  local_graph_artifact: "Knowledge graph",
  local_graph_reasoning_index: "Reasoning paths",
  received: "Received",
  relationship_evidence_cards: "Relationship evidence",
  relationship_sample_categories: "Relationship categories",
  representative_embedding_inputs: "Embedding examples",
  retrieval_rehearsals: "Search rehearsals",
  structure_extraction: "Extracted text",
  structured_ingest_document: "Structured ingest",
  upstream_ingest: "GraphRAG ingest",
  upstream_preflight: "GraphRAG connection",
};

const friendlyLabelKey = (value: string): string =>
  value.trim().toLowerCase().replace(/[\s-]+/g, "_");

const formatLabel = (value: string) => {
  const friendly = FRIENDLY_GRAPHRAG_LABELS[friendlyLabelKey(value)];
  if (friendly) return friendly;

  return value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
};

const formatGraphRelationLabel = (value: string) => {
  const normalized = value.trim().toLowerCase();
  if (
    [
      "co_occurs",
      "co_occurs_with",
      "co-occurs-with",
      "co occurs with",
      "related_text",
      "related text",
    ].includes(normalized)
  ) {
    return "Mentions";
  }
  if (normalized === "has_section" || normalized === "contains_section") {
    return "Section";
  }
  if (normalized.startsWith("has_column")) {
    return "Column";
  }
  if (["contains_metric", "contains_chart_series"].includes(normalized)) {
    return "Metric";
  }
  if (["contains_chart_point", "has_value"].includes(normalized)) {
    return "Value";
  }
  if (["contains_table", "has_column"].includes(normalized)) {
    return "Table";
  }
  return formatLabel(value);
};

const formatGraphRAGArtifactContentLabel = (value: string): string => {
  const normalized = value.toLowerCase();
  if (normalized.includes("presentationml") || normalized.includes("powerpoint")) {
    return "PowerPoint";
  }
  if (normalized.includes("spreadsheetml") || normalized.includes("excel")) {
    return "Excel";
  }
  if (normalized.includes("wordprocessingml") || normalized.includes("word")) {
    return "Word";
  }
  if (normalized.includes("pdf")) return "PDF";
  if (normalized.includes("markdown")) return "Markdown";
  if (normalized.includes("json")) return "JSON";
  if (normalized.startsWith("image/")) return "Image";
  if (normalized.startsWith("text/")) return "Text";

  return formatLabel(normalized.split(";")[0].split("/").pop() || value);
};

const graphRAGArtifactActionLabel = (
  artifact: GraphRAGPipelineArtifactDownload,
): string => {
  if (artifact.kind === "source_document") return "Original";
  if (artifact.kind === "structured_ingest") return "Text";
  if (artifact.kind === "local_graph") return "Graph";
  if (artifact.kind === "local_context") return "Retrieval";
  if (artifact.kind === "local_graph_reasoning") return "Paths JSON";
  if (artifact.kind === "interaction_handoff") return "Chat handoff";
  return artifact.shortLabel || artifact.label;
};

const graphRAGArtifactColor = (kind: string): string => {
  if (kind === "source_document") return accentColors.primary;
  if (kind === "structured_ingest") return accentColors.info;
  if (kind === "local_graph") return accentColors.success;
  if (kind === "local_context") return accentColors.purple;
  return accentColors.slate;
};

const openGraphRAGPipelineArtifact = (endpoint: string): void => {
  if (!endpoint || typeof window === "undefined") return;
  window.open(endpoint, "_blank", "noopener,noreferrer");
};

const cleanPipelineGraphLabel = (value: string, fallback = "Source"): string => {
  const text = value.replace(/\s+/g, " ").trim();
  if (!text) return fallback;
  if (isGraphDocumentName(text) || isInternalGraphArtifactEndpoint(text)) {
    return "source document";
  }
  return cleanGraphReferenceEndpointLabel(text, fallback);
};

const formatPipelineRelationTitleLabel = (value: string): string =>
  formatLabel(formatGraphReferenceRelation(value));

const formatPipelineGraphChainLabel = (
  values: string[],
  fallback = "Evidence",
): string => {
  const labels = values
    .map((value, index) =>
      cleanPipelineGraphLabel(
        value,
        index === 0 ? fallback : index === values.length - 1 ? "Target" : "Bridge",
      ),
    )
    .filter(Boolean);
  return labels.join(" to ");
};

const formatPipelineGraphPairLabel = (
  source: string,
  target: string,
): string => formatPipelineGraphChainLabel([source, target], "Source");

const humanizePipelineVisibleGraphText = (value: string): string =>
  value
    .replace(/\s+--?\[[^\]]+\]--?>\s+/g, " to ")
    .replace(/\s*->\s*/g, " to ")
    .replace(/\s*→\s*/g, " to ")
    .replace(/(\S)[.,;:!?]+\s+to\s+/g, "$1 to ")
    .replace(/\s+/g, " ")
    .trim();

const normalizePipelineVisibleGraphText = (value: string): string =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const formatPipelineRelationshipStatement = (
  source: string,
  relation: string,
  target: string,
): string => {
  if (!source.trim() || !target.trim()) return "";
  const sourceLabel = cleanPipelineGraphLabel(source, "Source");
  const targetLabel = cleanPipelineGraphLabel(target, "Target");
  const relationLabel = formatGraphReferenceRelation(relation);
  if (!sourceLabel || !targetLabel) return "";
  return `${sourceLabel} ${relationLabel} ${targetLabel}`;
};

const cleanPipelineSampleBodyText = (value: string): string =>
  humanizePipelineVisibleGraphText(cleanGraphRAGTraceSampleText(
    value
      .replace(
        /\b[^\s,;:|()]+?\.(?:pptx?|pdf|docx?|xlsx?|csv|md|markdown|txt)\b/gi,
        "source document",
      )
      .replace(/\bsource document\s+source document\b/gi, "source document")
      .replace(/\s+/g, " ")
      .trim(),
    180,
  ));

const formatGraphEdgeTitle = (value: string) => {
  const match = value.match(/^(.*?)\s+--\[(.*?)\]-->\s+(.*)$/);
  if (!match) {
    return value.replace(
      /([^:\n]{2,48}):\s*([^>\n]+?)\s*->\s*([^\n]+)$/i,
      (_match, relation: string, source: string, target: string) =>
        `${formatPipelineRelationTitleLabel(relation)}: ${formatPipelineGraphPairLabel(source, target)}`,
    );
  }

  const [, source, relation, target] = match;
  return `${formatPipelineRelationTitleLabel(relation)}: ${formatPipelineGraphPairLabel(source, target)}`;
};

const compactGraphRAGActionText = (value: string, limit = 52): string => {
  const text = value
    .replace(/\s*:\s*[a-z][\w -]{1,48}$/i, "")
    .replace(/\s+/g, " ")
    .trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(1, limit - 3)).trim()}...`;
};

const isGraphDocumentName = (value: string): boolean =>
  /\.(?:csv|docx?|html?|md|pdf|pptx?|txt|xlsx?)$/i.test(value.trim());

const isExternalGraphEndpoint = (value: string): boolean =>
  /\b(?:https?:\/\/|www\.)\S+/i.test(value.trim());

const isGraphLinkRelation = (value: string): boolean =>
  /^(?:links?_to|contains?_link|has_link|references?_url)$/i.test(
    value.trim().replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, ""),
  );

const isInternalGraphArtifactEndpoint = (value: string): boolean =>
  /(?:^|[/"'\\\s])(?:ppt|word|xl|docProps|_rels)\/|(?:chartWorkbook|oleObject|embeddings\/|media\/image|\.rels\b|\.xml\b|\.bin\b)/i.test(value.trim());

const graphRAGQuestionLabel = (
  question: Pick<GraphRAGBatchQuestion, "kind" | "label" | "source" | "target" | "bridge" | "relation">,
): string => {
  const source = question.source?.trim() || "";
  const target = question.target?.trim() || "";
  const bridge = question.bridge?.trim() || "";
  const relation = question.relation?.trim() || "";
  const label = question.label?.trim() || "Ask about evidence";
  const linkSubject = [target, source, label]
    .map((value) => value.replace(/\s+--?\[.*?\]--?>\s+/g, " -> ").trim())
    .find(
      (value) =>
        value &&
        !value.includes("->") &&
        !isExternalGraphEndpoint(value) &&
        !isGraphDocumentName(value),
    );

  if (question.kind === "path" && source && target) {
    return bridge
      ? `Trace ${compactGraphRAGActionText(source, 24)} via ${compactGraphRAGActionText(bridge, 24)}`
      : `Trace ${compactGraphRAGActionText(source, 24)} to ${compactGraphRAGActionText(target, 24)}`;
  }

  if (question.kind === "relationship" && (source || target)) {
    if (isGraphLinkRelation(relation)) {
      return linkSubject
        ? `Trace ${compactGraphRAGActionText(linkSubject, 30)} link`
        : "Trace evidence link";
    }
    if (target.includes("->")) {
      return `Trace ${compactGraphRAGActionText(target.replace(/\s*->\s*/g, " to "))}`;
    }
    if (source && target) {
      if (relation) {
        return `Trace ${formatGraphRelationLabel(relation)}: ${compactGraphRAGActionText(target, 30)}`;
      }
      return isGraphDocumentName(source)
        ? `Trace ${compactGraphRAGActionText(target)}`
        : `Trace ${compactGraphRAGActionText(source, 24)} to ${compactGraphRAGActionText(target, 24)}`;
    }
    return `Trace ${compactGraphRAGActionText(source || target)}`;
  }

  return compactGraphRAGActionText(label);
};

const isNumericGraphEndpoint = (value: string): boolean => {
  const text = value.trim();
  if (!text) return false;
  return /^[$€£]?\s*-?\d+(?:[.,]\d+)*(?:\s*%|\s*(?:k|m|b|mm|bn))?$/i.test(text);
};

const graphNodeLooksSemantic = (value: string): boolean => {
  const text = value.trim();
  if (!text || isNumericGraphEndpoint(text) || isGraphDocumentName(text)) return false;
  return /[a-z]/i.test(text) && text.replace(/[^a-z0-9]+/gi, "").length >= 4;
};

const isLowValueGraphEndpointLabel = (value: string): boolean => {
  const text = value
    .toLowerCase()
    .replace(/\bhf[_\s-]?token\b/g, "hf token")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
  return [
    "hf hub",
    "hf token",
    "hub",
    "please",
    "token",
    "warning",
    "you",
  ].includes(text);
};

const graphRelationUsefulnessScore = ({
  relation,
  source,
  target,
  family,
  structural,
  weakMention,
}: {
  relation: string;
  source: string;
  target: string;
  family: string;
  structural: boolean;
  weakMention: boolean;
}): number => {
  const normalizedRelation = relation.toLowerCase();
  let score = 0;

  score += graphEvidenceFamilyPriority(family) / 10;
  if (weakMention) score -= 12;
  if (structural) score -= 6;
  if (graphNodeLooksSemantic(source)) score += 3;
  if (graphNodeLooksSemantic(target)) score += 3;
  if (isNumericGraphEndpoint(source) || isNumericGraphEndpoint(target)) score -= 18;
  if (isGraphDocumentName(source) || isGraphDocumentName(target)) score -= 4;
  if (isExternalGraphEndpoint(source) || isExternalGraphEndpoint(target)) score -= 24;
  if (isInternalGraphArtifactEndpoint(source) || isInternalGraphArtifactEndpoint(target)) score -= 32;
  if (/^contains_table/.test(normalizedRelation)) score -= 10;
  if (/^has_column_\d+/.test(normalizedRelation)) score -= 12;
  if (/[|]/.test(source) || /[|]/.test(target)) score -= 8;

  if (/(drive|depend|support|block|risk|reference|link|own|unblock|calculate|influence|cause|map|route|connect)/i.test(normalizedRelation)) {
    score += 14;
  }
  if (/(q\d|value|point|cell|created|modified|keyword|subject|title|slide_count|id)$/i.test(normalizedRelation)) {
    score -= 10;
  }

  return score;
};

const formatCompactEndpoint = (value: string) => {
  if (!value) return "";

  try {
    const url = new URL(value);
    return `${url.host}${url.pathname}${url.search}`;
  } catch {
    return value.replace(/^https?:\/\//, "");
  }
};

type PipelineEvidenceSampleRow = {
  key: string;
  label: string;
  title: string;
  body: string;
  linkTarget?: { label: string; url: string } | null;
  countLabel: string;
  color: string;
  priority: number;
};

type PipelineSampleOutputGroup = {
  key: string;
  label: string;
  count: number;
  role: GraphRAGTraceSampleRole;
};

type PipelineStageOutputRole = "evidence" | "support";

type PipelineStageOutputRow = {
  key: string;
  label: string;
  stageName: string;
  outputRole: PipelineStageOutputRole;
  status: string;
  statusColor: string;
  durationMs: number | null;
  durationLabel: string;
  sampleGroups: string[];
  sampleCount: number;
  title: string;
  body: string;
  linkTarget?: { label: string; url: string } | null;
  warningCount: number;
  warningText: string;
};

const PipelineSampleLinkTargetRow = ({
  target,
  color,
  compact = false,
}: {
  target?: { label: string; url: string } | null;
  color: string;
  compact?: boolean;
}) => {
  if (!target) return null;

  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: { xs: "1fr", sm: "auto minmax(0, 1fr)" },
        alignItems: "baseline",
        gap: { xs: 0.15, sm: 0.55 },
        minWidth: 0,
        mt: compact ? 0.35 : 0.55,
        px: compact ? 0.5 : 0.6,
        py: compact ? 0.35 : 0.45,
        borderRadius: "6px",
        bgcolor: alpha(color, 0.075),
      }}
    >
      <Typography
        variant="caption"
        noWrap
        sx={{
          color,
          fontSize: compact ? "0.58rem" : "0.62rem",
          fontWeight: 900,
          lineHeight: 1.15,
        }}
      >
        Link target
      </Typography>
      <Typography
        variant="caption"
        title={target.url}
        sx={{
          minWidth: 0,
          color: "text.primary",
          display: "-webkit-box",
          overflow: "hidden",
          WebkitBoxOrient: "vertical",
          WebkitLineClamp: compact ? { xs: 2, sm: 1 } : 2,
          overflowWrap: "anywhere",
          wordBreak: "break-word",
          fontSize: compact ? "0.6rem" : "0.64rem",
          fontWeight: 850,
          lineHeight: 1.18,
        }}
      >
        {target.label !== "External link" ? `${target.label}: ` : ""}
        {target.url}
      </Typography>
    </Box>
  );
};

type PipelineLaneProgressRow = {
  key: string;
  label: string;
  color: string;
  status: string;
  stageCount: number;
  completedCount: number;
  sampleCount: number;
  warningCount: number;
  durationMs: number;
  sampleTitle: string;
  sampleBody: string;
  sampleCountLabel: string;
  warningText: string;
};

type ActiveGraphRow = {
  key: string;
  label: string;
  text: string;
  title: string;
  color: string;
  family?: string;
  structural?: boolean;
  weakMention?: boolean;
  lowValue?: boolean;
  priority: number;
  kind?: "path" | "relationship";
  relation?: string;
  source?: string;
  bridge?: string;
  target?: string;
};

const dedupeActiveGraphRows = (rows: ActiveGraphRow[]): ActiveGraphRow[] => {
  const seen = new Set<string>();

  return rows.filter((row) => {
    const normalizedKey = [
      row.kind || "relationship",
      row.relation || "",
      row.source || "",
      row.bridge || "",
      row.target || "",
      row.text,
    ]
      .join("|")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();

    if (!normalizedKey || seen.has(normalizedKey)) return false;
    seen.add(normalizedKey);
    return true;
  });
};

type GraphRAGChatContextOverride = Partial<GraphRAGTraceChatContextRequest>;

type PipelineEvidenceSampleConfig = {
  id: string;
  label: string;
  stageNames: string[];
  sampleKeys: string[];
  countKeys: string[];
  unit: string;
  color: string;
  priority: number;
};

const PIPELINE_EVIDENCE_SAMPLE_CONFIGS: PipelineEvidenceSampleConfig[] = [
  {
    id: "content",
    label: "Readable text",
    stageNames: ["structure_extraction", "structured_ingest_document"],
    sampleKeys: ["elements", "text"],
    countKeys: ["elements", "element_count", "char_count"],
    unit: "items",
    color: accentColors.primary,
    priority: 1,
  },
  {
    id: "chunks",
    label: "Citable chunks",
    stageNames: ["chunking", "local_context_cache"],
    sampleKeys: ["chunk_evidence_cards", "chunks"],
    countKeys: ["chunks", "cached_chunks", "cached_chunk_count"],
    unit: "chunks",
    color: accentColors.info,
    priority: 2,
  },
  {
    id: "relationships",
    label: "Relationships",
    stageNames: ["graph_candidates", "local_graph_artifact"],
    sampleKeys: ["relationship_evidence_cards", "relationships", "edges"],
    countKeys: ["relationships", "edges", "relationship_count"],
    unit: "links",
    color: accentColors.success,
    priority: 3,
  },
  {
    id: "paths",
    label: "Reasoning paths",
    stageNames: ["local_graph_reasoning_index"],
    sampleKeys: ["path_candidates", "graph_paths", "bridge_entities"],
    countKeys: ["path_candidates", "graph_paths", "bridge_entities"],
    unit: "paths",
    color: accentColors.purple,
    priority: 4,
  },
  {
    id: "chat",
    label: "Chat context",
    stageNames: ["graph_interaction_plan"],
    sampleKeys: [
      "retrieval_rehearsals",
      "graph_anchors",
      "starter_prompts",
      "suggested_questions",
      "capability_cards",
    ],
    countKeys: [
      "retrieval_rehearsals",
      "graph_anchors",
      "starter_prompts",
      "suggested_questions",
    ],
    unit: "ready",
    color: accentColors.warning,
    priority: 5,
  },
];

const PIPELINE_EVIDENCE_EXAMPLE_PRIORITY: Record<string, number> = {
  "Citable chunks": 1,
  Relationships: 2,
  "Reasoning paths": 3,
  "Chat context": 4,
  "Readable text": 5,
};

const pipelineEvidenceExamplePriority = (row: PipelineEvidenceSampleRow): number =>
  PIPELINE_EVIDENCE_EXAMPLE_PRIORITY[row.label] ?? row.priority + 10;

const SUPPORT_PIPELINE_STAGE_NAMES = new Set([
  "received",
  "document_profile",
  "upstream_preflight",
  "upstream_ingest",
]);

const pipelineStageOutputRole = (
  stageName: string,
  sampleOutputs: Record<string, unknown>,
  groups: PipelineSampleOutputGroup[],
): PipelineStageOutputRole => {
  const normalizedStageName = stageName.toLowerCase();
  const matchesEvidenceConfig = PIPELINE_EVIDENCE_SAMPLE_CONFIGS.some(
    (config) =>
      config.stageNames.some((name) => normalizedStageName.includes(name)) ||
      config.sampleKeys.some((key) => sampleOutputs[key] !== undefined),
  );

  if (matchesEvidenceConfig) return "evidence";
  if (SUPPORT_PIPELINE_STAGE_NAMES.has(normalizedStageName)) return "support";
  return groups.some((group) => group.role === "evidence" || group.role === "interaction")
    ? "evidence"
    : "support";
};

const compactSampleText = (value: unknown, limit = 180): string => {
  const cleanPreview = (input: string): string =>
    cleanDocumentTextForReading(cleanGraphRAGTraceSampleText(input, Math.max(limit, 240)))
      .replace(/\b(?:ppt|xl|word)\/[^\s|;,)]+/gi, (match) =>
        cleanGraphReferenceLabel(match, "document package evidence"),
      )
      .replace(/\bembedded[_\s-]*package\b/gi, "embedded data")
      .replace(/\b(?:media|document|workbook|chart|table|sheet|slide)\s+part\s+(embedded data|chart evidence|table evidence|sheet evidence|slide evidence|media evidence)\b/gi, "$1")
      .replace(/(?:^|\s)\[[a-z][a-z0-9_ -]*(?:\s*\|[^\]]*)?\]\s*/gi, " ")
      .replace(/\s*;\s*[a-z][a-z0-9_]*=(?:"[^"]*"|\[[^\]]*\]|[^;\]]+)\]?/gi, " ")
      .replace(/\s+/g, " ")
      .trim();

  const normalize = (input: unknown): string => {
    if (input === null || input === undefined) return "";
    if (typeof input === "string") return cleanPreview(input);
    if (typeof input === "number" || typeof input === "boolean") {
      return String(input);
    }
    if (Array.isArray(input)) {
      return input
        .map((item) => normalize(item))
        .filter(Boolean)
        .slice(0, 4)
        .join(" ");
    }

    const record = asRecord(input);
    const preferredKeys = [
      "evidence_preview",
      "evidence",
      "reason",
      "detail",
      "text",
      "content",
      "provenance",
      "title",
      "label",
      "name",
      "answer_hint",
      "prompt",
    ];

    for (const key of preferredKeys) {
      const candidate = normalize(record[key]);
      if (candidate) return candidate;
    }

    try {
      return JSON.stringify(input);
    } catch {
      return "";
    }
  };

  const text = normalize(value);
  return text.length > limit ? `${text.slice(0, limit - 1).trim()}...` : text;
};

const pipelineStageStatusColor = (status: string): string => {
  const normalized = status.toLowerCase();
  if (PIPELINE_SUCCESS_STATUSES.has(normalized)) {
    return accentColors.success;
  }
  if (PIPELINE_RUNNING_STATUSES.has(normalized)) {
    return accentColors.info;
  }
  if (PIPELINE_WARNING_STATUSES.has(normalized)) {
    return accentColors.warning;
  }
  if (PIPELINE_FAILURE_STATUSES.has(normalized)) {
    return accentColors.rose;
  }
  return accentColors.slate;
};

const sampleOutputCount = (value: unknown): number => {
  if (Array.isArray(value)) return value.length;
  if (value && typeof value === "object") return Object.keys(asRecord(value)).length > 0 ? 1 : 0;
  return compactSampleText(value) ? 1 : 0;
};

const sampleOutputGroups = (
  sampleOutputs: Record<string, unknown>,
): PipelineSampleOutputGroup[] =>
  rankGraphRAGTraceSampleGroups(sampleOutputs)
    .map((group) => ({
      key: group.key,
      label: graphRAGTraceSampleGroupDisplayLabel(group.key, group.count),
      count: group.count || sampleOutputCount(sampleOutputs[group.key]),
      role: group.role,
    }));

const firstSampleRecord = (
  sampleOutputs: Record<string, unknown>,
  keys: string[],
): { key: string; record: Record<string, unknown>; index: number } | null => {
  const sample = pickBestGraphRAGTraceSample(sampleOutputs, keys);
  return sample
    ? { key: sample.key, record: sample.record, index: sample.index }
    : null;
};

const sampleCount = (
  counts: Record<string, unknown>,
  sampleOutputs: Record<string, unknown>,
  keys: string[],
): number => {
  for (const key of keys) {
    const count = positiveCount(counts[key]);
    if (count !== null) return count;
  }

  for (const key of keys) {
    const count = positiveCount(sampleOutputs[key]);
    if (count !== null) return count;
    if (Array.isArray(sampleOutputs[key]) && sampleOutputs[key].length > 0) {
      return sampleOutputs[key].length;
    }
  }

  return 0;
};

const pipelineSampleTitle = (
  key: string,
  record: Record<string, unknown>,
  index: number,
): string => {
  if (key === "elements") {
    const text = asString(record.text);
    if (text) return compactSampleText(text, 96);
    return [
      formatLabel(asString(record.type, "Element")),
      formatLabel(asString(record.location)),
    ]
      .filter(Boolean)
      .join(" · ");
  }

  if (key === "chunks" || key === "chunk_evidence_cards") {
    return asString(
      record.title,
      asString(
        record.scope_label,
        asString(record.chunk_id, `${formatLabel(key)} ${index + 1}`),
      ),
    );
  }

  if (
    key === "relationships" ||
    key === "relationship_evidence_cards" ||
    key === "edges"
  ) {
    const title = asString(record.title);
    const source = asString(record.source);
    const target = asString(record.target);
    const relation = asString(
      record.relation,
      asString(record.label, asString(record.type, "related")),
    );
    if (title) return formatGraphEdgeTitle(title);
    if (source || target) {
      return `${formatPipelineRelationTitleLabel(relation)}: ${formatPipelineGraphPairLabel(source, target)}`;
    }
  }

  if (key === "path_candidates" || key === "graph_paths") {
    const nodes = displayPathNodes(
      asArray(record.nodes).map((node) => compactSampleText(node, 48)),
    );
    if (nodes.length >= 2) return formatPipelineGraphChainLabel(nodes);
    return [
      asString(record.source, "Source"),
      asString(record.bridge),
      asString(record.target, "Target"),
    ]
      .filter(Boolean)
      .join(" to ");
  }

  if (key === "bridge_entities") {
    return `Bridge: ${asString(record.name, asString(record.label, `Entity ${index + 1}`))}`;
  }

  if (key === "retrieval_rehearsals") {
    if (asString(record.kind).toLowerCase() === "summary") {
      return "Document chat context";
    }
    return asString(record.label, asString(record.prompt_id, `Rehearsal ${index + 1}`));
  }

  if (key === "graph_anchors") {
    return asString(record.title, asString(record.label, `Anchor ${index + 1}`));
  }

  if (key === "starter_prompts" || key === "suggested_questions") {
    return asString(record.label, `Question ${index + 1}`);
  }

  if (key === "capability_cards") {
    return asString(record.label, `Capability ${index + 1}`);
  }

  return asString(record.title, asString(record.label, compactSampleText(record, 72)));
};

const pipelineSampleBody = (
  key: string,
  record: Record<string, unknown>,
): string => {
  if (key === "elements") {
    return compactSampleText(
      formatLabel(asString(record.location)) ||
        record.evidence_preview ||
        record.evidence ||
        record.metadata ||
        record,
    );
  }

  if (key === "chunks") {
    const metadata = asRecord(record.metadata);
    return compactSampleText(
      metadata.evidence_preview || record.evidence_preview || record.text,
    );
  }

  if (key === "chunk_evidence_cards") {
    return compactSampleText(record.evidence_preview || record.reason || record);
  }

  if (
    key === "relationships" ||
    key === "relationship_evidence_cards" ||
    key === "edges"
  ) {
    const rawSource = asString(record.source);
    const rawRelation = asString(record.relation, asString(record.label, asString(record.type)));
    const rawTarget = asString(record.target);
    const evidence = compactSampleText(record.evidence || record.reason || record.provenance || record);
    const statement = formatPipelineRelationshipStatement(rawSource, rawRelation, rawTarget);
    const visibleEvidence = humanizePipelineVisibleGraphText(cleanGraphReferenceEvidenceText(evidence, {
        source: rawSource,
        relation: rawRelation,
        target: rawTarget,
        filename: asString(record.filename),
    }));
    const evidenceKey = normalizePipelineVisibleGraphText(visibleEvidence);
    const statementKey = normalizePipelineVisibleGraphText(statement);
    const compactPairKey = normalizePipelineVisibleGraphText(
      `${formatPipelineGraphPairLabel(rawSource, rawTarget)} ${formatGraphReferenceRelation(rawRelation)}`,
    );
    if (statement && (!visibleEvidence || evidenceKey === compactPairKey || evidenceKey === statementKey)) {
      return compactSampleText(statement);
    }
    return compactSampleText(visibleEvidence || statement);
  }

  if (key === "path_candidates" || key === "graph_paths") {
    return cleanPipelineSampleBodyText(
      graphRAGTraceSampleEvidenceSummary(key, record) ||
      compactSampleText(record.explanation || record.evidence || record),
    );
  }

  if (key === "retrieval_rehearsals") {
    if (asString(record.kind).toLowerCase() === "summary") {
      const topChunk = asRecord(asArray(record.top_chunks)[0]);
      const preview = compactSampleText(
        topChunk.evidence_preview ||
          topChunk.text ||
          record.evidence_preview ||
          record.prompt,
        180,
      );
      return cleanPipelineSampleBodyText(preview || "Document evidence is ready for chat.");
    }
    return cleanPipelineSampleBodyText(
      graphRAGTraceSampleEvidenceSummary(key, record) ||
      compactSampleText(record.matched_terms || record.top_anchors || record.top_chunks),
    );
  }

  return cleanPipelineSampleBodyText(
    compactSampleText(record.evidence_preview || record.evidence || record.text || record),
  );
};

const pipelineStageDirectPreview = (
  stageName: string,
  sampleOutputs: Record<string, unknown>,
): { title: string; body: string } | null => {
  const normalizedStageName = stageName.toLowerCase();

  if (normalizedStageName === "received") {
    const filename = asString(sampleOutputs.filename, "Document");
    const extension = asString(sampleOutputs.extension);
    const sha = asString(sampleOutputs.sha256);
    const kind = filename !== "Document"
      ? formatFileKind(filename)
      : extension
        ? formatLabel(extension.replace(/^\./, ""))
        : "File";

    return {
      title: `Received ${filename}`,
      body: [
        kind,
        extension,
        sha ? "fingerprint recorded" : "",
      ].filter(Boolean).join(" · "),
    };
  }

  if (normalizedStageName === "document_profile") {
    const complexity = asString(sampleOutputs.complexity, "Document");
    const readinessScore = asNumber(sampleOutputs.readiness_score);
    const capabilities = asArray(sampleOutputs.detected_capabilities)
      .map((capability) => formatLabel(String(capability || "")))
      .filter(Boolean)
      .slice(0, 5);

    return {
      title: `${formatLabel(complexity)} document profile`,
      body: [
        readinessScore > 0 ? `${formatNumber(readinessScore)} readiness` : "",
        capabilities.length ? capabilities.join(", ") : "",
      ].filter(Boolean).join(" · "),
    };
  }

  if (normalizedStageName === "structured_ingest_document") {
    const filename = asString(sampleOutputs.filename, "Structured ingest");
    const usesStructuredText = sampleOutputs.uses_structured_text === true;
    const text = asString(sampleOutputs.text);

    return {
      title: usesStructuredText ? "Structured ingest ready" : "Document ingest ready",
      body: [
        filename,
        text ? `${formatNumber(text.length)} chars prepared` : "",
      ].filter(Boolean).join(" · "),
    };
  }

  if (normalizedStageName === "upstream_preflight") {
    const targetUrl = asString(sampleOutputs.target_url);
    const healthStatus = asString(sampleOutputs.health_response) ||
      asString(sampleOutputs.health_status_code);
    const nextStep = asString(sampleOutputs.next_step);

    return {
      title: targetUrl ? "GraphRAG endpoint checked" : "GraphRAG preflight checked",
      body: [
        targetUrl ? "remote endpoint configured" : "",
        healthStatus ? `health ${healthStatus}` : "",
        nextStep,
      ].filter(Boolean).join(" · "),
    };
  }

  if (normalizedStageName === "upstream_ingest") {
    const endpoint = asString(
      sampleOutputs.selected_endpoint,
      asString(sampleOutputs.target_url),
    );
    const payloadKind = asString(sampleOutputs.payload_kind);
    const response = asRecord(sampleOutputs.response);
    const responseStatus = asString(response.status, asString(response.message));

    return {
      title: endpoint ? "Submitted to GraphRAG" : "GraphRAG submission recorded",
      body: [
        endpoint ? "remote endpoint accepted" : "",
        payloadKind ? formatLabel(payloadKind) : "",
        responseStatus,
      ].filter(Boolean).join(" · "),
    };
  }

  return null;
};

const pipelineStageWarningText = (
  stageName: string,
  status: string,
  warnings: unknown[],
  error: string,
  sampleOutputs: Record<string, unknown>,
): string => {
  if (error) return compactSampleText(error, 180);

  const explicitWarning = warnings
    .map((warning) => compactSampleText(warning, 180))
    .find(Boolean);
  if (explicitWarning) return explicitWarning;

  const normalizedStatus = status.toLowerCase();
  if (!PIPELINE_WARNING_STATUSES.has(normalizedStatus)) {
    return "";
  }

  const normalizedStageName = stageName.toLowerCase();
  if (normalizedStageName.includes("local_graph_reasoning")) {
    const pathCount =
      sampleOutputCount(sampleOutputs.path_candidates) +
      sampleOutputCount(sampleOutputs.graph_paths) +
      sampleOutputCount(sampleOutputs.bridge_entities);
    if (pathCount === 0) {
      return "No multi-hop paths were found. Direct graph links and retrieval chunks are still available for chat.";
    }
    return "Path reasoning is partial. Review the graph links before relying on multi-hop answers.";
  }

  if (normalizedStageName.includes("upstream")) {
    return "Remote GraphRAG submission needs review. Local document chat remains available.";
  }

  return `${formatLabel(stageName)} returned ${formatLabel(status)}. Review the stage output before relying on this lane.`;
};

const buildPipelineEvidenceSampleRow = (
  stage: Record<string, unknown>,
  stageIndex: number,
): PipelineEvidenceSampleRow | null => {
  const stageName = asString(stage.name, `stage_${stageIndex + 1}`);
  const normalizedStageName = stageName.toLowerCase();
  const sampleOutputs = asRecord(stage.sample_outputs);
  if (Object.keys(sampleOutputs).length === 0) return null;

  const config = PIPELINE_EVIDENCE_SAMPLE_CONFIGS.find(
    (candidate) =>
      candidate.stageNames.some((name) => normalizedStageName.includes(name)) ||
      candidate.sampleKeys.some((key) => sampleOutputs[key] !== undefined),
  );
  if (!config) return null;

  const sample = firstSampleRecord(sampleOutputs, config.sampleKeys);
  if (!sample) return null;

  const counts = asRecord(stage.counts);
  const count = sampleCount(counts, sampleOutputs, config.countKeys);
  const title = humanizePipelineVisibleGraphText(compactSampleText(
    pipelineSampleTitle(sample.key, sample.record, sample.index),
    96,
  ));
  const body = humanizePipelineVisibleGraphText(pipelineSampleBody(sample.key, sample.record));
  const linkTarget = graphRAGTraceSampleLinkTarget(sample.record);
  if (!title && !body) return null;

  return {
    key: `${config.id}-${stageName}-${stageIndex}`,
    label: config.label,
    title: title || config.label,
    body,
    linkTarget,
    countLabel:
      count > 0
        ? `${formatNumber(count)} ${config.unit}`
        : formatLabel(asString(stage.status, "ready")),
    color: config.color,
    priority: config.priority + stageIndex / 100,
  };
};

const buildPipelineStageOutputRow = (
  stage: Record<string, unknown>,
  stageIndex: number,
): PipelineStageOutputRow => {
  const stageName = asString(stage.name, `stage_${stageIndex + 1}`);
  const status = asString(stage.status, "unknown");
  const sampleOutputs = asRecord(stage.sample_outputs);
  const groups = sampleOutputGroups(sampleOutputs);
  const visibleGroups = groups.filter(
    (group) => group.role === "evidence" || group.role === "interaction",
  );
  const sample = firstSampleRecord(
    sampleOutputs,
    groups.map((group) => group.key),
  );
  const outputRole = pipelineStageOutputRole(stageName, sampleOutputs, groups);
  const durationMs = asNumber(stage.duration_ms);
  const warnings = asArray(stage.warnings).filter(Boolean);
  const error = asString(stage.error);
  const message = asString(stage.message);
  const warningText = pipelineStageWarningText(
    stageName,
    status,
    warnings,
    error,
    sampleOutputs,
  );
  const directPreview = pipelineStageDirectPreview(stageName, sampleOutputs);
  const title = humanizePipelineVisibleGraphText(directPreview
    ? compactSampleText(directPreview.title, 92)
    : sample
    ? compactSampleText(
        pipelineSampleTitle(sample.key, sample.record, sample.index),
        92,
      )
    : compactSampleText(message || error || "No evidence preview recorded", 92));
  const body = humanizePipelineVisibleGraphText(directPreview
    ? compactSampleText(directPreview.body, 160)
    : sample
    ? pipelineSampleBody(sample.key, sample.record)
    : compactSampleText(error || message || warnings[0] || "", 160));
  const linkTarget = sample ? graphRAGTraceSampleLinkTarget(sample.record) : null;

  return {
    key: `${stageName}-${stageIndex}`,
    label: formatLabel(stageName),
    stageName,
    outputRole,
    status,
    statusColor: pipelineStageStatusColor(status),
    durationMs,
    durationLabel: durationMs !== null ? formatDuration(durationMs) : "instant",
    sampleGroups: visibleGroups.map((group) => group.label),
    sampleCount: visibleGroups.reduce((total, group) => total + group.count, 0),
    title,
    body,
    linkTarget,
    warningCount: warnings.length + (error ? 1 : 0),
    warningText,
  };
};

const pipelineLaneStatus = (rows: PipelineStageOutputRow[]): string => {
  if (rows.some((row) => ["failed", "error", "unavailable", "unhealthy", "misconfigured"].includes(row.status.toLowerCase()))) {
    return "error";
  }
  if (rows.some((row) => row.warningCount > 0 || PIPELINE_WARNING_STATUSES.has(row.status.toLowerCase()))) {
    return "warning";
  }
  if (rows.some((row) => PIPELINE_RUNNING_STATUSES.has(row.status.toLowerCase()))) {
    return "running";
  }
  if (rows.length > 0 && rows.every((row) => PIPELINE_SUCCESS_STATUSES.has(row.status.toLowerCase()))) {
    return "complete";
  }
  return rows.length > 0 ? "ready" : "waiting";
};

const buildPipelineLaneProgressRows = (
  stages: Record<string, unknown>[],
  stageRows: PipelineStageOutputRow[],
): PipelineLaneProgressRow[] =>
  PIPELINE_EVIDENCE_SAMPLE_CONFIGS.map((config) => {
    const matchedStageRows = stageRows.filter((row, index) => {
      const normalizedStageName = row.stageName.toLowerCase();
      const stage = stages[index] || {};
      const sampleOutputs = asRecord(stage.sample_outputs);
      return (
        config.stageNames.some((name) => normalizedStageName.includes(name)) ||
        config.sampleKeys.some((key) => sampleOutputs[key] !== undefined)
      );
    });
    const representative =
      matchedStageRows.find((row) => row.sampleCount > 0 && (row.title || row.body)) ||
      matchedStageRows.find((row) => row.title || row.body);
    const sampleRow = stages.reduce<PipelineEvidenceSampleRow | null>((found, stage, index) => {
      if (found) return found;
      const row = buildPipelineEvidenceSampleRow(stage, index);
      return row?.label === config.label ? row : null;
    }, null);
    const sampleCount = matchedStageRows.reduce((total, row) => total + row.sampleCount, 0);
    const durationMs = matchedStageRows.reduce((total, row) => total + (row.durationMs || 0), 0);
    const warningCount = matchedStageRows.reduce((total, row) => total + row.warningCount, 0);
    const warningText = matchedStageRows
      .map((row) => row.warningText)
      .find(Boolean) || "";
    const completedCount = matchedStageRows.filter((row) =>
      PIPELINE_SUCCESS_STATUSES.has(row.status.toLowerCase()),
    ).length;

    return {
      key: config.id,
      label: config.label,
      color: config.color,
      status: pipelineLaneStatus(matchedStageRows),
      stageCount: matchedStageRows.length,
      completedCount,
      sampleCount,
      warningCount,
      durationMs,
      sampleTitle: sampleRow?.title || representative?.title || "Waiting for evidence",
      sampleBody: sampleRow?.body || representative?.body || warningText || "",
      sampleCountLabel:
        sampleRow?.countLabel ||
        (sampleCount > 0 ? `${formatNumber(sampleCount)} evidence items` : `${formatNumber(matchedStageRows.length)} checks`),
      warningText,
    };
  }).filter((row) => row.stageCount > 0);

const hasPipelineStageSampleOutputs = (run: unknown): boolean =>
  asArray(asRecord(run).stages).some((stage) => {
    const sampleOutputs = asRecord(asRecord(stage).sample_outputs);
    return Object.keys(sampleOutputs).length > 0;
  });

const pipelineRunsShareDocument = graphRAGPipelineRunsShareDocument;
const preferredPipelineRunForDocumentChat =
  preferredGraphRAGPipelineRunForDocumentChat;

const parseJsonObject = (
  value: string,
  label: string,
): Record<string, unknown> | undefined => {
  const trimmed = value.trim();
  if (!trimmed) return undefined;

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    throw new Error(`${label} must be valid JSON`);
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object`);
  }

  return parsed as Record<string, unknown>;
};

const parseJsonValue = (value: string, label: string): unknown => {
  const trimmed = value.trim();
  if (!trimmed) return undefined;

  try {
    return JSON.parse(trimmed);
  } catch {
    throw new Error(`${label} must be valid JSON`);
  }
};

const extractCount = (value: unknown, keys: string[]) => {
  const record = asRecord(value);

  for (const key of keys) {
    const candidate = record[key];

    if (typeof candidate === "number") return candidate;
    if (Array.isArray(candidate)) return candidate.length;
  }

  return 0;
};

const settledValue = <T,>(result: PromiseSettledResult<T>) =>
  result.status === "fulfilled"
    ? result.value
    : { error: getErrorMessage(result.reason) };

interface JsonResultProps {
  value: unknown;
  emptyLabel?: string;
}

const JsonResult: React.FC<JsonResultProps> = ({
  value,
  emptyLabel = "No result yet",
}) => {
  if (value === null || value === undefined) {
    return (
      <Box
        sx={{
          borderRadius: "8px",
          border: "1px dashed",
          borderColor: "divider",
          px: 1.25,
          py: 1,
          color: "text.secondary",
          fontSize: "0.8rem",
        }}
      >
        {emptyLabel}
      </Box>
    );
  }

  return (
    <Box
      component="details"
      sx={{
        borderRadius: "8px",
        border: "1px solid",
        borderColor: "divider",
        overflow: "hidden",
        bgcolor: alpha(accentColors.slate, 0.08),
        "& pre": {
          fontSize: "0.74rem !important",
          lineHeight: "1.45 !important",
        },
        "&[open]": {
          maxHeight: 420,
        },
      }}
    >
      <Box
        component="summary"
        sx={{
          cursor: "pointer",
          px: 1.25,
          py: 0.8,
          fontSize: "0.78rem",
          fontWeight: 800,
          color: "text.secondary",
          listStyle: "none",
          "&::-webkit-details-marker": { display: "none" },
          "&:hover": { color: "text.primary" },
        }}
      >
        Raw response
      </Box>
      <CodeHighlighter
        code={JSON.stringify(value, null, 2)}
        language="json"
        customStyle={{
          maxHeight: 360,
          overflow: "auto",
          background: "rgba(15, 23, 42, 0.72)",
        }}
      />
    </Box>
  );
};

const countNestedArrayItems = (value: unknown, key: string): number => {
  const direct = asArray(asRecord(value)[key]).length;
  if (direct > 0) return direct;

  return asArray(value).reduce<number>(
    (total, item) => total + countNestedArrayItems(item, key),
    0,
  );
};

const QueryResultSummary: React.FC<{ value: unknown }> = ({ value }) => {
  if (value === null || value === undefined) {
    return (
      <JsonResult
        value={value}
        emptyLabel="Run a query or build context to see results."
      />
    );
  }

  const result = asRecord(value);
  const enhanced = asRecord(result.enhanced);
  const chatContext = asRecord(result.chat_context);
  const chunks = asArray(chatContext.chunks);
  const entities = asArray(chatContext.entities);
  const relationships = asArray(chatContext.relationships);
  const graphPathCount =
    countNestedArrayItems(chunks, "graph_paths") +
    asArray(chatContext.graph_paths).length;
  const answer = asString(enhanced.answer);
  const enhancedResults = asArray(enhanced.results);
  const inferredRelationships = asArray(enhanced.inferred_relationships);
  const hasContext =
    chunks.length > 0 ||
    entities.length > 0 ||
    relationships.length > 0 ||
    graphPathCount > 0;
  const noEnhancedAnswer = answer
    .toLowerCase()
    .includes("no enhanced reasoning results");

  return (
    <Stack spacing={1}>
      {answer && (
        <Alert
          severity={noEnhancedAnswer ? "warning" : "success"}
          variant="outlined"
          sx={{ alignItems: "flex-start" }}
        >
          <Stack spacing={0.5}>
            <Typography variant="body2" fontWeight={800}>
              {noEnhancedAnswer
                ? "Enhanced query did not find an answer"
                : "Answer"}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {answer}
            </Typography>
          </Stack>
        </Alert>
      )}
      {hasContext && (
        <Box
          sx={{
            border: "1px solid",
            borderColor: alpha(accentColors.primary, 0.24),
            borderRadius: "8px",
            p: 1.1,
            bgcolor: alpha(accentColors.info, 0.04),
          }}
        >
          <Stack spacing={1}>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 1,
                flexWrap: "wrap",
              }}
            >
              <Typography variant="subtitle2" fontWeight={800}>
                Graph Context
              </Typography>
              <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                {chunks.length > 0 && (
                  <Chip
                    label={`${formatNumber(chunks.length)} chunks`}
                    size="small"
                    variant="outlined"
                    sx={{ height: 22 }}
                  />
                )}
                {entities.length > 0 && (
                  <Chip
                    label={`${formatNumber(entities.length)} entities`}
                    size="small"
                    variant="outlined"
                    sx={{ height: 22 }}
                  />
                )}
                {relationships.length > 0 && (
                  <Chip
                    label={`${formatNumber(relationships.length)} relationships`}
                    size="small"
                    variant="outlined"
                    sx={{ height: 22 }}
                  />
                )}
                {graphPathCount > 0 && (
                  <Chip
                    label={`${formatNumber(graphPathCount)} graph paths`}
                    size="small"
                    color="info"
                    variant="outlined"
                    sx={{ height: 22 }}
                  />
                )}
              </Stack>
            </Box>
            {chunks.slice(0, 3).map((chunk, index) => {
              const chunkRecord = asRecord(chunk);
              const title = asString(
                chunkRecord.document_title,
                `Chunk ${index + 1}`,
              );
              const content = asString(chunkRecord.content);

              return (
                <Box
                  key={`${title}-${index}`}
                  sx={{
                    border: "1px solid",
                    borderColor: "divider",
                    borderRadius: "8px",
                    p: 1,
                    bgcolor: alpha(accentColors.slate, 0.09),
                  }}
                >
                  <Typography
                    variant="caption"
                    fontWeight={800}
                    sx={{ display: "block" }}
                  >
                    {title}
                  </Typography>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{
                      display: "-webkit-box",
                      WebkitLineClamp: 3,
                      WebkitBoxOrient: "vertical",
                      overflow: "hidden",
                      lineHeight: 1.45,
                    }}
                  >
                    {content}
                  </Typography>
                </Box>
              );
            })}
          </Stack>
        </Box>
      )}
      {!answer && !hasContext && (
        <Alert severity="info" variant="outlined">
          No readable query result was returned.
        </Alert>
      )}
      {(enhancedResults.length > 0 || inferredRelationships.length > 0) && (
        <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
          {enhancedResults.length > 0 && (
            <Chip
              label={`${formatNumber(enhancedResults.length)} enhanced results`}
              size="small"
              variant="outlined"
              sx={{ height: 22 }}
            />
          )}
          {inferredRelationships.length > 0 && (
            <Chip
              label={`${formatNumber(inferredRelationships.length)} inferred relationships`}
              size="small"
              variant="outlined"
              sx={{ height: 22 }}
            />
          )}
        </Stack>
      )}
      <JsonResult value={value} />
    </Stack>
  );
};

interface TabPanelProps {
  children: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, index, value }) => (
  <Box hidden={value !== index} role="tabpanel">
    {value === index && <Box sx={{ pt: 1.25 }}>{children}</Box>}
  </Box>
);

export const GraphRAGWorkbenchPage: React.FC = () => {
  const theme = useTheme();
  const desktopWorkspaceLayout = useMediaQuery(theme.breakpoints.up("md"));
  const navigate = useNavigate();
  const location = useLocation();
  const routeState = asRecord(location.state);
  const routeSelectedGraphRAGPipelineRunId = asString(
    routeState.selectedGraphRAGPipelineRunId,
  );
  const routeSelectedGraphRAGFilename = asString(
    routeState.selectedGraphRAGFilename,
  );
  const [activeTab, setActiveTab] = useState(6);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewError, setOverviewError] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<unknown>(null);
  const [deploymentDiagnostics, setDeploymentDiagnostics] =
    useState<unknown>(null);
  const [health, setHealth] = useState<unknown>(null);
  const [serviceStatus, setServiceStatus] = useState<unknown>(null);
  const [graphStats] = useState<unknown>(null);
  const [domainStats] = useState<unknown>(null);
  const [evaluationStatus] = useState<unknown>(null);
  const [codeStats, setCodeStats] = useState<unknown>(null);
  const [memoryHealth] = useState<unknown>(null);
  const [extractionCapabilities] = useState<unknown>(null);
  const [redditHealth] = useState<unknown>(null);

  const [queryText, setQueryText] = useState(
    "Which entities connect semantic chunking to answer quality?",
  );
  const [queryDomain, setQueryDomain] = useState("general");
  const [querySearchType, setQuerySearchType] = useState<SearchType>("hybrid");
  const [queryTopK, setQueryTopK] = useState(8);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [queryResult, setQueryResult] = useState<unknown>(null);

  const [bridgeMethod, setBridgeMethod] = useState<UpstreamMethod>("GET");
  const [bridgePath, setBridgePath] = useState("api/query-complexity-analysis");
  const [bridgeParams, setBridgeParams] = useState(
    '{\n  "query": "How does GraphRAG improve retrieval quality?"\n}',
  );
  const [bridgeBody, setBridgeBody] = useState("");
  const [bridgeLoading, setBridgeLoading] = useState(false);
  const [bridgeError, setBridgeError] = useState<string | null>(null);
  const [bridgeResult, setBridgeResult] = useState<unknown>(null);

  const [graphDomain, setGraphDomain] = useState("general");
  const [graphEntityName, setGraphEntityName] = useState("GraphRAG");
  const [graphLimit, setGraphLimit] = useState(25);
  const [graphLoading, setGraphLoading] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [graphResult, setGraphResult] = useState<unknown>(null);

  const [corpusDomain, setCorpusDomain] = useState("general");
  const [corpusDocumentName, setCorpusDocumentName] = useState("");
  const [corpusLoading, setCorpusLoading] = useState(false);
  const [corpusError, setCorpusError] = useState<string | null>(null);
  const [corpusResult, setCorpusResult] = useState<unknown>(null);
  const [corpusUploadFile, setCorpusUploadFile] = useState<File | null>(null);
  const [corpusUploadFiles, setCorpusUploadFiles] = useState<File[]>([]);
  const [corpusDragActive, setCorpusDragActive] = useState(false);
  const [corpusUseSemanticChunking, setCorpusUseSemanticChunking] =
    useState(true);
  const [corpusBuildKnowledgeGraph, setCorpusBuildKnowledgeGraph] =
    useState(true);
  const [pipelineRuns, setPipelineRuns] = useState<Record<string, unknown>[]>(
    [],
  );
  const [corpusRunSearch, setCorpusRunSearch] = useState("");
  const [selectedCorpusPipelineRunId, setSelectedCorpusPipelineRunId] =
    useState<string | null>(null);
  const [selectedEvidenceRowKey, setSelectedEvidenceRowKey] =
    useState<string | null>(null);
  const [documentInsightView, setDocumentInsightView] =
    useState<DocumentInsightView>("overview");
  const [mobileWorkspaceSection, setMobileWorkspaceSection] =
    useState<GraphRAGDocumentWorkspaceSection | null>(null);
  const [graphChatQuestion, setGraphChatQuestion] = useState("");
  const [stageSamplesExpanded, setStageSamplesExpanded] = useState(false);
  const [advancedTraceExpanded, setAdvancedTraceExpanded] = useState(false);
  const hydratingCorpusPipelineRunIdRef = useRef<string | null>(null);
  const appliedRouteSelectedRunIdRef = useRef<string | null>(null);
  const reviewReadyAutoFocusedRunIdRef = useRef<string | null>(null);
  const uploadIntakeRef = useRef<HTMLDivElement | null>(null);
  const documentLibraryRef = useRef<HTMLDivElement | null>(null);
  const selectedDocumentPanelRef = useRef<HTMLDivElement | null>(null);
  const selectedDocumentQuestionRef = useRef<HTMLFormElement | null>(null);
  const stageSamplesDetailsRef = useRef<HTMLDetailsElement | null>(null);
  const advancedTraceDetailsRef = useRef<HTMLDetailsElement | null>(null);
  const revealProcessingDetails = useCallback(
    (
      detailsRef: React.RefObject<HTMLDetailsElement | null>,
      setExpanded: (expanded: boolean) => void,
    ) => {
      setExpanded(true);
      window.requestAnimationFrame(() => {
        detailsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    },
    [],
  );
  const chatContextPipelineRunIds = useMemo(
    () => {
      const preferredDocumentRuns = groupGraphRAGDocumentPipelineRuns(
        pipelineRuns,
        {
          fallbackDomain: corpusDomain.trim() || "general",
          groupAcrossDomains: true,
        },
      ).map(
        (group) => preferredPipelineRunForDocumentChat(group.runs) || group.run,
      );

      return uniqueStringList([
        selectedCorpusPipelineRunId || "",
        ...preferredDocumentRuns.map((run) => asString(run.run_id)),
      ]).slice(0, MAX_CHAT_CONTEXT_PIPELINE_RUNS);
    },
    [corpusDomain, pipelineRuns, selectedCorpusPipelineRunId],
  );

  const [extractionText, setExtractionText] = useState(
    "Ada Lovelace collaborated with Charles Babbage on the Analytical Engine and described algorithms for symbolic computation.",
  );
  const [extractionDomain, setExtractionDomain] = useState("general");
  const [useSpanbert, setUseSpanbert] = useState(true);
  const [useDependency, setUseDependency] = useState(true);
  const [useEntityLinking, setUseEntityLinking] = useState(true);
  const [extractionLoading, setExtractionLoading] = useState(false);
  const [extractionError, setExtractionError] = useState<string | null>(null);
  const [extractionResult, setExtractionResult] = useState<unknown>(null);

  const [testSuite, setTestSuite] = useState<TestSuite>("quality");
  const [qualityQuery, setQualityQuery] = useState(
    "What evidence supports graph-aware retrieval?",
  );
  const [expectedAnswer, setExpectedAnswer] = useState(
    "The answer should cite retrieved graph context and relevant source snippets.",
  );
  const [feedbackQuery, setFeedbackQuery] = useState(
    "Which entities connect semantic chunking to answer quality?",
  );
  const [feedbackRating, setFeedbackRating] = useState<FeedbackRating>(1);
  const [feedbackComment, setFeedbackComment] = useState(
    "Relevant sources and graph context were useful.",
  );
  const [feedbackDomain, setFeedbackDomain] = useState("general");
  const [feedbackDays, setFeedbackDays] = useState(30);
  const [qualityLoading, setQualityLoading] = useState(false);
  const [qualityError, setQualityError] = useState<string | null>(null);
  const [qualityResult, setQualityResult] = useState<unknown>(null);

  const [projectName, setProjectName] = useState("llama-nexus");
  const [codeLanguage, setCodeLanguage] = useState("python");
  const [codePaths, setCodePaths] = useState(
    "backend/routes/graphrag/search.py\nbackend/routes/graphrag/extraction.py",
  );
  const [codeAnalyzePath, setCodeAnalyzePath] = useState(
    "backend/routes/graphrag/search.py",
  );
  const [codeTaskId, setCodeTaskId] = useState("");
  const [codeLoading, setCodeLoading] = useState(false);
  const [codeError, setCodeError] = useState<string | null>(null);
  const [codeResult, setCodeResult] = useState<unknown>(null);

  const [memoryPlatform, setMemoryPlatform] =
    useState<SocialPlatform>("twitch");
  const [memoryUserId, setMemoryUserId] = useState("demo-user");
  const [memoryDisplayName, setMemoryDisplayName] = useState("Demo User");
  const [memoryMessage, setMemoryMessage] = useState(
    "I am comparing GraphRAG retrieval quality across project documents.",
  );
  const [memoryQuery, setMemoryQuery] = useState(
    "What should I remember about this user?",
  );
  const [memoryPersonId, setMemoryPersonId] = useState("");
  const [memoryAdminKey, setMemoryAdminKey] = useState("");
  const [memoryEvalSuite, setMemoryEvalSuite] =
    useState<SocialMemoryEvalSuite>("all");
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [memoryResult, setMemoryResult] = useState<unknown>(null);

  const [redditQuery, setRedditQuery] = useState("local LLM retrieval quality");
  const [redditSubreddits, setRedditSubreddits] = useState(
    "LocalLLaMA, MachineLearning",
  );
  const [redditMaxPosts, setRedditMaxPosts] = useState(25);
  const [redditAutoIngest, setRedditAutoIngest] = useState(true);
  const [redditCrawlId, setRedditCrawlId] = useState("");
  const [redditLoading, setRedditLoading] = useState(false);
  const [redditError, setRedditError] = useState<string | null>(null);
  const [redditResult, setRedditResult] = useState<unknown>(null);

  const capabilityRecord = asRecord(capabilities);
  const diagnosticsRecord = asRecord(deploymentDiagnostics);
  const healthRecord = asRecord(health);
  const normalizedServiceStatus = normalizeGraphRAGServiceStatus(
    serviceStatus,
    health,
  );
  const graphStatsRecord = asRecord(graphStats);
  const serviceRecord = asRecord(capabilityRecord.services);
  const serviceStatusEntries = normalizedServiceStatus.entries;
  const serviceReadyCount = normalizedServiceStatus.readyCount;
  const serviceConfiguredCount = normalizedServiceStatus.configuredCount;
  const graphRAGCoreService = normalizedServiceStatus.graphRAGCoreService;
  const serviceStatusLoaded = normalizedServiceStatus.loaded;
  const remoteGraphRAGReady = normalizedServiceStatus.remoteGraphRAGReady;
  const serviceSmokeStatus = normalizedServiceStatus.smokeStatus;
  const serviceStatusSummary = normalizedServiceStatus.summary;
  const serviceSourceSummary =
    normalizedServiceStatus.sourceSummary ||
    asString(diagnosticsRecord.source_summary);
  const serviceStatusNextSteps = normalizedServiceStatus.nextSteps;
  const serviceOperatingModeLabel = normalizedServiceStatus.operatingModeLabel;
  const serviceOperatingModeSummary =
    normalizedServiceStatus.operatingModeSummary;
  const serviceOperatingModeAction =
    normalizedServiceStatus.operatingModeAction;
  const serviceContractItems = normalizedServiceStatus.contractItems;
  const serviceEvidenceCapabilities =
    normalizedServiceStatus.evidenceCapabilities;
  const serviceSmokeTests = normalizedServiceStatus.smokeTests;
  const serviceWorkflowEntries = normalizedServiceStatus.workflowEntries;
  const serviceOperatorChecklist = normalizedServiceStatus.operatorChecklist;
  const serviceDeploymentQuickstart =
    normalizedServiceStatus.deploymentQuickstart;
  const serviceInteractionPathways =
    normalizedServiceStatus.interactionPathways;
  const serviceRecommendedPathwayId =
    normalizedServiceStatus.recommendedPathwayId;
  const serviceWorkflowReadyCount = normalizedServiceStatus.workflowReadyCount;
  const serviceWorkflowCount = normalizedServiceStatus.workflowCount;
  const serviceStatusSeverity = normalizedServiceStatus.severity;
  const diagnosticFeatureRecord = asRecord(diagnosticsRecord.features);
  const diagnosticFeatures = Object.entries(diagnosticFeatureRecord).map(
    ([name, feature]) => ({
      name,
      data: asRecord(feature),
    }),
  );
  const diagnosticWarnings = asArray(diagnosticsRecord.warnings).map(String);
  const capabilityList = asArray(capabilityRecord.capabilities);
  const enabledServices = Object.values(serviceRecord).filter(
    (service) => asRecord(service).enabled === true,
  ).length;
  const graphStatus =
    healthRecord.status === "healthy" || healthRecord.status === "ok"
      ? "online"
      : healthRecord.status === "partial"
        ? "partial"
        : "offline";
  const graphStatusColor =
    graphStatus === "online"
      ? accentColors.success
      : graphStatus === "partial"
        ? accentColors.warning
        : accentColors.rose;
  const deploymentStatus = String(
    diagnosticsRecord.status || healthRecord.status || "disabled",
  );
  const deploymentChipStatus =
    deploymentStatus === "ready" ||
    deploymentStatus === "healthy" ||
    deploymentStatus === "ok"
      ? "ready"
      : deploymentStatus === "partial"
        ? "partial"
        : deploymentStatus === "local"
          ? "local"
          : deploymentStatus === "disabled"
            ? "disabled"
            : "offline";
  const bridgeRecord = asRecord(capabilityRecord.upstream_bridge);
  const bridgeAvailable = bridgeRecord.available === true;
  const allowedPrefixes = asArray(bridgeRecord.allowed_prefixes).map(String);

  const extractionSummary = useMemo(() => {
    const resultRecord = asRecord(extractionResult);
    const enhanced = resultRecord.enhanced;
    const ner = resultRecord.ner;
    const relations = resultRecord.relations;

    return {
      entities:
        extractCount(enhanced, ["entity_count", "entities"]) ||
        extractCount(ner, ["entity_count", "entities"]),
      relationships:
        extractCount(enhanced, ["relationship_count", "relationships"]) ||
        extractCount(relations, ["relation_count", "relations"]),
      diagnostics: Object.keys(asRecord(resultRecord.diagnostics)).length,
    };
  }, [extractionResult]);

  const loadOverview = useCallback(async () => {
    setOverviewLoading(true);
    setOverviewError(null);

    const errors: string[] = [];
    const [
      diagnosticsResult,
      capabilitiesResult,
      healthResult,
      serviceStatusResult,
    ] = await Promise.allSettled([
      api.getGraphRAGDeploymentDiagnostics(),
      api.getGraphRAGCapabilities(),
      api.getGraphRAGHealth(),
      api.getGraphRAGServicesStatus(),
    ]);

    if (diagnosticsResult.status === "fulfilled")
      setDeploymentDiagnostics(diagnosticsResult.value);
    else
      errors.push(`Deployment: ${getErrorMessage(diagnosticsResult.reason)}`);

    if (capabilitiesResult.status === "fulfilled")
      setCapabilities(capabilitiesResult.value);
    else
      errors.push(
        `Capabilities: ${getErrorMessage(capabilitiesResult.reason)}`,
      );

    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    else errors.push(`Health: ${getErrorMessage(healthResult.reason)}`);

    if (serviceStatusResult.status === "fulfilled")
      setServiceStatus(serviceStatusResult.value);
    else
      errors.push(`Services: ${getErrorMessage(serviceStatusResult.reason)}`);

    setOverviewError(errors.length ? errors.join(" | ") : null);
    setOverviewLoading(false);
  }, []);

  const scrollDocumentWorkspaceSectionIntoView = useCallback(
    (section: GraphRAGDocumentWorkspaceSection) => {
      if (typeof window === "undefined") return;

      setMobileWorkspaceSection(section);
      window.setTimeout(() => {
        const target =
          section === "upload"
            ? uploadIntakeRef.current
            : section === "documents"
              ? documentLibraryRef.current
              : selectedDocumentQuestionRef.current ||
                selectedDocumentPanelRef.current;

        target?.scrollIntoView({
          behavior: "smooth",
          block: section === "selected" ? "center" : "start",
        });
      }, 40);
    },
    [],
  );

  useEffect(() => {
    loadOverview();
  }, [loadOverview]);

  const handleRunQuery = async () => {
    const query = queryText.trim();
    if (!query) return;

    setQueryLoading(true);
    setQueryError(null);
    setQueryResult(null);

    try {
      const [complexity, enhanced] = await Promise.all([
        api.analyzeGraphRAGQueryComplexity(query),
        api.enhancedGraphRAGQuery({
          query,
          search_type: querySearchType,
          top_k: queryTopK,
          domain: queryDomain.trim() || undefined,
        }),
      ]);

      setQueryResult({ complexity, enhanced });
    } catch (error) {
      setQueryError(getErrorMessage(error));
    } finally {
      setQueryLoading(false);
    }
  };

  const handleBuildChatContext = async () => {
    const query = queryText.trim();
    if (!query) return;

    setQueryLoading(true);
    setQueryError(null);
    setQueryResult(null);

    try {
      const pipelineRunIds =
        chatContextPipelineRunIds.length > 0
          ? chatContextPipelineRunIds
          : undefined;
      const contextRequest = {
        query,
        mode: "v2",
        top_k: queryTopK,
        domain: queryDomain.trim() || "general",
        rag_enabled: true,
        rag_search_mode: "domain",
        pipeline_run_ids: pipelineRunIds,
        use_local_pipeline_context: pipelineRunIds ? true : undefined,
        include_local_pipeline: pipelineRunIds ? true : undefined,
        include_graph: true,
      } as const;
      const context = await api.getGraphRAGChatContext(contextRequest);

      setQueryResult({
        chat_context_request: contextRequest,
        chat_context: context,
      });
    } catch (error) {
      setQueryError(getErrorMessage(error));
    } finally {
      setQueryLoading(false);
    }
  };

  const handleRunBridge = async () => {
    const path = bridgePath.trim();
    if (!path) return;

    setBridgeLoading(true);
    setBridgeError(null);
    setBridgeResult(null);

    try {
      const params = parseJsonObject(bridgeParams, "Params");
      const data = parseJsonValue(bridgeBody, "Body");
      const response = await api.callGraphRAGUpstream(path, {
        method: bridgeMethod,
        params,
        data,
        timeout: 300000,
      });

      setBridgeResult({
        request: {
          method: bridgeMethod,
          path: path.replace(/^\/+/, ""),
          params: params ?? {},
          body: data ?? null,
        },
        response,
      });
    } catch (error) {
      setBridgeError(getErrorMessage(error));
    } finally {
      setBridgeLoading(false);
    }
  };

  const runGraphOperation = async (
    operation: () => Promise<Record<string, unknown> | unknown>,
    resultKey: string,
  ) => {
    setGraphLoading(true);
    setGraphError(null);
    setGraphResult(null);

    try {
      const result = await operation();
      setGraphResult({ [resultKey]: result });
    } catch (error) {
      setGraphError(getErrorMessage(error));
    } finally {
      setGraphLoading(false);
    }
  };

  const graphDomainParam = () => graphDomain.trim() || undefined;

  const handleGraphNeighborhood = async () => {
    const entityName = graphEntityName.trim();
    if (!entityName) return;

    await runGraphOperation(
      () =>
        api.getGraphRAGEntityNeighborhood({
          entity_name: entityName,
          domain: graphDomainParam(),
          limit: graphLimit,
        }),
      "neighborhood",
    );
  };

  const handleGraphCommunities = async () => {
    await runGraphOperation(
      () => api.getGraphRAGCommunities(graphDomainParam()),
      "communities",
    );
  };

  const handleGraphFiltered = async () => {
    await runGraphOperation(
      () =>
        api.getGraphRAGGraph({
          domain: graphDomainParam(),
          max_entities: graphLimit,
          max_relationships: graphLimit * 2,
        }),
      "graph",
    );
  };

  const handleGraphTopEntities = async () => {
    await runGraphOperation(
      () =>
        api.getGraphRAGTopEntities({
          domain: graphDomainParam(),
          limit: graphLimit,
        }),
      "top_entities",
    );
  };

  const handleGraphTopRelationships = async () => {
    await runGraphOperation(
      () =>
        api.getGraphRAGTopRelationships({
          domain: graphDomainParam(),
          limit: graphLimit,
        }),
      "top_relationships",
    );
  };

  const corpusPipelineOptions = () => ({
    domain: corpusDomain.trim() || "general",
    useSemanticChunking: corpusUseSemanticChunking,
    buildKnowledgeGraph: corpusBuildKnowledgeGraph,
    sampleLimit: 8,
  });

  const rememberCorpusPipelineRun = (result: unknown) => {
    const run = asRecord(result);
    const runId = asString(run.run_id);
    if (!runId) return;
    setPipelineRuns((prev) =>
      [run, ...prev.filter((item) => asString(item.run_id) !== runId)].slice(
        0,
        12,
      ),
    );
  };

  const rememberCorpusPipelineRuns = (runs: unknown[]) => {
    const normalizedRuns = runs
      .filter(
        (run): run is Record<string, unknown> =>
          Boolean(run) && typeof run === "object" && !Array.isArray(run),
      )
      .filter((run) => Boolean(asString(run.run_id)));
    if (normalizedRuns.length === 0) return;

    setPipelineRuns((prev) => {
      const seen = new Set<string>();
      const merged = [...normalizedRuns, ...prev].filter((run) => {
        const runId = asString(run.run_id);
        if (!runId || seen.has(runId)) return false;
        seen.add(runId);
        return true;
      });
      return merged.slice(0, 12);
    });
  };

  const setPreferredDocumentInsightViewForRuns = (
    runs: Record<string, unknown>[],
  ) => {
    const view = preferredDocumentInsightViewForRuns(runs);
    if (view === "overview") {
      const runId = asString(runs.find((run) => asString(run.run_id))?.run_id);
      if (runId) {
        reviewReadyAutoFocusedRunIdRef.current = runId;
      }
    }
    setDocumentInsightView(view);
  };

  const applyCorpusBatchUploadResult = (
    result: Record<string, unknown>,
    uploadedFiles: File[],
  ) => {
    const runs = asArray(result.runs).filter(
      (run): run is Record<string, unknown> =>
        Boolean(run) && typeof run === "object" && !Array.isArray(run),
    );
    const firstRun = runs.find((run) =>
      Boolean(asString(asRecord(run).run_id)),
    );

    setCorpusResult({ pipeline_batch_upload: result });
    setSelectedCorpusPipelineRunId(
      firstRun ? asString(asRecord(firstRun).run_id) : null,
    );
    setPreferredDocumentInsightViewForRuns(runs);
    setMobileWorkspaceSection("selected");
    setCorpusDocumentName(
      uploadedFiles.length === 1 ? uploadedFiles[0].name : "",
    );
    rememberCorpusPipelineRuns(runs);
    setCorpusUploadFile(null);
    setCorpusUploadFiles([]);
  };

  const recoverCorpusPipelineBatchUpload = async (
    files: File[],
    uploadStartedAtMs: number,
  ): Promise<Record<string, unknown> | null> => {
    const domain = corpusDomain.trim() || "general";
    const deadlineMs = Date.now() + BATCH_UPLOAD_RECOVERY_TIMEOUT_MS;
    const limit = Math.max(24, files.length * 8);

    while (Date.now() <= deadlineMs) {
      try {
        const result = await api.listGraphRAGIngestionPipelineRuns({
          limit,
          domain,
        });
        const runs = asArray(asRecord(result).runs).filter(
          (run): run is Record<string, unknown> =>
            Boolean(run) && typeof run === "object" && !Array.isArray(run),
        );
        const recoveredRuns = completedRecentPipelineRunsForFiles(
          runs,
          files,
          uploadStartedAtMs,
        );

        if (recoveredRuns.length > 0) {
          return buildRecoveredBatchUploadResult({
            files,
            runs: recoveredRuns,
            domain,
            uploadStartedAtMs,
          });
        }
      } catch {
        return null;
      }

      await delay(BATCH_UPLOAD_RECOVERY_POLL_MS);
    }

    return null;
  };

  const handleCorpusFilesSelected = (files: File[]) => {
    const file = files[0] ?? null;
    setCorpusUploadFile(file);
    setCorpusUploadFiles(files);
    setCorpusError(null);
    setCorpusResult(null);
    if (file) {
      setCorpusDocumentName(file.name);
      setMobileWorkspaceSection("upload");
    }
  };

  const handleCorpusUploadFileChange = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const files = Array.from(event.target.files ?? []);
    handleCorpusFilesSelected(files);
    event.target.value = "";
  };

  const handleCorpusDrop = (event: React.DragEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setCorpusDragActive(false);
    if (corpusLoading) return;

    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length > 0) {
      handleCorpusFilesSelected(files);
    }
  };

  const handleCorpusPipelinePreview = async () => {
    if (!corpusUploadFile) return;

    setCorpusLoading(true);
    setCorpusError(null);
    setCorpusResult(null);

    try {
      const result = await api.previewGraphRAGIngestionPipeline(
        corpusUploadFile,
        corpusPipelineOptions(),
      );
      setCorpusResult({ pipeline_preview: result });
      setSelectedCorpusPipelineRunId(null);
      setPreferredDocumentInsightViewForRuns([asRecord(result)]);
      setMobileWorkspaceSection("selected");
      setCorpusDocumentName(corpusUploadFile.name);
      rememberCorpusPipelineRun(result);
    } catch (error) {
      setCorpusError(getErrorMessage(error));
    } finally {
      setCorpusLoading(false);
    }
  };

  const handleCorpusPipelineUpload = async () => {
    if (!corpusUploadFile) return;

    setCorpusLoading(true);
    setCorpusError(null);
    setCorpusResult(null);

    try {
      const result = await api.uploadGraphRAGDocumentWithPipeline(
        corpusUploadFile,
        {
          ...corpusPipelineOptions(),
          submitAsync: true,
        },
      );
      const runId = asString(asRecord(result).run_id);
      setCorpusResult({ pipeline_upload: result });
      setSelectedCorpusPipelineRunId(runId || null);
      setPreferredDocumentInsightViewForRuns([asRecord(result)]);
      setMobileWorkspaceSection("selected");
      setCorpusDocumentName(corpusUploadFile.name);
      rememberCorpusPipelineRun(result);
      setCorpusUploadFile(null);
      setCorpusUploadFiles([]);
    } catch (error) {
      setCorpusError(getErrorMessage(error));
    } finally {
      setCorpusLoading(false);
    }
  };

  const handleCorpusPipelineBatchUpload = async () => {
    if (corpusUploadFiles.length === 0) return;

    const uploadFiles = [...corpusUploadFiles];
    const uploadStartedAtMs = Date.now();
    setCorpusLoading(true);
    setCorpusError(null);
    setCorpusResult(null);

    try {
      const result = await api.uploadGraphRAGDocumentsWithPipeline(
        uploadFiles,
        {
          ...corpusPipelineOptions(),
          maxConcurrency: Math.min(4, Math.max(1, uploadFiles.length)),
        },
      );
      applyCorpusBatchUploadResult(asRecord(result), uploadFiles);
    } catch (error) {
      const recoveredResult = await recoverCorpusPipelineBatchUpload(
        uploadFiles,
        uploadStartedAtMs,
      );
      if (recoveredResult) {
        applyCorpusBatchUploadResult(recoveredResult, uploadFiles);
        return;
      }
      setCorpusError(getErrorMessage(error));
    } finally {
      setCorpusLoading(false);
    }
  };

  const handleCorpusPipelineRuns = async () => {
    setCorpusLoading(true);
    setCorpusError(null);

    try {
      const result = await api.listGraphRAGIngestionPipelineRuns({
        limit: 12,
        domain: corpusDomain.trim() || undefined,
      });
      const runs = asArray(asRecord(result).runs).filter(
        (item): item is Record<string, unknown> =>
          Boolean(item) && typeof item === "object" && !Array.isArray(item),
      );
      setPipelineRuns(runs);
      setCorpusResult({ pipeline_runs: result });
      setSelectedCorpusPipelineRunId(asString(runs[0]?.run_id) || null);
      setMobileWorkspaceSection("documents");
    } catch (error) {
      setCorpusError(getErrorMessage(error));
    } finally {
      setCorpusLoading(false);
    }
  };

  useEffect(() => {
    let mounted = true;

    const loadRecentPipelineRuns = async () => {
      setCorpusLoading(true);
      setCorpusError(null);

      try {
        const result = await api.listGraphRAGIngestionPipelineRuns({
          limit: 12,
        });
        if (!mounted) return;
        const runs = asArray(asRecord(result).runs).filter(
          (item): item is Record<string, unknown> =>
            Boolean(item) && typeof item === "object" && !Array.isArray(item),
        );
        setPipelineRuns(runs);
        if (runs.length > 0) {
          setSelectedCorpusPipelineRunId(
            (current) => current || asString(runs[0]?.run_id) || null,
          );
          setMobileWorkspaceSection((current) => current || "selected");
        }
        if (runs.length > 0) {
          setCorpusResult({ pipeline_runs: result });
        }
      } catch (error) {
        if (mounted) {
          setCorpusError(getErrorMessage(error));
        }
      } finally {
        if (mounted) {
          setCorpusLoading(false);
        }
      }
    };

    void loadRecentPipelineRuns();

    return () => {
      mounted = false;
    };
  }, []);

  const scrollSelectedDocumentPanelIntoView = useCallback(() => {
    if (typeof window === "undefined") return;
    const shouldGuideToPanel = window.matchMedia("(max-width: 700px)").matches;
    if (!shouldGuideToPanel) return;

    window.setTimeout(() => {
      const target =
        selectedDocumentQuestionRef.current || selectedDocumentPanelRef.current;
      target?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }, 80);
  }, []);

  const handleCorpusLoadPipelineRun = async (
    runId: string,
    options: { guideToSelectedPanel?: boolean } = {},
  ) => {
    if (!runId) return;
    const selectedSummaryRun = pipelineRuns.find(
      (run) => asString(run.run_id) === runId,
    );
    setSelectedCorpusPipelineRunId(runId);
    if (options.guideToSelectedPanel) {
      setMobileWorkspaceSection("selected");
      scrollSelectedDocumentPanelIntoView();
    }
    if (selectedSummaryRun) {
      setCorpusResult((prev: unknown) => ({
        ...asRecord(prev),
        pipeline_run: selectedSummaryRun,
      }));
    }
    setCorpusLoading(true);
    setCorpusError(null);

    try {
      const result = await api.getGraphRAGIngestionPipelineRun(runId);
      setCorpusResult({ pipeline_run: result });
      setPipelineRuns((prev) =>
        [
          result,
          ...prev.filter((item) => asString(item.run_id) !== runId),
        ].slice(0, 12),
      );
    } catch (error) {
      setCorpusError(getErrorMessage(error));
    } finally {
      setCorpusLoading(false);
    }
  };

  useEffect(() => {
    if (
      !routeSelectedGraphRAGPipelineRunId ||
      appliedRouteSelectedRunIdRef.current === routeSelectedGraphRAGPipelineRunId
    ) {
      return;
    }

    appliedRouteSelectedRunIdRef.current = routeSelectedGraphRAGPipelineRunId;
    void handleCorpusLoadPipelineRun(routeSelectedGraphRAGPipelineRunId);
  }, [routeSelectedGraphRAGPipelineRunId]);

  useEffect(() => {
    const runId = selectedCorpusPipelineRunId || asString(pipelineRuns[0]?.run_id);
    if (!runId || hydratingCorpusPipelineRunIdRef.current === runId) return;

    const selectedRun = pipelineRuns.find((run) => asString(run.run_id) === runId);
    const selectedRunHasSamples = hasPipelineStageSampleOutputs(selectedRun);
    const corpusResultRecord = asRecord(corpusResult);
    const activeRun = asRecord(
      corpusResultRecord.pipeline_run ||
        corpusResultRecord.pipeline_upload ||
        corpusResultRecord.pipeline_preview,
    );
    const activeRunHasSamples =
      asString(activeRun.run_id) === runId && hasPipelineStageSampleOutputs(activeRun);

    if (selectedRunHasSamples || activeRunHasSamples) return;

    let mounted = true;
    hydratingCorpusPipelineRunIdRef.current = runId;

    api
      .getGraphRAGIngestionPipelineRun(runId)
      .then((result) => {
        if (!mounted) return;
        setCorpusResult((prev: unknown) => ({
          ...asRecord(prev),
          pipeline_run: result,
        }));
        setPipelineRuns((prev) =>
          [
            result,
            ...prev.filter((item) => asString(item.run_id) !== runId),
          ].slice(0, 12),
        );
        setSelectedCorpusPipelineRunId((current: string | null) => current || runId);
      })
      .catch((error) => {
        if (mounted) setCorpusError(getErrorMessage(error));
      })
      .finally(() => {
        if (hydratingCorpusPipelineRunIdRef.current === runId) {
          hydratingCorpusPipelineRunIdRef.current = null;
        }
      });

    return () => {
      mounted = false;
    };
  }, [
    corpusResult,
    pipelineRuns,
    selectedCorpusPipelineRunId,
  ]);

  const handleRunExtraction = async () => {
    const text = extractionText.trim();
    if (!text) return;

    setExtractionLoading(true);
    setExtractionError(null);
    setExtractionResult(null);

    try {
      const [ner, relations, relationsBatch, enhanced, diagnostics] =
        await Promise.allSettled([
          api.extractGraphRAGNER(text),
          api.extractGraphRAGRelations({ text }),
          api.extractGraphRAGRelationsBatch({ texts: [text] }),
          api.enhancedGraphRAGExtraction({
            text,
            domain: extractionDomain.trim() || "general",
            useSpanbert,
            useDependency,
            useEntityLinking,
          }),
          api.testGraphRAGEnhancedExtraction({
            text,
            domain: extractionDomain.trim() || "general",
          }),
        ]);

      setExtractionResult({
        ner: settledValue(ner),
        relations: settledValue(relations),
        relations_batch: settledValue(relationsBatch),
        enhanced: settledValue(enhanced),
        diagnostics: settledValue(diagnostics),
      });
    } catch (error) {
      setExtractionError(getErrorMessage(error));
    } finally {
      setExtractionLoading(false);
    }
  };

  const handleRunTestSuite = async () => {
    setQualityLoading(true);
    setQualityError(null);
    setQualityResult(null);

    try {
      const result = await api.runGraphRAGTestSuite(testSuite);
      setQualityResult({ test_suite: testSuite, result });
    } catch (error) {
      setQualityError(getErrorMessage(error));
    } finally {
      setQualityLoading(false);
    }
  };

  const handleGraphCompleteness = async () => {
    setQualityLoading(true);
    setQualityError(null);
    setQualityResult(null);

    try {
      const result = await api.evaluateGraphRAGGraphCompleteness();
      setQualityResult({ graph_completeness: result });
    } catch (error) {
      setQualityError(getErrorMessage(error));
    } finally {
      setQualityLoading(false);
    }
  };

  const handleQueryResponseEval = async () => {
    const query = qualityQuery.trim();
    const expected = expectedAnswer.trim();
    if (!query || !expected) return;

    setQualityLoading(true);
    setQualityError(null);
    setQualityResult(null);

    try {
      const result = await api.evaluateGraphRAGQueryResponses({
        test_queries: [query],
        expected_answers: [expected],
      });
      setQualityResult({ query_response_evaluation: result });
    } catch (error) {
      setQualityError(getErrorMessage(error));
    } finally {
      setQualityLoading(false);
    }
  };

  const handleSubmitFeedback = async () => {
    const query = feedbackQuery.trim();
    if (!query) return;

    setQualityLoading(true);
    setQualityError(null);
    setQualityResult(null);

    try {
      const result = await api.submitGraphRAGFeedback({
        query,
        rating: feedbackRating,
        comment: feedbackComment.trim() || undefined,
        domain: feedbackDomain.trim() || "general",
        source: "workbench",
      });
      setQualityResult({ feedback_submission: result });
    } catch (error) {
      setQualityError(getErrorMessage(error));
    } finally {
      setQualityLoading(false);
    }
  };

  const handleFeedbackAnalytics = async () => {
    setQualityLoading(true);
    setQualityError(null);
    setQualityResult(null);

    try {
      const result = await api.getGraphRAGFeedbackAnalytics({
        days: feedbackDays,
        domain: feedbackDomain.trim() || undefined,
      });
      setQualityResult({ feedback_analytics: result });
    } catch (error) {
      setQualityError(getErrorMessage(error));
    } finally {
      setQualityLoading(false);
    }
  };

  const handleCodeIngest = async () => {
    const file_paths = codePaths
      .split(/\r?\n/)
      .map((path) => path.trim())
      .filter(Boolean);

    if (!file_paths.length || !projectName.trim()) return;

    setCodeLoading(true);
    setCodeError(null);
    setCodeResult(null);

    try {
      const result = await api.ingestGraphRAGCodePaths({
        file_paths,
        project_name: projectName.trim(),
        language: codeLanguage.trim() || undefined,
      });
      const resultRecord = asRecord(result);
      const taskId = resultRecord.task_id ?? resultRecord.id;

      if (typeof taskId === "string") {
        setCodeTaskId(taskId);
      }

      setCodeResult({ ingestion: result });
    } catch (error) {
      setCodeError(getErrorMessage(error));
    } finally {
      setCodeLoading(false);
    }
  };

  const handleCodePoll = async () => {
    if (!codeTaskId.trim()) return;

    setCodeLoading(true);
    setCodeError(null);
    setCodeResult(null);

    try {
      const result = await api.getGraphRAGCodeIngestionStatus(
        codeTaskId.trim(),
      );
      setCodeResult({ ingestion_status: result });
    } catch (error) {
      setCodeError(getErrorMessage(error));
    } finally {
      setCodeLoading(false);
    }
  };

  const handleCodeAnalyze = async () => {
    if (!codeAnalyzePath.trim()) return;

    setCodeLoading(true);
    setCodeError(null);
    setCodeResult(null);

    try {
      const result = await api.analyzeGraphRAGCodeFile(codeAnalyzePath.trim());
      setCodeResult({ analysis: result });
    } catch (error) {
      setCodeError(getErrorMessage(error));
    } finally {
      setCodeLoading(false);
    }
  };

  const handleCodeStats = async () => {
    setCodeLoading(true);
    setCodeError(null);
    setCodeResult(null);

    try {
      const result = await api.getGraphRAGCodeStatistics();
      setCodeStats(result);
      setCodeResult({ statistics: result });
    } catch (error) {
      setCodeError(getErrorMessage(error));
    } finally {
      setCodeLoading(false);
    }
  };

  const handleMemoryIngest = async () => {
    if (
      !memoryUserId.trim() ||
      !memoryDisplayName.trim() ||
      !memoryMessage.trim()
    )
      return;

    setMemoryLoading(true);
    setMemoryError(null);
    setMemoryResult(null);

    try {
      const result = await api.ingestGraphRAGSocialMemoryMessage(
        {
          platform: memoryPlatform,
          platform_user_id: memoryUserId.trim(),
          display_name: memoryDisplayName.trim(),
          text: memoryMessage.trim(),
          timestamp: new Date().toISOString(),
          message_id: `manual-${Date.now()}`,
        },
        { sync: true },
      );
      const resultRecord = asRecord(result);
      if (typeof resultRecord.person_id === "string") {
        setMemoryPersonId(resultRecord.person_id);
      }
      setMemoryResult({ ingest: result });
    } catch (error) {
      setMemoryError(getErrorMessage(error));
    } finally {
      setMemoryLoading(false);
    }
  };

  const handleMemoryContext = async () => {
    if (!memoryUserId.trim()) return;

    setMemoryLoading(true);
    setMemoryError(null);
    setMemoryResult(null);

    try {
      const result = await api.getGraphRAGSocialMemoryContext({
        platform: memoryPlatform,
        platform_user_id: memoryUserId.trim(),
        query: memoryQuery.trim() || undefined,
        max_facts: 8,
        include_recent_episodes: 3,
      });
      const resultRecord = asRecord(result);
      if (typeof resultRecord.person_id === "string") {
        setMemoryPersonId(resultRecord.person_id);
      }
      setMemoryResult({ context: result });
    } catch (error) {
      setMemoryError(getErrorMessage(error));
    } finally {
      setMemoryLoading(false);
    }
  };

  const handleMemoryExplain = async () => {
    if (!memoryUserId.trim() || !memoryQuery.trim()) return;

    setMemoryLoading(true);
    setMemoryError(null);
    setMemoryResult(null);

    try {
      const result = await api.explainGraphRAGSocialMemoryRetrieval({
        platform: memoryPlatform,
        platform_user_id: memoryUserId.trim(),
        query: memoryQuery.trim(),
        max_facts: 8,
      });
      setMemoryResult({ explanation: result });
    } catch (error) {
      setMemoryError(getErrorMessage(error));
    } finally {
      setMemoryLoading(false);
    }
  };

  const handleMemoryFacts = async () => {
    if (!memoryPersonId.trim()) return;

    setMemoryLoading(true);
    setMemoryError(null);
    setMemoryResult(null);

    try {
      const result = await api.getGraphRAGSocialMemoryPersonFacts(
        memoryPersonId.trim(),
      );
      setMemoryResult({ facts: result });
    } catch (error) {
      setMemoryError(getErrorMessage(error));
    } finally {
      setMemoryLoading(false);
    }
  };

  const handleMemoryEval = async () => {
    setMemoryLoading(true);
    setMemoryError(null);
    setMemoryResult(null);

    try {
      const result = await api.runGraphRAGSocialMemoryEval(memoryEvalSuite, {
        adminApiKey: memoryAdminKey.trim() || undefined,
      });
      setMemoryResult({ eval: result });
    } catch (error) {
      setMemoryError(getErrorMessage(error));
    } finally {
      setMemoryLoading(false);
    }
  };

  const handleMemoryCommunityReport = async () => {
    setMemoryLoading(true);
    setMemoryError(null);
    setMemoryResult(null);

    try {
      const result = await api.getGraphRAGSocialMemoryCommunityReport(
        memoryAdminKey.trim() || undefined,
      );
      setMemoryResult({ community_report: result });
    } catch (error) {
      setMemoryError(getErrorMessage(error));
    } finally {
      setMemoryLoading(false);
    }
  };

  const handleMemoryMigrate = async () => {
    setMemoryLoading(true);
    setMemoryError(null);
    setMemoryResult(null);

    try {
      const result = await api.migrateGraphRAGSocialMemory(
        memoryAdminKey.trim() || undefined,
      );
      setMemoryResult({ migration: result });
    } catch (error) {
      setMemoryError(getErrorMessage(error));
    } finally {
      setMemoryLoading(false);
    }
  };

  const handleRedditStart = async () => {
    const query = redditQuery.trim();
    if (!query) return;

    setRedditLoading(true);
    setRedditError(null);
    setRedditResult(null);

    try {
      const subreddits = redditSubreddits
        .split(",")
        .map((subreddit) => subreddit.trim().replace(/^r\//i, ""))
        .filter(Boolean);
      const result = await api.startGraphRAGRedditCrawl({
        query,
        subreddits: subreddits.length ? subreddits : undefined,
        max_posts: redditMaxPosts,
        auto_ingest: redditAutoIngest,
      });
      const resultRecord = asRecord(result);
      const nextCrawlId = resultRecord.crawl_id;

      if (typeof nextCrawlId === "string") {
        setRedditCrawlId(nextCrawlId);
      }

      setRedditResult({ started: result });
    } catch (error) {
      setRedditError(getErrorMessage(error));
    } finally {
      setRedditLoading(false);
    }
  };

  const handleRedditStatus = async () => {
    if (!redditCrawlId.trim()) return;

    setRedditLoading(true);
    setRedditError(null);
    setRedditResult(null);

    try {
      const result = await api.getGraphRAGRedditCrawlStatus(
        redditCrawlId.trim(),
      );
      setRedditResult({ status: result });
    } catch (error) {
      setRedditError(getErrorMessage(error));
    } finally {
      setRedditLoading(false);
    }
  };

  const handleRedditResults = async () => {
    if (!redditCrawlId.trim()) return;

    setRedditLoading(true);
    setRedditError(null);
    setRedditResult(null);

    try {
      const result = await api.getGraphRAGRedditCrawlResults(
        redditCrawlId.trim(),
      );
      setRedditResult({ results: result });
    } catch (error) {
      setRedditError(getErrorMessage(error));
    } finally {
      setRedditLoading(false);
    }
  };

  const handleRedditList = async () => {
    setRedditLoading(true);
    setRedditError(null);
    setRedditResult(null);

    try {
      const result = await api.listGraphRAGRedditCrawls();
      setRedditResult({ crawls: result });
    } catch (error) {
      setRedditError(getErrorMessage(error));
    } finally {
      setRedditLoading(false);
    }
  };

  const handleRedditCancel = async () => {
    if (!redditCrawlId.trim()) return;

    setRedditLoading(true);
    setRedditError(null);
    setRedditResult(null);

    try {
      const result = await api.cancelGraphRAGRedditCrawl(redditCrawlId.trim());
      setRedditResult({ cancelled: result });
    } catch (error) {
      setRedditError(getErrorMessage(error));
    } finally {
      setRedditLoading(false);
    }
  };

  const corpusResultRecord = asRecord(corpusResult);
  const corpusBatchUploadRecord = asRecord(
    corpusResultRecord.pipeline_batch_upload,
  );
  const corpusBatchRuns = asArray(corpusBatchUploadRecord.runs).filter(
    (run): run is Record<string, unknown> =>
      Boolean(run) && typeof run === "object" && !Array.isArray(run),
  );
  const corpusBatchRunIds = corpusBatchRuns
    .map((run) => asString(run.run_id))
    .filter(Boolean);
  const corpusBatchFilenames = corpusBatchRuns
    .map((run) => asString(asRecord(run.document).filename))
    .filter(Boolean);
  const corpusBatchResults = asArray(corpusBatchUploadRecord.results).filter(
    (result): result is Record<string, unknown> =>
      Boolean(result) && typeof result === "object" && !Array.isArray(result),
  );
  const corpusHasBatchResult = Boolean(
    corpusResultRecord.pipeline_batch_upload,
  );
  const corpusBatchSummaryRecord = asRecord(corpusBatchUploadRecord.summary);
  const corpusBatchUploadManifest = asRecord(
    corpusBatchUploadRecord.upload_manifest ||
      corpusBatchSummaryRecord.upload_manifest,
  );
  const corpusBatchManifestFiles = asArray(
    corpusBatchUploadManifest.files,
  ).filter(
    (result): result is Record<string, unknown> =>
      Boolean(result) && typeof result === "object" && !Array.isArray(result),
  );
  const corpusBatchManifestRows =
    corpusBatchManifestFiles.length > 0
      ? corpusBatchManifestFiles
      : corpusBatchResults;
  const corpusBatchManifestLanes = asArray(
    corpusBatchUploadManifest.lanes,
  ).filter(
    (lane): lane is Record<string, unknown> =>
      Boolean(lane) && typeof lane === "object" && !Array.isArray(lane),
  );
  const corpusBatchManifestLargestFile = asRecord(
    corpusBatchUploadManifest.largest_file,
  );
  const corpusBatchManifestLargestFilename = asString(
    corpusBatchManifestLargestFile.filename,
  );
  const corpusBatchManifestLargestBytes = asNumber(
    corpusBatchManifestLargestFile.size_bytes,
  );
  const corpusBatchManifestTotalBytes = asNumber(
    corpusBatchUploadManifest.total_bytes,
  );
  const corpusBatchManifestDomains = asStringArray(
    corpusBatchUploadManifest.domains,
  );
  const corpusBatchManifestNextStep = asString(
    corpusBatchUploadManifest.next_step,
  );
  const corpusBatchTotal = asNumber(
    corpusBatchUploadRecord.total_files ?? corpusBatchUploadManifest.file_count,
    corpusBatchResults.length || corpusBatchRuns.length,
  );
  const corpusBatchProcessed = asNumber(
    corpusBatchUploadRecord.processed ?? corpusBatchUploadManifest.accepted,
    corpusBatchRuns.length,
  );
  const corpusBatchFailed = asNumber(
    corpusBatchUploadRecord.failed ?? corpusBatchUploadManifest.failed,
    Math.max(0, corpusBatchTotal - corpusBatchProcessed),
  );
  const corpusBatchExecutionRecord = asRecord(
    corpusBatchSummaryRecord.execution,
  );
  const corpusBatchOperationalHandoff = asRecord(
    corpusBatchSummaryRecord.operational_handoff,
  );
  const corpusBatchResumeHandoff = asRecord(
    corpusBatchUploadRecord.batch_resume_handoff ||
      corpusBatchSummaryRecord.batch_resume_handoff ||
      corpusBatchUploadManifest.batch_resume_handoff,
  );
  const corpusBatchResumeRouteState = asRecord(
    corpusBatchResumeHandoff.route_state,
  );
  const corpusBatchResumeRouteHandoff = asRecord(
    corpusBatchResumeRouteState.graphragHandoff,
  );
  const corpusBatchOperationalRouteHandoff = asRecord(
    asRecord(corpusBatchOperationalHandoff.route_state).graphragHandoff,
  );
  const corpusBatchManifestRouteHandoff = asRecord(
    asRecord(corpusBatchUploadManifest.route_state).graphragHandoff,
  );
  const corpusBatchPreparedRouteHandoff =
    Object.keys(corpusBatchResumeRouteHandoff).length > 0
      ? corpusBatchResumeRouteHandoff
      : Object.keys(corpusBatchOperationalRouteHandoff).length > 0
        ? corpusBatchOperationalRouteHandoff
        : corpusBatchManifestRouteHandoff;
  const corpusBatchResumeChatContextRequest = asRecord(
    corpusBatchResumeHandoff.chat_context_request ||
      corpusBatchUploadManifest.chat_context_request ||
      corpusBatchOperationalHandoff.chat_context_request ||
      corpusBatchPreparedRouteHandoff.chatContextRequest,
  );
  const corpusBatchResumeSuggestedQuestions = (
    asArray(corpusBatchPreparedRouteHandoff.suggestedQuestions).length > 0
      ? asArray(corpusBatchPreparedRouteHandoff.suggestedQuestions)
      : asArray(corpusBatchResumeHandoff.suggested_questions)
  ).filter(
    (question): question is Record<string, unknown> =>
      Boolean(question) &&
      typeof question === "object" &&
      !Array.isArray(question),
  );
  const corpusBatchHandoffStatus = asString(
    corpusBatchOperationalHandoff.status,
  );
  const corpusBatchHandoffLabel = asString(corpusBatchOperationalHandoff.label);
  const corpusBatchHandoffMessage = asString(
    corpusBatchOperationalHandoff.message,
  );
  const corpusBatchHandoffNextStep = asString(
    corpusBatchOperationalHandoff.next_step,
  );
  const corpusBatchTraceReady = asNumber(
    corpusBatchOperationalHandoff.trace_ready,
  );
  const corpusBatchChatReady = asNumber(
    corpusBatchOperationalHandoff.chat_ready,
  );
  const corpusBatchRemoteWaiting = asNumber(
    corpusBatchOperationalHandoff.remote_waiting,
  );
  const corpusBatchDurationMs = asNumber(
    corpusBatchUploadRecord.duration_ms ??
      corpusBatchExecutionRecord.duration_ms,
  );
  const corpusBatchThroughput =
    corpusBatchUploadRecord.throughput_files_per_second ??
    corpusBatchExecutionRecord.throughput_files_per_second;
  const corpusBatchMaxConcurrency = asNumber(
    corpusBatchUploadRecord.max_concurrency ??
      corpusBatchUploadRecord.concurrency ??
      corpusBatchExecutionRecord.max_concurrency,
  );
  const corpusBatchPeakConcurrency = asNumber(
    corpusBatchUploadRecord.peak_concurrency ??
      corpusBatchExecutionRecord.peak_concurrency,
  );
  const corpusBatchEvidenceTotals = asRecord(
    corpusBatchSummaryRecord.evidence_totals,
  );
  const corpusBatchGraphOverview = asRecord(
    corpusBatchSummaryRecord.graph_overview,
  );
  const corpusBatchGraphRelationshipCount = asNumber(
    corpusBatchGraphOverview.relationship_count,
  );
  const corpusBatchGraphSharedEntityCount = asNumber(
    corpusBatchGraphOverview.shared_entity_count,
  );
  const corpusBatchGraphBridgeEntityCount = asNumber(
    corpusBatchGraphOverview.bridge_entity_count,
  );
  const corpusBatchGraphPathCount = asNumber(
    corpusBatchGraphOverview.cross_document_path_count,
  );
  const corpusBatchGraphTopRelationships = asArray(
    corpusBatchGraphOverview.top_relationships,
  ).filter(
    (relationship): relationship is Record<string, unknown> =>
      Boolean(relationship) &&
      typeof relationship === "object" &&
      !Array.isArray(relationship),
  );
  const corpusBatchGraphSharedEntities = asArray(
    corpusBatchGraphOverview.shared_entities,
  ).filter(
    (entity): entity is Record<string, unknown> =>
      Boolean(entity) && typeof entity === "object" && !Array.isArray(entity),
  );
  const corpusBatchGraphBridgeEntities = asArray(
    corpusBatchGraphOverview.bridge_entities,
  ).filter(
    (entity): entity is Record<string, unknown> =>
      Boolean(entity) && typeof entity === "object" && !Array.isArray(entity),
  );
  const corpusBatchGraphPaths = asArray(
    corpusBatchGraphOverview.cross_document_paths,
  ).filter(
    (path): path is Record<string, unknown> =>
      Boolean(path) && typeof path === "object" && !Array.isArray(path),
  );
  const corpusBatchGraphQuestions = buildGraphRAGBatchQuestions(
    {
      filenames: corpusBatchFilenames,
      overview: corpusBatchGraphOverview,
    },
    5,
  );
  const corpusBatchPayloadKinds = Object.entries(
    asRecord(corpusBatchSummaryRecord.payload_kinds),
  )
    .map(([kind, count]) => ({ kind, count: asNumber(count) }))
    .filter(({ count }) => count > 0);
  const corpusBatchInteractionModes = Object.entries(
    asRecord(
      corpusBatchUploadManifest.interaction_mode_counts ||
        corpusBatchSummaryRecord.interaction_mode_counts,
    ),
  )
    .map(([mode, count]) => ({ mode, count: asNumber(count) }))
    .filter(({ count }) => count > 0);
  const corpusBatchInteractionRehearsalTotals = asRecord(
    corpusBatchUploadManifest.interaction_rehearsal_totals ||
      corpusBatchSummaryRecord.interaction_rehearsal_totals,
  );
  const corpusBatchManifestRehearsals = asNumber(
    corpusBatchUploadManifest.retrieval_rehearsals ??
      corpusBatchInteractionRehearsalTotals.retrieval_rehearsals,
  );
  const corpusBatchManifestRehearsedChunks = asNumber(
    corpusBatchUploadManifest.rehearsed_chunk_matches ??
      corpusBatchInteractionRehearsalTotals.chunk_matches,
  );
  const corpusBatchManifestRehearsedAnchors = asNumber(
    corpusBatchUploadManifest.rehearsed_anchor_matches ??
      corpusBatchInteractionRehearsalTotals.anchor_matches,
  );
  const corpusBatchRemoteSubmitted = asNumber(
    corpusBatchSummaryRecord.remote_submitted,
  );
  const corpusBatchLocalPreviewOnly = asNumber(
    corpusBatchSummaryRecord.local_preview_only,
  );
  const corpusBatchEndpointFallbacks = corpusBatchManifestRows.filter(
    (row) => row.endpoint_fallback_used === true,
  ).length;
  const corpusBatchEndpointAttemptTotal = corpusBatchManifestRows.reduce(
    (total, row) => total + asNumber(row.endpoint_attempt_count),
    0,
  );
  const corpusBatchDeploymentBlocked = corpusBatchManifestRows.filter((row) => {
    const status = asString(row.deployment_status).toLowerCase();
    return [
      "remote_ingest_failed",
      "health_unavailable",
      "health_unhealthy",
      "preflight_blocked",
      "not_configured",
      "disabled",
    ].includes(status);
  }).length;
  const corpusBatchEvidenceChips = [
    {
      label: "archives",
      value: positiveCount(corpusBatchEvidenceTotals.archive_entries),
    },
    {
      label: "attachments",
      value: positiveCount(corpusBatchEvidenceTotals.attachments),
    },
    {
      label: "OCR",
      value: positiveCount(corpusBatchEvidenceTotals.ocr_blocks),
    },
    {
      label: "layout regions",
      value: positiveCount(corpusBatchEvidenceTotals.layout_elements),
    },
    {
      label: "tables",
      value: positiveCount(corpusBatchEvidenceTotals.distinct_tables),
    },
    {
      label: "table cells",
      value: positiveCount(corpusBatchEvidenceTotals.table_elements),
    },
    { label: "charts", value: positiveCount(corpusBatchEvidenceTotals.charts) },
    {
      label: "diagrams",
      value: positiveCount(corpusBatchEvidenceTotals.diagrams),
    },
    { label: "shapes", value: positiveCount(corpusBatchEvidenceTotals.shapes) },
    {
      label: "connectors",
      value: positiveCount(corpusBatchEvidenceTotals.connectors),
    },
    { label: "forms", value: positiveCount(corpusBatchEvidenceTotals.forms) },
    { label: "links", value: positiveCount(corpusBatchEvidenceTotals.links) },
    {
      label: "code blocks",
      value: positiveCount(corpusBatchEvidenceTotals.code_blocks),
    },
    {
      label: "comments",
      value: positiveCount(corpusBatchEvidenceTotals.comments),
    },
    {
      label: "revisions",
      value: positiveCount(corpusBatchEvidenceTotals.revisions),
    },
    {
      label: "speaker notes",
      value: positiveCount(corpusBatchEvidenceTotals.speaker_notes),
    },
    {
      label: "styled text",
      value: positiveCount(corpusBatchEvidenceTotals.styled_text_blocks),
    },
    {
      label: "messages",
      value: positiveCount(corpusBatchEvidenceTotals.message_body_parts),
    },
    { label: "media", value: positiveCount(corpusBatchEvidenceTotals.media) },
    { label: "images", value: positiveCount(corpusBatchEvidenceTotals.images) },
    {
      label: "pivot tables",
      value: positiveCount(corpusBatchEvidenceTotals.pivot_tables),
    },
    {
      label: "hidden slides",
      value: positiveCount(corpusBatchEvidenceTotals.hidden_slides),
    },
    {
      label: "hidden sheets",
      value: positiveCount(corpusBatchEvidenceTotals.hidden_sheets),
    },
    {
      label: "legacy streams",
      value: positiveCount(corpusBatchEvidenceTotals.legacy_office_streams),
    },
    {
      label: "legacy text",
      value: positiveCount(corpusBatchEvidenceTotals.legacy_office_text_blocks),
    },
    {
      label: "graph edges",
      value: positiveCount(corpusBatchEvidenceTotals.graph_relationships),
    },
    {
      label: "semantic edges",
      value: positiveCount(corpusBatchEvidenceTotals.semantic_relationships),
    },
    {
      label: "table facts",
      value: positiveCount(corpusBatchEvidenceTotals.table_facts),
    },
    {
      label: "chart edges",
      value: positiveCount(corpusBatchEvidenceTotals.chart_relationships),
    },
    {
      label: "layout edges",
      value: positiveCount(corpusBatchEvidenceTotals.layout_relationships),
    },
    {
      label: "link edges",
      value: positiveCount(corpusBatchEvidenceTotals.link_relationships),
    },
    {
      label: "code edges",
      value: positiveCount(corpusBatchEvidenceTotals.code_block_relationships),
    },
    {
      label: "diagram edges",
      value: positiveCount(corpusBatchEvidenceTotals.diagram_relationships),
    },
    {
      label: "style edges",
      value: positiveCount(corpusBatchEvidenceTotals.styled_text_relationships),
    },
    {
      label: "attachment edges",
      value: positiveCount(corpusBatchEvidenceTotals.attachment_relationships),
    },
    {
      label: "archive edges",
      value: positiveCount(corpusBatchEvidenceTotals.archive_relationships),
    },
    {
      label: "OCR edges",
      value: positiveCount(corpusBatchEvidenceTotals.ocr_relationships),
    },
    {
      label: "notes edges",
      value: positiveCount(
        corpusBatchEvidenceTotals.speaker_notes_relationships,
      ),
    },
  ].filter(
    (chip): chip is { label: string; value: number } => chip.value !== null,
  );
  const corpusUploadFileCount = corpusUploadFiles.length;
  const corpusUploadManifest = buildGraphRAGUploadManifest(corpusUploadFiles, {
    buildKnowledgeGraph: corpusBuildKnowledgeGraph,
    useSemanticChunking: corpusUseSemanticChunking,
  });
  const corpusUploadTotalBytes = corpusUploadManifest.totalBytes;
  const corpusSelectedFileNames = corpusUploadFiles
    .map((file) => file.name)
    .join(", ");
  const corpusUploadPreviewFiles = corpusUploadFiles.slice(0, 3);
  const corpusUploadOverflowCount = Math.max(
    0,
    corpusUploadFiles.length - corpusUploadPreviewFiles.length,
  );
  const corpusHasMultipleFiles = corpusUploadFileCount > 1;
  const corpusUploadInProgress = corpusLoading && corpusUploadFileCount > 0;
  const corpusUploadActive = corpusUploadFileCount > 0;
  const corpusComplexFormatCount = corpusUploadManifest.richFormatCount;
  const corpusUploadModeLabel = corpusUploadManifest.modeLabel;
  const corpusUploadModeDetail =
    corpusUploadFileCount > 0
      ? corpusUploadManifest.sizeAndKindSummary
      : "No files selected";
  const corpusIntakeChips = [
    {
      label: corpusUploadModeLabel,
      color: accentColors.primary,
      active: true,
    },
    {
      label: corpusUseSemanticChunking ? "Semantic chunks" : "Standard chunks",
      color: accentColors.info,
      active: true,
    },
    {
      label: corpusUploadManifest.chatModeLabel,
      color: corpusBuildKnowledgeGraph
        ? accentColors.success
        : accentColors.warning,
      active: true,
    },
    {
      label:
        corpusUploadManifest.richFormatLabel ||
        `${formatNumber(corpusComplexFormatCount)} rich files`,
      color: accentColors.purple,
      active: corpusComplexFormatCount > 0,
    },
  ].filter((chip) => chip.active);
  const corpusProcessButtonLabel = corpusUploadInProgress
    ? corpusHasMultipleFiles
      ? `Processing ${formatNumber(corpusUploadFileCount)} documents`
      : "Processing"
    : corpusHasMultipleFiles
      ? `Process ${formatNumber(corpusUploadFileCount)} documents`
      : "Process";
  const selectedCorpusBatchRun =
    corpusBatchRuns.find(
      (run) => asString(run.run_id) === selectedCorpusPipelineRunId,
    ) || corpusBatchRuns[0];
  const selectedCorpusBatchRunIndex = selectedCorpusBatchRun
    ? Math.max(
        0,
        corpusBatchRuns.findIndex(
          (run) =>
            asString(run.run_id) === asString(selectedCorpusBatchRun.run_id),
        ),
      )
    : -1;
  const selectedCorpusBatchDocument = asRecord(
    selectedCorpusBatchRun?.document,
  );
  const selectedCorpusBatchFilename = asString(
    selectedCorpusBatchDocument.filename,
    selectedCorpusBatchRunIndex >= 0
      ? `File ${selectedCorpusBatchRunIndex + 1}`
      : "Batch file",
  );
  const corpusBatchDisplayFileCount =
    corpusBatchProcessed ||
    corpusBatchTotal ||
    corpusBatchFilenames.length ||
    corpusBatchRunIds.length;
  const corpusBatchDocumentSetTitle =
    corpusBatchDisplayFileCount > 1
      ? `${formatNumber(corpusBatchDisplayFileCount)} documents`
      : selectedCorpusBatchFilename;
  const selectedRecentPipelineRun =
    pipelineRuns.find(
      (run) => asString(run.run_id) === selectedCorpusPipelineRunId,
    ) || (selectedCorpusPipelineRunId ? undefined : pipelineRuns[0]);
  const corpusPipelineResult = selectGraphRAGWorkbenchPipelineResult({
    corpusResult,
    selectedPipelineRunId: selectedCorpusPipelineRunId,
    selectedBatchRun: selectedCorpusBatchRun,
    selectedRecentRun: selectedRecentPipelineRun,
  });
  const corpusPipelineTitle = corpusResultRecord.pipeline_preview
    ? "Latest Corpus Preview"
    : corpusResultRecord.pipeline_run
      ? "Loaded Document Import"
      : corpusResultRecord.pipeline_batch_upload
        ? `Batch Import: ${selectedCorpusBatchFilename}`
        : selectedRecentPipelineRun
          ? "Recent Document Import"
          : "Latest Corpus Upload";
  const handleAskChatAcrossBatch = (
    promptOverride?: string,
    chatContextOverride: GraphRAGChatContextOverride = {},
    displayPromptOverride?: string,
  ) => {
    const preparedRunIds = asStringArray(
      corpusBatchResumeChatContextRequest.pipeline_run_ids,
    );
    const routeRunIds = asStringArray(corpusBatchPreparedRouteHandoff.runIds);
    const batchRunIds =
      preparedRunIds.length > 0
        ? preparedRunIds
        : routeRunIds.length > 0
          ? routeRunIds
          : corpusBatchRunIds;
    if (batchRunIds.length === 0) return;
    const fileText =
      corpusBatchFilenames.length > 0
        ? corpusBatchFilenames.slice(0, 6).join(", ")
        : `${batchRunIds.length} uploaded files`;
    const fallbackPrompt = buildGraphRAGBatchPrompt({
      filenames:
        corpusBatchFilenames.length > 0 ? corpusBatchFilenames : [fileText],
      overview: corpusBatchGraphOverview,
    });
    const prompt =
      promptOverride ||
      asString(corpusBatchPreparedRouteHandoff.prompt) ||
      asString(corpusBatchResumeChatContextRequest.query) ||
      fallbackPrompt;
    const shouldAutoSubmit = Boolean(promptOverride);
    const visiblePrompt =
      shouldAutoSubmit
        ? displayPromptOverride ||
          buildGraphRAGTraceDisplayPrompt(fileText, prompt)
        : "";
    const contextQuery = asString(
      chatContextOverride.query,
      buildGraphRAGTraceContextQuery(fileText, prompt),
    );
    const domain = asString(
      corpusBatchResumeChatContextRequest.domain,
      asString(
        corpusBatchPreparedRouteHandoff.domain,
        corpusDomain.trim() || "general",
      ),
    );

    navigate(getRoutePath("chat"), {
      state: {
        graphragHandoff: {
          ...corpusBatchPreparedRouteHandoff,
          runIds: batchRunIds,
          filename: asString(
            corpusBatchPreparedRouteHandoff.filename,
            `${batchRunIds.length} file GraphRAG batch`,
          ),
          domain,
          prompt: shouldAutoSubmit ? prompt : "",
          displayPrompt: visiblePrompt,
          autoSubmit: shouldAutoSubmit,
          chatContextEndpoint: asString(
            corpusBatchResumeHandoff.chat_context_endpoint,
            asString(
              corpusBatchPreparedRouteHandoff.chatContextEndpoint,
              "/api/v1/graphrag/chat-context-v2",
            ),
          ),
          chatContextRequest: {
            ...corpusBatchResumeChatContextRequest,
            ...chatContextOverride,
            query: contextQuery,
            domain,
            pipeline_run_ids: batchRunIds,
            include_graph:
              corpusBatchResumeChatContextRequest.include_graph === false
                ? false
                : true,
            use_local_pipeline_context:
              corpusBatchResumeChatContextRequest.use_local_pipeline_context ===
              false
                ? false
                : true,
            include_local_pipeline:
              corpusBatchResumeChatContextRequest.include_local_pipeline ===
              false
                ? false
                : true,
          },
          contextQuery,
          suggestedQuestions:
            corpusBatchResumeSuggestedQuestions.length > 0
              ? corpusBatchResumeSuggestedQuestions
              : corpusBatchGraphQuestions,
        },
      },
    });
  };
  const handleAskChatAcrossRecent = (
    promptOverride?: string,
    chatContextOverride: GraphRAGChatContextOverride = {},
  ) => {
    const recentRunIds = chatContextPipelineRunIds;
    if (recentRunIds.length === 0) return;

    const filenames = pipelineRuns
      .filter((run) => recentRunIds.includes(asString(run.run_id)))
      .map((run) => asString(asRecord(run.document).filename))
      .filter(Boolean);
    const prompt =
      promptOverride ||
      buildGraphRAGBatchPrompt({
        filenames,
        overview: {
          relationship_count: pipelineRunTotals.relationships,
        },
      });
    const shouldAutoSubmit = Boolean(promptOverride);
    const fileText =
      filenames.length > 0
        ? filenames.slice(0, 6).join(", ")
        : `${recentRunIds.length} recent document import${recentRunIds.length === 1 ? "" : "s"}`;
    const contextQuery = asString(
      chatContextOverride.query,
      buildGraphRAGTraceContextQuery(fileText, prompt),
    );

    navigate(getRoutePath("chat"), {
      state: {
        graphragHandoff: {
          runIds: recentRunIds,
          filename:
            filenames.length > 0
              ? `${filenames.length} recent GraphRAG document${filenames.length === 1 ? "" : "s"}`
              : `${recentRunIds.length} recent document import${recentRunIds.length === 1 ? "" : "s"}`,
          domain: corpusDomain.trim() || "general",
          prompt: shouldAutoSubmit ? prompt : "",
          displayPrompt: shouldAutoSubmit
            ? buildGraphRAGTraceDisplayPrompt(fileText, prompt)
            : "",
          contextQuery,
          chatContextEndpoint: "/api/v1/graphrag/chat-context-v2",
          autoSubmit: shouldAutoSubmit,
          chatContextRequest: {
            ...chatContextOverride,
            query: contextQuery,
            domain: corpusDomain.trim() || "general",
            mode: "v2",
            top_k: queryTopK,
            rag_enabled: true,
            rag_search_mode: "domain",
            pipeline_run_ids: recentRunIds,
            use_local_pipeline_context: true,
            include_local_pipeline: true,
            include_graph: true,
          },
          suggestedQuestions: buildGraphRAGBatchQuestions(
            {
              filenames,
              overview: {
                relationship_count: pipelineRunTotals.relationships,
              },
            },
            4,
          ),
          recommendedMode: "local_pipeline_graph",
        },
      },
    });
  };
  const buildTraceSummaryPromptForRun = (run: Record<string, unknown>): string => {
    const document = asRecord(run.document);
    return buildGraphRAGTracePrompt(
      asString(document.filename, activeCorpusFilename || corpusPipelineTitle || "Document"),
      asRecord(run.summary),
      asRecord(run.document_profile),
      asArray(run.stages).map((stage) => asRecord(asRecord(stage).sample_outputs)),
      run.artifacts,
    );
  };
  const normalizeRunQuestion = (
    question: Record<string, unknown>,
    index: number,
    run: Record<string, unknown>,
  ): GraphRAGBatchQuestion => {
    const kind = asString(question.kind);
    const normalizedKind =
      kind === "summary" ||
      kind === "bridge" ||
      kind === "path" ||
      kind === "relationship" ||
      kind === "shared_entity"
        ? kind
        : "relationship";
    return {
      id: asString(question.id, `selected-question-${index}`),
      label:
        normalizedKind === "summary"
          ? "Summarize document"
          : asString(
              question.label,
              asString(question.kind, "Ask about evidence"),
            ),
      prompt:
        normalizedKind === "summary"
          ? buildTraceSummaryPromptForRun(run)
          : asString(question.prompt),
      kind: normalizedKind,
      source: asString(question.source) || undefined,
      target: asString(question.target) || undefined,
      bridge: asString(question.bridge) || undefined,
      relation: asString(question.relation) || undefined,
      focus: asString(question.focus) || undefined,
    };
  };
  const handleAskChatFromPipelineRun = (
    run: Record<string, unknown>,
    promptOverride?: string,
    chatContextOverride: GraphRAGChatContextOverride = {},
    displayPromptOverride?: string,
  ) => {
    const runId = asString(run.run_id);
    if (!runId) return;

    const document = asRecord(run.document);
    const resumeHandoff = asRecord(run.resume_handoff);
    const routeState = asRecord(resumeHandoff.route_state);
    const routeHandoff = asRecord(routeState.graphragHandoff);
    const chatContextRequest = asRecord(resumeHandoff.chat_context_request);
    const starterPrompts = asArray(resumeHandoff.starter_prompts).filter(
      (prompt): prompt is Record<string, unknown> =>
        Boolean(prompt) && typeof prompt === "object" && !Array.isArray(prompt),
    );
    const suggestedQuestions = asArray(
      resumeHandoff.suggested_questions,
    ).filter(
      (question): question is Record<string, unknown> =>
        Boolean(question) &&
        typeof question === "object" &&
        !Array.isArray(question),
    );
    const upgradedSummaryPrompt = buildTraceSummaryPromptForRun(run);
    const routePrompt = asString(routeHandoff.prompt);
    const starterPrompt = asString(starterPrompts[0]?.prompt);
    const prompt =
      promptOverride ||
      (routePrompt && /summar/i.test(routePrompt) ? upgradedSummaryPrompt : routePrompt) ||
      (starterPrompt && /summar/i.test(starterPrompt) ? upgradedSummaryPrompt : starterPrompt) ||
      `Use the recent document import for ${asString(document.filename, "this document")} to summarize key evidence, relationships, and provenance.`;
    const shouldAutoSubmit = Boolean(promptOverride);
    const filename = asString(
      routeHandoff.filename,
      asString(document.filename, "GraphRAG pipeline run"),
    );
    const visiblePrompt =
      shouldAutoSubmit
        ? displayPromptOverride ||
          buildGraphRAGTraceDisplayPrompt(filename, prompt)
        : "";
    const domain = asString(
      routeHandoff.domain,
      asString(document.domain, corpusDomain.trim() || "general"),
    );
    const contextQuery = asString(
      chatContextOverride.query,
      buildGraphRAGTraceContextQuery(filename, prompt),
    );

    navigate(getRoutePath("chat"), {
      state: {
        graphragHandoff: {
          ...routeHandoff,
          runId,
          runIds: [runId],
          filename,
          domain,
          prompt: shouldAutoSubmit ? prompt : "",
          displayPrompt: visiblePrompt,
          contextQuery,
          autoSubmit: shouldAutoSubmit,
          chatContextEndpoint: asString(
            resumeHandoff.chat_context_endpoint,
            "/api/v1/graphrag/chat-context-v2",
          ),
          chatContextRequest: {
            ...chatContextRequest,
            ...chatContextOverride,
            query: contextQuery,
            domain: asString(
              chatContextRequest.domain,
              domain,
            ),
            pipeline_run_ids: [runId],
            include_graph:
              chatContextRequest.include_graph === false ? false : true,
            use_local_pipeline_context:
              chatContextRequest.use_local_pipeline_context === false
                ? false
                : true,
            include_local_pipeline:
              chatContextRequest.include_local_pipeline === false
                ? false
                : true,
          },
          suggestedQuestions:
            suggestedQuestions.length > 0
              ? suggestedQuestions.map((question, index) =>
                  normalizeRunQuestion(question, index, run),
                )
              : starterPrompts
                  .map((starterPrompt, index) =>
                    normalizeRunQuestion(starterPrompt, index, run),
                  )
                  .filter((starterPrompt) => starterPrompt.prompt),
        },
      },
    });
  };
  const handleAskChatFromPipelineRuns = (
    runs: Record<string, unknown>[],
    promptOverride?: string,
    chatContextOverride: GraphRAGChatContextOverride = {},
    displayPromptOverride?: string,
  ) => {
    const usableRuns = runs.filter((run) => asString(run.run_id));
    if (usableRuns.length === 0) return;
    if (usableRuns.length > 1 && pipelineRunsShareDocument(usableRuns)) {
      const preferredRun = preferredPipelineRunForDocumentChat(usableRuns);
      if (preferredRun) {
        handleAskChatFromPipelineRun(
          preferredRun,
          promptOverride,
          chatContextOverride,
          displayPromptOverride,
        );
        return;
      }
    }
    if (usableRuns.length === 1) {
      handleAskChatFromPipelineRun(
        usableRuns[0],
        promptOverride,
        chatContextOverride,
        displayPromptOverride,
      );
      return;
    }

    const runIds = uniqueStringList(
      usableRuns.map((run) => asString(run.run_id)),
    );
    const firstDocument = asRecord(usableRuns[0].document);
    const filename = asString(firstDocument.filename, "Selected document");
    const domain = asString(
      firstDocument.domain,
      corpusDomain.trim() || "general",
    );
    const totals = usableRuns.reduce<{
      chunks: number;
      entities: number;
      relationships: number;
    }>(
      (accumulator, run) => {
        const summary = asRecord(run.summary);
        return {
          chunks: accumulator.chunks + asNumber(summary.chunk_count),
          entities:
            accumulator.entities + asNumber(summary.entity_candidate_count),
          relationships:
            accumulator.relationships +
            asNumber(summary.relationship_candidate_count),
        };
      },
      { chunks: 0, entities: 0, relationships: 0 },
    );
    const prompt =
      promptOverride ||
      buildGraphRAGBatchPrompt({
        filenames: [filename],
        overview: {
          relationship_count: totals.relationships,
          entity_count: totals.entities,
          chunk_count: totals.chunks,
        },
      });
    const shouldAutoSubmit = Boolean(promptOverride);
    const visiblePrompt =
      shouldAutoSubmit
        ? displayPromptOverride ||
          buildGraphRAGTraceDisplayPrompt(`${filename} (${formatNumber(runIds.length)} imports)`, prompt)
        : "";
    const contextQuery = asString(
      chatContextOverride.query,
      buildGraphRAGTraceContextQuery(filename, prompt),
    );

    navigate(getRoutePath("chat"), {
      state: {
        graphragHandoff: {
          runIds,
          filename: `${filename} (${formatNumber(runIds.length)} imports)`,
          domain,
          prompt: shouldAutoSubmit ? prompt : "",
          displayPrompt: visiblePrompt,
          contextQuery,
          autoSubmit: shouldAutoSubmit,
          chatContextEndpoint: "/api/v1/graphrag/chat-context-v2",
          chatContextRequest: {
            ...chatContextOverride,
            query: contextQuery,
            domain,
            mode: "v2",
            top_k: queryTopK,
            rag_enabled: true,
            rag_search_mode: "domain",
            pipeline_run_ids: runIds,
            use_local_pipeline_context: true,
            include_local_pipeline: true,
            include_graph: true,
          },
          suggestedQuestions: buildGraphRAGBatchQuestions(
            {
              filenames: [filename],
              overview: {
                relationship_count: totals.relationships,
                entity_count: totals.entities,
                chunk_count: totals.chunks,
              },
            },
            4,
          ),
          recommendedMode: "local_pipeline_graph",
        },
      },
    });
  };

  const workflowStep = corpusUploadInProgress
    ? 1
    : corpusPipelineResult
    ? 3
    : pipelineRuns.length > 0
      ? 2
      : corpusUploadFileCount > 0
        ? 1
        : 0;
  const activeCorpusPipelineRecord = asRecord(corpusPipelineResult);
  const activeCorpusPipelineRunId = asString(activeCorpusPipelineRecord.run_id);
  const activeCorpusPipelineStatus = asString(activeCorpusPipelineRecord.status).toLowerCase();
  const activeAsyncSubmissionStatus = asString(
    asRecord(activeCorpusPipelineRecord.async_submission).status,
  ).toLowerCase();
  useEffect(() => {
    if (documentInsightView !== "processing") return;
    if (corpusUploadInProgress || corpusUploadFileCount > 0) return;
    if (stageSamplesExpanded || advancedTraceExpanded) return;
    if (!activeCorpusPipelineRunId || !pipelineRunIsTerminal(activeCorpusPipelineRecord)) return;
    if (reviewReadyAutoFocusedRunIdRef.current === activeCorpusPipelineRunId) return;

    reviewReadyAutoFocusedRunIdRef.current = activeCorpusPipelineRunId;
    setDocumentInsightView("overview");
  }, [
    activeCorpusPipelineRecord,
    activeCorpusPipelineRunId,
    advancedTraceExpanded,
    corpusUploadFileCount,
    corpusUploadInProgress,
    documentInsightView,
    stageSamplesExpanded,
  ]);
  const activeCorpusPipelineSummary = asRecord(activeCorpusPipelineRecord.summary);
  const activeCorpusPipelineDocument = asRecord(activeCorpusPipelineRecord.document);
  const activeCorpusFilename = asString(
    activeCorpusPipelineDocument.filename,
    corpusHasBatchResult ? selectedCorpusBatchFilename : "Selected document",
  );
  const activeCorpusDocumentTitle =
    activeCorpusFilename === "Selected document"
      ? routeSelectedGraphRAGFilename || activeCorpusFilename
      : activeCorpusFilename;
  const activeLocalGraph = asRecord(activeCorpusPipelineRecord.local_graph);
  const activeLocalGraphEdges = asArray(activeLocalGraph.edges).filter(
    (edge): edge is Record<string, unknown> =>
      Boolean(edge) && typeof edge === "object" && !Array.isArray(edge),
  );
  const allActiveLocalGraphRows: ActiveGraphRow[] = activeLocalGraphEdges
    .map((edge, index) => {
      const relation = asString(
        edge.label,
        asString(edge.relation, asString(edge.type, "related")),
      );
      const normalizedRelation = relation.toLowerCase();
      const source = asString(edge.source, "Source");
      const target = asString(edge.target, "Target");
      const family = graphEdgeEvidenceFamily(edge);
      const evidence = asString(edge.evidence);
      const metadata = asRecord(edge.metadata);
      const properties = asRecord(edge.properties);
      const evidenceTitle = cleanGraphRAGTraceSampleText(
        evidence ||
          asString(metadata.provenance_summary) ||
          asString(properties.provenance_summary) ||
          `${formatPipelineRelationTitleLabel(relation)}: ${formatPipelineGraphPairLabel(source, target)}`,
        180,
      );
      const weight = asNumber(
        edge.weight ?? properties.weight ?? properties.confidence,
        1,
      );
      const structural =
        normalizedRelation.startsWith("contains") ||
        normalizedRelation.startsWith("has_");
      const weakMention = [
        "co_occurs_with",
        "co-occurs-with",
        "co occurs with",
        "related_to",
      ].includes(normalizedRelation);
      const lowValue =
        weakMention &&
        (
          isLowValueGraphEndpointLabel(source) ||
          isLowValueGraphEndpointLabel(target) ||
          isLowValueGraphRAGTraceSampleText(`${source} -> ${target}`)
        );

      return {
        key: `local-edge-${source}-${target}-${relation}-${index}`,
        label: formatGraphRelationLabel(relation),
        text: formatPipelineGraphPairLabel(source, target),
        kind: "relationship" as const,
        relation,
        source,
        target,
        title: evidenceTitle,
        color:
          family === "semantic"
            ? accentColors.success
            : family === "structural"
              ? accentColors.slate
              : accentColors.info,
        family,
        structural,
        weakMention,
        lowValue,
        priority:
          graphRelationUsefulnessScore({
            relation,
            source,
            target,
            family,
            structural,
            weakMention,
          }) + weight - (lowValue ? 120 : 0),
      };
    })
    .filter((row) => row.text !== "Source to Target")
    .sort((left, right) => right.priority - left.priority);
  const usefulActiveLocalGraphRows = allActiveLocalGraphRows.filter(
    (row) => !row.lowValue,
  );
  const hasPrimaryLocalGraphRows = usefulActiveLocalGraphRows.some(
    (row) => row.family === "semantic" && !row.structural && !row.weakMention,
  );
  const activeLocalGraphRows = (
    hasPrimaryLocalGraphRows
      ? usefulActiveLocalGraphRows.filter((row) => !row.weakMention && !row.structural)
      : usefulActiveLocalGraphRows
  ).slice(0, 4);
  const activeLocalEvidenceChips = Object.entries(
    activeLocalGraphEdges.reduce<Record<string, number>>((counts, edge) => {
      const family = graphEdgeEvidenceFamily(edge);
      counts[family] = (counts[family] || 0) + 1;
      return counts;
    }, {}),
  )
    .map(([family, value]) => ({
      label: `${formatLabel(family)} edges`,
      value,
    }))
    .sort(
      (left, right) =>
        graphEvidenceFamilyPriority(right.label.replace(/\s+edges$/i, "")) -
          graphEvidenceFamilyPriority(left.label.replace(/\s+edges$/i, "")) ||
        right.value - left.value,
    );
  const activeResumeHandoff = asRecord(activeCorpusPipelineRecord.resume_handoff);
  const activeRunPromptCandidates = [
    ...asArray(activeResumeHandoff.suggested_questions),
    ...asArray(activeResumeHandoff.starter_prompts),
  ]
    .filter(
      (question): question is Record<string, unknown> =>
        Boolean(question) &&
        typeof question === "object" &&
        !Array.isArray(question),
    )
    .map((question, index) =>
      normalizeRunQuestion(question, index, activeCorpusPipelineRecord),
    )
    .filter((question) => question.prompt);
  const pipelineRunTotals = pipelineRuns.reduce<{ chunks: number; entities: number; relationships: number; chatReady: number }>(
    (totals, run) => {
      const summary = asRecord(run.summary);
      const resumeHandoff = asRecord(run.resume_handoff);
      const interactionHandoff = asRecord(run.interaction_handoff);
      const status = asString(run.status).toLowerCase();
      return {
        chunks: totals.chunks + asNumber(summary.chunk_count),
        entities: totals.entities + asNumber(summary.entity_candidate_count),
        relationships:
          totals.relationships + asNumber(summary.relationship_candidate_count),
        chatReady:
          totals.chatReady +
          (resumeHandoff.ready === true ||
          interactionHandoff.ready === true ||
          PIPELINE_SUCCESS_STATUSES.has(status)
            ? 1
            : 0),
      };
    },
    { chunks: 0, entities: 0, relationships: 0, chatReady: 0 },
  );
  const normalizedCorpusRunSearch = corpusRunSearch.trim().toLowerCase();
  const visiblePipelineRuns = normalizedCorpusRunSearch
    ? pipelineRuns.filter((run) => {
        const document = asRecord(run.document);
        const summary = asRecord(run.summary);
        return [
          run.run_id,
          run.status,
          document.filename,
          document.name,
          document.domain,
          summary.chunk_count,
          summary.entity_candidate_count,
          summary.relationship_candidate_count,
        ]
          .map((value) => String(value || "").toLowerCase())
          .join(" ")
          .includes(normalizedCorpusRunSearch);
      })
    : pipelineRuns;
  const visibleDocumentRuns = groupGraphRAGDocumentPipelineRuns(
    visiblePipelineRuns,
    {
      fallbackDomain: corpusDomain.trim() || "general",
      groupAcrossDomains: true,
    },
  );
  const visibleDocumentTotals =
    summarizeGraphRAGDocumentLibrary(visibleDocumentRuns);
  const visibleDocumentChatReadyCount = visibleDocumentTotals.chatReadyCount;
  const corpusChatButtonLabel =
    chatContextPipelineRunIds.length > 1 ? "Ask all" : "Ask latest";
  const activeDocumentGroup =
    !corpusHasBatchResult &&
    !corpusResultRecord.pipeline_upload &&
    !corpusResultRecord.pipeline_preview
      ? visibleDocumentRuns.find(({ run, runs }) => {
          const activeRunId = activeCorpusPipelineRunId || selectedCorpusPipelineRunId;
          if (activeRunId) {
            return runs.some((candidate) => asString(candidate.run_id) === activeRunId);
          }
          return run === selectedRecentPipelineRun;
        })
      : undefined;
  const activeDocumentGroupDocument = asRecord(activeDocumentGroup?.run.document);
  const activeDocumentGroupFilename = asString(
    activeDocumentGroupDocument.filename,
    activeCorpusFilename,
  );
  const activeDocumentDefaultRun = activeDocumentGroup
    ? preferredPipelineRunForDocumentChat(activeDocumentGroup.runs)
    : null;
  const activeDocumentDefaultSummary = asRecord(activeDocumentDefaultRun?.summary);
  const readyDocumentReviewStats = [
    {
      label: "Chunks",
      value: corpusHasBatchResult
        ? corpusBatchRuns.reduce(
            (total, run) => total + asNumber(asRecord(run.summary).chunk_count),
            0,
          )
        : activeDocumentGroup
          ? asNumber(activeDocumentDefaultSummary.chunk_count, activeDocumentGroup.chunks)
          : asNumber(activeCorpusPipelineSummary.chunk_count),
    },
    {
      label: "Entities",
      value: corpusHasBatchResult
        ? corpusBatchRuns.reduce(
            (total, run) =>
              total + asNumber(asRecord(run.summary).entity_candidate_count),
            0,
          )
        : activeDocumentGroup
          ? asNumber(
              activeDocumentDefaultSummary.entity_candidate_count,
              activeDocumentGroup.entities,
            )
          : asNumber(activeCorpusPipelineSummary.entity_candidate_count),
    },
    {
      label: "Relations",
      value:
        corpusBatchGraphRelationshipCount ||
        (corpusHasBatchResult
          ? corpusBatchRuns.reduce(
              (total, run) =>
                total +
                asNumber(asRecord(run.summary).relationship_candidate_count),
              0,
            )
          : activeDocumentGroup
            ? asNumber(
                activeDocumentDefaultSummary.relationship_candidate_count,
                activeDocumentGroup.relationships,
              )
            : asNumber(activeCorpusPipelineSummary.relationship_candidate_count)),
    },
    {
      label: "Paths",
      value: corpusBatchGraphPathCount,
    },
  ];
  const workbenchPanelState = buildGraphRAGWorkbenchPanelState({
    uploadFileCount: corpusUploadFileCount,
    uploadInProgress: corpusUploadInProgress,
    selectedFileNames: corpusSelectedFileNames,
    uploadPrimarySummary: corpusUploadManifest.primarySummary,
    uploadPipelineStepCount: corpusUploadManifest.pipelineSteps.length,
    hasMultipleUploadFiles: corpusHasMultipleFiles,
    hasBatchResult: corpusHasBatchResult,
    batchProcessedCount: corpusBatchProcessed,
    hasLatestImport: Boolean(
      corpusResultRecord.pipeline_upload || corpusResultRecord.pipeline_preview,
    ),
    activeDocumentRunCount: activeDocumentGroup?.runCount ?? 0,
    activeDocumentTitle: corpusHasBatchResult
      ? corpusBatchDocumentSetTitle
      : activeDocumentGroup
        ? activeDocumentGroupFilename
        : activeCorpusDocumentTitle || corpusPipelineTitle,
    fallbackTitle: routeSelectedGraphRAGFilename || corpusPipelineTitle,
    readyStats: readyDocumentReviewStats,
    canAskContext: Boolean(
      corpusHasBatchResult ||
        activeDocumentGroup ||
        activeCorpusPipelineRunId ||
        chatContextPipelineRunIds.length > 0,
    ),
    formatCount: formatNumber,
  });
  const activeEvidenceTitle = workbenchPanelState.evidenceTitle;
  const activeEvidenceChipLabel = workbenchPanelState.evidenceChipLabel;
  const activeDocumentHistoryLabel = workbenchPanelState.documentHistoryLabel;
  const activePanelEyebrow = workbenchPanelState.eyebrow;
  const activePanelTitle = workbenchPanelState.title;
  const activePanelSubtitle = workbenchPanelState.subtitle;
  const activePanelDisplayEyebrow = workbenchPanelState.showDocumentInsights
    ? corpusHasBatchResult
      ? "Document set"
      : ""
    : activePanelEyebrow;
  const activePanelDisplayTitle = workbenchPanelState.showDocumentInsights
    ? activeEvidenceTitle
    : activePanelTitle;
  const activePanelReadySummary = workbenchPanelState.stats.length
    ? workbenchPanelState.stats
        .slice(0, 3)
        .map((stat) => `${formatNumber(stat.value)} ${stat.label.toLowerCase()}`)
        .join(" · ")
    : activePanelSubtitle;
  const activePanelDisplaySubtitle = workbenchPanelState.showDocumentInsights
    ? corpusHasBatchResult
      ? activePanelTitle
      : activePanelReadySummary
    : activePanelSubtitle;
  const showActiveEvidenceChip =
    !workbenchPanelState.showDocumentInsights ||
    corpusHasBatchResult ||
    activeEvidenceChipLabel === "Processing" ||
    activeEvidenceChipLabel.includes("selected");
  const activeReviewStats = workbenchPanelState.stats;
  const readyDocumentInteractionVisible =
    workbenchPanelState.showDocumentInsights &&
    (workbenchPanelState.canAsk || pipelineRuns.length > 0 || corpusHasBatchResult);
  const canCompactUploadIntake =
    readyDocumentInteractionVisible && !corpusUploadActive;
  const defaultMobileWorkspaceLeadSection: GraphRAGDocumentWorkspaceSection =
    corpusUploadActive || !readyDocumentInteractionVisible
      ? "upload"
      : corpusPipelineResult || corpusHasBatchResult
        ? "selected"
        : "documents";
  const mobileWorkspaceLeadSection =
    mobileWorkspaceSection || defaultMobileWorkspaceLeadSection;
  const mobileWorkspaceGridAreas =
    graphRAGDocumentWorkspaceMobileGridAreas(mobileWorkspaceLeadSection);
  const mobileUploadIntakeIsLead = mobileWorkspaceLeadSection === "upload";
  const compactUploadIntake = canCompactUploadIntake;
  const activeWorkspaceStepTarget = graphRAGDocumentWorkspaceActiveStepTarget({
    workflowStep,
    desktopLayout: desktopWorkspaceLayout,
    mobileLeadSection: mobileWorkspaceLeadSection,
    mobileWorkspaceSection: mobileWorkspaceSection || "",
  });
  const workflowStepTargets = graphRAGDocumentWorkspaceStepTargets(
    mobileWorkspaceLeadSection,
  );
  const workspaceSelectedFirst =
    readyDocumentInteractionVisible &&
    !corpusUploadActive &&
    mobileWorkspaceLeadSection === "selected";
  const workflowStepByTarget: Record<
    GraphRAGDocumentWorkspaceSection,
    {
      label: string;
      target: GraphRAGDocumentWorkspaceSection;
      detail: string;
      complete: boolean;
      icon: React.ReactNode;
    }
  > = {
    upload: {
      label: "Add",
      target: "upload",
      detail: corpusUploadInProgress
        ? "Processing"
        : corpusUploadFileCount > 0
          ? `${formatNumber(corpusUploadFileCount)} selected`
          : compactUploadIntake
            ? "Add more"
            : "Choose files",
      complete: corpusUploadInProgress || corpusUploadFileCount > 0,
      icon: <UploadIcon />,
    },
    documents: {
      label: "Review",
      target: "documents",
      detail: corpusUploadInProgress
        ? "Reading"
        : pipelineRuns.length > 0
          ? visibleDocumentChatReadyCount > 0
            ? `${formatNumber(visibleDocumentChatReadyCount)} ready`
            : `${formatNumber(pipelineRuns.length)} indexed`
          : "No documents",
      complete: pipelineRuns.length > 0 || corpusHasBatchResult,
      icon: <GraphIcon />,
    },
    selected: {
      label: "Ask",
      target: "selected",
      detail: corpusUploadInProgress
        ? "Waiting"
        : workbenchPanelState.canAsk
          ? "Context ready"
          : "Select source",
      complete: workbenchPanelState.canAsk,
      icon: <ChatIcon />,
    },
  };
  const workflowSteps = workflowStepTargets.map(
    (target) => workflowStepByTarget[target],
  );
  const activeEvidenceChips = (
    corpusUploadActive
      ? []
      : corpusBatchEvidenceChips.length > 0
      ? corpusBatchEvidenceChips
      : activeLocalEvidenceChips
  ).slice(0, 8);
  const activeEvidencePreviewChips = [...activeEvidenceChips]
    .sort(
      (left, right) =>
        graphEvidenceFamilyPreviewPriority(right.label) -
          graphEvidenceFamilyPreviewPriority(left.label) ||
        right.value - left.value,
    )
    .slice(0, 7);
  const activeGraphRows: ActiveGraphRow[] = dedupeActiveGraphRows([
    ...(corpusUploadActive ? [] : corpusBatchGraphPaths.slice(0, 2).map((path) => {
      const source = asString(path.source, "Source");
      const bridge = asString(path.bridge, "Bridge");
      const target = asString(path.target, "Target");
      return {
        key: `path-${asString(path.source)}-${asString(path.bridge)}-${asString(path.target)}`,
        label: "Path",
        text: formatPipelineGraphChainLabel([source, bridge, target]),
        title: asString(path.text),
        kind: "path" as const,
        relation: "Path",
        source,
        bridge,
        target,
        color: accentColors.info,
        priority: 90,
      };
    })),
    ...(corpusUploadActive ? [] : corpusBatchGraphTopRelationships.slice(0, 2).map((relationship, index) => {
      const source = asString(relationship.source, "Source");
      const relation = asString(relationship.relation, "related");
      const target = asString(relationship.target, "Target");
      return {
        key: `relationship-${asString(relationship.source)}-${asString(relationship.target)}-${index}`,
        label: formatLabel(relation),
        text: formatPipelineGraphPairLabel(source, target),
        title: `${formatLabel(relation)}: ${formatPipelineGraphPairLabel(source, target)}`,
        kind: "relationship" as const,
        relation,
        source,
        target,
        color: accentColors.success,
        priority: 86,
      };
    })),
    ...(corpusUploadActive ? [] : activeLocalGraphRows),
  ]).slice(0, 3);
  useEffect(() => {
    if (
      selectedEvidenceRowKey &&
      !activeGraphRows.some((row) => row.key === selectedEvidenceRowKey)
    ) {
      setSelectedEvidenceRowKey(null);
    }
  }, [activeGraphRows, selectedEvidenceRowKey]);
  const selectedActiveGraphRow =
    activeGraphRows.find((row) => row.key === selectedEvidenceRowKey) ||
    activeGraphRows[0];
  const evidenceRelationshipReviewVisible =
    documentInsightView === "evidence" &&
    activeGraphRows.length > 0 &&
    Boolean(selectedEvidenceRowKey);
  const selectedActiveGraphPrompt = selectedActiveGraphRow
    ? buildSelectedGraphRAGPrompt(selectedActiveGraphRow, activeEvidenceTitle)
    : "";
  const selectedActiveGraphContextOverride =
    buildSelectedGraphRAGContextOverride(selectedActiveGraphRow);
  const handleAskAboutSelectedGraphRow = () => {
    if (!selectedActiveGraphPrompt) return;

    if (corpusHasBatchResult) {
      handleAskChatAcrossBatch(
        selectedActiveGraphPrompt,
        selectedActiveGraphContextOverride,
      );
      return;
    }

    if (activeDocumentGroup) {
      handleAskChatFromPipelineRuns(
        activeDocumentGroup.runs,
        selectedActiveGraphPrompt,
        selectedActiveGraphContextOverride,
      );
      return;
    }

    if (activeCorpusPipelineRunId) {
      handleAskChatFromPipelineRun(
        activeCorpusPipelineRecord,
        selectedActiveGraphPrompt,
        selectedActiveGraphContextOverride,
      );
    }
  };
  const activeDocumentQuestionOverview = activeReviewStats.reduce<
    Record<string, number>
  >((overview, stat) => {
    if (stat.label === "Chunks") overview.chunk_count = stat.value;
    if (stat.label === "Entities") overview.entity_candidate_count = stat.value;
    if (stat.label === "Relations") {
      overview.relationship_candidate_count = stat.value;
    }
    if (stat.label === "Paths") overview.path_count = stat.value;
    return overview;
  }, {});
  const activeDocumentGroupQuestions = buildGraphRAGDocumentQuestions(
    {
      filename: activeEvidenceTitle,
      overview: activeDocumentQuestionOverview,
      graphRows: activeGraphRows.map((row) => ({
        kind: row.kind,
        label: row.label,
        relation: row.relation,
        source: row.source,
        target: row.target,
        bridge: row.bridge,
        title: row.title,
        text: row.text,
      })),
      evidenceFamilies: activeEvidenceChips,
      preferredQuestions: activeRunPromptCandidates,
    },
    3,
  );
  const rawActiveSuggestedQuestions = (
    corpusUploadActive
      ? []
      : corpusHasBatchResult
        ? corpusBatchGraphQuestions
        : activeDocumentGroupQuestions.length > 0
          ? activeDocumentGroupQuestions
          : activeRunPromptCandidates.length > 0
            ? activeRunPromptCandidates
            : activeLocalGraphRows.slice(0, 2).map((row, index): GraphRAGBatchQuestion => ({
                id: `selected-edge-question-${index}`,
                label: row.label,
                kind: row.kind === "path" ? "path" : "relationship",
                source: row.source,
                target: row.target,
                bridge: row.bridge,
                relation: row.relation,
                prompt: `Explain the evidence behind ${row.text} in ${activeEvidenceTitle}. Cite the relevant document context and graph relationship.`,
              }))
  );
  const graphFirstSuggestedQuestions = rawActiveSuggestedQuestions.filter(
    (question) =>
      question.kind !== "summary" &&
      question.focus !== "provenance" &&
      (
        question.kind === "path" ||
        question.kind === "bridge" ||
        question.kind === "shared_entity" ||
        (
          question.kind === "relationship" &&
          Boolean(question.source || question.target) &&
          ![
            "co_occurs",
            "co_occurs_with",
            "co-occurs-with",
            "co occurs with",
            "mentioned_with",
            "mentioned with",
            "related_to",
            "related to",
          ].includes(String(question.relation || "").toLowerCase())
        )
      ),
  );
  const graphFirstSuggestedQuestionIds = new Set(
    graphFirstSuggestedQuestions.map((question) => question.id),
  );
  const activeSuggestedQuestions = (
    graphFirstSuggestedQuestions.length > 0
      ? [
          ...graphFirstSuggestedQuestions,
          ...rawActiveSuggestedQuestions.filter(
            (question) => !graphFirstSuggestedQuestionIds.has(question.id),
          ),
        ]
      : rawActiveSuggestedQuestions
  ).slice(0, 2);
  const trimmedGraphChatQuestion = graphChatQuestion.trim();
  const canAskGraphChatQuestion = workbenchPanelState.canAsk;
  const handleAskGraphChatQuestion = () => {
    if (!trimmedGraphChatQuestion || !canAskGraphChatQuestion) return;

    if (corpusHasBatchResult) {
      handleAskChatAcrossBatch(trimmedGraphChatQuestion);
      return;
    }

    if (activeDocumentGroup) {
      const preferredRun =
        activeDocumentDefaultRun ||
        preferredPipelineRunForDocumentChat(activeDocumentGroup.runs);
      if (preferredRun) {
        handleAskChatFromPipelineRun(
          preferredRun,
          trimmedGraphChatQuestion,
        );
        return;
      }
      handleAskChatFromPipelineRun(
        activeDocumentGroup.run,
        trimmedGraphChatQuestion,
      );
      return;
    }

    if (activeCorpusPipelineRunId) {
      handleAskChatFromPipelineRun(
        activeCorpusPipelineRecord,
        trimmedGraphChatQuestion,
      );
      return;
    }

    handleAskChatAcrossRecent(trimmedGraphChatQuestion);
  };
  const activePipelineStages = corpusUploadInProgress
    ? []
    : asArray(activeCorpusPipelineRecord.stages).filter(
        (stage): stage is Record<string, unknown> =>
          Boolean(stage) && typeof stage === "object" && !Array.isArray(stage),
      );
  const activeStageOutputRows = activePipelineStages.map((stage, index) =>
    buildPipelineStageOutputRow(stage, index),
  );
  const activeEvidenceStageOutputRows = activeStageOutputRows.filter(
    (row) => row.outputRole === "evidence",
  );
  const activeCoreEvidenceStageOutputRows = activeEvidenceStageOutputRows.filter((row) => {
    const stageName = row.stageName.toLowerCase();
    const hasExtractedText = activeEvidenceStageOutputRows.some(
      (candidate) => candidate.stageName.toLowerCase() === "structure_extraction",
    );
    const hasRelationshipExtraction = activeEvidenceStageOutputRows.some(
      (candidate) => candidate.stageName.toLowerCase() === "graph_candidates",
    );
    const hasRetrievalOutput = activeEvidenceStageOutputRows.some((candidate) =>
      ["chunking", "local_context_cache"].includes(candidate.stageName.toLowerCase()),
    );

    if (stageName === "structured_ingest_document" && hasExtractedText) return false;
    if (stageName === "local_graph_artifact" && hasRelationshipExtraction) return false;
    if (stageName === "embedding_plan" && hasRetrievalOutput) return false;
    return true;
  });
  const activeProcessingStageOutputRows = activeCoreEvidenceStageOutputRows.length > 0
    ? activeCoreEvidenceStageOutputRows
    : activeEvidenceStageOutputRows.length > 0
      ? activeEvidenceStageOutputRows
      : activeStageOutputRows;
  const activeProcessingStageOutputKeys = new Set(
    activeProcessingStageOutputRows.map((row) => row.key),
  );
  const activeSupportStageOutputRows = activeStageOutputRows.filter(
    (row) => !activeProcessingStageOutputKeys.has(row.key),
  );
  const activePipelineLaneRows = buildPipelineLaneProgressRows(
    activePipelineStages,
    activeStageOutputRows,
  );
  const activeReadinessRun = asRecord(
    activeDocumentDefaultRun || activeCorpusPipelineRecord,
  );
  const activeReadinessRunDocument = asRecord(activeReadinessRun.document);
  const activeReadinessRunSummary = asRecord(activeReadinessRun.summary);
  const activeReadinessRunArtifacts = buildGraphRAGPipelineArtifactManifest(
    activeReadinessRun,
  );
  const activeReadinessRunOperationalHandoff = asRecord(
    activeReadinessRun.operational_handoff,
  );
  const activeReadinessRunId = asString(
    activeReadinessRun.run_id,
    activeCorpusPipelineRunId,
  );
  const activeArtifactDownloads = normalizeGraphRAGPipelineArtifactDownloads(
    activeReadinessRunArtifacts,
    {
      includeManifest: false,
      runId: activeReadinessRunId,
      documentFilename: activeEvidenceTitle,
    },
  );
  const activeRawArtifactMap = asArray(activeReadinessRunArtifacts.artifacts)
    .filter(
      (artifact): artifact is Record<string, unknown> =>
        Boolean(artifact) && typeof artifact === "object" && !Array.isArray(artifact),
    )
    .reduce<Map<string, Record<string, unknown>>>((artifacts, artifact) => {
      const kind = asString(artifact.kind).toLowerCase().replace(/[\s-]+/g, "_");
      if (kind) artifacts.set(kind, artifact);
      return artifacts;
    }, new Map());
  const activeArtifactByKind = new Map(
    activeArtifactDownloads.map((artifact) => [artifact.kind, artifact]),
  );
  const activeSourceArtifactRows = [
    "source_document",
    "structured_ingest",
    "local_graph",
    "local_context",
  ]
    .map((kind) => activeArtifactByKind.get(kind))
    .filter(
      (artifact): artifact is GraphRAGPipelineArtifactDownload =>
        Boolean(artifact && artifact.available && artifact.endpoint),
    )
    .map((artifact) => {
      const rawArtifact = activeRawArtifactMap.get(artifact.kind) || {};
      const sizeBytes = asNumber(rawArtifact.size_bytes);
      const contentLabel = formatGraphRAGArtifactContentLabel(artifact.contentType);
      const detail = [
        contentLabel,
        sizeBytes > 0 ? formatGraphRAGUploadFileSize(sizeBytes) : "",
        artifact.charCount && artifact.charCount > 0
          ? `${formatNumber(artifact.charCount)} chars`
          : "",
        artifact.filename || artifact.downloadFilename,
      ]
        .filter(Boolean)
        .join(" · ");
      return {
        ...artifact,
        actionLabel: graphRAGArtifactActionLabel(artifact),
        color: graphRAGArtifactColor(artifact.kind),
        contentLabel,
        detail,
        sizeBytes,
      };
    });
  const activeSourceArtifactAvailableInRows = activeSourceArtifactRows.some(
    (artifact) => artifact.kind === "source_document",
  );
  const activeStructuredArtifactAvailableInRows = activeSourceArtifactRows.some(
    (artifact) => artifact.kind === "structured_ingest",
  );
  const activeReadinessRunDocumentProfile = asRecord(
    activeReadinessRun.document_profile,
  );
  const activeDocumentEvidenceProfile = asRecord(
    activeReadinessRunSummary.evidence_profile ||
      activeReadinessRunDocumentProfile.evidence_profile,
  );
  const activeRichEvidenceOutputSummary =
    buildGraphRAGRichEvidenceOutputSummary(activeDocumentEvidenceProfile);
  const activePipelineOutputRows = (() => {
    if (!activeRichEvidenceOutputSummary) return activeProcessingStageOutputRows;

    const richEvidenceRow: PipelineStageOutputRow = {
      key: "rich-evidence-mix",
      label: "Rich evidence mix",
      stageName: "rich_evidence_mix",
      outputRole: "evidence",
      status: "complete",
      statusColor: accentColors.purple,
      durationMs: null,
      durationLabel: "ready",
      sampleGroups: activeRichEvidenceOutputSummary.sampleGroups,
      sampleCount: activeRichEvidenceOutputSummary.count,
      title: activeRichEvidenceOutputSummary.title,
      body: activeRichEvidenceOutputSummary.body,
      linkTarget: null,
      warningCount: 0,
      warningText: "",
    };
    const insertIndex = activeProcessingStageOutputRows.findIndex(
      (row) => row.stageName.toLowerCase() === "graph_candidates",
    );
    if (insertIndex < 0) {
      return [...activeProcessingStageOutputRows, richEvidenceRow];
    }
    return [
      ...activeProcessingStageOutputRows.slice(0, insertIndex),
      richEvidenceRow,
      ...activeProcessingStageOutputRows.slice(insertIndex),
    ];
  })();
  const activeSourceQuickCounts = asRecord(activeDocumentEvidenceProfile.quick_counts);
  const activeSourceLocationChips = [
    {
      key: "slides",
      label: "slides",
      value: asNumber(activeSourceQuickCounts.slides),
      color: accentColors.primary,
    },
    {
      key: "charts",
      label: "charts",
      value: asNumber(activeSourceQuickCounts.charts),
      color: accentColors.success,
    },
    {
      key: "tables",
      label: "tables",
      value:
        asNumber(activeSourceQuickCounts.tables) ||
        asNumber(activeSourceQuickCounts.table_elements),
      color: accentColors.info,
    },
    {
      key: "forms",
      label: "forms",
      value: asNumber(activeSourceQuickCounts.forms),
      color: accentColors.success,
    },
    {
      key: "styled_text",
      label: "styled text",
      value:
        asNumber(activeSourceQuickCounts.styled_text_blocks) ||
        asNumber(activeSourceQuickCounts.styled_text),
      color: accentColors.warning,
    },
    {
      key: "connectors",
      label: "connectors",
      value: asNumber(activeSourceQuickCounts.connectors),
      color: accentColors.purple,
    },
    {
      key: "diagrams",
      label: "diagrams",
      value: asNumber(activeSourceQuickCounts.diagrams),
      color: accentColors.purple,
    },
    {
      key: "notes",
      label: "notes",
      value: asNumber(activeSourceQuickCounts.speaker_notes),
      color: accentColors.warning,
    },
    {
      key: "links",
      label: "links",
      value: asNumber(activeSourceQuickCounts.links),
      color: accentColors.info,
    },
    {
      key: "attachments",
      label: "attachments",
      value: asNumber(activeSourceQuickCounts.attachments),
      color: accentColors.slate,
    },
  ].filter((chip) => chip.value > 0);
  const activeSourceSignalChips =
    activeSourceLocationChips.length > 0
      ? activeSourceLocationChips
      : activeEvidenceChips.slice(0, 6).map((chip) => ({
          ...chip,
          color: accentColors.info,
        }));
  const visibleActiveSourceSignalChips = activeSourceSignalChips.slice(0, 6);
  const hiddenActiveSourceSignalChipCount = Math.max(
    0,
    activeSourceSignalChips.length - visibleActiveSourceSignalChips.length,
  );
  const activeChildSourceArtifacts = asArray(
    activeDocumentEvidenceProfile.source_artifacts,
  )
    .filter(
      (artifact): artifact is Record<string, unknown> =>
        Boolean(artifact) && typeof artifact === "object" && !Array.isArray(artifact),
    )
    .map((artifact, index) => {
      const label = asString(
        artifact.filename,
        asString(artifact.label, `Source artifact ${index + 1}`),
      );
      const kind = asString(artifact.kind, "artifact");
      const sizeBytes = asNumber(artifact.size_bytes);
      const parsedChildElements = asNumber(artifact.parsed_child_elements);
      const detail = [
        formatLabel(kind),
        sizeBytes > 0 ? formatGraphRAGUploadFileSize(sizeBytes) : "",
        parsedChildElements > 0
          ? `${formatNumber(parsedChildElements)} parsed item${parsedChildElements === 1 ? "" : "s"}`
          : "",
      ]
        .filter(Boolean)
        .join(" · ");
      return {
        key: `${label}-${index}`,
        label,
        detail,
      };
    })
    .filter((artifact) => artifact.label)
    .slice(0, 3);
  const activeSourceDetailsAvailable =
    hiddenActiveSourceSignalChipCount > 0 || activeChildSourceArtifacts.length > 0;
  const activeSourceDetailItemCount =
    activeSourceSignalChips.length + activeChildSourceArtifacts.length;
  const activeSourceDocumentRawArtifact =
    activeRawArtifactMap.get("source_document") || {};
  const activeSourceDocumentSizeBytes = asNumber(
    activeReadinessRunDocument.size_bytes ??
      activeSourceDocumentRawArtifact.size_bytes,
  );
  const activeSourceDocumentContentLabel = formatGraphRAGArtifactContentLabel(
    asString(
      activeReadinessRunDocument.content_type,
      asString(activeReadinessRunDocument.extension, "Document"),
    ),
  );
  const activeSourceDocumentSummaryText = [
    activeSourceDocumentContentLabel,
    activeSourceDocumentSizeBytes > 0
      ? formatGraphRAGUploadFileSize(activeSourceDocumentSizeBytes)
      : "",
    activeSourceArtifactAvailableInRows ? "source saved" : "",
    activeStructuredArtifactAvailableInRows ? "structured text ready" : "",
  ]
    .filter(Boolean)
    .join(" · ");
  const activeDocumentReadinessSummary =
    summarizeGraphRAGDocumentPipelineRun(activeReadinessRun);
  const activeRemoteIngestStage = activePipelineStages.find(
    (stage) => asString(stage.name).toLowerCase() === "upstream_ingest",
  );
  const activeRemoteIngestStatus = asString(
    activeReadinessRunOperationalHandoff.remote_ingest_status,
    asString(
      asRecord(activeRemoteIngestStage).status,
      asString(activeReadinessRunOperationalHandoff.status),
    ),
  ).toLowerCase();
  const activeRemoteAccepted = PIPELINE_SUCCESS_STATUSES.has(activeRemoteIngestStatus);
  const activeRemoteSubmitted =
    activeReadinessRunOperationalHandoff.remote_ingest_submitted === true ||
    activeRemoteIngestStatus === "submitted" ||
    asString(activeReadinessRunOperationalHandoff.status).toLowerCase() ===
      "remote_graph_submitted";
  const activeRemoteTimedOut =
    activeRemoteIngestStatus === "timed_out" ||
    activeRemoteIngestStatus === "remote_ingest_timed_out";
  const shouldPollActivePipelineRun = Boolean(
    activeCorpusPipelineRunId &&
      (
        PIPELINE_RUNNING_STATUSES.has(activeCorpusPipelineStatus) ||
        PIPELINE_RUNNING_STATUSES.has(activeRemoteIngestStatus) ||
        PIPELINE_RUNNING_STATUSES.has(activeAsyncSubmissionStatus)
      ),
  );
  useEffect(() => {
    if (!shouldPollActivePipelineRun || !activeCorpusPipelineRunId) return;

    let cancelled = false;
    let timer: number | null = null;

    const schedule = (delayMs: number) => {
      timer = window.setTimeout(() => {
        void poll();
      }, delayMs);
    };

    const poll = async () => {
      try {
        const result = await api.getGraphRAGIngestionPipelineRun(activeCorpusPipelineRunId);
        if (cancelled) return;
        setCorpusResult((prev: unknown) => ({
          ...asRecord(prev),
          pipeline_run: result,
        }));
        setPipelineRuns((prev) =>
          [
            result,
            ...prev.filter((item) => asString(item.run_id) !== activeCorpusPipelineRunId),
          ].slice(0, 12),
        );

        const resultRecord = asRecord(result);
        const upstream = asRecord(resultRecord.upstream_ingest);
        const asyncSubmission = asRecord(resultRecord.async_submission);
        const nextStatuses = [
          asString(resultRecord.status).toLowerCase(),
          asString(upstream.status).toLowerCase(),
          asString(asyncSubmission.status).toLowerCase(),
        ];
        if (
          nextStatuses.some((status) => PIPELINE_RUNNING_STATUSES.has(status))
        ) {
          schedule(1500);
        }
      } catch (error) {
        if (!cancelled) {
          setCorpusError(getErrorMessage(error));
          schedule(3000);
        }
      }
    };

    schedule(900);

    return () => {
      cancelled = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [
    activeAsyncSubmissionStatus,
    activeCorpusPipelineRunId,
    activeCorpusPipelineStatus,
    activeRemoteIngestStatus,
    shouldPollActivePipelineRun,
  ]);
  const activeSourceArtifactAvailable =
    activeReadinessRunDocument.source_artifact_available === true ||
    asArray(activeReadinessRunArtifacts.artifacts).some((artifact) => {
      const artifactRecord = asRecord(artifact);
      return (
        asString(artifactRecord.kind).toLowerCase() === "source_document" &&
        artifactRecord.available === true
      );
    });
  const activeLocalGraphSummary = asRecord(activeReadinessRunSummary.local_graph);
  const activeLocalContextSummary = asRecord(activeReadinessRunSummary.local_context);
  const activeLocalGraphReady =
    asNumber(activeLocalGraphSummary.edge_count ?? activeLocalGraphSummary.relationship_count) > 0 ||
    activeLocalGraphEdges.length > 0 ||
    activePipelineLaneRows.some(
      (lane) =>
        lane.key === "relationships" &&
        PIPELINE_SUCCESS_STATUSES.has(lane.status.toLowerCase()),
    );
  const activePipelineCompletedStageCount = activeStageOutputRows.filter((row) =>
    PIPELINE_SUCCESS_STATUSES.has(row.status.toLowerCase()),
  ).length;
  const activeDocumentPipelineSignals = !corpusUploadActive &&
    workbenchPanelState.showDocumentInsights
      ? [
          {
            key: "chat",
            label: activeDocumentReadinessSummary.readyForChat
              ? "Ready for chat"
              : "Chat pending",
            title: activeDocumentReadinessSummary.readyForChat
              ? "Local handoff can answer with document evidence"
              : "Waiting for a chat-ready handoff",
            color: activeDocumentReadinessSummary.readyForChat
              ? accentColors.success
              : accentColors.info,
          },
          {
            key: "local-graph",
            label: activeLocalGraphReady ? "Graph indexed" : "Graph pending",
            title: activeLocalGraphReady
              ? `${formatNumber(asNumber(activeLocalGraphSummary.edge_count ?? activeLocalGraphSummary.relationship_count, activeLocalGraphEdges.length))} local relationships available`
              : "No local graph artifact detected yet",
            color: activeLocalGraphReady ? accentColors.primary : accentColors.info,
          },
          {
            key: "remote",
            label: activeRemoteTimedOut
              ? "Remote pending"
              : activeRemoteAccepted
              ? "Remote accepted"
              : activeRemoteSubmitted
                ? "Remote submitted"
                : "Local only",
            title: activeRemoteTimedOut
              ? "Remote ingest timed out, but local trace and chat context are ready"
              : activeRemoteAccepted
              ? "Remote GraphRAG API accepted the structured payload"
              : activeRemoteSubmitted
                ? "Structured payload was submitted upstream"
                : "Local trace is available before remote ingest",
            color: activeRemoteTimedOut
              ? accentColors.warning
              : activeRemoteAccepted
              ? accentColors.success
              : activeRemoteSubmitted
                ? accentColors.info
                : accentColors.warning,
          },
          {
            key: "source",
            label: activeSourceArtifactAvailable ? "Source saved" : "Text indexed",
            title: activeSourceArtifactAvailable
              ? "Original source artifact can be opened from citations"
              : `${formatNumber(asNumber(activeLocalContextSummary.chunk_count, activeDocumentReadinessSummary.chunkCount))} retrieval chunks available`,
            color: activeSourceArtifactAvailable
              ? accentColors.success
              : accentColors.slate,
          },
          {
            key: "stages",
            label:
              activePipelineOutputRows.length > 0
                ? `${formatNumber(activePipelineOutputRows.length)} outputs`
                : "No checks",
            title: activePipelineOutputRows.length > 0
              ? "Pipeline outputs are ready to inspect"
              : "No processing evidence has been recorded",
            color:
              activeStageOutputRows.length > 0 &&
              activePipelineCompletedStageCount === activeStageOutputRows.length
                ? accentColors.success
                : accentColors.info,
          },
        ]
      : [];
  const activeDocumentPrimarySignals = activeDocumentPipelineSignals
    .filter((signal) => {
      if (signal.key === "chat") return !activeDocumentReadinessSummary.readyForChat;
      if (signal.key === "local-graph") return !activeLocalGraphReady;
      if (signal.key === "source") return false;
      if (signal.key !== "remote") return false;
      return activeRemoteTimedOut || (activeRemoteSubmitted && !activeRemoteAccepted);
    })
    .slice(0, 3);
  const activePipelineCompletedLaneCount = activePipelineLaneRows.filter((lane) =>
    PIPELINE_SUCCESS_STATUSES.has(lane.status.toLowerCase()),
  ).length;
  const activePipelineSampleTotal = activePipelineLaneRows.reduce(
    (total, lane) => total + lane.sampleCount,
    0,
  );
  const activePipelineIssueRows = activePipelineLaneRows
    .filter((lane) =>
      lane.warningText ||
      lane.warningCount > 0 ||
      PIPELINE_WARNING_STATUSES.has(lane.status.toLowerCase()),
    )
    .map((lane) => ({
      key: lane.key,
      label: lane.label,
      color: lane.color,
      message:
        lane.warningText ||
        `${formatLabel(lane.status)} while preparing ${lane.label.toLowerCase()}.`,
    }));
  const activePipelineWarningTotal = activePipelineLaneRows.reduce(
    (total, lane) =>
      total +
      lane.warningCount +
      (lane.warningCount === 0 &&
      (lane.warningText || PIPELINE_WARNING_STATUSES.has(lane.status.toLowerCase()))
        ? 1
        : 0),
    0,
  );
  const activePipelineDurationTotal = activePipelineLaneRows.reduce(
    (total, lane) => total + lane.durationMs,
    0,
  );
  const activePipelinePerformance = asRecord(
    asRecord(activeCorpusPipelineRecord.summary).pipeline_performance ||
      activeCorpusPipelineRecord.pipeline_performance,
  );
  const activePipelineTotalDurationMs = asNumber(
    activePipelinePerformance.total_duration_ms ??
      activePipelinePerformance.totalDurationMs,
  );
  const activePipelineSlowestStage = asRecord(
    activePipelinePerformance.slowest_stage,
  );
  const activePipelineSlowestStageName = asString(
    activePipelineSlowestStage.name,
  );
  const activePipelineSlowestStageDurationMs = asNumber(
    activePipelineSlowestStage.duration_ms ??
      activePipelineSlowestStage.durationMs,
  );
  const activePipelineSlowestStageDisplayName =
    activePipelineSlowestStageName === "upstream_ingest"
      ? "ingest"
      : formatLabel(activePipelineSlowestStageName).replace(/^GraphRAG\s+/i, "");
  const activePipelineElapsedLabel =
    activePipelineTotalDurationMs > 0
      ? `${formatDuration(activePipelineTotalDurationMs)} total`
      : activePipelineDurationTotal > 0
        ? `${formatDuration(activePipelineDurationTotal)} total`
        : "";
  const activePipelineSlowestStageNeedsAttention =
    activePipelineWarningTotal > 0 || activePipelineSlowestStageDurationMs >= 30_000;
  const activePipelineSlowestStageLabel =
    activePipelineSlowestStageNeedsAttention &&
    activePipelineSlowestStageName &&
    activePipelineSlowestStageDurationMs > 0
      ? `${activePipelineSlowestStageDisplayName} ${formatDuration(
          activePipelineSlowestStageDurationMs,
        )}`
      : "";
  const activePipelineReadiness = buildGraphRAGDocumentReadiness({
    lanes: activePipelineLaneRows,
    evidence: activeEvidenceChips,
    questionCount: activeSuggestedQuestions.length,
  });
  const activePipelineSummaryChips = activePipelineLaneRows.length > 0
    ? [
        {
          key: "lanes",
          label: `${formatNumber(activePipelineCompletedLaneCount)}/${formatNumber(activePipelineLaneRows.length)} checks`,
          title: "Processing checks completed",
          color: activePipelineCompletedLaneCount === activePipelineLaneRows.length
            ? accentColors.success
            : accentColors.info,
        },
        activePipelineSampleTotal > 0
          ? {
              key: "samples",
              label: `${formatNumber(activePipelineSampleTotal)} evidence items`,
              title: "Representative evidence captured across processing checks",
              color: accentColors.info,
            }
          : null,
        activePipelineElapsedLabel
          ? {
              key: "elapsed",
              label: activePipelineElapsedLabel,
              title: "Total processing time",
              color: accentColors.slate,
            }
          : null,
        activePipelineSlowestStageLabel
          ? {
              key: "slowest",
              label: activePipelineSlowestStageLabel,
              title: `Slowest pipeline stage: ${activePipelineSlowestStageLabel}`,
              color: accentColors.warning,
            }
          : null,
        activePipelineWarningTotal > 0
          ? {
              key: "issues",
              label: `${formatNumber(activePipelineWarningTotal)} issue${activePipelineWarningTotal === 1 ? "" : "s"}`,
              title: "Pipeline warnings or partial stages",
              color: accentColors.warning,
            }
          : null,
      ].filter((chip): chip is { key: string; label: string; title: string; color: string } => Boolean(chip))
    : [];
  const activeStageSampleRows = activePipelineStages
    .reduce<PipelineEvidenceSampleRow[]>((rows, stage, index) => {
      const row = buildPipelineEvidenceSampleRow(stage, index);
      if (!row || rows.some((item) => item.label === row.label)) return rows;
      return [...rows, row];
    }, [])
    .sort((left, right) => left.priority - right.priority)
    .slice(0, 5);
  const activeEvidenceExampleRows = [...activeStageSampleRows]
    .sort((left, right) =>
      pipelineEvidenceExamplePriority(left) - pipelineEvidenceExamplePriority(right) ||
      left.priority - right.priority,
    )
    .slice(0, 2);
  const handleAskAboutPipelineSampleRow = (row: PipelineEvidenceSampleRow) => {
    const prompt = buildGraphRAGPipelineEvidenceSamplePrompt(row, activeEvidenceTitle);

    if (corpusHasBatchResult) {
      handleAskChatAcrossBatch(prompt);
      return;
    }

    if (activeDocumentGroup) {
      handleAskChatFromPipelineRuns(activeDocumentGroup.runs, prompt);
      return;
    }

    if (activeCorpusPipelineRunId) {
      handleAskChatFromPipelineRun(activeCorpusPipelineRecord, prompt);
    }
  };
  const evidenceInsightCount =
    activeSourceArtifactRows.length +
    activeSourceLocationChips.length +
    activeChildSourceArtifacts.length +
    activeGraphRows.length +
    activeStageSampleRows.length;
  const processingInsightCount =
    activePipelineLaneRows.length || activeStageOutputRows.length;
  const evidenceInsightAvailable = evidenceInsightCount > 0;
  const processingInsightAvailable = processingInsightCount > 0;
  const documentInsightTabs: Array<{
    value: DocumentInsightView;
    label: string;
    shortLabel?: string;
    count?: number;
    disabled?: boolean;
  }> = [
    {
      value: "overview",
      label: "Review",
      count: activeReviewStats.length + activeSuggestedQuestions.length,
    },
    {
      value: "evidence",
      label: "Evidence",
      count: evidenceInsightCount,
      disabled: !evidenceInsightAvailable,
    },
    {
      value: "processing",
      label: "Pipeline",
      count: processingInsightCount,
      disabled: !processingInsightAvailable,
    },
  ];
  useEffect(() => {
    if (corpusUploadInProgress && documentInsightView !== "overview") {
      setDocumentInsightView("overview");
      return;
    }

    if (documentInsightView === "evidence" && !evidenceInsightAvailable) {
      setDocumentInsightView("overview");
      return;
    }

    if (documentInsightView === "processing" && !processingInsightAvailable) {
      setDocumentInsightView("overview");
    }
  }, [
    corpusUploadInProgress,
    documentInsightView,
    evidenceInsightAvailable,
    processingInsightAvailable,
  ]);
  const documentChatReadyCount =
    corpusBatchChatReady ||
    corpusBatchRunIds.length ||
    visibleDocumentChatReadyCount ||
    pipelineRunTotals.chatReady;
  const documentWorkspaceStats = [
    {
      label: "Selected",
      value: corpusUploadFileCount || 0,
      detail:
        corpusUploadFileCount > 0
          ? formatFileSize(corpusUploadTotalBytes)
          : "Add files",
      color: accentColors.primary,
      icon: <UploadIcon />,
    },
    {
      label: "Docs",
      value: corpusHasBatchResult
        ? `${formatNumber(corpusBatchProcessed)}/${formatNumber(corpusBatchTotal)}`
        : visibleDocumentRuns.length,
      detail:
        visibleDocumentTotals.chunks > 0
          ? `${formatNumber(visibleDocumentTotals.chunks)} chunks`
          : "Library",
      color: accentColors.success,
      icon: <StorageIcon />,
    },
    {
      label: "Relations",
      value:
        corpusBatchGraphRelationshipCount ||
        visibleDocumentTotals.relationships ||
        0,
      detail:
        corpusBatchGraphPathCount > 0
          ? `${formatNumber(corpusBatchGraphPathCount)} paths`
          : visibleDocumentTotals.entities > 0
            ? `${formatNumber(visibleDocumentTotals.entities)} entities`
            : "Evidence",
      color: accentColors.info,
      icon: <GraphIcon />,
    },
    {
      label: "Ready",
      value: documentChatReadyCount,
      detail: documentChatReadyCount > 0
        ? "Ask now"
        : corpusPipelineResult
          ? "Context attached"
          : "No docs",
      color: accentColors.warning,
      icon: <ChatIcon />,
    },
  ];

  return (
    <PageShell sx={{ py: { xs: 0.85, md: 1.5 } }}>
      <Stack spacing={{ xs: 0.75, md: 1.5 }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "space-between",
            gap: 1,
            flexWrap: "wrap",
          }}
        >
          <Box sx={{ minWidth: 0, flex: "1 1 190px" }}>
            <Typography
              variant="h4"
              sx={{
                fontWeight: 800,
                fontSize: { xs: "1.12rem", sm: "1.6rem", md: "2rem" },
                lineHeight: 1.1,
              }}
            >
              Document Chat
            </Typography>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{
                display: { xs: "none", sm: "block" },
                mt: 0.5,
                maxWidth: 900,
              }}
            >
              Sources, relationships, and cited answers.
            </Typography>
          </Box>
          <Stack
            direction="row"
            spacing={0.75}
            alignItems="center"
            sx={{ flexShrink: 0, ml: "auto" }}
          >
            <StatusChip
              status={serviceStatusLoaded ? serviceSmokeStatus : "running"}
              label={
                serviceStatusLoaded
                  ? remoteGraphRAGReady
                    ? "Ready"
                    : "Local"
                  : "Checking"
              }
              statusConfig={STATUS_CONFIG}
              height={20}
              sx={{
                fontSize: { xs: "0.64rem", sm: "0.7rem" },
                "& .MuiChip-label": { px: { xs: 0.7, sm: 0.8 } },
              }}
            />
            <Tooltip title="Refresh GraphRAG status" describeChild>
              <span>
                <IconButton
                  aria-label="Refresh GraphRAG status"
                  onClick={loadOverview}
                  disabled={overviewLoading}
                  size="small"
                  sx={{
                    borderRadius: "8px",
                    border: "1px solid",
                    borderColor: alpha(accentColors.info, 0.32),
                    color: accentColors.info,
                    width: { xs: 32, sm: 40 },
                    height: { xs: 32, sm: 40 },
                  }}
                >
                  {overviewLoading ? (
                    <CircularProgress size={18} />
                  ) : (
                    <RefreshIcon />
                  )}
                </IconButton>
              </span>
            </Tooltip>
          </Stack>
        </Box>

        {overviewError && (
          <Alert severity="warning" variant="outlined">
            {overviewError}
          </Alert>
        )}

        {false && (
          <>
            <Grid container spacing={1}>
              <Grid item xs={12} sm={6} lg={3}>
                <MetricTile
                  label="GraphRAG"
                  value={formatLabel(String(healthRecord.status || "offline"))}
                  detail={
                    (healthRecord.message || healthRecord.url) as
                      string | undefined
                  }
                  color={graphStatusColor}
                  icon={<GraphIcon />}
                  pulse={graphStatus === "online"}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <MetricTile
                  label="Capabilities"
                  value={capabilityList.length}
                  detail={`${enabledServices} services enabled`}
                  color={accentColors.info}
                  icon={<AnalyticsIcon />}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <MetricTile
                  label="Graph Size"
                  value={`${formatNumber(graphStatsRecord.nodes)} nodes`}
                  detail={`${formatNumber(graphStatsRecord.edges)} edges`}
                  color={accentColors.primary}
                  icon={<StorageIcon />}
                />
              </Grid>
              <Grid item xs={12} sm={6} lg={3}>
                <MetricTile
                  label="Bridge"
                  value={bridgeAvailable ? "Typed + fallback" : "Typed only"}
                  detail={String(
                    bridgeRecord.base_path || "/api/v1/graphrag/upstream",
                  )}
                  color={
                    bridgeAvailable ? accentColors.success : accentColors.slate
                  }
                  icon={<ReasoningIcon />}
                />
              </Grid>
            </Grid>

            <SectionCard
              title="Service Smoke Check"
              subtitle={`${serviceWorkflowReadyCount}/${serviceWorkflowCount || serviceWorkflowEntries.length || 0} workflows ready, ${serviceReadyCount}/${serviceStatusEntries.length || 0} services reachable, ${serviceConfiguredCount} configured`}
              icon={<StorageIcon />}
              accentColor={
                serviceSmokeStatus === "ready"
                  ? accentColors.success
                  : serviceSmokeStatus === "partial"
                    ? accentColors.warning
                    : serviceSmokeStatus === "local" ||
                        serviceSmokeStatus === "disabled"
                      ? accentColors.info
                      : accentColors.rose
              }
              action={
                <StatusChip
                  status={serviceSmokeStatus}
                  label={formatLabel(serviceSmokeStatus)}
                  statusConfig={STATUS_CONFIG}
                  height={22}
                />
              }
              contentSx={{ p: 1.25, "&:last-child": { pb: 1.25 } }}
            >
              <Stack spacing={1}>
                {(serviceStatusSummary ||
                  serviceStatusNextSteps.length > 0) && (
                  <Alert
                    severity={serviceStatusSeverity}
                    variant="outlined"
                    sx={{ py: 0.5 }}
                  >
                    <Stack spacing={0.35}>
                      {(serviceOperatingModeLabel ||
                        serviceOperatingModeSummary) && (
                        <Typography variant="caption" sx={{ lineHeight: 1.35 }}>
                          {serviceOperatingModeLabel
                            ? `${serviceOperatingModeLabel}: `
                            : ""}
                          {serviceOperatingModeSummary ||
                            serviceOperatingModeAction}
                        </Typography>
                      )}
                      {serviceStatusSummary && (
                        <Typography variant="caption" sx={{ lineHeight: 1.35 }}>
                          {serviceStatusSummary}
                        </Typography>
                      )}
                      {serviceSourceSummary && (
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ lineHeight: 1.35 }}
                        >
                          {serviceSourceSummary}
                        </Typography>
                      )}
                      {serviceStatusNextSteps.slice(0, 2).map((step) => (
                        <Typography
                          key={step}
                          variant="caption"
                          sx={{ lineHeight: 1.35, fontFamily: "monospace" }}
                        >
                          {step}
                        </Typography>
                      ))}
                    </Stack>
                  </Alert>
                )}

                {serviceInteractionPathways.length > 0 && (
                  <Stack spacing={0.75}>
                    <Typography
                      variant="caption"
                      fontWeight={800}
                      color="text.secondary"
                    >
                      Interaction pathways
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: {
                          xs: "1fr",
                          md: "repeat(2, minmax(0, 1fr))",
                          xl: "repeat(4, minmax(0, 1fr))",
                        },
                        gap: 0.75,
                      }}
                    >
                      {serviceInteractionPathways.map((pathway) => {
                        const pathwayColor =
                          pathway.status === "ready"
                            ? accentColors.success
                            : pathway.status === "local"
                              ? accentColors.info
                              : pathway.status === "partial"
                                ? accentColors.warning
                                : pathway.status === "disabled" ||
                                    pathway.status === "not_configured"
                                  ? accentColors.slate
                                  : accentColors.rose;
                        const isRecommended =
                          pathway.recommended ||
                          pathway.id === serviceRecommendedPathwayId;
                        const title = [
                          pathway.summary,
                          pathway.action,
                          pathway.endpoint,
                          pathway.secondaryEndpoint,
                          ...pathway.checks,
                          ...pathway.outputs,
                        ]
                          .filter(Boolean)
                          .join(" ");

                        return (
                          <Box
                            key={pathway.id}
                            title={title}
                            sx={{
                              borderRadius: "8px",
                              border: "1px solid",
                              borderColor: alpha(
                                isRecommended
                                  ? accentColors.primary
                                  : pathwayColor,
                                isRecommended ? 0.45 : 0.26,
                              ),
                              bgcolor: alpha(
                                isRecommended
                                  ? accentColors.primary
                                  : pathwayColor,
                                pathway.ready ? 0.05 : 0.03,
                              ),
                              p: 0.85,
                              minWidth: 0,
                            }}
                          >
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 0.55,
                                minWidth: 0,
                              }}
                            >
                              <StatusChip
                                status={pathway.status}
                                label={
                                  isRecommended
                                    ? "Recommended"
                                    : formatLabel(pathway.status)
                                }
                                statusConfig={STATUS_CONFIG}
                                height={18}
                                sx={{ flexShrink: 0 }}
                              />
                              <Typography
                                variant="caption"
                                fontWeight={800}
                                noWrap
                                title={pathway.label}
                                sx={{ minWidth: 0 }}
                              >
                                {pathway.label}
                              </Typography>
                            </Box>
                            {pathway.summary && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                title={pathway.summary}
                                sx={{
                                  display: "-webkit-box",
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: "vertical",
                                  overflow: "hidden",
                                  mt: 0.45,
                                  lineHeight: 1.35,
                                }}
                              >
                                {pathway.summary}
                              </Typography>
                            )}
                            {(pathway.usesLocalPipeline ||
                              pathway.usesRemoteGraph ||
                              pathway.mode) && (
                              <Box
                                sx={{
                                  display: "flex",
                                  flexWrap: "wrap",
                                  gap: 0.35,
                                  mt: 0.55,
                                }}
                              >
                                {pathway.usesLocalPipeline && (
                                  <Chip
                                    label="local trace"
                                    size="small"
                                    variant="outlined"
                                    sx={{
                                      height: 18,
                                      fontSize: "0.62rem",
                                      borderColor: alpha(
                                        accentColors.info,
                                        0.25,
                                      ),
                                      color: "text.secondary",
                                      "& .MuiChip-label": { px: 0.65 },
                                    }}
                                  />
                                )}
                                {pathway.usesRemoteGraph && (
                                  <Chip
                                    label="remote graph"
                                    size="small"
                                    variant="outlined"
                                    sx={{
                                      height: 18,
                                      fontSize: "0.62rem",
                                      borderColor: alpha(
                                        accentColors.success,
                                        0.25,
                                      ),
                                      color: "text.secondary",
                                      "& .MuiChip-label": { px: 0.65 },
                                    }}
                                  />
                                )}
                                {pathway.mode && (
                                  <Chip
                                    label={formatLabel(pathway.mode)}
                                    size="small"
                                    variant="outlined"
                                    title={pathway.mode}
                                    sx={{
                                      height: 18,
                                      maxWidth: "100%",
                                      fontSize: "0.62rem",
                                      borderColor: alpha(pathwayColor, 0.25),
                                      color: "text.secondary",
                                      "& .MuiChip-label": {
                                        px: 0.65,
                                        overflow: "hidden",
                                        textOverflow: "ellipsis",
                                      },
                                    }}
                                  />
                                )}
                              </Box>
                            )}
                            {pathway.action && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                title={pathway.action}
                                sx={{
                                  display: "-webkit-box",
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: "vertical",
                                  overflow: "hidden",
                                  mt: 0.55,
                                  lineHeight: 1.35,
                                }}
                              >
                                {pathway.action}
                              </Typography>
                            )}
                            {(pathway.endpoint ||
                              pathway.secondaryEndpoint) && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                noWrap
                                title={[
                                  pathway.endpoint,
                                  pathway.secondaryEndpoint,
                                ]
                                  .filter(Boolean)
                                  .join(" then ")}
                                sx={{
                                  display: "block",
                                  mt: 0.5,
                                  fontFamily: "monospace",
                                  opacity: 0.82,
                                }}
                              >
                                {[pathway.endpoint, pathway.secondaryEndpoint]
                                  .filter(Boolean)
                                  .join(" then ")}
                              </Typography>
                            )}
                          </Box>
                        );
                      })}
                    </Box>
                  </Stack>
                )}

                {serviceDeploymentQuickstart.length > 0 && (
                  <Stack spacing={0.75}>
                    <Typography
                      variant="caption"
                      fontWeight={800}
                      color="text.secondary"
                    >
                      Deployment quickstart
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: {
                          xs: "1fr",
                          md: "repeat(2, minmax(0, 1fr))",
                          xl: "repeat(4, minmax(0, 1fr))",
                        },
                        gap: 0.75,
                      }}
                    >
                      {serviceDeploymentQuickstart.map((item) => {
                        const quickstartColor =
                          item.status === "ready"
                            ? accentColors.success
                            : item.status === "local"
                              ? accentColors.info
                              : item.status === "partial"
                                ? accentColors.warning
                                : item.status === "disabled" ||
                                    item.status === "not_configured"
                                  ? accentColors.slate
                                  : accentColors.rose;
                        const title = [
                          item.summary,
                          item.command,
                          item.nextStep,
                          item.endpoint,
                          ...item.envVars,
                          ...item.profiles,
                          ...item.checks,
                        ]
                          .filter(Boolean)
                          .join(" ");

                        return (
                          <Box
                            key={item.id}
                            title={title}
                            sx={{
                              borderRadius: "8px",
                              border: "1px solid",
                              borderColor: alpha(
                                quickstartColor,
                                item.ready ? 0.34 : 0.24,
                              ),
                              bgcolor: alpha(
                                quickstartColor,
                                item.ready ? 0.045 : 0.03,
                              ),
                              p: 0.85,
                              minWidth: 0,
                            }}
                          >
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 0.55,
                                minWidth: 0,
                              }}
                            >
                              <StatusChip
                                status={item.status}
                                label={formatLabel(item.status)}
                                statusConfig={STATUS_CONFIG}
                                height={18}
                                sx={{ flexShrink: 0 }}
                              />
                              <Typography
                                variant="caption"
                                fontWeight={800}
                                noWrap
                                title={item.label}
                                sx={{ minWidth: 0 }}
                              >
                                {item.label}
                              </Typography>
                            </Box>
                            {item.summary && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                title={item.summary}
                                sx={{
                                  display: "-webkit-box",
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: "vertical",
                                  overflow: "hidden",
                                  mt: 0.45,
                                  lineHeight: 1.35,
                                }}
                              >
                                {item.summary}
                              </Typography>
                            )}
                            {item.command && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                noWrap
                                title={item.command}
                                sx={{
                                  display: "block",
                                  mt: 0.5,
                                  fontFamily: "monospace",
                                  opacity: 0.86,
                                }}
                              >
                                {item.command}
                              </Typography>
                            )}
                            {(item.envVars.length > 0 ||
                              item.profiles.length > 0 ||
                              item.checks.length > 0) && (
                              <Box
                                sx={{
                                  display: "flex",
                                  flexWrap: "wrap",
                                  gap: 0.35,
                                  mt: 0.55,
                                }}
                              >
                                {item.envVars.slice(0, 2).map((envVar) => (
                                  <Chip
                                    key={`${item.id}-${envVar}`}
                                    label={envVar}
                                    size="small"
                                    variant="outlined"
                                    title={envVar}
                                    sx={{
                                      height: 18,
                                      maxWidth: "100%",
                                      fontSize: "0.62rem",
                                      borderColor: alpha(quickstartColor, 0.25),
                                      color: "text.secondary",
                                      "& .MuiChip-label": {
                                        px: 0.65,
                                        overflow: "hidden",
                                        textOverflow: "ellipsis",
                                      },
                                    }}
                                  />
                                ))}
                                {item.profiles.slice(0, 2).map((profile) => (
                                  <Chip
                                    key={`${item.id}-${profile}`}
                                    label={`profile ${profile}`}
                                    size="small"
                                    variant="outlined"
                                    title={profile}
                                    sx={{
                                      height: 18,
                                      maxWidth: "100%",
                                      fontSize: "0.62rem",
                                      borderColor: alpha(
                                        accentColors.info,
                                        0.25,
                                      ),
                                      color: "text.secondary",
                                      "& .MuiChip-label": {
                                        px: 0.65,
                                        overflow: "hidden",
                                        textOverflow: "ellipsis",
                                      },
                                    }}
                                  />
                                ))}
                                {item.checks.slice(0, 2).map((check) => (
                                  <Chip
                                    key={`${item.id}-${check}`}
                                    label={check}
                                    size="small"
                                    variant="outlined"
                                    title={check}
                                    sx={{
                                      height: 18,
                                      maxWidth: "100%",
                                      fontSize: "0.62rem",
                                      borderColor: alpha(
                                        accentColors.slate,
                                        0.24,
                                      ),
                                      color: "text.secondary",
                                      "& .MuiChip-label": {
                                        px: 0.65,
                                        overflow: "hidden",
                                        textOverflow: "ellipsis",
                                      },
                                    }}
                                  />
                                ))}
                              </Box>
                            )}
                            {(item.nextStep || item.endpoint) && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                noWrap
                                title={item.nextStep || item.endpoint}
                                sx={{
                                  display: "block",
                                  mt: 0.55,
                                  fontFamily: item.nextStep
                                    ? "inherit"
                                    : "monospace",
                                  opacity: 0.82,
                                }}
                              >
                                {item.nextStep || item.endpoint}
                              </Typography>
                            )}
                          </Box>
                        );
                      })}
                    </Box>
                  </Stack>
                )}

                {serviceContractItems.length > 0 && (
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: {
                        xs: "1fr",
                        md: "repeat(2, minmax(0, 1fr))",
                        xl: `repeat(${Math.min(4, Math.max(1, serviceContractItems.length))}, minmax(0, 1fr))`,
                      },
                      gap: 0.75,
                    }}
                  >
                    {serviceContractItems.map((item) => (
                      <Box
                        key={item.id}
                        sx={{
                          borderRadius: "8px",
                          border: "1px solid",
                          borderColor:
                            item.status === "ready"
                              ? alpha(accentColors.success, 0.3)
                              : item.status === "local"
                                ? alpha(accentColors.info, 0.3)
                                : alpha(accentColors.rose, 0.28),
                          bgcolor:
                            item.status === "ready"
                              ? alpha(accentColors.success, 0.045)
                              : item.status === "local"
                                ? alpha(accentColors.info, 0.045)
                                : alpha(accentColors.rose, 0.035),
                          p: 0.85,
                          minWidth: 0,
                        }}
                      >
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            gap: 0.65,
                            minWidth: 0,
                          }}
                        >
                          <StatusChip
                            status={item.status}
                            label={formatLabel(item.status)}
                            statusConfig={STATUS_CONFIG}
                            height={19}
                            sx={{ flexShrink: 0 }}
                          />
                          <Typography
                            variant="caption"
                            fontWeight={800}
                            noWrap
                            title={item.label}
                            sx={{ minWidth: 0 }}
                          >
                            {item.label}
                          </Typography>
                        </Box>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          noWrap
                          title={item.detail}
                          sx={{ display: "block", mt: 0.5 }}
                        >
                          {item.detail}
                        </Typography>
                        {item.meta && (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            noWrap
                            title={item.meta}
                            sx={{
                              display: "block",
                              fontFamily: "monospace",
                              opacity: 0.82,
                            }}
                          >
                            {item.meta}
                          </Typography>
                        )}
                      </Box>
                    ))}
                  </Box>
                )}

                {serviceEvidenceCapabilities.length > 0 && (
                  <Stack spacing={0.75}>
                    <Typography
                      variant="caption"
                      fontWeight={800}
                      color="text.secondary"
                    >
                      Evidence families
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: {
                          xs: "1fr",
                          sm: "repeat(2, minmax(0, 1fr))",
                          lg: "repeat(3, minmax(0, 1fr))",
                        },
                        gap: 0.75,
                      }}
                    >
                      {serviceEvidenceCapabilities.map((capability) => {
                        const familyColor = capability.ready
                          ? accentColors.success
                          : capability.status === "local"
                            ? accentColors.info
                            : accentColors.slate;
                        const meta = [
                          capability.formats.length
                            ? `${capability.formats.length} formats`
                            : "",
                          capability.relationshipTypes
                            .slice(0, 2)
                            .map(formatLabel)
                            .join(" + "),
                        ]
                          .filter(Boolean)
                          .join(" | ");
                        const detail = [
                          capability.detail,
                          capability.interactionOutputs
                            .map(formatLabel)
                            .join(", "),
                        ]
                          .filter(Boolean)
                          .join(" ");

                        return (
                          <Tooltip
                            key={capability.id}
                            title={detail || capability.label}
                          >
                            <Box
                              sx={{
                                borderRadius: "8px",
                                border: "1px solid",
                                borderColor: alpha(
                                  familyColor,
                                  capability.ready ? 0.32 : 0.22,
                                ),
                                bgcolor: alpha(
                                  familyColor,
                                  capability.ready ? 0.045 : 0.03,
                                ),
                                px: 0.85,
                                py: 0.65,
                                minWidth: 0,
                              }}
                            >
                              <Box
                                sx={{
                                  display: "flex",
                                  alignItems: "center",
                                  gap: 0.55,
                                  minWidth: 0,
                                }}
                              >
                                <StatusChip
                                  status={capability.status}
                                  label={
                                    capability.status === "local"
                                      ? "Local"
                                      : formatLabel(capability.status)
                                  }
                                  statusConfig={STATUS_CONFIG}
                                  height={18}
                                  sx={{ flexShrink: 0 }}
                                />
                                <Typography
                                  variant="caption"
                                  fontWeight={800}
                                  noWrap
                                  title={capability.label}
                                  sx={{ minWidth: 0 }}
                                >
                                  {capability.label}
                                </Typography>
                              </Box>
                              {meta && (
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  noWrap
                                  title={meta}
                                  sx={{ display: "block", mt: 0.45 }}
                                >
                                  {meta}
                                </Typography>
                              )}
                            </Box>
                          </Tooltip>
                        );
                      })}
                    </Box>
                  </Stack>
                )}

                {serviceSmokeTests.length > 0 && (
                  <Stack spacing={0.75}>
                    <Typography
                      variant="caption"
                      fontWeight={800}
                      color="text.secondary"
                    >
                      Smoke tests
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: {
                          xs: "1fr",
                          md: "repeat(2, minmax(0, 1fr))",
                          xl: "repeat(4, minmax(0, 1fr))",
                        },
                        gap: 0.75,
                      }}
                    >
                      {serviceSmokeTests.map((test) => {
                        const smokeColor =
                          test.status === "ready"
                            ? accentColors.success
                            : test.status === "local"
                              ? accentColors.info
                              : test.status === "not_configured" ||
                                  test.status === "disabled"
                                ? accentColors.slate
                                : accentColors.rose;
                        const endpointTitle = [
                          test.endpoint,
                          test.command,
                          ...test.candidates,
                          ...test.traceFields,
                        ]
                          .filter(Boolean)
                          .join(" ");
                        const endpointDisplay =
                          test.endpoint ||
                          test.candidates[0] ||
                          test.command ||
                          test.scope;

                        return (
                          <Box
                            key={test.id}
                            title={endpointTitle || test.expected}
                            sx={{
                              borderRadius: "8px",
                              border: "1px solid",
                              borderColor: alpha(
                                smokeColor,
                                test.ready ? 0.34 : 0.24,
                              ),
                              bgcolor: alpha(
                                smokeColor,
                                test.ready ? 0.045 : 0.03,
                              ),
                              p: 0.85,
                              minWidth: 0,
                            }}
                          >
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 0.55,
                                minWidth: 0,
                              }}
                            >
                              <Chip
                                label={test.method}
                                size="small"
                                sx={{
                                  height: 18,
                                  fontSize: "0.62rem",
                                  bgcolor: alpha(smokeColor, 0.12),
                                  color: smokeColor,
                                  fontWeight: 800,
                                  flexShrink: 0,
                                  "& .MuiChip-label": { px: 0.65 },
                                }}
                              />
                              <StatusChip
                                status={test.status}
                                label={formatLabel(test.status)}
                                statusConfig={STATUS_CONFIG}
                                height={18}
                                sx={{ flexShrink: 0 }}
                              />
                              <Typography
                                variant="caption"
                                fontWeight={800}
                                noWrap
                                title={test.label}
                                sx={{ minWidth: 0 }}
                              >
                                {test.label}
                              </Typography>
                            </Box>
                            {endpointDisplay && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                noWrap
                                title={endpointTitle || endpointDisplay}
                                sx={{
                                  display: "block",
                                  mt: 0.5,
                                  fontFamily: "monospace",
                                  opacity: 0.84,
                                }}
                              >
                                {endpointDisplay}
                              </Typography>
                            )}
                            {test.expected && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                title={test.expected}
                                sx={{
                                  display: "-webkit-box",
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: "vertical",
                                  overflow: "hidden",
                                  mt: 0.45,
                                  lineHeight: 1.35,
                                }}
                              >
                                {test.expected}
                              </Typography>
                            )}
                            {(test.candidates.length > 1 ||
                              test.traceFields.length > 0) && (
                              <Box
                                sx={{
                                  display: "flex",
                                  flexWrap: "wrap",
                                  gap: 0.35,
                                  mt: 0.55,
                                }}
                              >
                                {test.candidates
                                  .slice(1, 3)
                                  .map((candidate) => (
                                    <Chip
                                      key={`${test.id}-${candidate}`}
                                      label={formatCompactEndpoint(candidate)}
                                      size="small"
                                      variant="outlined"
                                      sx={{
                                        height: 18,
                                        maxWidth: "100%",
                                        fontSize: "0.62rem",
                                        borderColor: alpha(smokeColor, 0.25),
                                        color: "text.secondary",
                                      }}
                                    />
                                  ))}
                                {test.traceFields.slice(0, 2).map((field) => (
                                  <Chip
                                    key={`${test.id}-${field}`}
                                    label={field}
                                    size="small"
                                    variant="outlined"
                                    sx={{
                                      height: 18,
                                      maxWidth: "100%",
                                      fontSize: "0.62rem",
                                      borderColor: alpha(smokeColor, 0.25),
                                      color: "text.secondary",
                                    }}
                                  />
                                ))}
                              </Box>
                            )}
                          </Box>
                        );
                      })}
                    </Box>
                  </Stack>
                )}

                {serviceOperatorChecklist.length > 0 && (
                  <Stack spacing={0.75}>
                    <Typography
                      variant="caption"
                      fontWeight={800}
                      color="text.secondary"
                    >
                      Operator checklist
                    </Typography>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: {
                          xs: "1fr",
                          md: "repeat(2, minmax(0, 1fr))",
                          xl: "repeat(3, minmax(0, 1fr))",
                        },
                        gap: 0.75,
                      }}
                    >
                      {serviceOperatorChecklist.map((item) => {
                        const checklistColor =
                          item.status === "ready"
                            ? accentColors.success
                            : item.status === "local"
                              ? accentColors.info
                              : item.status === "partial"
                                ? accentColors.warning
                                : item.status === "disabled" ||
                                    item.status === "not_configured"
                                  ? accentColors.slate
                                  : accentColors.rose;
                        const title = [item.detail, item.action, item.endpoint]
                          .filter(Boolean)
                          .join(" ");

                        return (
                          <Box
                            key={item.id}
                            title={title}
                            sx={{
                              borderRadius: "8px",
                              border: "1px solid",
                              borderColor: alpha(
                                checklistColor,
                                item.ready ? 0.34 : 0.24,
                              ),
                              borderLeft: `3px solid ${alpha(checklistColor, 0.78)}`,
                              bgcolor: alpha(
                                checklistColor,
                                item.ready ? 0.045 : 0.03,
                              ),
                              p: 0.85,
                              minWidth: 0,
                            }}
                          >
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 0.65,
                                minWidth: 0,
                              }}
                            >
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ fontWeight: 800, flexShrink: 0 }}
                              >
                                {item.index + 1}
                              </Typography>
                              <StatusChip
                                status={item.status}
                                label={formatLabel(item.status)}
                                statusConfig={STATUS_CONFIG}
                                height={19}
                                sx={{ flexShrink: 0 }}
                              />
                              <Typography
                                variant="caption"
                                fontWeight={800}
                                noWrap
                                title={item.label}
                                sx={{ minWidth: 0 }}
                              >
                                {item.label}
                              </Typography>
                            </Box>
                            {item.detail && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{
                                  display: "block",
                                  mt: 0.45,
                                  lineHeight: 1.35,
                                }}
                              >
                                {item.detail}
                              </Typography>
                            )}
                            {item.action && (
                              <Typography
                                variant="caption"
                                sx={{
                                  display: "block",
                                  mt: 0.45,
                                  lineHeight: 1.35,
                                  color: checklistColor,
                                  fontWeight: 700,
                                }}
                              >
                                {item.action}
                              </Typography>
                            )}
                            {(item.endpoint || item.mode) && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                noWrap
                                title={item.endpoint || item.mode}
                                sx={{
                                  display: "block",
                                  mt: 0.45,
                                  fontFamily: "monospace",
                                  opacity: 0.82,
                                }}
                              >
                                {item.endpoint || item.mode}
                              </Typography>
                            )}
                            {item.checks.length > 0 && (
                              <Box
                                sx={{
                                  display: "flex",
                                  flexWrap: "wrap",
                                  gap: 0.35,
                                  mt: 0.55,
                                }}
                              >
                                {item.checks.slice(0, 4).map((check) => (
                                  <Chip
                                    key={`${item.id}-${check}`}
                                    label={formatLabel(check)}
                                    size="small"
                                    variant="outlined"
                                    sx={{
                                      height: 18,
                                      fontSize: "0.62rem",
                                      borderColor: alpha(checklistColor, 0.25),
                                      color: "text.secondary",
                                    }}
                                  />
                                ))}
                              </Box>
                            )}
                          </Box>
                        );
                      })}
                    </Box>
                  </Stack>
                )}

                {serviceWorkflowEntries.length > 0 && (
                  <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap" }}>
                    {serviceWorkflowEntries.map((workflow) => {
                      const chipLabel =
                        workflow.status === "local"
                          ? "Local"
                          : formatLabel(workflow.status);

                      return (
                        <Tooltip
                          key={workflow.id}
                          title={
                            workflow.tooltip ||
                            workflow.endpoint ||
                            workflow.label
                          }
                        >
                          <Box
                            sx={{
                              minWidth: { xs: "100%", sm: 210 },
                              maxWidth: { xs: "100%", md: 280 },
                              borderRadius: "8px",
                              border: "1px solid",
                              borderColor: workflow.ready
                                ? alpha(accentColors.success, 0.28)
                                : alpha(accentColors.rose, 0.28),
                              bgcolor: workflow.ready
                                ? alpha(accentColors.success, 0.045)
                                : alpha(accentColors.rose, 0.035),
                              px: 0.85,
                              py: 0.65,
                              display: "flex",
                              alignItems: "center",
                              gap: 0.75,
                            }}
                          >
                            <StatusChip
                              status={workflow.status}
                              label={chipLabel}
                              statusConfig={STATUS_CONFIG}
                              height={20}
                              sx={{ flexShrink: 0 }}
                            />
                            <Typography
                              variant="caption"
                              fontWeight={700}
                              noWrap
                              title={workflow.label}
                              sx={{ minWidth: 0 }}
                            >
                              {workflow.label}
                            </Typography>
                          </Box>
                        </Tooltip>
                      );
                    })}
                  </Box>
                )}

                {serviceStatusEntries.length > 0 ? (
                  <Grid container spacing={1}>
                    {serviceStatusEntries.map((service) => {
                      const chipStatus = service.ready
                        ? "ready"
                        : service.configured
                          ? service.status
                          : service.status === "disabled"
                            ? "disabled"
                            : "not_configured";
                      const endpoint = service.healthUrl || service.url;
                      const envText = service.requiredEnv.join(" / ");
                      const deploymentText =
                        service.deploymentHint ||
                        (!service.ready ? envText : "");
                      const sourceKnown =
                        service.urlSource && service.urlSource !== "unset";
                      const sourceText = sourceKnown
                        ? service.autoDiscovered
                          ? "Docker default"
                          : service.urlSourceLabel || service.urlSource
                        : "";
                      const healthCandidateTitle =
                        service.compatibleHealthUrls.length > 0
                          ? `Compatible health URLs: ${service.compatibleHealthUrls.join(", ")}`
                          : "";
                      const healthAttemptTitle =
                        service.healthAttempts.length > 0
                          ? `Health probe attempts: ${service.healthAttempts
                              .map((attempt) => {
                                const parts = [
                                  attempt.url,
                                  attempt.statusCode
                                    ? String(attempt.statusCode)
                                    : formatLabel(attempt.status),
                                  attempt.error,
                                ].filter(Boolean);
                                return parts.join(" ");
                              })
                              .join(" | ")}`
                          : service.attemptedHealthUrls.length > 0
                            ? `Health probe attempts: ${service.attemptedHealthUrls.join(", ")}`
                            : "";

                      return (
                        <Grid
                          item
                          xs={12}
                          sm={6}
                          lg={serviceStatusEntries.length > 4 ? 2 : 3}
                          key={service.name}
                        >
                          <Box
                            sx={{
                              borderRadius: "8px",
                              border: "1px solid",
                              borderColor: service.ready
                                ? alpha(accentColors.success, 0.32)
                                : "divider",
                              bgcolor: service.ready
                                ? alpha(accentColors.success, 0.045)
                                : alpha(accentColors.slate, 0.04),
                              p: 1,
                              minHeight: 146,
                              display: "flex",
                              flexDirection: "column",
                              gap: 0.5,
                            }}
                          >
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "space-between",
                                gap: 0.75,
                              }}
                            >
                              <Typography
                                variant="caption"
                                fontWeight={800}
                                noWrap
                                title={formatLabel(service.name)}
                                sx={{ minWidth: 0 }}
                              >
                                {formatLabel(service.name)}
                              </Typography>
                              <StatusChip
                                status={chipStatus}
                                label={formatLabel(service.status)}
                                statusConfig={STATUS_CONFIG}
                                height={20}
                                sx={{ flexShrink: 0 }}
                              />
                            </Box>
                            {service.description && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ lineHeight: 1.35 }}
                              >
                                {service.description}
                              </Typography>
                            )}
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              sx={{ lineHeight: 1.35 }}
                            >
                              {service.message}
                            </Typography>
                            {sourceText && (
                              <Chip
                                label={sourceText}
                                size="small"
                                variant="outlined"
                                title={
                                  service.urlSourceLabel || service.urlSource
                                }
                                sx={{
                                  alignSelf: "flex-start",
                                  maxWidth: "100%",
                                  height: 18,
                                  fontSize: "0.62rem",
                                  borderColor: service.autoDiscovered
                                    ? alpha(accentColors.info, 0.32)
                                    : alpha(accentColors.success, 0.28),
                                  color: "text.secondary",
                                  "& .MuiChip-label": {
                                    px: 0.65,
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                  },
                                }}
                              />
                            )}
                            {(service.compatibleHealthUrls.length > 1 ||
                              service.attemptedHealthUrls.length > 0) && (
                              <Box
                                sx={{
                                  display: "flex",
                                  gap: 0.35,
                                  flexWrap: "wrap",
                                  minWidth: 0,
                                }}
                              >
                                {service.compatibleHealthUrls.length > 1 && (
                                  <Chip
                                    label={`${service.compatibleHealthUrls.length} health paths`}
                                    size="small"
                                    variant="outlined"
                                    title={healthCandidateTitle}
                                    sx={{
                                      height: 18,
                                      fontSize: "0.62rem",
                                      borderColor: alpha(
                                        accentColors.info,
                                        0.26,
                                      ),
                                      color: "text.secondary",
                                      "& .MuiChip-label": { px: 0.65 },
                                    }}
                                  />
                                )}
                                {service.attemptedHealthUrls.length > 0 && (
                                  <Chip
                                    label={`Tried ${service.attemptedHealthUrls.length}`}
                                    size="small"
                                    variant="outlined"
                                    title={healthAttemptTitle}
                                    sx={{
                                      height: 18,
                                      fontSize: "0.62rem",
                                      borderColor: alpha(
                                        service.ready
                                          ? accentColors.success
                                          : accentColors.warning,
                                        0.3,
                                      ),
                                      color: "text.secondary",
                                      "& .MuiChip-label": { px: 0.65 },
                                    }}
                                  />
                                )}
                                {service.healthAttempts
                                  .slice(0, 2)
                                  .map((attempt, index) => {
                                    const labelParts = [
                                      formatCompactEndpoint(attempt.url),
                                      attempt.statusCode
                                        ? String(attempt.statusCode)
                                        : formatLabel(attempt.status),
                                    ].filter(Boolean);

                                    return (
                                      <Chip
                                        key={`${service.name}-health-attempt-${attempt.url || index}-${attempt.status}`}
                                        label={labelParts.join(" ")}
                                        size="small"
                                        variant="outlined"
                                        title={[
                                          attempt.url,
                                          attempt.status,
                                          attempt.statusCode,
                                          attempt.error,
                                        ]
                                          .filter(Boolean)
                                          .join(" ")}
                                        sx={{
                                          height: 18,
                                          maxWidth: "100%",
                                          fontSize: "0.62rem",
                                          borderColor: alpha(
                                            accentColors.slate,
                                            0.24,
                                          ),
                                          color: "text.secondary",
                                          "& .MuiChip-label": {
                                            px: 0.65,
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                          },
                                        }}
                                      />
                                    );
                                  })}
                              </Box>
                            )}
                            {(endpoint || envText) && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                noWrap
                                title={endpoint || envText}
                                sx={{
                                  mt: "auto",
                                  fontFamily: "monospace",
                                  opacity: 0.82,
                                }}
                              >
                                {endpoint || envText}
                              </Typography>
                            )}
                            {!service.ready && deploymentText && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                title={deploymentText}
                                sx={{
                                  display: "-webkit-box",
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: "vertical",
                                  overflow: "hidden",
                                  lineHeight: 1.35,
                                }}
                              >
                                {deploymentText}
                              </Typography>
                            )}
                          </Box>
                        </Grid>
                      );
                    })}
                  </Grid>
                ) : (
                  <Alert severity="info" variant="outlined">
                    Service status has not been loaded yet.
                  </Alert>
                )}
              </Stack>
            </SectionCard>

            <Grid container spacing={1.25} alignItems="stretch">
              <Grid item xs={12} lg={7}>
                <SectionCard
                  title="Capability Surface"
                  subtitle={`${capabilityList.length} typed capabilities exposed`}
                  icon={<AnalyticsIcon />}
                  accentColor={accentColors.info}
                  fullHeight
                  action={
                    <StatusChip
                      status={
                        capabilityRecord.enabled === true
                          ? "available"
                          : "disabled"
                      }
                      statusConfig={STATUS_CONFIG}
                      height={22}
                    />
                  }
                >
                  <Stack spacing={1.25}>
                    <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap" }}>
                      {Object.entries(serviceRecord).map(([name, service]) => {
                        const serviceEnabled =
                          asRecord(service).enabled === true;

                        return (
                          <StatusChip
                            key={name}
                            status={serviceEnabled ? "online" : "offline"}
                            label={formatLabel(name)}
                            statusConfig={STATUS_CONFIG}
                            height={22}
                          />
                        );
                      })}
                    </Box>
                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: {
                          xs: "1fr",
                          md: "repeat(2, minmax(0, 1fr))",
                        },
                        gap: 0.75,
                        maxHeight: 220,
                        overflow: "auto",
                        pr: 0.25,
                      }}
                    >
                      {capabilityList.slice(0, 18).map((capability, index) => {
                        const capabilityInfo = asRecord(capability);
                        const label = String(
                          capabilityInfo.label ||
                            capabilityInfo.id ||
                            `Capability ${index + 1}`,
                        );
                        const path = String(
                          capabilityInfo.llama_nexus_path || "",
                        );

                        return (
                          <Box
                            key={`${label}-${index}`}
                            sx={{
                              p: 1,
                              borderRadius: "8px",
                              border: "1px solid",
                              borderColor: "divider",
                              bgcolor: alpha(accentColors.info, 0.05),
                              minWidth: 0,
                            }}
                          >
                            <Typography
                              sx={{ fontWeight: 700, fontSize: "0.78rem" }}
                              noWrap
                              title={label}
                            >
                              {label}
                            </Typography>
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              noWrap
                              title={path}
                              sx={{ display: "block" }}
                            >
                              {path}
                            </Typography>
                          </Box>
                        );
                      })}
                    </Box>
                  </Stack>
                </SectionCard>
              </Grid>
              <Grid item xs={12} md={5}>
                <SectionCard
                  title="Readiness"
                  subtitle={String(
                    diagnosticsRecord.summary ||
                      "Deployment and service readiness",
                  )}
                  icon={<QualityIcon />}
                  accentColor={accentColors.purple}
                  fullHeight
                  action={
                    <StatusChip
                      status={deploymentChipStatus}
                      label={formatLabel(deploymentStatus)}
                      statusConfig={STATUS_CONFIG}
                      height={22}
                    />
                  }
                >
                  <Stack spacing={1.1}>
                    {diagnosticWarnings.length > 0 && (
                      <Alert
                        severity="warning"
                        variant="outlined"
                        sx={{ py: 0.5 }}
                      >
                        <Stack spacing={0.35}>
                          {diagnosticWarnings.slice(0, 3).map((warning) => (
                            <Typography
                              key={warning}
                              variant="caption"
                              sx={{ lineHeight: 1.35 }}
                            >
                              {warning}
                            </Typography>
                          ))}
                        </Stack>
                      </Alert>
                    )}

                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: {
                          xs: "1fr",
                          sm: "repeat(2, minmax(0, 1fr))",
                        },
                        gap: 0.75,
                      }}
                    >
                      {diagnosticFeatures.map(({ name, data }) => {
                        const label = String(data.label || formatLabel(name));
                        const ready = data.ready === true;
                        const configured = data.configured === true;
                        const featureStatus = String(
                          data.status ||
                            (ready
                              ? "ready"
                              : configured
                                ? "partial"
                                : "offline"),
                        );
                        const chipStatus = ready
                          ? "ready"
                          : featureStatus === "optional" || configured
                            ? "partial"
                            : featureStatus === "disabled"
                              ? "disabled"
                              : "offline";
                        const urls = asRecord(data.urls);
                        const urlEntries = Object.entries(urls).filter(
                          ([, value]) => Boolean(value),
                        );
                        const directUrl =
                          typeof data.url === "string" ? data.url : "";
                        const directSource = asString(
                          data.url_source_label,
                          asString(data.url_source),
                        );
                        const sourceEntries = Object.entries(
                          asRecord(data.url_sources),
                        )
                          .map(([sourceName, sourceValue]) => {
                            const sourceData = asRecord(sourceValue);
                            const source = asString(
                              sourceData.source,
                              asString(sourceData.url_source),
                            );
                            const label = asString(
                              sourceData.label,
                              asString(
                                sourceData.url_source_label,
                                source ? formatLabel(source) : "",
                              ),
                            );
                            return source && source !== "unset"
                              ? `${formatLabel(sourceName)} ${label}`
                              : "";
                          })
                          .filter(Boolean);
                        const sourceText =
                          directSource && directSource !== "Not Configured"
                            ? directSource
                            : sourceEntries.join(" | ");
                        const message =
                          typeof data.message === "string" &&
                          data.message.trim()
                            ? data.message
                            : "No readiness detail available.";
                        const nextStep =
                          typeof data.next_step === "string"
                            ? data.next_step
                            : "";
                        const urlText = directUrl
                          ? directUrl
                          : urlEntries
                              .map(
                                ([urlName, value]) =>
                                  `${formatLabel(urlName)} ${String(value)}`,
                              )
                              .join(" | ");
                        const envText = asArray(data.required_env)
                          .map(String)
                          .join(" / ");

                        return (
                          <Box
                            key={name}
                            sx={{
                              borderRadius: "8px",
                              border: "1px solid",
                              borderColor: ready
                                ? alpha(accentColors.success, 0.32)
                                : "divider",
                              bgcolor: ready
                                ? alpha(accentColors.success, 0.05)
                                : alpha(accentColors.slate, 0.04),
                              p: 1,
                              minWidth: 0,
                            }}
                          >
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "space-between",
                                gap: 0.75,
                              }}
                            >
                              <Typography
                                sx={{ fontWeight: 800, fontSize: "0.78rem" }}
                                noWrap
                                title={label}
                              >
                                {label}
                              </Typography>
                              <StatusChip
                                status={chipStatus}
                                label={formatLabel(featureStatus)}
                                statusConfig={STATUS_CONFIG}
                                height={20}
                              />
                            </Box>
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              sx={{
                                display: "block",
                                mt: 0.45,
                                lineHeight: 1.35,
                              }}
                            >
                              {message}
                            </Typography>
                            {sourceText && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                noWrap
                                title={sourceText}
                                sx={{
                                  display: "block",
                                  mt: 0.35,
                                  fontFamily: "monospace",
                                  opacity: 0.78,
                                }}
                              >
                                {sourceText}
                              </Typography>
                            )}
                            {(urlText || envText) && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                noWrap
                                title={urlText || envText}
                                sx={{
                                  display: "block",
                                  mt: 0.5,
                                  fontFamily: "monospace",
                                  opacity: 0.82,
                                }}
                              >
                                {urlText || envText}
                              </Typography>
                            )}
                            {!ready && nextStep && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{
                                  display: "block",
                                  mt: 0.35,
                                  lineHeight: 1.35,
                                }}
                              >
                                {nextStep}
                              </Typography>
                            )}
                          </Box>
                        );
                      })}
                    </Box>

                    <Divider />
                    <JsonResult
                      value={{
                        deploymentDiagnostics,
                        serviceStatus,
                        evaluationStatus,
                        codeStats,
                        redditHealth,
                        memoryHealth,
                        domainStats,
                        extractionCapabilities,
                      }}
                    />
                  </Stack>
                </SectionCard>
              </Grid>
            </Grid>
          </>
        )}

        <SectionCard
          title="Document Workspace"
          icon={<GraphIcon />}
          accentColor={accentColors.primary}
          headerSx={{
            display: { xs: "none", sm: "flex" },
          }}
          contentSx={{
            p: { xs: 0.75, md: 1.25 },
            "&:last-child": { pb: { xs: 0.85, md: 1.25 } },
          }}
        >
          {false && (
            <Tabs
              value={activeTab}
              onChange={(_, value: number) => setActiveTab(value)}
              variant="scrollable"
              scrollButtons="auto"
              sx={{
                minHeight: 36,
                borderBottom: "1px solid",
                borderColor: "divider",
                "& .MuiTab-root": {
                  minHeight: 36,
                  px: 1.25,
                  textTransform: "none",
                  fontWeight: 700,
                },
              }}
            >
              <Tab
                value={6}
                icon={<StorageIcon />}
                iconPosition="start"
                label="Corpus"
              />
              <Tab
                value={0}
                icon={<ReasoningIcon />}
                iconPosition="start"
                label="Query"
              />
              <Tab
                value={7}
                icon={<GraphIcon />}
                iconPosition="start"
                label="Graph"
              />
              <Tab
                value={8}
                icon={<GraphIcon />}
                iconPosition="start"
                label="Bridge"
              />
            </Tabs>
          )}

          <TabPanel value={activeTab} index={0}>
            <Grid container spacing={1.25}>
              <Grid item xs={12} lg={7}>
                <Stack spacing={1}>
                  <TextField
                    label="Query"
                    value={queryText}
                    onChange={(event) => setQueryText(event.target.value)}
                    multiline
                    minRows={4}
                    fullWidth
                  />
                  <Box sx={fieldGridSx}>
                    <FormControl size="small">
                      <InputLabel>Search</InputLabel>
                      <Select
                        label="Search"
                        value={querySearchType}
                        onChange={(event) =>
                          setQuerySearchType(event.target.value as SearchType)
                        }
                      >
                        {SEARCH_TYPES.map((option) => (
                          <MenuItem key={option.value} value={option.value}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <TextField
                      label="Domain"
                      size="small"
                      value={queryDomain}
                      onChange={(event) => setQueryDomain(event.target.value)}
                    />
                    <TextField
                      label="Top K"
                      size="small"
                      type="number"
                      inputProps={{ min: 1, max: 50 }}
                      value={queryTopK}
                      onChange={(event) =>
                        setQueryTopK(
                          Math.max(1, Number(event.target.value) || 1),
                        )
                      }
                    />
                  </Box>
                  {queryError && (
                    <Alert severity="error" variant="outlined">
                      {queryError}
                    </Alert>
                  )}
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Button
                      startIcon={
                        queryLoading ? (
                          <CircularProgress size={16} />
                        ) : (
                          <RunIcon />
                        )
                      }
                      disabled={queryLoading || !queryText.trim()}
                      onClick={handleRunQuery}
                      sx={actionButtonSx(accentColors.rose)}
                    >
                      Analyze and Query
                    </Button>
                    <Button
                      startIcon={<GraphIcon />}
                      disabled={queryLoading || !queryText.trim()}
                      onClick={handleBuildChatContext}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Build Context
                    </Button>
                  </Stack>
                  {chatContextPipelineRunIds.length > 0 && (
                    <Tooltip
                      title={chatContextPipelineRunIds.slice(0, 4).join(", ")}
                    >
                      <Chip
                        label={`Using ${chatContextPipelineRunIds.length} recent document trace${chatContextPipelineRunIds.length === 1 ? "" : "s"}`}
                        size="small"
                        sx={{
                          alignSelf: "flex-start",
                          height: 24,
                          fontSize: "0.72rem",
                          bgcolor: alpha(accentColors.info, 0.12),
                        }}
                      />
                    </Tooltip>
                  )}
                </Stack>
              </Grid>
              <Grid item xs={12} lg={7}>
                <QueryResultSummary value={queryResult} />
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={1}>
            <Grid container spacing={1.25}>
              <Grid item xs={12} md={4}>
                <Stack spacing={1}>
                  <TextField
                    label="Text"
                    value={extractionText}
                    onChange={(event) => setExtractionText(event.target.value)}
                    multiline
                    minRows={5}
                    fullWidth
                  />
                  <TextField
                    label="Domain"
                    size="small"
                    value={extractionDomain}
                    onChange={(event) =>
                      setExtractionDomain(event.target.value)
                    }
                  />
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={useSpanbert}
                          onChange={(event) =>
                            setUseSpanbert(event.target.checked)
                          }
                        />
                      }
                      label="SpanBERT"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={useDependency}
                          onChange={(event) =>
                            setUseDependency(event.target.checked)
                          }
                        />
                      }
                      label="Dependency"
                    />
                    <FormControlLabel
                      control={
                        <Switch
                          checked={useEntityLinking}
                          onChange={(event) =>
                            setUseEntityLinking(event.target.checked)
                          }
                        />
                      }
                      label="Linking"
                    />
                  </Stack>
                  <Grid container spacing={1}>
                    <Grid item xs={4}>
                      <MetricTile
                        label="Entities"
                        value={extractionSummary.entities}
                        color={accentColors.success}
                        icon={<ExtractionIcon />}
                      />
                    </Grid>
                    <Grid item xs={4}>
                      <MetricTile
                        label="Relations"
                        value={extractionSummary.relationships}
                        color={accentColors.warning}
                        icon={<GraphIcon />}
                      />
                    </Grid>
                    <Grid item xs={4}>
                      <MetricTile
                        label="Diagnostics"
                        value={extractionSummary.diagnostics}
                        color={accentColors.info}
                        icon={<AnalyticsIcon />}
                      />
                    </Grid>
                  </Grid>
                  {extractionError && (
                    <Alert severity="error" variant="outlined">
                      {extractionError}
                    </Alert>
                  )}
                  <Button
                    startIcon={
                      extractionLoading ? (
                        <CircularProgress size={16} />
                      ) : (
                        <RunIcon />
                      )
                    }
                    disabled={extractionLoading || !extractionText.trim()}
                    onClick={handleRunExtraction}
                    sx={actionButtonSx(accentColors.warning)}
                  >
                    Run Extraction Stack
                  </Button>
                </Stack>
              </Grid>
              <Grid item xs={12} lg={7}>
                <JsonResult value={extractionResult} />
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={2}>
            <Grid container spacing={1.25}>
              <Grid item xs={12} lg={5}>
                <Stack spacing={1}>
                  <FormControl size="small">
                    <InputLabel>Test suite</InputLabel>
                    <Select
                      label="Test suite"
                      value={testSuite}
                      onChange={(event) =>
                        setTestSuite(event.target.value as TestSuite)
                      }
                    >
                      {TEST_SUITES.map((option) => (
                        <MenuItem key={option.value} value={option.value}>
                          {option.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <Button
                    startIcon={
                      qualityLoading ? (
                        <CircularProgress size={16} />
                      ) : (
                        <QualityIcon />
                      )
                    }
                    disabled={qualityLoading}
                    onClick={handleRunTestSuite}
                    sx={actionButtonSx(accentColors.purple)}
                  >
                    Run Suite
                  </Button>
                  <Button
                    startIcon={<GraphIcon />}
                    disabled={qualityLoading}
                    onClick={handleGraphCompleteness}
                    variant="outlined"
                    sx={{ borderRadius: "8px" }}
                  >
                    Graph Completeness
                  </Button>
                  <Divider />
                  <TextField
                    label="Evaluation query"
                    size="small"
                    value={qualityQuery}
                    onChange={(event) => setQualityQuery(event.target.value)}
                    fullWidth
                  />
                  <TextField
                    label="Expected answer"
                    size="small"
                    value={expectedAnswer}
                    onChange={(event) => setExpectedAnswer(event.target.value)}
                    multiline
                    minRows={3}
                    fullWidth
                  />
                  {qualityError && (
                    <Alert severity="error" variant="outlined">
                      {qualityError}
                    </Alert>
                  )}
                  <Button
                    startIcon={<RunIcon />}
                    disabled={
                      qualityLoading ||
                      !qualityQuery.trim() ||
                      !expectedAnswer.trim()
                    }
                    onClick={handleQueryResponseEval}
                    sx={actionButtonSx(accentColors.info)}
                  >
                    Evaluate Answer
                  </Button>
                  <Divider />
                  <Box sx={fieldGridSx}>
                    <TextField
                      label="Feedback query"
                      size="small"
                      value={feedbackQuery}
                      onChange={(event) => setFeedbackQuery(event.target.value)}
                    />
                    <FormControl size="small">
                      <InputLabel>Rating</InputLabel>
                      <Select
                        label="Rating"
                        value={feedbackRating}
                        onChange={(event) =>
                          setFeedbackRating(
                            Number(event.target.value) as FeedbackRating,
                          )
                        }
                      >
                        {FEEDBACK_RATINGS.map((option) => (
                          <MenuItem key={option.value} value={option.value}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <TextField
                      label="Feedback domain"
                      size="small"
                      value={feedbackDomain}
                      onChange={(event) =>
                        setFeedbackDomain(event.target.value)
                      }
                    />
                    <TextField
                      label="Analytics days"
                      size="small"
                      type="number"
                      value={feedbackDays}
                      onChange={(event) =>
                        setFeedbackDays(
                          Math.max(1, Number(event.target.value) || 1),
                        )
                      }
                    />
                  </Box>
                  <TextField
                    label="Feedback comment"
                    size="small"
                    value={feedbackComment}
                    onChange={(event) => setFeedbackComment(event.target.value)}
                    multiline
                    minRows={2}
                    fullWidth
                  />
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Button
                      startIcon={<QualityIcon />}
                      disabled={qualityLoading || !feedbackQuery.trim()}
                      onClick={handleSubmitFeedback}
                      sx={actionButtonSx(accentColors.success)}
                    >
                      Submit Feedback
                    </Button>
                    <Button
                      startIcon={<AnalyticsIcon />}
                      disabled={qualityLoading}
                      onClick={handleFeedbackAnalytics}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Feedback Analytics
                    </Button>
                  </Stack>
                </Stack>
              </Grid>
              <Grid item xs={12} lg={7}>
                <JsonResult value={qualityResult ?? evaluationStatus} />
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={3}>
            <Grid container spacing={1.25}>
              <Grid item xs={12} lg={5}>
                <Stack spacing={1}>
                  <Box sx={fieldGridSx}>
                    <TextField
                      label="Project"
                      size="small"
                      value={projectName}
                      onChange={(event) => setProjectName(event.target.value)}
                    />
                    <TextField
                      label="Language"
                      size="small"
                      value={codeLanguage}
                      onChange={(event) => setCodeLanguage(event.target.value)}
                    />
                    <TextField
                      label="Task ID"
                      size="small"
                      value={codeTaskId}
                      onChange={(event) => setCodeTaskId(event.target.value)}
                    />
                  </Box>
                  <TextField
                    label="File paths"
                    value={codePaths}
                    onChange={(event) => setCodePaths(event.target.value)}
                    multiline
                    minRows={4}
                    fullWidth
                  />
                  <TextField
                    label="Analyze file"
                    size="small"
                    value={codeAnalyzePath}
                    onChange={(event) => setCodeAnalyzePath(event.target.value)}
                    fullWidth
                  />
                  {codeError && (
                    <Alert severity="error" variant="outlined">
                      {codeError}
                    </Alert>
                  )}
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Button
                      startIcon={
                        codeLoading ? (
                          <CircularProgress size={16} />
                        ) : (
                          <RunIcon />
                        )
                      }
                      disabled={
                        codeLoading || !codePaths.trim() || !projectName.trim()
                      }
                      onClick={handleCodeIngest}
                      sx={actionButtonSx(accentColors.warning)}
                    >
                      Ingest
                    </Button>
                    <Button
                      disabled={codeLoading || !codeTaskId.trim()}
                      onClick={handleCodePoll}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Poll
                    </Button>
                    <Button
                      disabled={codeLoading || !codeAnalyzePath.trim()}
                      onClick={handleCodeAnalyze}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Analyze
                    </Button>
                    <Button
                      disabled={codeLoading}
                      onClick={handleCodeStats}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Stats
                    </Button>
                  </Stack>
                </Stack>
              </Grid>
              <Grid item xs={12} lg={7}>
                <JsonResult value={codeResult ?? codeStats} />
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={4}>
            <Grid container spacing={1.25}>
              <Grid item xs={12} lg={5}>
                <Stack spacing={1}>
                  <TextField
                    label="Query"
                    value={redditQuery}
                    onChange={(event) => setRedditQuery(event.target.value)}
                    fullWidth
                  />
                  <TextField
                    label="Subreddits"
                    value={redditSubreddits}
                    onChange={(event) =>
                      setRedditSubreddits(event.target.value)
                    }
                    helperText="Comma-separated subreddit names"
                    fullWidth
                  />
                  <Box sx={fieldGridSx}>
                    <TextField
                      label="Max posts"
                      size="small"
                      type="number"
                      inputProps={{ min: 1, max: 500 }}
                      value={redditMaxPosts}
                      onChange={(event) =>
                        setRedditMaxPosts(
                          Math.max(1, Number(event.target.value) || 1),
                        )
                      }
                    />
                    <TextField
                      label="Crawl ID"
                      size="small"
                      value={redditCrawlId}
                      onChange={(event) => setRedditCrawlId(event.target.value)}
                      sx={{ gridColumn: { md: "span 2" } }}
                    />
                  </Box>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={redditAutoIngest}
                        onChange={(event) =>
                          setRedditAutoIngest(event.target.checked)
                        }
                      />
                    }
                    label="Auto-ingest into GraphRAG"
                  />
                  {redditError && (
                    <Alert severity="error" variant="outlined">
                      {redditError}
                    </Alert>
                  )}
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Button
                      startIcon={
                        redditLoading ? (
                          <CircularProgress size={16} />
                        ) : (
                          <RunIcon />
                        )
                      }
                      disabled={redditLoading || !redditQuery.trim()}
                      onClick={handleRedditStart}
                      sx={actionButtonSx(accentColors.rose)}
                    >
                      Start Crawl
                    </Button>
                    <Button
                      disabled={redditLoading || !redditCrawlId.trim()}
                      onClick={handleRedditStatus}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Status
                    </Button>
                    <Button
                      disabled={redditLoading || !redditCrawlId.trim()}
                      onClick={handleRedditResults}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Results
                    </Button>
                    <Button
                      disabled={redditLoading}
                      onClick={handleRedditList}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Crawls
                    </Button>
                    <Button
                      startIcon={<StopIcon />}
                      disabled={redditLoading || !redditCrawlId.trim()}
                      onClick={handleRedditCancel}
                      variant="outlined"
                      color="warning"
                      sx={{ borderRadius: "8px" }}
                    >
                      Cancel
                    </Button>
                  </Stack>
                </Stack>
              </Grid>
              <Grid item xs={12} lg={7}>
                <JsonResult value={redditResult ?? redditHealth} />
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={5}>
            <Grid container spacing={1.25}>
              <Grid item xs={12} lg={5}>
                <Stack spacing={1}>
                  <Box sx={fieldGridSx}>
                    <FormControl size="small">
                      <InputLabel>Platform</InputLabel>
                      <Select
                        label="Platform"
                        value={memoryPlatform}
                        onChange={(event) =>
                          setMemoryPlatform(
                            event.target.value as SocialPlatform,
                          )
                        }
                      >
                        {SOCIAL_PLATFORMS.map((option) => (
                          <MenuItem key={option.value} value={option.value}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <TextField
                      label="User ID"
                      size="small"
                      value={memoryUserId}
                      onChange={(event) => setMemoryUserId(event.target.value)}
                    />
                    <TextField
                      label="Display"
                      size="small"
                      value={memoryDisplayName}
                      onChange={(event) =>
                        setMemoryDisplayName(event.target.value)
                      }
                    />
                  </Box>
                  <TextField
                    label="Message"
                    value={memoryMessage}
                    onChange={(event) => setMemoryMessage(event.target.value)}
                    multiline
                    minRows={3}
                    fullWidth
                  />
                  <TextField
                    label="Memory query"
                    value={memoryQuery}
                    onChange={(event) => setMemoryQuery(event.target.value)}
                    multiline
                    minRows={2}
                    fullWidth
                  />
                  <Box sx={fieldGridSx}>
                    <TextField
                      label="Person ID"
                      size="small"
                      value={memoryPersonId}
                      onChange={(event) =>
                        setMemoryPersonId(event.target.value)
                      }
                    />
                    <TextField
                      label="Admin API key"
                      size="small"
                      type="password"
                      value={memoryAdminKey}
                      onChange={(event) =>
                        setMemoryAdminKey(event.target.value)
                      }
                    />
                    <FormControl size="small">
                      <InputLabel>Eval suite</InputLabel>
                      <Select
                        label="Eval suite"
                        value={memoryEvalSuite}
                        onChange={(event) =>
                          setMemoryEvalSuite(
                            event.target.value as SocialMemoryEvalSuite,
                          )
                        }
                      >
                        {SOCIAL_MEMORY_EVAL_SUITES.map((option) => (
                          <MenuItem key={option.value} value={option.value}>
                            {option.label}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </Box>
                  {memoryError && (
                    <Alert severity="error" variant="outlined">
                      {memoryError}
                    </Alert>
                  )}
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Button
                      startIcon={
                        memoryLoading ? (
                          <CircularProgress size={16} />
                        ) : (
                          <RunIcon />
                        )
                      }
                      disabled={
                        memoryLoading ||
                        !memoryUserId.trim() ||
                        !memoryDisplayName.trim() ||
                        !memoryMessage.trim()
                      }
                      onClick={handleMemoryIngest}
                      sx={actionButtonSx(accentColors.info)}
                    >
                      Ingest
                    </Button>
                    <Button
                      disabled={memoryLoading || !memoryUserId.trim()}
                      onClick={handleMemoryContext}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Context
                    </Button>
                    <Button
                      disabled={
                        memoryLoading ||
                        !memoryUserId.trim() ||
                        !memoryQuery.trim()
                      }
                      onClick={handleMemoryExplain}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Explain
                    </Button>
                    <Button
                      disabled={memoryLoading || !memoryPersonId.trim()}
                      onClick={handleMemoryFacts}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Facts
                    </Button>
                    <Button
                      disabled={memoryLoading}
                      onClick={handleMemoryEval}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Eval
                    </Button>
                    <Button
                      disabled={memoryLoading}
                      onClick={handleMemoryCommunityReport}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Report
                    </Button>
                    <Button
                      disabled={memoryLoading}
                      onClick={handleMemoryMigrate}
                      variant="outlined"
                      color="warning"
                      sx={{ borderRadius: "8px" }}
                    >
                      Migrate
                    </Button>
                  </Stack>
                </Stack>
              </Grid>
              <Grid item xs={12} lg={7}>
                <JsonResult value={memoryResult ?? memoryHealth} />
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={6}>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: {
                  xs: "1fr",
                  lg: "minmax(0, 1fr) auto",
                },
                alignItems: "center",
                gap: 1,
                mb: { xs: 0.75, md: 1.25 },
                border: "1px solid",
                borderColor: alpha(accentColors.primary, 0.18),
                borderRadius: "8px",
                px: { xs: 0.75, md: 1 },
                py: { xs: 0.55, md: 0.75 },
                bgcolor: alpha(accentColors.slate, 0.028),
              }}
            >
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, minmax(0, 1fr))",
                  gap: 0.65,
                  minWidth: 0,
                }}
              >
                {workflowSteps.map((step, index) => {
                  const stepNumber = index + 1;
                  const isCurrent = activeWorkspaceStepTarget === step.target;
                  const isComplete = !isCurrent && step.complete;
                  const color = isComplete
                    ? accentColors.success
                    : isCurrent
                      ? accentColors.primary
                      : accentColors.slate;

                  return (
                    <Box
                      key={step.label}
                      component="button"
                      type="button"
                      aria-current={isCurrent ? "step" : undefined}
                      aria-label={`Go to ${step.label.toLowerCase()} step: ${step.detail}`}
                      onClick={() =>
                        scrollDocumentWorkspaceSectionIntoView(step.target)
                      }
                      sx={{
                        appearance: "none",
                        width: "100%",
                        display: "grid",
                        gridTemplateColumns: "auto minmax(0, 1fr)",
                        alignItems: "center",
                        gap: 0.65,
                        minWidth: 0,
                        border: "1px solid",
                        borderColor: alpha(color, isCurrent ? 0.46 : 0.18),
                        borderRadius: "8px",
                        px: { xs: 0.55, md: 0.75 },
                        py: 0.55,
                        bgcolor: alpha(color, isCurrent ? 0.08 : 0.025),
                        color: "inherit",
                        font: "inherit",
                        textAlign: "left",
                        cursor: "pointer",
                        transition:
                          "background-color 120ms ease, border-color 120ms ease",
                        "&:hover": {
                          borderColor: alpha(color, 0.52),
                          bgcolor: alpha(color, isCurrent ? 0.12 : 0.07),
                        },
                        "&:focus-visible": {
                          outline: "2px solid",
                          outlineColor: alpha(accentColors.primary, 0.74),
                          outlineOffset: 2,
                        },
                      }}
                    >
                      <Box
                        sx={{
                          width: { xs: 22, md: 24 },
                          height: { xs: 22, md: 24 },
                          borderRadius: "8px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          color,
                          bgcolor: alpha(color, 0.12),
                          flexShrink: 0,
                          "& .MuiSvgIcon-root": { fontSize: 15 },
                        }}
                      >
                        {step.icon}
                      </Box>
                      <Box sx={{ minWidth: 0 }}>
                        <Typography
                          variant="caption"
                          fontWeight={900}
                          noWrap
                          sx={{ display: "block", lineHeight: 1.1 }}
                        >
                          {stepNumber}. {step.label}
                        </Typography>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          noWrap
                          sx={{
                            display: { xs: "none", sm: "block" },
                            fontSize: "0.66rem",
                            lineHeight: 1.15,
                          }}
                        >
                          {step.detail}
                        </Typography>
                      </Box>
                    </Box>
                  );
                })}
              </Box>
              <Box
                sx={{
                  display: { xs: "none", xl: "grid" },
                  gridTemplateColumns: {
                    xs: "repeat(2, minmax(0, 1fr))",
                    sm: "repeat(4, minmax(0, 1fr))",
                    lg: "repeat(4, auto)",
                  },
                  gap: 0.65,
                  minWidth: { lg: 460 },
                }}
              >
                {documentWorkspaceStats.map((stat) => (
                  <Box
                    key={stat.label}
                    sx={{
                      minWidth: { xs: 0, lg: 92 },
                      borderRadius: "8px",
                      px: 0.8,
                      py: 0.55,
                      border: "1px solid",
                      borderColor: alpha(stat.color, 0.18),
                      bgcolor: alpha(stat.color, 0.035),
                    }}
                  >
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      noWrap
                      sx={{ display: "block", fontSize: "0.64rem" }}
                    >
                      {stat.label}
                    </Typography>
                    <Typography
                      noWrap
                      sx={{
                        color: stat.color,
                        fontSize: "0.9rem",
                        fontWeight: 900,
                        lineHeight: 1.1,
                      }}
                    >
                      {formatNumber(stat.value)}
                    </Typography>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      noWrap
                      sx={{ display: "block", fontSize: "0.61rem", lineHeight: 1.1 }}
                    >
                      {stat.detail}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Box>
            <Box
              sx={{
                display: "grid",
                gap: 1.25,
                gridTemplateColumns: {
                  xs: "minmax(0, 1fr)",
                  md: workspaceSelectedFirst
                    ? "minmax(0, 1fr) minmax(280px, 320px)"
                    : "minmax(300px, 340px) minmax(0, 1fr)",
                  xl: workspaceSelectedFirst
                    ? "minmax(0, 1fr) minmax(320px, 360px)"
                    : "minmax(320px, 360px) minmax(0, 1fr)",
                },
                gridTemplateAreas: {
                  xs: mobileWorkspaceGridAreas,
                  md: workspaceSelectedFirst ? `"selected left"` : `"left selected"`,
                },
                alignItems: "start",
                justifyItems: "stretch",
              }}
            >
              <Box
                sx={{
                  display: { xs: "contents", md: "block" },
                  gridArea: { md: "left" },
                  minWidth: 0,
                  width: "100%",
                  position: { md: "sticky" },
                  top: { md: 72 },
                  alignSelf: { md: "start" },
                  maxHeight: { md: "calc(100vh - 88px)" },
                  overflowY: { md: "auto" },
                  pr: { md: 0.25 },
                  scrollbarWidth: "thin",
                }}
              >
                <Stack spacing={1} sx={{ display: { xs: "contents", md: "flex" } }}>
                  <Box
                    ref={uploadIntakeRef}
                    sx={{
                      display: {
                        xs:
                          compactUploadIntake && !mobileUploadIntakeIsLead
                            ? "none"
                            : "grid",
                        md: "grid",
                      },
                      gap: 1,
                      gridArea: { xs: "upload", md: "auto" },
                      minWidth: 0,
                    }}
                  >
                  <Box
                    sx={{
                      border: "1px solid",
                      borderColor: alpha(
                        corpusUploadFileCount > 0
                          ? accentColors.success
                          : accentColors.primary,
                        0.24,
                      ),
                      borderRadius: "8px",
                      p: { xs: 0.9, md: 1.25 },
                      bgcolor: alpha(accentColors.primary, 0.04),
                    }}
                  >
                    <Stack spacing={{ xs: 0.85, md: 1.25 }}>
                      <Box
                        sx={{
                          display:
                            corpusUploadFileCount > 0 || corpusUploadInProgress
                              ? "flex"
                              : "none",
                          alignItems: "center",
                          justifyContent: "space-between",
                          gap: 1,
                        }}
                      >
                        <Box sx={{ minWidth: 0 }}>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ display: "block", fontWeight: 900 }}
                          >
                            {compactUploadIntake && !mobileUploadIntakeIsLead
                              ? "Add more"
                              : "Selected"}
                          </Typography>
                          <Typography
                            variant="body2"
                            noWrap
                            sx={{ fontWeight: 850 }}
                          >
                            {corpusUploadFileCount > 0
                              ? `${formatNumber(corpusUploadFileCount)} selected`
                              : compactUploadIntake && !mobileUploadIntakeIsLead
                              ? "Add sources"
                              : "Add documents"}
                          </Typography>
                        </Box>
                        <Chip
                          size="small"
                          label={
                            corpusUploadFileCount > 0
                              ? formatFileSize(corpusUploadTotalBytes)
                              : compactUploadIntake && !mobileUploadIntakeIsLead
                                ? "Ready"
                                : "Select files"
                          }
                          variant="outlined"
                          sx={{ height: 22, flexShrink: 0 }}
                        />
                      </Box>
                      {serviceStatusLoaded &&
                        !remoteGraphRAGReady &&
                        !compactUploadIntake && (
                        <Alert
                          severity={graphRAGCoreService?.configured ? "warning" : "info"}
                          variant="outlined"
                          sx={{
                            py: 0.45,
                            "& .MuiAlert-message": {
                              fontSize: "0.78rem",
                              lineHeight: 1.45,
                            },
                          }}
                        >
                          {graphRAGCoreService?.configured
                            ? "Remote GraphRAG is offline; local document chat still works."
                            : "Local indexing is active. Remote GraphRAG can be connected later."}
                        </Alert>
                      )}
                      <Button
                        data-testid="graphrag-document-upload-dropzone"
                        component="label"
                        disabled={corpusLoading}
                        variant="outlined"
                        fullWidth
                        onDragEnter={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          if (!corpusLoading) setCorpusDragActive(true);
                        }}
                        onDragOver={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          if (!corpusLoading) setCorpusDragActive(true);
                        }}
                        onDragLeave={(event) => {
                          const nextTarget = event.relatedTarget;
                          if (
                            nextTarget instanceof Node &&
                            event.currentTarget.contains(nextTarget)
                          ) {
                            return;
                          }
                          setCorpusDragActive(false);
                        }}
                        onDrop={handleCorpusDrop}
                        sx={{
                          borderRadius: "8px",
                          minHeight: compactUploadIntake
                            ? {
                                xs: mobileUploadIntakeIsLead ? 86 : 42,
                                sm: mobileUploadIntakeIsLead ? 128 : 46,
                                md: 48,
                              }
                            : { xs: 86, sm: 128, md: 142 },
                          borderStyle: "dashed",
                          borderWidth: corpusDragActive ? 2 : 1,
                          borderColor: corpusDragActive
                            ? accentColors.primary
                            : corpusUploadFileCount > 0
                              ? alpha(accentColors.success, 0.52)
                              : alpha(accentColors.primary, 0.42),
                          bgcolor: corpusDragActive
                            ? alpha(accentColors.primary, 0.12)
                            : corpusUploadFileCount > 0
                              ? alpha(accentColors.success, 0.055)
                              : alpha(accentColors.primary, 0.035),
                          flexDirection: "column",
                          justifyContent: "center",
                          gap: compactUploadIntake
                            ? {
                                xs: mobileUploadIntakeIsLead ? 0.45 : 0.25,
                                md: 0.25,
                              }
                            : { xs: 0.45, md: 0.7 },
                          px: { xs: 1, md: 1.5 },
                          textTransform: "none",
                          transition:
                            "background-color 120ms ease, border-color 120ms ease",
                        }}
                      >
                        <UploadIcon
                          sx={{
                            fontSize: compactUploadIntake
                              ? {
                                  xs: mobileUploadIntakeIsLead ? 24 : 20,
                                  md: 22,
                                }
                              : { xs: 24, md: 28 },
                          }}
                        />
                        <Typography variant="body2" fontWeight={800}>
                          {corpusUploadFileCount > 0 ? (
                            `${formatNumber(corpusUploadFileCount)} selected`
                          ) : compactUploadIntake ? (
                            <>
                              <Box
                                component="span"
                                sx={{
                                  display: {
                                    xs: mobileUploadIntakeIsLead ? "inline" : "none",
                                    md: "none",
                                  },
                                }}
                              >
                                Drop or select documents
                              </Box>
                              <Box
                                component="span"
                                sx={{
                                  display: {
                                    xs: mobileUploadIntakeIsLead ? "none" : "inline",
                                    md: "inline",
                                  },
                                }}
                              >
                                Add documents
                              </Box>
                            </>
                          ) : (
                            "Drop or select documents"
                          )}
                        </Typography>
                        {(!compactUploadIntake || mobileUploadIntakeIsLead) && (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{
                              display: compactUploadIntake
                                ? { xs: "block", md: "none" }
                                : "block",
                            }}
                          >
                            {corpusUploadFileCount > 0
                              ? corpusUploadModeDetail
                              : "PDF, Office, Markdown, images"}
                          </Typography>
                        )}
                        <input
                          hidden
                          type="file"
                          multiple
                          accept={GRAPHRAG_DOCUMENT_ACCEPT}
                          onChange={handleCorpusUploadFileChange}
                        />
                      </Button>
                      {corpusUploadFileCount > 0 && (
                        <Box
                          sx={{
                            border: "1px solid",
                            borderColor: alpha(accentColors.primary, 0.18),
                            borderRadius: "8px",
                            overflow: "hidden",
                            bgcolor: alpha(accentColors.slate, 0.06),
                          }}
                        >
                          <Box
                            sx={{
                              px: 1,
                              py: 0.75,
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                              gap: 1,
                              borderBottom: "1px solid",
                              borderColor: "divider",
                            }}
                          >
                            <Box sx={{ minWidth: 0 }}>
                              <Typography variant="caption" fontWeight={850}>
                                Selected files
                              </Typography>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                title={corpusSelectedFileNames}
                                noWrap
                                sx={{ display: "block", maxWidth: 320 }}
                              >
                                {corpusUploadManifest.primarySummary}
                              </Typography>
                            </Box>
                            {corpusUploadInProgress ? (
                              <Chip
                                size="small"
                                label="Processing"
                                color="success"
                                variant="outlined"
                                sx={{ height: 22, flexShrink: 0 }}
                              />
                            ) : (
                              <Button
                                size="small"
                                variant="text"
                                disabled={corpusLoading}
                                onClick={() => {
                                  setCorpusUploadFile(null);
                                  setCorpusUploadFiles([]);
                                  setCorpusDragActive(false);
                                }}
                                sx={{
                                  minHeight: 24,
                                  px: 0.75,
                                  borderRadius: "8px",
                                  textTransform: "none",
                                }}
                              >
                                Clear
                              </Button>
                            )}
                            </Box>
                          {(corpusUploadManifest.pipelineSteps.length > 0 ||
                            corpusIntakeChips.length > 0) && (
                            <Box
                              component="details"
                              sx={{
                                borderTop: "1px solid",
                                borderColor: alpha(accentColors.slate, 0.14),
                                mt: 0.65,
                                "& summary": {
                                  cursor: "pointer",
                                  listStyle: "none",
                                },
                                "& summary::-webkit-details-marker": {
                                  display: "none",
                                },
                              }}
                            >
                              <Box
                                component="summary"
                                title={corpusUploadManifest.pipelineSummary}
                                sx={{
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "space-between",
                                  gap: 1,
                                  px: 0.85,
                                  py: 0.7,
                                }}
                              >
                                <Box sx={{ minWidth: 0 }}>
                                  <Typography
                                    variant="caption"
                                    sx={{ display: "block", fontWeight: 850 }}
                                  >
                                    What will be read
                                  </Typography>
                                  {!corpusUploadFileCount &&
                                    corpusUploadManifest.pipelineSummary && (
                                    <Typography
                                      variant="caption"
                                      color="text.secondary"
                                      title={corpusUploadManifest.pipelineSummary}
                                      noWrap
                                      sx={{
                                        display: "block",
                                        maxWidth: { xs: 210, sm: 300 },
                                        fontSize: "0.64rem",
                                        lineHeight: 1.15,
                                      }}
                                    >
                                      {corpusUploadManifest.pipelineSummary}
                                    </Typography>
                                  )}
                                </Box>
                                <Chip
                                  size="small"
                                  label={`${formatNumber(corpusUploadManifest.pipelineSteps.length || 1)} checks`}
                                  variant="outlined"
                                  sx={{ height: 20, flexShrink: 0, fontSize: "0.62rem" }}
                                />
                              </Box>
                              {corpusUploadManifest.pipelineSteps.length > 0 && (
                                <Stack spacing={0.55} sx={{ px: 0.85, pb: 0.7 }}>
                                  {corpusUploadManifest.pipelineSteps.map(
                                    (step, index) => (
                                      <Box
                                        key={step.label}
                                        sx={{
                                          display: "grid",
                                          gridTemplateColumns: "auto minmax(0, 1fr)",
                                          alignItems: "center",
                                          gap: 0.55,
                                          minWidth: 0,
                                        }}
                                      >
                                        <Chip
                                          size="small"
                                          label={index + 1}
                                          sx={{
                                            width: 22,
                                            height: 20,
                                            fontSize: "0.62rem",
                                            bgcolor: alpha(
                                              index === 3
                                                ? accentColors.success
                                                : accentColors.primary,
                                              0.12,
                                            ),
                                          }}
                                        />
                                        <Box sx={{ minWidth: 0 }}>
                                          <Typography
                                            variant="caption"
                                            noWrap
                                            sx={{
                                              display: "block",
                                              fontWeight: 850,
                                              lineHeight: 1.18,
                                            }}
                                          >
                                            {step.label}
                                          </Typography>
                                          <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            noWrap
                                            sx={{
                                              display: "block",
                                              fontSize: "0.63rem",
                                              lineHeight: 1.18,
                                            }}
                                          >
                                            {step.detail}
                                          </Typography>
                                        </Box>
                                      </Box>
                                    ),
                                  )}
                                </Stack>
                              )}
                              {corpusIntakeChips.length > 0 && (
                                <Box
                                  sx={{
                                    px: 0.85,
                                    pb: 0.75,
                                    display: "flex",
                                    gap: 0.45,
                                    flexWrap: "wrap",
                                  }}
                                >
                                  {corpusIntakeChips.map((chip) => (
                                    <Chip
                                      key={chip.label}
                                      size="small"
                                      label={chip.label}
                                      variant="outlined"
                                      sx={{
                                        height: 20,
                                        fontSize: "0.62rem",
                                        borderColor: alpha(chip.color, 0.32),
                                        bgcolor: alpha(chip.color, 0.045),
                                      }}
                                    />
                                  ))}
                                </Box>
                              )}
                            </Box>
                          )}
                          <Stack spacing={0.5} sx={{ p: 0.75 }}>
                            {corpusUploadPreviewFiles.map((file) => (
                              <Box
                                key={`${file.name}-${file.size}-${file.lastModified}`}
                                sx={{
                                  display: "grid",
                                  gridTemplateColumns: "auto minmax(0, 1fr) auto",
                                  alignItems: "center",
                                  gap: 0.75,
                                  minWidth: 0,
                                }}
                              >
                                <Chip
                                  size="small"
                                  label={formatFileKind(file.name)}
                                  variant="outlined"
                                  sx={{
                                    height: 20,
                                    minWidth: 58,
                                    fontSize: "0.66rem",
                                    borderColor: alpha(accentColors.info, 0.34),
                                  }}
                                />
                                <Typography
                                  variant="caption"
                                  noWrap
                                  title={file.name}
                                  sx={{ minWidth: 0, fontWeight: 700 }}
                                >
                                  {file.name}
                                </Typography>
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{ flexShrink: 0 }}
                                >
                                  {formatFileSize(file.size)}
                                </Typography>
                              </Box>
                            ))}
                            {corpusUploadOverflowCount > 0 && (
                              <Typography variant="caption" color="text.secondary">
                                +{corpusUploadOverflowCount} more selected
                              </Typography>
                            )}
                          </Stack>
                        </Box>
                      )}
                      {corpusUploadFileCount > 0 && (
                        <Stack spacing={1}>
                          <Button
                            aria-label={
                              corpusHasMultipleFiles
                                ? `Process ${formatNumber(corpusUploadFileCount)} selected uploads`
                                : "Process selected upload"
                            }
                            startIcon={
                              corpusLoading ? (
                                <CircularProgress size={16} />
                              ) : (
                                <RunIcon />
                              )
                            }
                            disabled={corpusLoading}
                            onClick={
                              corpusHasMultipleFiles
                                ? handleCorpusPipelineBatchUpload
                                : handleCorpusPipelineUpload
                            }
                            fullWidth
                            sx={{
                              ...actionButtonSx(accentColors.success),
                              display: { xs: "flex", md: "none" },
                            }}
                          >
                            {corpusProcessButtonLabel}
                          </Button>
                        </Stack>
                      )}
                      {corpusUploadFileCount > 0 && (
                      <Box
                        component="details"
                        sx={{
                          display: "block",
                          borderTop: "1px solid",
                          borderColor: "divider",
                          pt: 0.75,
                          "& summary": {
                            cursor: "pointer",
                            listStyle: "none",
                          },
                          "& summary::-webkit-details-marker": {
                            display: "none",
                          },
                        }}
                      >
                        <Box
                          component="summary"
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: 1,
                          }}
                        >
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ fontWeight: 850 }}
                          >
                            Options
                          </Typography>
                          <Chip
                            size="small"
                            label={
                              corpusBuildKnowledgeGraph
                                ? "Relationships on"
                                : "Text only"
                            }
                            variant="outlined"
                            sx={{ height: 20, flexShrink: 0, fontSize: "0.62rem" }}
                          />
                        </Box>
                        <Stack spacing={1} sx={{ mt: 1 }}>
                          <Box sx={fieldGridSx}>
                            <TextField
                              label="Domain"
                              size="small"
                              value={corpusDomain}
                              onChange={(event) =>
                                setCorpusDomain(event.target.value)
                              }
                            />
                            <TextField
                              label="Document name"
                              size="small"
                              value={corpusDocumentName}
                              onChange={(event) =>
                                setCorpusDocumentName(event.target.value)
                              }
                              sx={{ gridColumn: { md: "span 2" } }}
                            />
                          </Box>
                          <Stack
                            direction="row"
                            spacing={1}
                            alignItems="center"
                            flexWrap="wrap"
                            useFlexGap
                          >
                            <FormControlLabel
                              control={
                                <Switch
                                  size="small"
                                  checked={corpusUseSemanticChunking}
                                  onChange={(event) =>
                                    setCorpusUseSemanticChunking(
                                      event.target.checked,
                                    )
                                  }
                                />
                              }
                              label="Semantic chunking"
                            />
                            <FormControlLabel
                              control={
                                <Switch
                                  size="small"
                                  checked={corpusBuildKnowledgeGraph}
                                  onChange={(event) =>
                                    setCorpusBuildKnowledgeGraph(
                                      event.target.checked,
                                    )
                                  }
                                />
                              }
                              label="Build graph"
                            />
                          </Stack>
                          <Stack
                            direction="row"
                            spacing={1}
                            flexWrap="wrap"
                            useFlexGap
                          >
                            <Button
                              startIcon={
                                corpusLoading ? (
                                  <CircularProgress size={16} />
                                ) : (
                                  <RunIcon />
                                )
                              }
                              aria-label="Preview document extraction"
                              disabled={corpusLoading || !corpusUploadFile}
                              onClick={handleCorpusPipelinePreview}
                              variant="outlined"
                              sx={{ borderRadius: "8px" }}
                            >
                              Preview extraction
                            </Button>
                          </Stack>
                        </Stack>
                      </Box>
                      )}
                    </Stack>
                  </Box>
                  {corpusError && (
                    <Alert severity="error" variant="outlined">
                      {corpusError}
                    </Alert>
                  )}
                  </Box>
                  <Box
                    ref={documentLibraryRef}
                    sx={{
                      border: "1px solid",
                      borderColor: "divider",
                      borderRadius: "8px",
                      gridArea: { xs: "documents", md: "auto" },
                      p: { xs: 0.85, md: 1 },
                      bgcolor: alpha(accentColors.slate, 0.035),
                    }}
                  >
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: 1,
                        mb: 0.9,
                        flexWrap: "wrap",
                      }}
                    >
                      <Box sx={{ minWidth: 0 }}>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ fontWeight: 850, display: "block" }}
                        >
                          Documents
                        </Typography>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: "block" }}
                        >
                          {pipelineRuns.length > 0
                            ? [
                                normalizedCorpusRunSearch
                                  ? `${visibleDocumentRuns.length}/${pipelineRuns.length} shown`
                                  : `${visibleDocumentRuns.length} document${visibleDocumentRuns.length === 1 ? "" : "s"}`,
                                documentChatReadyCount > 0
                                  ? `${formatNumber(visibleDocumentChatReadyCount || documentChatReadyCount)} ready`
                                  : "",
                                visibleDocumentTotals.relationships > 0
                                  ? `${formatNumber(visibleDocumentTotals.relationships)} relations`
                                  : "",
                              ]
                                .filter(Boolean)
                                .join(" · ")
                            : "Processed documents appear here"}
                        </Typography>
                      </Box>
                      <Stack direction="row" spacing={0.75} sx={{ flexShrink: 0 }}>
                        {compactUploadIntake && !corpusUploadActive && (
                          <Button
                            component="label"
                            size="small"
                            variant="outlined"
                            startIcon={<UploadIcon sx={{ fontSize: 16 }} />}
                            disabled={corpusLoading}
                            sx={{
                              display: { xs: "inline-flex", md: "none" },
                              minHeight: 30,
                              px: 0.85,
                              borderRadius: "8px",
                              textTransform: "none",
                              fontSize: "0.74rem",
                              fontWeight: 850,
                            }}
                          >
                            Add
                            <input
                              hidden
                              type="file"
                              multiple
                              accept={GRAPHRAG_DOCUMENT_ACCEPT}
                              onChange={handleCorpusUploadFileChange}
                            />
                          </Button>
                        )}
                        <Tooltip title="Load recent documents" describeChild>
                          <span>
                            <IconButton
                              size="small"
                              aria-label="Load recent documents"
                              disabled={corpusLoading}
                              onClick={handleCorpusPipelineRuns}
                              sx={{
                                borderRadius: "8px",
                                border: "1px solid",
                                borderColor: alpha(accentColors.primary, 0.26),
                                color: accentColors.primary,
                              }}
                            >
                              {corpusLoading ? (
                                <CircularProgress size={16} />
                              ) : (
                                <RefreshIcon fontSize="small" />
                              )}
                            </IconButton>
                          </span>
                        </Tooltip>
                      </Stack>
                    </Box>
                    {(pipelineRuns.length > 4 || corpusRunSearch.trim()) && (
                      <TextField
                        value={corpusRunSearch}
                        onChange={(event) =>
                          setCorpusRunSearch(event.target.value)
                        }
                        placeholder="Find a document"
                        size="small"
                        fullWidth
                        sx={{
                        mb: 0.85,
                        "& .MuiOutlinedInput-root": {
                            height: { xs: 32, md: 34 },
                            borderRadius: "8px",
                            fontSize: "0.78rem",
                          },
                        }}
                        InputProps={{
                          startAdornment: (
                            <InputAdornment position="start">
                              <SearchIcon sx={{ fontSize: 17 }} />
                            </InputAdornment>
                          ),
                        }}
                      />
                    )}
                    {pipelineRuns.length > 0 ? (
                      <Box sx={{ maxHeight: { xs: "none", sm: 220, md: 520 }, overflow: "auto", pr: 0.25 }}>
                        <Stack spacing={0.7}>
                          {visibleDocumentRuns.map((documentGroup, index) => {
                            const {
                              run,
                              runs,
                              runCount,
                              domains,
                              chatReady: groupChatReady,
                            } = documentGroup;
                            const runId = asString(run.run_id);
                            const document = asRecord(run.document);
                            const filename = asString(
                              document.filename,
                              `Run ${index + 1}`,
                            );
                            const status = asString(run.status, "unknown");
                            const preferredRun =
                              preferredPipelineRunForDocumentChat(runs) || run;
                            const rowSummary =
                              summarizeGraphRAGDocumentPipelineRun(preferredRun);
                            const rowStatusColor = groupChatReady
                              ? accentColors.success
                              : pipelineStageStatusColor(status);
                            const latestRelationshipCount =
                              rowSummary.relationshipCount;
                            const latestChunkCount = rowSummary.chunkCount;
                            const rowChips = [
                              {
                                label: groupChatReady ? "Ready" : formatLabel(status),
                                color: rowStatusColor,
                                emphasis: groupChatReady,
                              },
                              runCount > 1
                                ? {
                                    label: `${formatNumber(runCount)} imports`,
                                    color: accentColors.primary,
                                  }
                                : null,
                              latestRelationshipCount > 0
                                ? {
                                    label: `${formatNumber(latestRelationshipCount)} rels`,
                                    color: accentColors.success,
                                  }
                                : null,
                              latestChunkCount > 0
                                ? {
                                    label: `${formatNumber(latestChunkCount)} chunks`,
                                    color: accentColors.info,
                                  }
                                : null,
                            ].filter(
                              (
                                chip,
                              ): chip is {
                                label: string;
                                color: string;
                                emphasis?: boolean;
                              } => Boolean(chip),
                            );
                            const isSelected = Boolean(
                              runs.some((candidate) => {
                                const candidateRunId = asString(candidate.run_id);
                                return (
                                  candidateRunId &&
                                  (candidateRunId === selectedCorpusPipelineRunId ||
                                    candidateRunId === activeCorpusPipelineRunId)
                                );
                              }),
                            );
                            const domainSummary =
                              domains.length > 1
                                ? `${domains.length} domains: ${domains.join(", ")}`
                                : domains[0] || "";
                            return (
                              <Box
                                key={runId || index}
                                role="button"
                                tabIndex={corpusLoading || !runId ? -1 : 0}
                                aria-disabled={corpusLoading || !runId}
                                onClick={() => {
                                  if (!corpusLoading && runId) {
                                    handleCorpusLoadPipelineRun(runId, {
                                      guideToSelectedPanel: true,
                                    });
                                  }
                                }}
                                onKeyDown={(event) => {
                                  if (
                                    (event.key === "Enter" ||
                                      event.key === " ") &&
                                    !corpusLoading &&
                                    runId
                                  ) {
                                    event.preventDefault();
                                    handleCorpusLoadPipelineRun(runId, {
                                      guideToSelectedPanel: true,
                                    });
                                  }
                                }}
                                sx={{
                                  textAlign: "left",
                                  borderRadius: "8px",
                                  border: "1px solid",
                                  borderColor: isSelected
                                    ? alpha(accentColors.primary, 0.66)
                                    : alpha(accentColors.slate, 0.22),
                                  px: { xs: 0.85, md: 1 },
                                  py: { xs: 0.72, md: 0.9 },
                                  cursor:
                                    corpusLoading || !runId
                                      ? "default"
                                      : "pointer",
                                  bgcolor: isSelected
                                    ? alpha(accentColors.primary, 0.1)
                                    : alpha(accentColors.slate, 0.025),
                                  transition:
                                    "background-color 120ms ease, border-color 120ms ease",
                                  "&:hover": {
                                    borderColor: alpha(
                                      accentColors.primary,
                                      0.42,
                                    ),
                                    bgcolor: alpha(accentColors.primary, 0.05),
                                  },
                                }}
                              >
                                <Box
                                  sx={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 1,
                                    minWidth: 0,
                                  }}
                                >
                                  <Box
                                    sx={{
                                      width: 34,
                                      height: { xs: 30, md: 34 },
                                      minWidth: { xs: 30, md: 34 },
                                      borderRadius: "8px",
                                      display: "flex",
                                      alignItems: "center",
                                      justifyContent: "center",
                                      color: isSelected
                                        ? accentColors.primary
                                        : accentColors.info,
                                      bgcolor: alpha(
                                        isSelected
                                          ? accentColors.primary
                                          : accentColors.info,
                                        0.11,
                                      ),
                                      flexShrink: 0,
                                    }}
                                  >
                                    <StorageIcon sx={{ fontSize: 18 }} />
                                  </Box>
                                  <Box sx={{ minWidth: 0, flex: 1 }}>
                                    <Typography
                                      variant="body2"
                                      fontWeight={850}
                                      noWrap
                                      title={
                                        domainSummary
                                          ? `${filename} · ${domainSummary}`
                                          : filename
                                      }
                                    >
                                      {filename}
                                    </Typography>
                                    <Box
                                      sx={{
                                        display: "flex",
                                        gap: 0.35,
                                        flexWrap: "wrap",
                                        mt: 0.35,
                                      }}
                                    >
                                      {rowChips.slice(0, 3).map((chip) => (
                                        <Chip
                                          key={`${runId || filename}-${chip.label}`}
                                          size="small"
                                          label={chip.label}
                                          variant={chip.emphasis ? "filled" : "outlined"}
                                          sx={{
                                            height: 18,
                                            maxWidth: 118,
                                            fontSize: "0.58rem",
                                            color: chip.emphasis
                                              ? "primary.contrastText"
                                              : chip.color,
                                            borderColor: alpha(chip.color, 0.34),
                                            bgcolor: chip.emphasis
                                              ? alpha(chip.color, 0.82)
                                              : alpha(chip.color, 0.055),
                                            "& .MuiChip-label": {
                                              px: 0.58,
                                              minWidth: 0,
                                              overflow: "hidden",
                                              textOverflow: "ellipsis",
                                            },
                                          }}
                                        />
                                      ))}
                                    </Box>
                                  </Box>
                                  <Box sx={{ flexShrink: 0 }}>
                                    <Tooltip
                                      title={
                                        runCount > 1
                                          ? `Ask using ${formatNumber(runCount)} imports for ${filename}`
                                          : `Ask about ${filename}`
                                      }
                                      describeChild
                                    >
                                      <span>
                                        <IconButton
                                          size="small"
                                          disabled={runs.length === 0}
                                          aria-label={
                                            runCount > 1
                                              ? `Ask using ${formatNumber(runCount)} imports for ${filename}`
                                              : `Ask about ${filename}`
                                          }
                                          onClick={(event) => {
                                            event.stopPropagation();
                                            handleAskChatFromPipelineRuns(runs);
                                          }}
                                          sx={{
                                            width: 32,
                                            height: 32,
                                            borderRadius: "8px",
                                            border: "1px solid",
                                            borderColor: alpha(
                                              groupChatReady
                                                ? accentColors.primary
                                                : accentColors.slate,
                                              groupChatReady ? 0.42 : 0.24,
                                            ),
                                            color: groupChatReady
                                              ? accentColors.primary
                                              : "text.secondary",
                                            bgcolor: alpha(
                                              groupChatReady
                                                ? accentColors.primary
                                                : accentColors.slate,
                                              groupChatReady ? 0.12 : 0.045,
                                            ),
                                            "&:hover": {
                                              bgcolor: alpha(accentColors.primary, 0.18),
                                            },
                                          }}
                                        >
                                          <ChatIcon sx={{ fontSize: 17 }} />
                                        </IconButton>
                                      </span>
                                    </Tooltip>
                                  </Box>
                                </Box>
                              </Box>
                            );
                          })}
                          {visibleDocumentRuns.length === 0 && (
                            <Box
                              sx={{
                                border: "1px dashed",
                                borderColor: alpha(accentColors.primary, 0.24),
                                borderRadius: "8px",
                                px: 1,
                                py: 1,
                                color: "text.secondary",
                              }}
                            >
                              <Typography variant="caption">
                                No documents match this search.
                              </Typography>
                            </Box>
                          )}
                        </Stack>
                      </Box>
                    ) : (
                      <Box
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          gap: 1,
                          borderRadius: "8px",
                          border: "1px dashed",
                          borderColor: alpha(accentColors.primary, 0.2),
                          px: 1,
                          py: 1.15,
                          color: "text.secondary",
                        }}
                      >
                        <StorageIcon sx={{ fontSize: 18, flexShrink: 0 }} />
                        <Typography variant="caption">
                          Processed documents appear here.
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </Stack>
              </Box>
              <Box
                sx={{
                  display: { xs: corpusUploadActive ? "none" : "block", md: "block" },
                  gridArea: "selected",
                  minWidth: 0,
                  width: "100%",
                }}
              >
                {corpusPipelineResult ? (
                  <Stack spacing={1} sx={{ width: "100%" }}>
                    {corpusHasBatchResult && (
                      <Alert
                        severity={corpusBatchFailed > 0 ? "warning" : "success"}
                        variant="outlined"
                        sx={{
                          alignItems: "center",
                          py: { xs: 0.45, sm: 0.65 },
                          "& .MuiAlert-message": {
                            width: "100%",
                            py: { xs: 0.25, sm: 0.5 },
                          },
                        }}
                      >
                        <Box
                          sx={{
                            display: "grid",
                            gridTemplateColumns:
                              corpusBatchRunIds.length > 1
                                ? "minmax(0, 1fr) auto"
                                : "minmax(0, 1fr)",
                            alignItems: "center",
                            gap: { xs: 0.75, sm: 1 },
                            minWidth: 0,
                          }}
                        >
                          <Box sx={{ minWidth: 0 }}>
                            <Typography variant="body2" noWrap fontWeight={800}>
                              {corpusBatchFailed > 0
                                ? "Processed with issues"
                                : "Documents processed"}
                            </Typography>
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              noWrap
                              sx={{ display: "block" }}
                            >
                              {[
                                `${formatNumber(corpusBatchProcessed)}/${formatNumber(corpusBatchTotal)} ready`,
                                corpusBatchGraphRelationshipCount > 0
                                  ? `${formatNumber(corpusBatchGraphRelationshipCount)} relationships`
                                  : "",
                              ]
                                .filter(Boolean)
                                .join(" · ")}
                            </Typography>
                          </Box>
                          {corpusBatchRunIds.length > 1 && (
                            <Button
                              size="small"
                              variant="contained"
                              startIcon={<ChatIcon fontSize="small" />}
                              aria-label="Ask chat across processed documents"
                              onClick={() => handleAskChatAcrossBatch()}
                              sx={{
                                borderRadius: "8px",
                                minHeight: { xs: 32, sm: 34 },
                                minWidth: { xs: 74, sm: 104 },
                                px: { xs: 0.85, sm: 1.25 },
                                textTransform: "none",
                                boxShadow: "none",
                                "& .MuiButton-startIcon": {
                                  mr: { xs: 0.35, sm: 0.55 },
                                },
                              }}
                            >
                              <Box component="span" sx={{ display: { xs: "none", sm: "inline" } }}>
                                Ask Chat
                              </Box>
                              <Box component="span" sx={{ display: { xs: "inline", sm: "none" } }}>
                                Ask
                              </Box>
                            </Button>
                          )}
                        </Box>
                      </Alert>
                    )}
                    {false && corpusHasBatchResult && (
                      <Alert
                        severity={corpusBatchFailed > 0 ? "warning" : "success"}
                        variant="outlined"
                        sx={{ alignItems: "flex-start" }}
                      >
                        <Stack spacing={0.75}>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 0.75,
                              flexWrap: "wrap",
                            }}
                          >
                            <Typography variant="body2" fontWeight={800}>
                              {corpusBatchFailed > 0
                                ? "Batch completed with issues"
                                : "Batch ready for chat"}
                            </Typography>
                            <Chip
                              label={`${formatNumber(corpusBatchProcessed)}/${formatNumber(corpusBatchTotal)} processed`}
                              size="small"
                              color="success"
                              sx={{ height: 22 }}
                            />
                            {corpusBatchGraphRelationshipCount > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchGraphRelationshipCount)} relationships`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchGraphPathCount > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchGraphPathCount)} paths`}
                                size="small"
                                color="info"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchChatReady > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchChatReady)} chat-ready`}
                                size="small"
                                color="success"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchRunIds.length > 1 && (
                              <Button
                                size="small"
                                variant="contained"
                                startIcon={<ChatIcon fontSize="small" />}
                                onClick={() => handleAskChatAcrossBatch()}
                                sx={{
                                  minHeight: 24,
                                  px: 1,
                                  py: 0.15,
                                  borderRadius: "8px",
                                  textTransform: "none",
                                  boxShadow: "none",
                                }}
                              >
                                Ask Chat
                              </Button>
                            )}
                          </Box>
                          {(corpusBatchGraphSharedEntities.length > 0 ||
                            corpusBatchGraphBridgeEntities.length > 0 ||
                            corpusBatchGraphPaths.length > 0) && (
                            <Stack spacing={0.5}>
                              {corpusBatchGraphPaths
                                .slice(0, 2)
                                .map((path, index) => {
                                  const bridge = asString(
                                    path.bridge,
                                    "Bridge",
                                  );
                                  const source = asString(
                                    path.source,
                                    "Source",
                                  );
                                  const target = asString(
                                    path.target,
                                    "Target",
                                  );
                                  const label = formatPipelineGraphChainLabel([source, bridge, target]);

                                  return (
                                    <Typography
                                      key={`${label}-${index}`}
                                      variant="caption"
                                      color="text.secondary"
                                      noWrap
                                      title={asString(path.text, label)}
                                    >
                                      Path: {label}
                                    </Typography>
                                  );
                                })}
                              <Box
                                sx={{
                                  display: "flex",
                                  gap: 0.5,
                                  flexWrap: "wrap",
                                }}
                              >
                                {corpusBatchGraphSharedEntities
                                  .slice(0, 3)
                                  .map((entity, index) => (
                                    <Chip
                                      key={`${asString(entity.name, "entity")}-${index}`}
                                      label={asString(
                                        entity.name,
                                        `Shared entity ${index + 1}`,
                                      )}
                                      size="small"
                                      variant="outlined"
                                      sx={{ height: 22, maxWidth: 180 }}
                                    />
                                  ))}
                                {corpusBatchGraphBridgeEntities
                                  .slice(0, 2)
                                  .map((entity, index) => (
                                    <Chip
                                      key={`${asString(entity.name, "bridge")}-${index}`}
                                      label={`Bridge: ${asString(entity.name, `Bridge ${index + 1}`)}`}
                                      size="small"
                                      variant="outlined"
                                      sx={{ height: 22, maxWidth: 200 }}
                                    />
                                  ))}
                              </Box>
                            </Stack>
                          )}
                          <Typography variant="caption" color="text.secondary">
                            {corpusBatchHandoffMessage ||
                              corpusBatchManifestNextStep ||
                              "Open chat to ask over this upload with graph context."}
                          </Typography>
                        </Stack>
                      </Alert>
                    )}
                    {false && corpusHasBatchResult && (
                      <Alert
                        severity={corpusBatchFailed > 0 ? "warning" : "success"}
                        variant="outlined"
                        sx={{ alignItems: "flex-start" }}
                      >
                        <Stack spacing={0.75}>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 0.75,
                              flexWrap: "wrap",
                            }}
                          >
                            <Typography variant="body2" fontWeight={700}>
                              {corpusBatchHandoffLabel ||
                                (corpusBatchFailed > 0
                                  ? "Batch completed with issues"
                                  : "Batch upload complete")}
                            </Typography>
                            {corpusBatchHandoffStatus && (
                              <Chip
                                label={formatLabel(corpusBatchHandoffStatus)}
                                size="small"
                                color={
                                  corpusBatchHandoffStatus ===
                                  "remote_graph_submitted"
                                    ? "success"
                                    : "warning"
                                }
                                sx={{ height: 22 }}
                              />
                            )}
                            <Chip
                              label={`${formatNumber(corpusBatchProcessed)}/${formatNumber(corpusBatchTotal)} processed`}
                              size="small"
                              sx={{ height: 22 }}
                            />
                            {corpusBatchManifestTotalBytes > 0 && (
                              <Chip
                                label={`${formatFileSize(corpusBatchManifestTotalBytes)} uploaded`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchManifestLargestFilename &&
                              corpusBatchManifestLargestBytes > 0 && (
                                <Chip
                                  label={`Largest ${corpusBatchManifestLargestFilename} (${formatFileSize(corpusBatchManifestLargestBytes)})`}
                                  title={corpusBatchManifestLargestFilename}
                                  size="small"
                                  variant="outlined"
                                  sx={{
                                    height: 22,
                                    maxWidth: 260,
                                    "& .MuiChip-label": {
                                      minWidth: 0,
                                      overflow: "hidden",
                                      textOverflow: "ellipsis",
                                    },
                                  }}
                                />
                              )}
                            {corpusBatchFailed > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchFailed)} failed`}
                                size="small"
                                color="warning"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchDurationMs > 0 && (
                              <Chip
                                label={`Elapsed ${formatDuration(corpusBatchDurationMs)}`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {asNumber(corpusBatchThroughput) > 0 && (
                              <Chip
                                label={formatThroughput(corpusBatchThroughput)}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchMaxConcurrency > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchMaxConcurrency)} workers`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchPeakConcurrency > 0 &&
                              corpusBatchPeakConcurrency !==
                                corpusBatchMaxConcurrency && (
                                <Chip
                                  label={`Peak ${formatNumber(corpusBatchPeakConcurrency)}`}
                                  size="small"
                                  variant="outlined"
                                  sx={{ height: 22 }}
                                />
                              )}
                            {corpusBatchRemoteSubmitted > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchRemoteSubmitted)} remote`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchEndpointAttemptTotal > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchEndpointAttemptTotal)} endpoint attempts`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchEndpointFallbacks > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchEndpointFallbacks)} used fallback`}
                                size="small"
                                color="info"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchDeploymentBlocked > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchDeploymentBlocked)} deployment blocked`}
                                size="small"
                                color="warning"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchLocalPreviewOnly > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchLocalPreviewOnly)} local traces`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchTraceReady > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchTraceReady)} trace-ready`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchChatReady > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchChatReady)} chat-ready`}
                                size="small"
                                color="success"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchRemoteWaiting > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchRemoteWaiting)} waiting remote`}
                                size="small"
                                color="warning"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchGraphRelationshipCount > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchGraphRelationshipCount)} graph relationships`}
                                size="small"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchGraphSharedEntityCount > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchGraphSharedEntityCount)} shared entities`}
                                size="small"
                                color="info"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchGraphBridgeEntityCount > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchGraphBridgeEntityCount)} bridge entities`}
                                size="small"
                                color="info"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchGraphPathCount > 0 && (
                              <Chip
                                label={`${formatNumber(corpusBatchGraphPathCount)} cross-doc paths`}
                                size="small"
                                color="info"
                                variant="outlined"
                                sx={{ height: 22 }}
                              />
                            )}
                            {corpusBatchRunIds.length > 1 && (
                              <Button
                                size="small"
                                variant="contained"
                                startIcon={<ChatIcon fontSize="small" />}
                                onClick={() => handleAskChatAcrossBatch()}
                                sx={{
                                  minHeight: 24,
                                  px: 1,
                                  py: 0.15,
                                  borderRadius: "8px",
                                  textTransform: "none",
                                  boxShadow: "none",
                                }}
                              >
                                Ask Chat Across Batch
                              </Button>
                            )}
                          </Box>
                          {(corpusBatchHandoffMessage ||
                            corpusBatchHandoffNextStep ||
                            corpusBatchManifestNextStep) && (
                            <Typography
                              variant="caption"
                              color="text.secondary"
                            >
                              {[
                                corpusBatchHandoffMessage,
                                corpusBatchHandoffNextStep ||
                                  corpusBatchManifestNextStep,
                              ]
                                .filter(Boolean)
                                .join(" ")}
                            </Typography>
                          )}
                          {corpusBatchManifestLanes.length > 0 && (
                            <Box
                              sx={{
                                display: "flex",
                                gap: 0.5,
                                flexWrap: "wrap",
                              }}
                            >
                              {corpusBatchManifestLanes.map((lane) => {
                                const id = asString(lane.id);
                                const label = asString(
                                  lane.label,
                                  formatLabel(id || "lane"),
                                );
                                const count = asNumber(lane.count);
                                const waiting = asNumber(lane.waiting);
                                const complete = lane.complete === true;
                                const ready = lane.ready === true;
                                return (
                                  <Chip
                                    key={id || label}
                                    label={`${label}: ${formatNumber(count)}${waiting > 0 ? ` · ${formatNumber(waiting)} waiting` : ""}`}
                                    size="small"
                                    color={
                                      complete
                                        ? "success"
                                        : waiting > 0
                                          ? "warning"
                                          : ready
                                            ? "info"
                                            : "default"
                                    }
                                    variant="outlined"
                                    sx={{ height: 22 }}
                                  />
                                );
                              })}
                            </Box>
                          )}
                          {(corpusBatchPayloadKinds.length > 0 ||
                            corpusBatchEvidenceChips.length > 0) && (
                            <Box
                              sx={{
                                display: "flex",
                                gap: 0.5,
                                flexWrap: "wrap",
                              }}
                            >
                              {corpusBatchManifestDomains
                                .slice(0, 3)
                                .map((domain) => (
                                  <Chip
                                    key={domain}
                                    label={`Domain ${domain}`}
                                    size="small"
                                    variant="outlined"
                                    sx={{ height: 22 }}
                                  />
                                ))}
                              {corpusBatchPayloadKinds.map(
                                ({ kind, count }) => (
                                  <Chip
                                    key={kind}
                                    label={`${formatNumber(count)} ${formatLabel(kind)}`}
                                    size="small"
                                    variant="outlined"
                                    sx={{ height: 22 }}
                                  />
                                ),
                              )}
                              {corpusBatchInteractionModes.map(
                                ({ mode, count }) => (
                                  <Chip
                                    key={mode}
                                    label={`${formatNumber(count)} ${formatLabel(mode)}`}
                                    size="small"
                                    color="success"
                                    variant="outlined"
                                    sx={{ height: 22 }}
                                  />
                                ),
                              )}
                              {corpusBatchManifestRehearsals > 0 && (
                                <Tooltip
                                  title={`${formatNumber(corpusBatchManifestRehearsedAnchors)} anchor match${corpusBatchManifestRehearsedAnchors === 1 ? "" : "es"} and ${formatNumber(corpusBatchManifestRehearsedChunks)} chunk match${corpusBatchManifestRehearsedChunks === 1 ? "" : "es"} across starter prompts`}
                                >
                                  <Chip
                                    label={`${formatNumber(corpusBatchManifestRehearsals)} rehearsals`}
                                    size="small"
                                    color="success"
                                    variant="outlined"
                                    sx={{ height: 22 }}
                                  />
                                </Tooltip>
                              )}
                              {corpusBatchEvidenceChips.map((chip) => (
                                <Chip
                                  key={chip.label}
                                  label={`${formatNumber(chip.value)} ${chip.label}`}
                                  size="small"
                                  variant="outlined"
                                  sx={{ height: 22 }}
                                />
                              ))}
                            </Box>
                          )}
                          {(corpusBatchGraphSharedEntities.length > 0 ||
                            corpusBatchGraphTopRelationships.length > 0 ||
                            corpusBatchGraphBridgeEntities.length > 0 ||
                            corpusBatchGraphPaths.length > 0) && (
                            <Box
                              sx={{
                                display: "flex",
                                gap: 0.5,
                                flexWrap: "wrap",
                              }}
                            >
                              {corpusBatchGraphSharedEntities
                                .slice(0, 4)
                                .map((entity, index) => {
                                  const name = asString(
                                    entity.name,
                                    `Shared entity ${index + 1}`,
                                  );
                                  const runCount = asNumber(entity.run_count);
                                  return (
                                    <Chip
                                      key={`${name}-${index}`}
                                      label={`${name}${runCount > 0 ? ` (${formatNumber(runCount)} files)` : ""}`}
                                      title={name}
                                      size="small"
                                      color="info"
                                      variant="outlined"
                                      sx={{
                                        height: 22,
                                        maxWidth: 240,
                                        "& .MuiChip-label": {
                                          minWidth: 0,
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                        },
                                      }}
                                    />
                                  );
                                })}
                              {corpusBatchGraphBridgeEntities
                                .slice(0, 4)
                                .map((entity, index) => {
                                  const name = asString(
                                    entity.name,
                                    `Bridge ${index + 1}`,
                                  );
                                  const relationshipCount = asNumber(
                                    entity.relationship_count,
                                  );
                                  return (
                                    <Chip
                                      key={`bridge-${name}-${index}`}
                                      label={`Bridge: ${name}${relationshipCount > 0 ? ` (${formatNumber(relationshipCount)} links)` : ""}`}
                                      title={name}
                                      size="small"
                                      color="info"
                                      variant="outlined"
                                      sx={{
                                        height: 22,
                                        maxWidth: 260,
                                        "& .MuiChip-label": {
                                          minWidth: 0,
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                        },
                                      }}
                                    />
                                  );
                                })}
                              {corpusBatchGraphPaths
                                .slice(0, 4)
                                .map((path, index) => {
                                  const bridge = asString(
                                    path.bridge,
                                    "Bridge",
                                  );
                                  const source = asString(
                                    path.source,
                                    "Source",
                                  );
                                  const target = asString(
                                    path.target,
                                    "Target",
                                  );
                                  const label = formatPipelineGraphChainLabel([source, bridge, target]);
                                  return (
                                    <Chip
                                      key={`path-${source}-${bridge}-${target}-${index}`}
                                      label={`Path: ${label}`}
                                      title={asString(path.text, label)}
                                      size="small"
                                      color="info"
                                      variant="outlined"
                                      sx={{
                                        height: 22,
                                        maxWidth: 320,
                                        "& .MuiChip-label": {
                                          minWidth: 0,
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                        },
                                      }}
                                    />
                                  );
                                })}
                              {corpusBatchGraphTopRelationships
                                .slice(0, 4)
                                .map((relationship, index) => {
                                  const source = asString(
                                    relationship.source,
                                    "Source",
                                  );
                                  const relation = asString(
                                    relationship.relation,
                                    "related_to",
                                  );
                                  const target = asString(
                                    relationship.target,
                                    "Target",
                                  );
                                  const label = formatPipelineGraphPairLabel(source, target);
                                  return (
                                    <Chip
                                      key={`${source}-${relation}-${target}-${index}`}
                                      label={`${formatLabel(relation)}: ${label}`}
                                      title={`${formatLabel(relation)}: ${label}`}
                                      size="small"
                                      variant="outlined"
                                      sx={{
                                        height: 22,
                                        maxWidth: 280,
                                        "& .MuiChip-label": {
                                          minWidth: 0,
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                        },
                                      }}
                                    />
                                  );
                                })}
                            </Box>
                          )}
                          {corpusBatchGraphQuestions.length > 0 && (
                            <Box
                              sx={{
                                display: "flex",
                                gap: 0.5,
                                flexWrap: "wrap",
                              }}
                            >
                              {corpusBatchGraphQuestions
                                .slice(0, 5)
                                .map((question) => {
                                  const questionLabel = graphRAGQuestionLabel(question);
                                  return (
                                  <Tooltip
                                    key={question.id}
                                    title={question.prompt}
                                    describeChild
                                  >
                                    <Button
                                      size="small"
                                      variant={
                                        question.kind === "summary"
                                          ? "contained"
                                          : "outlined"
                                      }
                                      color={
                                        question.kind === "summary"
                                          ? "primary"
                                          : "inherit"
                                      }
                                      startIcon={<ChatIcon fontSize="small" />}
                                      onClick={() =>
                                        handleAskChatAcrossBatch(
                                          question.prompt,
                                          {},
                                          questionLabel,
                                        )
                                      }
                                      sx={{
                                        minHeight: 24,
                                        maxWidth: { xs: "100%", sm: 260 },
                                        px: 1,
                                        py: 0.15,
                                        borderRadius: "8px",
                                        textTransform: "none",
                                        boxShadow: "none",
                                        justifyContent: "flex-start",
                                        "& .MuiButton-startIcon": {
                                          flexShrink: 0,
                                        },
                                      }}
                                    >
                                      <Box
                                        component="span"
                                        sx={{
                                          minWidth: 0,
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                          whiteSpace: "nowrap",
                                        }}
                                      >
                                        {questionLabel}
                                      </Box>
                                    </Button>
                                  </Tooltip>
                                  );
                                })}
                            </Box>
                          )}
                          {corpusBatchManifestRows.length > 0 && (
                            <Box
                              sx={{
                                display: "flex",
                                gap: 0.5,
                                flexWrap: "wrap",
                              }}
                            >
                              {corpusBatchManifestRows
                                .slice(0, 8)
                                .map((result, index) => {
                                  const filename = asString(
                                    result.filename,
                                    `File ${index + 1}`,
                                  );
                                  const status = asString(
                                    result.status,
                                    result.success === false
                                      ? "error"
                                      : "complete",
                                  );
                                  const success = result.success !== false;
                                  const traceReady =
                                    result.trace_ready === true;
                                  const chatReady = result.chat_ready === true;
                                  const remoteSubmitted =
                                    result.remote_submitted === true;
                                  const remoteWaiting =
                                    result.remote_waiting === true;
                                  const payloadKind = asString(
                                    result.payload_kind,
                                  );
                                  const interactionMode = asString(
                                    result.interaction_mode,
                                  );
                                  const graphAnchorCount = asNumber(
                                    result.graph_anchor_count,
                                  );
                                  const rehearsalCount = asNumber(
                                    result.retrieval_rehearsal_count,
                                  );
                                  const deploymentStatus = asString(
                                    result.deployment_status,
                                  );
                                  const deploymentSeverity = asString(
                                    result.deployment_severity,
                                  );
                                  const endpointAttemptCount = asNumber(
                                    result.endpoint_attempt_count,
                                  );
                                  const endpointFallbackUsed =
                                    result.endpoint_fallback_used === true;
                                  const selectedEndpoint = asString(
                                    result.selected_endpoint,
                                  );
                                  const compactEndpoint = formatCompactEndpoint(
                                    selectedEndpoint ||
                                      asString(result.target_url),
                                  );
                                  const readinessLabel = remoteSubmitted
                                    ? "remote"
                                    : chatReady
                                      ? "chat-ready"
                                      : traceReady
                                        ? "trace-ready"
                                        : remoteWaiting
                                          ? "waiting remote"
                                          : "";
                                  const color = success
                                    ? remoteWaiting
                                      ? accentColors.warning
                                      : chatReady || remoteSubmitted
                                        ? accentColors.success
                                        : accentColors.info
                                    : accentColors.rose;
                                  const runId = asString(result.run_id);
                                  const isSelected =
                                    runId &&
                                    asString(selectedCorpusBatchRun?.run_id) ===
                                      runId;

                                  return (
                                    <Chip
                                      key={runId || `${filename}-${index}`}
                                      title={[
                                        runId
                                          ? `Show processing details for ${filename}`
                                          : filename,
                                        payloadKind
                                          ? `Payload ${formatLabel(payloadKind)}`
                                          : "",
                                        interactionMode
                                          ? `Mode ${formatLabel(interactionMode)}`
                                          : "",
                                        graphAnchorCount > 0
                                          ? `${formatNumber(graphAnchorCount)} graph anchors`
                                          : "",
                                        rehearsalCount > 0
                                          ? `${formatNumber(rehearsalCount)} rehearsals`
                                          : "",
                                        deploymentStatus
                                          ? `Deployment ${formatLabel(deploymentStatus)}`
                                          : "",
                                        endpointAttemptCount > 0
                                          ? `${formatNumber(endpointAttemptCount)} endpoint attempts`
                                          : "",
                                        endpointFallbackUsed
                                          ? "Endpoint fallback used"
                                          : "",
                                        compactEndpoint
                                          ? `Endpoint ${compactEndpoint}`
                                          : "",
                                        asString(result.next_step),
                                      ]
                                        .filter(Boolean)
                                        .join(" · ")}
                                      label={[
                                        `${filename}: ${formatLabel(status)}`,
                                        readinessLabel,
                                        deploymentStatus &&
                                        deploymentStatus !== status
                                          ? formatLabel(deploymentStatus)
                                          : "",
                                        endpointFallbackUsed ? "fallback" : "",
                                        endpointAttemptCount > 1
                                          ? `${formatNumber(endpointAttemptCount)} tries`
                                          : "",
                                      ]
                                        .filter(Boolean)
                                        .join(" · ")}
                                      size="small"
                                      color={
                                        deploymentSeverity === "error"
                                          ? "warning"
                                          : endpointFallbackUsed
                                            ? "info"
                                            : undefined
                                      }
                                      variant={
                                        isSelected ? "filled" : "outlined"
                                      }
                                      clickable={Boolean(runId)}
                                      onClick={
                                        runId
                                          ? () =>
                                              setSelectedCorpusPipelineRunId(
                                                runId,
                                              )
                                          : undefined
                                      }
                                      sx={{
                                        height: 22,
                                        maxWidth: 240,
                                        borderColor: isSelected
                                          ? color
                                          : alpha(color, 0.42),
                                        bgcolor: isSelected
                                          ? alpha(color, 0.22)
                                          : alpha(color, 0.08),
                                        "& .MuiChip-label": {
                                          minWidth: 0,
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                        },
                                      }}
                                    />
                                  );
                                })}
                              {corpusBatchManifestRows.length > 8 && (
                                <Chip
                                  label={`+${corpusBatchManifestRows.length - 8} more`}
                                  size="small"
                                  variant="outlined"
                                  sx={{ height: 22 }}
                                />
                              )}
                            </Box>
                          )}
                        </Stack>
                      </Alert>
                    )}
                    <Box
                      ref={selectedDocumentPanelRef}
                      sx={{
                        gridArea: { md: "selected" },
                        minWidth: 0,
                        width: "100%",
                        alignSelf: { md: "start" },
                        maxHeight: { md: "calc(100vh - 112px)" },
                        overflowY: { md: "auto" },
                        scrollbarWidth: "thin",
                        border: "1px solid",
                        borderColor: alpha(accentColors.primary, 0.32),
                        borderRadius: "8px",
                        p: { xs: 1, md: 1.35 },
                        bgcolor: "background.paper",
                      }}
                    >
                      <Stack spacing={{ xs: 0.85, sm: 1.1 }}>
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "flex-start",
                            justifyContent: "space-between",
                            gap: 1,
                            flexWrap: "wrap",
                          }}
                        >
                          <Box sx={{ minWidth: 0, flex: 1 }}>
                            {activePanelDisplayEyebrow && (
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{ display: "block", fontWeight: 900 }}
                              >
                                {activePanelDisplayEyebrow}
                              </Typography>
                            )}
                            <Stack
                              direction="row"
                              spacing={0.75}
                              alignItems="center"
                              flexWrap="wrap"
                              useFlexGap
                              sx={{ mb: 0.25 }}
                            >
                              <Typography
                                variant="subtitle2"
                                noWrap
                                title={activePanelDisplayTitle}
                                sx={{ minWidth: 0, fontWeight: 900 }}
                              >
                                {activePanelDisplayTitle}
                              </Typography>
                              {showActiveEvidenceChip && (
                                <Chip
                                  size="small"
                                  label={activeEvidenceChipLabel}
                                  color="success"
                                  variant="outlined"
                                  sx={{ height: 22 }}
                                />
                              )}
                              {activeDocumentHistoryLabel && (
                                <Chip
                                  size="small"
                                  label={activeDocumentHistoryLabel}
                                  variant="outlined"
                                  sx={{
                                    height: 22,
                                    color: "text.secondary",
                                    borderColor: alpha(accentColors.slate, 0.28),
                                  }}
                                />
                              )}
                            </Stack>
                            <Typography
                              variant="body2"
                              color="text.secondary"
                              title={activePanelDisplaySubtitle}
                              sx={{
                                display: { xs: "none", sm: "-webkit-box" },
                                overflow: "hidden",
                                WebkitBoxOrient: "vertical",
                                WebkitLineClamp: 1,
                                lineHeight: 1.35,
                              }}
                            >
                              {activePanelDisplaySubtitle}
                            </Typography>
                          </Box>
                          {!corpusUploadActive &&
                            !workbenchPanelState.showDocumentInsights && (
                            <Button
                              size="small"
                              variant="outlined"
                              startIcon={
                                workbenchPanelState.primaryAction === "processing" ? (
                                  <CircularProgress size={16} />
                                ) : (
                                  <ChatIcon fontSize="small" />
                                )
                              }
                              onClick={() =>
                                corpusHasBatchResult
                                  ? handleAskChatAcrossBatch()
                                  : activeDocumentGroup
                                    ? handleAskChatFromPipelineRuns(
                                        activeDocumentGroup.runs,
                                      )
                                    : activeCorpusPipelineRunId
                                      ? handleAskChatFromPipelineRun(
                                          activeCorpusPipelineRecord,
                                        )
                                      : undefined
                              }
                              disabled={
                                !corpusHasBatchResult &&
                                !activeDocumentGroup &&
                                !activeCorpusPipelineRunId
                              }
                              sx={{
                                borderRadius: "8px",
                                minHeight: { xs: 32, sm: 34 },
                                px: { xs: 1, sm: 1.25 },
                                py: 0.35,
                                textTransform: "none",
                                fontSize: { xs: "0.78rem", sm: "0.82rem" },
                                fontWeight: 900,
                                boxShadow: "none",
                                flexShrink: 0,
                                width: "auto",
                                borderColor: alpha(accentColors.primary, 0.38),
                                bgcolor: alpha(accentColors.primary, 0.08),
                                color: accentColors.primary,
                                "&:hover": {
                                  borderColor: alpha(accentColors.primary, 0.56),
                                  bgcolor: alpha(accentColors.primary, 0.14),
                                },
                              }}
                            >
                              {workbenchPanelState.primaryActionLabel}
                            </Button>
                          )}
                        </Box>

                        {corpusUploadActive && (
                          <Box
                            sx={{
                              order: 0,
                              display: "grid",
                              gap: 0.85,
                              border: "1px solid",
                              borderColor: alpha(accentColors.success, 0.34),
                              borderRadius: "8px",
                              px: { xs: 0.9, sm: 1 },
                              py: { xs: 0.85, sm: 0.95 },
                              bgcolor: alpha(accentColors.success, 0.035),
                            }}
                          >
                            <Box
                              sx={{
                                display: "grid",
                                gridTemplateColumns: {
                                  xs: "minmax(0, 1fr)",
                                  sm: "minmax(0, 1fr) auto",
                                },
                                alignItems: "center",
                                gap: 0.85,
                                minWidth: 0,
                              }}
                            >
                              <Box
                                sx={{
                                  display: "grid",
                                  gridTemplateColumns: "auto minmax(0, 1fr)",
                                  alignItems: "center",
                                  gap: 0.75,
                                  minWidth: 0,
                                }}
                              >
                                <Box
                                  sx={{
                                    width: 32,
                                    height: 32,
                                    borderRadius: "8px",
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    color: accentColors.success,
                                    bgcolor: alpha(accentColors.success, 0.12),
                                  }}
                                >
                                  <UploadIcon sx={{ fontSize: 18 }} />
                                </Box>
                                <Box sx={{ minWidth: 0 }}>
                                  <Typography
                                    variant="body2"
                                    sx={{ fontWeight: 850, lineHeight: 1.28 }}
                                  >
                                    {workbenchPanelState.uploadNotice}
                                  </Typography>
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    noWrap
                                    title={corpusSelectedFileNames}
                                    sx={{ display: "block" }}
                                  >
                                    {corpusUploadManifest.sizeAndKindSummary}
                                  </Typography>
                                </Box>
                              </Box>
                              <Button
                                size="small"
                                variant="contained"
                                aria-label={
                                  corpusHasMultipleFiles
                                    ? `Process ${formatNumber(corpusUploadFileCount)} selected uploads`
                                    : "Process selected upload"
                                }
                                startIcon={
                                  corpusLoading ? (
                                    <CircularProgress size={15} />
                                  ) : (
                                    <RunIcon fontSize="small" />
                                  )
                                }
                                disabled={corpusLoading || corpusUploadFileCount === 0}
                                onClick={
                                  corpusHasMultipleFiles
                                    ? handleCorpusPipelineBatchUpload
                                    : handleCorpusPipelineUpload
                                }
                                sx={{
                                  justifySelf: { xs: "stretch", sm: "end" },
                                  minHeight: 34,
                                  px: 1.25,
                                  borderRadius: "8px",
                                  textTransform: "none",
                                  fontSize: "0.82rem",
                                  fontWeight: 900,
                                  boxShadow: "none",
                                }}
                              >
                                {corpusProcessButtonLabel}
                              </Button>
                            </Box>
                            <Box
                              sx={{
                                display: "grid",
                                gridTemplateColumns: {
                                  xs: "minmax(0, 1fr)",
                                  md: "repeat(2, minmax(0, 1fr))",
                                },
                                gap: 0.55,
                                minWidth: 0,
                              }}
                            >
                              {corpusUploadPreviewFiles.map((file) => (
                                <Box
                                  key={`${file.name}-${file.size}`}
                                  sx={{
                                    display: "grid",
                                    gridTemplateColumns: "auto minmax(0, 1fr) auto",
                                    alignItems: "center",
                                    gap: 0.55,
                                    minWidth: 0,
                                    borderTop: "1px solid",
                                    borderColor: alpha(accentColors.slate, 0.16),
                                    pt: 0.55,
                                  }}
                                >
                                  <Chip
                                    size="small"
                                    label={formatFileKind(file.name)}
                                    variant="outlined"
                                    sx={{
                                      height: 20,
                                      maxWidth: 82,
                                      fontSize: "0.62rem",
                                      color: accentColors.info,
                                      borderColor: alpha(accentColors.info, 0.34),
                                      bgcolor: alpha(accentColors.info, 0.06),
                                    }}
                                  />
                                  <Typography
                                    variant="caption"
                                    noWrap
                                    title={file.name}
                                    sx={{ minWidth: 0, fontWeight: 850 }}
                                  >
                                    {file.name}
                                  </Typography>
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    noWrap
                                    sx={{ justifySelf: "end" }}
                                  >
                                    {formatFileSize(file.size)}
                                  </Typography>
                                </Box>
                              ))}
                              {corpusUploadOverflowCount > 0 && (
                                <Chip
                                  size="small"
                                  label={`+${formatNumber(corpusUploadOverflowCount)} more`}
                                  variant="outlined"
                                  sx={{ height: 22, justifySelf: "start" }}
                                />
                              )}
                            </Box>
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 0.45,
                                flexWrap: "wrap",
                                minWidth: 0,
                              }}
                            >
                              {corpusUploadManifest.pipelineSteps.map((step, index) => (
                                <Chip
                                  key={step.label}
                                  size="small"
                                  label={`${index + 1}. ${step.label}`}
                                  title={step.detail}
                                  variant="outlined"
                                  sx={{
                                    height: 20,
                                    maxWidth: 112,
                                    fontSize: "0.62rem",
                                    color:
                                      index === corpusUploadManifest.pipelineSteps.length - 1
                                        ? accentColors.success
                                        : accentColors.primary,
                                    borderColor: alpha(
                                      index === corpusUploadManifest.pipelineSteps.length - 1
                                        ? accentColors.success
                                        : accentColors.primary,
                                      0.3,
                                    ),
                                    bgcolor: alpha(
                                      index === corpusUploadManifest.pipelineSteps.length - 1
                                        ? accentColors.success
                                        : accentColors.primary,
                                      0.055,
                                    ),
                                    "& .MuiChip-label": {
                                      minWidth: 0,
                                      overflow: "hidden",
                                      textOverflow: "ellipsis",
                                    },
                                  }}
                                />
                              ))}
                            </Box>
                          </Box>
                        )}

                        {activeDocumentPrimarySignals.length > 0 && (
                          <Box
                            sx={{
                              order: { xs: 5, sm: 0 },
                              display: "flex",
                              alignItems: "center",
                              gap: 0.45,
                              flexWrap: "wrap",
                              minWidth: 0,
                            }}
                          >
                            {activeDocumentPrimarySignals.map((signal) => (
                              <Tooltip key={signal.key} title={signal.title}>
                                <Chip
                                  size="small"
                                  label={signal.label}
                                  variant="outlined"
                                  sx={{
                                    height: 22,
                                    maxWidth: 150,
                                    borderColor: alpha(signal.color, 0.36),
                                    color: signal.color,
                                    bgcolor: alpha(signal.color, 0.055),
                                    fontSize: "0.64rem",
                                    "& .MuiChip-label": {
                                      minWidth: 0,
                                      overflow: "hidden",
                                      textOverflow: "ellipsis",
                                    },
                                  }}
                                />
                              </Tooltip>
                            ))}
                          </Box>
                        )}

                        {workbenchPanelState.showDocumentInsights && (
                          <Box
                            component="form"
                            ref={selectedDocumentQuestionRef}
                            onSubmit={(event) => {
                              event.preventDefault();
                              handleAskGraphChatQuestion();
                            }}
                            sx={{
                              order: -2,
                              minWidth: 0,
                            }}
                          >
                            <Box
                              sx={{
                                display: { xs: "grid", sm: "block" },
                                gridTemplateColumns: { xs: "minmax(0, 1fr) 44px", sm: "1fr" },
                                alignItems: "stretch",
                                gap: 0.5,
                                minWidth: 0,
                              }}
                            >
                              <TextField
                                value={graphChatQuestion}
                                onChange={(event) =>
                                  setGraphChatQuestion(event.target.value)
                                }
                                size="small"
                                fullWidth
                                multiline
                                minRows={1}
                                maxRows={5}
                                placeholder={
                                  corpusHasBatchResult
                                    ? "Ask across these documents"
                                    : "Ask about the selected document"
                                }
                                disabled={!canAskGraphChatQuestion}
                                InputProps={{
                                  startAdornment: (
                                    <InputAdornment position="start">
                                      <ChatIcon sx={{ fontSize: 17 }} />
                                    </InputAdornment>
                                  ),
                                  endAdornment: (
                                    <InputAdornment
                                      position="end"
                                      sx={{ display: { xs: "none", sm: "flex" } }}
                                    >
                                      <Tooltip
                                        title={
                                          corpusHasBatchResult
                                            ? "Ask across this document set"
                                            : "Ask the selected document"
                                        }
                                        describeChild
                                      >
                                        <span>
                                          <Button
                                            type="submit"
                                            aria-label={
                                              corpusHasBatchResult
                                                ? "Ask set: ask across this document set"
                                                : "Ask: ask the selected document"
                                            }
                                            data-testid="graphrag-selected-document-ask"
                                            size="small"
                                            variant="contained"
                                            endIcon={<SendIcon sx={{ fontSize: 15 }} />}
                                            disabled={
                                              !trimmedGraphChatQuestion ||
                                              !canAskGraphChatQuestion
                                            }
                                            sx={{
                                              minWidth: 82,
                                              height: 31,
                                              px: 1,
                                              borderRadius: "8px",
                                              textTransform: "none",
                                              fontSize: "0.74rem",
                                              fontWeight: 900,
                                              boxShadow: "none",
                                              "&.Mui-disabled": {
                                                bgcolor: alpha(accentColors.primary, 0.12),
                                                backgroundImage: "none",
                                                color: alpha("#fff", 0.46),
                                              },
                                              "& .MuiButton-endIcon": {
                                                ml: 0.45,
                                              },
                                            }}
                                          >
                                            {corpusHasBatchResult ? "Ask set" : "Ask"}
                                          </Button>
                                        </span>
                                      </Tooltip>
                                    </InputAdornment>
                                  ),
                                }}
                                sx={{
                                  "& .MuiOutlinedInput-root": {
                                    minHeight: { xs: 44, sm: 42 },
                                    alignItems: "flex-start",
                                    borderRadius: "8px",
                                    fontSize: "0.86rem",
                                    bgcolor: alpha(accentColors.primary, 0.035),
                                  },
                                  "& textarea": {
                                    lineHeight: 1.35,
                                  },
                                  "& .MuiInputAdornment-root": {
                                    alignSelf: "flex-start",
                                    mt: 0.35,
                                  },
                                }}
                              />
                              <Tooltip
                                title={
                                  corpusHasBatchResult
                                    ? "Ask across this document set"
                                    : "Ask the selected document"
                                }
                                describeChild
                              >
                                <span>
                                  <IconButton
                                    type="submit"
                                    aria-label={
                                      corpusHasBatchResult
                                        ? "Ask set: ask across this document set"
                                        : "Ask: ask the selected document"
                                    }
                                    data-testid="graphrag-selected-document-ask-mobile"
                                    size="small"
                                    disabled={
                                      !trimmedGraphChatQuestion ||
                                      !canAskGraphChatQuestion
                                    }
                                    sx={{
                                      display: { xs: "inline-flex", sm: "none" },
                                      width: 44,
                                      minWidth: 44,
                                      minHeight: 44,
                                      alignSelf: "stretch",
                                      borderRadius: "8px",
                                      bgcolor: alpha(accentColors.primary, 0.92),
                                      color: "#fff",
                                      border: "1px solid",
                                      borderColor: alpha(accentColors.primary, 0.35),
                                      "&:hover": {
                                        bgcolor: accentColors.primary,
                                      },
                                      "&.Mui-disabled": {
                                        bgcolor: alpha(accentColors.primary, 0.1),
                                        color: alpha("#fff", 0.44),
                                        borderColor: alpha(accentColors.primary, 0.2),
                                      },
                                    }}
                                  >
                                    <SendIcon sx={{ fontSize: 18 }} />
                                  </IconButton>
                                </span>
                              </Tooltip>
                            </Box>
                          </Box>
                        )}

                        {workbenchPanelState.showDocumentInsights && (
                          <Box
                            sx={{
                              order: 2,
                              border: "1px solid",
                              borderColor: alpha(accentColors.slate, 0.16),
                              borderRadius: "8px",
                              bgcolor: alpha(accentColors.slate, 0.035),
                              px: { xs: 0.45, sm: 0.6 },
                              py: { xs: 0.3, sm: 0.45 },
                            }}
                          >
                            <Tabs
                              value={documentInsightView}
                              onChange={(_event, value: DocumentInsightView) =>
                                setDocumentInsightView(value)
                              }
                              aria-label="Document insight views"
                              variant="fullWidth"
                              selectionFollowsFocus
                              sx={{
                                minHeight: { xs: 32, sm: 34 },
                                "& .MuiTabs-flexContainer": {
                                  gap: 0.25,
                                },
                                "& .MuiTabs-indicator": {
                                  height: 2,
                                  borderRadius: 999,
                                },
                                "& .MuiTab-root": {
                                  flex: "1 1 0",
                                  minWidth: 0,
                                  maxWidth: "none",
                                  minHeight: { xs: 32, sm: 34 },
                                  px: { xs: 0.65, sm: 0.9 },
                                  py: { xs: 0.25, sm: 0.35 },
                                  mr: 0.25,
                                  borderRadius: "8px",
                                  fontSize: "0.72rem",
                                  fontWeight: 850,
                                  textTransform: "none",
                                },
                                "& .MuiTab-iconWrapper": {
                                  mr: 0.45,
                                },
                              }}
                            >
                              {documentInsightTabs.map((tab) => (
                                <Tab
                                  key={tab.value}
                                  value={tab.value}
                                  disabled={tab.disabled}
                                  aria-label={[
                                    tab.label,
                                    tab.count !== undefined && tab.count > 0
                                      ? `${formatNumber(tab.count)} ${
                                          tab.value === "overview"
                                            ? "insights"
                                            : tab.value === "evidence"
                                              ? "source references"
                                              : "pipeline lanes"
                                        }`
                                      : "",
                                  ]
                                    .filter(Boolean)
                                    .join(", ")}
                                  icon={
                                    tab.value === "overview" ? (
                                      <AnalyticsIcon fontSize="small" />
                                    ) : tab.value === "evidence" ? (
                                      <GraphIcon fontSize="small" />
                                    ) : (
                                      <StorageIcon fontSize="small" />
                                    )
                                  }
                                  iconPosition="start"
                                  label={
                                    <Box
                                      component="span"
                                      sx={{
                                        display: "inline-flex",
                                        alignItems: "center",
                                        gap: 0.5,
                                        minWidth: 0,
                                      }}
                                    >
                                      <Box
                                        component="span"
                                        sx={{
                                          display: {
                                            xs: tab.shortLabel ? "none" : "inline",
                                            sm: "inline",
                                          },
                                        }}
                                      >
                                        {tab.label}
                                      </Box>
                                      {tab.shortLabel && (
                                        <Box
                                          component="span"
                                          sx={{ display: { xs: "inline", sm: "none" } }}
                                        >
                                          {tab.shortLabel}
                                        </Box>
                                      )}
                                      {tab.count !== undefined && tab.count > 0
                                        ? (
                                          <Box
                                            component="span"
                                            aria-hidden
                                            sx={{
                                              display: "inline-grid",
                                              minWidth: 20,
                                              height: 18,
                                              placeItems: "center",
                                              px: 0.45,
                                              borderRadius: "999px",
                                              bgcolor: alpha(accentColors.primary, 0.13),
                                              color: accentColors.primary,
                                              fontSize: "0.62rem",
                                              lineHeight: 1,
                                              fontWeight: 950,
                                            }}
                                          >
                                            {formatNumber(tab.count)}
                                          </Box>
                                        )
                                        : null}
                                    </Box>
                                  }
                                />
                              ))}
                            </Tabs>
                          </Box>
                        )}

                        {documentInsightView === "overview" &&
                          activeReviewStats.length > 0 && (
                          <Box
                            sx={{
                              order: 3,
                              display: "flex",
                              gap: 0.45,
                              flexWrap: "wrap",
                              minWidth: 0,
                            }}
                          >
                            {activeReviewStats
                              .filter((stat) => stat.value > 0)
                              .slice(0, 4)
                              .map((stat) => {
                                const statColor =
                                  stat.label === "Relations"
                                    ? accentColors.primary
                                    : stat.label === "Entities"
                                      ? accentColors.info
                                      : stat.label === "Paths"
                                        ? accentColors.purple
                                        : accentColors.success;

                                return (
                                  <Chip
                                    key={`overview-${stat.label}`}
                                    size="small"
                                    label={`${formatNumber(stat.value)} ${stat.label.toLowerCase()}`}
                                    variant="outlined"
                                    sx={{
                                      height: 23,
                                      maxWidth: 160,
                                      borderColor: alpha(statColor, 0.28),
                                      color: statColor,
                                      bgcolor: alpha(statColor, 0.04),
                                      fontSize: "0.64rem",
                                      "& .MuiChip-label": {
                                        minWidth: 0,
                                        overflow: "hidden",
                                        textOverflow: "ellipsis",
                                      },
                                    }}
                                  />
                                );
                              })}
                          </Box>
                        )}

                        {documentInsightView === "overview" &&
                          activeGraphRows.length > 0 && (
                          <Box
                            sx={{
                              order: 4,
                              display: { xs: "none", sm: "grid" },
                              gridTemplateColumns: {
                                xs: "1fr",
                                xl: "1fr",
                              },
                              gap: 0.65,
                              alignItems: "stretch",
                            }}
                          >
                            {activeGraphRows.length > 0 && (
                              <Box
                                sx={{
                                  minWidth: 0,
                                  border: "1px solid",
                                  borderColor: alpha(accentColors.success, 0.18),
                                  borderRadius: "8px",
                                  bgcolor: alpha(accentColors.success, 0.035),
                                  p: 0.75,
                                }}
                              >
                                <Box
                                  sx={{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "space-between",
                                    gap: 0.65,
                                    mb: 0.55,
                                    minWidth: 0,
                                  }}
                                >
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    noWrap
                                    sx={{ fontWeight: 900 }}
                                  >
                                    Key relationships
                                  </Typography>
                                  <Chip
                                    size="small"
                                    label={`${formatNumber(activeGraphRows.length)} ready`}
                                    variant="outlined"
                                    sx={{ height: 20, flexShrink: 0, fontSize: "0.62rem" }}
                                  />
                                </Box>
                                <Box
                                  sx={{
                                    display: "grid",
                                    gridTemplateColumns: {
                                      xs: "1fr",
                                      md: "repeat(2, minmax(0, 1fr))",
                                    },
                                    gap: 0.5,
                                  }}
                                >
                                  {activeGraphRows.slice(0, 2).map((row) => {
                                    const relationLabel =
                                      row.kind === "path"
                                        ? "Path"
                                        : row.relation
                                          ? formatGraphRelationLabel(row.relation)
                                          : row.label;
                                    return (
                                      <Box
                                        key={`overview-graph-${row.key}`}
                                        role="button"
                                        tabIndex={0}
                                        title={row.title || row.text}
                                        onClick={() => {
                                          setSelectedEvidenceRowKey(row.key);
                                          setDocumentInsightView("evidence");
                                        }}
                                        onKeyDown={(event) => {
                                          if (event.key === "Enter" || event.key === " ") {
                                            event.preventDefault();
                                            setSelectedEvidenceRowKey(row.key);
                                            setDocumentInsightView("evidence");
                                          }
                                        }}
                                        sx={{
                                          minWidth: 0,
                                          minHeight: 72,
                                          border: "1px solid",
                                          borderColor: alpha(row.color, 0.18),
                                          borderRadius: "8px",
                                          bgcolor: alpha(row.color, 0.055),
                                          px: 0.7,
                                          py: 0.62,
                                          cursor: "pointer",
                                          outline: "none",
                                          transition:
                                            "background-color 120ms ease, border-color 120ms ease",
                                          "&:hover": {
                                            bgcolor: alpha(row.color, 0.1),
                                            borderColor: alpha(row.color, 0.42),
                                          },
                                          "&:focus-visible": {
                                            borderColor: row.color,
                                            boxShadow: `0 0 0 2px ${alpha(row.color, 0.16)}`,
                                          },
                                        }}
                                      >
                                        <Box
                                          sx={{
                                            display: "flex",
                                            alignItems: "center",
                                            gap: 0.45,
                                            minWidth: 0,
                                            mb: 0.35,
                                          }}
                                        >
                                          <Chip
                                            size="small"
                                            label={relationLabel}
                                            variant="outlined"
                                            sx={{
                                              height: 19,
                                              maxWidth: 122,
                                              flexShrink: 0,
                                              borderColor: alpha(row.color, 0.34),
                                              color: row.color,
                                              fontSize: "0.6rem",
                                              "& .MuiChip-label": {
                                                minWidth: 0,
                                                overflow: "hidden",
                                                textOverflow: "ellipsis",
                                              },
                                            }}
                                          />
                                          <Typography
                                            variant="caption"
                                            noWrap
                                            color="text.secondary"
                                            sx={{ minWidth: 0, fontSize: "0.64rem" }}
                                          >
                                            {row.kind === "path" ? "Graph path" : "Graph relation"}
                                          </Typography>
                                        </Box>
                                        <Typography
                                          variant="caption"
                                          sx={{
                                            display: "-webkit-box",
                                            overflow: "hidden",
                                            WebkitBoxOrient: "vertical",
                                            WebkitLineClamp: 2,
                                            color: "text.primary",
                                            fontWeight: 900,
                                            lineHeight: 1.24,
                                          }}
                                        >
                                          {row.text}
                                        </Typography>
                                        {row.title && row.title !== row.text && (
                                          <Typography
                                            variant="caption"
                                            color="text.secondary"
                                            sx={{
                                              display: "-webkit-box",
                                              mt: 0.2,
                                              overflow: "hidden",
                                              WebkitBoxOrient: "vertical",
                                              WebkitLineClamp: 1,
                                              fontSize: "0.64rem",
                                              lineHeight: 1.2,
                                            }}
                                          >
                                            {row.title}
                                          </Typography>
                                        )}
                                      </Box>
                                    );
                                  })}
                                </Box>
                              </Box>
                            )}

                          </Box>
                        )}

                        {documentInsightView === "evidence" &&
                          (activeSourceArtifactRows.length > 0 ||
                            activeSourceLocationChips.length > 0 ||
                            activeEvidenceChips.length > 0 ||
                            activeChildSourceArtifacts.length > 0) && (
                          <Box
                            component="section"
                            sx={{
                              order: 3,
                              border: "1px solid",
                              borderColor: alpha(accentColors.primary, 0.18),
                              borderRadius: "8px",
                              overflow: "hidden",
                              bgcolor: alpha(accentColors.primary, 0.025),
                            }}
                          >
                            <Box
                              sx={{
                                display: "grid",
                                gridTemplateColumns: {
                                  xs: "1fr",
                                  md: activeSourceArtifactRows.length > 0
                                    ? "minmax(0, 1fr) auto"
                                    : "1fr",
                                },
                                alignItems: "center",
                                gap: 0.85,
                                px: 1,
                                py: 0.85,
                                borderBottom: "1px solid",
                                borderColor: alpha(accentColors.primary, 0.12),
                              }}
                            >
                              <Box sx={{ minWidth: 0 }}>
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{ display: "block", fontWeight: 900 }}
                                >
                                  Source
                                </Typography>
                                <Typography
                                  variant="body2"
                                  noWrap
                                  title={activeEvidenceTitle}
                                  sx={{ fontWeight: 950, minWidth: 0 }}
                                >
                                  {activeEvidenceTitle}
                                </Typography>
                                {activeSourceDocumentSummaryText && (
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    noWrap
                                    sx={{ display: "block", mt: 0.2 }}
                                  >
                                    {activeSourceDocumentSummaryText}
                                  </Typography>
                                )}
                              </Box>

                              {activeSourceArtifactRows.length > 0 && (
                                <Stack
                                  direction="row"
                                  spacing={0.5}
                                  alignItems="center"
                                  useFlexGap
                                  sx={{
                                    width: { xs: "100%", md: "auto" },
                                    maxWidth: { xs: "100%", md: 560 },
                                    overflowX: "visible",
                                    overflowY: "hidden",
                                    flexWrap: "wrap",
                                    justifyContent: { xs: "flex-start", md: "flex-end" },
                                    flexShrink: 0,
                                  }}
                                >
                                  {activeSourceArtifactRows.map((artifact) => {
                                    const isPrimary =
                                      artifact.kind === "source_document";
                                    const Icon =
                                      artifact.kind === "source_document"
                                        ? DocumentIcon
                                        : artifact.contentLabel === "JSON"
                                          ? DataIcon
                                          : ArticleIcon;
                                    return (
                                      <Tooltip
                                        key={artifact.kind}
                                        title={[
                                          artifact.label,
                                          artifact.detail || artifact.downloadFilename,
                                        ].filter(Boolean).join(" - ")}
                                        describeChild
                                      >
                                        <Button
                                          size="small"
                                          variant={isPrimary ? "contained" : "outlined"}
                                          startIcon={<Icon fontSize="small" />}
                                          endIcon={<OpenIcon fontSize="small" />}
                                          aria-label={`Open ${artifact.label || artifact.actionLabel}`}
                                          onClick={() =>
                                            openGraphRAGPipelineArtifact(artifact.endpoint)
                                          }
                                          sx={{
                                            minHeight: 28,
                                            minWidth: 0,
                                            maxWidth: { xs: 118, sm: 152 },
                                            flex: "0 1 auto",
                                            px: { xs: 0.65, sm: 0.9 },
                                            py: 0.15,
                                            borderRadius: "8px",
                                            borderColor: alpha(artifact.color, 0.36),
                                            bgcolor: isPrimary
                                              ? artifact.color
                                              : alpha(artifact.color, 0.045),
                                            color: isPrimary
                                              ? "primary.contrastText"
                                              : artifact.color,
                                            textTransform: "none",
                                            boxShadow: "none",
                                            justifyContent: "flex-start",
                                            "&:hover": {
                                              borderColor: alpha(artifact.color, 0.58),
                                              bgcolor: isPrimary
                                                ? artifact.color
                                                : alpha(artifact.color, 0.09),
                                              boxShadow: "none",
                                            },
                                            "& .MuiButton-startIcon, & .MuiButton-endIcon": {
                                              flexShrink: 0,
                                            },
                                            "& .MuiButton-startIcon": {
                                              mr: { xs: 0.35, sm: 0.5 },
                                            },
                                            "& .MuiButton-endIcon": {
                                              ml: { xs: 0.3, sm: 0.5 },
                                            },
                                          }}
                                        >
                                          <Box
                                            component="span"
                                            sx={{
                                              minWidth: 0,
                                              overflow: "hidden",
                                              textOverflow: "ellipsis",
                                              whiteSpace: "nowrap",
                                            }}
                                          >
                                            {artifact.actionLabel}
                                          </Box>
                                        </Button>
                                      </Tooltip>
                                    );
                                  })}
                                </Stack>
                              )}
                            </Box>

                            {activeSourceSignalChips.length > 0 && (
                              <Box
                                sx={{
                                  display: "flex",
                                  gap: 0.45,
                                  flexWrap: "wrap",
                                  px: 1,
                                  py: 0.75,
                                }}
                              >
                                {visibleActiveSourceSignalChips.map((chip) => (
                                  <Chip
                                    key={`source-signal-${chip.label}`}
                                    label={`${formatNumber(chip.value)} ${chip.label}`}
                                    size="small"
                                    variant="outlined"
                                    sx={{
                                      height: 22,
                                      maxWidth: 176,
                                      borderColor: alpha(chip.color, 0.34),
                                      color: chip.color,
                                      bgcolor: alpha(chip.color, 0.045),
                                      fontSize: "0.66rem",
                                      "& .MuiChip-label": {
                                        minWidth: 0,
                                        overflow: "hidden",
                                        textOverflow: "ellipsis",
                                      },
                                    }}
                                  />
                                ))}
                                {hiddenActiveSourceSignalChipCount > 0 && (
                                  <Chip
                                    label={`+${formatNumber(hiddenActiveSourceSignalChipCount)} more`}
                                    size="small"
                                    variant="outlined"
                                    sx={{
                                      height: 22,
                                      borderColor: alpha(accentColors.slate, 0.24),
                                      color: "text.secondary",
                                      bgcolor: alpha(accentColors.slate, 0.035),
                                      fontSize: "0.66rem",
                                    }}
                                  />
                                )}
                              </Box>
                            )}

                            {activeSourceDetailsAvailable && (
                              <Box
                                component="details"
                                data-testid="graphrag-source-details"
                                sx={{
                                  mx: 1,
                                  mb: 0.85,
                                  border: "1px solid",
                                  borderColor: alpha(accentColors.slate, 0.18),
                                  borderRadius: "8px",
                                  bgcolor: alpha(accentColors.slate, 0.035),
                                  overflow: "hidden",
                                  "& summary": {
                                    cursor: "pointer",
                                    listStyle: "none",
                                  },
                                  "& summary::-webkit-details-marker": {
                                    display: "none",
                                  },
                                }}
                              >
                                <Box
                                  component="summary"
                                  sx={{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "space-between",
                                    gap: 0.75,
                                    px: 0.8,
                                    py: 0.65,
                                  }}
                                >
                                  <Box sx={{ minWidth: 0 }}>
                                    <Typography
                                      variant="caption"
                                      sx={{ display: "block", fontWeight: 900 }}
                                    >
                                      Source details
                                    </Typography>
                                    <Typography
                                      variant="caption"
                                      color="text.secondary"
                                      noWrap
                                      sx={{ display: "block", fontSize: "0.64rem" }}
                                    >
                                      Extra evidence signals and embedded files
                                    </Typography>
                                  </Box>
                                  <Chip
                                    size="small"
                                    label={`${formatNumber(activeSourceDetailItemCount)} items`}
                                    variant="outlined"
                                    sx={{ height: 21, flexShrink: 0, fontSize: "0.62rem" }}
                                  />
                                </Box>
                                <Box
                                  sx={{
                                    display: "grid",
                                    gap: 0.65,
                                    px: 0.8,
                                    pb: 0.75,
                                  }}
                                >
                                  {activeSourceSignalChips.length > 0 && (
                                    <Box
                                      sx={{
                                        display: "flex",
                                        gap: 0.45,
                                        flexWrap: "wrap",
                                      }}
                                    >
                                      {activeSourceSignalChips.map((chip) => (
                                        <Chip
                                          key={`source-detail-signal-${chip.label}`}
                                          data-testid="graphrag-source-detail-signal"
                                          label={`${formatNumber(chip.value)} ${chip.label}`}
                                          size="small"
                                          variant="outlined"
                                          sx={{
                                            height: 22,
                                            maxWidth: 176,
                                            borderColor: alpha(chip.color, 0.34),
                                            color: chip.color,
                                            bgcolor: alpha(chip.color, 0.045),
                                            fontSize: "0.66rem",
                                            "& .MuiChip-label": {
                                              minWidth: 0,
                                              overflow: "hidden",
                                              textOverflow: "ellipsis",
                                            },
                                          }}
                                        />
                                      ))}
                                    </Box>
                                  )}
                                  {activeChildSourceArtifacts.length > 0 && (
                                    <Box
                                      sx={{
                                        display: "flex",
                                        gap: 0.45,
                                        flexWrap: "wrap",
                                      }}
                                    >
                                      {activeChildSourceArtifacts.map((artifact) => (
                                        <Tooltip
                                          key={artifact.key}
                                          title={artifact.detail || "Embedded source artifact"}
                                        >
                                          <Chip
                                            data-testid="graphrag-source-artifact-chip"
                                            icon={<StorageIcon sx={{ fontSize: 14 }} />}
                                            label={artifact.label}
                                            size="small"
                                            variant="outlined"
                                            sx={{
                                              height: 23,
                                              maxWidth: 220,
                                              borderColor: alpha(accentColors.slate, 0.24),
                                              bgcolor: alpha(accentColors.slate, 0.045),
                                              fontSize: "0.64rem",
                                              "& .MuiChip-label": {
                                                minWidth: 0,
                                                overflow: "hidden",
                                                textOverflow: "ellipsis",
                                              },
                                            }}
                                          />
                                        </Tooltip>
                                      ))}
                                    </Box>
                                  )}
                                </Box>
                              </Box>
                            )}

                            {activeEvidenceExampleRows.length > 0 && (
                              <Box
                                sx={{
                                  px: 1,
                                  pb: 0.9,
                                }}
                              >
                                <Box
                                  sx={{
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "space-between",
                                    gap: 0.75,
                                    mb: 0.6,
                                    minWidth: 0,
                                  }}
                                >
                                  <Box sx={{ minWidth: 0 }}>
                                    <Typography
                                      variant="caption"
                                      sx={{ display: "block", fontWeight: 900 }}
                                    >
                                      Evidence examples
                                    </Typography>
                                    <Typography
                                      variant="caption"
                                      color="text.secondary"
                                      noWrap
                                      sx={{ display: "block", fontSize: "0.66rem" }}
                                    >
                                      The most useful extracted context for chat
                                    </Typography>
                                  </Box>
                                  <Chip
                                    size="small"
                                    label={`${formatNumber(activeStageSampleRows.length)} found`}
                                    variant="outlined"
                                    sx={{ height: 21, flexShrink: 0, fontSize: "0.62rem" }}
                                  />
                                </Box>
                                <Box
                                  sx={{
                                    display: "grid",
                                    gridTemplateColumns: {
                                      xs: "1fr",
                                      lg: "repeat(2, minmax(0, 1fr))",
                                    },
                                    gap: 0.65,
                                  }}
                                >
                                  {activeEvidenceExampleRows.map((row) => (
                                    <Box
                                      key={`evidence-example-${row.key}`}
                                      data-testid="graphrag-evidence-example-row"
                                      data-evidence-sample-key={row.key}
                                      data-evidence-sample-label={row.label}
                                      data-evidence-sample-title={row.title}
                                      title={`${row.title}: ${row.body}${row.linkTarget ? ` Link target: ${row.linkTarget.url}` : ""}`}
                                      sx={{
                                        minWidth: 0,
                                        border: "1px solid",
                                        borderColor: alpha(row.color, 0.22),
                                        borderRadius: "8px",
                                        bgcolor: alpha(row.color, 0.045),
                                        px: 0.85,
                                        py: 0.72,
                                      }}
                                    >
                                      <Box
                                        sx={{
                                          display: "grid",
                                          gridTemplateColumns: "minmax(0, 1fr) auto",
                                          gap: 0.65,
                                          alignItems: "start",
                                          minWidth: 0,
                                        }}
                                      >
                                        <Box sx={{ minWidth: 0 }}>
                                          <Typography
                                            variant="caption"
                                            noWrap
                                            sx={{
                                              display: "block",
                                              color: row.color,
                                              fontSize: "0.65rem",
                                              fontWeight: 950,
                                              lineHeight: 1.15,
                                            }}
                                          >
                                            {row.label}
                                          </Typography>
                                          <Typography
                                            variant="caption"
                                            sx={{
                                              display: "-webkit-box",
                                              mt: 0.3,
                                              minHeight: "2.25em",
                                              overflow: "hidden",
                                              WebkitBoxOrient: "vertical",
                                              WebkitLineClamp: 2,
                                              color: "text.primary",
                                              fontWeight: 900,
                                              lineHeight: 1.18,
                                            }}
                                          >
                                            {row.title}
                                          </Typography>
                                        </Box>
                                        <Stack
                                          spacing={0.35}
                                          alignItems="flex-end"
                                          sx={{ flexShrink: 0 }}
                                        >
                                          <Chip
                                            size="small"
                                            label={row.countLabel}
                                            variant="outlined"
                                            sx={{
                                              height: 19,
                                              maxWidth: 92,
                                              borderColor: alpha(row.color, 0.34),
                                              fontSize: "0.6rem",
                                              "& .MuiChip-label": {
                                                px: 0.65,
                                                overflow: "hidden",
                                                textOverflow: "ellipsis",
                                              },
                                            }}
                                          />
                                          <Button
                                            data-testid="graphrag-evidence-example-ask"
                                            size="small"
                                            variant="text"
                                            startIcon={<ChatIcon fontSize="small" />}
                                            title={`Ask about ${row.label}: ${row.title}`}
                                            onClick={() => handleAskAboutPipelineSampleRow(row)}
                                            sx={{
                                              minHeight: 22,
                                              px: 0.65,
                                              borderRadius: "8px",
                                              color: row.color,
                                              textTransform: "none",
                                              fontSize: "0.64rem",
                                              "& .MuiButton-startIcon": {
                                                mr: 0.35,
                                              },
                                            }}
                                          >
                                            Ask
                                          </Button>
                                        </Stack>
                                      </Box>
                                      {row.body && (
                                        <Typography
                                          variant="caption"
                                          color="text.secondary"
                                          sx={{
                                            display: "-webkit-box",
                                            mt: 0.45,
                                            overflow: "hidden",
                                            WebkitBoxOrient: "vertical",
                                            WebkitLineClamp: 2,
                                            lineHeight: 1.25,
                                          }}
                                        >
                                          {row.body}
                                        </Typography>
                                      )}
                                      <PipelineSampleLinkTargetRow
                                        target={row.linkTarget}
                                        color={row.color}
                                        compact
                                      />
                                    </Box>
                                  ))}
                                </Box>
                              </Box>
                            )}
                          </Box>
                        )}

                        {documentInsightView === "processing" &&
                          activePipelineLaneRows.length > 0 && (
                          <Box
                            component="details"
                            sx={{
                              order: 5,
                              border: "1px solid",
                              borderColor: alpha(accentColors.info, 0.18),
                              borderRadius: "8px",
                              bgcolor: alpha(accentColors.slate, 0.03),
                              overflow: "hidden",
                              "& summary": {
                                cursor: "pointer",
                                listStyle: "none",
                              },
                              "& summary::-webkit-details-marker": {
                                display: "none",
                              },
                            }}
                          >
                            <Box
                              component="summary"
                              sx={{
                                display: "grid",
                                gridTemplateColumns: {
                                  xs: "1fr",
                                  sm: "minmax(0, 1fr) auto",
                                },
                                alignItems: "start",
                                gap: 0.8,
                                px: 1,
                                py: 0.9,
                                borderBottom: "1px solid",
                                borderColor: "divider",
                              }}
                            >
                              <Box
                                sx={{
                                  display: "grid",
                                  gridTemplateColumns: {
                                    xs: "1fr",
                                    md: activePipelineReadiness.primaryCard
                                      ? "minmax(0, 0.58fr) minmax(260px, 1fr)"
                                      : "1fr",
                                  },
                                  alignItems: "stretch",
                                  gap: 0.75,
                                  minWidth: 0,
                                }}
                              >
                                <Box sx={{ minWidth: 0 }}>
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{ display: "block", fontWeight: 850 }}
                                  >
                                    {activePipelineReadiness.headline}
                                  </Typography>
                                  <Typography
                                    variant="body2"
                                    noWrap
                                    sx={{ fontWeight: 900 }}
                                    title={activePipelineReadiness.detail}
                                  >
                                    {activePipelineReadiness.detail}
                                  </Typography>
                                  {activePipelineReadiness.chips.length > 0 && (
                                    <Box
                                      sx={{
                                        display: "flex",
                                        gap: 0.35,
                                        flexWrap: "wrap",
                                        mt: 0.55,
                                      }}
                                    >
                                      {activePipelineReadiness.chips.slice(0, 2).map((chip) => (
                                        <Chip
                                          key={`readiness-chip-${chip}`}
                                          size="small"
                                          label={chip}
                                          variant="outlined"
                                          sx={{
                                            height: 19,
                                            maxWidth: 158,
                                            fontSize: "0.6rem",
                                            bgcolor: alpha(accentColors.info, 0.045),
                                            "& .MuiChip-label": {
                                              minWidth: 0,
                                              overflow: "hidden",
                                              textOverflow: "ellipsis",
                                            },
                                          }}
                                        />
                                      ))}
                                    </Box>
                                  )}
                                </Box>
                                {activePipelineReadiness.primaryCard && (
                                  <Box
                                    sx={{
                                      display: "grid",
                                      gridTemplateColumns: "minmax(0, 1fr) auto",
                                      alignContent: "center",
                                      gap: 0.35,
                                      minWidth: 0,
                                      px: 0.75,
                                      py: 0.62,
                                      border: "1px solid",
                                      borderColor: alpha(accentColors.primary, 0.22),
                                      borderRadius: "8px",
                                      bgcolor: alpha(accentColors.primary, 0.055),
                                    }}
                                  >
                                    <Box sx={{ minWidth: 0 }}>
                                      <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{
                                          display: "block",
                                          fontSize: "0.64rem",
                                          fontWeight: 850,
                                          lineHeight: 1.15,
                                        }}
                                      >
                                        {activePipelineReadiness.primaryCard.label
                                          .toLowerCase()
                                          .includes("relationship")
                                          ? "Strongest relationship"
                                          : "Key evidence"}
                                      </Typography>
                                      <Typography
                                        variant="caption"
                                        title={[
                                          activePipelineReadiness.primaryCard.title,
                                          activePipelineReadiness.primaryCard.detail,
                                        ].filter(Boolean).join(" - ")}
                                        sx={{
                                          display: "-webkit-box",
                                          minWidth: 0,
                                          overflow: "hidden",
                                          WebkitBoxOrient: "vertical",
                                          WebkitLineClamp: 2,
                                          color: "text.primary",
                                          fontWeight: 900,
                                          lineHeight: 1.28,
                                        }}
                                      >
                                        {activePipelineReadiness.primaryCard.title ||
                                          activePipelineReadiness.primaryCard.detail}
                                      </Typography>
                                    </Box>
                                    <Chip
                                      size="small"
                                      label={activePipelineReadiness.primaryCard.label}
                                      variant="outlined"
                                      sx={{
                                        height: 20,
                                        maxWidth: 112,
                                        alignSelf: "center",
                                        justifySelf: "end",
                                        fontSize: "0.6rem",
                                        borderColor: alpha(accentColors.primary, 0.34),
                                        color: accentColors.primary,
                                        "& .MuiChip-label": {
                                          minWidth: 0,
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                        },
                                      }}
                                    />
                                  </Box>
                                )}
                              </Box>
                              <Chip
                                size="small"
                                label={activePipelineReadiness.statusLabel}
                                variant="outlined"
                                sx={{
                                  height: 22,
                                  maxWidth: { xs: 160, sm: 210 },
                                  flexShrink: 0,
                                  justifySelf: { xs: "flex-start", sm: "end" },
                                  "& .MuiChip-label": {
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                  },
                                }}
                              />
                            </Box>
                            {activePipelineIssueRows.length > 0 && (
                              <Box
                                sx={{
                                  mx: 0.6,
                                  mt: 0.6,
                                  px: 0.85,
                                  py: 0.7,
                                  border: "1px solid",
                                  borderColor: alpha(accentColors.warning, 0.3),
                                  borderRadius: "8px",
                                  bgcolor: alpha(accentColors.warning, 0.06),
                                }}
                              >
                                <Typography
                                  variant="caption"
                                  sx={{
                                    display: "block",
                                    color: accentColors.warning,
                                    fontWeight: 950,
                                    mb: 0.35,
                                  }}
                                >
                                  Review needed
                                </Typography>
                                <Stack spacing={0.35}>
                                  {activePipelineIssueRows.slice(0, 2).map((issue) => (
                                    <Box
                                      key={`pipeline-issue-${issue.key}`}
                                      sx={{
                                        display: "grid",
                                        gridTemplateColumns: "auto minmax(0, 1fr)",
                                        gap: 0.5,
                                        minWidth: 0,
                                        alignItems: "baseline",
                                      }}
                                    >
                                      <Chip
                                        size="small"
                                        label={issue.label}
                                        variant="outlined"
                                        sx={{
                                          height: 19,
                                          maxWidth: 128,
                                          color: issue.color,
                                          borderColor: alpha(issue.color, 0.38),
                                          fontSize: "0.6rem",
                                          "& .MuiChip-label": {
                                            minWidth: 0,
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                          },
                                        }}
                                      />
                                      <Typography
                                        variant="caption"
                                        title={issue.message}
                                        sx={{
                                          minWidth: 0,
                                          color: "text.primary",
                                          display: "-webkit-box",
                                          overflow: "hidden",
                                          WebkitBoxOrient: "vertical",
                                          WebkitLineClamp: 2,
                                          lineHeight: 1.25,
                                        }}
                                      >
                                        {issue.message}
                                      </Typography>
                                    </Box>
                                  ))}
                                </Stack>
                              </Box>
                            )}
                            <Box
                              sx={{
                                display: "grid",
                                gridTemplateColumns: {
                                  xs: "1fr",
                                  sm: "repeat(2, minmax(0, 1fr))",
                                  xl: "repeat(3, minmax(0, 1fr))",
                                },
                                gap: 0.45,
                                p: 0.6,
                              }}
                            >
                              {activePipelineReadiness.cards.slice(0, 5).map((card) => {
                                const lane = activePipelineLaneRows.find((item) => item.key === card.key);
                                const cardColor = lane?.color || accentColors.info;
                                const laneStatusColor =
                                  card.tone === "warning"
                                    ? accentColors.warning
                                    : card.tone === "running"
                                      ? accentColors.info
                                      : card.tone === "success"
                                        ? accentColors.success
                                        : accentColors.slate;
                                return (
                                  <Box
                                    key={`trace-lane-${card.key}`}
                                    title={`${card.label}: ${card.title}${card.detail ? ` - ${card.detail}` : ""}`}
                                    sx={{
                                      minWidth: 0,
                                      border: "1px solid",
                                      borderColor: alpha(cardColor, 0.22),
                                      borderRadius: "8px",
                                      bgcolor: alpha(cardColor, 0.045),
                                      px: 0.7,
                                      py: 0.58,
                                    }}
                                  >
                                    <Box
                                      sx={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 0.5,
                                        minWidth: 0,
                                      }}
                                    >
                                      <Typography
                                        variant="caption"
                                        noWrap
                                        sx={{
                                          flex: 1,
                                          minWidth: 0,
                                          color: cardColor,
                                          fontWeight: 950,
                                        }}
                                      >
                                        {card.label}
                                      </Typography>
                                      <Chip
                                        size="small"
                                        label={card.statusLabel}
                                        variant="outlined"
                                        sx={{
                                          height: 18,
                                          maxWidth: 86,
                                          flexShrink: 0,
                                          borderColor: alpha(laneStatusColor, 0.4),
                                          color: laneStatusColor,
                                          fontSize: "0.58rem",
                                          "& .MuiChip-label": {
                                            px: 0.65,
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                          },
                                        }}
                                      />
                                    </Box>
                                    <Typography
                                      variant="caption"
                                      sx={{
                                        display: "-webkit-box",
                                        mt: 0.45,
                                        minHeight: "2.1em",
                                        overflow: "hidden",
                                        WebkitBoxOrient: "vertical",
                                        WebkitLineClamp: 2,
                                        color: "text.primary",
                                        fontWeight: 850,
                                        lineHeight: 1.18,
                                      }}
                                    >
                                      {card.title}
                                    </Typography>
                                    {card.detail && (
                                      <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{
                                          display: "-webkit-box",
                                          mt: 0.3,
                                          overflow: "hidden",
                                          WebkitBoxOrient: "vertical",
                                          WebkitLineClamp: 2,
                                          lineHeight: 1.2,
                                        }}
                                      >
                                        {card.detail}
                                      </Typography>
                                    )}
                                    <Box
                                      sx={{
                                        display: "flex",
                                        gap: 0.35,
                                        flexWrap: "wrap",
                                        mt: 0.45,
                                      }}
                                    >
                                      <Chip
                                        size="small"
                                        label={card.countLabel}
                                        sx={{
                                          height: 18,
                                          maxWidth: 118,
                                          fontSize: "0.6rem",
                                          bgcolor: alpha(cardColor, 0.1),
                                          "& .MuiChip-label": {
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                          },
                                        }}
                                      />
                                      {lane && lane.durationMs > 0 && (
                                        <Chip
                                          size="small"
                                          label={formatDuration(lane.durationMs)}
                                          variant="outlined"
                                          sx={{ height: 18, fontSize: "0.6rem" }}
                                        />
                                      )}
                                      {card.warningCount > 0 && (
                                        <Chip
                                          size="small"
                                          label={`${card.warningCount} issue${card.warningCount === 1 ? "" : "s"}`}
                                          variant="outlined"
                                          sx={{
                                            height: 18,
                                            fontSize: "0.6rem",
                                            borderColor: alpha(accentColors.warning, 0.36),
                                            color: accentColors.warning,
                                          }}
                                        />
                                      )}
                                    </Box>
                                  </Box>
                                );
                              })}
                            </Box>
                          </Box>
                        )}

                        {documentInsightView === "overview" &&
                          activeSuggestedQuestions.length > 0 && (
                          <Box sx={{ order: 1, display: { xs: "none", sm: "block" } }}>
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              sx={{
                                display: "block",
                                fontWeight: 850,
                                mb: 0.5,
                              }}
                            >
                              Try
                            </Typography>
                            <Box
                              sx={{
                                display: "flex",
                                gap: 0.5,
                                flexWrap: "wrap",
                              }}
                            >
                              {activeSuggestedQuestions.map((question, index) => {
                                const questionLabel = graphRAGQuestionLabel(question);
                                return (
                                  <Button
                                    key={question.id}
                                    data-testid="graphrag-starter-prompt"
                                    data-prompt-label={questionLabel}
                                    size="small"
                                    variant="outlined"
                                    startIcon={<ChatIcon fontSize="small" />}
                                    onClick={() =>
                                      corpusHasBatchResult
                                        ? handleAskChatAcrossBatch(
                                            question.prompt,
                                            {},
                                            questionLabel,
                                          )
                                        : activeDocumentGroup
                                          ? handleAskChatFromPipelineRuns(
                                              activeDocumentGroup.runs,
                                              question.prompt,
                                              {},
                                              questionLabel,
                                            )
                                          : handleAskChatFromPipelineRun(
                                              activeCorpusPipelineRecord,
                                              question.prompt,
                                              {},
                                              questionLabel,
                                            )
                                    }
                                    sx={{
                                      display: {
                                        xs: index === 0 ? "inline-flex" : "none",
                                        sm: "inline-flex",
                                      },
                                      width: "auto",
                                      minHeight: { xs: 28, sm: 28 },
                                      maxWidth: { xs: "100%", sm: 340 },
                                      px: 1,
                                      py: 0.2,
                                      borderRadius: "8px",
                                      textTransform: "none",
                                      boxShadow: "none",
                                      color: accentColors.primary,
                                      borderColor: alpha(accentColors.primary, 0.32),
                                      bgcolor: alpha(accentColors.primary, 0.035),
                                      "&:hover": {
                                        borderColor: alpha(accentColors.primary, 0.48),
                                        bgcolor: alpha(accentColors.primary, 0.08),
                                      },
                                      justifyContent: "flex-start",
                                      alignItems: "flex-start",
                                      textAlign: "left",
                                      "& .MuiButton-startIcon": {
                                        flexShrink: 0,
                                        mt: { xs: 0.15, sm: 0 },
                                      },
                                    }}
                                  >
                                    <Box
                                      component="span"
                                      sx={{
                                        minWidth: 0,
                                        display: "block",
                                        overflow: { xs: "visible", sm: "hidden" },
                                        overflowWrap: "anywhere",
                                        textOverflow: { xs: "clip", sm: "ellipsis" },
                                        whiteSpace: { xs: "normal", sm: "nowrap" },
                                        lineHeight: 1.18,
                                      }}
                                    >
                                      {questionLabel}
                                    </Box>
                                  </Button>
                                );
                              })}
                            </Box>
                          </Box>
                        )}

                        {documentInsightView === "processing" &&
                          activeStageOutputRows.length > 0 && (
                          <Box
                            component="section"
                            sx={{
                              order: 6,
                              border: "1px solid",
                              borderColor: alpha(accentColors.info, 0.18),
                              borderRadius: "8px",
                              overflow: "hidden",
                              bgcolor: alpha(accentColors.slate, 0.025),
                            }}
                          >
                            <Box
                              sx={{
                                display: "grid",
                                gridTemplateColumns: {
                                  xs: "1fr",
                                  sm: "minmax(0, 1fr) auto",
                                },
                                alignItems: "start",
                                gap: 0.75,
                                px: 1,
                                py: 0.85,
                                minWidth: 0,
                              }}
                            >
                              <Box sx={{ minWidth: 0 }}>
                                <Typography
                                  variant="caption"
                                  sx={{
                                    display: "block",
                                    fontWeight: 900,
                                  }}
                                >
                                  Flow
                                </Typography>
                                <Stack
                                  direction="row"
                                  spacing={0.45}
                                  useFlexGap
                                  flexWrap="wrap"
                                  sx={{
                                    mt: 0.45,
                                    minWidth: 0,
                                  }}
                                >
                                  {activePipelineSummaryChips.map((chip) => (
                                    <Tooltip key={chip.key} title={chip.title}>
                                      <Chip
                                        size="small"
                                        label={chip.label}
                                        variant="outlined"
                                        sx={{
                                          height: 21,
                                          maxWidth: { xs: 132, sm: 180 },
                                          borderColor: alpha(chip.color, 0.34),
                                          bgcolor: alpha(chip.color, 0.055),
                                          color: chip.color,
                                          fontSize: "0.61rem",
                                          "& .MuiChip-label": {
                                            minWidth: 0,
                                            overflow: "hidden",
                                            textOverflow: "ellipsis",
                                          },
                                        }}
                                      />
                                    </Tooltip>
                                  ))}
                                </Stack>
                              </Box>
                              <Stack
                                direction="row"
                                spacing={0.5}
                                alignItems="center"
                                useFlexGap
                                flexWrap="wrap"
                                sx={{
                                  justifyContent: { xs: "flex-start", sm: "flex-end" },
                                  flexShrink: 0,
                                  minWidth: 0,
                                }}
                              >
                                <Button
                                  size="small"
                                  variant="outlined"
                                  startIcon={<DataIcon fontSize="small" />}
                                  aria-expanded={stageSamplesExpanded}
                                  aria-controls="graphrag-stage-samples"
                                  onClick={() =>
                                    stageSamplesExpanded
                                      ? setStageSamplesExpanded(false)
                                      : revealProcessingDetails(
                                          stageSamplesDetailsRef,
                                          setStageSamplesExpanded,
                                        )
                                  }
                                  sx={{
                                    minHeight: 22,
                                    px: 0.75,
                                    borderRadius: "8px",
                                    textTransform: "none",
                                    fontSize: "0.66rem",
                                    "& .MuiButton-startIcon": {
                                      mr: 0.4,
                                    },
                                  }}
                                >
                                  {stageSamplesExpanded ? "Hide outputs" : "View outputs"}
                                </Button>
                                <Button
                                  size="small"
                                  variant="text"
                                  startIcon={<ArticleIcon fontSize="small" />}
                                  aria-expanded={advancedTraceExpanded}
                                  aria-controls="graphrag-advanced-trace"
                                  onClick={() =>
                                    advancedTraceExpanded
                                      ? setAdvancedTraceExpanded(false)
                                      : revealProcessingDetails(
                                          advancedTraceDetailsRef,
                                          setAdvancedTraceExpanded,
                                        )
                                  }
                                  sx={{
                                    minHeight: 22,
                                    px: 0.75,
                                    borderRadius: "8px",
                                    textTransform: "none",
                                    fontSize: "0.66rem",
                                    "& .MuiButton-startIcon": {
                                      mr: 0.4,
                                    },
                                  }}
                                >
                                  {advancedTraceExpanded ? "Hide diagnostics" : "Diagnostics"}
                                </Button>
                              </Stack>
                            </Box>
                            {activePipelineIssueRows.length > 0 && (
                              <Box
                                sx={{
                                  mx: 1,
                                  mb: 0.65,
                                  px: 0.85,
                                  py: 0.65,
                                  display: "grid",
                                  gridTemplateColumns: {
                                    xs: "1fr",
                                    md: "auto minmax(0, 1fr)",
                                  },
                                  gap: { xs: 0.35, md: 0.65 },
                                  alignItems: "baseline",
                                  border: "1px solid",
                                  borderColor: alpha(accentColors.warning, 0.3),
                                  borderRadius: "8px",
                                  bgcolor: alpha(accentColors.warning, 0.06),
                                }}
                              >
                                <Typography
                                  variant="caption"
                                  sx={{
                                    color: accentColors.warning,
                                    fontWeight: 950,
                                    lineHeight: 1.2,
                                  }}
                                >
                                  Review needed
                                </Typography>
                                <Typography
                                  variant="caption"
                                  title={activePipelineIssueRows
                                    .map((issue) => `${issue.label}: ${issue.message}`)
                                    .join(" · ")}
                                  sx={{
                                    minWidth: 0,
                                    color: "text.primary",
                                    display: "-webkit-box",
                                    overflow: "hidden",
                                    WebkitBoxOrient: "vertical",
                                    WebkitLineClamp: { xs: 2, md: 1 },
                                    lineHeight: 1.25,
                                  }}
                                >
                                  {activePipelineIssueRows
                                    .slice(0, 2)
                                    .map((issue) => `${issue.label}: ${issue.message}`)
                                    .join(" · ")}
                                </Typography>
                              </Box>
                            )}
                            {activePipelineLaneRows.length > 0 && (
                              <Box
                                sx={{
                                  display: "grid",
                                  gridTemplateColumns: {
                                    xs: "repeat(2, minmax(0, 1fr))",
                                    md: "repeat(5, minmax(0, 1fr))",
                                  },
                                  gap: 0.5,
                                  px: 1,
                                  pb: 0.8,
                                }}
                              >
                                {activePipelineLaneRows.map((lane, laneIndex) => {
                                  const laneStatusColor = pipelineStageStatusColor(lane.status);
                                  const laneComplete = lane.status.toLowerCase() === "complete";
                                  return (
                                    <Box
                                      key={lane.key}
                                      data-testid="graphrag-pipeline-lane-summary"
                                      title={`${lane.label}: ${lane.sampleTitle}${lane.sampleBody ? ` - ${lane.sampleBody}` : ""}`}
                                      sx={{
                                        minWidth: 0,
                                        border: "1px solid",
                                        borderColor: alpha(lane.color, 0.24),
                                        borderRadius: "8px",
                                        bgcolor: alpha(lane.color, 0.055),
                                        px: 0.75,
                                        py: 0.65,
                                        display: "flex",
                                        flexDirection: "column",
                                        gap: 0.45,
                                      }}
                                    >
                                      <Box
                                        sx={{
                                          display: "flex",
                                          alignItems: "center",
                                          gap: 0.55,
                                          minWidth: 0,
                                        }}
                                      >
                                        <Box
                                          aria-hidden
                                          sx={{
                                            width: 19,
                                            height: 19,
                                            flexShrink: 0,
                                            display: "grid",
                                            placeItems: "center",
                                            borderRadius: "50%",
                                            border: "1px solid",
                                            borderColor: alpha(lane.color, 0.38),
                                            bgcolor: alpha(lane.color, 0.1),
                                            color: lane.color,
                                            fontSize: "0.62rem",
                                            fontWeight: 950,
                                            lineHeight: 1,
                                          }}
                                        >
                                          {laneIndex + 1}
                                        </Box>
                                        <Typography
                                          variant="caption"
                                          noWrap
                                          sx={{
                                            flex: 1,
                                            minWidth: 0,
                                            color: lane.color,
                                            fontSize: "0.68rem",
                                            fontWeight: 950,
                                          }}
                                        >
                                          {lane.label}
                                        </Typography>
                                        <Chip
                                          size="small"
                                          label={laneComplete ? "Ready" : formatLabel(lane.status)}
                                          variant="outlined"
                                          sx={{
                                            height: 18,
                                            maxWidth: 72,
                                            flexShrink: 0,
                                            borderColor: alpha(laneStatusColor, 0.42),
                                            color: laneStatusColor,
                                            fontSize: "0.58rem",
                                            "& .MuiChip-label": {
                                              px: 0.55,
                                              overflow: "hidden",
                                              textOverflow: "ellipsis",
                                            },
                                          }}
                                        />
                                      </Box>
                                      <Typography
                                        variant="caption"
                                        noWrap
                                        sx={{
                                          overflow: "hidden",
                                          color: "text.primary",
                                          fontWeight: 850,
                                          lineHeight: 1.2,
                                          textOverflow: "ellipsis",
                                        }}
                                      >
                                        {lane.sampleTitle}
                                      </Typography>
                                      <Box
                                        sx={{
                                          display: "flex",
                                          gap: 0.35,
                                          flexWrap: "wrap",
                                          mt: "auto",
                                        }}
                                      >
                                        <Chip
                                          size="small"
                                          label={lane.sampleCountLabel}
                                          sx={{
                                            height: 18,
                                            maxWidth: 124,
                                            fontSize: "0.6rem",
                                            bgcolor: alpha(lane.color, 0.1),
                                            "& .MuiChip-label": {
                                              overflow: "hidden",
                                              textOverflow: "ellipsis",
                                            },
                                          }}
                                        />
                                        <Chip
                                          size="small"
                                          label={`${formatNumber(lane.completedCount)}/${formatNumber(lane.stageCount)}`}
                                          variant="outlined"
                                          sx={{
                                            height: 18,
                                            maxWidth: 56,
                                            fontSize: "0.6rem",
                                            borderColor: alpha(lane.color, 0.28),
                                            color: "text.secondary",
                                            "& .MuiChip-label": {
                                              overflow: "hidden",
                                              textOverflow: "ellipsis",
                                            },
                                          }}
                                        />
                                        {lane.durationMs > 0 && (!laneComplete || lane.warningCount > 0) && (
                                          <Chip
                                            size="small"
                                            label={formatDuration(lane.durationMs)}
                                            variant="outlined"
                                            sx={{
                                              height: 18,
                                              maxWidth: 82,
                                              fontSize: "0.6rem",
                                              borderColor: alpha(lane.color, 0.28),
                                              color: "text.secondary",
                                              "& .MuiChip-label": {
                                                overflow: "hidden",
                                                textOverflow: "ellipsis",
                                              },
                                            }}
                                          />
                                        )}
                                        {lane.warningCount > 0 && (
                                          <Chip
                                            size="small"
                                            label={`${lane.warningCount} issue${lane.warningCount === 1 ? "" : "s"}`}
                                            variant="outlined"
                                            sx={{
                                              height: 18,
                                              fontSize: "0.6rem",
                                              borderColor: alpha(accentColors.warning, 0.36),
                                              color: accentColors.warning,
                                            }}
                                          />
                                        )}
                                      </Box>
                                    </Box>
                                  );
                                })}
                              </Box>
                            )}
                            <Box
                              component="details"
                              id="graphrag-stage-samples"
                              ref={stageSamplesDetailsRef}
                              open={stageSamplesExpanded}
                              sx={{
                                borderTop: "1px solid",
                                borderColor: alpha(accentColors.slate, 0.16),
                                "& summary": {
                                  cursor: "pointer",
                                  listStyle: "none",
                                },
                                "& summary::-webkit-details-marker": {
                                  display: "none",
                                },
                              }}
                            >
                              <Box
                                component="summary"
                                onClick={(event) => {
                                  event.preventDefault();
                                  setStageSamplesExpanded((expanded) => !expanded);
                                }}
                                sx={{
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "space-between",
                                  gap: 1,
                                  px: 1,
                                  py: 0.75,
                                }}
                              >
                                <Box sx={{ minWidth: 0 }}>
                                  <Typography
                                    variant="caption"
                                    sx={{ display: "block", fontWeight: 900 }}
                                  >
                                    Pipeline outputs
                                  </Typography>
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    noWrap
                                    sx={{ display: "block", fontSize: "0.66rem" }}
                                  >
                                    Read / cite / connect / reason / chat
                                  </Typography>
                                </Box>
                                <Stack direction="row" spacing={0.45} sx={{ flexShrink: 0 }}>
                                  <Chip
                                    size="small"
                                    label={`${formatNumber(activePipelineOutputRows.length)} outputs`}
                                    variant="outlined"
                                    sx={{ height: 20, flexShrink: 0, fontSize: "0.62rem" }}
                                  />
                                  {activeSupportStageOutputRows.length > 0 && (
                                    <Chip
                                      size="small"
                                      label={`${formatNumber(activeSupportStageOutputRows.length)} checks`}
                                      variant="outlined"
                                      sx={{ height: 20, flexShrink: 0, fontSize: "0.62rem" }}
                                    />
                                  )}
                                  <Chip
                                    size="small"
                                    label={stageSamplesExpanded ? "Hide" : "Show"}
                                    sx={{ height: 20, flexShrink: 0, fontSize: "0.62rem" }}
                                  />
                                </Stack>
                              </Box>
                              <Box
                                sx={{
                                  display: "grid",
                                  gridTemplateColumns: {
                                    xs: "1fr",
                                    lg: "repeat(2, minmax(0, 1fr))",
                                  },
                                  gap: 0.55,
                                  px: 1,
                                  pb: 1,
                                }}
                              >
                                {activePipelineOutputRows.map((stage) => (
                                  <Box
                                    key={stage.key}
                                    data-testid="graphrag-pipeline-output-row"
                                    title={`${stage.label}: ${stage.title}${stage.body ? ` - ${stage.body}` : ""}`}
                                    sx={{
                                      minWidth: 0,
                                      border: "1px solid",
                                      borderColor: alpha(stage.statusColor, 0.2),
                                      borderRadius: "8px",
                                      bgcolor: alpha(stage.statusColor, 0.045),
                                      px: 0.9,
                                      py: 0.7,
                                    }}
                                  >
                                    <Box
                                      sx={{
                                        display: "grid",
                                        gridTemplateColumns: "minmax(0, 1fr) auto",
                                        alignItems: "center",
                                        gap: 0.75,
                                        minWidth: 0,
                                      }}
                                    >
                                      <Box sx={{ minWidth: 0 }}>
                                        <Typography
                                          variant="caption"
                                          noWrap
                                          sx={{
                                            display: "block",
                                            color: stage.statusColor,
                                            fontWeight: 900,
                                            lineHeight: 1.2,
                                          }}
                                        >
                                          {stage.label}
                                        </Typography>
                                        <Typography
                                          variant="caption"
                                          color="text.secondary"
                                          noWrap
                                          sx={{ display: "block", fontSize: "0.66rem" }}
                                        >
                                          {[
                                            formatLabel(stage.status),
                                            stage.durationMs && stage.durationMs > 0
                                              ? stage.durationLabel
                                              : "",
                                          ].filter(Boolean).join(" · ")}
                                        </Typography>
                                      </Box>
                                      {stage.warningCount > 0 ? (
                                        <Chip
                                          size="small"
                                          label={`${stage.warningCount} issue${stage.warningCount === 1 ? "" : "s"}`}
                                          variant="outlined"
                                          sx={{
                                            height: 20,
                                            maxWidth: 86,
                                            borderColor: alpha(accentColors.warning, 0.36),
                                            color: accentColors.warning,
                                            fontSize: "0.62rem",
                                          }}
                                        />
                                      ) : (
                                        <Chip
                                          size="small"
                                          label={stage.sampleGroups[0] || "ready"}
                                          variant="outlined"
                                          sx={{
                                            height: 20,
                                            maxWidth: 112,
                                            borderColor: alpha(stage.statusColor, 0.34),
                                            fontSize: "0.62rem",
                                            "& .MuiChip-label": {
                                              overflow: "hidden",
                                              textOverflow: "ellipsis",
                                            },
                                          }}
                                        />
                                      )}
                                    </Box>
                                    <Typography
                                      variant="caption"
                                      sx={{
                                        display: "-webkit-box",
                                        mt: 0.45,
                                        overflow: "hidden",
                                        WebkitBoxOrient: "vertical",
                                        WebkitLineClamp: 2,
                                        color: "text.primary",
                                        fontWeight: 800,
                                        lineHeight: 1.22,
                                      }}
                                    >
                                      {stage.title}
                                    </Typography>
                                    {stage.body && (
                                      <Typography
                                        variant="caption"
                                        color="text.secondary"
                                        sx={{
                                          display: "-webkit-box",
                                          mt: 0.25,
                                          overflow: "hidden",
                                          WebkitBoxOrient: "vertical",
                                          WebkitLineClamp: 2,
                                          lineHeight: 1.24,
                                        }}
                                      >
                                        {stage.body}
                                      </Typography>
                                    )}
                                    <PipelineSampleLinkTargetRow
                                      target={stage.linkTarget}
                                      color={stage.statusColor}
                                      compact
                                    />
                                    {stage.sampleGroups.length > 1 && (
                                      <Box
                                        sx={{
                                          display: "flex",
                                          gap: 0.35,
                                          flexWrap: "wrap",
                                          mt: 0.55,
                                        }}
                                      >
                                        {stage.sampleGroups.slice(1, 3).map((group) => (
                                          <Chip
                                            key={`${stage.key}-${group}`}
                                            size="small"
                                            label={group}
                                            sx={{
                                              height: 18,
                                              maxWidth: 130,
                                              fontSize: "0.6rem",
                                              bgcolor: alpha(stage.statusColor, 0.08),
                                              "& .MuiChip-label": {
                                                overflow: "hidden",
                                                textOverflow: "ellipsis",
                                              },
                                            }}
                                          />
                                        ))}
                                      </Box>
                                    )}
                                  </Box>
                                ))}
                              </Box>
                            </Box>
                          </Box>
                        )}

                        {evidenceRelationshipReviewVisible && (
                          <Box
                            sx={{
                              order: 5,
                              border: "1px solid",
                              borderColor: alpha(accentColors.success, 0.2),
                              borderRadius: "8px",
                              overflow: "hidden",
                              bgcolor: alpha(accentColors.slate, 0.035),
                            }}
                          >
                            <Box
                              sx={{
                                display: "grid",
                                gridTemplateColumns: {
                                  xs: "1fr",
                                  sm: "minmax(0, 1fr) auto",
                                },
                                alignItems: "center",
                                gap: 0.75,
                                px: 1,
                                py: 0.8,
                                borderBottom: "1px solid",
                                borderColor: "divider",
                              }}
                            >
                              <Box sx={{ minWidth: 0 }}>
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{ display: "block", fontWeight: 850 }}
                                >
                                  Key relationships
                                </Typography>
                                <Typography
                                  variant="body2"
                                  noWrap
                                  sx={{ fontWeight: 900 }}
                                >
                                  {selectedActiveGraphRow?.text || "Document references"}
                                </Typography>
                              </Box>
                              <Stack
                                direction="row"
                                spacing={0.5}
                                alignItems="center"
                                sx={{
                                  justifySelf: { xs: "flex-start", sm: "flex-end" },
                                  flexShrink: 0,
                                }}
                              >
                                <Chip
                                  size="small"
                                  label={`${formatNumber(activeGraphRows.length)} items`}
                                  variant="outlined"
                                  sx={{ height: 22, flexShrink: 0 }}
                                />
                                <Button
                                  size="small"
                                  variant="contained"
                                  startIcon={<ChatIcon fontSize="small" />}
                                  aria-label={
                                    selectedActiveGraphRow?.text
                                      ? `Ask about ${selectedActiveGraphRow.text}`
                                      : "Ask about selected relationship"
                                  }
                                  disabled={!selectedActiveGraphPrompt}
                                  onClick={handleAskAboutSelectedGraphRow}
                                  sx={{
                                    borderRadius: "8px",
                                    minHeight: 28,
                                    px: 1,
                                    textTransform: "none",
                                    boxShadow: "none",
                                  }}
                                >
                                  Ask about this
                                </Button>
                              </Stack>
                            </Box>
                            <Box
                              sx={{
                                display: "grid",
                                gridTemplateColumns: {
                                  xs: "1fr",
                                  xl: "repeat(3, minmax(0, 1fr))",
                                },
                                gap: 0.55,
                                px: 0.75,
                                py: 0.75,
                              }}
                            >
                              {activeGraphRows.slice(0, 3).map((row) => {
                                const isSelected =
                                  selectedActiveGraphRow?.key === row.key;
                                const rowKind =
                                  row.kind === "path" ? "Path" : "Relation";
                                const relationLabel =
                                  row.kind === "path"
                                    ? "Path"
                                    : row.relation
                                      ? formatGraphRelationLabel(row.relation)
                                      : row.label;
                                return (
                                  <Box
                                    key={row.key}
                                    role="button"
                                    tabIndex={0}
                                    title={row.title || row.text}
                                    onClick={() => setSelectedEvidenceRowKey(row.key)}
                                    onKeyDown={(event) => {
                                      if (event.key === "Enter" || event.key === " ") {
                                        event.preventDefault();
                                        setSelectedEvidenceRowKey(row.key);
                                      }
                                    }}
                                    sx={{
                                      display: "grid",
                                      gridTemplateColumns: {
                                        xs: "1fr",
                                        md: "minmax(0, 1fr) auto",
                                      },
                                      alignItems: "center",
                                      gap: 0.55,
                                      minWidth: 0,
                                      px: 0.85,
                                      py: 0.68,
                                      borderRadius: "8px",
                                      bgcolor: alpha(row.color, isSelected ? 0.14 : 0.055),
                                      border: "1px solid",
                                      borderColor: alpha(
                                        row.color,
                                        isSelected ? 0.62 : 0.16,
                                      ),
                                      cursor: "pointer",
                                      outline: "none",
                                      transition:
                                        "background-color 120ms ease, border-color 120ms ease",
                                      "&:hover": {
                                        bgcolor: alpha(row.color, 0.1),
                                        borderColor: alpha(row.color, 0.42),
                                      },
                                      "&:focus-visible": {
                                        borderColor: row.color,
                                        boxShadow: `0 0 0 2px ${alpha(row.color, 0.16)}`,
                                      },
                                    }}
                                  >
                                    <Box sx={{ minWidth: 0 }}>
                                      <Box
                                        sx={{
                                          display: "flex",
                                          alignItems: "center",
                                          gap: 0.45,
                                          minWidth: 0,
                                        }}
                                      >
                                        <Chip
                                          size="small"
                                          label={rowKind}
                                          variant="outlined"
                                          sx={{
                                            height: 20,
                                            fontSize: "0.62rem",
                                            borderColor: alpha(row.color, 0.34),
                                            flexShrink: 0,
                                          }}
                                        />
                                        <Typography
                                          variant="caption"
                                          noWrap
                                          sx={{ minWidth: 0, fontWeight: 900 }}
                                        >
                                          {row.source || row.text}
                                        </Typography>
                                        {row.bridge && (
                                          <>
                                            <Typography
                                              variant="caption"
                                              color="text.secondary"
                                              sx={{ flexShrink: 0 }}
                                            >
                                              to
                                            </Typography>
                                            <Typography
                                              variant="caption"
                                              noWrap
                                              sx={{
                                                minWidth: 0,
                                                maxWidth: "28%",
                                                color: row.color,
                                                fontWeight: 900,
                                              }}
                                            >
                                              {row.bridge}
                                            </Typography>
                                          </>
                                        )}
                                        {row.target && (
                                          <>
                                            <Typography
                                              variant="caption"
                                              color="text.secondary"
                                              sx={{ flexShrink: 0 }}
                                            >
                                              to
                                            </Typography>
                                            <Typography
                                              variant="caption"
                                              noWrap
                                              sx={{ minWidth: 0, fontWeight: 900 }}
                                            >
                                              {row.target}
                                            </Typography>
                                          </>
                                        )}
                                      </Box>
                                      {row.title && row.title !== row.text && (
                                        <Typography
                                          variant="caption"
                                          noWrap
                                          color="text.secondary"
                                          sx={{ display: "block", mt: 0.25 }}
                                        >
                                          {row.title}
                                        </Typography>
                                      )}
                                    </Box>
                                    <Chip
                                      size="small"
                                      label={relationLabel}
                                      variant={isSelected ? "filled" : "outlined"}
                                      sx={{
                                        height: 21,
                                        maxWidth: { xs: "100%", sm: 160 },
                                        justifySelf: { xs: "flex-start", sm: "flex-end" },
                                        bgcolor: isSelected
                                          ? alpha(row.color, 0.18)
                                          : "transparent",
                                        borderColor: alpha(row.color, 0.36),
                                        color: isSelected ? row.color : "text.secondary",
                                        fontSize: "0.62rem",
                                        "& .MuiChip-label": {
                                          minWidth: 0,
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                        },
                                      }}
                                    />
                                  </Box>
                                );
                              })}
                              {activeGraphRows.length > 3 && (
                                <Box
                                  sx={{
                                    display: "flex",
                                    alignItems: "center",
                                    minHeight: 56,
                                    px: 0.85,
                                    py: 0.68,
                                    borderRadius: "8px",
                                    border: "1px dashed",
                                    borderColor: alpha(accentColors.success, 0.2),
                                    color: "text.secondary",
                                  }}
                                >
                                  <Typography variant="caption" fontWeight={800}>
                                    +{formatNumber(activeGraphRows.length - 3)} more in details
                                  </Typography>
                                </Box>
                              )}
                            </Box>
                            {false && selectedActiveGraphRow && (
                              <Box
                                sx={{
                                  display: "grid",
                                  gridTemplateColumns: {
                                    xs: "1fr",
                                    sm: "minmax(0, 1fr) auto",
                                  },
                                  alignItems: "center",
                                  gap: 0.85,
                                  px: 1,
                                  py: 0.9,
                                  borderTop: "1px solid",
                                  borderColor: "divider",
                                  bgcolor: alpha(selectedActiveGraphRow.color, 0.055),
                                }}
                              >
                                <Box sx={{ minWidth: 0 }}>
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{ display: "block", fontWeight: 850 }}
                                  >
                                    Selected{" "}
                                    {selectedActiveGraphRow.kind === "path"
                                      ? "path"
                                      : "relationship"}
                                  </Typography>
                                  <Typography
                                    variant="body2"
                                    title={selectedActiveGraphRow.text}
                                    noWrap
                                    sx={{ fontWeight: 900, minWidth: 0 }}
                                  >
                                    {selectedActiveGraphRow.text}
                                  </Typography>
                                  {selectedActiveGraphRow.title && (
                                    <Typography
                                      variant="caption"
                                      color="text.secondary"
                                      sx={{
                                        display: "-webkit-box",
                                        mt: 0.25,
                                        overflow: "hidden",
                                        WebkitBoxOrient: "vertical",
                                        WebkitLineClamp: 2,
                                        lineHeight: 1.25,
                                      }}
                                    >
                                      {selectedActiveGraphRow.title}
                                    </Typography>
                                  )}
                                </Box>
                                <Button
                                  size="small"
                                  variant="contained"
                                  startIcon={<ChatIcon fontSize="small" />}
                                  disabled={!selectedActiveGraphPrompt}
                                  onClick={handleAskAboutSelectedGraphRow}
                                  sx={{
                                    borderRadius: "8px",
                                    minHeight: 30,
                                    px: 1.1,
                                    textTransform: "none",
                                    boxShadow: "none",
                                    justifySelf: { xs: "flex-start", sm: "flex-end" },
                                  }}
                                >
                                  Ask with this reference
                                </Button>
                              </Box>
                            )}
                          </Box>
                        )}

                        {documentInsightView === "evidence" &&
                          (activeStageSampleRows.length > 0 ||
                            activeEvidenceChips.length > 0) && (
                          <Box
                            component="details"
                            data-testid="graphrag-extraction-evidence-details"
                            sx={{
                              order: 4,
                              border: "1px solid",
                              borderColor: alpha(accentColors.info, 0.18),
                              borderRadius: "8px",
                              overflow: "hidden",
                              bgcolor: alpha(accentColors.slate, 0.035),
                              "& summary": {
                                cursor: "pointer",
                                listStyle: "none",
                              },
                              "& summary::-webkit-details-marker": {
                                display: "none",
                              },
                            }}
                          >
                            <Box
                              component="summary"
                              sx={{
                                px: 1,
                                py: 0.85,
                                display: "grid",
                                gridTemplateColumns: "minmax(0, 1fr)",
                                gap: 1,
                                minWidth: 0,
                              }}
                            >
                              <Box
                                sx={{
                                  display: "flex",
                                  alignItems: "center",
                                  justifyContent: "space-between",
                                  gap: 1,
                                  minWidth: 0,
                                }}
                              >
                                <Box sx={{ minWidth: 0 }}>
                                  <Typography
                                    variant="caption"
                                    sx={{ display: "block", fontWeight: 900 }}
                                  >
                                    Extraction evidence
                                  </Typography>
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    noWrap
                                    sx={{ display: "block" }}
                                  >
                                    {[
                                      activeStageSampleRows.length
                                        ? `${activeStageSampleRows.length} evidence samples`
                                        : "",
                                      activeEvidenceChips.length
                                        ? `${activeEvidenceChips.length} evidence types`
                                        : "",
                                    ]
                                      .filter(Boolean)
                                      .join(" · ")}
                                  </Typography>
                                </Box>
                                <Chip
                                  size="small"
                                  label="Show evidence"
                                  variant="outlined"
                                  sx={{
                                    height: 22,
                                    flexShrink: 0,
                                    fontSize: "0.62rem",
                                  }}
                                />
                              </Box>
                              {activeEvidencePreviewChips.length > 0 && (
                                <Box
                                  sx={{
                                    display: "flex",
                                    gap: 0.4,
                                    flexWrap: "wrap",
                                  }}
                                >
                                  {activeEvidencePreviewChips.map((chip) => (
                                    <Chip
                                      key={`sample-preview-${chip.label}`}
                                      label={`${formatNumber(chip.value)} ${chip.label}`}
                                      size="small"
                                      variant="outlined"
                                      sx={{
                                        height: 21,
                                        maxWidth: 170,
                                        fontSize: "0.62rem",
                                        bgcolor: alpha(accentColors.info, 0.045),
                                        "& .MuiChip-label": {
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                        },
                                      }}
                                    />
                                  ))}
                                </Box>
                              )}
                            </Box>
                            <Stack spacing={1} sx={{ p: 1, pt: 0 }}>
                              {activeStageSampleRows.length > 0 && (
                                <Box
                                  sx={{
                                    display: "grid",
                                    gridTemplateColumns: {
                                      xs: "1fr",
                                      xl: "repeat(2, minmax(0, 1fr))",
                                    },
                                    gap: 0.75,
                                  }}
                                >
                                  {activeStageSampleRows.map((row) => (
                                    <Box
                                      key={row.key}
                                      data-testid="graphrag-extraction-evidence-row"
                                      data-evidence-sample-key={row.key}
                                      data-evidence-sample-label={row.label}
                                      data-evidence-sample-title={row.title}
                                      title={`${row.title}: ${row.body}${row.linkTarget ? ` Link target: ${row.linkTarget.url}` : ""}`}
                                      sx={{
                                        minWidth: 0,
                                        border: "1px solid",
                                        borderColor: alpha(row.color, 0.2),
                                        borderRadius: "8px",
                                        px: 1,
                                        py: 0.85,
                                        bgcolor: alpha(row.color, 0.045),
                                      }}
                                    >
                                      <Box
                                        sx={{
                                          display: "flex",
                                          alignItems: "center",
                                          justifyContent: "space-between",
                                          gap: 0.75,
                                          minWidth: 0,
                                          mb: 0.45,
                                        }}
                                      >
                                        <Typography
                                          variant="caption"
                                          noWrap
                                          sx={{
                                            minWidth: 0,
                                            color: row.color,
                                            fontSize: "0.67rem",
                                            fontWeight: 900,
                                          }}
                                        >
                                          {row.label}
                                        </Typography>
                                        <Stack
                                          direction="row"
                                          spacing={0.35}
                                          alignItems="center"
                                          sx={{ flexShrink: 0 }}
                                        >
                                          <Chip
                                            size="small"
                                            label={row.countLabel}
                                            variant="outlined"
                                            sx={{
                                              height: 19,
                                              maxWidth: 92,
                                              borderColor: alpha(row.color, 0.35),
                                              fontSize: "0.62rem",
                                              flexShrink: 0,
                                              "& .MuiChip-label": {
                                                px: 0.65,
                                                overflow: "hidden",
                                                textOverflow: "ellipsis",
                                              },
                                            }}
                                          />
                                          <Button
                                            data-testid="graphrag-extraction-evidence-ask"
                                            size="small"
                                            variant="text"
                                            startIcon={<ChatIcon fontSize="small" />}
                                            title={`Ask about ${row.label}: ${row.title}`}
                                            onClick={() => handleAskAboutPipelineSampleRow(row)}
                                            sx={{
                                              minHeight: 22,
                                              px: 0.55,
                                              borderRadius: "8px",
                                              color: row.color,
                                              textTransform: "none",
                                              fontSize: "0.62rem",
                                              "& .MuiButton-startIcon": {
                                                mr: 0.3,
                                              },
                                            }}
                                          >
                                            Ask
                                          </Button>
                                        </Stack>
                                      </Box>
                                      <Typography
                                        variant="caption"
                                        sx={{
                                          display: "-webkit-box",
                                          minHeight: "2.3em",
                                          overflow: "hidden",
                                          WebkitBoxOrient: "vertical",
                                          WebkitLineClamp: 2,
                                          color: "text.primary",
                                          fontWeight: 850,
                                          lineHeight: 1.15,
                                        }}
                                      >
                                        {row.title}
                                      </Typography>
                                      {row.body && (
                                        <Typography
                                          variant="caption"
                                          color="text.secondary"
                                          sx={{
                                            display: "-webkit-box",
                                            mt: 0.45,
                                            minHeight: "2.6em",
                                            overflow: "hidden",
                                            WebkitBoxOrient: "vertical",
                                            WebkitLineClamp: 2,
                                            lineHeight: 1.25,
                                          }}
                                        >
                                          {row.body}
                                        </Typography>
                                      )}
                                      <PipelineSampleLinkTargetRow
                                        target={row.linkTarget}
                                        color={row.color}
                                      />
                                    </Box>
                                  ))}
                                </Box>
                              )}
                            </Stack>
                          </Box>
                        )}

                        {false && activeSuggestedQuestions.length > 0 && (
                          <Box
                            sx={{
                              display: "grid",
                              gridTemplateColumns: {
                                xs: "1fr",
                                xl: "minmax(0, 1fr)",
                              },
                              gap: 1,
                            }}
                          >
                            <Box>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                sx={{
                                  display: "block",
                                  fontWeight: 850,
                                  mb: 0.5,
                                }}
                              >
                                Ask next
                              </Typography>
                              <Box
                                sx={{
                                  display: "flex",
                                  gap: 0.5,
                                  flexWrap: "wrap",
                                }}
                              >
                                {activeSuggestedQuestions.map((question) => {
                                  const questionLabel = graphRAGQuestionLabel(question);
                                  return (
                                  <Tooltip
                                    key={question.id}
                                    title={question.prompt}
                                    describeChild
                                  >
                                    <Button
                                      size="small"
                                      variant="outlined"
                                      startIcon={<ChatIcon fontSize="small" />}
                                      onClick={() =>
                                        corpusHasBatchResult
                                          ? handleAskChatAcrossBatch(
                                              question.prompt,
                                              {},
                                              questionLabel,
                                            )
                                          : activeDocumentGroup
                                            ? handleAskChatFromPipelineRuns(
                                                activeDocumentGroup.runs,
                                                question.prompt,
                                                {},
                                                questionLabel,
                                              )
                                          : handleAskChatFromPipelineRun(
                                              activeCorpusPipelineRecord,
                                              question.prompt,
                                              {},
                                              questionLabel,
                                            )
                                      }
                                      sx={{
                                        minHeight: 26,
                                        maxWidth: { xs: "100%", sm: 260 },
                                        px: 1,
                                        py: 0.15,
                                        borderRadius: "8px",
                                        textTransform: "none",
                                        justifyContent: "flex-start",
                                        "& .MuiButton-startIcon": {
                                          flexShrink: 0,
                                        },
                                      }}
                                    >
                                      <Box
                                        component="span"
                                        sx={{
                                          minWidth: 0,
                                          overflow: "hidden",
                                          textOverflow: "ellipsis",
                                          whiteSpace: "nowrap",
                                        }}
                                      >
                                        {questionLabel}
                                      </Box>
                                    </Button>
                                  </Tooltip>
                                  );
                                })}
                              </Box>
                            </Box>
                          </Box>
                        )}
                      </Stack>
                    </Box>
                    {documentInsightView === "processing" && (
                      <Box
                        component="details"
                        id="graphrag-advanced-trace"
                        ref={advancedTraceDetailsRef}
                        open={advancedTraceExpanded}
                        sx={{
                          border: "1px solid",
                          borderColor: "divider",
                          borderRadius: "8px",
                          overflow: "hidden",
                          "& summary": {
                            cursor: "pointer",
                            px: 1.25,
                            py: 0.85,
                            fontSize: "0.78rem",
                            fontWeight: 800,
                            color: "text.secondary",
                            listStyle: "none",
                          },
                          "& summary::-webkit-details-marker": {
                            display: "none",
                          },
                        }}
                        >
                          <Box
                            component="summary"
                            onClick={(event) => {
                              event.preventDefault();
                              setAdvancedTraceExpanded((expanded) => !expanded);
                            }}
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "space-between",
                              gap: 1,
                            }}
                          >
                            <Box sx={{ minWidth: 0 }}>
                              <Typography
                                variant="caption"
                                sx={{ display: "block", fontWeight: 900 }}
                              >
                                Advanced diagnostics
                              </Typography>
                              <Typography
                                variant="caption"
                                color="text.secondary"
                                noWrap
                                sx={{ display: "block", fontSize: "0.66rem" }}
                              >
                                Full run diagnostics and artifact-level context
                              </Typography>
                            </Box>
                            <Chip
                              size="small"
                              label={advancedTraceExpanded ? "Hide" : "Show"}
                              sx={{ height: 20, flexShrink: 0, fontSize: "0.62rem" }}
                            />
                          </Box>
                          <Box sx={{ p: 1 }}>
                            <GraphRAGPipelineTrace
                              result={corpusPipelineResult}
                              title={corpusPipelineTitle}
                            />
                          </Box>
                      </Box>
                    )}
                    {pipelineRuns.length === 0 && (
                      <Box
                        component="details"
                        sx={{
                          border: "1px solid",
                          borderColor: "divider",
                          borderRadius: "8px",
                          overflow: "hidden",
                          "& summary": {
                            cursor: "pointer",
                            px: 1.25,
                            py: 0.85,
                            fontSize: "0.78rem",
                            fontWeight: 800,
                            color: "text.secondary",
                            listStyle: "none",
                          },
                          "& summary::-webkit-details-marker": {
                            display: "none",
                          },
                        }}
                      >
                        <Box component="summary">Raw response</Box>
                        <Box sx={{ p: 1 }}>
                          <JsonResult value={corpusResult} />
                        </Box>
                      </Box>
                    )}
                  </Stack>
                ) : Boolean(corpusResult) || pipelineRuns.length > 0 ? (
                  <Stack spacing={1}>
                    {pipelineRuns.length > 0 ? (
                      <Box
                        sx={{
                          border: "1px solid",
                          borderColor: alpha(accentColors.primary, 0.24),
                          borderRadius: "8px",
                          p: { xs: 1, sm: 1.35 },
                          minHeight: 300,
                          bgcolor: alpha(accentColors.primary, 0.035),
                        }}
                      >
                        <Stack spacing={1.25}>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "flex-start",
                              justifyContent: "space-between",
                              gap: 1,
                              flexWrap: "wrap",
                            }}
                          >
                            <Box sx={{ minWidth: 0 }}>
                              <Typography variant="subtitle2" fontWeight={900}>
                                Ask your documents
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Ask across the set, or choose a document from the list for focused references.
                              </Typography>
                            </Box>
                            <Button
                              variant="contained"
                              startIcon={<ChatIcon fontSize="small" />}
                              disabled={
                                corpusLoading ||
                                chatContextPipelineRunIds.length === 0
                              }
                              onClick={() => handleAskChatAcrossRecent()}
                              sx={{
                                borderRadius: "8px",
                                minHeight: 38,
                                px: 1.5,
                                textTransform: "none",
                                boxShadow: "none",
                                flexShrink: 0,
                            }}
                          >
                              {corpusChatButtonLabel}
                            </Button>
                          </Box>
                          <Box
                            component="form"
                            onSubmit={(event) => {
                              event.preventDefault();
                              handleAskGraphChatQuestion();
                            }}
                            sx={{
                              display: "grid",
                              gridTemplateColumns: {
                                xs: "minmax(0, 1fr) 44px",
                                sm: "minmax(0, 1fr) auto",
                              },
                              gap: 0.75,
                              alignItems: "center",
                            }}
                          >
                            <TextField
                              value={graphChatQuestion}
                              onChange={(event) =>
                                setGraphChatQuestion(event.target.value)
                              }
                              size="small"
                              fullWidth
                              placeholder="Ask across your processed documents"
                              disabled={!canAskGraphChatQuestion}
                              InputProps={{
                                startAdornment: (
                                  <InputAdornment position="start">
                                    <ChatIcon sx={{ fontSize: 17 }} />
                                  </InputAdornment>
                                ),
                              }}
                              sx={{
                                "& .MuiOutlinedInput-root": {
                                  minHeight: 42,
                                  borderRadius: "8px",
                                  fontSize: "0.86rem",
                                  bgcolor: alpha(accentColors.slate, 0.035),
                                },
                              }}
                            />
                            <Button
                              type="submit"
                              variant={
                                trimmedGraphChatQuestion && canAskGraphChatQuestion
                                  ? "contained"
                                  : "outlined"
                              }
                              startIcon={<SendIcon fontSize="small" />}
                              aria-label="Ask"
                              disabled={
                                !trimmedGraphChatQuestion ||
                                !canAskGraphChatQuestion
                              }
                              sx={{
                                borderRadius: "8px",
                                minWidth: { xs: 44, sm: 76 },
                                minHeight: 42,
                                px: { xs: 0, sm: 1.35 },
                                textTransform: "none",
                                boxShadow: "none",
                                whiteSpace: "nowrap",
                                "& .MuiButton-startIcon": {
                                  ml: 0,
                                  mr: { xs: 0, sm: 0.75 },
                                },
                              }}
                            >
                              <Box
                                component="span"
                                sx={{ display: { xs: "none", sm: "inline" } }}
                              >
                                Ask
                              </Box>
                            </Button>
                          </Box>
                          <Box
                            sx={{
                              display: "flex",
                              gap: 0.45,
                              flexWrap: "wrap",
                            }}
                          >
                            {[
                              { label: "Documents", value: pipelineRuns.length },
                              {
                                label: "Chat ready",
                                value: documentChatReadyCount,
                              },
                              {
                                label: "Relationships",
                                value: pipelineRunTotals.relationships,
                              },
                              {
                                label: "Chunks",
                                value: pipelineRunTotals.chunks,
                              },
                            ].map((stat) => (
                              <Chip
                                key={stat.label}
                                label={`${formatNumber(stat.value)} ${stat.label.toLowerCase()}`}
                                size="small"
                                variant="outlined"
                                sx={{
                                  height: 22,
                                  maxWidth: 160,
                                  borderColor: alpha(accentColors.primary, 0.24),
                                  color: "text.secondary",
                                  bgcolor: alpha(accentColors.slate, 0.035),
                                  fontSize: "0.64rem",
                                  "& .MuiChip-label": {
                                    minWidth: 0,
                                    overflow: "hidden",
                                    textOverflow: "ellipsis",
                                  },
                                }}
                              />
                            ))}
                          </Box>
                          <Box>
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              sx={{ display: "block", fontWeight: 850, mb: 0.5 }}
                            >
                              Recent documents
                            </Typography>
                            <Stack spacing={0.55}>
                              {pipelineRuns.slice(0, 3).map((run, index) => {
                                const document = asRecord(run.document);
                                const summary = asRecord(run.summary);
                                const filename = asString(
                                  document.filename,
                                  `Run ${index + 1}`,
                                );
                                return (
                                  <Box
                                    key={`${filename}-${index}`}
                                    sx={{
                                      display: "grid",
                                      gridTemplateColumns: "minmax(0, 1fr) auto",
                                      alignItems: "center",
                                      gap: 1,
                                      px: 1,
                                      py: 0.7,
                                      borderRadius: "8px",
                                      border: "1px solid",
                                      borderColor: alpha(accentColors.success, 0.16),
                                      bgcolor: alpha(accentColors.slate, 0.035),
                                    }}
                                  >
                                    <Typography
                                      variant="caption"
                                      noWrap
                                      title={filename}
                                      sx={{ fontWeight: 800, minWidth: 0 }}
                                    >
                                      {filename}
                                    </Typography>
                                    <Typography
                                      variant="caption"
                                      color="text.secondary"
                                      sx={{ flexShrink: 0 }}
                                    >
                                      {asNumber(summary.relationship_candidate_count) > 0
                                        ? `${formatNumber(asNumber(summary.relationship_candidate_count))} relations`
                                        : "Ready"}
                                    </Typography>
                                  </Box>
                                );
                              })}
                            </Stack>
                          </Box>
                        </Stack>
                      </Box>
                    ) : (
                      <Box
                        sx={{
                          border: "1px solid",
                          borderColor: alpha(accentColors.primary, 0.2),
                          borderRadius: "8px",
                          p: 1.25,
                          bgcolor: alpha(accentColors.info, 0.04),
                        }}
                      >
                        <Typography variant="subtitle2" fontWeight={800}>
                          Corpus response ready
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Open the raw response only when you need the service payload.
                        </Typography>
                      </Box>
                    )}
                    {pipelineRuns.length === 0 && Boolean(corpusResult) && (
                      <Box
                        component="details"
                        sx={{
                          border: "1px solid",
                          borderColor: "divider",
                          borderRadius: "8px",
                          overflow: "hidden",
                          "& summary": {
                            cursor: "pointer",
                            px: 1.25,
                            py: 0.85,
                            fontSize: "0.78rem",
                            fontWeight: 800,
                            color: "text.secondary",
                            listStyle: "none",
                          },
                          "& summary::-webkit-details-marker": {
                            display: "none",
                          },
                        }}
                      >
                        <Box component="summary">Raw response</Box>
                        <Box sx={{ p: 1 }}>
                          <JsonResult value={corpusResult} />
                        </Box>
                      </Box>
                    )}
                  </Stack>
                ) : (
                  <Box
                    sx={{
                      border: "1px solid",
                      borderColor: alpha(accentColors.primary, 0.2),
                      borderRadius: "8px",
                      p: 2,
                      minHeight: 300,
                      bgcolor: alpha(accentColors.info, 0.04),
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <Stack spacing={1} alignItems="center" textAlign="center">
                      <Box
                        sx={{
                          width: 44,
                          height: 44,
                          borderRadius: "8px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          bgcolor: alpha(accentColors.primary, 0.1),
                          color: accentColors.primary,
                        }}
                      >
                        <GraphIcon />
                      </Box>
                      <Typography variant="subtitle2" fontWeight={800}>
                        Waiting for a document
                      </Typography>
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ maxWidth: 360 }}
                      >
                        Process a file or load recent runs. The relationship
                        summary and chat handoff will appear here.
                      </Typography>
                    </Stack>
                  </Box>
                )}
              </Box>
            </Box>
          </TabPanel>

          <TabPanel value={activeTab} index={7}>
            <Grid container spacing={1.25}>
              <Grid item xs={12} lg={5}>
                <Stack spacing={1}>
                  <Box sx={fieldGridSx}>
                    <TextField
                      label="Domain"
                      size="small"
                      value={graphDomain}
                      onChange={(event) => setGraphDomain(event.target.value)}
                    />
                    <TextField
                      label="Entity"
                      size="small"
                      value={graphEntityName}
                      onChange={(event) =>
                        setGraphEntityName(event.target.value)
                      }
                    />
                    <TextField
                      label="Limit"
                      size="small"
                      type="number"
                      inputProps={{ min: 1, max: 500 }}
                      value={graphLimit}
                      onChange={(event) =>
                        setGraphLimit(
                          Math.max(1, Number(event.target.value) || 1),
                        )
                      }
                    />
                  </Box>
                  {graphError && (
                    <Alert severity="error" variant="outlined">
                      {graphError}
                    </Alert>
                  )}
                  <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                    <Button
                      startIcon={
                        graphLoading ? (
                          <CircularProgress size={16} />
                        ) : (
                          <RunIcon />
                        )
                      }
                      disabled={graphLoading || !graphEntityName.trim()}
                      onClick={handleGraphNeighborhood}
                      sx={actionButtonSx(accentColors.primary)}
                    >
                      Neighborhood
                    </Button>
                    <Button
                      disabled={graphLoading}
                      onClick={handleGraphFiltered}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Filtered Graph
                    </Button>
                    <Button
                      disabled={graphLoading}
                      onClick={handleGraphTopEntities}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Top Entities
                    </Button>
                    <Button
                      disabled={graphLoading}
                      onClick={handleGraphTopRelationships}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Top Relationships
                    </Button>
                    <Button
                      disabled={graphLoading}
                      onClick={handleGraphCommunities}
                      variant="outlined"
                      sx={{ borderRadius: "8px" }}
                    >
                      Communities
                    </Button>
                  </Stack>
                </Stack>
              </Grid>
              <Grid item xs={12} lg={7}>
                <JsonResult
                  value={
                    graphResult ?? { stats: graphStats, domains: domainStats }
                  }
                />
              </Grid>
            </Grid>
          </TabPanel>

          <TabPanel value={activeTab} index={8}>
            <Grid container spacing={1.25}>
              <Grid item xs={12} lg={5}>
                <Stack spacing={1}>
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: {
                        xs: "1fr",
                        sm: "120px minmax(0, 1fr)",
                      },
                      gap: 1,
                    }}
                  >
                    <FormControl size="small">
                      <InputLabel>Method</InputLabel>
                      <Select
                        label="Method"
                        value={bridgeMethod}
                        onChange={(event) =>
                          setBridgeMethod(event.target.value as UpstreamMethod)
                        }
                      >
                        {UPSTREAM_METHODS.map((method) => (
                          <MenuItem key={method} value={method}>
                            {method}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                    <TextField
                      label="Path"
                      size="small"
                      value={bridgePath}
                      onChange={(event) => setBridgePath(event.target.value)}
                      fullWidth
                    />
                  </Box>
                  <TextField
                    label="Params JSON"
                    value={bridgeParams}
                    onChange={(event) => setBridgeParams(event.target.value)}
                    multiline
                    minRows={4}
                    fullWidth
                  />
                  <TextField
                    label="Body JSON"
                    value={bridgeBody}
                    onChange={(event) => setBridgeBody(event.target.value)}
                    multiline
                    minRows={4}
                    fullWidth
                  />
                  <Box
                    sx={{
                      display: "flex",
                      gap: 0.5,
                      flexWrap: "wrap",
                      maxHeight: 82,
                      overflow: "auto",
                    }}
                  >
                    {allowedPrefixes.slice(0, 24).map((prefix) => (
                      <StatusChip
                        key={prefix}
                        label={prefix}
                        status="available"
                        statusConfig={STATUS_CONFIG}
                        height={22}
                      />
                    ))}
                  </Box>
                  {bridgeError && (
                    <Alert severity="error" variant="outlined">
                      {bridgeError}
                    </Alert>
                  )}
                  <Button
                    startIcon={
                      bridgeLoading ? (
                        <CircularProgress size={16} />
                      ) : (
                        <RunIcon />
                      )
                    }
                    disabled={
                      bridgeLoading || !bridgeAvailable || !bridgePath.trim()
                    }
                    onClick={handleRunBridge}
                    sx={actionButtonSx(accentColors.primary)}
                  >
                    Call Bridge
                  </Button>
                </Stack>
              </Grid>
              <Grid item xs={12} lg={7}>
                <JsonResult
                  value={
                    bridgeResult ?? {
                      upstream_bridge: bridgeRecord,
                      allowed_prefixes: allowedPrefixes,
                    }
                  }
                />
              </Grid>
            </Grid>
          </TabPanel>
        </SectionCard>
      </Stack>
    </PageShell>
  );
};

export default GraphRAGWorkbenchPage;
