import React, {
  lazy,
  Suspense,
  useState,
  useRef,
  useEffect,
  useCallback,
  useMemo,
  ChangeEvent,
  KeyboardEvent,
} from "react";
import { flushSync } from "react-dom";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Paper,
  Avatar,
  Stack,
  Chip,
  IconButton,
  Tooltip,
  FormControl,
  Slider,
  Grid,
  Collapse,
  Drawer,
  Divider,
  Link,
  CircularProgress,
  Badge,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Switch,
  FormControlLabel,
  Select,
  MenuItem,
} from "@mui/material";
import { alpha } from "@mui/material/styles";
import {
  Send as SendIcon,
  Person as UserIcon,
  SmartToy as BotIcon,
  Settings as SettingsIcon,
  ContentCopy as CopyIcon,
  Edit as EditIcon,
  Check as CheckIcon,
  Build as ToolIcon,
  CheckBox as CheckBoxIcon,
  CheckBoxOutlineBlank as CheckBoxOutlineBlankIcon,
  History as HistoryIcon,
  Save as SaveIcon,
  Add as AddIcon,
  AttachFile as AttachFileIcon,
  Close as CloseIcon,
  Mic as MicIcon,
  Stop as StopIcon,
  VolumeUp as VolumeUpIcon,
  VolumeOff as VolumeOffIcon,
  PlayArrow as PlayIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  HeadsetMic as HeadsetMicIcon,
  CallSplit as ForkIcon,
  Code as CodeIcon,
} from "@mui/icons-material";
import { useVoiceInput } from "@/hooks/useVoiceInput";
import { useTTS } from "@/hooks/useTTS";
import { useClipboard } from "@/hooks/useClipboard";
import { apiService } from "@/services/api";
import { ToolsService, ExtendedTool } from "@/services/tools";
import type { RAGChunk } from "@/components/chat/RAGContextBlock";
import type {
  UnifiedChunk,
  UnifiedRelationship,
  QualitySignals,
  GraphReasoning,
  PipelineArtifactContext,
} from "@/components/chat/UnifiedContextBlock";
import type { ChatReferenceItem } from "@/components/chat/ReferenceExplorer";
import {
  buildChatOutputConstraints,
  structuredOutputConflictsWithTools,
} from "@/utils/chatOutputConstraints";
import {
  applyKnowledgeSystemPrompt,
  isExplicitGraphRAGQuestion,
} from "@/utils/chatSystemPrompt";
import { debugLog } from "@/utils/debug";
import { buildGraphRAGReferenceSourceLocations } from "@/utils/graphragReferenceLocations";
import { formatGraphRAGReasoningLocation } from "@/utils/graphragReasoningResults";
import { selectAnswerEvidenceDirectRelationships } from "@/utils/chatReferenceEvidenceRanking";
import { buildChatReferenceContextSummary } from "@/utils/chatReferenceSummary";
import { cleanGraphRAGAssistantAnswer } from "@/utils/graphragAnswerText";
import {
  buildGraphRAGDocumentEvidenceAnswer,
  buildGraphRAGDocumentFallbackAnswer,
  type GraphRAGDocumentFallbackContext,
} from "@/utils/graphragDocumentAnswerFallback";
import {
  cleanGraphRAGTraceSampleText,
  isLowValueGraphRAGTraceSampleText,
} from "@/utils/graphragTraceSamples";
import type {
  GraphRAGTraceCapabilityCard,
  GraphRAGTraceChatContextRequest,
  GraphRAGTraceQuestion,
} from "@/utils/graphragPipelinePrompts";
import {
  buildGraphRAGChatContextPayload,
  normalizeGraphRAGChatContextEndpoint,
  shouldUseUnifiedGraphRAGChatContext,
} from "@/utils/graphragPipelinePrompts";
import {
  PageShell,
  StatusChip,
  type StatusChipConfigMap,
} from "@/components/common";
import { getRoutePath } from "@/config/navigation";
import type {
  ChatMessage,
  ChatCompletionRequest,
  Tool,
  ToolCall,
  ConversationListItem,
} from "@/types/api";
import type {
  DeploymentRouteContext,
  RuntimeRouteState,
} from "@/types/runtimeHandoff";
import { runtimeBackendLabel } from "@/types/runtimeHandoff";
import { accentColors } from "@/utils/theme";

interface ChatPageProps {}

interface GraphRAGChatHandoff {
  runId?: string;
  runIds?: string[];
  filename?: string;
  filenames?: string[];
  domain?: string;
  prompt?: string;
  displayPrompt?: string;
  contextQuery?: string;
  autoSubmit?: boolean;
  chatContextEndpoint?: string;
  chatContextRequest?: GraphRAGTraceChatContextRequest;
  suggestedQuestions?: GraphRAGTraceQuestion[];
  recommendedMode?: string;
  answerGuidance?: string;
  citationScopes?: string[];
  capabilityCards?: GraphRAGTraceCapabilityCard[];
  graphAnchorCount?: number;
  retrievalRehearsalCount?: number;
}

type ChatRouteState = RuntimeRouteState & {
  graphragHandoff?: GraphRAGChatHandoff;
};

type LocalChatMessage = ChatMessage & {
  _clientMessageId?: string;
};

type ChatBusyPhase = "idle" | "preparing_context" | "writing_answer";

const createClientMessageId = (): string => {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `chat-message-${Date.now()}-${Math.random().toString(36).slice(2)}`;
};

const updateAssistantMessageByClientId = (
  messages: ChatMessage[],
  clientMessageId: string,
  updater: (message: LocalChatMessage) => LocalChatMessage,
): ChatMessage[] => {
  const messageIndex = messages.findIndex(
    (message) =>
      (message as LocalChatMessage)._clientMessageId === clientMessageId &&
      message.role === "assistant",
  );
  if (messageIndex < 0) return messages;

  const updatedMessages = [...messages];
  updatedMessages[messageIndex] = updater({
    ...(updatedMessages[messageIndex] as LocalChatMessage),
  });
  return updatedMessages;
};

const getGraphRAGHandoffRunIds = (
  handoff?: GraphRAGChatHandoff | null,
): string[] => {
  const ids = [
    ...(Array.isArray(handoff?.runIds) ? handoff?.runIds || [] : []),
    handoff?.runId || "",
  ];
  const seen = new Set<string>();
  return ids
    .map((id) => String(id || "").trim())
    .filter((id) => {
      if (!id || seen.has(id)) return false;
      seen.add(id);
      return true;
    });
};

const getGraphRAGHandoffFilenames = (
  handoff?: GraphRAGChatHandoff | null,
): string[] =>
  uniqueReferenceStrings([
    ...(handoff?.filenames || []),
    ...referenceStringList((handoff?.chatContextRequest as any)?.filenames),
  ]);

const hasGraphRAGDocumentContextHandoff = (
  handoff?: GraphRAGChatHandoff | null,
): boolean => {
  const request = handoff?.chatContextRequest;
  return Boolean(
    handoff &&
      (getGraphRAGHandoffRunIds(handoff).length > 0 ||
        handoff.chatContextEndpoint ||
        request?.pipeline_run_id ||
        (request?.pipeline_run_ids?.length || 0) > 0 ||
        request?.use_local_pipeline_context ||
        request?.include_local_pipeline),
  );
};

const isRequestedGraphRAGDocumentEvidenceQuestion = (queryText: string): boolean => {
  const text = String(queryText || "").replace(/\s+/g, " ").trim();
  if (!text) return false;
  const asksForEachUploadedSource =
    /\b(?:list\s+each|each|every|all|per[-\s]?(?:file|document|source)|filenames?|file names?)\b/i.test(text) &&
    /\b(?:uploaded\s+)?(?:files?|documents?|sources?|filenames?)\b/i.test(text);
  const asksForEvidence =
    /\b(?:audit|cite|extract|extracted|extraction|inspect|list|review|show)\b/i.test(text) ||
    /\bwhat\b[\s\S]*\bextract(?:ed|ion)?\b/i.test(text);
  const namesRichEvidence =
    /\b(?:attachments?|charts?|comments?|connectors?|embedded files?|embedded objects?|form fields?|forms?|images?|links?|media|speaker notes?|styled text|tables?)\b/i.test(text) ||
    /\b(?:evidence\s+)?(?:families|family|kinds|types)\b/i.test(text);
  return asksForEachUploadedSource || (asksForEvidence && namesRichEvidence);
};

const GRAPH_RAG_AUTOSUBMIT_LOCK_MS = 30_000;

type GraphRAGAutoSubmitWindow = Window & {
  __llamaNexusGraphRAGAutoSubmitLocks?: Map<string, number>;
};

const claimGraphRAGAutoSubmit = (key: string): boolean => {
  if (typeof window === "undefined") return true;

  const now = Date.now();
  const autoSubmitWindow = window as GraphRAGAutoSubmitWindow;
  const locks =
    autoSubmitWindow.__llamaNexusGraphRAGAutoSubmitLocks ?? new Map();
  autoSubmitWindow.__llamaNexusGraphRAGAutoSubmitLocks = locks;

  locks.forEach((timestamp, lockKey) => {
    if (now - timestamp > GRAPH_RAG_AUTOSUBMIT_LOCK_MS) {
      locks.delete(lockKey);
    }
  });

  const previous = locks.get(key);
  if (previous && now - previous < GRAPH_RAG_AUTOSUBMIT_LOCK_MS) {
    return false;
  }

  locks.set(key, now);
  return true;
};

const asReferenceRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};

const referenceRecordList = (value: unknown): Record<string, unknown>[] =>
  Array.isArray(value)
    ? value
        .map(asReferenceRecord)
        .filter((record) => Object.keys(record).length > 0)
    : [];

const asReferenceString = (value: unknown): string =>
  typeof value === "string" ? value.trim() : "";

const referenceStringList = (value: unknown): string[] =>
  Array.isArray(value)
    ? Array.from(
        new Set(
          value
            .map((item) => String(item || "").trim())
            .filter(Boolean),
        ),
      )
    : [];

const uniqueReferenceStrings = (
  values: Array<string | undefined | null>,
): string[] => {
  const seen = new Set<string>();
  const output: string[] = [];

  values.forEach((value) => {
    const normalized = String(value || "").trim();
    const key = normalized.toLowerCase();
    if (!normalized || seen.has(key)) return;
    seen.add(key);
    output.push(normalized);
  });

  return output;
};

const referenceBasename = (value: string): string =>
  value.split(/[\\/]/).filter(Boolean).pop() || value;

const normalizeReferenceKey = (value: unknown): string =>
  referenceBasename(String(value || "").trim()).toLowerCase();

const formatReferenceKind = (value: string): string =>
  value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const PIPELINE_EVIDENCE_FAMILY_LABELS: Record<
  string,
  { singular: string; plural: string }
> = {
  chart_points: { singular: "chart point", plural: "chart points" },
  charts: { singular: "chart", plural: "charts" },
  code_blocks: { singular: "code block", plural: "code blocks" },
  comments: { singular: "comment", plural: "comments" },
  connectors: { singular: "connector", plural: "connectors" },
  diagram_relationships: { singular: "diagram edge", plural: "diagram edges" },
  embedded_files: { singular: "embedded file", plural: "embedded files" },
  forms: { singular: "form field", plural: "form fields" },
  form_fields: { singular: "form field", plural: "form fields" },
  images: { singular: "image", plural: "images" },
  links: { singular: "link", plural: "links" },
  media: { singular: "media item", plural: "media items" },
  notes: { singular: "note", plural: "notes" },
  ocr: { singular: "OCR result", plural: "OCR results" },
  speaker_notes: { singular: "speaker note", plural: "speaker notes" },
  styled_text: { singular: "styled text", plural: "styled text" },
  tables: { singular: "table", plural: "tables" },
};

const asReferenceNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value !== "string") return undefined;
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : undefined;
};

const formatPipelineEvidenceFamilyLabel = (
  family: string,
  count?: number,
): string => {
  const key = family
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const mapped = PIPELINE_EVIDENCE_FAMILY_LABELS[key];
  if (mapped) return count === 1 ? mapped.singular : mapped.plural;
  return family.replace(/[_-]+/g, " ").toLowerCase();
};

const formatAnswerEvidenceRole = (role?: string): string => {
  if (!role) return "";
  if (role === "pipeline_evidence_family") return "Pipeline evidence";
  if (role === "requested_evidence_family") return "Requested evidence";
  return formatReferenceKind(role);
};

const summarizePipelineEvidenceLocation = (
  locationMetadata: Record<string, unknown>,
): string => {
  const family = asReferenceString(
    locationMetadata.evidence_family ||
      locationMetadata.pipeline_evidence_family,
  );
  const count = asReferenceNumber(
    locationMetadata.evidence_count ||
      locationMetadata.pipeline_evidence_count,
  );
  const positiveCount =
    count !== undefined && count > 0 ? count : undefined;
  const familyLabel = family
    ? formatPipelineEvidenceFamilyLabel(family, positiveCount)
    : "";

  if (positiveCount !== undefined && familyLabel) return `${positiveCount} ${familyLabel}`;
  if (familyLabel) return formatReferenceKind(familyLabel);
  if (positiveCount !== undefined) {
    return `${positiveCount} evidence ${positiveCount === 1 ? "item" : "items"}`;
  }
  return "";
};

const isInlinePreviewableArtifact = (
  artifact: Record<string, unknown> | undefined,
): boolean => {
  const contentType = String(artifact?.content_type || "").toLowerCase();
  const filename = String(artifact?.filename || artifact?.endpoint || "").toLowerCase();

  return (
    contentType.startsWith("text/") ||
    contentType.includes("application/pdf") ||
    contentType.includes("application/json") ||
    contentType.includes("application/xml") ||
    contentType.includes("application/xhtml") ||
    /\.(?:txt|md|markdown|pdf|json|csv|html|xml)$/i.test(filename)
  );
};

const referenceArtifactForRun = (
  run: NonNullable<PipelineArtifactContext["runs"]>[number] | undefined,
  preferredKinds: string[] = ["source_document", "structured_ingest"],
) => {
  const artifacts = Array.isArray(run?.artifacts) ? run?.artifacts || [] : [];

  for (const kind of preferredKinds) {
    const match = artifacts.find(
      (artifact) =>
        String(artifact.kind || "").toLowerCase() === kind &&
        artifact.endpoint &&
        isInlinePreviewableArtifact(artifact as Record<string, unknown>),
    );
    if (match) return match;
  }

  const structured = artifacts.find(
    (artifact) =>
      String(artifact.kind || "").toLowerCase() === "structured_ingest" &&
      artifact.endpoint,
  );
  if (structured) return structured;

  return artifacts.find((artifact) => artifact.endpoint);
};

const findReferenceRun = (
  identifiers: unknown[],
  pipelineArtifacts?: PipelineArtifactContext | null,
) => {
  const runs = pipelineArtifacts?.runs || [];
  if (!runs.length) return undefined;

  const keys = identifiers
    .map(normalizeReferenceKey)
    .filter(Boolean);

  if (keys.length > 0) {
    const exact = runs.find((run) => {
      const runKeys = [
        run.run_id,
        run.filename,
        run.structured_ingest?.filename,
      ]
        .map(normalizeReferenceKey)
        .filter(Boolean);
      return runKeys.some((runKey) => keys.includes(runKey));
    });
    if (exact) return exact;

    const fuzzy = runs.find((run) => {
      const runKeys = [
        run.run_id,
        run.filename,
        run.structured_ingest?.filename,
      ]
        .map(normalizeReferenceKey)
        .filter(Boolean);
      return runKeys.some((runKey) =>
        keys.some((key) => key.includes(runKey) || runKey.includes(key)),
      );
    });
    if (fuzzy) return fuzzy;
  }

  return runs.length === 1 ? runs[0] : undefined;
};

const chunkReferenceTerms = (
  chunk: Partial<UnifiedChunk> & Partial<RAGChunk>,
): string[] => {
  const metadata = asReferenceRecord(chunk.metadata);
  return uniqueReferenceStrings([
    ...(chunk.matched_entities || []),
    ...(chunk.matched_graph_path_terms || []),
    ...referenceStringList(metadata.matched_local_pipeline_evidence_terms),
    ...referenceStringList(metadata.matched_local_chunk_evidence_terms),
    ...referenceStringList(metadata.local_chunk_evidence_terms),
    ...referenceStringList(metadata.evidence_families),
    ...referenceStringList(metadata.location_hints),
    ...referenceStringList(metadata.retrieval_queries),
    asReferenceString(metadata.primary_evidence_family),
    asReferenceString(metadata.primary_location_hint),
    asReferenceString(metadata.filename),
    asReferenceString(metadata.local_graph_filename),
  ]);
};

type ReferenceRelationshipLike = Partial<UnifiedRelationship> & {
  specific_matched_query_terms?: unknown;
  relationship_type?: unknown;
  evidence?: unknown;
  rationale?: unknown;
  provenance_summary?: unknown;
  evidence_families?: unknown;
  weight?: unknown;
};

const relationshipReferenceTerms = (
  relationship: ReferenceRelationshipLike,
): string[] => {
  const metadata = asReferenceRecord(relationship.metadata);
  return uniqueReferenceStrings([
    relationship.source,
    relationship.target,
    relationship.relation,
    ...(relationship.matched_query_terms || []),
    ...referenceStringList(relationship.specific_matched_query_terms),
    asReferenceString(relationship.relationship_type),
    asReferenceString(relationship.evidence),
    asReferenceString(relationship.rationale),
    asReferenceString(relationship.provenance),
    asReferenceString(relationship.provenance_summary),
    ...referenceStringList(relationship.provenance_summary),
    ...referenceStringList(relationship.evidence_families),
    ...referenceStringList(metadata.evidence_families),
    ...referenceStringList(metadata.child_labels),
    ...referenceStringList(metadata.location_hints),
    ...referenceStringList(metadata.provenance_summary),
    asReferenceString(metadata.primary_evidence_family),
    asReferenceString(metadata.relationship_type),
    asReferenceString(metadata.element_type),
    asReferenceString(metadata.shape_type),
    asReferenceString(metadata.shape_label),
    asReferenceString(metadata.group_label),
    asReferenceString(metadata.parent_group),
    asReferenceString(metadata.parent_group_label),
    asReferenceString(metadata.provenance_summary),
    asReferenceString(metadata.evidence),
    asReferenceString(metadata.local_graph_filename),
    asReferenceString(metadata.filename),
  ]);
};

const pathReferenceTerms = (
  path: NonNullable<GraphReasoning["paths"]>[number],
): string[] =>
  uniqueReferenceStrings([
    path.source,
    path.target,
    ...(path.nodes || []),
    ...(path.matched_query_terms || []),
    ...(path.evidence_families || []),
    ...(path.provenance_summary || []),
    ...(path.relationships || []).flatMap((relationship) => [
      relationship.source,
      relationship.relation,
      relationship.target,
      relationship.evidence,
      relationship.provenance,
      relationship.provenance_summary,
      ...referenceStringList(relationship.evidence_families),
    ]),
  ]);

const cleanReferencePreview = (value: string, limit = 260): string => {
  const normalized = value
    .replace(/^#\s*Structured GraphRAG Ingest:\s*/i, "")
    .replace(/^##\s*Source\s*/i, "")
    .replace(/^-\s*(?:File|Filename|Content type|Extraction method|Domain|SHA256|Hash):\s*.*$/gim, "")
    .replace(/^##\s*Extracted Structure\s*/i, "")
    .replace(/^\s*Scope:\s*.*?\s+Evidence focus:\s*/i, "")
    .replace(/\bline\s*\d+\s*:\s*/gi, "")
    .replace(
      /^\s*(?:\d+\.\s*)?\[(?:paragraph|heading|list|code block|code|table|link|image|section)[^\]]*\]\s*/gim,
      "",
    )
    .replace(
      /\[(?:paragraph|heading|list|code block|code|table|link|image|section)[^\]]*\]\s*line\s*\d+\s*:\s*/gi,
      "",
    )
    .replace(
      /\s*(?:\d+\.\s*)?\[(?:paragraph|heading|list|code block|code|table|link|image|section)[^\]]*\]\s*/gi,
      " ",
    )
    .replace(/^\s*(?:\d+\.\s*)?\[[a-z][a-z0-9_ -]*(?:\s*\|[^\]]*)?\]\s*/gim, "")
    .replace(/\s*\(level=\d+\)/gi, "")
    .replace(
      /\s*--\[\s*([^\]]+?)\s*\]--?>\s*/g,
      (_match, relation: string) => ` ${formatReferenceKind(relation).toLowerCase()} `,
    )
    .replace(
      /\s*--\[\s*([^\]]+?)\s*\]--\s*/g,
      (_match, relation: string) => ` ${formatReferenceKind(relation).toLowerCase()} `,
    )
    .replace(/\s*\[[a-z][a-z_ -]*\]\s*-->\s*[A-Za-z0-9_./:-]+/g, "")
    .replace(/\s+/g, " ")
    .trim();
  return cleanGraphRAGTraceSampleText(normalized, limit);
};

const structuredIngestExcerptForRun = (
  run: NonNullable<PipelineArtifactContext["runs"]>[number] | undefined,
  limit = 300,
): string => {
  const structuredIngest = asReferenceRecord(run?.structured_ingest);
  const structuredArtifact = asReferenceRecord(
    (run as Record<string, unknown> | undefined)?.structured_ingest_artifact,
  );
  const rawExcerpt =
    asReferenceString(structuredIngest.excerpt) ||
    asReferenceString(structuredArtifact.excerpt) ||
    asReferenceString(structuredArtifact.content);
  if (!rawExcerpt) return "";

  const warningPreview =
    rawExcerpt.match(/\bWarning:\s*[^=\n\r]{20,420}/i)?.[0] || "";
  return cleanReferencePreview(
    warningPreview || rawExcerpt,
    warningPreview ? Math.max(limit, 420) : limit,
  );
};

const isLowValueReferenceRelationship = (
  source: unknown,
  relation: unknown,
  target: unknown,
): boolean => {
  const sourceText = asReferenceString(source) || "Source";
  const relationText = asReferenceString(relation) || "related";
  const targetText = asReferenceString(target) || "Target";
  return (
    isLowValueGraphRAGTraceSampleText(
      `${sourceText} ${relationText} ${targetText}`,
    ) ||
    isLowValueGraphRAGTraceSampleText(`${sourceText} -> ${targetText}`)
  );
};

const WEAK_REFERENCE_RELATION_PATTERN =
  /^(?:appears?_with|co_occurs(?:_with)?|mentioned?_with|mentions?|related(?:_to|_text)?|contains(?:_text)?)$/i;

const isWeakReferenceRelation = (relation: unknown): boolean =>
  WEAK_REFERENCE_RELATION_PATTERN.test(
    String(relation || "")
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, ""),
  );

const buildChatReferenceItems = ({
  unifiedChunks,
  ragContext,
  relationships,
  graphReasoning,
  pipelineArtifacts,
}: {
  unifiedChunks: UnifiedChunk[];
  ragContext: RAGChunk[];
  relationships: UnifiedRelationship[];
  graphReasoning?: GraphReasoning | null;
  pipelineArtifacts?: PipelineArtifactContext | null;
}): ChatReferenceItem[] => {
  const references: ChatReferenceItem[] = [];
  const seen = new Set<string>();
  const answerPlanEvidenceCards = Array.isArray(
    graphReasoning?.answer_plan?.evidence_cards,
  )
    ? graphReasoning?.answer_plan?.evidence_cards || []
    : [];
  const chunks: Array<Partial<UnifiedChunk> & Partial<RAGChunk>> =
    unifiedChunks.length > 0 ? unifiedChunks : ragContext;

  chunks.slice(0, 8).forEach((chunk, index) => {
    const metadata = asReferenceRecord(chunk.metadata);
    const evidenceFamilies = uniqueReferenceStrings([
      asReferenceString(metadata.primary_evidence_family),
      ...referenceStringList(metadata.evidence_families),
    ]);
    const locationHints = uniqueReferenceStrings([
      asReferenceString(metadata.primary_location_hint),
      ...referenceStringList(metadata.location_hints),
      ...referenceStringList(metadata.provenance),
    ]);
    const matchedTerms = uniqueReferenceStrings([
      ...referenceStringList(metadata.matched_local_chunk_evidence_terms),
      ...referenceStringList(metadata.matched_local_pipeline_evidence_terms),
      ...referenceStringList(metadata.matched_local_graph_terms),
    ]);
    const run = findReferenceRun(
      [
        chunk.document_id,
        chunk.document_title,
        metadata.run_id,
        metadata.local_graph_run_id,
        metadata.filename,
        metadata.local_graph_filename,
      ],
      pipelineArtifacts,
    );
    const artifact = referenceArtifactForRun(run);
    const id =
      chunk.id ||
      `chunk-${chunk.document_id || "document"}-${chunk.chunk_index ?? index}`;
    const key = `passage:${id}`;
    if (seen.has(key)) return;
    seen.add(key);
    const body = cleanReferencePreview(
      asReferenceString(metadata.semantic_text) || chunk.content || "",
      300,
    );
    if (!body || isLowValueGraphRAGTraceSampleText(body)) return;

    references.push({
      id: key,
      kind: "passage",
      sourceRole: "retrieved_passage",
      title:
        chunk.document_title ||
        asReferenceString(metadata.filename) ||
        asReferenceString(metadata.local_graph_filename) ||
        run?.filename ||
        `Passage ${index + 1}`,
      subtitle: [
        chunk.chunk_index !== undefined ? `Chunk ${chunk.chunk_index}` : "",
        typeof chunk.score === "number" ? `${Math.round(chunk.score * 100)}% match` : "",
      ]
        .filter(Boolean)
        .join(" · "),
      body,
      score: chunk.score,
      documentId: chunk.document_id,
      chunkIndex: chunk.chunk_index,
      endpoint: artifact?.endpoint,
      contentType: artifact?.content_type,
      filename: run?.filename || run?.structured_ingest?.filename || artifact?.filename,
      runId: run?.run_id,
      terms: chunkReferenceTerms(chunk),
      evidenceFamilies,
      locationHints,
      sourceLocations: buildGraphRAGReferenceSourceLocations(
        metadata,
        ...referenceRecordList(metadata.source_locations),
      ),
      matchedTerms,
    });
  });

  (graphReasoning?.paths || []).slice(0, 6).forEach((path, index) => {
    const run = findReferenceRun(
      [
        ...(path.relationships || []).flatMap((relationship) => {
          const metadata = asReferenceRecord(relationship.metadata);
          return [
            metadata.local_graph_run_id,
            metadata.run_id,
            metadata.local_graph_filename,
            metadata.filename,
          ];
        }),
        ...(path.provenance_summary || []),
      ],
      pipelineArtifacts,
    );
    const artifact = referenceArtifactForRun(run);
    const nodes = path.nodes?.length
      ? path.nodes
      : [path.source, path.target].filter(Boolean);
    const pathTitle =
      nodes.length > 1
        ? nodes.join(" -> ")
        : path.text || `${path.source || "Source"} -> ${path.target || "Target"}`;
    const key = `path:${pathTitle}:${index}`;
    if (seen.has(key)) return;
    seen.add(key);

    const evidenceFamilies = uniqueReferenceStrings([
      ...(path.evidence_families || []),
      ...(path.relationships || []).flatMap((relationship) =>
        referenceStringList(relationship.evidence_families),
      ),
    ]);
    const locationHints = uniqueReferenceStrings([
      ...(path.provenance_summary || []),
      ...(path.relationships || []).flatMap((relationship) => [
        relationship.provenance,
        relationship.provenance_summary,
      ]),
    ]);
    const sourceLocations = buildGraphRAGReferenceSourceLocations(
      ...(path.relationships || []).map((relationship) =>
        asReferenceRecord(relationship.metadata),
      ),
    );
    const relationshipHops = (path.relationships || []).map((relationship) => ({
      source: relationship.source,
      relation: formatReferenceKind(relationship.relation || "related_to"),
      target: relationship.target,
      evidence: cleanReferencePreview(
        relationship.evidence ||
          relationship.rationale ||
          relationship.provenance ||
          relationship.provenance_summary ||
          "",
      ),
    }));
    const body = [
      path.explanation,
      path.answer_hint,
      path.text,
      ...relationshipHops.map((relationship) =>
        [
          relationship.source,
          relationship.relation,
          relationship.target,
          relationship.evidence,
        ]
          .filter(Boolean)
          .join(" "),
      ),
    ]
      .filter(Boolean)
      .join("\n");

    references.push({
      id: key,
      kind: "path",
      sourceRole: "reasoning_path",
      title: pathTitle,
      subtitle: [
        path.hops ? `${path.hops} hop${path.hops === 1 ? "" : "s"}` : "",
        path.path_type ? formatReferenceKind(path.path_type) : "",
        typeof path.score === "number" ? `${Math.round(path.score * 100)}% score` : "",
      ]
        .filter(Boolean)
        .join(" · "),
      body: cleanReferencePreview(body),
      score: path.score,
      endpoint: artifact?.endpoint,
      contentType: artifact?.content_type,
      filename: run?.filename || run?.structured_ingest?.filename || artifact?.filename,
      runId: run?.run_id,
      terms: pathReferenceTerms(path),
      evidenceFamilies,
      locationHints,
      sourceLocations,
      matchedTerms: uniqueReferenceStrings(path.matched_query_terms || []),
      path: {
        nodes,
        hops: path.hops,
        score: path.score,
        relationships: relationshipHops,
      },
    });
  });

  selectAnswerEvidenceDirectRelationships(
    graphReasoning?.direct_relationships || [],
    answerPlanEvidenceCards,
    10,
  )
    .forEach((relationship, index) => {
      const directRelationship =
        relationship as ReferenceRelationshipLike;
      const metadata = asReferenceRecord(directRelationship.metadata);
      const relation =
        asReferenceString(directRelationship.relation) ||
        asReferenceString(directRelationship.relationship_type) ||
        "related_to";
      const relationLabel = formatReferenceKind(relation);
      const source = asReferenceString(directRelationship.source) || "Source";
      const target = asReferenceString(directRelationship.target) || "Target";
      if (isLowValueReferenceRelationship(source, relation || relationLabel, target)) return;
      const key = `direct-relationship:${source}:${relation}:${target}:${index}`;
      if (seen.has(key)) return;
      seen.add(key);

      const run = findReferenceRun(
        [
          metadata.local_graph_run_id,
          metadata.run_id,
          metadata.local_graph_filename,
          metadata.filename,
          directRelationship.provenance,
          directRelationship.provenance_summary,
        ],
        pipelineArtifacts,
      );
      const artifact = referenceArtifactForRun(run);
      const sourceLocations = buildGraphRAGReferenceSourceLocations(
        metadata,
        ...referenceRecordList(metadata.source_locations),
      );
      const evidenceBundleSummary =
        summarizePipelineEvidenceLocation(metadata);
      const body = [
        directRelationship.evidence,
        directRelationship.rationale,
        directRelationship.provenance_summary,
        directRelationship.provenance,
        asReferenceString(metadata.provenance_summary),
        asReferenceString(metadata.evidence),
        `${source} ${relationLabel} ${target}`,
      ]
        .filter(Boolean)
        .join("\n");
      const previewBody = cleanReferencePreview(body, 300);
      if (!previewBody || isLowValueGraphRAGTraceSampleText(previewBody)) return;
      const evidenceFamilies = uniqueReferenceStrings([
        relation,
        asReferenceString(directRelationship.relationship_type),
        ...referenceStringList(directRelationship.evidence_families),
        ...referenceStringList(metadata.evidence_families),
        asReferenceString(metadata.primary_evidence_family),
        asReferenceString(metadata.relationship_type),
        asReferenceString(metadata.element_type),
        asReferenceString(metadata.shape_type),
        asReferenceString(metadata.group_label) ? "shape_group" : "",
      ]);

      references.push({
        id: key,
        kind: "relationship",
        sourceRole: "answer_evidence",
        title: `${source} -> ${target}`,
        citationLabel: evidenceBundleSummary || relationLabel,
        subtitle: [
          relationLabel,
          evidenceBundleSummary,
          run?.filename ||
            asReferenceString(metadata.local_graph_filename) ||
            asReferenceString(metadata.filename),
        ]
          .filter(Boolean)
          .join(" · "),
        body: previewBody,
        score:
          typeof directRelationship.score === "number"
            ? directRelationship.score
            : typeof directRelationship.weight === "number"
              ? directRelationship.weight
              : undefined,
        endpoint: artifact?.endpoint,
        contentType: artifact?.content_type,
        filename:
          run?.filename ||
          run?.structured_ingest?.filename ||
          artifact?.filename ||
          asReferenceString(metadata.local_graph_filename) ||
          asReferenceString(metadata.filename),
        runId: run?.run_id || asReferenceString(metadata.run_id),
        terms: relationshipReferenceTerms(directRelationship),
        evidenceFamilies,
        locationHints: uniqueReferenceStrings([
          evidenceBundleSummary,
          directRelationship.provenance,
          asReferenceString(directRelationship.provenance_summary),
          ...referenceStringList(directRelationship.provenance_summary),
          asReferenceString(metadata.provenance_summary),
          ...referenceStringList(metadata.location_hints),
        ]),
        sourceLocations,
        matchedTerms: uniqueReferenceStrings([
          source,
          target,
          relation,
          ...(directRelationship.matched_query_terms || []),
          ...referenceStringList(directRelationship.specific_matched_query_terms),
        ]),
        relationship: {
          source,
          relation: relationLabel,
          target,
        },
      });
    });

  answerPlanEvidenceCards
    .slice(0, 12)
    .forEach((card, index) => {
      const locationMetadata = asReferenceRecord(card.location);
      const locationLabel = formatGraphRAGReasoningLocation(locationMetadata);
      const run = findReferenceRun(
        [
          locationMetadata.local_graph_run_id,
          locationMetadata.run_id,
          locationMetadata.local_graph_filename,
          locationMetadata.filename,
          card.citation_label,
          card.provenance,
          card.provenance_summary,
        ],
        pipelineArtifacts,
      );
      const artifact = referenceArtifactForRun(run);
      const evidenceBundleSummary =
        summarizePipelineEvidenceLocation(locationMetadata);
      const roleLabel = formatAnswerEvidenceRole(card.role);
      const isRequestedEvidenceCard = card.role === "requested_evidence_family";
      const requestedEvidenceTitle =
        isRequestedEvidenceCard
          ? asReferenceString(card.title) ||
            formatReferenceKind(asReferenceString(card.kind) || "Evidence")
          : "";
      const relationLabel = formatReferenceKind(
        card.source || card.target
          ? card.relation || card.relationship_type || card.role || "answer_evidence"
          : card.relationship_type || card.relation || card.role || "answer_evidence",
      );
      const weakRelationWithEvidence =
        !isRequestedEvidenceCard &&
        isWeakReferenceRelation(card.relation || card.relationship_type || relationLabel) &&
        Boolean(card.evidence);
      const evidenceTitle = weakRelationWithEvidence
        ? cleanReferencePreview(asReferenceString(card.evidence), 180)
        : "";
      const cardIdentity =
        asReferenceString(card.id) ||
        asReferenceString(card.citation_label) ||
        `${card.index ?? index}`;
      const key = `answer-card:${cardIdentity}:${index}`;
      if (seen.has(key)) return;
      seen.add(key);

      const relationSentence =
        card.source || card.target
          ? `${card.source || "Source"} ${relationLabel} ${card.target || "Target"}`
          : "";
      if (
        relationSentence &&
        isLowValueReferenceRelationship(
          card.source,
          card.relation || card.relationship_type || relationLabel,
          card.target,
        )
      ) {
        return;
      }
      const relationTitle =
        card.source || card.target
          ? `${card.source || "Source"} -> ${card.target || "Target"}`
          : "";
      const body = (
        isRequestedEvidenceCard
          ? [card.evidence || card.text]
          : weakRelationWithEvidence
            ? [card.evidence]
          : [
              card.text,
              card.evidence,
              card.provenance_summary,
              card.provenance,
              relationSentence,
            ]
      )
        .filter(Boolean)
        .join("\n");
      const previewBody = cleanReferencePreview(body, 300);
      if (!previewBody || isLowValueGraphRAGTraceSampleText(previewBody)) return;
      const kind: ChatReferenceItem["kind"] =
        !isRequestedEvidenceCard && (card.source || card.target)
          ? "relationship"
          : "passage";

      references.push({
        id: key,
        kind,
        sourceRole: "answer_evidence",
        title:
          requestedEvidenceTitle ||
          evidenceTitle ||
          relationTitle ||
          evidenceBundleSummary ||
          cleanReferencePreview(card.citation_label || "") ||
          relationLabel ||
          `Evidence ${index + 1}`,
        citationLabel: card.citation_label || evidenceBundleSummary || locationLabel,
        subtitle: [
          roleLabel,
          evidenceBundleSummary && evidenceBundleSummary !== card.citation_label
            ? evidenceBundleSummary
            : "",
          locationLabel && locationLabel !== evidenceBundleSummary
            ? locationLabel
            : "",
          run?.filename,
        ]
          .filter(Boolean)
          .join(" · "),
        body: previewBody,
        endpoint: artifact?.endpoint,
        contentType: artifact?.content_type,
        filename:
          run?.filename ||
          run?.structured_ingest?.filename ||
          artifact?.filename ||
          asReferenceString(locationMetadata.local_graph_filename) ||
          asReferenceString(locationMetadata.filename),
        runId: run?.run_id || asReferenceString(locationMetadata.run_id),
        terms: uniqueReferenceStrings([
          isRequestedEvidenceCard ? "" : card.source,
          isRequestedEvidenceCard ? "" : card.target,
          isRequestedEvidenceCard ? "" : card.relation,
          isRequestedEvidenceCard ? "" : card.relationship_type,
          card.kind,
          card.citation_label,
          evidenceBundleSummary,
          locationLabel,
          ...referenceStringList(card.evidence_families),
        ]),
        evidenceFamilies: isRequestedEvidenceCard
          ? uniqueReferenceStrings([
              card.role,
              card.kind,
            ])
          : uniqueReferenceStrings([
              card.role,
              card.kind,
              card.relationship_type,
              ...referenceStringList(card.evidence_families),
            ]),
        locationHints: uniqueReferenceStrings([
          locationLabel,
          card.provenance,
          card.provenance_summary,
        ]),
        sourceLocations: buildGraphRAGReferenceSourceLocations(locationMetadata),
        matchedTerms: isRequestedEvidenceCard
          ? uniqueReferenceStrings([card.kind, requestedEvidenceTitle])
          : uniqueReferenceStrings([
              card.source,
              card.target,
              card.relation,
            ]),
        relationship: !isRequestedEvidenceCard && (card.source || card.target)
          ? {
              source: card.source,
              relation: relationLabel,
              target: card.target,
            }
          : undefined,
      });
    });

  (pipelineArtifacts?.runs || []).slice(0, 6).forEach((run) => {
    const artifact = referenceArtifactForRun(run);
    (run.evidence_cards || []).slice(0, 4).forEach((card, index) => {
      const metadata = asReferenceRecord(card.metadata);
      const sourceLocationRecords = referenceRecordList(card.source_locations);
      const cardIdentity =
        asReferenceString(card.chunk_id) ||
        asReferenceString(card.title) ||
        `${index}`;
      const key = `run-evidence:${run.run_id || run.filename || "run"}:${cardIdentity}:${index}`;
      if (seen.has(key)) return;
      seen.add(key);

      const sourceLocations = buildGraphRAGReferenceSourceLocations(
        metadata,
        ...sourceLocationRecords,
      );
      const evidenceFamilies = uniqueReferenceStrings([
        card.category,
        ...(card.categories || []),
        ...(card.evidence_families || []),
      ]);
      const locationHints = uniqueReferenceStrings([
        card.scope_label,
        ...(card.location_hints || []),
        ...sourceLocations.map((location) => `${location.label}: ${location.value}`),
      ]);
      const representativeDocumentChunk =
        /representative document chunk/i.test(asReferenceString(card.reason)) ||
        (
          asReferenceString(card.title).toLowerCase() === "document" &&
          asReferenceString(card.category).toLowerCase() === "document"
        );
      const previewSource =
        card.evidence_preview ||
        (representativeDocumentChunk ? structuredIngestExcerptForRun(run, 420) : "") ||
        card.reason ||
        "";
      const representativeWarningPreview =
        representativeDocumentChunk
          ? asReferenceString(previewSource).match(
              /\bWarning:\s*[^=\n\r]{20,420}/i,
            )?.[0]
          : "";
      const previewBody = cleanReferencePreview(
        representativeWarningPreview || previewSource,
        representativeDocumentChunk ? 420 : 300,
      );
      if (!previewBody || isLowValueGraphRAGTraceSampleText(previewBody)) return;

      references.push({
        id: key,
        kind: "passage",
        sourceRole: "source_document",
        title: card.title || run.filename || `Source ${index + 1}`,
        citationLabel: card.scope_label || card.title,
        subtitle: [
          card.reason,
          card.scope_label && card.scope_label !== card.title ? card.scope_label : "",
          run.filename,
        ]
          .filter(Boolean)
          .join(" · "),
        body: previewBody,
        endpoint: artifact?.endpoint,
        contentType: artifact?.content_type,
        filename: run.filename || run.structured_ingest?.filename || artifact?.filename,
        runId: run.run_id,
        terms: uniqueReferenceStrings([
          run.filename,
          card.title,
          card.reason,
          card.scope_label,
          ...(card.categories || []),
          ...(card.evidence_families || []),
          ...(card.location_hints || []),
        ]),
        evidenceFamilies,
        locationHints,
        sourceLocations,
        matchedTerms: uniqueReferenceStrings([
          card.title,
          card.scope_label,
          card.category,
        ]),
      });
    });
  });

  (pipelineArtifacts?.runs || []).slice(0, 16).forEach((run) => {
    const artifact = referenceArtifactForRun(run);
    if (!artifact?.endpoint) return;

    const key = `document:${run.run_id || artifact.endpoint}`;
    if (seen.has(key)) return;
    seen.add(key);
    const body = structuredIngestExcerptForRun(run, 300);
    if (body && isLowValueGraphRAGTraceSampleText(body)) return;

    references.push({
      id: key,
      kind: "document",
      sourceRole: "source_document",
      title: run.filename || run.structured_ingest?.filename || "Source document",
      subtitle: [
        run.status ? formatReferenceKind(run.status) : "",
        run.structured_ingest?.char_count
          ? `${run.structured_ingest.char_count.toLocaleString()} chars`
          : "",
      ]
        .filter(Boolean)
        .join(" · "),
      body,
      endpoint: artifact.endpoint,
      contentType: artifact.content_type,
      filename: run.filename || run.structured_ingest?.filename || artifact.filename,
      runId: run.run_id,
      terms: uniqueReferenceStrings([run.filename, run.structured_ingest?.filename]),
      sourceLocations: buildGraphRAGReferenceSourceLocations(
        asReferenceRecord(run.structured_ingest),
      ),
    });
  });

  relationships.slice(0, 6).forEach((relationship, index) => {
    const metadata = asReferenceRecord(relationship.metadata);
    const run = findReferenceRun(
      [
        metadata.local_graph_run_id,
        metadata.run_id,
        metadata.local_graph_filename,
        metadata.filename,
      ],
      pipelineArtifacts,
    );
    const artifact = referenceArtifactForRun(run);
    const key = `relationship:${relationship.source || "source"}:${relationship.relation || "related"}:${relationship.target || "target"}:${index}`;
    if (seen.has(key)) return;
    seen.add(key);

    const relationLabel = formatReferenceKind(relationship.relation || "related_to");
    if (
      isLowValueReferenceRelationship(
        relationship.source,
        relationship.relation || relationLabel,
        relationship.target,
      )
    ) {
      return;
    }
    const body = [
      relationship.provenance,
      asReferenceString(metadata.provenance_summary),
      asReferenceString(metadata.evidence),
      `${relationship.source || "Source"} ${relationLabel} ${relationship.target || "Target"}`,
    ]
      .filter(Boolean)
      .join("\n");
    const previewBody = cleanReferencePreview(body, 300);
    if (!previewBody || isLowValueGraphRAGTraceSampleText(previewBody)) return;

    references.push({
      id: key,
      kind: "relationship",
      sourceRole: "graph_relationship",
      title: relationLabel,
      subtitle:
        asReferenceString(metadata.local_graph_filename) ||
        asReferenceString(metadata.filename) ||
        run?.filename,
      body: previewBody,
      score: relationship.score,
      endpoint: artifact?.endpoint,
      contentType: artifact?.content_type,
      filename: run?.filename || run?.structured_ingest?.filename || artifact?.filename,
      runId: run?.run_id,
      terms: relationshipReferenceTerms(relationship),
      evidenceFamilies: uniqueReferenceStrings([
        relationship.relation,
        ...referenceStringList(metadata.evidence_families),
        asReferenceString(metadata.primary_evidence_family),
        asReferenceString(metadata.element_type),
        asReferenceString(metadata.shape_type),
      ]),
      sourceLocations: buildGraphRAGReferenceSourceLocations(metadata),
      relationship: {
        source: relationship.source,
        relation: relationLabel,
        target: relationship.target,
      },
    });
  });

  return references;
};

const formatGraphRAGLabel = (value: string): string =>
  value
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());

const compactGraphRAGActionText = (value: string, limit = 52): string => {
  const text = value
    .replace(/\s*:\s*[a-z][\w -]{1,48}$/i, "")
    .replace(/\s+/g, " ")
    .trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(1, limit - 3)).trim()}...`;
};

const isDocumentFilenameLabel = (value: string): boolean =>
  /\.(?:csv|docx?|html?|md|pdf|pptx?|txt|xlsx?)$/i.test(value.trim());

const isExternalGraphEndpoint = (value: string): boolean =>
  /\b(?:https?:\/\/|www\.)\S+/i.test(value.trim());

const isGraphLinkRelation = (value: string): boolean =>
  /^(?:links?_to|contains?_link|has_link|references?_url)$/i.test(
    value.trim().replace(/[^a-z0-9]+/gi, "_").replace(/^_+|_+$/g, ""),
  );

const graphRAGQuestionLabel = (question: GraphRAGTraceQuestion): string => {
  const source = question.source?.trim() || "";
  const target = question.target?.trim() || "";
  const bridge = question.bridge?.trim() || "";
  const relation = question.relation?.trim() || "";
  const label = question.label?.trim() || "Ask document";
  const linkSubject = [target, source, label]
    .map((value) => value.replace(/\s+--?\[.*?\]--?>\s+/g, " -> ").trim())
    .find(
      (value) =>
        value &&
        !value.includes("->") &&
        !isExternalGraphEndpoint(value) &&
        !isDocumentFilenameLabel(value),
    );

  if (question.kind === "path" && source && target) {
    return bridge
      ? `Trace ${compactGraphRAGActionText(source, 24)} via ${compactGraphRAGActionText(bridge, 24)}`
      : `Trace ${compactGraphRAGActionText(source, 24)} -> ${compactGraphRAGActionText(target, 24)}`;
  }

  if (question.kind === "relationship" && (source || target)) {
    if (isGraphLinkRelation(relation)) {
      return linkSubject
        ? `Trace ${compactGraphRAGActionText(linkSubject, 30)} link`
        : "Trace evidence link";
    }
    if (target.includes("->")) {
      return `Trace ${compactGraphRAGActionText(target)}`;
    }
    if (source && target) {
      return isDocumentFilenameLabel(source)
        ? `Trace ${compactGraphRAGActionText(target)}`
        : `Trace ${compactGraphRAGActionText(source, 24)} -> ${compactGraphRAGActionText(target, 24)}`;
    }
    return `Trace ${compactGraphRAGActionText(source || target)}`;
  }

  return compactGraphRAGActionText(label);
};

const CHAT_STATUS_CONFIG: StatusChipConfigMap = {
  workbench: { color: accentColors.purple, label: "Conversation workbench" },
  unsaved: { color: accentColors.warning, label: "Unsaved" },
  saved: { color: accentColors.success, label: "Saved" },
  draft: { color: accentColors.slate, label: "Draft" },
  reasoning: { color: accentColors.info, label: "Reasoning" },
  structured: { color: accentColors.info, label: "Structured output" },
  pipeline: { color: accentColors.purple, label: "Pipeline trace" },
  tools: { color: accentColors.warning, label: "Tools" },
  runtime: { color: accentColors.success, label: "Runtime loop" },
};

const chatActionButtonSx = {
  width: 30,
  height: 30,
  p: 0,
  borderRadius: "8px",
  border: "1px solid",
  borderColor: "divider",
  bgcolor: "action.hover",
  "&:hover": {
    bgcolor: "action.selected",
  },
};

const MarkdownRenderer = lazy(() =>
  import("@/components/chat/MarkdownRenderer").then((module) => ({
    default: module.MarkdownRenderer,
  })),
);
const RAGSettingsPanel = lazy(() =>
  import("@/components/chat/RAGSettingsPanel").then((module) => ({
    default: module.RAGSettingsPanel,
  })),
);
const RAGContextBlock = lazy(() =>
  import("@/components/chat/RAGContextBlock").then((module) => ({
    default: module.RAGContextBlock,
  })),
);
const GraphContextBlock = lazy(() =>
  import("@/components/chat/GraphContextBlock").then((module) => ({
    default: module.GraphContextBlock,
  })),
);
const EntityExplorer = lazy(() =>
  import("@/components/chat/EntityExplorer").then((module) => ({
    default: module.EntityExplorer,
  })),
);
const ConversationEntities = lazy(() =>
  import("@/components/chat/ConversationEntities").then((module) => ({
    default: module.ConversationEntities,
  })),
);
const UnifiedContextBlock = lazy(() =>
  import("@/components/chat/UnifiedContextBlock").then((module) => ({
    default: module.UnifiedContextBlock,
  })),
);
const ReferenceExplorer = lazy(() =>
  import("@/components/chat/ReferenceExplorer").then((module) => ({
    default: module.ReferenceExplorer,
  })),
);
const RetrievalFeedbackBar = lazy(() =>
  import("@/components/chat/RetrievalFeedbackBar").then((module) => ({
    default: module.RetrievalFeedbackBar,
  })),
);
const ConstraintEditor = lazy(() =>
  import("@/components/chat/ConstraintEditor").then((module) => ({
    default: module.ConstraintEditor,
  })),
);
const ConversationSidebar = lazy(() =>
  import("@/components/chat/ConversationSidebar").then((module) => ({
    default: module.ConversationSidebar,
  })),
);

interface ChatSettings {
  // Connection settings
  baseUrl: string;
  endpoint: string;
  apiKey: string;
  openaiApiKey: string; // For audio transcription (legacy, now uses local STT)
  model: string;
  // Model parameters
  temperature: number;
  topP: number;
  topK: number;
  maxTokens: number;
  streamResponse: boolean;
  enableTools: boolean;
  selectedTools: string[];
  // Reasoning settings
  reasoningLevel: "low" | "medium" | "high" | "none";
  // Display settings
  showPerformanceMetrics: boolean;
  // Voice settings
  voiceModeEnabled: boolean;
  ttsEnabled: boolean;
  ttsAutoPlay: boolean;
  ttsVoice: string;
  ttsSpeed: number;
  sttModel: string;
  sttLanguage: string;
  vadSilenceThreshold: number;
  vadSilenceDuration: number;
  noSpeechThreshold: number; // Max no_speech_prob to accept (0-1)
  // RAG settings
  ragEnabled: boolean;
  ragSearchMode: "global" | "domains";
  ragSelectedDomains: string[]; // domain IDs when mode is 'domains'
  ragTopK: number; // number of chunks to retrieve
  ragShowContext: boolean; // show retrieved context in UI
  // GraphRAG settings
  graphEnabled: boolean; // augment with knowledge graph context
  // Structured output (JSON schema + GBNF grammar)
  enableStructuredOutput: boolean;
  structuredOutputSchema: Record<string, unknown> | null;
  structuredOutputGrammar: string | null;
  structuredOutputSchemaName: string;
}

interface PerformanceMetrics {
  startTime: number | null;
  firstTokenTime: number | null;
  endTime: number | null;
  tokensGenerated: number;
  tokensPerSecond: number | null;
  promptTokens: number | null;
  timeToFirstToken: number | null;
}

interface NonStreamingResponseOptions {
  cleanGraphRAGAnswer?: boolean;
  graphRAGSourceLabel?: string;
  documentFirstGraphRAGAnswer?: boolean;
  timeoutMs?: number;
  fallbackAnswer?: (reason: "timeout" | "error") => string;
}

const defaultSettings: ChatSettings = {
  // Connection settings
  baseUrl: "", // Empty means use current domain
  endpoint: "/v1/chat/completions",
  apiKey: "",
  openaiApiKey: "", // For audio transcription (legacy)
  model: "",
  // Model parameters
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  maxTokens: 1024,
  streamResponse: true,
  enableTools: false,
  selectedTools: [],
  // Reasoning settings
  reasoningLevel: "low", // Default to low to prevent excessive thinking
  // Display settings
  showPerformanceMetrics: false,
  // Voice settings
  voiceModeEnabled: false,
  ttsEnabled: true,
  ttsAutoPlay: true,
  ttsVoice: "alloy",
  ttsSpeed: 1.0,
  sttModel: "base",
  sttLanguage: "auto",
  vadSilenceThreshold: 0.04, // 4% - increase if it triggers too easily
  vadSilenceDuration: 1500, // 1.5 seconds
  noSpeechThreshold: 0.6, // Reject if no_speech_prob > 60%
  // RAG settings
  ragEnabled: false,
  ragSearchMode: "global",
  ragSelectedDomains: [],
  ragTopK: 5,
  ragShowContext: true,
  // GraphRAG settings
  graphEnabled: false,
  // Structured output
  enableStructuredOutput: false,
  structuredOutputSchema: null,
  structuredOutputGrammar: null,
  structuredOutputSchemaName: "response",
};

const initialPerformanceMetrics: PerformanceMetrics = {
  startTime: null,
  firstTokenTime: null,
  endTime: null,
  tokensGenerated: 0,
  tokensPerSecond: null,
  promptTokens: null,
  timeToFirstToken: null,
};

/** llama.cpp can emit ~1e6 tok/s when eval time is 0 ms; client token/Δt can also spike. */
const MAX_SANE_TOKENS_PER_SEC = 10_000;
const MIN_CHAT_OUTPUT_TOKENS = 128;
const DEFAULT_CHAT_OUTPUT_TOKENS = 1024;
const MAX_CHAT_OUTPUT_TOKENS = 2048;
const DOCUMENT_CHAT_OUTPUT_TOKENS = 896;
const CONTEXT_TOKEN_BUFFER = 512;
const MAX_KNOWLEDGE_CONTEXT_CHARS = 7_500;
const MIN_KNOWLEDGE_CONTEXT_CHARS = 900;
const DOCUMENT_CHAT_CONTEXT_MAX_CHARS = 4_800;
const DOCUMENT_CHAT_CONTEXT_MIN_CHARS = 650;
const MAX_KNOWLEDGE_CONTEXT_LINE_CHARS = 420;
const STREAM_FIRST_TOKEN_TIMEOUT_MS = 120_000;
const STREAM_IDLE_COMPLETE_MS = 18_000;
const DOCUMENT_CHAT_MODEL_TIMEOUT_MS = 18_000;

function coerceFiniteNumber(v: unknown): number | undefined {
  if (v == null) return undefined;
  if (typeof v === "number") return Number.isFinite(v) ? v : undefined;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  }
  return undefined;
}

/** Reject absurd server/client rates; accepts numeric strings from JSON. */
function sanitizeTokensPerSecond(v: unknown): number | undefined {
  const n = coerceFiniteNumber(v);
  if (n == null || n <= 0) return undefined;
  if (n > MAX_SANE_TOKENS_PER_SEC) return undefined;
  return n;
}

function estimateDeltaTokens(text: string): number {
  const newTokens = text
    .split(/[\s\n]+/)
    .filter((t: string) => t.length > 0).length;
  return Math.max(1, newTokens);
}

function estimateMessageTokens(msg: ChatMessage): number {
  let count = 0;

  if (typeof msg.content === "string") {
    count += msg.content.length / 3.5;
  } else if (Array.isArray(msg.content)) {
    msg.content.forEach((part: any) => {
      if (part.type === "text") count += (part.text?.length || 0) / 3.5;
      if (part.type === "image_url") count += 1000;
    });
  }

  if (msg.reasoning_content) {
    count += msg.reasoning_content.length / 3.5;
  }

  return Math.ceil(count);
}

function compactKnowledgeContext(
  contextText: string,
  maxChars: number,
): string {
  if (!contextText || contextText.length <= maxChars) return contextText;

  const suffix = `\n\n[GraphRAG context trimmed from ${contextText.length.toLocaleString()} characters so chat generation has room to answer.]`;
  const availableChars = Math.max(0, maxChars - suffix.length);
  const lines = contextText.split("\n");
  const kept: string[] = [];
  const keptIndexes = new Set<number>();
  let length = 0;

  const pushLine = (line: string, index: number): boolean => {
    if (keptIndexes.has(index)) return true;
    const trimmedLine =
      line.length > MAX_KNOWLEDGE_CONTEXT_LINE_CHARS
        ? `${line.slice(0, MAX_KNOWLEDGE_CONTEXT_LINE_CHARS - 3)}...`
        : line;
    const nextLength = length + trimmedLine.length + 1;
    if (nextLength > availableChars) return false;

    kept.push(trimmedLine);
    keptIndexes.add(index);
    length = nextLength;
    return true;
  };

  const nextMajorSectionIndex = (startIndex: number): number => {
    for (let i = startIndex + 1; i < lines.length; i += 1) {
      if (/^(=== .+ ===|--- End Knowledge Context ---)/.test(lines[i])) {
        return i;
      }
    }
    return lines.length;
  };

  const pushSection = (headerPattern: RegExp, maxLines: number): void => {
    const start = lines.findIndex((line) => headerPattern.test(line));
    if (start < 0) return;

    const end = Math.min(nextMajorSectionIndex(start), start + maxLines);
    if (kept.length > 0) {
      const spacerIndex = Number.MIN_SAFE_INTEGER + kept.length;
      if (!pushLine("", spacerIndex)) return;
    }
    for (let i = start; i < end; i += 1) {
      if (!pushLine(lines[i], i)) return;
    }
  };

  const pushLeadingContext = (maxLines: number): void => {
    for (let i = 0; i < Math.min(lines.length, maxLines); i += 1) {
      if (!pushLine(lines[i], i)) return;
    }
  };

  pushSection(/^=== Requested Evidence Coverage ===$/i, 16);
  pushSection(/^=== Knowledge Context ===$/i, 24);
  pushSection(/^=== Graph Answer Plan ===$/i, 42);
  pushSection(/^=== Graph Reasoning Summary ===$/i, 18);
  pushSection(/^=== Relevant Graph Relationships ===$/i, 18);
  pushSection(/^=== Graph Paths ===$/i, 10);
  pushSection(/^=== Bridge Entities ===$/i, 10);
  pushSection(/^=== Pipeline Interaction Handoff ===$/i, 18);
  pushLeadingContext(28);

  const priorityPattern =
    /(requested evidence coverage|relevant graph relationships|groups_visual|shape_group|grouped visual|answer plan|strongest path|direct relationships|knowledge graph paths|local graph relationships|local graph paths|structured graphrag ingest|extracted entities|graphrag|llama nexus|qdrant|typed bridge|pipeline|relationship|evidence|cite|file=|--\[)/i;
  for (let i = 0; i < lines.length; i += 1) {
    if (keptIndexes.has(i)) continue;

    const line = lines[i];
    const shouldKeep = kept.length < 40 || priorityPattern.test(line);
    if (!shouldKeep) continue;

    if (!pushLine(line, i)) break;
  }

  if (kept.length === 0) {
    return `${contextText.slice(0, maxChars - 180)}\n\n[GraphRAG context trimmed to fit the model context window.]`;
  }

  return `${kept.join("\n")}${suffix}`;
}

function safeOutputTokenBudget(
  messages: ChatMessage[],
  requestedMaxTokens: number,
  contextLimit: number,
): number {
  const promptTokens = messages.reduce(
    (total, msg) => total + estimateMessageTokens(msg),
    0,
  );
  const serverRoom = Math.max(
    0,
    contextLimit - promptTokens - CONTEXT_TOKEN_BUFFER,
  );
  const requested =
    Number.isFinite(requestedMaxTokens) && requestedMaxTokens > 0
      ? requestedMaxTokens
      : DEFAULT_CHAT_OUTPUT_TOKENS;

  return Math.max(
    MIN_CHAT_OUTPUT_TOKENS,
    Math.min(
      requested,
      MAX_CHAT_OUTPUT_TOKENS,
      serverRoom || MIN_CHAT_OUTPUT_TOKENS,
    ),
  );
}

async function readErrorDetail(response: Response): Promise<string> {
  try {
    const text = await response.text();
    if (!text) return "";

    try {
      const parsed = JSON.parse(text);
      const detail =
        parsed?.detail?.error?.message ||
        parsed?.detail?.message ||
        parsed?.detail ||
        parsed?.error?.message ||
        parsed?.message;
      return typeof detail === "string" ? detail : text;
    } catch {
      return text;
    }
  } catch {
    return "";
  }
}

function asStreamingChatRequest(
  request: ChatCompletionRequest,
): ChatCompletionRequest {
  return {
    ...request,
    stream: true,
    stream_options: request.stream_options || { include_usage: true },
  };
}

function asNonStreamingChatRequest(
  request: ChatCompletionRequest,
): ChatCompletionRequest {
  const rest = { ...request };
  delete rest.stream_options;
  return {
    ...rest,
    stream: false,
  };
}

/** Generation-phase tok/s (after first token, excluding prefill). */
function computeGenerationTps(
  tokens: number,
  startMs: number,
  firstTokenMs: number | null,
  endMs: number,
): number | null {
  const genStart = firstTokenMs ?? startMs;
  const genTimeSec = (endMs - genStart) / 1000;
  if (tokens <= 0 || genTimeSec <= 0) return null;
  return sanitizeTokensPerSecond(tokens / genTimeSec) ?? null;
}

/** Only OpenAI delta string content — never __verbose (can inject timing garbage into chat). */
function normalizeDeltaContent(raw: unknown): string {
  if (raw == null) return "";
  if (typeof raw !== "string") return "";
  const t = raw;
  if (/^\s*\d+(?:\.\d+)?\s*(?:tok\/s|tokens?\s*\/\s*s)/i.test(t.trim()))
    return "";
  return t;
}

function normalizeChatSettings(settings: ChatSettings): ChatSettings {
  const maxTokens = Number.isFinite(settings.maxTokens)
    ? settings.maxTokens
    : DEFAULT_CHAT_OUTPUT_TOKENS;

  return {
    ...settings,
    maxTokens: Math.min(
      Math.max(maxTokens, MIN_CHAT_OUTPUT_TOKENS),
      MAX_CHAT_OUTPUT_TOKENS,
    ),
  };
}

/** Direct port URLs bypass the backend router (8600=llama.cpp, 8601=vLLM, 8603=NInfer). */
function isDirectInferencePortBaseUrl(baseUrl: string): boolean {
  const trimmed = baseUrl?.trim();
  if (!trimmed) return false;
  try {
    const u = new URL(trimmed);
    const port = u.port || (u.protocol === "https:" ? "443" : "80");
    return port === "8600" || port === "8601" || port === "8603";
  } catch {
    return /:8600\b|:8601\b|:8603\b/.test(trimmed);
  }
}

function isUsableChatApiKey(key?: string): boolean {
  const trimmed = key?.trim() ?? "";
  return Boolean(trimmed) && trimmed !== "placeholder-api-key";
}

function inferenceBackendLabel(backend?: string | null): string {
  if (backend === "ninfer") return "NInfer";
  if (backend === "vllm") return "vLLM";
  if (backend === "llamacpp") return "llama.cpp";
  return backend || "runtime";
}

function stringifyChatContent(
  content: ChatMessage["content"] | null | undefined,
): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return String(content);

  const parts = content
    .map((part) => {
      if (part.type === "text") return part.text?.trim() ?? "";
      if (part.type === "image_url") return "[image]";
      return `[${part.type}]`;
    })
    .filter(Boolean);

  return parts.join("\n");
}

function isLegacyAssistantGreeting(content: string): boolean {
  const normalized = content.trim();
  return (
    normalized.startsWith("Hello! I'm your AI assistant") &&
    normalized.includes("weather lookup, calculator, code execution")
  );
}

function isStarterAssistantPlaceholder(content: string): boolean {
  const normalized = content.trim();
  return (
    normalized === "Ready when you are." || normalized.startsWith("Ready on ")
  );
}

function normalizeLoadedMessages(rawMessages: unknown): ChatMessage[] {
  if (!Array.isArray(rawMessages)) return [];

  const normalized = rawMessages
    .map((raw): ChatMessage | null => {
      if (!raw || typeof raw !== "object") return null;

      const msg = raw as Partial<ChatMessage>;
      if (
        !msg.role ||
        !["system", "user", "assistant", "tool"].includes(msg.role)
      )
        return null;

      const reasoningContent =
        typeof msg.reasoning_content === "string"
          ? msg.reasoning_content
          : undefined;
      const toolCalls = Array.isArray(msg.tool_calls)
        ? msg.tool_calls
        : undefined;
      let content = msg.content;

      if (content == null) {
        content = "";
      } else if (typeof content !== "string" && !Array.isArray(content)) {
        content = String(content);
      }

      if (msg.role === "assistant" && typeof content === "string") {
        if (
          !content.trim() &&
          !reasoningContent?.trim() &&
          !toolCalls?.length
        ) {
          return null;
        }

        if (
          isLegacyAssistantGreeting(content) ||
          isStarterAssistantPlaceholder(content)
        ) {
          return null;
        }
      }

      return {
        role: msg.role,
        content,
        reasoning_content: reasoningContent,
        tool_calls: toolCalls,
        tool_call_id:
          typeof msg.tool_call_id === "string" ? msg.tool_call_id : undefined,
      };
    })
    .filter((msg): msg is ChatMessage => msg !== null);

  return normalized;
}

export const ChatPage: React.FC<ChatPageProps> = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const copyTextToClipboard = useClipboard();
  const deployPath = getRoutePath("deploy");
  const routeState = location.state as ChatRouteState | null;
  const deploymentContext = routeState?.deployment;
  const graphRAGRouteHandoff = routeState?.graphragHandoff;
  const graphRAGRouteRunIdsKey =
    getGraphRAGHandoffRunIds(graphRAGRouteHandoff).join("|");
  const graphRAGRouteRequestKey = graphRAGRouteHandoff?.chatContextRequest
    ? JSON.stringify(graphRAGRouteHandoff.chatContextRequest)
    : "";

  // Load settings from localStorage or use defaults
  const loadSettings = (): ChatSettings => {
    try {
      const saved = localStorage.getItem("chat-settings");
      if (saved) {
        const parsed = JSON.parse(saved);
        return normalizeChatSettings({ ...defaultSettings, ...parsed });
      }
    } catch (error) {
      console.warn("Failed to load chat settings from localStorage:", error);
    }
    return normalizeChatSettings(defaultSettings);
  };

  const [currentModelName, setCurrentModelName] = useState<string>("AI Model");
  const [activeInferenceBackend, setActiveInferenceBackend] = useState<
    string | null
  >(null);
  const [deployWorkloadProfile, setDeployWorkloadProfile] = useState<
    string | null
  >(null);
  const [deploymentHandoff, setDeploymentHandoff] =
    useState<DeploymentRouteContext | null>(null);
  const [graphRAGChatHandoff, setGraphRAGChatHandoff] =
    useState<GraphRAGChatHandoff | null>(null);
  const appliedGraphRAGHandoffRef = useRef<string | null>(null);
  const autoSubmittedGraphRAGHandoffRef = useRef<string | null>(null);
  const sendInFlightRef = useRef(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [chatBusyPhase, setChatBusyPhase] =
    useState<ChatBusyPhase>("idle");
  const [settings, setSettings] = useState(loadSettings);
  const [showSettings, setShowSettings] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availableTools, setAvailableTools] = useState<ExtendedTool[]>(
    ToolsService.getBuiltInToolsExtended(),
  );
  const [mcpToolsLoading, setMcpToolsLoading] = useState(false);
  const [constraintEditorOpen, setConstraintEditorOpen] = useState(false);

  // Conversation management state
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [currentConversationId, setCurrentConversationId] = useState<
    string | null
  >(null);
  const [conversationTitle, setConversationTitle] =
    useState<string>("New Conversation");
  const [isSaving, setIsSaving] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [saveStatus, setSaveStatus] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
  const saveConversationRef = useRef<(() => Promise<void>) | null>(null);
  const [editingTitle, setEditingTitle] = useState(false);
  const [editTitleValue, setEditTitleValue] = useState("");

  // Context window tracking state
  const [tokenCount, setTokenCount] = useState({ prompt: 0, total: 0 });
  const [contextLimit, setContextLimit] = useState(4096);

  // Performance metrics tracking
  const [performanceMetrics, setPerformanceMetrics] =
    useState<PerformanceMetrics>(initialPerformanceMetrics);
  const performanceRef = useRef<PerformanceMetrics>(initialPerformanceMetrics);

  // Multi-modal inputs state
  const [uploadedImages, setUploadedImages] = useState<
    Array<{ file: File; preview: string; base64: string }>
  >([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Audio recording state (legacy manual recording)
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [showAudioDialog, setShowAudioDialog] = useState(false);

  // Voice mode state
  const [sttServiceAvailable, setSttServiceAvailable] = useState(false);
  const [ttsServiceAvailable, setTtsServiceAvailable] = useState(false);
  const [voicePanelExpanded, setVoicePanelExpanded] = useState(false);
  const [ttsStarting, setTtsStarting] = useState(false);
  const [ttsStopping, setTtsStopping] = useState(false);

  // Message editing state
  const [editingMessageIndex, setEditingMessageIndex] = useState<number | null>(
    null,
  );
  const [editingContent, setEditingContent] = useState<string>("");

  // RAG context state
  const [ragContext, setRagContext] = useState<RAGChunk[]>([]);
  const [ragLoading, setRagLoading] = useState(false);

  // Graph context state
  const [graphEntities, setGraphEntities] = useState<any[]>([]);
  const [graphRelationships, setGraphRelationships] = useState<
    UnifiedRelationship[]
  >([]);
  const [graphLoading, setGraphLoading] = useState(false);

  // Unified context state (when both RAG + Graph are enabled)
  const [unifiedChunks, setUnifiedChunks] = useState<UnifiedChunk[]>([]);
  const [unifiedLoading, setUnifiedLoading] = useState(false);
  const [unifiedCrossRefs, setUnifiedCrossRefs] = useState<any[]>([]);
  const [unifiedQuality, setUnifiedQuality] = useState<QualitySignals | null>(
    null,
  );
  const [unifiedGraphConnections, setUnifiedGraphConnections] = useState<
    string[]
  >([]);
  const [unifiedGraphReasoning, setUnifiedGraphReasoning] =
    useState<GraphReasoning | null>(null);
  const [unifiedPipelineArtifacts, setUnifiedPipelineArtifacts] =
    useState<PipelineArtifactContext | null>(null);
  const [conversationEntities, setConversationEntities] = useState<
    Array<{ name: string; type: string }>
  >([]);

  // Entity explorer state
  const [explorerAnchorEl, setExplorerAnchorEl] = useState<HTMLElement | null>(
    null,
  );
  const [explorerEntity, setExplorerEntity] = useState({ name: "", type: "" });

  useEffect(() => {
    if (!deploymentContext?.modelName) return;

    setDeploymentHandoff(deploymentContext);
    setCurrentModelName(deploymentContext.modelName);
    setActiveInferenceBackend(deploymentContext.backend || null);
    setSettings((prev) => {
      const updated = {
        ...prev,
        model: deploymentContext.modelName || prev.model,
        baseUrl: "",
        endpoint: "/v1/chat/completions",
      };

      try {
        localStorage.setItem("chat-settings", JSON.stringify(updated));
      } catch (error) {
        console.warn("Failed to save deployed chat settings:", error);
      }

      return updated;
    });
  }, [
    deploymentContext?.backend,
    deploymentContext?.endpoint,
    deploymentContext?.modelName,
  ]);

  useEffect(() => {
    const routeRunIds = getGraphRAGHandoffRunIds(graphRAGRouteHandoff);
    if (routeRunIds.length === 0 && !graphRAGRouteHandoff?.prompt) return;
    const handoffKey = `${routeRunIds.join("|")}:${graphRAGRouteHandoff?.prompt || ""}:${graphRAGRouteRequestKey}`;
    if (appliedGraphRAGHandoffRef.current === handoffKey) return;
    appliedGraphRAGHandoffRef.current = handoffKey;
    if (routeRunIds.length > 0 || graphRAGRouteHandoff?.autoSubmit) {
      setCurrentConversationId(null);
      setConversationTitle("New Conversation");
      setMessages([]);
      setHasUnsavedChanges(false);
      setSaveStatus("idle");
      setError(null);
      setRagContext([]);
      setGraphEntities([]);
      setGraphRelationships([]);
      setUnifiedChunks([]);
      setUnifiedCrossRefs([]);
      setUnifiedQuality(null);
      setUnifiedGraphConnections([]);
      setUnifiedGraphReasoning(null);
      setUnifiedPipelineArtifacts(null);
      setConversationEntities([]);
    }
    setGraphRAGChatHandoff(graphRAGRouteHandoff || null);
    if (
      graphRAGRouteHandoff?.prompt &&
      !graphRAGRouteHandoff.autoSubmit &&
      !(graphRAGRouteHandoff.suggestedQuestions?.length || 0)
    ) {
      setInput((prev) =>
        prev.trim() ? prev : graphRAGRouteHandoff?.prompt || "",
      );
    }
    setSettings((prev) => {
      const updated = {
        ...prev,
        ragEnabled: true,
        graphEnabled: true,
        ragShowContext: true,
      };

      try {
        localStorage.setItem("chat-settings", JSON.stringify(updated));
      } catch (error) {
        console.warn("Failed to save GraphRAG handoff chat settings:", error);
      }

      return updated;
    });
  }, [
    graphRAGRouteHandoff?.domain,
    graphRAGRouteHandoff?.autoSubmit,
    graphRAGRouteHandoff?.filename,
    graphRAGRouteHandoff?.prompt,
    graphRAGRouteRequestKey,
    graphRAGRouteRunIdsKey,
  ]);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const messagesScrollRef = useRef<HTMLDivElement | null>(null);
  const inlineAnswerReferencesRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  // Voice input hook with VAD
  const handleVoiceTranscription = useCallback((text: string) => {
    if (text.trim()) {
      setInput((prev) => (prev ? `${prev} ${text}` : text));
    }
  }, []);

  const voiceInput = useVoiceInput({
    silenceThreshold: settings.vadSilenceThreshold,
    silenceDuration: settings.vadSilenceDuration,
    minRecordingTime: 500,
    maxRecordingTime: 60000,
    sttModel: settings.sttModel,
    sttLanguage: settings.sttLanguage,
    noSpeechThreshold: settings.noSpeechThreshold,
    onTranscription: handleVoiceTranscription,
    onError: (error) => setAudioError(error),
  });

  // TTS hook
  const tts = useTTS({
    voice: settings.ttsVoice,
    model: "tts-1",
    speed: settings.ttsSpeed,
    autoPlay: settings.ttsAutoPlay,
    onError: (error) => console.warn("TTS error:", error),
  });

  // Check STT/TTS service availability
  useEffect(() => {
    const checkServices = async () => {
      try {
        const sttStatus = await apiService.getSTTStatus();
        setSttServiceAvailable(sttStatus.running);
      } catch {
        setSttServiceAvailable(false);
      }

      try {
        const ttsStatus = await apiService.getTTSStatus();
        setTtsServiceAvailable(ttsStatus.running);
      } catch {
        setTtsServiceAvailable(false);
      }
    };

    checkServices();
    // Check periodically
    const interval = setInterval(checkServices, 30000);
    return () => clearInterval(interval);
  }, []);

  // Load MCP tools from connected servers
  useEffect(() => {
    const loadMCPTools = async () => {
      setMcpToolsLoading(true);
      try {
        const mcpTools = await ToolsService.getMCPTools();
        setAvailableTools([
          ...ToolsService.getBuiltInToolsExtended(),
          ...mcpTools,
        ]);
      } catch (error) {
        debugLog("chat", "Failed to load MCP tools", error);
      } finally {
        setMcpToolsLoading(false);
      }
    };
    loadMCPTools();
  }, []);

  // Start TTS service
  const handleStartTTS = async () => {
    setTtsStarting(true);
    setError(null);
    try {
      const result = await apiService.startTTSService({
        voice: settings.ttsVoice,
      });
      if (result.success) {
        setTtsServiceAvailable(true);
      } else {
        setError(result.message || "Failed to start TTS service");
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start TTS service",
      );
    } finally {
      setTtsStarting(false);
    }
  };

  // Stop TTS service
  const handleStopTTS = async () => {
    setTtsStopping(true);
    try {
      await apiService.stopTTSService();
      setTtsServiceAvailable(false);
      tts.stop(); // Stop any playing audio
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to stop TTS service",
      );
    } finally {
      setTtsStopping(false);
    }
  };

  // [2026-10-15 | INFO | ChatPage.handleToggleVoiceMode | Toggle the fully-local voice loop (STT -> LLM -> TTS); requires the local STT service to be running]
  const handleToggleVoiceMode = () => {
    if (!sttServiceAvailable) return;
    const next = !settings.voiceModeEnabled;
    if (!next) {
      // Leaving voice mode: release the microphone if the local VAD loop is active
      if (voiceInput.isListening) {
        voiceInput.stopListening();
      }
      tts.stop();
    }
    saveSettings({ ...settings, voiceModeEnabled: next });
  };

  // Auto-send when voice input transcription is complete and VAD detects end of speech
  useEffect(() => {
    if (
      settings.voiceModeEnabled &&
      !voiceInput.isRecording &&
      !voiceInput.isTranscribing &&
      input.trim() &&
      voiceInput.isListening
    ) {
      // Small delay to allow for multi-utterance input
      const timer = setTimeout(() => {
        if (input.trim()) {
          handleSendMessage();
        }
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [
    voiceInput.isRecording,
    voiceInput.isTranscribing,
    input,
    settings.voiceModeEnabled,
    voiceInput.isListening,
  ]);

  const scrollToBottom = () => {
    const scroller = messagesScrollRef.current;

    if (scroller) {
      scroller.scrollTo({ top: scroller.scrollHeight, behavior: "smooth" });
      return;
    }

    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  };

  const handleChatSurfaceWheel = useCallback(
    (event: React.WheelEvent<HTMLDivElement>) => {
      const scroller = messagesScrollRef.current;
      const target = event.target as HTMLElement | null;

      if (!scroller || !target || scroller.contains(target)) return;
      if (
        target.closest(
          'input, textarea, [role="listbox"], [data-preserve-wheel="true"]',
        )
      )
        return;
      if (Math.abs(event.deltaY) <= Math.abs(event.deltaX)) return;

      const atTop = scroller.scrollTop <= 0;
      const atBottom =
        Math.ceil(scroller.scrollTop + scroller.clientHeight) >=
        scroller.scrollHeight;

      if ((event.deltaY < 0 && atTop) || (event.deltaY > 0 && atBottom)) return;

      scroller.scrollTop += event.deltaY;
      event.preventDefault();
    },
    [],
  );

  // Save settings to localStorage whenever they change
  const saveSettings = (newSettings: ChatSettings) => {
    const normalizedSettings = normalizeChatSettings(newSettings);
    try {
      localStorage.setItem("chat-settings", JSON.stringify(normalizedSettings));
      setSettings(normalizedSettings);
    } catch (error) {
      console.warn("Failed to save chat settings to localStorage:", error);
      setSettings(normalizedSettings); // Still update state even if save fails
    }
  };

  // Estimate token count (rough approximation: ~4 chars per token)
  const estimateTokenCount = useCallback((msgs: ChatMessage[]) => {
    let totalChars = 0;
    msgs.forEach((msg) => {
      // Handle string content
      if (typeof msg.content === "string") {
        totalChars += msg.content.length;
      }
      // Handle array content (multi-modal)
      else if (Array.isArray(msg.content)) {
        msg.content.forEach((part: any) => {
          if (part.type === "text" && part.text) {
            totalChars += part.text.length;
          }
          // Approximate image tokens (images typically use ~85-170 tokens in vision models)
          if (part.type === "image_url") {
            totalChars += 100 * 4; // Estimate 100 tokens per image
          }
        });
      }
      if (msg.reasoning_content) {
        totalChars += msg.reasoning_content.length;
      }
    });
    return Math.ceil(totalChars / 4);
  }, []);

  // Save conversation to backend
  const saveConversation = useCallback(
    async (forceNew: boolean = false) => {
      if (messages.length <= 1) return; // Don't save empty conversations

      setIsSaving(true);
      try {
        const conversationData = {
          title: conversationTitle || "New Conversation",
          messages: messages.map((msg) => ({
            role: msg.role,
            content: stringifyChatContent(msg.content),
            reasoning_content: msg.reasoning_content,
            tool_calls: msg.tool_calls,
            tool_call_id: msg.tool_call_id,
          })),
          model: currentModelName,
          settings: {
            temperature: settings.temperature,
            topP: settings.topP,
            topK: settings.topK,
            maxTokens: settings.maxTokens,
            reasoningLevel: settings.reasoningLevel,
          },
        };

        if (currentConversationId && !forceNew) {
          // Update existing conversation
          await apiService.updateConversation(
            currentConversationId,
            conversationData,
          );
        } else {
          // Create new conversation
          // Auto-generate title from first user message
          const firstUserMsg = messages.find((m) => m.role === "user");
          if (firstUserMsg && conversationTitle === "New Conversation") {
            let titleText = "";
            if (typeof firstUserMsg.content === "string") {
              titleText = firstUserMsg.content;
            } else if (Array.isArray(firstUserMsg.content)) {
              // Extract text from multi-modal content
              const textParts = firstUserMsg.content
                .filter((part: any) => part.type === "text" && part.text)
                .map((part: any) => part.text);
              titleText = textParts.join(" ");
            }
            conversationData.title =
              titleText.slice(0, 50) + (titleText.length > 50 ? "..." : "");
            setConversationTitle(conversationData.title);
          }

          const newConversation =
            await apiService.createConversation(conversationData);
          setCurrentConversationId(newConversation.id);
        }
        setHasUnsavedChanges(false);
      } catch (err) {
        console.error("Failed to save conversation:", err);
        setError("Failed to save conversation");
      } finally {
        setIsSaving(false);
      }
    },
    [
      messages,
      conversationTitle,
      currentModelName,
      settings,
      currentConversationId,
    ],
  );

  // Keep ref in sync for use in non-stale callbacks
  useEffect(() => {
    saveConversationRef.current = saveConversation;
  }, [saveConversation]);

  // Load a conversation from the sidebar
  const handleSelectConversation = useCallback(
    async (conversation: ConversationListItem) => {
      try {
        const fullConversation = await apiService.getConversation(
          conversation.id,
        );
        setCurrentConversationId(fullConversation.id);
        setConversationTitle(fullConversation.title);
        setMessages(normalizeLoadedMessages(fullConversation.messages));
        setHasUnsavedChanges(false);
        setError(null);
      } catch (err) {
        console.error("Failed to load conversation:", err);
        setError("Failed to load conversation");
      }
    },
    [],
  );

  // Start a new conversation
  const handleNewConversation = useCallback(() => {
    setCurrentConversationId(null);
    setConversationTitle("New Conversation");
    setMessages([]);
    setHasUnsavedChanges(false);
    setError(null);
  }, []);

  // Save title to backend
  const handleTitleSave = useCallback(async () => {
    const trimmed = editTitleValue.trim();
    if (!trimmed || trimmed === conversationTitle) {
      setEditingTitle(false);
      return;
    }
    setConversationTitle(trimmed);
    setEditingTitle(false);
    if (currentConversationId) {
      try {
        await apiService.updateConversation(currentConversationId, {
          title: trimmed,
        });
      } catch (err) {
        console.error("Failed to update title:", err);
      }
    }
  }, [editTitleValue, conversationTitle, currentConversationId]);

  // Fork conversation from a specific message index
  const handleForkConversation = useCallback(
    async (fromIndex: number) => {
      try {
        const forkedMessages = messages
          .slice(0, fromIndex + 1)
          .map((msg: any) => ({
            role: msg.role,
            content: msg.content,
            reasoning_content: msg.reasoning_content,
            tool_calls: msg.tool_calls,
            tool_call_id: msg.tool_call_id,
          }));
        const firstUserMsg = forkedMessages.find((m: any) => m.role === "user");
        let forkTitle = conversationTitle;
        if (firstUserMsg && typeof firstUserMsg.content === "string") {
          forkTitle = firstUserMsg.content.slice(0, 40) + " (fork)";
        } else {
          forkTitle = conversationTitle + " (fork)";
        }
        const newConv = await apiService.createConversation({
          title: forkTitle,
          messages: forkedMessages,
          model: currentModelName,
        });
        setCurrentConversationId(newConv.id);
        setConversationTitle(forkTitle);
        setMessages(forkedMessages);
        setHasUnsavedChanges(false);
        setSaveStatus("saved");
        setTimeout(() => setSaveStatus("idle"), 2000);
      } catch (err) {
        console.error("Failed to fork conversation:", err);
      }
    },
    [messages, conversationTitle, currentModelName],
  );

  // Update token count when messages change
  useEffect(() => {
    const count = estimateTokenCount(messages);
    setTokenCount({ prompt: count, total: count });

    // Mark as having unsaved changes when messages change (except initial load)
    if (messages.length > 1) {
      setHasUnsavedChanges(true);
    }
  }, [messages, estimateTokenCount]);

  // Cleanup image previews on unmount
  useEffect(() => {
    return () => {
      uploadedImages.forEach((img) => URL.revokeObjectURL(img.preview));
    };
  }, [uploadedImages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Use backend LLM proxy (not :8600/:8601) so chat follows whichever backend is deployed
  useEffect(() => {
    const alignConnectionWithActiveBackend = async () => {
      try {
        const backends = await apiService.getBackendsStatus();
        setActiveInferenceBackend(backends.active || null);
        const llamacpp = backends.backends?.llamacpp;
        const ninfer = backends.backends?.ninfer;
        const profile = llamacpp?.config?.mtp?.workload_profile;
        setDeployWorkloadProfile(typeof profile === "string" ? profile : null);

        const current = loadSettings();
        const updated = { ...current };
        const ninferRunning = Boolean(ninfer?.running) || backends.active === "ninfer";
        const ninferServer = ninfer?.config?.server as
          { api_key?: string; apiKey?: string } | undefined;
        const ninferServed =
          (ninfer?.config?.model as { served_name?: string; artifact?: string } | undefined)
            ?.served_name ||
          (ninfer?.model as { name?: string } | undefined)?.name ||
          "";

        // [2026-09-05 | INFO | ChatPage.alignConnectionWithActiveBackend | Direct :8603
        // without the deploy key is a 401. Prefer the backend proxy when NInfer is live.
        if (ninferRunning || !isDirectInferencePortBaseUrl(current.baseUrl)) {
          updated.baseUrl = "";
        }
        if (ninferRunning && ninferServed) {
          updated.model = ninferServed;
        }
        if (!isUsableChatApiKey(updated.apiKey)) {
          updated.apiKey = "";
        }

        // Sync API key from deployed backend when using a direct inference port.
        if (isDirectInferencePortBaseUrl(updated.baseUrl)) {
          const serviceConfig = await apiService
            .getServiceConfig()
            .catch(() => null);
          const llamacppServer = llamacpp?.config?.server as
            { api_key?: string; apiKey?: string } | undefined;
          const serviceServer = serviceConfig?.server as
            { api_key?: string; apiKey?: string } | undefined;
          const deployKey =
            ninferServer?.api_key ||
            ninferServer?.apiKey ||
            llamacppServer?.api_key ||
            llamacppServer?.apiKey ||
            serviceServer?.api_key ||
            serviceServer?.apiKey;
          const usableDeployKey = deployKey?.trim() ?? "";
          if (isUsableChatApiKey(usableDeployKey)) {
            updated.apiKey = usableDeployKey;
          }
        }

        if (JSON.stringify(updated) !== JSON.stringify(current)) {
          localStorage.setItem("chat-settings", JSON.stringify(updated));
          setSettings(updated);
        }
      } catch (error) {
        console.warn(
          "Failed to align chat connection with active backend:",
          error,
        );
      }
    };
    alignConnectionWithActiveBackend();
  }, []);

  // Fetch current model info on mount
  useEffect(() => {
    const fetchCurrentModel = async () => {
      try {
        const modelInfo = await apiService.getCurrentModel();
        if (modelInfo) {
          // Set context limit
          const limit =
            modelInfo.context_size ||
            modelInfo.contextLength ||
            modelInfo.model?.context_size ||
            4096;
          if (limit) setContextLimit(limit);

          if (modelInfo.name) {
            setCurrentModelName(modelInfo.name);
            // Always sync model setting from the active backend so the
            // correct model ID is used in chat requests (e.g. vLLM served
            // model name may differ from the llama.cpp config model name).
            setSettings((prev) => {
              if (prev.model === modelInfo.name) return prev;
              const updated = { ...prev, model: modelInfo.name };
              try {
                localStorage.setItem("chat-settings", JSON.stringify(updated));
              } catch (error) {
                console.warn(
                  "Failed to save chat settings to localStorage:",
                  error,
                );
              }
              return updated;
            });
          }
        }
      } catch (error) {
        console.warn("Failed to fetch current model:", error);
        // Keep default generic message if model fetch fails
      }
    };
    fetchCurrentModel();
  }, []);

  const resetUnifiedContext = () => {
    setUnifiedChunks([]);
    setUnifiedCrossRefs([]);
    setUnifiedQuality(null);
    setUnifiedGraphConnections([]);
    setUnifiedGraphReasoning(null);
    setUnifiedPipelineArtifacts(null);
  };

  const handleSendMessage = async (
    messageOverride?: string,
    displayOverride?: string,
    handoffOverride?: GraphRAGChatHandoff | null,
  ) => {
    const activeGraphRAGChatHandoff =
      handoffOverride !== undefined ? handoffOverride : graphRAGChatHandoff;
    const outgoingText =
      typeof messageOverride === "string" ? messageOverride : input;
    const visibleOutgoingText =
      typeof displayOverride === "string" && displayOverride.trim()
        ? displayOverride
        : outgoingText;
    if (
      (!outgoingText.trim() && uploadedImages.length === 0) ||
      isLoading ||
      sendInFlightRef.current
    ) {
      return;
    }
    sendInFlightRef.current = true;

    // Construct message content with images if present
    let messageContent: any = visibleOutgoingText.trim();
    let requestMessageContent: any = outgoingText.trim();

    // If images are uploaded, format the content for multi-modal models (OpenAI format)
    if (uploadedImages.length > 0) {
      // OpenAI multi-modal format: content is an array of parts
      const contentParts: any[] = [];
      const requestContentParts: any[] = [];

      // Add text if present
      if (visibleOutgoingText.trim()) {
        contentParts.push({
          type: "text",
          text: visibleOutgoingText.trim(),
        });
      }
      if (outgoingText.trim()) {
        requestContentParts.push({
          type: "text",
          text: outgoingText.trim(),
        });
      }

      // Add images
      uploadedImages.forEach((img) => {
        const imagePart = {
          type: "image_url",
          image_url: {
            url: img.base64,
          },
        };
        contentParts.push(imagePart);
        requestContentParts.push(imagePart);
      });

      messageContent = contentParts;
      requestMessageContent = requestContentParts;
    }

    const userMessage: ChatMessage = {
      role: "user",
      content: messageContent,
    };
    const requestUserMessage: ChatMessage = {
      role: "user",
      content: requestMessageContent,
    };

    setMessages((prev: ChatMessage[]) => [...prev, userMessage]);
    setInput("");

    // Clear uploaded images after sending
    uploadedImages.forEach((img) => URL.revokeObjectURL(img.preview));
    setUploadedImages([]);

    setIsLoading(true);
    setChatBusyPhase(
      hasGraphRAGDocumentContextHandoff(activeGraphRAGChatHandoff)
        ? "preparing_context"
        : "writing_answer",
    );
    setError(null);
    resetUnifiedContext();

    // Reset performance metrics for new generation
    performanceRef.current = initialPerformanceMetrics;
    setPerformanceMetrics(initialPerformanceMetrics);

    try {
      const graphRAGContextRequest = activeGraphRAGChatHandoff?.chatContextRequest;
      const hasGraphRAGHandoffContext =
        hasGraphRAGDocumentContextHandoff(activeGraphRAGChatHandoff);
      let contextLimitForRequest = contextLimit;
      let modelForRequest = settings.model;
      if (
        hasGraphRAGHandoffContext ||
        !modelForRequest ||
        modelForRequest === "AI Model"
      ) {
        const modelInfo = await apiService.getCurrentModel().catch(() => null);
        const detectedContextLimit = coerceFiniteNumber(
          modelInfo?.context_size ||
            modelInfo?.contextLength ||
            modelInfo?.model?.context_size,
        );
        if (detectedContextLimit && detectedContextLimit > 0) {
          contextLimitForRequest = detectedContextLimit;
          setContextLimit(detectedContextLimit);
        }
        const activeModelName =
          typeof modelInfo?.name === "string" ? modelInfo.name : "";
        if (
          activeModelName &&
          (hasGraphRAGHandoffContext || !modelForRequest || modelForRequest === "AI Model")
        ) {
          modelForRequest = activeModelName;
          setCurrentModelName(activeModelName);
          setSettings((prev) => {
            if (prev.model === activeModelName) return prev;
            const updated = { ...prev, model: activeModelName };
            try {
              localStorage.setItem("chat-settings", JSON.stringify(updated));
            } catch (error) {
              console.warn(
                "Failed to save active chat model to localStorage:",
                error,
              );
            }
            return updated;
          });
        }
      }
      if (
        (!modelForRequest || modelForRequest === "AI Model") &&
        !hasGraphRAGHandoffContext
      ) {
        throw new Error("No active chat model is loaded.");
      }

      const selectedToolsForRequest = settings.enableTools
        ? availableTools.filter((tool: Tool) =>
            settings.selectedTools.includes(tool.function.name),
          )
        : [];
      const contextSettingsForRequest: ChatSettings = hasGraphRAGHandoffContext
        ? {
            ...settings,
            graphEnabled: true,
            ragEnabled:
              typeof graphRAGContextRequest?.rag_enabled === "boolean"
                ? graphRAGContextRequest.rag_enabled
                : true,
            ragShowContext: true,
          }
        : settings;

      const shouldForceGraphRAGV2Context =
        hasGraphRAGHandoffContext &&
        (getGraphRAGHandoffRunIds(activeGraphRAGChatHandoff).length > 0 ||
          graphRAGContextRequest?.pipeline_run_id ||
          (graphRAGContextRequest?.pipeline_run_ids?.length || 0) > 0 ||
          graphRAGContextRequest?.use_local_pipeline_context ||
          graphRAGContextRequest?.include_local_pipeline);
      const preparedGraphRAGEndpointForRequest = shouldForceGraphRAGV2Context
        ? "/api/v1/graphrag/chat-context-v2"
        : normalizeGraphRAGChatContextEndpoint(
            activeGraphRAGChatHandoff?.chatContextEndpoint,
          );
      const shouldUseUnifiedGraphRAGContext =
        contextSettingsForRequest.graphEnabled &&
        (shouldForceGraphRAGV2Context ||
          shouldUseUnifiedGraphRAGChatContext(
            activeGraphRAGChatHandoff?.chatContextEndpoint,
            contextSettingsForRequest,
          ));

      // Fetch RAG context if enabled
      let ragContextText = "";
      if (
        contextSettingsForRequest.ragEnabled &&
        shouldUseUnifiedGraphRAGContext
      ) {
        setRagContext([]);
      } else if (contextSettingsForRequest.ragEnabled) {
        setRagLoading(true);
        try {
          const queryText =
            typeof requestMessageContent === "string"
              ? requestMessageContent
              : requestMessageContent.find((p: any) => p.type === "text")?.text || "";

          if (queryText) {
            const ragPayload: any = {
              query: queryText,
              top_k: contextSettingsForRequest.ragTopK,
            };

            if (contextSettingsForRequest.ragSearchMode === "global") {
              ragPayload.search_all_domains = true;
            } else if (contextSettingsForRequest.ragSelectedDomains.length > 0) {
              ragPayload.domain_id =
                contextSettingsForRequest.ragSelectedDomains[0]; // Primary domain
            }

            const ragResponse = await fetch("/api/v1/rag/retrieve", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(ragPayload),
            });

            if (ragResponse.ok) {
              const ragData = await ragResponse.json();
              const chunks: RAGChunk[] = (ragData.results || []).map(
                (r: any) => ({
                  id: r.id || r.chunk_id,
                  content: r.content || r.text,
                  score: r.score || r.similarity || 0,
                  document_id: r.document_id,
                  document_title: r.document_title || r.metadata?.title,
                  chunk_index: r.chunk_index ?? 0,
                  metadata: r.metadata,
                }),
              );

              setRagContext(chunks);

              if (chunks.length > 0) {
                ragContextText =
                  "\n\n--- Retrieved Context ---\n" +
                  chunks.map((c, i) => `[${i + 1}] ${c.content}`).join("\n\n") +
                  "\n--- End Context ---\n\nUse the above context to help answer the user's question.";
              }
            }
          }
        } catch (ragError) {
          console.warn("RAG retrieval failed:", ragError);
        } finally {
          setRagLoading(false);
        }
      } else {
        setRagContext([]);
      }

      // Fetch graph context if enabled
      let graphContextText = "";
      const handoffFilenames = getGraphRAGHandoffFilenames(activeGraphRAGChatHandoff);
      const graphRAGFallbackContext: GraphRAGDocumentFallbackContext = {
        query: outgoingText,
        sourceLabel: activeGraphRAGChatHandoff?.filename,
        chunks: handoffFilenames.map((filename) => ({
          content:
            "Uploaded document is available in this GraphRAG batch, but no top retrieved evidence surfaced for it in this response.",
          metadata: { filename },
        })),
        relationships: [],
        graphReasoning: null,
      };
      if (contextSettingsForRequest.graphEnabled) {
        setGraphLoading(true);
        try {
          const queryText =
            typeof requestMessageContent === "string"
              ? requestMessageContent
              : requestMessageContent.find((p: any) => p.type === "text")?.text || "";

          if (queryText) {
            const primaryGraphRAGEndpoint = preparedGraphRAGEndpointForRequest;
            // ── Unified mode: v2 handles graph-only and graph+RAG document context ──
            if (shouldUseUnifiedGraphRAGContext) {
              setUnifiedLoading(true);
              try {
                const unifiedGraphRAGPayload = buildGraphRAGChatContextPayload(
                  activeGraphRAGChatHandoff,
                  queryText,
                  contextSettingsForRequest,
                );
                const unifiedResponse = await fetch(primaryGraphRAGEndpoint, {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify(unifiedGraphRAGPayload),
                });

                if (unifiedResponse.ok) {
                  const uData = await unifiedResponse.json();
                  const nextUnifiedChunks = uData.chunks || [];
                  const nextRelationships = uData.relationships || [];
                  const nextGraphReasoning = uData.graph_reasoning || null;
                  const nextPipelineArtifacts = uData.pipeline_artifacts || null;
                  // Populate unified state
                  setUnifiedChunks(nextUnifiedChunks);
                  setUnifiedCrossRefs(uData.cross_references || []);
                  setUnifiedQuality(uData.quality_signals || null);
                  setUnifiedGraphConnections(uData.graph_connections || []);
                  setUnifiedGraphReasoning(nextGraphReasoning);
                  setUnifiedPipelineArtifacts(nextPipelineArtifacts);
                  graphRAGFallbackContext.chunks = [
                    ...(graphRAGFallbackContext.chunks || []),
                    ...nextUnifiedChunks,
                  ];
                  graphRAGFallbackContext.relationships = nextRelationships;
                  graphRAGFallbackContext.graphReasoning = nextGraphReasoning;
                  // Also populate the individual states so old components work
                  setGraphEntities(uData.entities || []);
                  setGraphRelationships(nextRelationships);
                  setRagContext(
                    nextUnifiedChunks.map((c: any) => ({
                      id: c.id,
                      content: c.content,
                      score: c.score,
                      document_id: c.document_id,
                      document_title: c.document_title,
                      chunk_index: c.chunk_index,
                      metadata: c.metadata,
                    })),
                  );
                  // Accumulate entities for conversation sidebar
                  if (uData.entities?.length > 0) {
                    setConversationEntities((prev: any[]) => {
                      const existing = new Set(
                        prev.map((e: { name: string }) => e.name.toLowerCase()),
                      );
                      const newEntities = uData.entities.filter(
                        (e: { name: string }) =>
                          !existing.has(e.name.toLowerCase()),
                      );
                      return [...prev, ...newEntities];
                    });
                  }
                  // Use the unified context text
                  graphContextText = uData.context_text || "";
                  // Clear individual ragContextText since unified has everything
                  ragContextText = "";
                }
              } catch (unifiedError) {
                console.warn(
                  "Unified context failed, falling back to separate fetches:",
                  unifiedError,
                );
                // Fall through to the normal graph fetch below
              } finally {
                setUnifiedLoading(false);
              }
            }

            // If unified mode didn't produce results, fall back to old graph context
            if (!graphContextText) {
              const graphContextPayload = buildGraphRAGChatContextPayload(
                activeGraphRAGChatHandoff,
                queryText,
                contextSettingsForRequest,
              );
              const graphFallbackEndpoint =
                primaryGraphRAGEndpoint === "/api/v1/graphrag/chat-context-v2"
                  ? "/api/v1/graphrag/chat-context"
                  : primaryGraphRAGEndpoint;
              const graphResponse = await fetch(graphFallbackEndpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(graphContextPayload),
              });

              if (graphResponse.ok) {
                const graphData = await graphResponse.json();
                const nextRelationships = graphData.relationships || [];
                const nextGraphReasoning = graphData.graph_reasoning || null;
                const nextPipelineArtifacts =
                  graphData.pipeline_artifacts || null;
                setGraphEntities(graphData.entities || []);
                setGraphRelationships(nextRelationships);
                setUnifiedGraphConnections(graphData.graph_connections || []);
                setUnifiedGraphReasoning(nextGraphReasoning);
                setUnifiedPipelineArtifacts(nextPipelineArtifacts);
                graphRAGFallbackContext.chunks = [
                  ...(graphRAGFallbackContext.chunks || []),
                  ...(graphData.chunks || []),
                ];
                graphRAGFallbackContext.relationships = nextRelationships;
                graphRAGFallbackContext.graphReasoning = nextGraphReasoning;
                if (graphData.entities?.length > 0) {
                  setConversationEntities((prev: any[]) => {
                    const existing = new Set(
                      prev.map((e: { name: string }) => e.name.toLowerCase()),
                    );
                    const newEntities = graphData.entities.filter(
                      (e: { name: string }) =>
                        !existing.has(e.name.toLowerCase()),
                    );
                    return [...prev, ...newEntities];
                  });
                }
                graphContextText = graphData.context_text || "";
              }
            }
          }
        } catch (graphError) {
          console.warn("Graph context retrieval failed:", graphError);
        } finally {
          setGraphLoading(false);
        }
      } else {
        setGraphEntities([]);
        setGraphRelationships([]);
        resetUnifiedContext();
      }

      if (
        isExplicitGraphRAGQuestion(outgoingText) ||
        isRequestedGraphRAGDocumentEvidenceQuestion(outgoingText)
      ) {
        const evidenceAnswer =
          buildGraphRAGDocumentEvidenceAnswer(graphRAGFallbackContext);
        if (evidenceAnswer) {
          const assistantMessage: ChatMessage = {
            role: "assistant",
            content: evidenceAnswer,
          };
          setMessages((prev: ChatMessage[]) => [...prev, assistantMessage]);
          return;
        }
      }

      if (!modelForRequest || modelForRequest === "AI Model") {
        if (hasGraphRAGHandoffContext) {
          const fallbackMessage: ChatMessage = {
            role: "assistant",
            content: buildGraphRAGDocumentFallbackAnswer(
              graphRAGFallbackContext,
              "error",
            ),
          };
          setMessages((prev: ChatMessage[]) => [...prev, fallbackMessage]);
          return;
        }
        throw new Error("No active chat model is loaded.");
      }

      // Filter out UI-only messages and prepare messages for LLM
      // This is critical: UI messages like "✅ Tool executed:" break the chat template
      const cleanedMessages = [...messages, requestUserMessage]
        .filter(
          (msg) =>
            !msg.content?.toString().includes("🔧 Executing tool:") &&
            !msg.content?.toString().includes("✅ Tool executed:"),
        )
        .map((msg) => {
          // Clean up message for API
          const cleanMsg: ChatMessage = {
            role: msg.role,
            content: msg.content,
          };

          // For assistant messages with tool_calls, content must be null (not empty string)
          if (
            msg.role === "assistant" &&
            msg.tool_calls &&
            msg.tool_calls.length > 0
          ) {
            cleanMsg.content = null as any; // llama.cpp requires null, not empty string
            cleanMsg.tool_calls = msg.tool_calls;
          }

          // Include tool_call_id for tool messages
          if (msg.role === "tool" && msg.tool_call_id) {
            cleanMsg.tool_call_id = msg.tool_call_id;
          }

          return cleanMsg;
        });

      // Prepare messages with reasoning level system prompt if needed
      let requestMessages = cleanedMessages;

      const rawContextText = ragContextText + graphContextText;
      const currentMessageTokens = estimateMessageTokens(requestUserMessage);
      const requestedOutputTokens = hasGraphRAGHandoffContext
        ? DOCUMENT_CHAT_OUTPUT_TOKENS
        : Math.min(
            settings.maxTokens || DEFAULT_CHAT_OUTPUT_TOKENS,
            MAX_CHAT_OUTPUT_TOKENS,
          );
      const maxKnowledgeContextChars = hasGraphRAGHandoffContext
        ? DOCUMENT_CHAT_CONTEXT_MAX_CHARS
        : MAX_KNOWLEDGE_CONTEXT_CHARS;
      const minKnowledgeContextChars = hasGraphRAGHandoffContext
        ? DOCUMENT_CHAT_CONTEXT_MIN_CHARS
        : MIN_KNOWLEDGE_CONTEXT_CHARS;
      const contextCharBudget = Math.min(
        maxKnowledgeContextChars,
        Math.max(
          minKnowledgeContextChars,
          Math.floor(
            (contextLimitForRequest -
              requestedOutputTokens -
              currentMessageTokens -
              CONTEXT_TOKEN_BUFFER) *
              3.5,
          ),
        ),
      );

      // Add or update system message with reasoning level, RAG context, and graph context
      const allContextText = compactKnowledgeContext(
        rawContextText,
        contextCharBudget,
      );
      requestMessages = applyKnowledgeSystemPrompt(requestMessages, {
        contextText: allContextText,
        reasoningLevel: settings.reasoningLevel,
        userQuery: outgoingText,
      });

      // Truncate messages to fit context window
      // Estimate tokens: Text ~3.5 chars/token, Image ~1000 tokens
      // Reserve output tokens plus a small server-side buffer.
      const reservedOutputTokens = Math.min(
        requestedOutputTokens,
        MAX_CHAT_OUTPUT_TOKENS,
      );

      // Keep the system prompt and always preserve the active user turn.
      // GraphRAG handoff prompts can be long while model metadata is still
      // hydrating; dropping the latest user message yields invalid chat
      // requests and a visibly dead handoff.
      if (requestMessages.length > 1) {
        const systemMsg = requestMessages[0];
        const systemTokens = estimateMessageTokens(systemMsg);
        const budget =
          contextLimitForRequest -
          reservedOutputTokens -
          CONTEXT_TOKEN_BUFFER -
          systemTokens;
        const latestMessage = requestMessages[requestMessages.length - 1];
        const latestMessageCost = estimateMessageTokens(latestMessage);

        let used = latestMessageCost;
        const keptMessages: ChatMessage[] = [latestMessage];

        // Iterate backwards through older messages only.
        for (let i = requestMessages.length - 2; i >= 1; i--) {
          const msg = requestMessages[i];
          const cost = estimateMessageTokens(msg);
          if (used + cost <= budget) {
            keptMessages.unshift(msg);
            used += cost;
          } else {
            debugLog(
              "chat",
              `Truncating history: dropping message with ~${cost} tokens. Budget remaining: ${budget - used}`,
            );
            break;
          }
        }

        if (latestMessageCost > budget) {
          debugLog(
            "chat",
            `Preserving latest user message with ~${latestMessageCost} tokens even though history budget is ${budget}.`,
          );
        }

        requestMessages = [systemMsg, ...keptMessages];
      }

      const safeMaxTokens = safeOutputTokenBudget(
        requestMessages,
        hasGraphRAGHandoffContext
          ? Math.max(settings.maxTokens, DOCUMENT_CHAT_OUTPUT_TOKENS)
          : settings.maxTokens,
        contextLimitForRequest,
      );
      const maxTokensForRequest = hasGraphRAGHandoffContext
        ? Math.min(safeMaxTokens, DOCUMENT_CHAT_OUTPUT_TOKENS)
        : safeMaxTokens;
      const responseShouldStream =
        settings.streamResponse && !hasGraphRAGHandoffContext;

      const request: ChatCompletionRequest = {
        messages: requestMessages,
        model: modelForRequest,
        temperature: settings.temperature,
        top_p: settings.topP,
        top_k: settings.topK,
        max_tokens: maxTokensForRequest,
        stream: responseShouldStream,
        chat_template_kwargs: { enable_thinking: false },
        ...(responseShouldStream
          ? { stream_options: { include_usage: true } }
          : {}),
        tools:
          selectedToolsForRequest.length > 0
            ? selectedToolsForRequest
            : undefined,
        tool_choice: selectedToolsForRequest.length > 0 ? "auto" : undefined,
        ...buildChatOutputConstraints(settings),
      };

      setChatBusyPhase("writing_answer");
      const nonStreamingResponseOptions: NonStreamingResponseOptions = {
        cleanGraphRAGAnswer: hasGraphRAGHandoffContext,
        graphRAGSourceLabel:
          activeGraphRAGChatHandoff?.filename ||
          getGraphRAGHandoffRunIds(activeGraphRAGChatHandoff)[0] ||
          "",
        documentFirstGraphRAGAnswer:
          hasGraphRAGHandoffContext && !isExplicitGraphRAGQuestion(outgoingText),
        ...(hasGraphRAGHandoffContext
          ? {
              timeoutMs: DOCUMENT_CHAT_MODEL_TIMEOUT_MS,
              fallbackAnswer: (reason: "timeout" | "error") =>
                buildGraphRAGDocumentFallbackAnswer(
                  graphRAGFallbackContext,
                  reason,
                ),
            }
          : {}),
      };

      if (responseShouldStream) {
        try {
          await handleStreamingResponse(request);
        } catch (streamError) {
          console.warn(
            "Streaming failed, falling back to non-streaming:",
            streamError,
          );
          // Fallback to non-streaming if streaming fails
          const nonStreamRequest = { ...request, stream: false };
          await handleNonStreamingResponse(
            nonStreamRequest,
            nonStreamingResponseOptions,
          );
        }
      } else {
        await handleNonStreamingResponse(request, nonStreamingResponseOptions);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send message");
      console.error("Chat error:", err);
      // Drop empty assistant placeholder left by a failed stream/non-stream attempt.
      setMessages((prev: ChatMessage[]) => {
        if (prev.length === 0) return prev;
        const last = prev[prev.length - 1];
        if (
          last.role === "assistant" &&
          !last.content &&
          !last.reasoning_content &&
          !(last.tool_calls && last.tool_calls.length > 0)
        ) {
          return prev.slice(0, -1);
        }
        return prev;
      });
    } finally {
      sendInFlightRef.current = false;
      setIsLoading(false);
      setChatBusyPhase("idle");
      // Auto-save after each assistant reply
      setTimeout(() => {
        saveConversationRef
          .current?.()
          .then(() => {
            setSaveStatus("saved");
            setTimeout(() => setSaveStatus("idle"), 2000);
          })
          .catch(() => setSaveStatus("error"));
      }, 300);
    }
  };

  useEffect(() => {
    if (!graphRAGChatHandoff?.autoSubmit || !graphRAGChatHandoff.prompt) {
      return;
    }
    if (isLoading) return;

    const runIds = getGraphRAGHandoffRunIds(graphRAGChatHandoff).join("|");
    const autoSubmitKey = `${runIds}:${graphRAGChatHandoff.prompt}`;
    if (autoSubmittedGraphRAGHandoffRef.current === autoSubmitKey) return;
    if (!claimGraphRAGAutoSubmit(autoSubmitKey)) return;
    autoSubmittedGraphRAGHandoffRef.current = autoSubmitKey;
    setInput("");
    void handleSendMessage(
      graphRAGChatHandoff.prompt,
      graphRAGChatHandoff.displayPrompt,
      graphRAGChatHandoff,
    );
  }, [
    graphRAGChatHandoff,
    handleSendMessage,
    isLoading,
  ]);

  // Custom API call function that uses chat-specific settings
  const createChatCompletionStream = async (
    request: ChatCompletionRequest,
  ): Promise<ReadableStream<any>> => {
    const authHeaders: Record<string, string> = isUsableChatApiKey(settings.apiKey)
      ? { Authorization: `Bearer ${settings.apiKey.trim()}` }
      : {};
    const fullUrl = settings.baseUrl
      ? `${settings.baseUrl.replace(/\/$/, "")}${settings.endpoint}`
      : settings.endpoint;
    const response = await fetch(fullUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
        "Cache-Control": "no-cache",
        ...authHeaders,
      },
      body: JSON.stringify(asStreamingChatRequest(request)),
    });

    if (!response.ok) {
      const detail = await readErrorDetail(response);
      throw new Error(
        `HTTP ${response.status}: ${response.statusText}${detail ? " — " + detail : ""}`,
      );
    }

    if (!response.body) {
      throw new Error("Response body is null");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    return new ReadableStream({
      start(controller) {
        let buffer = ""; // Buffer for incomplete SSE messages

        function pump(): Promise<void> {
          return reader
            .read()
            .then(({ done, value }) => {
              if (done) {
                controller.close();
                return;
              }

              // Append to buffer and parse complete SSE messages
              buffer += decoder.decode(value, { stream: true });

              // Split by double-newline (SSE message separator) or single newline for simple cases
              // SSE spec: messages are separated by blank lines
              const messages = buffer.split(/\n\n/);

              // Keep the last potentially incomplete message in buffer
              buffer = messages.pop() || "";

              for (const message of messages) {
                const lines = message.split("\n");
                let data = "";

                for (const line of lines) {
                  const trimmedLine = line.trim();
                  if (trimmedLine.startsWith("data: ")) {
                    // Accumulate data field (may be spread across lines in some servers)
                    data += trimmedLine.slice(6);
                  } else if (trimmedLine.startsWith("data:")) {
                    data += trimmedLine.slice(5);
                  }
                }

                if (data === "[DONE]") {
                  controller.close();
                  return;
                }

                if (data) {
                  try {
                    const parsed = JSON.parse(data);
                    controller.enqueue(parsed);
                  } catch (e) {
                    // Try to handle case where data was split across chunks
                    console.warn("Failed to parse SSE data:", data);
                  }
                }
              }

              return pump();
            })
            .catch((error) => {
              controller.error(error);
            });
        }
        return pump();
      },
      cancel(reason) {
        return reader.cancel(reason);
      },
    });
  };

  const createChatCompletion = async (
    request: ChatCompletionRequest,
    options: { timeoutMs?: number } = {},
  ): Promise<any> => {
    const authHeaders: Record<string, string> = isUsableChatApiKey(settings.apiKey)
      ? { Authorization: `Bearer ${settings.apiKey.trim()}` }
      : {};
    const fullUrl = settings.baseUrl
      ? `${settings.baseUrl.replace(/\/$/, "")}${settings.endpoint}`
      : settings.endpoint;
    const controller =
      options.timeoutMs && options.timeoutMs > 0 ? new AbortController() : null;
    const timeoutId = controller
      ? window.setTimeout(() => controller.abort(), options.timeoutMs)
      : null;
    let response: Response;
    try {
      response = await fetch(fullUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders,
        },
        body: JSON.stringify(asNonStreamingChatRequest(request)),
        signal: controller?.signal,
      });
    } catch (error) {
      if (controller?.signal.aborted) {
        const timeoutError = new Error(
          `Model endpoint timed out after ${Math.round((options.timeoutMs || 0) / 1000)}s.`,
        );
        timeoutError.name = "AbortError";
        throw timeoutError;
      }
      throw error;
    } finally {
      if (timeoutId !== null) window.clearTimeout(timeoutId);
    }

    if (!response.ok) {
      const detail = await readErrorDetail(response);
      throw new Error(
        `HTTP ${response.status}: ${response.statusText}${detail ? " — " + detail : ""}`,
      );
    }

    return response.json();
  };

  const handleStreamingResponse = async (request: ChatCompletionRequest) => {
    const assistantMessageId = createClientMessageId();
    const assistantMessage: LocalChatMessage = {
      role: "assistant",
      content: "",
      reasoning_content: "",
      tokensPerSecond: undefined,
      _clientMessageId: assistantMessageId,
    };
    setMessages((prev: ChatMessage[]) => [...prev, assistantMessage]);

    // Initialize performance tracking
    const startTime = performance.now();
    performanceRef.current = {
      ...initialPerformanceMetrics,
      startTime,
    };
    setPerformanceMetrics(performanceRef.current);

    try {
      const stream = await createChatCompletionStream(request);
      const reader = stream.getReader();

      try {
        // Tool calls are accumulated here across all deltas
        const accumulatedToolCalls: ToolCall[] = [];
        let tokenCount = 0;
        let firstTokenReceived = false;
        let toolCallsProcessed = false; // Flag to prevent processing tool calls multiple times

        const applyUsage = (usage?: {
          prompt_tokens?: number;
          completion_tokens?: number;
        }) => {
          if (!usage) return;
          if (usage.prompt_tokens != null) {
            performanceRef.current = {
              ...performanceRef.current,
              promptTokens: usage.prompt_tokens,
            };
          }
          if (usage.completion_tokens != null && usage.completion_tokens > 0) {
            tokenCount = usage.completion_tokens;
            performanceRef.current = {
              ...performanceRef.current,
              tokensGenerated: tokenCount,
            };
          }
        };

        const recordGenerationDelta = (deltaText: string) => {
          if (!deltaText) return;
          if (!firstTokenReceived) {
            firstTokenReceived = true;
            const firstTokenTime = performance.now();
            performanceRef.current = {
              ...performanceRef.current,
              firstTokenTime,
              timeToFirstToken: firstTokenTime - startTime,
            };
            setPerformanceMetrics({ ...performanceRef.current });
          }

          tokenCount += estimateDeltaTokens(deltaText);
          const currentTime = performance.now();
          const tps = computeGenerationTps(
            tokenCount,
            startTime,
            performanceRef.current.firstTokenTime,
            currentTime,
          );
          performanceRef.current = {
            ...performanceRef.current,
            tokensGenerated: tokenCount,
            tokensPerSecond: tps,
          };
          setPerformanceMetrics({ ...performanceRef.current });

          setMessages((prev: ChatMessage[]) => {
            if (tps == null) return prev;
            return updateAssistantMessageByClientId(
              prev,
              assistantMessageId,
              (message) => ({
                ...message,
                tokensPerSecond: tps,
              }),
            );
          });
        };

        // TTS sentence chunking - send sentences as they complete for lower latency
        let ttsSentenceBuffer = "";
        const shouldStreamTTS =
          settings.voiceModeEnabled &&
          settings.ttsEnabled &&
          ttsServiceAvailable;

        // Function to extract and speak complete sentences
        const processTTSBuffer = (forceFlush: boolean = false) => {
          if (!shouldStreamTTS || !ttsSentenceBuffer.trim()) return;

          // Look for sentence boundaries: . ! ? followed by space or end
          // Also split on newlines for code/list responses
          const sentenceBreakRegex = /([.!?])\s+|(\n\n)/g;

          let lastIndex = 0;
          let match;

          while (
            (match = sentenceBreakRegex.exec(ttsSentenceBuffer)) !== null
          ) {
            // Extract the complete sentence including the punctuation
            const sentence = ttsSentenceBuffer
              .slice(lastIndex, match.index + 1)
              .trim();
            if (sentence.length > 0) {
              tts.speak(sentence);
            }
            lastIndex = match.index + match[0].length;
          }

          // Keep the remainder (incomplete sentence) in the buffer
          ttsSentenceBuffer = ttsSentenceBuffer.slice(lastIndex);

          // If forcing flush (end of stream), speak any remaining content
          if (forceFlush && ttsSentenceBuffer.trim().length > 0) {
            tts.speak(ttsSentenceBuffer.trim());
            ttsSentenceBuffer = "";
          }
        };

        type StreamReadResult = ReadableStreamReadResult<any> & {
          idleTimeout?: boolean;
        };

        for (;;) {
          const readTimeoutMs = firstTokenReceived
            ? STREAM_IDLE_COMPLETE_MS
            : STREAM_FIRST_TOKEN_TIMEOUT_MS;
          const readResult = await new Promise<StreamReadResult>(
            (resolve, reject) => {
              const timeoutId = window.setTimeout(
                () =>
                  resolve({
                    done: true,
                    value: undefined,
                    idleTimeout: true,
                  }),
                readTimeoutMs,
              );
              reader
                .read()
                .then((result) => {
                  window.clearTimeout(timeoutId);
                  resolve({ ...result, idleTimeout: false });
                })
                .catch((error) => {
                  window.clearTimeout(timeoutId);
                  reject(error);
                });
            },
          );
          const { done, value, idleTimeout } = readResult;
          if (idleTimeout) {
            debugLog(
              "chat",
              `Stream finalized after ${readTimeoutMs}ms without new tokens`,
            );
            try {
              await reader.cancel("stream idle completion timeout");
            } catch (cancelError) {
              console.warn("Failed to cancel idle stream:", cancelError);
            }
            if (!firstTokenReceived) {
              throw new Error("Timed out waiting for the first streamed token");
            }
          }
          applyUsage(value?.usage);

          if (done || idleTimeout) {
            debugLog(
              "chat",
              idleTimeout ? "Stream ended after idle timeout" : "Stream ended naturally",
            );
            // Finalize performance metrics (generation phase, not including prefill)
            const endTime = performance.now();
            const finalTps = computeGenerationTps(
              tokenCount,
              startTime,
              performanceRef.current.firstTokenTime,
              endTime,
            );
            performanceRef.current = {
              ...performanceRef.current,
              endTime,
              tokensGenerated: tokenCount,
              tokensPerSecond: finalTps,
            };
            setPerformanceMetrics({ ...performanceRef.current });

            if (finalTps != null) {
              setMessages((prev: ChatMessage[]) => {
                return updateAssistantMessageByClientId(
                  prev,
                  assistantMessageId,
                  (message) => ({
                    ...message,
                    tokensPerSecond: finalTps,
                  }),
                );
              });
            }

            // Flush any remaining TTS content
            processTTSBuffer(true);

            // Process tool calls when stream ends (not when finish_reason is received)
            // llama.cpp sends finish_reason before all tool data is accumulated
            if (accumulatedToolCalls.length > 0 && !toolCallsProcessed) {
              toolCallsProcessed = true;
              debugLog(
                "chat",
                "Stream ended with accumulated tool calls",
                accumulatedToolCalls,
              );

              let messagesSnapshot: ChatMessage[] = [];
              flushSync(() => {
                setMessages((prev: ChatMessage[]) => {
                  const updatedMessages = updateAssistantMessageByClientId(
                    prev,
                    assistantMessageId,
                    (message) => ({
                      ...message,
                      tool_calls: accumulatedToolCalls,
                    }),
                  );
                  messagesSnapshot = [...updatedMessages];
                  return updatedMessages;
                });
              });

              await handleToolCalls(accumulatedToolCalls, messagesSnapshot);
            }
            break;
          }

          debugLog("chat", "Received stream chunk", value);
          debugLog("chat", "Delta object", value.choices?.[0]?.delta);

          // Only delta.content (string). Do not use __verbose.content — it can append timing lines to the assistant bubble.
          const contentDelta = normalizeDeltaContent(
            value.choices?.[0]?.delta?.content,
          );

          // Debug: log if we're getting reasoning_content
          if (
            value.choices?.[0]?.delta?.reasoning_content ||
            value.choices?.[0]?.delta?.reasoning
          ) {
            debugLog(
              "chat",
              "Reasoning content received",
              value.choices[0].delta.reasoning_content ||
                value.choices[0].delta.reasoning,
            );
          }

          if (contentDelta) {
            recordGenerationDelta(contentDelta);

            if (shouldStreamTTS) {
              ttsSentenceBuffer += contentDelta;
              processTTSBuffer(false);
            }

            setMessages((prev: ChatMessage[]) => {
              return updateAssistantMessageByClientId(
                prev,
                assistantMessageId,
                (message) => ({
                  ...message,
                  content:
                    (typeof message.content === "string"
                      ? message.content
                      : stringifyChatContent(message.content)) + contentDelta,
                }),
              );
            });
          }

          // Extract timing information if available (more accurate from server)
          const timings = value.timings || value.__verbose?.timings;
          if (timings != null && timings.predicted_per_second != null) {
            const tpsServer = sanitizeTokensPerSecond(
              timings.predicted_per_second,
            );
            performanceRef.current = {
              ...performanceRef.current,
              tokensPerSecond: tpsServer ?? null,
            };
            setPerformanceMetrics({ ...performanceRef.current });

            setMessages((prev: ChatMessage[]) => {
              return updateAssistantMessageByClientId(
                prev,
                assistantMessageId,
                (message) => {
                  const updatedMessage = { ...message };
                  if (tpsServer != null) {
                    updatedMessage.tokensPerSecond = tpsServer;
                  } else {
                    delete updatedMessage.tokensPerSecond;
                  }
                  return updatedMessage;
                },
              );
            });
          }

          // Extract prompt processing info if available
          if (timings?.prompt_per_second) {
            performanceRef.current = {
              ...performanceRef.current,
              promptTokens:
                timings.prompt_n || performanceRef.current.promptTokens,
            };
            setPerformanceMetrics({ ...performanceRef.current });
          }

          // Handle reasoning/thinking content (DeepSeek R1, QwQ, Qwen3, Nemotron, etc.)
          const reasoningDelta =
            value.choices?.[0]?.delta?.reasoning_content ||
            value.choices?.[0]?.delta?.reasoning;
          if (reasoningDelta) {
            // If there is no normal content delta, treat reasoning as the streamed response
            // (e.g. Qwen3 puts ALL output in reasoning_content)
            if (!contentDelta) {
              recordGenerationDelta(reasoningDelta);

              if (shouldStreamTTS) {
                ttsSentenceBuffer += reasoningDelta;
                processTTSBuffer(false);
              }
            }

            setMessages((prev: ChatMessage[]) => {
              return updateAssistantMessageByClientId(
                prev,
                assistantMessageId,
                (message) => ({
                  ...message,
                  reasoning_content:
                    (message.reasoning_content || "") + reasoningDelta,
                }),
              );
            });
          }

          if (value.choices?.[0]?.delta?.tool_calls) {
            // Tool calls arrive as deltas - accumulate into a single tool call
            // SIMPLE APPROACH: We only support one tool call at a time,
            // accumulate all deltas into index 0
            debugLog(
              "chat",
              "Tool call delta received",
              value.choices[0].delta.tool_calls,
            );

            for (const delta of value.choices[0].delta.tool_calls) {
              // Initialize the single tool call if not exists
              if (accumulatedToolCalls.length === 0) {
                accumulatedToolCalls[0] = {
                  id: delta.id || `call_${Date.now()}`,
                  type: "function",
                  function: {
                    name: "",
                    arguments: "",
                  },
                };
                debugLog("chat", "Initialized tool call accumulator");
              }

              // Always accumulate into the first (and only) tool call
              if (delta.function?.name) {
                accumulatedToolCalls[0].function.name += delta.function.name;
                debugLog("chat", "Added to tool name", delta.function.name);
              }
              if (delta.function?.arguments) {
                accumulatedToolCalls[0].function.arguments +=
                  delta.function.arguments;
                debugLog(
                  "chat",
                  "Added to tool args",
                  delta.function.arguments,
                );
              }
              // Capture ID if provided and we don't have one yet
              if (delta.id && !accumulatedToolCalls[0].id.startsWith("call_")) {
                accumulatedToolCalls[0].id = delta.id;
              }
            }
            debugLog(
              "chat",
              "Current accumulated tool call",
              accumulatedToolCalls[0],
            );
          }

          // Note: Tool calls are now processed at stream end (when done === true)
          // rather than when finish_reason === 'tool_calls' because llama.cpp
          // sends finish_reason before all tool data is fully accumulated
        }
      } finally {
        reader.releaseLock();

        // After streaming completes, if content is empty but reasoning_content exists,
        // use reasoning_content as the visible content (for models that put all output in thinking)
        setMessages((prev: ChatMessage[]) => {
          return updateAssistantMessageByClientId(
            prev,
            assistantMessageId,
            (message) => {
              if (
                !stringifyChatContent(message.content).trim() &&
                message.reasoning_content
              ) {
                return {
                  ...message,
                  content: message.reasoning_content,
                };
              }
              return message;
            },
          );
        });
      }
    } catch (streamError) {
      console.error("Streaming error:", streamError);
      throw streamError; // Re-throw to trigger fallback
    }
  };

  const handleNonStreamingResponse = async (
    request: ChatCompletionRequest,
    options: NonStreamingResponseOptions = {},
  ) => {
    const startTime = performance.now();
    let response: any;
    try {
      response = await createChatCompletion(request, {
        timeoutMs: options.timeoutMs,
      });
    } catch (error) {
      if (options.fallbackAnswer) {
        const reason =
          error instanceof Error && error.name === "AbortError"
            ? "timeout"
            : "error";
        console.warn("Using GraphRAG document answer fallback:", error);
        const fallbackContent = options.fallbackAnswer(reason);
        const fallbackMessage: ChatMessage = {
          role: "assistant",
          content: fallbackContent,
        };
        setMessages((prev: ChatMessage[]) => [...prev, fallbackMessage]);
        return;
      }
      throw error;
    }
    const endTime = performance.now();
    const usage = response.usage;
    const completionTokens = usage?.completion_tokens ?? 0;
    const finalTps = computeGenerationTps(
      completionTokens,
      startTime,
      startTime,
      endTime,
    );
    if (completionTokens > 0 || finalTps != null) {
      performanceRef.current = {
        ...performanceRef.current,
        startTime,
        endTime,
        firstTokenTime: startTime,
        timeToFirstToken: 0,
        tokensGenerated: completionTokens,
        tokensPerSecond: finalTps,
        promptTokens:
          usage?.prompt_tokens ?? performanceRef.current.promptTokens,
      };
      setPerformanceMetrics({ ...performanceRef.current });
    }

    const assistantMessage: ChatMessage = {
      ...response.choices[0].message,
      // Include reasoning_content if present (also check 'reasoning' field from vLLM)
      reasoning_content:
        response.choices[0].message.reasoning_content ||
        response.choices[0].message.reasoning,
      ...(finalTps != null ? { tokensPerSecond: finalTps } : {}),
    };

    // If content is empty but reasoning_content exists, use it as the visible content
    // This handles models (e.g. Qwen3) that put ALL output in reasoning_content
    if (!assistantMessage.content && assistantMessage.reasoning_content) {
      assistantMessage.content = assistantMessage.reasoning_content;
    }

    if (
      options.cleanGraphRAGAnswer &&
      typeof assistantMessage.content === "string"
    ) {
      assistantMessage.content = cleanGraphRAGAssistantAnswer(
        assistantMessage.content,
        {
          sourceLabel: options.graphRAGSourceLabel,
          documentFirst: options.documentFirstGraphRAGAnswer,
        },
      );
    }

    // For tool calls, we need to capture the snapshot synchronously
    // to pass to handleToolCalls (React state updates are async)
    let messagesSnapshot: ChatMessage[] | null = null;

    setMessages((prev: ChatMessage[]) => {
      const newMessages = [...prev, assistantMessage];
      // Capture snapshot if there are tool calls
      if (
        assistantMessage.tool_calls &&
        assistantMessage.tool_calls.length > 0
      ) {
        messagesSnapshot = newMessages;
      }
      return newMessages;
    });

    // Trigger TTS for the response
    const responseText =
      typeof assistantMessage.content === "string"
        ? assistantMessage.content
        : "";
    if (
      responseText.trim() &&
      settings.voiceModeEnabled &&
      settings.ttsEnabled &&
      ttsServiceAvailable
    ) {
      tts.speak(responseText.trim());
    }

    // Handle tool calls if present
    if (
      assistantMessage.tool_calls &&
      assistantMessage.tool_calls.length > 0 &&
      messagesSnapshot
    ) {
      await handleToolCalls(assistantMessage.tool_calls, messagesSnapshot);
    }
  };

  const handleToolCalls = async (
    toolCalls: ToolCall[],
    messagesSnapshot?: ChatMessage[],
  ) => {
    // Use provided snapshot or capture current messages from state
    let currentMessages: ChatMessage[] = messagesSnapshot || [];

    if (!messagesSnapshot) {
      // Fallback: Get current messages state for building the follow-up request
      setMessages((prev: ChatMessage[]) => {
        currentMessages = [...prev];
        return prev; // Don't modify, just capture
      });
    }

    for (const toolCall of toolCalls) {
      try {
        // Show tool execution indicator
        const toolMessage: ChatMessage = {
          role: "assistant",
          content: `🔧 Executing tool: ${toolCall.function.name}...`,
        };
        setMessages((prev: ChatMessage[]) => [...prev, toolMessage]);

        // Execute the tool
        const result = await ToolsService.executeToolCall(toolCall);

        // Add tool result as a tool message
        const toolResultMessage: ChatMessage = {
          role: "tool",
          content: result,
          tool_call_id: toolCall.id,
        };

        // Update the tool execution message with result and add tool result
        setMessages((prev: ChatMessage[]) => {
          const newMessages = [...prev];
          const lastMessage = newMessages[newMessages.length - 1];
          if (
            typeof lastMessage.content === "string" &&
            lastMessage.content.includes("🔧 Executing tool:")
          ) {
            lastMessage.content = `✅ Tool executed: ${toolCall.function.name}\n\nResult:\n${result}`;
          }
          return [...newMessages, toolResultMessage];
        });

        // Build proper message history for the LLM:
        // 1. Include all messages up to and including the assistant's tool_call message
        // 2. Add the tool result message
        // The chat template expects: [user msg] -> [assistant with tool_calls] -> [tool result]

        // Filter out UI-only messages (tool execution indicators)
        const messagesForLLM = currentMessages
          .filter(
            (msg) =>
              !msg.content?.toString().includes("🔧 Executing tool:") &&
              !msg.content?.toString().includes("✅ Tool executed:"),
          )
          .map((msg) => {
            // Clean up message for API - remove UI-only properties
            const cleanMsg: ChatMessage = {
              role: msg.role,
              content: msg.content,
            };

            // For assistant messages with tool_calls, content must be null (not empty string)
            if (
              msg.role === "assistant" &&
              msg.tool_calls &&
              msg.tool_calls.length > 0
            ) {
              cleanMsg.content = null as any; // llama.cpp requires null, not empty string
              cleanMsg.tool_calls = msg.tool_calls;
            }

            // Include tool_call_id for tool messages
            if (msg.role === "tool" && msg.tool_call_id) {
              cleanMsg.tool_call_id = msg.tool_call_id;
            }

            return cleanMsg;
          });

        // Add the tool result message and ensure reasoning level is set for follow-up
        let followUpMessages: ChatMessage[] = [
          ...messagesForLLM,
          toolResultMessage,
        ];

        followUpMessages = applyKnowledgeSystemPrompt(followUpMessages, {
          reasoningLevel: settings.reasoningLevel,
        });

        // Continue conversation with tool result
        const followUpRequest: ChatCompletionRequest = {
          messages: followUpMessages,
          model: settings.model,
          temperature: settings.temperature,
          top_p: settings.topP,
          top_k: settings.topK,
          max_tokens: safeOutputTokenBudget(
            followUpMessages,
            settings.maxTokens,
            contextLimit,
          ),
          stream: false, // Use non-streaming for tool follow-ups
          chat_template_kwargs: { enable_thinking: false },
        };

        debugLog("chat", "Tool follow-up request", {
          messages: followUpMessages,
          request: followUpRequest,
        });

        const followUpResponse = await createChatCompletion(followUpRequest);
        debugLog("chat", "Tool follow-up response", {
          response: followUpResponse,
          messageToAdd: followUpResponse.choices[0].message,
        });

        const responseMessage = followUpResponse.choices[0].message;
        // If content is empty but reasoning_content exists, use it as the visible content
        // This handles models that put all output in reasoning/thinking tokens
        if (!responseMessage.content && responseMessage.reasoning_content) {
          responseMessage.content = responseMessage.reasoning_content;
        }
        setMessages((prev: ChatMessage[]) => [...prev, responseMessage]);

        // Update currentMessages for the next iteration
        currentMessages = [
          ...currentMessages,
          toolResultMessage,
          followUpResponse.choices[0].message,
        ];
      } catch (error) {
        const errorMessage: ChatMessage = {
          role: "assistant",
          content: `❌ Error executing tool ${toolCall.function.name}: ${error instanceof Error ? error.message : "Unknown error"}`,
        };
        setMessages((prev: ChatMessage[]) => [...prev, errorMessage]);
      }
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Global keyboard shortcuts for chat
  useEffect(() => {
    const handleGlobalKeyDown = (e: globalThis.KeyboardEvent) => {
      // Cmd/Ctrl + L: Clear chat
      if ((e.metaKey || e.ctrlKey) && e.key === "l") {
        e.preventDefault();
        clearChat();
      }
      // Cmd/Ctrl + /: Toggle settings
      if ((e.metaKey || e.ctrlKey) && e.key === "/") {
        e.preventDefault();
        setShowSettings((prev) => !prev);
      }
      // Cmd/Ctrl + N: New conversation
      if ((e.metaKey || e.ctrlKey) && e.key === "n") {
        e.preventDefault();
        handleNewConversation();
      }
      // Cmd/Ctrl + S: Save conversation
      if ((e.metaKey || e.ctrlKey) && e.key === "s") {
        e.preventDefault();
        if (hasUnsavedChanges) {
          saveConversation();
        }
      }
      // Escape: Stop TTS / Close settings
      if (e.key === "Escape") {
        if (tts.isPlaying) {
          tts.stop();
        } else if (showSettings) {
          setShowSettings(false);
        }
      }
    };

    document.addEventListener("keydown", handleGlobalKeyDown);
    return () => document.removeEventListener("keydown", handleGlobalKeyDown);
  }, [hasUnsavedChanges, showSettings, tts.isPlaying]);

  const clearChat = () => {
    setCurrentConversationId(null);
    setConversationTitle("New Conversation");
    setMessages([]);
    setHasUnsavedChanges(false);
    setSaveStatus("idle");
    setError(null);
    setRagContext([]);
    setGraphEntities([]);
    setGraphRelationships([]);
    resetUnifiedContext();
  };

  const copyToClipboard = (
    content:
      | string
      | Array<{ type: string; text?: string; image_url?: { url: string } }>,
  ) => {
    let text = "";
    if (typeof content === "string") {
      text = content;
    } else if (Array.isArray(content)) {
      // Extract text parts from multi-modal content
      text = content
        .filter((part: any) => part.type === "text" && part.text)
        .map((part: any) => part.text)
        .join("\n");
    }
    copyTextToClipboard(text, {
      successMessage: "Message content copied.",
      successTitle: "Message copied",
    });
  };

  // Start editing a message
  const startEditingMessage = (index: number) => {
    const message = messages[index];
    if (typeof message.content === "string") {
      setEditingContent(message.content);
    } else if (Array.isArray(message.content)) {
      // Extract text from multi-modal content
      const textParts = message.content
        .filter((part: any) => part.type === "text" && part.text)
        .map((part: any) => part.text);
      setEditingContent(textParts.join("\n"));
    }
    setEditingMessageIndex(index);
  };

  // Cancel editing
  const cancelEditingMessage = () => {
    setEditingMessageIndex(null);
    setEditingContent("");
  };

  // Save edited message and regenerate response
  const saveEditedMessage = async (index: number) => {
    if (!editingContent.trim()) return;

    // Update the message at the given index
    const updatedMessages = messages.slice(0, index + 1);
    updatedMessages[index] = {
      ...updatedMessages[index],
      content: editingContent.trim(),
    };

    // Remove all messages after the edited one (will regenerate)
    setMessages(updatedMessages);
    setEditingMessageIndex(null);
    setEditingContent("");
    setHasUnsavedChanges(true);

    // Regenerate the response
    setIsLoading(true);
    setChatBusyPhase("writing_answer");
    setError(null);

    try {
      const selectedToolsForRequest = settings.enableTools
        ? availableTools.filter((tool: Tool) =>
            settings.selectedTools.includes(tool.function.name),
          )
        : [];

      // Prepare messages with reasoning level system prompt if needed
      const requestMessages = applyKnowledgeSystemPrompt(updatedMessages, {
        reasoningLevel: settings.reasoningLevel,
      });

      const request: ChatCompletionRequest = {
        messages: requestMessages,
        model: settings.model,
        temperature: settings.temperature,
        top_p: settings.topP,
        top_k: settings.topK,
        max_tokens: safeOutputTokenBudget(
          requestMessages,
          settings.maxTokens,
          contextLimit,
        ),
        stream: settings.streamResponse,
        chat_template_kwargs: { enable_thinking: false },
        ...(settings.streamResponse
          ? { stream_options: { include_usage: true } }
          : {}),
        tools:
          selectedToolsForRequest.length > 0
            ? selectedToolsForRequest
            : undefined,
        tool_choice: selectedToolsForRequest.length > 0 ? "auto" : undefined,
        ...buildChatOutputConstraints(settings),
      };

      if (settings.streamResponse) {
        await handleStreamingResponse(request);
      } else {
        await handleNonStreamingResponse(request);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to regenerate response",
      );
    } finally {
      setIsLoading(false);
      setChatBusyPhase("idle");
    }
  };

  const formatRole = (role: string) => {
    switch (role) {
      case "user":
        return "You";
      case "assistant":
        return "Assistant";
      case "tool":
        return "Tool Result";
      case "system":
        return "System";
      default:
        return role;
    }
  };

  const getRoleIcon = (role: string) => {
    switch (role) {
      case "user":
        return <UserIcon />;
      case "assistant":
        return <BotIcon />;
      case "tool":
        return <ToolIcon />;
      default:
        return <BotIcon />;
    }
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case "user":
        return "primary";
      case "assistant":
        return "secondary";
      case "tool":
        return "info";
      default:
        return "secondary";
    }
  };

  const handleApplyConstraints = (schema: object, grammar?: string) => {
    const record = schema as Record<string, unknown>;
    saveSettings({
      ...settings,
      enableStructuredOutput: true,
      structuredOutputSchema: record,
      structuredOutputSchemaName:
        typeof record.title === "string" ? record.title : "response",
      structuredOutputGrammar: grammar?.trim() ? grammar : null,
    });
  };

  const toggleStructuredOutput = () => {
    const next = !settings.enableStructuredOutput;
    if (next && !settings.structuredOutputSchema) {
      setConstraintEditorOpen(true);
      return;
    }
    saveSettings({ ...settings, enableStructuredOutput: next });
  };

  const clearStructuredOutput = () => {
    saveSettings({
      ...settings,
      enableStructuredOutput: false,
      structuredOutputSchema: null,
      structuredOutputGrammar: null,
      structuredOutputSchemaName: "response",
    });
  };

  const toggleToolSelection = (toolName: string) => {
    const newSelectedTools = settings.selectedTools.includes(toolName)
      ? settings.selectedTools.filter((name: string) => name !== toolName)
      : [...settings.selectedTools, toolName];

    saveSettings({ ...settings, selectedTools: newSelectedTools });
  };

  // Image upload handlers
  const handleImageUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const newImages: Array<{ file: File; preview: string; base64: string }> =
      [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];

      // Validate file type
      if (!file.type.startsWith("image/")) {
        setError(`File ${file.name} is not an image`);
        continue;
      }

      // Validate file size (max 20MB)
      if (file.size > 20 * 1024 * 1024) {
        setError(`File ${file.name} is too large (max 20MB)`);
        continue;
      }

      // Create preview URL
      const preview = URL.createObjectURL(file);

      // Convert to base64
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result as string);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      newImages.push({ file, preview, base64 });
    }

    setUploadedImages((prev) => [...prev, ...newImages]);

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleRemoveImage = (index: number) => {
    setUploadedImages((prev) => {
      const updated = [...prev];
      // Revoke the preview URL to free memory
      URL.revokeObjectURL(updated[index].preview);
      updated.splice(index, 1);
      return updated;
    });
  };

  // Audio recording handlers
  const startRecording = async () => {
    try {
      setAudioError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });
        await transcribeAudio(audioBlob);

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setShowAudioDialog(true);
    } catch (err) {
      console.error("Failed to start recording:", err);
      setAudioError("Failed to access microphone. Please check permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const cancelRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      audioChunksRef.current = [];
    }
    setShowAudioDialog(false);
    setAudioError(null);
  };

  const transcribeAudio = async (audioBlob: Blob) => {
    setIsTranscribing(true);
    try {
      // Convert webm to a format OpenAI accepts (mp3, mp4, mpeg, mpga, m4a, wav, or webm)
      const formData = new FormData();
      formData.append("file", audioBlob, "audio.webm");
      formData.append("model", "whisper-1");

      // Get OpenAI API key from settings
      const apiKey =
        settings.openaiApiKey || localStorage.getItem("openai-api-key") || "";

      if (!apiKey) {
        throw new Error(
          "OpenAI API key not configured. Please add it in Chat Settings.",
        );
      }

      const response = await fetch(
        "https://api.openai.com/v1/audio/transcriptions",
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${apiKey}`,
          },
          body: formData,
        },
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(
          errorData.error?.message ||
            `Transcription failed: ${response.statusText}`,
        );
      }

      const data = await response.json();
      const transcription = data.text;

      // Append transcription to input
      setInput((prev) => (prev ? `${prev} ${transcription}` : transcription));
      setShowAudioDialog(false);
      setAudioError(null);
    } catch (err) {
      console.error("Transcription error:", err);
      setAudioError(
        err instanceof Error ? err.message : "Transcription failed",
      );
    } finally {
      setIsTranscribing(false);
    }
  };

  const contextUsagePercent =
    contextLimit > 0
      ? Math.min((tokenCount.prompt / contextLimit) * 100, 100)
      : 0;
  const contextColor =
    tokenCount.prompt > contextLimit * 0.9
      ? accentColors.rose
      : tokenCount.prompt > contextLimit * 0.7
        ? accentColors.warning
        : accentColors.success;
  const backendLabel = activeInferenceBackend
    ? activeInferenceBackend === "vllm"
      ? "vLLM"
      : "llama.cpp"
    : settings.baseUrl
      ? "Custom endpoint"
      : "Backend proxy";
  const retrievalLabel =
    settings.ragEnabled && settings.graphEnabled
      ? "Hybrid"
      : settings.ragEnabled
        ? "RAG"
        : settings.graphEnabled
          ? "Graph"
          : "Off";
  const mediaLabel =
    uploadedImages.length > 0
      ? `${uploadedImages.length} image${uploadedImages.length === 1 ? "" : "s"}`
      : settings.voiceModeEnabled
        ? "Voice mode"
        : "Text";
  const normalizedRuntimeBackend =
    activeInferenceBackend === "llamacpp" || activeInferenceBackend === "vllm"
      ? activeInferenceBackend
      : null;
  const activeDeploymentContext: DeploymentRouteContext | null =
    deploymentHandoff ??
    (normalizedRuntimeBackend && currentModelName !== "AI Model"
      ? {
          backend: normalizedRuntimeBackend,
          modelName: currentModelName,
          endpoint: "Backend proxy",
        }
      : null);
  const navigateToDeploy = useCallback(() => {
    if (normalizedRuntimeBackend) {
      navigate(deployPath, { state: { backend: normalizedRuntimeBackend } });
      return;
    }

    navigate(deployPath);
  }, [deployPath, navigate, normalizedRuntimeBackend]);
  const runtimeEndpointLabel =
    activeDeploymentContext?.endpoint || "Backend proxy";
  const sessionSummary = `${currentModelName} · ${tokenCount.prompt.toLocaleString()} / ${contextLimit.toLocaleString()} ctx`;
  const graphRAGPipelineRunIds = getGraphRAGHandoffRunIds(graphRAGChatHandoff);
  const graphRAGTraceLabel =
    graphRAGPipelineRunIds.length > 1
      ? `${graphRAGPipelineRunIds.length} documents`
      : graphRAGChatHandoff?.filename || graphRAGPipelineRunIds[0] || "";
  const graphRAGHandoffQuestions = (
    Array.isArray(graphRAGChatHandoff?.suggestedQuestions)
      ? graphRAGChatHandoff?.suggestedQuestions || []
      : []
  )
    .filter((question) => question && question.prompt && question.label)
    .slice(0, 4);
  const graphRAGHandoffCapabilityCards = (
    Array.isArray(graphRAGChatHandoff?.capabilityCards)
      ? graphRAGChatHandoff?.capabilityCards || []
      : []
  )
    .filter((card) => card && card.label)
    .slice(0, 5);
  const graphRAGHandoffMeta = [
    graphRAGChatHandoff?.recommendedMode
      ? formatGraphRAGLabel(graphRAGChatHandoff.recommendedMode)
      : "",
    typeof graphRAGChatHandoff?.graphAnchorCount === "number"
      ? `${graphRAGChatHandoff.graphAnchorCount} anchors`
      : "",
    typeof graphRAGChatHandoff?.retrievalRehearsalCount === "number"
      ? `${graphRAGChatHandoff.retrievalRehearsalCount} rehearsals`
      : "",
  ].filter(Boolean);
  const graphRAGContextRequest = graphRAGChatHandoff?.chatContextRequest;
  const graphRAGContextFeatureChips = [
    graphRAGContextRequest?.include_graph === false ? "" : "Graph",
    graphRAGContextRequest?.include_local_pipeline === false &&
    graphRAGContextRequest?.use_local_pipeline_context === false
      ? ""
      : "Document sources",
    graphRAGPipelineRunIds.length > 1
      ? `${graphRAGPipelineRunIds.length} documents`
      : "",
    graphRAGChatHandoff?.domain ? `${graphRAGChatHandoff.domain} domain` : "",
  ].filter(Boolean);
  const graphRAGPromptPreview = graphRAGChatHandoff?.prompt?.trim() || "";
  const graphRAGDefaultPrompt =
    graphRAGPromptPreview ||
    graphRAGHandoffQuestions[0]?.prompt ||
    "Summarize the strongest document evidence and graph relationships.";
  const graphRAGPrimaryAskLabel =
    graphRAGPipelineRunIds.length > 1 ? "Ask across documents" : "Ask this document";
  const graphRAGSecondaryQuestions = graphRAGHandoffQuestions.filter(
    (question) => question.prompt !== graphRAGDefaultPrompt,
  );
  const sessionDetailItems = [
    `Model: ${currentModelName}`,
    `Context: ${tokenCount.prompt.toLocaleString()} / ${contextLimit.toLocaleString()} (${contextUsagePercent.toFixed(1)}%)`,
    `Tools: ${settings.enableTools ? `${settings.selectedTools.length || availableTools.length} available` : "off"}`,
    `Sources: ${retrievalLabel}`,
    graphRAGPipelineRunIds.length
      ? `Document context: ${graphRAGTraceLabel}`
      : "",
    `Input: ${mediaLabel}`,
    `Response: ${settings.streamResponse ? "streaming" : "complete response"}`,
    `Runtime: ${
      activeDeploymentContext
        ? `${runtimeBackendLabel(activeDeploymentContext.backend)} / ${runtimeEndpointLabel}`
        : backendLabel
    }`,
  ].filter(Boolean);
  const activeSessionChips = [
    settings.enableTools
      ? {
          key: "tools",
          status: "tools",
          label: `${settings.selectedTools.length || availableTools.length} tools`,
        }
      : null,
    retrievalLabel !== "Off"
      ? { key: "sources", status: "structured", label: retrievalLabel }
      : null,
    graphRAGPipelineRunIds.length
      ? {
          key: "graphrag-pipeline",
          status: "pipeline",
          label:
            graphRAGPipelineRunIds.length > 1
              ? "Document set"
              : "Document",
        }
      : null,
    mediaLabel !== "Text"
      ? { key: "media", status: "runtime", label: mediaLabel }
      : null,
    activeDeploymentContext
      ? {
          key: "runtime",
          status: "runtime",
          label: runtimeBackendLabel(activeDeploymentContext.backend),
        }
      : settings.baseUrl
        ? { key: "runtime", status: "runtime", label: "Custom endpoint" }
        : null,
  ].filter(Boolean) as Array<{ key: string; status: string; label: string }>;
  const saveStatusChipStatus = hasUnsavedChanges
    ? "unsaved"
    : saveStatus === "saved"
      ? "saved"
      : "draft";
  const showSaveStateChip = hasUnsavedChanges || saveStatus === "saving";
  const showReasoningChip =
    settings.reasoningLevel !== "low" && settings.reasoningLevel !== "none";
  const showContextUsageBar = contextUsagePercent >= 5;
  const generationTps = sanitizeTokensPerSecond(
    performanceMetrics.tokensPerSecond ?? undefined,
  );
  const compactMetricChips = settings.showPerformanceMetrics
    ? [
        {
          key: "generation",
          label:
            generationTps != null
              ? `${generationTps.toFixed(1)} tok/s`
              : "-- tok/s",
        },
        {
          key: "ttft",
          label: performanceMetrics.timeToFirstToken
            ? `${performanceMetrics.timeToFirstToken.toFixed(0)} ms TTFT`
            : "-- ms TTFT",
        },
        {
          key: "tokens",
          label:
            performanceMetrics.tokensGenerated > 0
              ? `${performanceMetrics.tokensGenerated} tokens`
              : "-- tokens",
        },
      ]
    : [];
  const canSendMessage = input.trim().length > 0 || uploadedImages.length > 0;
  const showSendButton = canSendMessage || isLoading;
  const hasGraphReasoningDisplay = Boolean(
    unifiedGraphReasoning?.summary?.summary_text ||
    unifiedGraphReasoning?.summary?.evidence?.length ||
    unifiedGraphReasoning?.paths?.length ||
    unifiedGraphReasoning?.bridge_entities?.length ||
    unifiedGraphReasoning?.direct_relationships?.length,
  );
  const hasPipelineArtifactDisplay = Boolean(
    (unifiedPipelineArtifacts?.run_count || 0) > 0 ||
    (unifiedQuality?.local_pipeline_artifact_run_count || 0) > 0,
  );
  const hasUnifiedContextDisplay = Boolean(
    unifiedLoading ||
    unifiedChunks.length > 0 ||
    graphEntities.length > 0 ||
    graphRelationships.length > 0 ||
    unifiedGraphConnections.length > 0 ||
    hasGraphReasoningDisplay ||
    hasPipelineArtifactDisplay,
  );
  const lastUserQuery = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message.role === "user") return stringifyChatContent(message.content);
    }
    return input;
  }, [input, messages]);
  const lastAssistantAnswer = useMemo(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (message.role === "assistant") {
        return stringifyChatContent(message.content);
      }
    }
    return "";
  }, [messages]);
  const referenceEvidenceQuery = useMemo(
    () => lastUserQuery || lastAssistantAnswer,
    [lastAssistantAnswer, lastUserQuery],
  );
  const chatReferences = useMemo(
    () =>
      buildChatReferenceItems({
        unifiedChunks,
        ragContext,
        relationships: graphRelationships,
        graphReasoning: unifiedGraphReasoning,
        pipelineArtifacts: unifiedPipelineArtifacts,
      }),
    [
      graphRelationships,
      ragContext,
      unifiedChunks,
      unifiedGraphReasoning,
      unifiedPipelineArtifacts,
    ],
  );
  const showChatSourcePanel = Boolean(
    unifiedLoading ||
    ragLoading ||
    graphLoading ||
    (messages.length > 0 &&
      graphRAGPipelineRunIds.length === 0 &&
      (hasUnifiedContextDisplay ||
        ragContext.length > 0 ||
        conversationEntities.length > 0)) ||
    (messages.length === 0 &&
      (graphRAGPipelineRunIds.length > 0 ||
        hasUnifiedContextDisplay ||
        ragContext.length > 0 ||
        conversationEntities.length > 0)),
  );
  const showPreChatSourceDetails = messages.length === 0;
  const chatSourceSummary = buildChatReferenceContextSummary({
    references: chatReferences,
    attachedDocumentCount: graphRAGPipelineRunIds.length,
    passageCount: unifiedChunks.length,
    fallbackPassageCount: ragContext.length,
    relationshipCount: graphRelationships.length,
    pathCount: unifiedGraphReasoning?.paths?.length,
  });
  const showInlineAnswerReferences = Boolean(
    messages.length > 0 && chatReferences.length > 0,
  );
  useEffect(() => {
    if (!showInlineAnswerReferences || isLoading) return;

    let attempts = 0;
    const maxAttempts = 12;
    const scrollReferenceTrayIntoView = () => {
      const scroller = messagesScrollRef.current;
      const section = inlineAnswerReferencesRef.current;
      if (!scroller) return;

      const tray = scroller.querySelector<HTMLElement>(
        '[data-testid="reference-tray"]',
      );
      const target = tray || section;
      if (!target) return;

      const scrollerRect = scroller.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      const topPadding = Math.max(
        8,
        Math.min(72, Math.round(scroller.clientHeight * 0.14)),
      );
      scroller.scrollTo({
        top: Math.max(
          0,
          scroller.scrollTop + targetRect.top - scrollerRect.top - topPadding,
        ),
        behavior: "auto",
      });
    };

    const interval = window.setInterval(() => {
      attempts += 1;
      scrollReferenceTrayIntoView();
      const scroller = messagesScrollRef.current;
      const section = inlineAnswerReferencesRef.current;
      const tray = scroller?.querySelector<HTMLElement>(
        '[data-testid="reference-tray"]',
      );
      const target = tray || section;
      if (!scroller || !target || attempts >= maxAttempts) {
        window.clearInterval(interval);
        return;
      }
      const scrollerRect = scroller.getBoundingClientRect();
      const targetRect = target.getBoundingClientRect();
      if (
        targetRect.top >= scrollerRect.top + 6 &&
        targetRect.top <=
          scrollerRect.top + Math.max(96, scroller.clientHeight * 0.36)
      ) {
        window.clearInterval(interval);
      }
    }, 120);
    return () => {
      window.clearInterval(interval);
    };
  }, [
    chatReferences.length,
    isLoading,
    messages.length,
    showInlineAnswerReferences,
  ]);
  const expandDocumentChatCanvas = Boolean(
    graphRAGPipelineRunIds.length > 0 && messages.length > 0,
  );
  const compactDocumentChatHeader = expandDocumentChatCanvas;
  const visibleGraphRAGContextFeatureChips = compactDocumentChatHeader
    ? graphRAGContextFeatureChips.slice(0, 2)
    : graphRAGContextFeatureChips;
  const visibleGraphRAGSecondaryQuestions = compactDocumentChatHeader
    ? []
    : graphRAGSecondaryQuestions;
  const isPreparingDocumentAnswer = Boolean(
    graphRAGPipelineRunIds.length > 0 &&
      (chatBusyPhase === "preparing_context" ||
        unifiedLoading ||
        ragLoading ||
        graphLoading),
  );
  const loadingStatusLabel =
    graphRAGPipelineRunIds.length > 0
      ? isPreparingDocumentAnswer
        ? "Finding sources"
        : "Writing"
      : "Generating";
  const assistantLoadingLabel =
    isPreparingDocumentAnswer
      ? "Finding document evidence..."
      : graphRAGPipelineRunIds.length > 0
        ? "Writing cited answer..."
        : "Assistant is typing...";
  const assistantLoadingDetail =
    graphRAGPipelineRunIds.length > 0
      ? isPreparingDocumentAnswer
        ? "Selecting passages, graph paths, and citations"
        : chatSourceSummary || "Document context is attached"
      : "";

  return (
    <PageShell
      sx={{
        height: "calc(100dvh - 120px)",
        display: "flex",
        flexDirection: "column",
        p: { xs: 1, sm: 1.25, md: 1.5 },
        width: "100%",
        minWidth: 0,
        maxWidth: "100%",
        overflow: "hidden",
      }}
    >
      <Box
        data-chat-shell
        onWheel={handleChatSurfaceWheel}
        sx={{
          display: "flex",
          flexDirection: "column",
          flex: 1,
          minHeight: 0,
          width: "100%",
          minWidth: 0,
          gap: 1,
        }}
      >
        <Paper
          data-chat-toolbar
          sx={{
            p: 0.75,
            borderRadius: "8px",
            border: "1px solid",
            borderColor: "divider",
            bgcolor: "background.paper",
            flexShrink: 0,
          }}
        >
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: {
                xs: "minmax(0, 1fr) auto",
                lg: "minmax(180px, 0.34fr) minmax(0, 1fr) auto",
              },
              gap: 0.75,
              alignItems: "center",
              minWidth: 0,
            }}
          >
            <Box sx={{ minWidth: 0 }}>
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 0.75,
                  minWidth: 0,
                }}
              >
                <Tooltip title="Conversation history">
                  <IconButton
                    onClick={() => setSidebarOpen(true)}
                    size="small"
                    sx={{
                      width: 28,
                      height: 28,
                      flexShrink: 0,
                      bgcolor: sidebarOpen ? "action.selected" : "action.hover",
                      border: "1px solid",
                      borderColor: "divider",
                    }}
                  >
                    <Badge
                      badgeContent={hasUnsavedChanges ? 1 : 0}
                      color="warning"
                      variant="dot"
                    >
                      <HistoryIcon fontSize="small" />
                    </Badge>
                  </IconButton>
                </Tooltip>
                {editingTitle ? (
                  <TextField
                    autoFocus
                    value={editTitleValue}
                    onChange={(e) => setEditTitleValue(e.target.value)}
                    onBlur={handleTitleSave}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleTitleSave();
                      if (e.key === "Escape") setEditingTitle(false);
                    }}
                    variant="standard"
                    fullWidth
                    sx={{
                      "& .MuiInput-input": {
                        fontWeight: 800,
                        fontSize: "0.95rem",
                        p: 0,
                      },
                    }}
                  />
                ) : (
                  <Tooltip title="Click to rename" placement="bottom-start">
                    <Typography
                      noWrap
                      onClick={() => {
                        setEditTitleValue(conversationTitle);
                        setEditingTitle(true);
                      }}
                      sx={{
                        fontWeight: 800,
                        color: "text.primary",
                        fontSize: "0.95rem",
                        lineHeight: 1.2,
                        cursor: "pointer",
                        minWidth: 0,
                      }}
                    >
                      {conversationTitle}
                    </Typography>
                  </Tooltip>
                )}
              </Box>
            </Box>

            <Box
              sx={{
                minWidth: 0,
                display: "flex",
                alignItems: "center",
                gap: 0.6,
                flexWrap: "wrap",
                gridColumn: { xs: "1 / -1", lg: "auto" },
              }}
            >
              {showSaveStateChip && (
                <StatusChip
                  status={saveStatusChipStatus}
                  statusConfig={CHAT_STATUS_CONFIG}
                  height={20}
                />
              )}
              {showReasoningChip && (
                <StatusChip
                  status="reasoning"
                  label={`Reasoning: ${settings.reasoningLevel}`}
                  statusConfig={CHAT_STATUS_CONFIG}
                  height={20}
                />
              )}
              {settings.enableStructuredOutput &&
                settings.structuredOutputSchema && (
                  <StatusChip
                    icon={<CodeIcon sx={{ fontSize: "0.875rem !important" }} />}
                    status="structured"
                    label={`JSON: ${settings.structuredOutputSchemaName}`}
                    onClick={() => setConstraintEditorOpen(true)}
                    clickable
                    statusConfig={CHAT_STATUS_CONFIG}
                    height={20}
                  />
                )}
              <Typography
                variant="caption"
                title={sessionDetailItems.join(" | ")}
                sx={{
                  minWidth: 0,
                  color: "text.secondary",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  flex: "1 1 220px",
                }}
              >
                {sessionSummary}
              </Typography>
              {activeSessionChips.map((item) => (
                <StatusChip
                  key={item.key}
                  status={item.status}
                  label={item.label}
                  statusConfig={CHAT_STATUS_CONFIG}
                  height={20}
                />
              ))}
              {compactMetricChips.map((item) => (
                <Chip
                  key={item.key}
                  size="small"
                  label={item.label}
                  variant="outlined"
                  sx={{
                    height: 20,
                    fontSize: "0.68rem",
                    color: "text.secondary",
                    borderColor: "divider",
                    "& .MuiChip-label": { px: 0.75 },
                  }}
                />
              ))}
              {showContextUsageBar && (
                <Box
                  sx={{
                    width: { xs: "100%", md: 112 },
                    height: 2,
                    borderRadius: 1,
                    bgcolor: alpha(contextColor, 0.18),
                    overflow: "hidden",
                  }}
                >
                  <Box
                    sx={{
                      width: `${contextUsagePercent}%`,
                      height: "100%",
                      bgcolor: contextColor,
                    }}
                  />
                </Box>
              )}
            </Box>

            <Box
              sx={{
                display: "flex",
                justifyContent: { xs: "flex-end", lg: "flex-end" },
                gap: 0.35,
                gridColumn: { xs: 2, lg: "auto" },
                gridRow: { xs: 1, lg: "auto" },
              }}
            >
              {(hasUnsavedChanges || isSaving) && (
                <Tooltip
                  title={
                    hasUnsavedChanges
                      ? "Save conversation"
                      : "Conversation saved"
                  }
                >
                  <span>
                    <IconButton
                      size="small"
                      aria-label="Save conversation"
                      onClick={() => saveConversation()}
                      disabled={isSaving || !hasUnsavedChanges}
                      sx={chatActionButtonSx}
                    >
                      {isSaving ? (
                        <CircularProgress size={15} />
                      ) : (
                        <SaveIcon fontSize="small" />
                      )}
                    </IconButton>
                  </span>
                </Tooltip>
              )}
              <Tooltip title="New conversation">
                <IconButton
                  size="small"
                  onClick={handleNewConversation}
                  sx={chatActionButtonSx}
                >
                  <AddIcon fontSize="small" />
                </IconButton>
              </Tooltip>
              <Tooltip
                title={
                  !sttServiceAvailable
                    ? "Voice mode unavailable — local STT service is not running. Deploy STT from the Deploy page."
                    : settings.voiceModeEnabled
                      ? "Voice mode: ON (local STT + TTS)"
                      : "Voice mode: OFF"
                }
              >
                <span>
                  <IconButton
                    size="small"
                    aria-label="Toggle voice mode"
                    color={settings.voiceModeEnabled ? "primary" : "default"}
                    onClick={handleToggleVoiceMode}
                    disabled={!sttServiceAvailable}
                    sx={chatActionButtonSx}
                  >
                    <HeadsetMicIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
              <Tooltip title="Settings">
                <IconButton
                  size="small"
                  color={showSettings ? "primary" : "default"}
                  onClick={() => setShowSettings(!showSettings)}
                  sx={chatActionButtonSx}
                >
                  <SettingsIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>

          {(saveStatus === "saving" || isLoading || voiceInput.isListening) && (
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 0.5 }}>
              {saveStatus === "saving" && (
                <StatusChip
                  status="draft"
                  label="Saving..."
                  statusConfig={CHAT_STATUS_CONFIG}
                  height={20}
                />
              )}
              {isLoading && (
                <StatusChip
                  icon={<CircularProgress size={12} />}
                  status="runtime"
                  label={loadingStatusLabel}
                  statusConfig={CHAT_STATUS_CONFIG}
                  height={20}
                />
              )}
              {voiceInput.isListening && (
                <StatusChip
                  status="saved"
                  label="Listening"
                  statusConfig={CHAT_STATUS_CONFIG}
                  height={20}
                />
              )}
            </Box>
          )}
        </Paper>

        {graphRAGPipelineRunIds.length > 0 && (
          <Paper
            data-document-scope
            sx={{
              p: { xs: compactDocumentChatHeader ? 0.75 : 1, sm: 1 },
              borderRadius: "8px",
              border: "1px solid",
              borderColor: alpha(accentColors.info, 0.24),
              bgcolor: alpha(accentColors.info, 0.045),
              flexShrink: 0,
              minWidth: 0,
            }}
          >
            <Stack spacing={{ xs: compactDocumentChatHeader ? 0.45 : 0.8, sm: 0.8 }}>
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: {
                    xs: "minmax(0, 1fr)",
                    md: "minmax(0, 1fr) auto",
                  },
                  alignItems: "center",
                  gap: 0.75,
                  minWidth: 0,
                }}
              >
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 0.75,
                    minWidth: 0,
                  }}
                >
                  <Box
                    sx={{
                      width: 30,
                      height: 30,
                      borderRadius: "8px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      color: accentColors.info,
                      bgcolor: alpha(accentColors.info, 0.12),
                      flexShrink: 0,
                    }}
                  >
                    <AttachFileIcon sx={{ fontSize: 17 }} />
                  </Box>
                  <Box sx={{ minWidth: 0, flex: 1 }}>
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 0.6,
                        minWidth: 0,
                        flexWrap: "wrap",
                      }}
                    >
                      <Typography
                        variant="body2"
                        fontWeight={850}
                        noWrap
                        title={graphRAGTraceLabel}
                        sx={{ minWidth: 0, maxWidth: { xs: "100%", md: 520 } }}
                      >
                        {graphRAGTraceLabel}
                      </Typography>
                      <StatusChip
                        status="pipeline"
                        label={
                          graphRAGPipelineRunIds.length > 1
                            ? "Document set"
                            : "Document"
                        }
                        statusConfig={CHAT_STATUS_CONFIG}
                        height={20}
                      />
                    </Box>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      noWrap
                      sx={{ display: "block" }}
                    >
                      {[
                        graphRAGChatHandoff?.domain
                          ? `${graphRAGChatHandoff.domain} domain`
                          : "",
                        graphRAGPipelineRunIds.length > 1
                          ? `${graphRAGPipelineRunIds.length} sources`
                          : "",
                        typeof graphRAGChatHandoff?.graphAnchorCount === "number"
                          ? `${graphRAGChatHandoff.graphAnchorCount} anchors`
                          : "",
                      ]
                        .filter(Boolean)
                        .slice(0, 3)
                        .join(" · ") || "Document context attached"}
                    </Typography>
                  </Box>
                </Box>
                <Stack
                  direction="row"
                  spacing={0.6}
                  flexWrap="wrap"
                  useFlexGap
                  sx={{
                    justifySelf: { xs: "flex-start", md: "flex-end" },
                    justifyContent: { xs: "flex-start", md: "flex-end" },
                  }}
                >
                  {!compactDocumentChatHeader && (
                    <Button
                      size="small"
                      variant="contained"
                      startIcon={<SendIcon fontSize="small" />}
                      onClick={() =>
                        handleSendMessage(
                          graphRAGDefaultPrompt,
                          graphRAGPrimaryAskLabel,
                        )
                      }
                      disabled={isLoading || !graphRAGDefaultPrompt}
                      sx={{
                        borderRadius: "8px",
                        textTransform: "none",
                        minHeight: 30,
                        boxShadow: "none",
                      }}
                    >
                      {graphRAGPrimaryAskLabel}
                    </Button>
                  )}
                  <Button
                    size="small"
                    variant={compactDocumentChatHeader ? "text" : "outlined"}
                    startIcon={<AttachFileIcon fontSize="small" />}
                    onClick={() =>
                      navigate(getRoutePath("graphrag-workbench"), {
                        state: {
                          selectedGraphRAGPipelineRunId:
                            graphRAGPipelineRunIds[0] || null,
                          selectedGraphRAGFilename:
                            graphRAGChatHandoff?.filename ||
                            graphRAGTraceLabel ||
                            "",
                        },
                      })
                    }
                    sx={{
                      borderRadius: "8px",
                      textTransform: "none",
                      minHeight: 30,
                    }}
                  >
                    {compactDocumentChatHeader ? "Documents" : "Workspace"}
                  </Button>
                </Stack>
              </Box>

              {(visibleGraphRAGContextFeatureChips.length > 0 ||
                visibleGraphRAGSecondaryQuestions.length > 0 ||
                (!compactDocumentChatHeader &&
                  (graphRAGChatHandoff?.answerGuidance ||
                    graphRAGHandoffCapabilityCards.length > 0))) && (
                <Box
                  sx={{
                    display: {
                      xs: compactDocumentChatHeader ? "none" : "flex",
                      sm: "flex",
                    },
                    alignItems: "center",
                    gap: 0.5,
                    flexWrap: "wrap",
                    minWidth: 0,
                  }}
                >
                  {visibleGraphRAGContextFeatureChips.map((label) => (
                    <Chip
                      key={`handoff-context-${label}`}
                      size="small"
                      label={label}
                      variant="outlined"
                      sx={{
                        height: 22,
                        maxWidth: 180,
                        fontSize: "0.66rem",
                        bgcolor: alpha(accentColors.info, 0.055),
                        "& .MuiChip-label": {
                          minWidth: 0,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        },
                      }}
                    />
                  ))}
                  {visibleGraphRAGSecondaryQuestions.map((question) => {
                    const questionLabel = graphRAGQuestionLabel(question);

                    return (
                      <Tooltip
                        key={question.id}
                        title={question.prompt}
                        enterDelay={500}
                        disableInteractive
                        disableTouchListener
                      >
                        <span>
                          <Button
                            size="small"
                            variant={
                              input === question.prompt
                                ? "contained"
                                : "outlined"
                            }
                            onClick={() =>
                              handleSendMessage(question.prompt, questionLabel)
                            }
                            disabled={isLoading}
                            sx={{
                              minHeight: 26,
                              maxWidth: { xs: "100%", sm: 260 },
                              borderRadius: "8px",
                              textTransform: "none",
                              px: 0.9,
                              fontSize: "0.72rem",
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
                        </span>
                      </Tooltip>
                    );
                  })}
                  {!compactDocumentChatHeader &&
                    graphRAGHandoffQuestions.length === 0 && (
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      noWrap
                      title={
                        graphRAGChatHandoff?.answerGuidance ||
                        graphRAGHandoffCapabilityCards
                          .map((card) => card.label)
                          .join(" | ")
                      }
                      sx={{ minWidth: 0, flex: 1 }}
                    >
                      {graphRAGChatHandoff?.answerGuidance ||
                        graphRAGHandoffCapabilityCards
                          .map((card) => card.label)
                          .join(" | ")}
                    </Typography>
                  )}
                </Box>
              )}
            </Stack>
          </Paper>
        )}

        {/* Settings Drawer */}
        <Drawer
          anchor="right"
          open={showSettings}
          onClose={() => setShowSettings(false)}
          PaperProps={{
            sx: {
              width: { xs: "100%", sm: 560, xl: 640 },
              maxWidth: "100vw",
              top: 56,
              height: "calc(100% - 56px)",
              bgcolor: "background.paper",
              borderLeft: "1px solid",
              borderColor: "divider",
            },
          }}
        >
          <Box
            sx={{
              height: "100%",
              display: "flex",
              flexDirection: "column",
              minWidth: 0,
            }}
          >
            <Box
              sx={{
                px: 2,
                py: 1.25,
                borderBottom: "1px solid",
                borderColor: "divider",
                display: "flex",
                alignItems: "center",
                gap: 1,
                flexShrink: 0,
                minWidth: 0,
              }}
            >
              <SettingsIcon
                sx={{ fontSize: 20, color: "primary.main", flexShrink: 0 }}
              />
              <Box sx={{ minWidth: 0, flex: 1 }}>
                <Typography
                  variant="subtitle1"
                  sx={{ fontWeight: 700, lineHeight: 1.2 }}
                >
                  Chat settings
                </Typography>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  title={sessionDetailItems.join(" | ")}
                  sx={{
                    display: "block",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {currentModelName} / {tokenCount.prompt.toLocaleString()} ctx
                  /{" "}
                  {settings.enableTools
                    ? `${settings.selectedTools.length} tools`
                    : "Tools off"}
                </Typography>
              </Box>
              <StatusChip
                status="reasoning"
                label={settings.reasoningLevel}
                statusConfig={CHAT_STATUS_CONFIG}
                height={20}
              />
              <Tooltip title="Close settings">
                <IconButton
                  aria-label="Close settings"
                  size="small"
                  onClick={() => setShowSettings(false)}
                  sx={{ ...chatActionButtonSx, flexShrink: 0 }}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Box>
            <Box
              data-preserve-wheel="true"
              sx={{
                flex: 1,
                minHeight: 0,
                overflowY: "auto",
                px: 2,
                py: 1.5,
                "&::-webkit-scrollbar": {
                  width: "8px",
                },
                "&::-webkit-scrollbar-track": {
                  bgcolor: "rgba(0, 0, 0, 0.1)",
                },
                "&::-webkit-scrollbar-thumb": {
                  bgcolor: "rgba(255, 255, 255, 0.2)",
                  borderRadius: "4px",
                  "&:hover": {
                    bgcolor: "rgba(255, 255, 255, 0.3)",
                  },
                },
              }}
            >
              {/* RAG Settings */}
              <Suspense fallback={null}>
                <RAGSettingsPanel
                  ragEnabled={settings.ragEnabled}
                  ragSearchMode={settings.ragSearchMode}
                  ragSelectedDomains={settings.ragSelectedDomains}
                  ragTopK={settings.ragTopK}
                  ragShowContext={settings.ragShowContext}
                  graphEnabled={settings.graphEnabled}
                  onSettingsChange={(ragSettings) =>
                    saveSettings({ ...settings, ...ragSettings })
                  }
                />
              </Suspense>

              {/* Connection Settings */}
              <Typography variant="h6" gutterBottom sx={{ mt: 2, mb: 1 }}>
                Connection Settings
              </Typography>
              {activeInferenceBackend && !settings.baseUrl && (
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mb: 1 }}
                >
                  Routing via backend proxy to active inference:{" "}
                  <Chip
                    size="small"
                    label={inferenceBackendLabel(activeInferenceBackend)}
                    color="primary"
                    variant="outlined"
                    sx={{ ml: 0.5, height: 22 }}
                  />
                </Typography>
              )}
              <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Base URL"
                    fullWidth
                    value={settings.baseUrl}
                    onChange={(e) =>
                      saveSettings({ ...settings, baseUrl: e.target.value })
                    }
                    placeholder="https://api.openai.com (leave empty for current domain)"
                    helperText="Base URL for the API service (optional)"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        fontSize: "0.875rem",
                        borderRadius: 1,
                        backgroundColor: "background.default",
                        "&.Mui-focused": {
                          borderColor: "primary.main",
                        },
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Endpoint Path"
                    fullWidth
                    value={settings.endpoint}
                    onChange={(e) =>
                      saveSettings({ ...settings, endpoint: e.target.value })
                    }
                    placeholder="/v1/chat/completions"
                    helperText="API endpoint path"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        fontSize: "0.875rem",
                        borderRadius: 1,
                        backgroundColor: "background.default",
                        "&.Mui-focused": {
                          borderColor: "primary.main",
                        },
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Model"
                    fullWidth
                    value={settings.model}
                    onChange={(e) =>
                      saveSettings({ ...settings, model: e.target.value })
                    }
                    placeholder="e.g. gpt-4o-mini or llama-3.1-8b-instruct"
                    helperText="Sent as the model parameter to the chat completions endpoint"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        fontSize: "0.875rem",
                        borderRadius: 1,
                        backgroundColor: "background.default",
                        "&.Mui-focused": {
                          borderColor: "primary.main",
                        },
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="API Key"
                    type="password"
                    fullWidth
                    value={settings.apiKey}
                    onChange={(e) =>
                      saveSettings({ ...settings, apiKey: e.target.value })
                    }
                    placeholder="Enter your API key (optional)"
                    helperText="API key for authentication (leave empty if not required)"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        fontSize: "0.875rem",
                        borderRadius: 1,
                        backgroundColor: "background.default",
                        "&.Mui-focused": {
                          borderColor: "primary.main",
                        },
                      },
                    }}
                  />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <TextField
                    label="OpenAI API Key (for voice)"
                    type="password"
                    fullWidth
                    value={settings.openaiApiKey}
                    onChange={(e) =>
                      saveSettings({
                        ...settings,
                        openaiApiKey: e.target.value,
                      })
                    }
                    placeholder="Enter OpenAI API key for transcription"
                    helperText="Required for voice input transcription"
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        fontSize: "0.875rem",
                        borderRadius: 1,
                        backgroundColor: "background.default",
                        "&.Mui-focused": {
                          borderColor: "primary.main",
                        },
                      },
                    }}
                  />
                </Grid>

                {/* Quick Presets */}
                <Grid item xs={12}>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    gutterBottom
                  >
                    Quick Presets:
                  </Typography>
                  <Box
                    sx={{ display: "flex", gap: 1, flexWrap: "wrap", mb: 2 }}
                  >
                    <Button
                      size="small"
                      variant="contained"
                      onClick={() =>
                        saveSettings({
                          ...settings,
                          baseUrl: "",
                          endpoint: "/v1/chat/completions",
                        })
                      }
                      sx={{ fontSize: "0.75rem", textTransform: "none" }}
                    >
                      Active backend (recommended)
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() =>
                        saveSettings({
                          ...settings,
                          baseUrl: "http://localhost:8600",
                          endpoint: "/v1/chat/completions",
                        })
                      }
                      sx={{ fontSize: "0.75rem", textTransform: "none" }}
                    >
                      LlamaCPP direct (8600)
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() =>
                        saveSettings({
                          ...settings,
                          baseUrl: "http://localhost:8601",
                          endpoint: "/v1/chat/completions",
                        })
                      }
                      sx={{ fontSize: "0.75rem", textTransform: "none" }}
                    >
                      vLLM direct (8601)
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() =>
                        saveSettings({
                          ...settings,
                          baseUrl: "",
                          endpoint: "/v1/chat/completions",
                          apiKey: "",
                          model: "Huihui-Qwen3.8-27B-abliterated",
                        })
                      }
                      sx={{ fontSize: "0.75rem", textTransform: "none" }}
                    >
                      NInfer via proxy
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() =>
                        saveSettings({
                          ...settings,
                          baseUrl: "https://api.openai.com",
                          endpoint: "/v1/chat/completions",
                        })
                      }
                      sx={{ fontSize: "0.75rem", textTransform: "none" }}
                    >
                      OpenAI API
                    </Button>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() =>
                        saveSettings({
                          ...settings,
                          baseUrl: "http://localhost:11434",
                          endpoint: "/v1/chat/completions",
                        })
                      }
                      sx={{ fontSize: "0.75rem", textTransform: "none" }}
                    >
                      Ollama (11434)
                    </Button>
                  </Box>
                </Grid>

                {/* URL Preview */}
                <Grid item xs={12}>
                  <Box
                    sx={{
                      p: 2,
                      backgroundColor: "rgba(0, 0, 0, 0.1)",
                      borderRadius: 1,
                      border: "1px solid rgba(255, 255, 255, 0.1)",
                    }}
                  >
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      gutterBottom
                    >
                      Full URL Preview:
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{ fontFamily: "monospace", color: "primary.main" }}
                    >
                      {settings.baseUrl
                        ? `${settings.baseUrl.replace(/\/$/, "")}${settings.endpoint}`
                        : `${window.location.origin}${settings.endpoint}`}
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              {/* Model Parameters */}
              <Typography variant="h6" gutterBottom sx={{ mt: 2, mb: 1 }}>
                Model Parameters
              </Typography>

              {/* Reasoning Level Control */}
              <Grid container spacing={3} sx={{ mb: 3 }}>
                <Grid item xs={12} sm={6}>
                  <Typography gutterBottom>Reasoning Level</Typography>
                  <Select
                    fullWidth
                    value={settings.reasoningLevel}
                    onChange={(e) =>
                      saveSettings({
                        ...settings,
                        reasoningLevel: e.target.value as
                          "low" | "medium" | "high" | "none",
                      })
                    }
                    sx={{
                      "& .MuiOutlinedInput-root": {
                        fontSize: "0.875rem",
                        borderRadius: 1,
                        backgroundColor: "background.default",
                        "&.Mui-focused": {
                          borderColor: "primary.main",
                        },
                      },
                    }}
                  >
                    <MenuItem value="none">None (Direct responses)</MenuItem>
                    <MenuItem value="low">Low (Fast responses)</MenuItem>
                    <MenuItem value="medium">Medium (Balanced)</MenuItem>
                    <MenuItem value="high">High (Deep analysis)</MenuItem>
                  </Select>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ mt: 0.5, display: "block" }}
                  >
                    Controls how much the model thinks before responding. Use
                    "Low" or "None" if the model is using too many tokens for
                    thinking.
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.showPerformanceMetrics}
                        onChange={(event) =>
                          saveSettings({
                            ...settings,
                            showPerformanceMetrics: event.target.checked,
                          })
                        }
                      />
                    }
                    label="Show performance metrics"
                  />
                </Grid>
              </Grid>

              <Grid container spacing={3}>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography gutterBottom>
                    Temperature: {settings.temperature}
                  </Typography>
                  <Slider
                    value={settings.temperature}
                    onChange={(_: Event, value: number | number[]) =>
                      saveSettings({
                        ...settings,
                        temperature: value as number,
                      })
                    }
                    min={0}
                    max={2}
                    step={0.1}
                    marks={[
                      { value: 0, label: "0" },
                      { value: 1, label: "1" },
                      { value: 2, label: "2" },
                    ]}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography gutterBottom>Top P: {settings.topP}</Typography>
                  <Slider
                    value={settings.topP}
                    onChange={(_: Event, value: number | number[]) =>
                      saveSettings({ ...settings, topP: value as number })
                    }
                    min={0}
                    max={1}
                    step={0.05}
                    marks={[
                      { value: 0, label: "0" },
                      { value: 0.5, label: "0.5" },
                      { value: 1, label: "1" },
                    ]}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography gutterBottom>Top K: {settings.topK}</Typography>
                  <Slider
                    value={settings.topK}
                    onChange={(_: Event, value: number | number[]) =>
                      saveSettings({ ...settings, topK: value as number })
                    }
                    min={1}
                    max={100}
                    step={1}
                    marks={[
                      { value: 1, label: "1" },
                      { value: 50, label: "50" },
                      { value: 100, label: "100" },
                    ]}
                  />
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography gutterBottom>
                    Max Tokens: {settings.maxTokens}
                  </Typography>
                  <Slider
                    value={settings.maxTokens}
                    onChange={(_: Event, value: number | number[]) =>
                      saveSettings({
                        ...settings,
                        maxTokens: Math.min(
                          value as number,
                          MAX_CHAT_OUTPUT_TOKENS,
                        ),
                      })
                    }
                    min={100}
                    max={MAX_CHAT_OUTPUT_TOKENS}
                    step={100}
                    marks={[
                      { value: 100, label: "100" },
                      { value: 1024, label: "1K" },
                      { value: MAX_CHAT_OUTPUT_TOKENS, label: "2K" },
                    ]}
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Tools Section */}
              <Typography variant="h6" gutterBottom>
                Function Calling Tools
              </Typography>
              <Box sx={{ mb: 2 }}>
                <FormControl component="fieldset">
                  <Button
                    variant={settings.enableTools ? "contained" : "outlined"}
                    startIcon={
                      settings.enableTools ? (
                        <CheckBoxIcon />
                      ) : (
                        <CheckBoxOutlineBlankIcon />
                      )
                    }
                    onClick={() =>
                      saveSettings({
                        ...settings,
                        enableTools: !settings.enableTools,
                      })
                    }
                    sx={{ mb: 2 }}
                  >
                    Enable Tool Calling
                  </Button>
                </FormControl>
              </Box>

              {settings.enableTools &&
                deployWorkloadProfile !== "agent" &&
                activeInferenceBackend === "llamacpp" && (
                  <Alert severity="info" sx={{ mb: 2, fontSize: "0.8125rem" }}>
                    Deploy is not using the Agent workload profile. On the{" "}
                    <Link
                      component="button"
                      type="button"
                      color="inherit"
                      underline="always"
                      onClick={navigateToDeploy}
                      sx={{ p: 0, font: "inherit", verticalAlign: "baseline" }}
                    >
                      Deploy page
                    </Link>
                    , select MTP → <strong>Agent</strong> for ~3× faster tool
                    calls (MTP n8, thinking off).
                  </Alert>
                )}

              {settings.enableTools && (
                <Grid container spacing={2}>
                  <Grid item xs={12}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <Typography variant="subtitle2" gutterBottom>
                        Available Tools (select which tools the model can use):
                      </Typography>
                      {mcpToolsLoading && (
                        <Chip
                          label="Loading MCP tools..."
                          size="small"
                          variant="outlined"
                        />
                      )}
                    </Box>
                  </Grid>
                  {/* Group built-in tools */}
                  {availableTools.filter(
                    (t: ExtendedTool) => t._source !== "mcp",
                  ).length > 0 && (
                    <Grid item xs={12}>
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ fontWeight: "bold" }}
                      >
                        📋 Built-in Tools
                      </Typography>
                    </Grid>
                  )}
                  {availableTools
                    .filter((t: ExtendedTool) => t._source !== "mcp")
                    .map((tool: ExtendedTool) => (
                      <Grid item xs={12} sm={6} md={4} key={tool.function.name}>
                        <Card
                          variant="outlined"
                          sx={{
                            cursor: "pointer",
                            bgcolor: settings.selectedTools.includes(
                              tool.function.name,
                            )
                              ? "action.selected"
                              : "background.paper",
                            "&:hover": { bgcolor: "action.hover" },
                          }}
                          onClick={() =>
                            toggleToolSelection(tool.function.name)
                          }
                        >
                          <CardContent sx={{ py: 1.5 }}>
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                mb: 1,
                              }}
                            >
                              <IconButton size="small" sx={{ mr: 1 }}>
                                {settings.selectedTools.includes(
                                  tool.function.name,
                                ) ? (
                                  <CheckBoxIcon color="primary" />
                                ) : (
                                  <CheckBoxOutlineBlankIcon />
                                )}
                              </IconButton>
                              <Typography
                                variant="subtitle2"
                                sx={{ fontWeight: "bold" }}
                              >
                                {tool.function.name}
                              </Typography>
                            </Box>
                            <Typography
                              variant="body2"
                              color="text.secondary"
                              sx={{ fontSize: "0.8rem" }}
                            >
                              {tool.function.description}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  {/* Group MCP tools by server */}
                  {availableTools.filter(
                    (t: ExtendedTool) => t._source === "mcp",
                  ).length > 0 && (
                    <Grid item xs={12}>
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ fontWeight: "bold", mt: 2 }}
                      >
                        🔌 MCP Tools
                      </Typography>
                    </Grid>
                  )}
                  {availableTools
                    .filter((t: ExtendedTool) => t._source === "mcp")
                    .map((tool: ExtendedTool) => (
                      <Grid item xs={12} sm={6} md={4} key={tool.function.name}>
                        <Card
                          variant="outlined"
                          sx={{
                            cursor: "pointer",
                            bgcolor: settings.selectedTools.includes(
                              tool.function.name,
                            )
                              ? "action.selected"
                              : "background.paper",
                            "&:hover": { bgcolor: "action.hover" },
                            borderColor: "primary.light",
                          }}
                          onClick={() =>
                            toggleToolSelection(tool.function.name)
                          }
                        >
                          <CardContent sx={{ py: 1.5 }}>
                            <Box
                              sx={{
                                display: "flex",
                                alignItems: "center",
                                mb: 1,
                                justifyContent: "space-between",
                              }}
                            >
                              <Box
                                sx={{ display: "flex", alignItems: "center" }}
                              >
                                <IconButton size="small" sx={{ mr: 1 }}>
                                  {settings.selectedTools.includes(
                                    tool.function.name,
                                  ) ? (
                                    <CheckBoxIcon color="primary" />
                                  ) : (
                                    <CheckBoxOutlineBlankIcon />
                                  )}
                                </IconButton>
                                <Typography
                                  variant="subtitle2"
                                  sx={{ fontWeight: "bold" }}
                                >
                                  {tool.function.name.replace(
                                    /^mcp_[^_]+_/,
                                    "",
                                  )}
                                </Typography>
                              </Box>
                              <Chip
                                label={tool._serverName || "MCP"}
                                size="small"
                                variant="outlined"
                                color="primary"
                                sx={{ fontSize: "0.65rem", height: 18 }}
                              />
                            </Box>
                            <Typography
                              variant="body2"
                              color="text.secondary"
                              sx={{ fontSize: "0.8rem" }}
                            >
                              {tool.function.description}
                            </Typography>
                          </CardContent>
                        </Card>
                      </Grid>
                    ))}
                  <Grid item xs={12}>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mt: 1 }}
                    >
                      Selected tools: {settings.selectedTools.length} /{" "}
                      {availableTools.length}
                      {availableTools.filter(
                        (t: ExtendedTool) => t._source === "mcp",
                      ).length > 0 && (
                        <span>
                          {" "}
                          (
                          {
                            availableTools.filter(
                              (t: ExtendedTool) => t._source === "mcp",
                            ).length
                          }{" "}
                          from MCP servers)
                        </span>
                      )}
                    </Typography>
                  </Grid>
                </Grid>
              )}

              <Divider sx={{ my: 3 }} />

              {/* Structured output */}
              <Typography variant="h6" gutterBottom>
                Structured Output
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Button
                  variant={
                    settings.enableStructuredOutput ? "contained" : "outlined"
                  }
                  startIcon={<CodeIcon />}
                  onClick={toggleStructuredOutput}
                  sx={{ mr: 1, mb: 1 }}
                >
                  {settings.enableStructuredOutput
                    ? "Structured Output On"
                    : "Enable Structured Output"}
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={() => setConstraintEditorOpen(true)}
                  sx={{ mb: 1 }}
                >
                  Edit JSON Schema
                </Button>
                {settings.structuredOutputSchema && (
                  <Button
                    variant="text"
                    size="small"
                    color="inherit"
                    onClick={clearStructuredOutput}
                    sx={{ mb: 1 }}
                  >
                    Clear
                  </Button>
                )}
              </Box>
              {settings.enableStructuredOutput &&
                settings.structuredOutputSchema && (
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ mb: 2, fontSize: "0.8rem" }}
                  >
                    Requests include <code>response_format: json_schema</code>
                    {settings.structuredOutputGrammar
                      ? " and a custom GBNF grammar."
                      : "."}
                  </Typography>
                )}
              {structuredOutputConflictsWithTools(settings) && (
                <Alert severity="warning" sx={{ mb: 2, fontSize: "0.8125rem" }}>
                  Tools and structured output are both enabled. llama.cpp uses a
                  single grammar slot — tool lazy-grammar may conflict with
                  custom JSON constraints. Disable one for reliable behavior.
                </Alert>
              )}

              <Divider sx={{ my: 3 }} />

              {/* Voice Settings Section */}
              <Typography variant="h6" gutterBottom>
                Voice Settings (STT/TTS)
              </Typography>

              {/* Service Status */}
              <Box
                sx={{
                  mb: 3,
                  display: "flex",
                  gap: 2,
                  flexWrap: "wrap",
                  alignItems: "center",
                }}
              >
                <Chip
                  label={
                    sttServiceAvailable
                      ? "STT Service Running"
                      : "STT Service Not Available"
                  }
                  color={sttServiceAvailable ? "success" : "default"}
                  size="small"
                  icon={<MicIcon />}
                />
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Chip
                    label={
                      ttsServiceAvailable
                        ? "TTS Service Running"
                        : "TTS Service Not Available"
                    }
                    color={ttsServiceAvailable ? "success" : "default"}
                    size="small"
                    icon={<VolumeUpIcon />}
                  />
                  {!ttsServiceAvailable ? (
                    <Button
                      size="small"
                      variant="contained"
                      color="primary"
                      onClick={handleStartTTS}
                      disabled={ttsStarting}
                      startIcon={
                        ttsStarting ? (
                          <CircularProgress size={14} />
                        ) : (
                          <PlayIcon />
                        )
                      }
                      sx={{
                        fontSize: "0.75rem",
                        textTransform: "none",
                        py: 0.5,
                      }}
                    >
                      {ttsStarting ? "Starting..." : "Start TTS"}
                    </Button>
                  ) : (
                    <Button
                      size="small"
                      variant="outlined"
                      color="error"
                      onClick={handleStopTTS}
                      disabled={ttsStopping}
                      startIcon={
                        ttsStopping ? (
                          <CircularProgress size={14} />
                        ) : (
                          <StopIcon />
                        )
                      }
                      sx={{
                        fontSize: "0.75rem",
                        textTransform: "none",
                        py: 0.5,
                      }}
                    >
                      {ttsStopping ? "Stopping..." : "Stop TTS"}
                    </Button>
                  )}
                </Box>
              </Box>

              {!sttServiceAvailable && (
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mb: 2 }}
                >
                  Deploy STT service from the Deploy page. TTS can be started
                  directly from here.
                </Typography>
              )}

              {/* [2026-10-15 | INFO | ChatPage.render | Voice mode master toggle — enables the fully-local voice loop (local STT + TTS) and re-routes the mic button] */}
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.voiceModeEnabled}
                    onChange={handleToggleVoiceMode}
                    disabled={!sttServiceAvailable}
                  />
                }
                label="Voice mode (fully-local STT → LLM → TTS loop)"
                sx={{ mb: 2 }}
              />

              <Grid container spacing={3}>
                {/* TTS Settings */}
                <Grid item xs={12} sm={6} md={4}>
                  <Typography gutterBottom>TTS Voice</Typography>
                  <Select
                    fullWidth
                    size="small"
                    value={settings.ttsVoice}
                    onChange={(e) =>
                      saveSettings({ ...settings, ttsVoice: e.target.value })
                    }
                    disabled={!ttsServiceAvailable}
                  >
                    <MenuItem value="alloy">Alloy</MenuItem>
                    <MenuItem value="echo">Echo</MenuItem>
                    <MenuItem value="fable">Fable</MenuItem>
                    <MenuItem value="onyx">Onyx</MenuItem>
                    <MenuItem value="nova">Nova</MenuItem>
                    <MenuItem value="shimmer">Shimmer</MenuItem>
                  </Select>
                </Grid>

                <Grid item xs={12} sm={6} md={4}>
                  <Typography gutterBottom>
                    TTS Speed: {settings.ttsSpeed}x
                  </Typography>
                  <Slider
                    value={settings.ttsSpeed}
                    onChange={(_: Event, value: number | number[]) =>
                      saveSettings({ ...settings, ttsSpeed: value as number })
                    }
                    min={0.5}
                    max={2.0}
                    step={0.1}
                    marks={[
                      { value: 0.5, label: "0.5x" },
                      { value: 1.0, label: "1x" },
                      { value: 2.0, label: "2x" },
                    ]}
                    disabled={!ttsServiceAvailable}
                  />
                </Grid>

                <Grid item xs={12} sm={6} md={4}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.ttsAutoPlay}
                        onChange={(e) =>
                          saveSettings({
                            ...settings,
                            ttsAutoPlay: e.target.checked,
                          })
                        }
                        disabled={!ttsServiceAvailable}
                      />
                    }
                    label="Auto-play TTS for responses"
                  />
                </Grid>

                {/* STT Settings */}
                <Grid item xs={12} sm={6} md={4}>
                  <Typography gutterBottom>STT Model</Typography>
                  <Select
                    fullWidth
                    size="small"
                    value={settings.sttModel}
                    onChange={(e) =>
                      saveSettings({ ...settings, sttModel: e.target.value })
                    }
                    disabled={!sttServiceAvailable}
                  >
                    <MenuItem value="tiny">Tiny (fastest)</MenuItem>
                    <MenuItem value="base">Base (balanced)</MenuItem>
                    <MenuItem value="small">Small (better)</MenuItem>
                    <MenuItem value="medium">Medium (great)</MenuItem>
                    <MenuItem value="large-v3">Large-v3 (best)</MenuItem>
                    <MenuItem value="distil-large-v3.5-ct2">
                      Distil-Large-v3.5 (fast & high quality)
                    </MenuItem>
                  </Select>
                </Grid>

                <Grid item xs={12} sm={6} md={4}>
                  <Typography gutterBottom>STT Language</Typography>
                  <Select
                    fullWidth
                    size="small"
                    value={settings.sttLanguage}
                    onChange={(e) =>
                      saveSettings({ ...settings, sttLanguage: e.target.value })
                    }
                    disabled={!sttServiceAvailable}
                  >
                    <MenuItem value="auto">Auto-detect</MenuItem>
                    <MenuItem value="en">English</MenuItem>
                    <MenuItem value="es">Spanish</MenuItem>
                    <MenuItem value="fr">French</MenuItem>
                    <MenuItem value="de">German</MenuItem>
                    <MenuItem value="it">Italian</MenuItem>
                    <MenuItem value="pt">Portuguese</MenuItem>
                    <MenuItem value="ru">Russian</MenuItem>
                    <MenuItem value="ja">Japanese</MenuItem>
                    <MenuItem value="ko">Korean</MenuItem>
                    <MenuItem value="zh">Chinese</MenuItem>
                  </Select>
                </Grid>

                {/* VAD Settings */}
                <Grid item xs={12} sm={6} md={4}>
                  <Typography gutterBottom>
                    Silence Duration: {settings.vadSilenceDuration}ms
                  </Typography>
                  <Slider
                    value={settings.vadSilenceDuration}
                    onChange={(_: Event, value: number | number[]) =>
                      saveSettings({
                        ...settings,
                        vadSilenceDuration: value as number,
                      })
                    }
                    min={500}
                    max={3000}
                    step={100}
                    marks={[
                      { value: 500, label: "0.5s" },
                      { value: 1500, label: "1.5s" },
                      { value: 3000, label: "3s" },
                    ]}
                    disabled={!sttServiceAvailable}
                  />
                  <Typography variant="caption" color="text.secondary">
                    How long to wait after you stop speaking before sending
                  </Typography>
                </Grid>

                <Grid item xs={12} sm={6} md={4}>
                  <Typography gutterBottom>
                    Silence Threshold:{" "}
                    {(settings.vadSilenceThreshold * 100).toFixed(0)}%
                  </Typography>
                  <Slider
                    value={settings.vadSilenceThreshold}
                    onChange={(_: Event, value: number | number[]) =>
                      saveSettings({
                        ...settings,
                        vadSilenceThreshold: value as number,
                      })
                    }
                    min={0.01}
                    max={0.1}
                    step={0.01}
                    marks={[
                      { value: 0.01, label: "1%" },
                      { value: 0.05, label: "5%" },
                      { value: 0.1, label: "10%" },
                    ]}
                    disabled={!sttServiceAvailable}
                  />
                  <Typography variant="caption" color="text.secondary">
                    Volume level below which is considered silence
                  </Typography>
                </Grid>
              </Grid>

              <Divider sx={{ my: 3 }} />

              {/* Keyboard Shortcuts */}
              <Typography variant="h6" gutterBottom>
                Keyboard Shortcuts
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
                {[
                  { keys: "Enter", action: "Send message" },
                  {
                    keys: navigator.platform.includes("Mac")
                      ? "Cmd+L"
                      : "Ctrl+L",
                    action: "Clear chat",
                  },
                  {
                    keys: navigator.platform.includes("Mac")
                      ? "Cmd+/"
                      : "Ctrl+/",
                    action: "Toggle settings",
                  },
                  {
                    keys: navigator.platform.includes("Mac")
                      ? "Cmd+N"
                      : "Ctrl+N",
                    action: "New conversation",
                  },
                  {
                    keys: navigator.platform.includes("Mac")
                      ? "Cmd+S"
                      : "Ctrl+S",
                    action: "Save conversation",
                  },
                  {
                    keys: navigator.platform.includes("Mac")
                      ? "Cmd+K"
                      : "Ctrl+K",
                    action: "Command palette",
                  },
                  { keys: "Esc", action: "Stop TTS / Close" },
                ].map(({ keys, action }) => (
                  <Box
                    key={keys}
                    sx={{ display: "flex", alignItems: "center", gap: 1 }}
                  >
                    <Chip
                      label={keys}
                      size="small"
                      sx={{
                        height: 24,
                        fontSize: "0.75rem",
                        fontFamily: "monospace",
                        bgcolor: "action.hover",
                      }}
                    />
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ fontSize: "0.8125rem" }}
                    >
                      {action}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Box>
          </Box>
        </Drawer>

        {/* Error Display */}
        {error && (
          <Paper
            sx={{
              p: 2,
              mb: 2,
              bgcolor: "error.dark",
              color: "error.contrastText",
              borderRadius: 1,
              border: "1px solid",
              borderColor: "error.main",
            }}
          >
            <Typography variant="body2" sx={{ fontSize: "0.8125rem" }}>
              Error: {error}
            </Typography>
          </Paper>
        )}

        {/* Messages */}
        <Box
          ref={messagesScrollRef}
          data-chat-messages
          sx={{
            flexGrow: 1,
            minHeight: 0,
            overflow: "auto",
            borderRadius: "8px",
            bgcolor: "background.paper",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            p: 0.75,
          }}
        >
          <Stack spacing={0.75}>
            {showChatSourcePanel && (
              <Paper
                variant="outlined"
                sx={{
                  border: "1px solid",
                  borderColor: alpha(accentColors.info, 0.22),
                  borderRadius: "8px",
                  bgcolor: alpha(accentColors.info, 0.035),
                  overflow: "hidden",
                }}
              >
                <Box
                  sx={{
                    px: 1,
                    py: 0.75,
                    display: "flex",
                    alignItems: "center",
                    gap: 0.75,
                    flexWrap: "wrap",
                  }}
                >
                  <StatusChip
                    status="structured"
                    label="Evidence"
                    statusConfig={CHAT_STATUS_CONFIG}
                    height={20}
                  />
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    noWrap
                    sx={{ minWidth: 0, flex: 1 }}
                  >
                    {chatSourceSummary ||
                      (isPreparingDocumentAnswer ||
                      unifiedLoading ||
                      ragLoading ||
                      graphLoading
                        ? "Finding sources"
                        : "Context available")}
                  </Typography>
                  {(isPreparingDocumentAnswer ||
                    unifiedLoading ||
                    ragLoading ||
                    graphLoading) && (
                    <CircularProgress size={14} />
                  )}
                </Box>
                <Stack spacing={0.75} sx={{ p: showPreChatSourceDetails ? 0.75 : 0, pt: 0 }}>
                  {showPreChatSourceDetails && graphRAGPipelineRunIds.length > 0 && (
                    <Box
                      sx={{
                        border: "1px solid",
                        borderColor: alpha(accentColors.purple, 0.18),
                        borderRadius: "8px",
                        bgcolor: alpha(accentColors.purple, 0.045),
                        p: 0.85,
                      }}
                    >
                      <Box
                        sx={{
                          display: "grid",
                          gridTemplateColumns: {
                            xs: "1fr",
                            md: "minmax(0, 1fr) auto",
                          },
                          gap: 0.75,
                          alignItems: "center",
                          minWidth: 0,
                        }}
                      >
                        <Box sx={{ minWidth: 0 }}>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              gap: 0.55,
                              minWidth: 0,
                              flexWrap: "wrap",
                            }}
                          >
                            <StatusChip
                              status="pipeline"
                              label="Document context"
                              statusConfig={CHAT_STATUS_CONFIG}
                              height={20}
                            />
                            <Typography
                              variant="caption"
                              noWrap
                              title={graphRAGTraceLabel}
                              sx={{ minWidth: 0, fontWeight: 850 }}
                            >
                              {graphRAGTraceLabel || "GraphRAG documents"}
                            </Typography>
                          </Box>
                          {graphRAGPromptPreview && (
                            <Typography
                              variant="caption"
                              color="text.secondary"
                              noWrap
                              title={graphRAGPromptPreview}
                              sx={{ display: "block", mt: 0.35 }}
                            >
                              {graphRAGPromptPreview}
                            </Typography>
                          )}
                        </Box>
                        <Box
                          sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: { xs: "flex-start", md: "flex-end" },
                            gap: 0.45,
                            flexWrap: "wrap",
                            minWidth: 0,
                          }}
                        >
                          {graphRAGContextFeatureChips.map((label) => (
                            <Chip
                              key={label}
                              size="small"
                              label={label}
                              variant="outlined"
                              sx={{
                                height: 20,
                                maxWidth: 160,
                                fontSize: "0.64rem",
                                bgcolor: alpha(accentColors.info, 0.04),
                                "& .MuiChip-label": {
                                  minWidth: 0,
                                  overflow: "hidden",
                                  textOverflow: "ellipsis",
                                },
                              }}
                            />
                          ))}
                          {graphRAGHandoffMeta.slice(0, 2).map((label) => (
                            <Chip
                              key={label}
                              size="small"
                              label={label}
                              sx={{
                                height: 20,
                                maxWidth: 160,
                                fontSize: "0.64rem",
                                bgcolor: alpha(accentColors.purple, 0.1),
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
                      {graphRAGHandoffQuestions.length > 0 && (
                        <Box
                          sx={{
                            display: "flex",
                            gap: 0.5,
                            flexWrap: "wrap",
                            mt: 0.75,
                          }}
                        >
                          {graphRAGHandoffQuestions
                            .slice(0, 2)
                            .map((question) => {
                              const questionLabel = graphRAGQuestionLabel(question);

                              return (
                                <Tooltip
                                  key={`evidence-handoff-${question.id}`}
                                  title={question.prompt}
                                >
                                  <span>
                                    <Button
                                      size="small"
                                      variant="outlined"
                                      startIcon={<SendIcon fontSize="small" />}
                                      onClick={() =>
                                        handleSendMessage(
                                          question.prompt,
                                          questionLabel,
                                        )
                                      }
                                      disabled={isLoading}
                                      sx={{
                                        minHeight: 26,
                                        maxWidth: { xs: "100%", sm: 260 },
                                        borderRadius: "8px",
                                        textTransform: "none",
                                        px: 0.9,
                                        fontSize: "0.72rem",
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
                                  </span>
                                </Tooltip>
                              );
                            })}
                        </Box>
                      )}
                    </Box>
                  )}
                  {showPreChatSourceDetails && chatReferences.length > 0 && (
                    <Suspense fallback={null}>
                      <ReferenceExplorer
                        references={chatReferences}
                        query={referenceEvidenceQuery}
                        density="cards"
                      />
                    </Suspense>
                  )}
                  {showPreChatSourceDetails && (
                  <Box
                    component="details"
                    sx={{
                      border: "1px solid",
                      borderColor: "divider",
                      borderRadius: "8px",
                      bgcolor: "background.paper",
                      overflow: "hidden",
                      "& summary": { listStyle: "none" },
                      "& summary::-webkit-details-marker": {
                        display: "none",
                      },
                    }}
                  >
                    <Box
                      component="summary"
                      sx={{
                        cursor: "pointer",
                        px: 1,
                        py: 0.65,
                        display: "flex",
                        alignItems: "center",
                        gap: 0.75,
                      }}
                    >
                      <StatusChip
                        status="draft"
                        label="Details"
                        statusConfig={CHAT_STATUS_CONFIG}
                        height={20}
                      />
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        noWrap
                        sx={{ minWidth: 0, flex: 1 }}
                      >
                        Retrieval, graph paths, and raw context
                      </Typography>
                    </Box>
                    <Stack spacing={0.75} sx={{ p: 0.75, pt: 0 }}>
                  {/* Unified Context Block (when graph reasoning is available) */}
                  {settings.graphEnabled && hasUnifiedContextDisplay && (
                    <Suspense fallback={null}>
                      <UnifiedContextBlock
                        chunks={unifiedChunks}
                        entities={graphEntities}
                        relationships={graphRelationships}
                        graphConnections={unifiedGraphConnections}
                        graphReasoning={unifiedGraphReasoning}
                        pipelineArtifacts={unifiedPipelineArtifacts}
                        crossReferences={unifiedCrossRefs}
                        qualitySignals={unifiedQuality || undefined}
                        isLoading={unifiedLoading}
                        onEntityClick={(
                          name: string,
                          type: string,
                          anchorEl: HTMLElement,
                        ) => {
                          setExplorerEntity({ name, type });
                          setExplorerAnchorEl(anchorEl);
                        }}
                      />
                    </Suspense>
                  )}
                  {/* Feedback bar for unified context */}
                  {settings.ragEnabled &&
                    settings.graphEnabled &&
                    unifiedChunks.length > 0 && (
                      <Suspense fallback={null}>
                        <RetrievalFeedbackBar
                          query={
                            messages.length > 0
                              ? typeof messages[messages.length - 1].content ===
                                "string"
                                ? (messages[messages.length - 1]
                                    .content as string)
                                : ""
                              : ""
                          }
                          qualitySignals={unifiedQuality}
                          domain={settings.ragSelectedDomains?.[0] || "general"}
                        />
                      </Suspense>
                    )}
                  {/* RAG Context Display (only when not in unified mode) */}
                  {settings.ragShowContext &&
                    !(
                      settings.ragEnabled &&
                      settings.graphEnabled &&
                      unifiedChunks.length > 0
                    ) &&
                    (ragLoading || ragContext.length > 0) && (
                      <Suspense fallback={null}>
                        <RAGContextBlock
                          chunks={ragContext}
                          isLoading={ragLoading}
                          onFetchNeighbors={async (
                            documentId,
                            chunkIndex,
                            direction,
                          ) => {
                            // Fetch neighboring chunks for exploration
                            try {
                              const response = await fetch(
                                "/api/v1/rag/chunks",
                                {
                                  method: "POST",
                                  headers: {
                                    "Content-Type": "application/json",
                                  },
                                  body: JSON.stringify({
                                    document_id: documentId,
                                    start_index:
                                      direction === "before"
                                        ? Math.max(0, chunkIndex - 2)
                                        : chunkIndex + 1,
                                    count: 2,
                                  }),
                                },
                              );
                              if (response.ok) {
                                const data = await response.json();
                                return (data.chunks || []).map((c: any) => ({
                                  id: c.id,
                                  content: c.content,
                                  score: 0,
                                  document_id: documentId,
                                  chunk_index: c.chunk_index,
                                  metadata: c.metadata,
                                }));
                              }
                            } catch (error) {
                              console.warn(
                                "Failed to fetch neighbor chunks:",
                                error,
                              );
                            }
                            return [];
                          }}
                        />
                      </Suspense>
                    )}
                  {/* Conversation Entity Sidebar */}
                  {settings.graphEnabled && conversationEntities.length > 0 && (
                    <Suspense fallback={null}>
                      <ConversationEntities
                        entities={conversationEntities}
                        onEntityClick={(name, type, anchorEl) => {
                          setExplorerEntity({ name, type });
                          setExplorerAnchorEl(anchorEl);
                        }}
                      />
                    </Suspense>
                  )}
                  {/* Graph Context Display (only when NOT in unified mode) */}
                  {settings.graphEnabled &&
                    !hasUnifiedContextDisplay &&
                    (graphLoading ||
                      graphEntities.length > 0 ||
                      graphRelationships.length > 0) && (
                      <Suspense fallback={null}>
                        <GraphContextBlock
                          entities={graphEntities}
                          relationships={graphRelationships}
                          isLoading={graphLoading}
                          onEntityClick={(name, type, anchorEl) => {
                            setExplorerEntity({ name, type });
                            setExplorerAnchorEl(anchorEl);
                          }}
                          onAskAbout={(name) => {
                            void handleSendMessage(
                              `Tell me more about ${name} and its connections`,
                            );
                          }}
                        />
                      </Suspense>
                      )}
                    </Stack>
                  </Box>
                  )}
                </Stack>
              </Paper>
            )}
            {/* Entity Explorer Popover */}
            <Suspense fallback={null}>
              <EntityExplorer
                anchorEl={explorerAnchorEl}
                entityName={explorerEntity.name}
                entityType={explorerEntity.type}
                onClose={() => setExplorerAnchorEl(null)}
                onEntityClick={(name, type) => {
                  setExplorerEntity({ name, type });
                }}
                onAskAbout={(name) => {
                  setExplorerAnchorEl(null);
                  void handleSendMessage(
                    `Tell me more about ${name} and its connections`,
                  );
                }}
              />
            </Suspense>
            {messages.length === 0 && graphRAGPipelineRunIds.length > 0 && (
              <Box
                sx={{
                  minHeight: "min(420px, 46vh)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  p: { xs: 1, md: 2 },
                }}
              >
                <Box
                  sx={{
                    width: "min(760px, 100%)",
                    border: "1px solid",
                    borderColor: alpha(accentColors.info, 0.24),
                    borderRadius: "8px",
                    bgcolor: alpha(accentColors.info, 0.035),
                    p: { xs: 1.25, md: 1.5 },
                  }}
                >
                  <Stack spacing={1}>
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        gap: 0.9,
                        minWidth: 0,
                      }}
                    >
                      <Box
                        sx={{
                          width: 34,
                          height: 34,
                          borderRadius: "8px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          color: accentColors.info,
                          bgcolor: alpha(accentColors.info, 0.12),
                          flexShrink: 0,
                        }}
                      >
                        <AttachFileIcon sx={{ fontSize: 18 }} />
                      </Box>
                      <Box sx={{ minWidth: 0, flex: 1 }}>
                        <Typography variant="subtitle2" fontWeight={850} noWrap>
                          Ready for document chat
                        </Typography>
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          noWrap
                          title={graphRAGTraceLabel}
                          sx={{ display: "block" }}
                        >
                          {[
                            graphRAGTraceLabel,
                            graphRAGChatHandoff?.domain
                              ? `${graphRAGChatHandoff.domain} domain`
                              : "",
                            ...graphRAGHandoffMeta,
                          ]
                            .filter(Boolean)
                            .slice(0, 4)
                            .join(" · ")}
                        </Typography>
                      </Box>
                    </Box>

                    <Box
                      sx={{
                        display: "grid",
                        gridTemplateColumns: {
                          xs: "repeat(2, minmax(0, 1fr))",
                          sm: "repeat(4, minmax(0, 1fr))",
                        },
                        border: "1px solid",
                        borderColor: alpha(accentColors.info, 0.16),
                        borderRadius: "8px",
                        overflow: "hidden",
                      }}
                    >
                      {[
                        {
                          label: "Documents",
                          value: graphRAGPipelineRunIds.length,
                        },
                        {
                          label: "Mode",
                          value: graphRAGChatHandoff?.recommendedMode
                            ? formatGraphRAGLabel(
                                graphRAGChatHandoff.recommendedMode,
                              )
                            : "GraphRAG",
                        },
                        {
                          label: "Anchors",
                          value:
                            typeof graphRAGChatHandoff?.graphAnchorCount ===
                            "number"
                              ? graphRAGChatHandoff.graphAnchorCount
                              : "Ready",
                        },
                        {
                          label: "Prompts",
                          value: graphRAGHandoffQuestions.length || 1,
                        },
                      ].map((item) => (
                        <Box
                          key={item.label}
                          sx={{
                            px: 1,
                            py: 0.75,
                            borderRight: { sm: "1px solid" },
                            borderBottom: { xs: "1px solid", sm: 0 },
                            borderColor: alpha(accentColors.info, 0.14),
                            "&:last-of-type": { borderRight: 0 },
                          }}
                        >
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            noWrap
                            sx={{ display: "block", fontSize: "0.66rem" }}
                          >
                            {item.label}
                          </Typography>
                          <Typography
                            title={String(item.value)}
                            sx={{
                              fontSize:
                                item.label === "Mode" ? "0.82rem" : "0.95rem",
                              fontWeight: 850,
                              lineHeight: 1.18,
                              color: accentColors.info,
                              overflowWrap: "anywhere",
                            }}
                          >
                            {item.value}
                          </Typography>
                        </Box>
                      ))}
                    </Box>

                    <Box sx={{ display: "flex", gap: 0.5, flexWrap: "wrap" }}>
                      <Button
                        size="small"
                        variant="contained"
                        startIcon={<SendIcon fontSize="small" />}
                        onClick={() =>
                          handleSendMessage(
                            graphRAGDefaultPrompt,
                            graphRAGPrimaryAskLabel,
                          )
                        }
                        disabled={isLoading || !graphRAGDefaultPrompt}
                        sx={{
                          minHeight: 28,
                          borderRadius: "8px",
                          textTransform: "none",
                          boxShadow: "none",
                        }}
                      >
                        {graphRAGPrimaryAskLabel}
                      </Button>
                      {graphRAGSecondaryQuestions.length > 0 &&
                        graphRAGSecondaryQuestions.map((question) => {
                          const questionLabel = graphRAGQuestionLabel(question);

                          return (
                            <Tooltip
                              key={`empty-${question.id}`}
                              title={question.prompt}
                            >
                              <span>
                                <Button
                                  size="small"
                                  variant="outlined"
                                  startIcon={<SendIcon fontSize="small" />}
                                  onClick={() =>
                                    handleSendMessage(
                                      question.prompt,
                                      questionLabel,
                                    )
                                  }
                                  disabled={isLoading}
                                  sx={{
                                    minHeight: 28,
                                    maxWidth: { xs: "100%", sm: 360 },
                                    minWidth: { sm: 250 },
                                    borderRadius: "8px",
                                    textTransform: "none",
                                    justifyContent: "flex-start",
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
                              </span>
                            </Tooltip>
                          );
                        })}
                    </Box>
                  </Stack>
                </Box>
              </Box>
            )}
            {messages.map((message: ChatMessage, index: number) => {
              const isPendingAssistantPlaceholder =
                isLoading &&
                index === messages.length - 1 &&
                message.role === "assistant" &&
                typeof message.content === "string" &&
                message.content.trim().length === 0 &&
                !message.reasoning_content?.trim() &&
                !message.tool_calls?.length;
              const isEmptyAssistantDiagnostic =
                !isPendingAssistantPlaceholder &&
                message.role === "assistant" &&
                typeof message.content === "string" &&
                message.content.trim().length === 0 &&
                !message.reasoning_content?.trim() &&
                !message.tool_calls?.length;

              if (isPendingAssistantPlaceholder) return null;

              if (isEmptyAssistantDiagnostic) {
                return (
                  <Box
                    key={index}
                    sx={{ display: "flex", justifyContent: "flex-start" }}
                  >
                    <Paper
                      sx={{
                        px: 0.85,
                        py: 0.45,
                        maxWidth: "76%",
                        borderRadius: 1,
                        border: "1px dashed",
                        borderColor: alpha(accentColors.warning, 0.45),
                        bgcolor: alpha(accentColors.warning, 0.08),
                      }}
                    >
                      <Typography
                        variant="caption"
                        sx={{
                          color: "warning.main",
                          display: "block",
                          fontFamily: "monospace",
                          lineHeight: 1.25,
                        }}
                      >
                        Empty assistant response from the model server. Retry
                        the prompt.
                      </Typography>
                    </Paper>
                  </Box>
                );
              }

              return (
                <Box
                  key={index}
                  sx={{
                    display: "flex",
                    justifyContent:
                      message.role === "user" ? "flex-end" : "flex-start",
                  }}
                >
                  <Paper
                    sx={{
                      position: "relative",
                      px: message.role === "user" ? 0.9 : 1,
                      pr: message.role === "user" ? { xs: 8.75, sm: 9.25 } : 1,
                      py: message.role === "user" ? 0.75 : 1,
                      maxWidth:
                        message.role === "user"
                          ? { xs: "92%", md: "62%", xl: "58%" }
                          : expandDocumentChatCanvas
                            ? "100%"
                            : "82%",
                      width:
                        message.role === "assistant" && expandDocumentChatCanvas
                          ? "100%"
                          : "auto",
                      bgcolor:
                        message.role === "user"
                          ? "primary.dark"
                          : "background.paper",
                      color:
                        message.role === "user"
                          ? "primary.contrastText"
                          : "text.primary",
                      border: "1px solid",
                      borderColor:
                        message.role === "user"
                          ? "primary.main"
                          : "rgba(255, 255, 255, 0.1)",
                      borderRadius: 1,
                      ...(message.role === "user" && {
                        "& .MuiTypography-body1": {
                          fontSize: "0.875rem",
                          lineHeight: 1.45,
                        },
                      }),
                      "& .message-actions": {
                        opacity: 1,
                        pointerEvents: "auto",
                        transition: "opacity 140ms ease",
                      },
                      "@media (hover: hover) and (pointer: fine)": {
                        "& .message-actions": {
                          opacity: 0,
                          pointerEvents: "none",
                        },
                        "&:hover .message-actions, &:focus-within .message-actions":
                          {
                            opacity: 1,
                            pointerEvents: "auto",
                          },
                      },
                    }}
                  >
                    <Box
                      sx={{
                        display: "flex",
                        alignItems: "center",
                        mb: message.role === "user" ? 0 : 0.5,
                        minHeight: message.role === "user" ? 0 : 20,
                      }}
                    >
                      {message.role !== "user" && (
                        <Avatar
                          sx={{
                            width: 18,
                            height: 18,
                            mr: 0.6,
                            bgcolor: getRoleColor(message.role) + ".main",
                            "& .MuiSvgIcon-root": { fontSize: 13 },
                          }}
                        >
                          {getRoleIcon(message.role)}
                        </Avatar>
                      )}
                      {message.role !== "user" && (
                        <Chip
                          label={formatRole(message.role)}
                          size="small"
                          color={
                            getRoleColor(message.role) as
                              "primary" | "secondary" | "info"
                          }
                          sx={{ height: 18, fontSize: "0.65rem" }}
                        />
                      )}
                      {message.role === "assistant" &&
                        (() => {
                          const bubbleTps = sanitizeTokensPerSecond(
                            message.tokensPerSecond,
                          );
                          return bubbleTps != null ? (
                            <Chip
                              label={`${bubbleTps.toFixed(2)} tok/s`}
                              size="small"
                              variant="outlined"
                              sx={{ ml: 0.75, fontSize: "0.65rem", height: 18 }}
                            />
                          ) : null;
                        })()}
                      <Box
                        className="message-actions"
                        sx={{
                          ml: message.role === "user" ? 0 : "auto",
                          display: "flex",
                          gap: 0.25,
                          ...(message.role === "user" && {
                            position: "absolute",
                            top: 3,
                            right: 3,
                            borderRadius: 1,
                            bgcolor: alpha(accentColors.slate, 0.5),
                          }),
                          "& .MuiIconButton-root": {
                            width: 22,
                            height: 22,
                            p: 0.2,
                          },
                          "& .MuiSvgIcon-root": { fontSize: 15 },
                        }}
                      >
                        {/* Edit button for user messages */}
                        {message.role === "user" &&
                          editingMessageIndex !== index && (
                            <Tooltip title="Edit message">
                              <span>
                                <IconButton
                                  size="small"
                                  aria-label="Edit message"
                                  onClick={() => startEditingMessage(index)}
                                  disabled={isLoading}
                                  sx={{ color: "text.secondary" }}
                                >
                                  <EditIcon fontSize="small" />
                                </IconButton>
                              </span>
                            </Tooltip>
                          )}
                        {/* TTS Play Button for assistant messages */}
                        {message.role === "assistant" &&
                          ttsServiceAvailable &&
                          typeof message.content === "string" &&
                          message.content.trim() && (
                            <Tooltip
                              title={
                                tts.isPlaying &&
                                tts.currentText === message.content
                                  ? "Stop speaking"
                                  : "Read aloud"
                              }
                            >
                              <IconButton
                                size="small"
                                aria-label={
                                  tts.isPlaying &&
                                  tts.currentText === message.content
                                    ? "Stop speaking"
                                    : "Read aloud"
                                }
                                onClick={() => {
                                  if (
                                    tts.isPlaying &&
                                    tts.currentText === message.content
                                  ) {
                                    tts.stop();
                                  } else {
                                    tts.speakNow(message.content as string);
                                  }
                                }}
                                sx={{
                                  color:
                                    tts.isPlaying &&
                                    tts.currentText === message.content
                                      ? "primary.main"
                                      : "text.secondary",
                                }}
                              >
                                {tts.isPlaying &&
                                tts.currentText === message.content ? (
                                  <StopIcon fontSize="small" />
                                ) : (
                                  <VolumeUpIcon fontSize="small" />
                                )}
                              </IconButton>
                            </Tooltip>
                          )}
                        <Tooltip title="Copy to clipboard">
                          <IconButton
                            size="small"
                            aria-label="Copy to clipboard"
                            onClick={() => copyToClipboard(message.content)}
                          >
                            <CopyIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Fork from here">
                          <IconButton
                            size="small"
                            aria-label="Fork from here"
                            onClick={() => handleForkConversation(index)}
                          >
                            <ForkIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    {/* Edit mode for user messages */}
                    {editingMessageIndex === index ? (
                      <Box sx={{ width: "100%" }}>
                        <TextField
                          fullWidth
                          multiline
                          minRows={2}
                          maxRows={8}
                          value={editingContent}
                          onChange={(e) => setEditingContent(e.target.value)}
                          autoFocus
                          sx={{
                            mb: 1,
                            "& .MuiOutlinedInput-root": {
                              fontSize: "0.875rem",
                              bgcolor: "rgba(0, 0, 0, 0.2)",
                            },
                          }}
                        />
                        <Box
                          sx={{
                            display: "flex",
                            gap: 1,
                            justifyContent: "flex-end",
                          }}
                        >
                          <Button
                            size="small"
                            onClick={cancelEditingMessage}
                            sx={{ textTransform: "none", fontSize: "0.75rem" }}
                          >
                            Cancel
                          </Button>
                          <Button
                            size="small"
                            variant="contained"
                            onClick={() => saveEditedMessage(index)}
                            disabled={!editingContent.trim() || isLoading}
                            startIcon={
                              isLoading ? (
                                <CircularProgress size={12} />
                              ) : (
                                <CheckIcon />
                              )
                            }
                            sx={{ textTransform: "none", fontSize: "0.75rem" }}
                          >
                            {isLoading
                              ? "Regenerating..."
                              : "Save & Regenerate"}
                          </Button>
                        </Box>
                      </Box>
                    ) : (
                      /* Normal message content */
                      (() => {
                        // Check if content is an array (multi-modal format)
                        if (Array.isArray(message.content)) {
                          const textParts: string[] = [];
                          const imageParts: any[] = [];

                          // Separate text and images
                          message.content.forEach((part: any) => {
                            if (part.type === "text") {
                              textParts.push(part.text);
                            } else if (part.type === "image_url") {
                              imageParts.push(part.image_url.url);
                            }
                          });

                          return (
                            <Box>
                              {/* Render images */}
                              {imageParts.length > 0 && (
                                <Box
                                  sx={{
                                    mb: 2,
                                    display: "flex",
                                    gap: 1,
                                    flexWrap: "wrap",
                                  }}
                                >
                                  {imageParts.map(
                                    (imageUrl: string, idx: number) => (
                                      <Box
                                        key={idx}
                                        sx={{
                                          maxWidth: 200,
                                          borderRadius: 1,
                                          overflow: "hidden",
                                          border:
                                            "1px solid rgba(255, 255, 255, 0.2)",
                                        }}
                                      >
                                        <img
                                          src={imageUrl}
                                          alt={`Attachment ${idx + 1}`}
                                          style={{
                                            width: "100%",
                                            height: "auto",
                                            display: "block",
                                          }}
                                        />
                                      </Box>
                                    ),
                                  )}
                                </Box>
                              )}
                              {/* Render text */}
                              {textParts.length > 0 && (
                                <Suspense
                                  fallback={
                                    <Typography
                                      variant="body1"
                                      sx={{
                                        whiteSpace: "pre-wrap",
                                        lineHeight: 1.7,
                                      }}
                                    >
                                      {textParts.join("\n")}
                                    </Typography>
                                  }
                                >
                                  <MarkdownRenderer
                                    content={textParts.join("\n")}
                                    reasoning_content={
                                      message.reasoning_content
                                    }
                                  />
                                </Suspense>
                              )}
                            </Box>
                          );
                        }

                        // String content - render normally
                        return (
                          <Suspense
                            fallback={
                              <Typography
                                variant="body1"
                                sx={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}
                              >
                                {message.content}
                              </Typography>
                            }
                          >
                            <MarkdownRenderer
                              content={message.content}
                              reasoning_content={message.reasoning_content}
                            />
                          </Suspense>
                        );
                      })()
                    )}
                  </Paper>
                </Box>
              );
            })}
            {showInlineAnswerReferences && (
              <Box
                ref={inlineAnswerReferencesRef}
                sx={{
                  width: {
                    xs: "100%",
                    md: expandDocumentChatCanvas ? "100%" : "min(820px, 82%)",
                  },
                  maxWidth: "100%",
                  alignSelf: "flex-start",
                }}
              >
                <Suspense fallback={null}>
                  <ReferenceExplorer
                    references={chatReferences}
                    query={referenceEvidenceQuery}
                    density="tray"
                  />
                </Suspense>
              </Box>
            )}
            {isLoading && (
              <Box sx={{ display: "flex", justifyContent: "flex-start" }}>
                <Paper
                  sx={{
                    p: 2,
                    bgcolor: "background.paper",
                    border: "1px solid rgba(255, 255, 255, 0.1)",
                    borderRadius: 1,
                  }}
                >
                  <Box sx={{ display: "flex", alignItems: "center" }}>
                    <CircularProgress size={20} sx={{ mr: 2 }} />
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="body2" color="text.secondary">
                        {assistantLoadingLabel}
                      </Typography>
                      {assistantLoadingDetail && (
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          noWrap
                          sx={{ display: "block", maxWidth: 520, opacity: 0.78 }}
                        >
                          {assistantLoadingDetail}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                </Paper>
              </Box>
            )}
            <div ref={messagesEndRef} />
          </Stack>
        </Box>

        {/* Voice Mode Compact Badge */}
        {settings.voiceModeEnabled && (
          <Box sx={{ mb: 2 }}>
            {/* Compact Badge - Always Visible */}
            <Paper
              onClick={() => setVoicePanelExpanded(!voicePanelExpanded)}
              sx={{
                p: 1,
                px: 1.5,
                borderRadius: 2,
                bgcolor: "background.paper",
                border: "1px solid",
                borderColor: voiceInput.isRecording
                  ? "error.main"
                  : voiceInput.isListening
                    ? "success.main"
                    : "rgba(255, 255, 255, 0.1)",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: 1.5,
                height: 48, // Fixed height to prevent bounce
                transition: "border-color 0.2s ease",
                "&:hover": {
                  borderColor: voiceInput.isRecording
                    ? "error.light"
                    : voiceInput.isListening
                      ? "success.light"
                      : "rgba(255, 255, 255, 0.2)",
                },
              }}
            >
              {/* Left: Animated mic icon */}
              <Box
                sx={{
                  position: "relative",
                  width: 32,
                  height: 32,
                  flexShrink: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                {/* Pulse animation ring */}
                <Box
                  sx={{
                    position: "absolute",
                    width: "100%",
                    height: "100%",
                    borderRadius: "50%",
                    bgcolor: voiceInput.isRecording
                      ? "error.main"
                      : voiceInput.isListening
                        ? "success.main"
                        : "grey.700",
                    opacity: 0.3,
                    transform: `scale(${1 + voiceInput.volume * 1.5})`,
                    transition: "transform 0.05s ease",
                  }}
                />
                <MicIcon
                  sx={{
                    fontSize: 18,
                    color: voiceInput.isRecording
                      ? "error.main"
                      : voiceInput.isListening
                        ? "success.main"
                        : "grey.500",
                    zIndex: 1,
                  }}
                />
              </Box>

              {/* Status text - fixed width */}
              <Typography
                variant="body2"
                sx={{
                  fontWeight: 600,
                  fontSize: "0.8rem",
                  width: 100, // Fixed width
                  flexShrink: 0,
                  whiteSpace: "nowrap",
                }}
              >
                {voiceInput.isTranscribing
                  ? "Processing..."
                  : voiceInput.isRecording
                    ? `Rec ${(voiceInput.recordingTime / 1000).toFixed(1)}s`
                    : voiceInput.isListening
                      ? "Listening"
                      : "Paused"}
              </Typography>

              {/* Volume display */}
              <Typography
                variant="caption"
                sx={{
                  fontFamily: "monospace",
                  fontSize: "0.7rem",
                  color:
                    voiceInput.volume > settings.vadSilenceThreshold
                      ? "success.main"
                      : "text.secondary",
                  width: 32,
                  flexShrink: 0,
                  textAlign: "right",
                }}
              >
                {(voiceInput.volume * 100).toFixed(0)}%
              </Typography>

              {/* Center: Volume bar - always visible, opacity changes */}
              <Box
                sx={{
                  flex: 1,
                  height: 8,
                  bgcolor: "rgba(0,0,0,0.3)",
                  borderRadius: 1,
                  overflow: "hidden",
                  position: "relative",
                  opacity: voiceInput.isListening ? 1 : 0.3,
                }}
              >
                {/* Volume bar */}
                <Box
                  sx={{
                    position: "absolute",
                    left: 0,
                    top: 0,
                    bottom: 0,
                    width: `${Math.min(voiceInput.volume * 200, 100)}%`,
                    bgcolor:
                      voiceInput.volume > settings.vadSilenceThreshold
                        ? voiceInput.isRecording
                          ? "error.main"
                          : "success.main"
                        : "grey.600",
                    borderRadius: 1,
                  }}
                />
                {/* Threshold marker */}
                <Box
                  sx={{
                    position: "absolute",
                    left: `${settings.vadSilenceThreshold * 200}%`,
                    top: 0,
                    bottom: 0,
                    width: 2,
                    bgcolor: "warning.main",
                  }}
                />
              </Box>

              {/* Countdown indicator - single element, only color changes */}
              <Typography
                variant="caption"
                sx={{
                  width: 36,
                  flexShrink: 0,
                  textAlign: "right",
                  fontFamily: "monospace",
                  fontSize: "0.7rem",
                  fontWeight: 600,
                  color:
                    voiceInput.isRecording && voiceInput.silenceTime > 0
                      ? "warning.main"
                      : "text.secondary",
                }}
              >
                {(voiceInput.isRecording && voiceInput.silenceTime > 0
                  ? (settings.vadSilenceDuration - voiceInput.silenceTime) /
                    1000
                  : settings.vadSilenceDuration / 1000
                ).toFixed(1)}
                s
              </Typography>

              {/* Right: Controls */}
              <Box
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 0.5,
                  flexShrink: 0,
                }}
                onClick={(e) => e.stopPropagation()}
              >
                {/* TTS toggle */}
                {ttsServiceAvailable && (
                  <Tooltip title={settings.ttsEnabled ? "TTS on" : "TTS off"}>
                    <IconButton
                      onClick={() =>
                        saveSettings({
                          ...settings,
                          ttsEnabled: !settings.ttsEnabled,
                        })
                      }
                      size="small"
                      sx={{
                        p: 0.5,
                        color: settings.ttsEnabled
                          ? "success.main"
                          : "grey.500",
                      }}
                    >
                      {settings.ttsEnabled ? (
                        <VolumeUpIcon fontSize="small" />
                      ) : (
                        <VolumeOffIcon fontSize="small" />
                      )}
                    </IconButton>
                  </Tooltip>
                )}

                {/* Listen toggle */}
                <Tooltip title={voiceInput.isListening ? "Stop" : "Start"}>
                  <IconButton
                    onClick={() => voiceInput.toggleListening()}
                    size="small"
                    sx={{
                      p: 0.5,
                      bgcolor: voiceInput.isListening
                        ? "error.main"
                        : "success.main",
                      color: "white",
                      "&:hover": {
                        bgcolor: voiceInput.isListening
                          ? "error.dark"
                          : "success.dark",
                      },
                    }}
                  >
                    {voiceInput.isListening ? (
                      <StopIcon fontSize="small" />
                    ) : (
                      <MicIcon fontSize="small" />
                    )}
                  </IconButton>
                </Tooltip>

                {/* Expand button */}
                <IconButton
                  size="small"
                  sx={{ p: 0.5, ml: 0.5 }}
                  onClick={() => setVoicePanelExpanded(!voicePanelExpanded)}
                >
                  {voicePanelExpanded ? (
                    <ExpandLessIcon fontSize="small" />
                  ) : (
                    <ExpandMoreIcon fontSize="small" />
                  )}
                </IconButton>
              </Box>
            </Paper>

            {/* Expanded Panel */}
            <Collapse in={voicePanelExpanded}>
              <Paper
                sx={{
                  mt: 0.5,
                  p: 1.5,
                  borderRadius: 1,
                  bgcolor: "background.paper",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                }}
              >
                {/* Full Audio Level Visualization */}
                <Box
                  sx={{
                    position: "relative",
                    height: 32,
                    bgcolor: "rgba(0,0,0,0.3)",
                    borderRadius: 1,
                    overflow: "hidden",
                    mb: 1.5,
                  }}
                >
                  {/* Current volume bar */}
                  <Box
                    sx={{
                      position: "absolute",
                      left: 0,
                      top: 4,
                      bottom: 4,
                      width: `${Math.min(voiceInput.volume * 200, 100)}%`,
                      bgcolor:
                        voiceInput.volume > settings.vadSilenceThreshold
                          ? voiceInput.isRecording
                            ? "error.main"
                            : "success.main"
                          : "grey.600",
                      borderRadius: 1,
                      transition: "width 0.05s ease-out",
                    }}
                  />

                  {/* Threshold line */}
                  <Box
                    sx={{
                      position: "absolute",
                      left: `${settings.vadSilenceThreshold * 200}%`,
                      top: 0,
                      bottom: 0,
                      width: 2,
                      bgcolor: "warning.main",
                      zIndex: 2,
                    }}
                  />

                  {/* Volume % and status */}
                  <Box
                    sx={{
                      position: "absolute",
                      inset: 0,
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      px: 1,
                      zIndex: 3,
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{ color: "white", fontSize: "0.7rem" }}
                    >
                      {voiceInput.volume > settings.vadSilenceThreshold
                        ? voiceInput.isRecording
                          ? "SPEAKING"
                          : "VOICE"
                        : "SILENCE"}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{
                        color: "white",
                        fontSize: "0.7rem",
                        fontWeight: 600,
                      }}
                    >
                      {(voiceInput.volume * 100).toFixed(0)}%
                    </Typography>
                  </Box>
                </Box>

                {/* Settings Grid */}
                <Box
                  sx={{
                    display: "flex",
                    gap: 2,
                    flexWrap: "wrap",
                    alignItems: "flex-start",
                  }}
                >
                  {/* Voice Threshold */}
                  <Box sx={{ minWidth: 140, flex: 1 }}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ fontSize: "0.7rem" }}
                      >
                        Threshold
                      </Typography>
                      <Typography
                        variant="caption"
                        sx={{
                          fontWeight: 600,
                          color: "primary.main",
                          fontSize: "0.7rem",
                        }}
                      >
                        {(settings.vadSilenceThreshold * 100).toFixed(0)}%
                      </Typography>
                    </Box>
                    <Slider
                      size="small"
                      value={settings.vadSilenceThreshold}
                      onChange={(_: Event, value: number | number[]) =>
                        saveSettings({
                          ...settings,
                          vadSilenceThreshold: value as number,
                        })
                      }
                      min={0.01}
                      max={0.3}
                      step={0.01}
                      sx={{ py: 0.5 }}
                    />
                  </Box>

                  {/* Silence Duration */}
                  <Box sx={{ minWidth: 140, flex: 1 }}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ fontSize: "0.7rem" }}
                      >
                        Silence Wait
                      </Typography>
                      <Typography
                        variant="caption"
                        sx={{
                          fontWeight: 600,
                          color: "primary.main",
                          fontSize: "0.7rem",
                        }}
                      >
                        {(settings.vadSilenceDuration / 1000).toFixed(1)}s
                      </Typography>
                    </Box>
                    <Slider
                      size="small"
                      value={settings.vadSilenceDuration}
                      onChange={(_: Event, value: number | number[]) =>
                        saveSettings({
                          ...settings,
                          vadSilenceDuration: value as number,
                        })
                      }
                      min={500}
                      max={5000}
                      step={100}
                      sx={{ py: 0.5 }}
                    />
                  </Box>

                  {/* Speech Filter */}
                  <Box sx={{ minWidth: 140, flex: 1 }}>
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ fontSize: "0.7rem" }}
                      >
                        Speech Filter
                      </Typography>
                      <Typography
                        variant="caption"
                        sx={{
                          fontWeight: 600,
                          color: "primary.main",
                          fontSize: "0.7rem",
                        }}
                      >
                        {(settings.noSpeechThreshold * 100).toFixed(0)}%
                        {voiceInput.lastNoSpeechProb !== null && (
                          <span
                            style={{
                              color:
                                voiceInput.lastNoSpeechProb >
                                settings.noSpeechThreshold
                                  ? "#f44336"
                                  : "#4caf50",
                            }}
                          >
                            {" "}
                            ({(voiceInput.lastNoSpeechProb * 100).toFixed(0)}%)
                          </span>
                        )}
                      </Typography>
                    </Box>
                    <Slider
                      size="small"
                      value={settings.noSpeechThreshold}
                      onChange={(_: Event, value: number | number[]) =>
                        saveSettings({
                          ...settings,
                          noSpeechThreshold: value as number,
                        })
                      }
                      min={0.1}
                      max={0.95}
                      step={0.05}
                      sx={{ py: 0.5 }}
                    />
                  </Box>

                  {/* STT Model */}
                  <Box sx={{ minWidth: 100 }}>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ display: "block", mb: 0.5, fontSize: "0.7rem" }}
                    >
                      STT Model
                    </Typography>
                    <Select
                      size="small"
                      value={settings.sttModel}
                      onChange={(e) =>
                        saveSettings({ ...settings, sttModel: e.target.value })
                      }
                      sx={{
                        fontSize: "0.7rem",
                        height: 26,
                        "& .MuiSelect-select": { py: 0.3, px: 1 },
                      }}
                    >
                      <MenuItem value="tiny">Tiny</MenuItem>
                      <MenuItem value="base">Base</MenuItem>
                      <MenuItem value="small">Small</MenuItem>
                      <MenuItem value="medium">Medium</MenuItem>
                      <MenuItem value="large-v3">Large</MenuItem>
                      <MenuItem value="distil-large-v3.5-ct2">
                        Distil-Large
                      </MenuItem>
                    </Select>
                  </Box>
                </Box>

                {/* Error display */}
                {voiceInput.error && (
                  <Typography
                    variant="caption"
                    color="error"
                    sx={{ mt: 1, display: "block" }}
                  >
                    {voiceInput.error}
                  </Typography>
                )}
              </Paper>
            </Collapse>
          </Box>
        )}

        {/* Input */}
        <Paper
          data-chat-input
          sx={{
            p: 1,
            borderRadius: "8px",
            bgcolor: "background.paper",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            width: "100%",
            minWidth: 0,
            flexShrink: 0,
          }}
        >
          {/* Image Preview Section */}
          {uploadedImages.length > 0 && (
            <Box sx={{ mb: 2, display: "flex", gap: 1, flexWrap: "wrap" }}>
              {uploadedImages.map((image, index) => (
                <Box
                  key={index}
                  sx={{
                    position: "relative",
                    width: 100,
                    height: 100,
                    borderRadius: 1,
                    overflow: "hidden",
                    border: "1px solid rgba(255, 255, 255, 0.2)",
                  }}
                >
                  <img
                    src={image.preview}
                    alt={`Upload ${index + 1}`}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "cover",
                    }}
                  />
                  <IconButton
                    size="small"
                    onClick={() => handleRemoveImage(index)}
                    sx={{
                      position: "absolute",
                      top: 2,
                      right: 2,
                      bgcolor: "rgba(0, 0, 0, 0.6)",
                      color: "white",
                      "&:hover": {
                        bgcolor: "rgba(0, 0, 0, 0.8)",
                      },
                      padding: "2px",
                    }}
                  >
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </Box>
              ))}
            </Box>
          )}

          <Box
            sx={{
              display: "flex",
              gap: 1,
              alignItems: "flex-end",
              width: "100%",
              minWidth: 0,
            }}
          >
            {/* Hidden file input */}
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              style={{ display: "none" }}
              onChange={handleImageUpload}
            />

            {/* Attachment button */}
            <Tooltip title="Attach images">
              <span>
                <IconButton
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isLoading}
                  size="small"
                  sx={{
                    flexShrink: 0,
                    color: "text.secondary",
                    "&:hover": { color: "primary.main" },
                  }}
                >
                  <AttachFileIcon />
                </IconButton>
              </span>
            </Tooltip>

            {/* Microphone button */}
            {/* [2026-10-15 | INFO | ChatPage.render | In voice mode the mic uses the local VAD STT path; otherwise it falls back to OpenAI cloud transcription] */}
            <Tooltip
              title={
                settings.voiceModeEnabled
                  ? voiceInput.isListening
                    ? "Stop local voice input (STT)"
                    : "Start local voice input (STT)"
                  : isRecording
                    ? "Recording..."
                    : "Voice input"
              }
            >
              <span>
                <IconButton
                  onClick={() => {
                    // [2026-10-15 | INFO | ChatPage.handleMicClick | Voice mode ON -> local STT; OFF -> legacy OpenAI cloud transcription]
                    if (settings.voiceModeEnabled) {
                      voiceInput.toggleListening();
                    } else if (isRecording) {
                      stopRecording();
                    } else {
                      startRecording();
                    }
                  }}
                  disabled={
                    isLoading ||
                    isTranscribing ||
                    (settings.voiceModeEnabled && voiceInput.isTranscribing)
                  }
                  size="small"
                  sx={{
                    flexShrink: 0,
                    color: settings.voiceModeEnabled
                      ? "primary.main"
                      : isRecording
                        ? "error.main"
                        : "text.secondary",
                    "&:hover": {
                      color: settings.voiceModeEnabled
                        ? "primary.dark"
                        : isRecording
                          ? "error.dark"
                          : "primary.main",
                    },
                    animation:
                      !settings.voiceModeEnabled && isRecording
                        ? "pulse 1.5s ease-in-out infinite"
                        : "none",
                    "@keyframes pulse": {
                      "0%, 100%": { opacity: 1 },
                      "50%": { opacity: 0.5 },
                    },
                  }}
                >
                  {settings.voiceModeEnabled ? (
                    voiceInput.isListening && !voiceInput.isTranscribing ? (
                      <StopIcon />
                    ) : (
                      <MicIcon />
                    )
                  ) : isRecording ? (
                    <StopIcon />
                  ) : (
                    <MicIcon />
                  )}
                </IconButton>
              </span>
            </Tooltip>

            <TextField
              ref={inputRef}
              multiline
              maxRows={4}
              value={input}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                setInput(e.target.value)
              }
              onKeyPress={handleKeyPress}
              placeholder="Type a message..."
              disabled={isLoading}
              sx={{
                flex: "1 1 0",
                minWidth: 0,
                width: 0,
                "& .MuiOutlinedInput-root": {
                  fontSize: "0.875rem",
                  borderRadius: 1,
                  backgroundColor: "background.default",
                  "&.Mui-focused": {
                    borderColor: "primary.main",
                  },
                },
              }}
            />
            {showSendButton && (
              <Button
                variant="contained"
                onClick={() => handleSendMessage()}
                disabled={isLoading}
                sx={{
                  flexShrink: 0,
                  minWidth: { xs: 44, sm: 100 },
                  px: { xs: 1, sm: 2 },
                  borderRadius: 1,
                  fontWeight: 500,
                  fontSize: "0.8125rem",
                  textTransform: "none",
                  boxShadow: "none",
                  "&:hover": {
                    boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
                  },
                  "& .MuiButton-startIcon": {
                    margin: { xs: 0, sm: "0 -4px 0 8px" },
                  },
                }}
                startIcon={
                  isLoading ? <CircularProgress size={16} /> : <SendIcon />
                }
              >
                <Box
                  component="span"
                  sx={{ display: { xs: "none", sm: "inline" } }}
                >
                  {isLoading
                    ? graphRAGPipelineRunIds.length > 0
                      ? isPreparingDocumentAnswer
                        ? "Finding"
                        : "Writing"
                      : "Sending"
                    : "Send"}
                </Box>
              </Button>
            )}
          </Box>
        </Paper>
      </Box>

      {/* Audio Recording Dialog */}
      <Dialog
        open={showAudioDialog}
        onClose={cancelRecording}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {isTranscribing
            ? "Transcribing Audio..."
            : isRecording
              ? "Recording Audio"
              : "Audio Transcription"}
        </DialogTitle>
        <DialogContent>
          <Box sx={{ textAlign: "center", py: 3 }}>
            {isRecording && (
              <Box>
                <Box
                  sx={{
                    width: 80,
                    height: 80,
                    borderRadius: "50%",
                    bgcolor: "error.main",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    margin: "0 auto",
                    animation: "pulse 1.5s ease-in-out infinite",
                    "@keyframes pulse": {
                      "0%, 100%": { transform: "scale(1)", opacity: 1 },
                      "50%": { transform: "scale(1.1)", opacity: 0.8 },
                    },
                  }}
                >
                  <MicIcon sx={{ fontSize: 40, color: "white" }} />
                </Box>
                <Typography variant="body1" sx={{ mt: 2 }}>
                  Recording in progress...
                </Typography>
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mt: 1 }}
                >
                  Click stop when you're done speaking
                </Typography>
              </Box>
            )}
            {isTranscribing && (
              <Box>
                <CircularProgress size={60} />
                <Typography variant="body1" sx={{ mt: 2 }}>
                  Transcribing your audio...
                </Typography>
              </Box>
            )}
            {audioError && (
              <Box>
                <Typography variant="body1" color="error" sx={{ mt: 2 }}>
                  {audioError}
                </Typography>
              </Box>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={cancelRecording} disabled={isTranscribing}>
            Cancel
          </Button>
          {isRecording && (
            <Button
              onClick={stopRecording}
              variant="contained"
              color="error"
              startIcon={<StopIcon />}
            >
              Stop Recording
            </Button>
          )}
        </DialogActions>
      </Dialog>

      <Suspense fallback={null}>
        <ConstraintEditor
          open={constraintEditorOpen}
          onClose={() => setConstraintEditorOpen(false)}
          onApply={handleApplyConstraints}
          initialSchema={settings.structuredOutputSchema ?? undefined}
        />
      </Suspense>

      {/* Conversation Sidebar */}
      <Suspense fallback={null}>
        <ConversationSidebar
          open={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          currentConversationId={currentConversationId || undefined}
        />
      </Suspense>
    </PageShell>
  );
};
