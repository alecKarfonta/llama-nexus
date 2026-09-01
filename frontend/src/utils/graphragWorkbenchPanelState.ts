export interface GraphRAGWorkbenchPanelStat {
  label: string;
  value: number;
}

export interface GraphRAGWorkbenchPanelStateInput {
  uploadFileCount: number;
  uploadInProgress?: boolean;
  selectedFileNames?: string;
  uploadPrimarySummary?: string;
  uploadPipelineStepCount?: number;
  hasMultipleUploadFiles?: boolean;
  hasBatchResult?: boolean;
  batchProcessedCount?: number;
  hasLatestImport?: boolean;
  activeDocumentRunCount?: number;
  activeDocumentTitle?: string;
  fallbackTitle?: string;
  readyStats?: GraphRAGWorkbenchPanelStat[];
  canAskContext?: boolean;
  formatCount?: (value: number) => string;
}

export interface GraphRAGWorkbenchPanelState {
  uploadActive: boolean;
  uploadQueued: boolean;
  evidenceTitle: string;
  evidenceChipLabel: string;
  documentHistoryLabel: string;
  eyebrow: string;
  title: string;
  subtitle: string;
  stats: GraphRAGWorkbenchPanelStat[];
  canAsk: boolean;
  showDocumentInsights: boolean;
  uploadNotice: string;
  primaryAction: "process" | "processing" | "chat";
  primaryActionLabel: string;
}

export type GraphRAGDocumentWorkspaceSection = "upload" | "documents" | "selected";

export interface GraphRAGDocumentWorkspaceActiveStepInput {
  workflowStep: number;
  desktopLayout: boolean;
  mobileLeadSection: GraphRAGDocumentWorkspaceSection;
  mobileWorkspaceSection?: GraphRAGDocumentWorkspaceSection | "";
}

const WORKSPACE_STEP_TARGETS: GraphRAGDocumentWorkspaceSection[] = [
  "upload",
  "documents",
  "selected",
];

export const graphRAGDocumentWorkspaceStepTargets = (
  leadSection: GraphRAGDocumentWorkspaceSection,
): GraphRAGDocumentWorkspaceSection[] => {
  void leadSection;
  return WORKSPACE_STEP_TARGETS;
};

const positiveStats = (stats: GraphRAGWorkbenchPanelStat[]) =>
  stats.filter((stat) => stat.value > 0);

export const graphRAGDocumentWorkspaceMobileGridAreas = (
  leadSection: GraphRAGDocumentWorkspaceSection,
): string => {
  if (leadSection === "selected") return `"selected" "documents" "upload"`;
  if (leadSection === "documents") return `"documents" "selected" "upload"`;
  return `"upload" "documents" "selected"`;
};

export const graphRAGDocumentWorkspaceActiveStepTarget = ({
  workflowStep,
  desktopLayout,
  mobileLeadSection,
  mobileWorkspaceSection = "",
}: GraphRAGDocumentWorkspaceActiveStepInput): GraphRAGDocumentWorkspaceSection => {
  if (mobileWorkspaceSection) return mobileWorkspaceSection;
  if (!desktopLayout) return mobileLeadSection;

  const stepIndex = Math.max(
    0,
    Math.min(WORKSPACE_STEP_TARGETS.length - 1, workflowStep - 1),
  );
  return WORKSPACE_STEP_TARGETS[stepIndex];
};

export const buildGraphRAGWorkbenchPanelState = ({
  uploadFileCount,
  uploadInProgress = false,
  selectedFileNames = "",
  uploadPrimarySummary = "",
  uploadPipelineStepCount = 0,
  hasMultipleUploadFiles = uploadFileCount > 1,
  hasBatchResult = false,
  batchProcessedCount = 0,
  hasLatestImport = false,
  activeDocumentRunCount = 0,
  activeDocumentTitle = "",
  fallbackTitle = "Selected document",
  readyStats = [],
  canAskContext = false,
  formatCount = (value) => String(value),
}: GraphRAGWorkbenchPanelStateInput): GraphRAGWorkbenchPanelState => {
  const uploadActive = uploadFileCount > 0;
  const uploadQueued = uploadActive && !uploadInProgress;
  const evidenceTitle = uploadActive
    ? selectedFileNames || "Selected upload"
    : activeDocumentTitle || fallbackTitle;
  const evidenceChipLabel = uploadInProgress
    ? "Processing"
    : uploadQueued
      ? `${formatCount(uploadFileCount)} selected`
      : hasBatchResult
        ? `${formatCount(batchProcessedCount)} docs`
        : canAskContext
          ? "Ready"
          : hasLatestImport || activeDocumentRunCount > 0
            ? "Indexed"
            : "Selected";
  const documentHistoryLabel = "";
  const eyebrow = uploadInProgress
    ? "Preparing"
    : uploadQueued
      ? "Selected upload"
      : hasBatchResult
        ? "Selected set"
        : "Document";
  const title = uploadInProgress
    ? hasMultipleUploadFiles
      ? `Processing ${formatCount(uploadFileCount)} documents`
      : "Processing document"
      : uploadQueued
        ? hasMultipleUploadFiles
          ? "Ready to process documents"
          : "Ready to process document"
      : hasBatchResult
        ? "Ask across these documents"
        : "Ask this document";
  const subtitle = uploadInProgress
    ? "Reading text, tables, relationships, and source locations."
    : uploadQueued
      ? uploadPrimarySummary || evidenceTitle
      : evidenceTitle;
  const stats = uploadActive
    ? positiveStats([
        { label: "Files", value: uploadFileCount },
        { label: "Steps", value: uploadPipelineStepCount },
      ])
    : positiveStats(readyStats);

  return {
    uploadActive,
    uploadQueued,
    evidenceTitle,
    evidenceChipLabel,
    documentHistoryLabel,
    eyebrow,
    title,
    subtitle,
    stats,
    canAsk: !uploadActive && canAskContext,
    showDocumentInsights: !uploadActive,
    uploadNotice: uploadInProgress
      ? "Building extraction, graph relationships, and chat context for the queued upload."
      : "Ready to extract relationships and citations for the selected upload.",
    primaryAction: uploadInProgress
      ? "processing"
      : uploadQueued
        ? "process"
        : "chat",
    primaryActionLabel: uploadInProgress
      ? "Processing"
      : uploadQueued
        ? "Process"
        : "Open chat",
  };
};
