export interface GraphRAGUploadManifestFile {
  name: string;
  size?: number | null;
}

export interface GraphRAGUploadKindCount {
  kind: string;
  count: number;
}

export interface GraphRAGUploadPipelineStep {
  label: string;
  detail: string;
}

export interface GraphRAGUploadManifest {
  fileCount: number;
  totalBytes: number;
  totalSizeLabel: string;
  kindCounts: GraphRAGUploadKindCount[];
  kindSummary: string;
  sizeAndKindSummary: string;
  primarySummary: string;
  richFormatCount: number;
  richFormatLabel: string | null;
  modeLabel: string;
  chatModeLabel: string;
  chunkingLabel: string;
  pipelineSteps: GraphRAGUploadPipelineStep[];
  pipelineSummary: string;
}

const RICH_FORMAT_KINDS = new Set([
  "Archive",
  "Doc",
  "EPUB",
  "Image",
  "PDF",
  "Slides",
  "Table",
]);

const pluralize = (count: number, singular: string, plural = `${singular}s`) =>
  `${count} ${count === 1 ? singular : plural}`;

const normalizeBytes = (bytes: number | null | undefined): number =>
  typeof bytes === "number" && Number.isFinite(bytes) && bytes > 0
    ? bytes
    : 0;

export const formatGraphRAGUploadFileSize = (bytes: number): string => {
  const safeBytes = normalizeBytes(bytes);
  if (safeBytes < 1024) return `${safeBytes} B`;
  if (safeBytes < 1024 * 1024) return `${(safeBytes / 1024).toFixed(1)} KB`;
  return `${(safeBytes / (1024 * 1024)).toFixed(1)} MB`;
};

export const formatGraphRAGUploadFileKind = (filename: string): string => {
  const extension = filename.split(".").pop()?.toLowerCase() || "";

  if (extension === "pdf") return "PDF";
  if (["doc", "docx", "docm", "dotx", "dotm", "rtf", "odt"].includes(extension)) {
    return "Doc";
  }
  if (["ppt", "pptx", "pptm", "potx", "potm", "odp", "ppsx", "ppsm"].includes(extension)) {
    return "Slides";
  }
  if (["xls", "xlsx", "xlsm", "xltx", "xltm", "csv", "ods"].includes(extension)) {
    return "Table";
  }
  if (["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp", "svg"].includes(extension)) {
    return "Image";
  }
  if (extension === "zip") return "Archive";
  if (extension === "epub") return "EPUB";
  if (extension === "md") return "Markdown";
  if (extension === "txt") return "Text";
  if (extension === "eml") return "Email";
  if (["json", "html", "xml"].includes(extension)) return extension.toUpperCase();

  return extension ? extension.toUpperCase() : "File";
};

const summarizeKinds = (kindCounts: GraphRAGUploadKindCount[]): string => {
  if (!kindCounts.length) return "No formats";

  const visibleKinds = kindCounts.slice(0, 3).map(({ count, kind }) =>
    count > 1 ? `${count} ${kind}` : kind,
  );
  const overflowCount = kindCounts.length - visibleKinds.length;
  if (overflowCount > 0) visibleKinds.push(`+${overflowCount}`);
  return visibleKinds.join(" + ");
};

export const buildGraphRAGUploadManifest = (
  files: GraphRAGUploadManifestFile[],
  options: {
    buildKnowledgeGraph?: boolean;
    useSemanticChunking?: boolean;
  } = {},
): GraphRAGUploadManifest => {
  const buildKnowledgeGraph = options.buildKnowledgeGraph !== false;
  const useSemanticChunking = options.useSemanticChunking !== false;
  const fileCount = files.length;
  const totalBytes = files.reduce(
    (total, file) => total + normalizeBytes(file.size),
    0,
  );
  const kindCounts = Object.entries(
    files.reduce<Record<string, number>>((counts, file) => {
      const kind = formatGraphRAGUploadFileKind(file.name);
      counts[kind] = (counts[kind] || 0) + 1;
      return counts;
    }, {}),
  )
    .map(([kind, count]) => ({ kind, count }))
    .sort(
      (left, right) =>
        right.count - left.count || left.kind.localeCompare(right.kind),
    );
  const kindSummary = summarizeKinds(kindCounts);
  const totalSizeLabel = formatGraphRAGUploadFileSize(totalBytes);
  const richFormatCount = files.filter((file) =>
    RICH_FORMAT_KINDS.has(formatGraphRAGUploadFileKind(file.name)),
  ).length;
  const richFormatLabel =
    richFormatCount > 0 ? pluralize(richFormatCount, "rich file") : null;
  const modeLabel =
    fileCount === 0
      ? "Ready"
      : fileCount === 1
        ? "Single import"
        : "Batch import";
  const chatModeLabel = buildKnowledgeGraph
    ? "Relationship chat"
    : "Document chat";
  const chunkingLabel = useSemanticChunking
    ? "Semantic chunks"
    : "Standard chunks";
  const pipelineSteps: GraphRAGUploadPipelineStep[] =
    fileCount === 0
      ? []
      : [
          { label: "Extract", detail: "Text and layout" },
          { label: "Chunk", detail: chunkingLabel },
          buildKnowledgeGraph
            ? { label: "Graph", detail: "Entities and relations" }
            : { label: "Embed", detail: "Vector context" },
          { label: "Chat", detail: chatModeLabel },
        ];

  return {
    fileCount,
    totalBytes,
    totalSizeLabel,
    kindCounts,
    kindSummary,
    sizeAndKindSummary:
      fileCount > 0 ? `${totalSizeLabel} - ${kindSummary}` : "No files queued",
    primarySummary:
      fileCount > 0
        ? `${pluralize(fileCount, "file")} - ${totalSizeLabel} - ${kindSummary}`
        : "No files selected",
    richFormatCount,
    richFormatLabel,
    modeLabel,
    chatModeLabel,
    chunkingLabel,
    pipelineSteps,
    pipelineSummary: pipelineSteps.map((step) => step.label).join(" -> "),
  };
};
