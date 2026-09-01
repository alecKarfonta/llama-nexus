const GENERATED_DIAGNOSTIC_HEADING_PATTERN =
  /^#{2,6}\s*(?:Candidate Entities|Candidate Graph Relationships|Embedding Chunks|Embedding Inputs|Graph Context|Graph Path Context|Local Graph|Relationship Candidates|Entity Candidates)\b/i;

const cleanRichOfficeLineMetadata = (value: string): string =>
  value
    .replace(/\s*;\s*(?:visibility|parse status|parse_status)\s*[:=]\s*[^;)\n]+/gi, "")
    .replace(/\s*;\s*(?:table|sheet|slide|chart|document|media)?\s*source\s+semantic_[a-z_]+\b[^;)\n]*/gi, "")
    .replace(/\s*;\s*(?:part|package path|relationship id|relationship_id|image relationship id|image_relationship_id)\s*(?:[:=]\s*)?[^;)\n]+/gi, "")
    .replace(/\s*;\s*(?:sheet|slide|chart|diagram|notes|media|control|text box)\s+(?:part|relationship)\s+[^;)\n]+/gi, "")
    .replace(/\s*;\s*(?:position|format)\s*[:=]?\s*[^;)\n]+/gi, "")
    .replace(/\s*;\s*\d+\s*x\s*\d+\s+image\b/gi, "")
    .replace(/\s*;\s*(?:image|audio|video)\/[a-z0-9.+-]+\b/gi, "")
    .replace(/\s*;\s*(?:layout\s*)?(?:bbox\s*)?x=\d+(?:\.\d+)?(?:in)?\s+y=\d+(?:\.\d+)?(?:in)?\s+w=\d+(?:\.\d+)?(?:in)?\s+h=\d+(?:\.\d+)?(?:in)?/gi, "")
    .replace(/\s+\bvisibility\s+(?:visible|hidden)\b/gi, "")
    .replace(/\s+\bsemantic_[a-z_]+\b/gi, "")
    .replace(/\s+\b(?:ppt|xl|word)\/[^\s;,)]+/gi, "")
    .replace(/\s*;\s*$/g, "")
    .replace(/\s+\)/g, ")")
    .replace(/\(\s+/g, "(")
    .replace(/\s{2,}/g, " ")
    .trimEnd();

const SOURCE_TEXT_TABLE_HEADER_PATTERN =
  /^(?:attachment|cell|comment|decision|evidence|field|label|media|note|owner|relation|relationship|source|status|target|type|value)$/i;

const isExtractionTableHeaderLine = (line: string): boolean => {
  const cells = line
    .split("|")
    .map((cell) => cell.trim())
    .filter(Boolean);
  return cells.length >= 2 && cells.length <= 6 && cells.every((cell) => SOURCE_TEXT_TABLE_HEADER_PATTERN.test(cell));
};

const isAdjacentToLabeledExtractionRow = (label: string, lines: string[], index: number): boolean => {
  const pattern = new RegExp(`\\b${label}\\s*:`, "i");
  const start = Math.max(0, index - 2);
  const end = Math.min(lines.length, index + 3);
  for (let cursor = start; cursor < end; cursor += 1) {
    if (cursor !== index && pattern.test(lines[cursor] || "")) return true;
  }
  return false;
};

export const cleanDocumentTextForReading = (value: string): string => {
  const readableLines: string[] = [];
  for (const line of value.replace(/\r\n?/g, "\n").split("\n")) {
    if (GENERATED_DIAGNOSTIC_HEADING_PATTERN.test(line.trim())) break;
    readableLines.push(line);
  }

  const lines = readableLines
    .join("\n")
    .split("\n")
    .map((line) =>
      line
        .replace(/^\s*(?:\d+\.\s*)?\[[^\n]*\]\s*/i, "")
        .replace(/^\s*(?:\d+\.\s*)?\[(?:paragraph|heading|list|code block|code|table|link|image|section)[^\]]*\]\s*/i, "")
        .replace(/^\s*(?:\d+\.\s*)?\[[a-z][a-z0-9_ -]*(?:\s*\|[^\]]*)?\]\s*/i, "")
        .replace(/^\s*\[(?:paragraph|heading|list|code block|code|table|link|image|section)[^\]]*\]\s*line\s*\d+\s*:\s*/i, "")
        .replace(/^\s*;\s*[^\]]+\]\s*/i, "")
        .replace(/^\s*line\s*\d+\s*:\s*/i, "")
        .replace(/^\s*Scope:\s*.*?\s+Evidence focus:\s*/i, "")
        .replace(/^\s*Evidence focus:\s*/i, "")
        .trimEnd(),
    )
    .map(cleanRichOfficeLineMetadata)
    .filter((line, index) => {
      const trimmed = line.trim();
      if (!trimmed) return true;
      if (index < 18 && /^#\s*Structured Ingest/i.test(trimmed)) return false;
      if (index < 18 && /^#\s*Structured GraphRAG Ingest/i.test(trimmed)) return false;
      if (index < 24 && /^##\s*(?:Document Text|Extracted Structure)$/i.test(trimmed)) return false;
      if (index < 24 && /^##\s*Source$/i.test(trimmed)) return false;
      if (index < 24 && /^-\s*(File|Filename|Content type|Extraction method|Structured text|Characters|Char count|Truncated|Run|Domain|Generated|Warnings|SHA256|Hash):/i.test(trimmed)) return false;
      if (/^Artifact manifest:/i.test(trimmed)) return false;
      return true;
    });
  const withoutExtractionTableHeaders = lines.filter((line, index) => {
    const trimmed = line.trim();
    if (isExtractionTableHeaderLine(trimmed)) return false;
    if (SOURCE_TEXT_TABLE_HEADER_PATTERN.test(trimmed) && isAdjacentToLabeledExtractionRow(trimmed, lines, index)) return false;
    return true;
  });
  const dedupedLines = withoutExtractionTableHeaders.filter((line, index) => {
    const trimmed = line.trim();
    if (!trimmed || index === 0) return true;
    const previous = withoutExtractionTableHeaders[index - 1]?.trim();
    return trimmed.toLowerCase() !== String(previous || "").toLowerCase();
  });

  return dedupedLines
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
};
