import { cleanDocumentTextForReading } from "./chatReferenceText";

export interface ChatReferenceDocumentLineRow {
  type: "line";
  lineNumber: number;
  text: string;
  matched: boolean;
}

export interface ChatReferenceDocumentGapRow {
  type: "gap";
  hiddenLines: number;
}

export type ChatReferenceDocumentReviewRow =
  | ChatReferenceDocumentLineRow
  | ChatReferenceDocumentGapRow;

export interface ChatReferenceDocumentReview {
  lineCount: number;
  matchCount: number;
  rows: ChatReferenceDocumentReviewRow[];
}

const DOCUMENT_TEXT_STOP_TERMS = new Set([
  "chunk",
  "content",
  "document",
  "evidence",
  "extract",
  "extracted",
  "file",
  "focus",
  "general",
  "heading",
  "hidden",
  "image",
  "images",
  "layout",
  "line",
  "logo",
  "markdown",
  "media",
  "microsoft",
  "paragraph",
  "passage",
  "part",
  "powerpoint",
  "scope",
  "section",
  "semantic",
  "slide",
  "slides",
  "source",
  "structured",
  "text",
  "visibility",
]);

const DOCUMENT_REVIEW_BOILERPLATE_LINE_PATTERN =
  /^(?:#{1,6}\s*document text|title|creator|author|keywords?|subject|company|application|app(?:lication)? name|created|modified|last modified by|revision|template|category|manager):\s*/i;

const isDocumentReviewBoilerplateLine = (line: string): boolean =>
  DOCUMENT_REVIEW_BOILERPLATE_LINE_PATTERN.test(line.trim());

const uniqueTerms = (terms: string[]): string[] => {
  const seen = new Set<string>();
  const output: string[] = [];

  terms.forEach((term) => {
    const normalized = String(term || "")
      .replace(/\s+/g, " ")
      .trim()
      .toLowerCase();
    if (
      normalized.length < 3 ||
      /^\d+$/.test(normalized) ||
      DOCUMENT_TEXT_STOP_TERMS.has(normalized) ||
      seen.has(normalized)
    ) {
      return;
    }

    seen.add(normalized);
    output.push(normalized);
  });

  return output.slice(0, 32);
};

export const buildChatReferenceDocumentReview = (
  text: string,
  terms: string[],
  options: {
    contextRadius?: number;
    maxRows?: number;
  } = {},
): ChatReferenceDocumentReview => {
  const lines = cleanDocumentTextForReading(String(text || ""))
    .replace(/\r\n?/g, "\n")
    .split("\n");
  const highlightTerms = uniqueTerms(terms);
  const contextRadius = Math.max(0, options.contextRadius ?? 1);
  const maxRows = Math.max(1, options.maxRows ?? 160);
  const matchedIndexes = new Set<number>();

  if (highlightTerms.length > 0) {
    lines.forEach((line, index) => {
      if (isDocumentReviewBoilerplateLine(line)) return;
      const normalizedLine = line.toLowerCase();
      if (highlightTerms.some((term) => normalizedLine.includes(term))) {
        matchedIndexes.add(index);
      }
    });
  }

  if (matchedIndexes.size === 0) {
    return {
      lineCount: lines.length,
      matchCount: 0,
      rows: [],
    };
  }

  const visibleIndexes = new Set<number>();
  matchedIndexes.forEach((index) => {
    for (
      let visibleIndex = Math.max(0, index - contextRadius);
      visibleIndex <= Math.min(lines.length - 1, index + contextRadius);
      visibleIndex += 1
    ) {
      visibleIndexes.add(visibleIndex);
    }
  });

  const sortedIndexes = Array.from(visibleIndexes)
    .filter((index) => !isDocumentReviewBoilerplateLine(lines[index] || ""))
    .sort((left, right) => left - right);
  const rows: ChatReferenceDocumentReviewRow[] = [];
  let previousIndex = -1;

  for (const index of sortedIndexes) {
    if (rows.length >= maxRows) break;

    if (previousIndex >= 0 && index - previousIndex > 1) {
      rows.push({
        type: "gap",
        hiddenLines: index - previousIndex - 1,
      });
      if (rows.length >= maxRows) break;
    }

    rows.push({
      type: "line",
      lineNumber: index + 1,
      text: lines[index],
      matched: matchedIndexes.has(index),
    });
    previousIndex = index;
  }

  if (sortedIndexes.length > 0 && previousIndex < sortedIndexes[sortedIndexes.length - 1]) {
    rows.push({
      type: "gap",
      hiddenLines: sortedIndexes[sortedIndexes.length - 1] - previousIndex,
    });
  }

  return {
    lineCount: lines.length,
    matchCount: matchedIndexes.size,
    rows,
  };
};
