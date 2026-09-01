const DOCUMENT_EXTENSION_PATTERN =
  /\.(?:pdf|pptx?|docx?|xlsx?|csv|tsv|md|markdown|txt|html?|json|xml|eml|msg|zip)(?:$|[?#\s])/i;

const normalizeFacetKey = (value: string): string =>
  value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

const formatFacet = (value: string): string =>
  value.replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const GENERIC_RELATION_KEYS = new Set([
  "edge",
  "link",
  "related",
  "related_to",
  "relation",
  "relationship",
]);

const LEADING_TARGET_RELATION_PATTERN =
  /^(drives?|supports?|references?|gates?|blocks?|requires?|owns?|contains?|connects?|groups?|funds?|reviews?|links?\s+to|relates\s+to|co[-\s]?occurs(?:\s+with)?)\s+(.+)$/i;

export const referenceBasenameForDisplay = (value: string): string =>
  value.split(/[\\/]/).filter(Boolean).pop() || value;

const officePackagePartLabel = (value: string): string => {
  const normalized = value.replace(/\\/g, "/").toLowerCase();
  if (!/(?:^|\/)(?:ppt|xl|word)\//.test(normalized)) return "";

  if (/(?:^|\/)(?:ppt|xl|word)\/charts\//.test(normalized)) return "chart evidence";
  if (/(?:^|\/)ppt\/slides\//.test(normalized)) return "slide evidence";
  if (/(?:^|\/)xl\/worksheets\//.test(normalized)) return "sheet evidence";
  if (/(?:^|\/)xl\/tables\//.test(normalized)) return "table evidence";
  if (/(?:^|\/)(?:ppt|xl|word)\/media\//.test(normalized)) return "media evidence";
  if (/(?:^|\/)(?:ppt|xl|word)\/embeddings\//.test(normalized)) return "embedded data";
  if (/(?:^|\/)(?:ppt|xl|word)\/comments\//.test(normalized)) return "comment evidence";
  if (/(?:^|\/)(?:ppt|word)\/notes/.test(normalized)) return "speaker notes";
  if (/(?:^|\/)(?:ppt|xl|word)\/_rels\//.test(normalized) || /\/_rels\//.test(normalized)) {
    return "relationship metadata";
  }
  return "document package evidence";
};

export const isDocumentLikeGraphLabel = (value?: string): boolean => {
  const text = String(value || "").trim();
  if (!text) return false;
  return Boolean(officePackagePartLabel(text)) || DOCUMENT_EXTENSION_PATTERN.test(referenceBasenameForDisplay(text));
};

export const cleanGraphReferenceLabel = (
  value: string | undefined,
  fallback = "Source",
): string => {
  const text = String(value || "")
    .replace(/\s+/g, " ")
    .replace(/^[-–]\s*/, "")
    .trim();
  if (!text) return fallback;

  const normalized = text.toLowerCase();
  if (normalized === "source | relation | target") return "Relationship table";
  if (normalized === "source") return "Source field";
  if (normalized === "relation") return "Relation field";
  if (normalized === "target") return "Target field";
  const packageLabel = officePackagePartLabel(text);
  if (packageLabel) return packageLabel;
  return isDocumentLikeGraphLabel(text) ? referenceBasenameForDisplay(text) : text;
};

const stripDisplayEndpointPunctuation = (value: string): string => {
  const text = value.replace(/\s+/g, " ").trim();
  if (!text) return "";
  if (isDocumentLikeGraphLabel(text)) return text;

  const lastToken = text.split(/\s+/).pop() || "";
  if (/^(?:[A-Za-z]\.){2,}$/.test(lastToken)) return text;

  return text.replace(/[.,;:!?]+$/g, "").trim() || text;
};

export const cleanGraphReferenceEndpointLabel = (
  value: string | undefined,
  fallback = "Source",
): string => stripDisplayEndpointPunctuation(cleanGraphReferenceLabel(value, fallback));

export const formatGraphReferenceRelation = (
  value: string | undefined,
  fallback = "related",
): string => {
  const raw = String(value || "").trim();
  if (!raw) return fallback;

  const key = normalizeFacetKey(raw);
  if (key === "handoff_path") return "graph path";
  if (key === "pipeline_handoff") return "handoff";
  if (
    [
      "answer_evidence",
      "references_chunk",
      "retrieved_passage",
    ].includes(key)
  ) {
    return "supports";
  }
  if (key === "pipeline_evidence_family") return "evidence";
  if (key === "source_document" || key === "source_document_context") return "source";
  if (key === "co_occurs" || key === "co_occurs_with" || key === "related_text") return "mentions";

  return formatFacet(raw).toLowerCase();
};

const isGenericGraphReferenceRelation = (value: string): boolean =>
  GENERIC_RELATION_KEYS.has(normalizeFacetKey(value));

const cleanInferredRelationLabel = (value: string): string => {
  const key = normalizeFacetKey(value);
  if (key === "co_occurs" || key === "co_occurs_with" || key === "related_text") return "mentions";
  if (key === "links_to") return "links to";
  if (key === "relates_to") return "relates to";
  return formatFacet(value).toLowerCase();
};

const inferRelationshipVerbFromTarget = (
  relationLabel: string,
  targetLabel: string,
): { relationLabel: string; targetLabel: string } | null => {
  if (!isGenericGraphReferenceRelation(relationLabel)) return null;
  const match = targetLabel.match(LEADING_TARGET_RELATION_PATTERN);
  if (!match) return null;

  const inferredRelation = cleanInferredRelationLabel(match[1]);
  const inferredTarget = match[2].replace(/\s+/g, " ").trim();
  if (!inferredRelation || !inferredTarget) return null;
  return {
    relationLabel: inferredRelation,
    targetLabel: inferredTarget,
  };
};

const graphRelationshipDisplayLabels = ({
  source,
  relation,
  target,
}: {
  source?: string;
  relation?: string;
  target?: string;
}): {
  sourceLabel: string;
  relationLabel: string;
  targetLabel: string;
  sourceIsDocumentBacked: boolean;
  targetIsDocumentBacked: boolean;
} => {
  const sourceLabel = cleanGraphReferenceEndpointLabel(source, "Source");
  let targetLabel = cleanGraphReferenceEndpointLabel(target, "Target");
  let relationLabel = formatGraphReferenceRelation(relation);
  const sourceIsDocumentBacked = isDocumentLikeGraphLabel(sourceLabel);
  let targetIsDocumentBacked = isDocumentLikeGraphLabel(targetLabel);

  if (!targetIsDocumentBacked) {
    const inferred = inferRelationshipVerbFromTarget(relationLabel, targetLabel);
    if (inferred) {
      relationLabel = inferred.relationLabel;
      targetLabel = cleanGraphReferenceEndpointLabel(inferred.targetLabel, "Target");
      targetIsDocumentBacked = isDocumentLikeGraphLabel(targetLabel);
    }
  }

  return {
    sourceLabel,
    relationLabel,
    targetLabel,
    sourceIsDocumentBacked,
    targetIsDocumentBacked,
  };
};

const relationshipLeadRelationLabel = (
  relationLabel: string,
  targetLabel: string,
): string => {
  const targetKey = normalizeFacetKey(targetLabel);
  const relationKey = normalizeFacetKey(relationLabel);
  const repeatedEvidenceNoun = relationKey.match(/^contains_(chart|table|sheet|slide|media)$/);
  if (repeatedEvidenceNoun && targetKey === `${repeatedEvidenceNoun[1]}_evidence`) {
    return "contains";
  }
  return relationLabel;
};

const sourceEvidenceNounFromRelation = (relation?: string): string => {
  const relationKey = normalizeFacetKey(String(relation || ""));
  const match = relationKey.match(/^contains_(chart|table|sheet|slide|media)$/);
  return match?.[1] || "";
};

export const briefGraphRelationshipTitle = ({
  source,
  relation,
  target,
}: {
  source?: string;
  relation?: string;
  target?: string;
}): string => {
  const {
    sourceLabel,
    targetLabel,
    sourceIsDocumentBacked,
    targetIsDocumentBacked,
  } = graphRelationshipDisplayLabels({ source, relation, target });

  if (sourceIsDocumentBacked && !targetIsDocumentBacked) return targetLabel;
  if (targetIsDocumentBacked && !sourceIsDocumentBacked) return sourceLabel;
  return [sourceLabel, targetLabel].filter(Boolean).join(" to ");
};

export const graphRelationshipEvidenceLead = ({
  source,
  relation,
  target,
}: {
  source?: string;
  relation?: string;
  target?: string;
}): string => {
  const {
    sourceLabel,
    relationLabel,
    targetLabel,
    sourceIsDocumentBacked,
    targetIsDocumentBacked,
  } = graphRelationshipDisplayLabels({ source, relation, target });
  const relationKey = normalizeFacetKey(String(relation || ""));
  const sourceEvidenceNoun = sourceEvidenceNounFromRelation(relation);

  if (sourceIsDocumentBacked && !targetIsDocumentBacked) {
    if (relationKey === "handoff_path") {
      return `${targetLabel} is connected by a graph path from source evidence`;
    }
    return `${targetLabel} is supported by source evidence`;
  }
  if (targetIsDocumentBacked && !sourceIsDocumentBacked) {
    if (relationKey === "handoff_path") {
      return `${sourceLabel} is connected by a graph path from source evidence`;
    }
    return `${sourceLabel} is supported by source evidence`;
  }
  if (
    sourceEvidenceNoun &&
    normalizeFacetKey(sourceLabel).includes(sourceEvidenceNoun) &&
    !normalizeFacetKey(targetLabel).includes(sourceEvidenceNoun)
  ) {
    return `${sourceLabel} is ${sourceEvidenceNoun} evidence from ${targetLabel}`;
  }

  return [
    sourceLabel,
    relationshipLeadRelationLabel(relationLabel, targetLabel),
    targetLabel,
  ].filter(Boolean).join(" ");
};

export interface GraphRelationshipEvidenceParts {
  sourceLabel: string;
  relationLabel: string;
  targetLabel: string;
  sourceDetail: string;
  targetDetail: string;
  lead: string;
  sourceIsDocumentBacked: boolean;
  targetIsDocumentBacked: boolean;
}

export const graphRelationshipEvidenceParts = ({
  source,
  relation,
  target,
}: {
  source?: string;
  relation?: string;
  target?: string;
}): GraphRelationshipEvidenceParts => {
  const {
    sourceLabel: rawSourceLabel,
    relationLabel,
    targetLabel: rawTargetLabel,
    sourceIsDocumentBacked,
    targetIsDocumentBacked,
  } = graphRelationshipDisplayLabels({ source, relation, target });

  return {
    sourceLabel: sourceIsDocumentBacked ? "Source document" : rawSourceLabel,
    relationLabel,
    targetLabel: targetIsDocumentBacked ? "Source document" : rawTargetLabel,
    sourceDetail: sourceIsDocumentBacked ? referenceBasenameForDisplay(rawSourceLabel) : "",
    targetDetail: targetIsDocumentBacked ? referenceBasenameForDisplay(rawTargetLabel) : "",
    lead: graphRelationshipEvidenceLead({ source, relation, target }),
    sourceIsDocumentBacked,
    targetIsDocumentBacked,
  };
};

export const graphRelationshipEvidenceMapNodes = ({
  source,
  relation,
  target,
}: {
  source?: string;
  relation?: string;
  target?: string;
}): string[] => {
  const {
    sourceLabel,
    relationLabel,
    targetLabel,
    sourceIsDocumentBacked,
    targetIsDocumentBacked,
  } = graphRelationshipDisplayLabels({ source, relation, target });

  return [
    sourceIsDocumentBacked ? "Source document" : sourceLabel,
    relationLabel,
    targetIsDocumentBacked ? "Source document" : targetLabel,
  ].filter(Boolean);
};

export const cleanGraphReferenceEvidenceText = (
  value: string,
  relationship?: {
    source?: string;
    relation?: string;
    target?: string;
    filename?: string;
  },
): string => {
  let text = value.replace(/\s+/g, " ").trim();
  if (!text) return "";

  const targetLabel = cleanGraphReferenceLabel(relationship?.target, "");
  const documentLabels = [
    relationship?.filename,
    relationship?.source,
    relationship?.target,
  ]
    .map(label => String(label || "").trim())
    .filter(isDocumentLikeGraphLabel)
    .map(referenceBasenameForDisplay);

  documentLabels.forEach(label => {
    const escaped = escapeRegExp(label);
    const relationWords = "(?:supports|references|contains|source|related(?:\\s+to)?)";
    text = text
      .replace(new RegExp(`${escaped}\\s*(?:->|${relationWords})\\s*`, "gi"), "")
      .replace(new RegExp(`\\s*(?:->|${relationWords})\\s*${escaped}`, "gi"), "")
      .replace(new RegExp(`\\bfile\\s+${escaped}`, "gi"), "source file");
    if (text.toLowerCase() !== label.toLowerCase()) {
      text = text.replace(new RegExp(escaped, "gi"), "source document");
    }
  });

  if (targetLabel && !isDocumentLikeGraphLabel(targetLabel)) {
    const escapedTarget = escapeRegExp(targetLabel);
    text = text.replace(new RegExp(`^${escapedTarget}\\s*[,.:;-]\\s*`, "i"), "");
    text = text.replace(
      new RegExp(`\\b([^.;:,]{2,60}?)\\s*->\\s*${escapedTarget}\\s+\\1\\b`, "i"),
      "$1",
    );
  }

  return text
    .replace(/\blocal pipeline\s+[^;()]+;?\s*/gi, "")
    .replace(/\s*;\s*layout\s*=\s*x=\d+(?:\.\d+)?(?:in)?\s+y=\d+(?:\.\d+)?(?:in)?\s+w=\d+(?:\.\d+)?(?:in)?\s+h=\d+(?:\.\d+)?(?:in)?/gi, "")
    .replace(/\s*;\s*(?:x|y|w|h)=\d+(?:\.\d+)?(?:in)?/gi, "")
    .replace(/\s*;\s*[^;.()]*\bvisibility\s+(?:visible|hidden)\b/gi, "")
    .replace(/\s*;\s*(?:table|sheet|slide|chart|document|media)?\s*source\s+semantic_[a-z_]+\b[^;.)]*/gi, "")
    .replace(/\s+\bvisibility\s+(?:visible|hidden)\b/gi, "")
    .replace(/\s+\bsemantic_[a-z_]+\b/gi, "")
    .replace(/\s*;\s*sheet\s+(?:part|relationship)\s+[^;)]+/gi, "")
    .replace(/\s*;\s*(?:parse_status|relationship_id|image_relationship_id)=[^;)]+/gi, "")
    .replace(/\bmedia\s+(?:embedded[_\s-]*package|source package part)\s+part\b/gi, "embedded package")
    .replace(/\bembedded[_\s-]*package\b/gi, "embedded package")
    .replace(/\bsource package part\b/gi, "source evidence")
    .replace(/\(\s*;\s*/g, "(")
    .replace(/\(\s*\)/g, "")
    .replace(/\s*--\[\s*([^\]]+?)\s*\]--?>\s*/g, (_match, relation: string) => ` ${formatGraphReferenceRelation(relation)} `)
    .replace(/\s*--\[\s*([^\]]+?)\s*\]--\s*/g, (_match, relation: string) => ` ${formatGraphReferenceRelation(relation)} `)
    .replace(/\s*->\s*/g, " -> ")
    .replace(/^(?:->|supports|references|contains|source)\s+/i, "")
    .replace(/\s+([,.;:])/g, "$1")
    .replace(/([,.;:])\s*\1+/g, "$1")
    .replace(/\b(Table\s+\d+)\s+\(table\)(?:\s*;\s*\1)?/gi, "$1")
    .replace(/\b(Table\s+\d+)(?:\s*;\s*\1\b)+/gi, "$1")
    .replace(/\bsource document\s+source document\b/gi, "source document")
    .replace(/^[,.;:\s-]+/, "")
    .replace(/\s+/g, " ")
    .trim();
};
