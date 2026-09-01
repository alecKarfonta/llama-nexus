import { cleanGraphRAGTraceSampleText, isLowValueGraphRAGTraceSampleText } from "./graphragTraceSamples";
import { formatGraphReferenceRelation } from "./graphragReferenceDisplay";

type GraphRAGRecord = Record<string, unknown>;

export interface GraphRAGDocumentFallbackChunk {
  content?: string;
  document_title?: string;
  document_id?: string;
  chunk_index?: number;
  metadata?: GraphRAGRecord;
}

export interface GraphRAGDocumentFallbackRelationship {
  source?: string;
  relation?: string;
  target?: string;
  relationship_type?: string;
  evidence?: string;
  rationale?: string;
  provenance?: string;
  provenance_summary?: string | string[];
  matched_query_terms?: string[];
  specific_matched_query_terms?: string[];
  score?: number;
  weight?: number;
  metadata?: GraphRAGRecord;
  location?: GraphRAGRecord;
}

export interface GraphRAGDocumentEvidenceCard
  extends GraphRAGDocumentFallbackRelationship {
  id?: string;
  role?: string;
  kind?: string;
  title?: string;
  text?: string;
  citation_label?: string;
  provenance_summary?: string | string[];
  location?: GraphRAGRecord;
  evidence_families?: string[];
  categories?: string[];
}

export interface GraphRAGDocumentFallbackPath {
  nodes?: string[];
  display_nodes?: string[];
  relationships?: GraphRAGDocumentFallbackRelationship[];
  explanation?: string;
  answer_hint?: string;
  text?: string;
  display_text?: string;
  traversal_text?: string;
  source?: string;
  target?: string;
  path_type?: string;
  provenance_summary?: string | string[];
  score?: number;
}

export interface GraphRAGDocumentFallbackContext {
  query: string;
  sourceLabel?: string;
  chunks?: GraphRAGDocumentFallbackChunk[];
  relationships?: GraphRAGDocumentFallbackRelationship[];
  graphReasoning?: {
    direct_relationships?: GraphRAGDocumentFallbackRelationship[];
    paths?: GraphRAGDocumentFallbackPath[];
    answer_plan?: {
      strategy?: string;
      primary_path?: GraphRAGDocumentFallbackPath;
      hops?: GraphRAGDocumentFallbackRelationship[];
      supporting_relationships?: GraphRAGDocumentFallbackRelationship[];
      evidence_cards?: GraphRAGDocumentEvidenceCard[];
    };
  } | null;
}

interface FallbackCandidate {
  source: string;
  relation: string;
  relationKey: string;
  target: string;
  statement: string;
  evidence: string;
  citation: string;
  score: number;
}

interface ChunkCandidate {
  text: string;
  citation: string;
  model: string;
  score: number;
}

interface PathCandidate {
  nodes: string[];
  relationships: FallbackCandidate[];
  text: string;
  explanation: string;
  score: number;
}

interface RequestedEvidenceCandidate {
  kind: string;
  title: string;
  evidence: string;
  citation: string;
  score: number;
  sourceIndex: number;
}

interface EntitySummaryCandidate {
  name: string;
  score: number;
}

interface SupplementalRequestedEvidence {
  entity: string;
  body: string;
  citation: string;
  relationship?: FallbackCandidate;
}

interface PerSourceEvidenceCandidate {
  source: string;
  body: string;
  citation: string;
  score: number;
}

const STOP_WORDS = new Set([
  "about",
  "after",
  "also",
  "and",
  "are",
  "before",
  "cite",
  "does",
  "document",
  "evidence",
  "from",
  "into",
  "report",
  "reports",
  "source",
  "that",
  "the",
  "this",
  "what",
  "which",
  "with",
]);

const DIRECT_RELATION_PATTERN =
  /^(?:approves?|blocks?|depends?|drives?|gates?|influences?|requires?|supports?|commented_on|decision|groups_visual|relation)$/i;

const WEAK_RELATION_PATTERN =
  /^(?:appears?_with|co_occurs(?:_with)?|mentioned?_with|mentions?|related_to)$/i;

const ATTRIBUTE_RELATION_PATTERN =
  /^(?:alias|category|confidence|domain|field_type|form_kind|kind|label|location|owner|region|source_owner|status|tag|title|type|value)$/i;

const URL_PATTERN = /https?:\/\/[^\s;,)]+/gi;
const DOCUMENT_FILE_EXTENSION_PATTERN =
  /\.(?:pdf|pptx?|docx?|xlsx?|csv|md|html?|txt|json|xml)(?:$|[?#])/i;

const asRecord = (value: unknown): GraphRAGRecord =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as GraphRAGRecord)
    : {};

const asString = (value: unknown): string =>
  typeof value === "string" ? value.trim() : "";

const stringList = (value: unknown): string[] =>
  Array.isArray(value)
    ? value.map((item) => String(item || "").trim()).filter(Boolean)
    : typeof value === "string" && value.trim()
      ? [value.trim()]
      : [];

const normalizeRelationKey = (value: unknown): string =>
  String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

const queryTerms = (query: string): string[] =>
  Array.from(
    new Set(
      query
        .toLowerCase()
        .replace(/[^a-z0-9/.-]+/g, " ")
        .split(/\s+/)
        .map((term) => term.trim())
        .filter((term) => term.length > 2 && !STOP_WORDS.has(term)),
    ),
  );

const sentenceCase = (value: string): string => {
  const text = value.trim();
  if (!text) return "";
  return `${text[0].toUpperCase()}${text.slice(1)}`;
};

const stripTrailingPunctuation = (value: string): string =>
  value.replace(/\s*[.;:,-]+\s*$/g, "").trim();

const escapeRegExp = (value: string): string =>
  value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const uniqueStrings = (values: string[]): string[] => {
  const seen = new Set<string>();
  return values
    .map((value) => value.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .filter((value) => {
      const key = value.toLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
};

const humanList = (values: string[]): string => {
  const items = uniqueStrings(values);
  if (items.length <= 2) return items.join(" and ");
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
};

const readableListItems = (values: string[]): string[] =>
  uniqueStrings(
    values.flatMap((value) => {
      const text = normalizeRequestedEvidenceText(value);
      if (!text) return [];
      try {
        const parsed = JSON.parse(text);
        if (Array.isArray(parsed)) {
          return parsed.map((item) => normalizeRequestedEvidenceText(String(item || "")));
        }
      } catch {
        // Fall through to forgiving display cleanup for partially serialized values.
      }

      const unwrapped = text
        .replace(/^\[\s*/, "")
        .replace(/\s*\]$/, "")
        .replace(/^["']|["']$/g, "");
      return unwrapped
        .split(/\s*["']?\s*,\s*["']?\s*/)
        .map((item) => item.replace(/^["']|["']$/g, "").trim())
        .filter(Boolean);
    }),
  );

const isGraphRAGSummaryQuery = (query: string): boolean => {
  const text = query.replace(/\s+/g, " ").trim();
  return (
    /\bsummar(?:y|ize|ise|izes|ises|ized|ised|izing|ising)\b/i.test(text) ||
    /\bkey evidence,\s*entities,\s*relationships\b/i.test(text) ||
    /\bnamed Entities,\s*Top relationships,\s*Best path,\s*and Source proof\b/i.test(text)
  );
};

const asksForRequestedEvidenceAnswer = (query: string): boolean => {
  if (isGraphRAGSummaryQuery(query)) return false;
  const text = query.replace(/\s+/g, " ").toLowerCase();
  const hasQuestionIntent =
    /\b(?:what|which|show|list|audit|inspect|review|cite|extract|extracted|extraction)\b/.test(text);
  const hasRichEvidenceFamily =
    /\b(?:charts?|tables?|forms?|form fields?|fields?|styled text|formatted text|bold|italic|underlined|layouts?|layout relationships?|visual relationships?|visual layout|nearby|spatial|positions?|visually near|slide evidence|source evidence|evidence famil(?:y|ies)|evidence types?|extracted evidence)\b/.test(text);
  const asksExtractionOutput =
    /\bwhat\b[\s\S]*\bextract(?:ed|ion)?\b/.test(text) ||
    /\bextract(?:ed|ion)?\b[\s\S]*\bevidence\b/.test(text);
  return hasRichEvidenceFamily && (hasQuestionIntent || asksExtractionOutput);
};

const isUsefulSummaryEntity = (value: string): boolean => {
  const text = value.replace(/\s+/g, " ").trim();
  if (text.length < 3 || text.length > 80) return false;
  if (/^source document$/i.test(text)) return false;
  if (/^https?:\/\//i.test(text)) return false;
  if (DOCUMENT_FILE_EXTENSION_PATTERN.test(text)) return false;
  if (/^(?:slide|page|table|chart|row|cell|source|target|document)$/i.test(text)) return false;
  return /[A-Za-z]/.test(text);
};

const normalizeLocation = (metadata: GraphRAGRecord): string => {
  const sourceLocations = Array.isArray(metadata.source_locations)
    ? metadata.source_locations.map(asRecord)
    : [];
  const primary = sourceLocations[0] || metadata;
  const slide = primary.slide ?? metadata.slide;
  const slideTitle =
    asString(primary.slide_title) || asString(metadata.slide_title);
  if (slide && slideTitle) return `slide ${slide}: ${slideTitle}`;
  if (slide) return `slide ${slide}`;

  const location = asString(primary.location) || asString(metadata.location);
  if (location) return location;

  const page = primary.page ?? metadata.page;
  if (page) return `page ${page}`;

  const table = primary.table ?? metadata.table;
  const row = primary.row ?? metadata.row;
  if (table && row) return `table ${table}, row ${row}`;
  if (table) return `table ${table}`;
  return "";
};

const firstProvenanceText = (
  relationship: GraphRAGDocumentFallbackRelationship,
): string =>
  [
    ...stringList(relationship.provenance_summary),
    relationship.provenance,
    asString((relationship as GraphRAGDocumentEvidenceCard).citation_label),
  ]
    .map((value) => String(value || "").trim())
    .find(Boolean) || "";

const explicitRelationshipSourceLabel = (
  relationship: GraphRAGDocumentFallbackRelationship,
): string => {
  const metadata = asRecord(relationship.metadata);
  const location = asRecord(relationship.location);
  const provenance = firstProvenanceText(relationship);
  return (
    asString(location.filename) ||
    asString(metadata.local_graph_filename) ||
    asString(metadata.filename) ||
    provenance.match(/(?:^|[;\s])file=([^;,\n]+)/)?.[1]?.trim() ||
    provenance.match(/\b([\w./ -]+\.(?:pdf|pptx?|docx?|xlsx?|csv|md|html?|txt|json|xml))\b/i)?.[1]?.trim() ||
    ""
  );
};

const provenanceLocation = (
  relationship: GraphRAGDocumentFallbackRelationship,
): string => {
  const text = firstProvenanceText(relationship);
  if (!text) return "";
  const slide =
    text.match(/(?:^|[;\s])slide=(?:slide\s*)?([^;\n]+)/i)?.[1]?.trim() ||
    text.match(/\bslide\s+(\d+[^;\n]*)/i)?.[0]?.trim() ||
    "";
  if (slide) return /^slide\b/i.test(slide) ? slide : `slide ${slide}`;
  const page =
    text.match(/(?:^|[;\s])page=(\d+)/i)?.[1]?.trim() ||
    text.match(/\bpage\s+(\d+)/i)?.[1]?.trim() ||
    "";
  if (page) return `page ${page}`;
  return "";
};

const relationshipSourceLabel = (
  relationship: GraphRAGDocumentFallbackRelationship,
  fallbackSourceLabel?: string,
): string => {
  return (
    explicitRelationshipSourceLabel(relationship) ||
    fallbackSourceLabel ||
    "selected document"
  );
};

const relationshipCitation = (
  relationship: GraphRAGDocumentFallbackRelationship,
  fallbackSourceLabel?: string,
): string => {
  const metadata = {
    ...asRecord(relationship.location),
    ...asRecord(relationship.metadata),
  };
  const source = relationshipSourceLabel(relationship, fallbackSourceLabel);
  const location = normalizeLocation(metadata) || provenanceLocation(relationship);
  return [source, location].filter(Boolean).join(", ");
};

const displaySourceLabel = (value: unknown): string => {
  const text = asString(value);
  if (!text) return "";
  const fileMatch = text.match(
    /(?:^|[\\/;,\s])([^\\/;,\s]+\.(?:pdf|pptx?|docx?|xlsx?|csv|md|html?|txt|json|xml))(?:$|[?#;,\s])/i,
  );
  const source = fileMatch?.[1] || text.split(/[\\/]/).pop() || text;
  return source.replace(/[.;,)]+$/g, "").trim();
};

const explicitChunkSourceLabel = (
  chunk: GraphRAGDocumentFallbackChunk,
): string => {
  const metadata = asRecord(chunk.metadata);
  return [
    metadata.filename,
    metadata.local_graph_filename,
    metadata.source_filename,
    metadata.source_file,
    metadata.file_name,
    metadata.document_filename,
    chunk.document_title,
    chunk.document_id,
  ]
    .map(displaySourceLabel)
    .find(Boolean) || "";
};

const isBatchSourceLabel = (value: string): boolean =>
  /\b(?:selected\s+)?(?:batch|document set|documents?|files?|sources?)\b/i.test(
    value,
  );

const preferredChunkSourceLabel = (
  chunk: GraphRAGDocumentFallbackChunk,
  fallbackSourceLabel?: string,
): string => {
  const explicitSource = explicitChunkSourceLabel(chunk);
  const fallback = displaySourceLabel(fallbackSourceLabel);
  if (!fallback || isBatchSourceLabel(fallback)) {
    return explicitSource || fallback || "selected document";
  }
  if (DOCUMENT_FILE_EXTENSION_PATTERN.test(fallback)) return fallback;
  return explicitSource || fallback || "selected document";
};

const relationshipEvidence = (
  relationship: GraphRAGDocumentFallbackRelationship,
): string => {
  const rawEvidence = [
    relationship.evidence,
    relationship.rationale,
    ...stringList(relationship.provenance_summary),
    relationship.provenance,
    asString(asRecord(relationship.metadata).evidence),
  ]
    .map((value) => String(value || "").trim())
    .find(Boolean);
  return rawEvidence ? cleanGraphRAGTraceSampleText(rawEvidence, 260) : "";
};

const relationStatement = (
  source: string,
  relationKey: string,
  relation: string,
  target: string,
): string => {
  const cleanSource = stripTrailingPunctuation(source);
  const cleanTarget = stripTrailingPunctuation(target);
  if (relationKey === "relation") {
    const relationTarget = cleanTarget.match(/^(gates?|requires?|supports?|blocks?|drives?)\s+(.+)$/i);
    if (relationTarget) {
      return `${cleanSource} ${relationTarget[1].toLowerCase()} ${stripTrailingPunctuation(relationTarget[2])}`;
    }
  }

  if (relationKey === "groups_visual") {
    return `${cleanSource} groups ${cleanTarget}`;
  }

  if (ATTRIBUTE_RELATION_PATTERN.test(relationKey)) {
    return `${cleanSource} ${relation} is ${cleanTarget}`;
  }

  return `${cleanSource} ${relation} ${cleanTarget}`;
};

const scoreRelationshipCandidate = (
  relationship: GraphRAGDocumentFallbackRelationship,
  statement: string,
  evidence: string,
  terms: string[],
): number => {
  const relationKey = normalizeRelationKey(
    relationship.relation || relationship.relationship_type,
  );
  const text = [
    statement,
    evidence,
    relationship.source,
    relationship.target,
    relationship.relation,
    relationship.relationship_type,
    ...(relationship.matched_query_terms || []),
    ...(relationship.specific_matched_query_terms || []),
  ]
    .join(" ")
    .toLowerCase();
  const queryScore = terms.reduce(
    (total, term) => total + (text.includes(term) ? 8 : 0),
    0,
  );
  const relationScore = DIRECT_RELATION_PATTERN.test(relationKey)
    ? 34
    : WEAK_RELATION_PATTERN.test(relationKey)
      ? -18
      : 8;
  const evidenceScore = evidence ? 14 : 0;
  const numericScore =
    typeof relationship.score === "number"
      ? relationship.score * 8
      : typeof relationship.weight === "number"
        ? relationship.weight * 8
        : 0;
  return relationScore + evidenceScore + queryScore + numericScore;
};

const relationshipCandidate = (
  relationship: GraphRAGDocumentFallbackRelationship,
  query: string,
  fallbackSourceLabel?: string,
): FallbackCandidate | null => {
  const source = asString(relationship.source);
  const target = asString(relationship.target);
  const relationKey = normalizeRelationKey(
    relationship.relation || relationship.relationship_type,
  );
  if (!source || !target || !relationKey) return null;

  const relation = formatGraphReferenceRelation(
    relationship.relation || relationship.relationship_type,
  );
  const statement = sentenceCase(
    relationStatement(source, relationKey, relation, target),
  );
  const relationshipSummary = `${source} ${relationKey} ${target}`;
  if (
    isLowValueGraphRAGTraceSampleText(relationshipSummary) ||
    isLowValueGraphRAGTraceSampleText(`${source} -> ${target}`)
  ) {
    return null;
  }

  const evidence = relationshipEvidence(relationship);
  if (evidence && isLowValueGraphRAGTraceSampleText(evidence)) return null;

  return {
    source,
    relation,
    relationKey,
    target,
    statement,
    evidence,
    citation: relationshipCitation(relationship, fallbackSourceLabel),
    score: scoreRelationshipCandidate(
      relationship,
      statement,
      evidence,
      queryTerms(query),
    ),
  };
};

const collectRelationshipCandidates = (
  context: GraphRAGDocumentFallbackContext,
): FallbackCandidate[] => {
  const records = [
    ...(context.graphReasoning?.answer_plan?.evidence_cards || []),
    ...(context.graphReasoning?.direct_relationships || []),
    ...(context.relationships || []),
    ...(context.graphReasoning?.paths || []).flatMap((path) => path.relationships || []),
  ];
  const seen = new Set<string>();
  return records
    .map((relationship) =>
      relationshipCandidate(relationship, context.query, context.sourceLabel),
    )
    .filter((candidate): candidate is FallbackCandidate => Boolean(candidate))
    .filter((candidate) => {
      const key = [
        candidate.source,
        candidate.relationKey,
        candidate.target,
      ]
        .join("|")
        .toLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .sort((a, b) => b.score - a.score);
};

const REQUESTED_EVIDENCE_ROLE_KEYS = new Set([
  "requested_evidence_family",
  "pipeline_evidence_family",
]);

const REQUESTED_EVIDENCE_FAMILY_ALIASES: Array<{
  key: string;
  aliases: RegExp[];
}> = [
  { key: "charts", aliases: [/\bcharts?\b/i, /\bplots?\b/i, /\bseries\b/i, /\baxis\b/i, /\btrendline\b/i] },
  { key: "tables", aliases: [/\btables?\b/i, /\brows?\b/i, /\bcolumns?\b/i, /\bcells?\b/i] },
  { key: "form_fields", aliases: [/\bform fields?\b/i, /\bforms?\b/i, /\bfields?\b/i, /\bcheckbox(?:es)?\b/i] },
  { key: "styled_text", aliases: [/\bstyled text\b/i, /\bformatted text\b/i, /\bemphasis\b/i, /\bemphasized\b/i, /\bbold\b/i, /\bitalic\b/i, /\bhighlight\b/i] },
  { key: "connectors", aliases: [/\bconnectors?\b/i, /\barrows?\b/i, /\bflows?\b/i, /\brelationship line\b/i] },
  { key: "speaker_notes", aliases: [/\bspeaker notes?\b/i, /\bpresenter notes?\b/i, /\bnotes slide\b/i] },
  { key: "comments", aliases: [/\bcomments?\b/i, /\bannotations?\b/i, /\breviewer\b/i] },
  { key: "media", aliases: [/\bmedia\b/i, /\baudio\b/i, /\bvideo\b/i, /\brecording\b/i, /\bembedded media\b/i] },
  { key: "attachments", aliases: [/\battachments?\b/i, /\bembedded files?\b/i, /\bembedded objects?\b/i, /\battached documents?\b/i] },
  { key: "links", aliases: [/\blinks?\b/i, /\burls?\b/i, /\buris?\b/i, /\bhref\b/i] },
  { key: "images", aliases: [/\bimages?\b/i, /\bpictures?\b/i, /\bfigures?\b/i, /\bscreenshots?\b/i] },
  { key: "layout", aliases: [/\blayout\b/i, /\bspatial\b/i, /\bvisual(?:ly)? near\b/i, /\bpositions?\b/i] },
];

const requestedEvidenceFamilyOrder = (query: string): Map<string, number> => {
  const text = query.replace(/\s+/g, " ").trim();
  const matches: Array<{ key: string; index: number }> = [];
  REQUESTED_EVIDENCE_FAMILY_ALIASES.forEach(({ key, aliases }) => {
    const indexes = aliases
      .map((alias) => {
        const match = text.match(alias);
        return match?.index ?? -1;
      })
      .filter((index) => index >= 0);
    if (indexes.length > 0) {
      matches.push({ key, index: Math.min(...indexes) });
    }
  });

  return new Map(
    matches
      .sort((a, b) => a.index - b.index)
      .map((match, index) => [match.key, index]),
  );
};

const requestedEvidenceCardKind = (card: GraphRAGDocumentEvidenceCard): string =>
  normalizeRelationKey(card.kind || card.relationship_type || card.role || card.title);

const requestedEvidenceCardTitle = (card: GraphRAGDocumentEvidenceCard): string => {
  const title = asString(card.title);
  if (title) return title;
  const kind = requestedEvidenceCardKind(card);
  if (!kind) return "Evidence";
  return kind
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
};

const isRequestedEvidenceCard = (card: GraphRAGDocumentEvidenceCard): boolean => {
  const role = normalizeRelationKey(card.role || card.relationship_type);
  return REQUESTED_EVIDENCE_ROLE_KEYS.has(role) || Boolean(card.kind && card.title);
};

const requestedEvidenceScore = (
  card: GraphRAGDocumentEvidenceCard,
  terms: string[],
): number => {
  const haystack = [
    card.title,
    card.kind,
    card.role,
    card.relationship_type,
    card.evidence,
    card.text,
    card.citation_label,
    ...stringList(card.provenance_summary),
    ...(card.evidence_families || []),
    ...(card.categories || []),
  ]
    .join(" ")
    .toLowerCase();
  const termScore = terms.reduce(
    (total, term) => total + (haystack.includes(term) ? 8 : 0),
    0,
  );
  const numericScore =
    typeof card.score === "number" ? card.score * 12 : 0;
  return 40 + termScore + numericScore;
};

const isTruncatedEvidenceLocation = (value: string): boolean => {
  const text = value.replace(/\s+/g, " ").trim();
  if (!text) return false;
  if (/\.\.\.?$/.test(text)) return true;
  return /\b(?:slide|page|sheet)\s+\d+\s*:\s*[A-Za-z]{1,4}\.?$/i.test(text);
};

const requestedEvidenceLocationFromText = (
  card: GraphRAGDocumentEvidenceCard,
  metadata: GraphRAGRecord,
  provenanceLocation: string,
): string => {
  const text = [
    asString(card.evidence),
    asString(card.text),
    asString(card.citation_label),
  ].filter(Boolean).join("; ");
  const explicitSlide = text.match(/(?:^|;\s*)slide\s*:\s*([^;]+)/i)?.[1]?.trim() || "";
  if (explicitSlide && !/^\d+$/.test(explicitSlide) && !isTruncatedEvidenceLocation(`slide: ${explicitSlide}`)) {
    const slideNumber =
      String(metadata.slide || "").trim() ||
      provenanceLocation.match(/\bslide\s+(\d+)/i)?.[1]?.trim() ||
      "";
    return slideNumber ? `slide ${slideNumber}: ${explicitSlide}` : `slide: ${explicitSlide}`;
  }

  const fullSlide = text.match(/\bslide\s+(\d+)\s*:\s*([^;]+)/i);
  const slideNumber = fullSlide?.[1]?.trim() || "";
  const slideTitle = fullSlide?.[2]?.trim() || "";
  if (slideNumber && slideTitle && !isTruncatedEvidenceLocation(`slide ${slideNumber}: ${slideTitle}`)) {
    return `slide ${slideNumber}: ${slideTitle}`;
  }
  return "";
};

const requestedEvidenceCardCitation = (
  card: GraphRAGDocumentEvidenceCard,
  fallbackSourceLabel?: string,
): string => {
  const location = asRecord(card.location);
  const metadata = {
    ...asRecord(card.metadata),
    ...location,
  };
  const source =
    fallbackSourceLabel ||
    asString(location.filename) ||
    asString(metadata.filename) ||
    asString(metadata.local_graph_filename);
  const provenanceLocation = stringList(card.provenance_summary)[0] || "";
  const structuredLocation = normalizeLocation(metadata);
  const evidenceLocation = requestedEvidenceLocationFromText(
    card,
    metadata,
    provenanceLocation,
  );
  const locationText =
    isTruncatedEvidenceLocation(provenanceLocation) && (structuredLocation || evidenceLocation)
      ? structuredLocation || evidenceLocation
      : provenanceLocation || structuredLocation || evidenceLocation;
  if (!source) return locationText;
  if (!locationText) return source;
  return locationText.toLowerCase().includes(source.toLowerCase())
    ? locationText
    : [source, locationText].filter(Boolean).join(", ");
};

const requestedEvidenceMetadata = (
  card: GraphRAGDocumentEvidenceCard,
): GraphRAGRecord => ({
  ...asRecord(card.metadata),
  ...asRecord(card.location),
});

const normalizeRequestedEvidenceText = (value: string): string =>
  value
    .replace(/\(\s*\$([A-Za-z]+)\s*\)/g, "(USD $1)")
    .replace(/\$\s*#/g, "USD #")
    .replace(/\$/g, "USD ")
    .replace(/\s+/g, " ")
    .replace(/\s+([,.;:])/g, "$1")
    .trim();

const colonFieldValues = (text: string, field: string): string[] => {
  const pattern = new RegExp(
    `(?:^|;\\s*)${escapeRegExp(field)}\\s*:\\s*([^;]+)`,
    "gi",
  );
  return uniqueStrings(
    Array.from(text.matchAll(pattern), (match) =>
      normalizeRequestedEvidenceText(match[1] || ""),
    ),
  );
};

const colonFieldValue = (text: string, field: string): string =>
  colonFieldValues(text, field)[0] || "";

const metadataText = (
  metadata: GraphRAGRecord,
  keys: string[],
): string =>
  keys
    .map((key) => asString(metadata[key]))
    .map(normalizeRequestedEvidenceText)
    .find(Boolean) || "";

const readableAssetName = (value: string): string => {
  const display = displaySourceLabel(value);
  if (!display) return "";
  if (/^(?:source|package|part|embedded|media|image|object)\b/i.test(display)) {
    return DOCUMENT_FILE_EXTENSION_PATTERN.test(display) ? display : "";
  }
  if (/^(?:text|image|audio|video|application)\/[-+.\w]+$/i.test(display)) return "";
  return display;
};

const rawFieldText = (text: string, field: string): string => {
  const match = text.match(
    new RegExp(`(?:^|;\\s*)${escapeRegExp(field)}\\s*:\\s*([\\s\\S]+)$`, "i"),
  );
  return normalizeRequestedEvidenceText(match?.[1] || "");
};

const cleanEvidencePhrase = (value: string): string =>
  normalizeRequestedEvidenceText(value)
    .replace(/^\[[a-z][a-z0-9_ -]*(?:\s*\|[^\]]*)?\]\s*/i, "")
    .replace(/^(?:evidence|evidence text|semantic text|text)\s*[:=]\s*/i, "")
    .replace(/^(?:speaker\s+note|note|comment|image|media|attachment)\s*:\s*/i, "")
    .replace(/^\[[a-z][a-z0-9_ -]*(?:\s*\|[^\]]*)?\]\s*/i, "")
    .replace(/^slide\s+\d+\s+(?:image|media|attachment)\s+\d+\s*:\s*/i, "")
    .replace(/\s*;\s*(?:provenance|location)\s*=.*$/i, "")
    .replace(/\s*;\s*(?:slide|page|sheet|table|chart)\s*:\s*[^;]+$/i, "")
    .trim();

const readableEvidenceFragments = (value: string): string[] =>
  uniqueStrings(
    normalizeRequestedEvidenceText(value)
      .split(/\s*;\s*/)
      .map(cleanEvidencePhrase)
      .filter((fragment) => {
        if (!fragment) return false;
        if (/^(?:slide|page|sheet|table|chart|row|cell)\b/i.test(fragment)) return false;
        if (/^(?:media|image|embedded file|attachment)\s+(?:kind|mime|part|filename)\b/i.test(fragment)) return false;
        if (/^(?:text|image|audio|video|application)\/[-+.\w]+$/i.test(fragment)) return false;
        if (/\b(?:ppt|xl|word)\/[^\s;,)]+/i.test(fragment)) return false;
        if (/^\d+(?:\.\d+)?x\d+(?:\.\d+)?\s+image\b/i.test(fragment)) return false;
        if (/\b(?:mime|relationship id|part)\b/i.test(fragment)) return false;
        if (/^(?:source|package|part|embedded|media|image|object)\b/i.test(fragment)) return false;
        return true;
      }),
  );

const evidenceTailFragments = (rawEvidence: string, field: string): string[] =>
  readableEvidenceFragments(rawFieldText(rawEvidence, field));

const evidenceAssignmentValues = (text: string): string[] => {
  const pattern =
    /(?:^|;\s*)evidence=([\s\S]*?)(?=;\s*(?:evidence|provenance|location)\s*=|$)/gi;
  return uniqueStrings(
    Array.from(text.matchAll(pattern), (match) =>
      normalizeRequestedEvidenceText(match[1] || "")
        .replace(/\s*\|\s*/g, "; ")
        .replace(/\s*;\s*/g, "; "),
    ),
  );
};

const requestedEvidenceFallbackText = (value: string): string =>
  cleanGraphRAGTraceSampleText(
    normalizeRequestedEvidenceText(value)
      .replace(/\s+--(?:\[[^\]]+\]|[a-z0-9_\s-]+)-->\s+/gi, " -> ")
      .replace(/\b(?:evidence|provenance|location)\s*=\s*/gi, "")
      .replace(/\s*;\s*/g, "; "),
    320,
  );

const requestedChartEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const semanticText = asString(metadata.semantic_text);
  const chartTitle =
    asString(metadata.chart_title) ||
    colonFieldValue(rawEvidence, "chart") ||
    rawEvidence.match(/\bchart\s+([^;]+)/i)?.[1]?.trim() ||
    "";
  const axes = uniqueStrings([
    asString(metadata.chart_axis_title),
    ...colonFieldValues(rawEvidence, "axis"),
    ...Array.from(
      semanticText.matchAll(/\b(?:value|category)\s+axis\s*;\s*([^;\n]+)/gi),
      (match) => normalizeRequestedEvidenceText(match[1] || ""),
    ),
    ...Array.from(
      semanticText.matchAll(/\b(?:value|category)\s+axis\s*:\s*([^;\n]+)/gi),
      (match) => normalizeRequestedEvidenceText(match[1] || ""),
    ),
    ...Array.from(
      rawEvidence.matchAll(/\b(?:value|category)\s+axis\s*:\s*([^;]+)/gi),
      (match) => normalizeRequestedEvidenceText(match[1] || ""),
    ),
  ]).filter((axis) => !/^(?:position|format)$/i.test(axis));
  const parts = [
    chartTitle ? `${chartTitle} chart` : "",
    axes.length > 0 ? `axes include ${humanList(axes.slice(0, 3))}` : "",
  ].filter(Boolean);
  return parts.join("; ");
};

const requestedTableEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const table = String(metadata.table ?? colonFieldValue(rawEvidence, "table") ?? "")
    .trim();
  const row = String(metadata.row ?? colonFieldValue(rawEvidence, "row") ?? "")
    .trim();
  const cellMatch = rawEvidence.match(/\b(R\d+C\d+)\s*:\s*([^;]+)/i);
  const evidenceValues = evidenceAssignmentValues(rawEvidence);
  const chartData = evidenceValues.find((value) =>
    /\b(?:metric|source owner|confidence)\s*:/i.test(value),
  );
  const primaryValue = evidenceValues.find((value) =>
    value && value !== chartData && !/^slide\s+\d+/i.test(value),
  );
  const parts = [
    table ? `Table ${table}${row ? ` row ${row}` : ""}` : "",
    cellMatch
      ? `cell ${cellMatch[1]} = ${normalizeRequestedEvidenceText(cellMatch[2] || "")}`
      : primaryValue
        ? primaryValue
        : "",
    chartData ? `chart data includes ${chartData.replace(/\s*;\s*/g, ", ")}` : "",
  ].filter(Boolean);
  return parts.join("; ");
};

const requestedFormEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const label =
    asString(metadata.label) ||
    asString(metadata.form_field_alias) ||
    colonFieldValue(rawEvidence, "label");
  const checkedRaw =
    typeof metadata.form_field_checked === "boolean"
      ? String(metadata.form_field_checked)
      : colonFieldValue(rawEvidence, "checked");
  const checked =
    checkedRaw.toLowerCase() === "true"
      ? "checked"
      : checkedRaw.toLowerCase() === "false"
        ? "not checked"
        : "";
  const value = asString(metadata.value) || colonFieldValue(rawEvidence, "value");
  const valueMatch =
    rawEvidence.match(/\b([^;]+?)\s+--has value-->\s*([^;]+)/i) ||
    rawEvidence.match(/evidence=([^:;]+):\s*([^;]+)/i);
  const valueLabel = normalizeRequestedEvidenceText(valueMatch?.[1] || "");
  const valueText = normalizeRequestedEvidenceText(valueMatch?.[2] || "");
  const parts = [
    label ? `${label}${checked ? ` is ${checked}` : value ? ` = ${value}` : ""}` : "",
    valueLabel && valueText ? `${valueLabel} is ${valueText}` : "",
  ].filter(Boolean);
  return uniqueStrings(parts).join("; ");
};

const requestedStyledTextEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const emphasisTarget =
    rawEvidence.match(/--emphasizes-->\s*([^;]+)/i)?.[1]?.trim() ||
    evidenceAssignmentValues(rawEvidence)[0] ||
    "";
  const style =
    asString(metadata.style_summary) ||
    colonFieldValue(rawEvidence, "style");
  const emphasisTerms = readableListItems([
    ...stringList(metadata.emphasis_terms),
    colonFieldValue(rawEvidence, "emphasis"),
  ]);
  const parts = [
    emphasisTarget && style
      ? `${normalizeRequestedEvidenceText(emphasisTarget)} is ${style}`
      : normalizeRequestedEvidenceText(emphasisTarget),
    emphasisTerms.length > 0
      ? `emphasized terms include ${humanList(emphasisTerms.slice(0, 3))}`
      : "",
  ].filter(Boolean);
  return parts.join("; ");
};

const requestedConnectorEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const edgeMatch = rawEvidence.match(/\b([^;\n]+?)\s*->\s*([^:;\n]+)\s*:\s*([^;\n]+)/i);
  const source =
    stripTrailingPunctuation(
      asString(metadata.source_label) ||
      asString(card.source) ||
      normalizeRequestedEvidenceText(edgeMatch?.[1] || ""),
    );
  const target =
    stripTrailingPunctuation(
      asString(metadata.target_label) ||
      asString(card.target) ||
      normalizeRequestedEvidenceText(edgeMatch?.[2] || ""),
    );
  const relation =
    asString(metadata.connector_title) ||
    asString(card.relation) ||
    normalizeRequestedEvidenceText(edgeMatch?.[3] || "");
  const description =
    asString(metadata.connector_description) ||
    rawEvidence.match(/\bdescription\s*:\s*([^;]+)/i)?.[1]?.trim() ||
    "";
  const connectorLabels = uniqueStrings([
    asString(metadata.connector_title),
    ...colonFieldValues(rawEvidence, "connector"),
    ...colonFieldValues(rawEvidence, "connector title"),
  ])
    .map(cleanEvidencePhrase)
    .filter(Boolean);
  const readableRelation = formatGraphReferenceRelation(relation);
  const relationship = source && target
    ? sentenceCase(relationStatement(
      source,
      normalizeRelationKey(readableRelation),
      readableRelation,
      target,
    ))
    : "";
  const parts = [
    relationship,
    description && !relationship.toLowerCase().includes(description.toLowerCase())
      ? description
      : "",
  ].filter(Boolean);
  const summary = uniqueStrings(parts).join("; ");
  if (summary) return summary;
  return connectorLabels.length > 0
    ? `connector labels include ${humanList(connectorLabels.slice(0, 3))}`
    : "";
};

const requestedLayoutEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const layoutLabel = (value: unknown): string =>
    asString(value).replace(/_/g, " ");
  const source =
    asString(card.source) ||
    rawEvidence.match(/^([^;]+?)\s+is visually near\b/i)?.[1]?.trim() ||
    "";
  const target =
    asString(metadata.related_label) ||
    asString(card.target) ||
    rawEvidence.match(/\bis visually near\s+([^;(]+)/i)?.[1]?.trim() ||
    "";
  const distanceValue = String(metadata.visual_distance_pct ?? "").trim();
  const distance =
    distanceValue ||
    rawEvidence.match(/\bdistance\s+([0-9.]+%?)/i)?.[1]?.trim() ||
    "";
  const zones = uniqueStrings([
    layoutLabel(metadata.layout_zone),
    layoutLabel(metadata.target_layout_zone),
  ]);
  const parts = [
    source && target
      ? `${source} is visually near ${target}`
      : "",
    distance ? `distance ${distance.endsWith("%") ? distance : `${distance}%`}` : "",
    zones.length > 0 ? `zones ${zones.join(" to ")}` : "",
  ].filter(Boolean);
  return parts.join("; ");
};

const requestedCommentEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const rawAuthor =
    metadataText(metadata, ["comment_author", "author", "reviewer"]) ||
    rawEvidence.match(/\b([A-Z][^:;\n]{2,60}?\s+Reviewer)\s*:/)?.[1]?.trim() ||
    "";
  const author = cleanEvidencePhrase(rawAuthor)
    .replace(/^.*?--[^>]*->\s*/i, "")
    .trim();
  const body =
    metadataText(metadata, ["comment_text", "comment", "text"]) ||
    evidenceAssignmentValues(rawEvidence)[0] ||
    rawEvidence.match(/\b[A-Z][^:;\n]{2,60}?\s+Reviewer\s*:\s*([^;]+)/)?.[1]?.trim() ||
    metadataText(metadata, ["semantic_text"]).match(/\b[A-Z][^:;\n]{2,60}?\s+Reviewer\s*:\s*([^.;]+[.]?)/)?.[1]?.trim() ||
    "";
  const cleanBody = cleanEvidencePhrase(body)
    .replace(/^.*?--[^>]*->\s*/i, "")
    .trim();
  if (author && cleanBody && !cleanBody.toLowerCase().startsWith(author.toLowerCase())) {
    return `${author}: ${cleanBody}`;
  }
  return cleanBody || author;
};

const requestedSpeakerNotesEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const note =
    metadataText(metadata, [
      "note_text",
      "speaker_note",
      "speaker_notes",
      "notes_text",
    ]) ||
    evidenceAssignmentValues(rawEvidence)[0] ||
    rawEvidence.match(/\bSpeaker note\s*:\s*([^;]+)/i)?.[1]?.trim() ||
    metadataText(metadata, ["semantic_text"]).match(/\bSpeaker note\s*:\s*([^.;]+[.]?)/i)?.[1]?.trim() ||
    "";
  return cleanEvidencePhrase(note);
};

const requestedEmbeddedAssetName = (
  metadata: GraphRAGRecord,
  rawEvidence: string,
): string =>
  [
    metadataText(metadata, [
      "embedded_file_filename",
      "attachment_filename",
      "attached_filename",
      "media_filename",
      "media_name",
      "image_filename",
      "image_name",
      "asset_filename",
      "file_name",
    ]),
    colonFieldValue(rawEvidence, "embedded file filename"),
    colonFieldValue(rawEvidence, "attachment filename"),
    colonFieldValue(rawEvidence, "media filename"),
    colonFieldValue(rawEvidence, "media name"),
    colonFieldValue(rawEvidence, "image filename"),
    colonFieldValue(rawEvidence, "image name"),
    colonFieldValue(rawEvidence, "image part"),
    colonFieldValue(rawEvidence, "media part"),
  ]
    .map(readableAssetName)
    .find(Boolean) || "";

const requestedEmbeddedAssetMime = (
  metadata: GraphRAGRecord,
  rawEvidence: string,
): string =>
  [
    metadataText(metadata, [
      "embedded_file_mime",
      "attachment_mime",
      "media_mime",
      "image_mime",
      "mime_type",
      "content_type",
    ]),
    colonFieldValue(rawEvidence, "embedded file mime"),
    colonFieldValue(rawEvidence, "attachment mime"),
    colonFieldValue(rawEvidence, "media mime"),
    colonFieldValue(rawEvidence, "image mime"),
    colonFieldValue(rawEvidence, "mime"),
    colonFieldValue(rawEvidence, "content type"),
  ]
    .map(normalizeRequestedEvidenceText)
    .find((value) => /^(?:text|image|audio|video|application)\/[-+.\w]+$/i.test(value)) || "";

const requestedImageEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const asset = requestedEmbeddedAssetName(metadata, rawEvidence);
  const semanticFragments = readableEvidenceFragments(
    metadataText(metadata, ["semantic_text"]),
  );
  const fragments = uniqueStrings([
    metadataText(metadata, [
      "image_description",
      "image_alt_text",
      "alt_text",
      "description",
    ]),
    ...semanticFragments,
    ...evidenceTailFragments(rawEvidence, "evidence text"),
    ...evidenceAssignmentValues(rawEvidence),
  ].map(cleanEvidencePhrase));
  const title =
    metadataText(metadata, ["image_title", "title", "label"]) ||
    fragments.find((fragment) => fragment.length <= 80 && !DOCUMENT_FILE_EXTENSION_PATTERN.test(fragment)) ||
    "";
  const detail =
    fragments.find((fragment) =>
      fragment &&
      fragment.length > Math.max(20, title.length) &&
      fragment.toLowerCase() !== title.toLowerCase() &&
      !DOCUMENT_FILE_EXTENSION_PATTERN.test(fragment),
    ) ||
    fragments.find((fragment) =>
      fragment &&
      fragment.toLowerCase() !== title.toLowerCase() &&
      !DOCUMENT_FILE_EXTENSION_PATTERN.test(fragment),
    ) || "";
  const subject = title
    ? `${title} image`
    : asset
      ? `Image ${asset}`
      : "Image evidence";
  return uniqueStrings([
    subject,
    detail,
  ]).join("; ");
};

const requestedMediaEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const asset = requestedEmbeddedAssetName(metadata, rawEvidence);
  const mime = requestedEmbeddedAssetMime(metadata, rawEvidence);
  const mediaKind =
    metadataText(metadata, ["media_kind", "kind", "type"]) ||
    colonFieldValue(rawEvidence, "media kind") ||
    "embedded media";
  const fragments = uniqueStrings([
    metadataText(metadata, ["media_description", "description"]),
    ...readableEvidenceFragments(metadataText(metadata, ["semantic_text"])),
    ...evidenceTailFragments(rawEvidence, "evidence text"),
    ...evidenceAssignmentValues(rawEvidence),
  ].map(cleanEvidencePhrase));
  const inferredKind =
    /^embedded media$/i.test(mediaKind) && /^image\//i.test(mime)
      ? "embedded image"
      : /^embedded media$/i.test(mediaKind) && /^audio\//i.test(mime)
        ? "embedded audio"
        : /^embedded media$/i.test(mediaKind) && /^video\//i.test(mime)
          ? "embedded video"
          : mediaKind;
  const displayKind = inferredKind
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/^\w/, (letter) => letter.toUpperCase());
  const mediaTitle = /^embedded image$/i.test(inferredKind)
    ? fragments.find((fragment) =>
      fragment.length <= 80 &&
      !DOCUMENT_FILE_EXTENSION_PATTERN.test(fragment) &&
      !/\b(?:blockers?|before|after|because|highlights?|launch|regional)\b/i.test(fragment),
    ) || ""
    : "";
  const label = mediaTitle
    ? `${displayKind} ${mediaTitle}`
    : asset
      ? `${displayKind} ${asset}`
      : displayKind;
  return uniqueStrings([
    label,
    fragments.find((fragment) => fragment.length > 20 && !/\bimage$/i.test(fragment)) ||
      fragments.find(Boolean) ||
      "",
  ]).join("; ");
};

const requestedAttachmentEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const asset = requestedEmbeddedAssetName(metadata, rawEvidence);
  const fragments = uniqueStrings([
    metadataText(metadata, ["attachment_description", "description"]),
    ...readableEvidenceFragments(metadataText(metadata, ["semantic_text"])),
    ...evidenceTailFragments(rawEvidence, "evidence text"),
    ...evidenceAssignmentValues(rawEvidence),
  ].map(cleanEvidencePhrase));
  const label = asset ? `Embedded file ${asset}` : "Embedded file";
  return uniqueStrings([
    label,
    fragments.find(Boolean) || "",
  ]).join("; ");
};

const cleanUrl = (value: string): string =>
  value.replace(/[.,;:!?]+$/g, "").trim();

const urlsFromText = (value: string): string[] =>
  uniqueStrings(
    Array.from(String(value || "").matchAll(URL_PATTERN), (match) =>
      cleanUrl(match[0] || ""),
    ),
  );

const requestedLinkEvidenceText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const metadata = requestedEvidenceMetadata(card);
  const urls = uniqueStrings([
    asString(metadata.uri),
    asString(metadata.link_uri),
    asString(metadata.link_target),
    asString(card.target),
    ...urlsFromText(rawEvidence),
  ].map(cleanUrl));
  if (urls.length === 0) return "";

  const edgeMatch =
    rawEvidence.match(/(https?:\/\/[^\s;,)]+)\s+links?\s+to\s+https?:\/\/[^\s;,)]+/i) ||
    rawEvidence.match(/(?:^|;\s*)([^;\n]{2,120}?)\s+links?\s+to\s+https?:\/\/[^\s;,)]+/i) ||
    rawEvidence.match(/(?:^|;\s*)([^;\n]{2,120}?)\s+contains\s+link\s+https?:\/\/[^\s;,)]+/i);
  const source = normalizeRequestedEvidenceText(
    asString(card.source) ||
    edgeMatch?.[1] ||
    "",
  ).replace(/^(?:evidence text|link)\s*:\s*/i, "");
  const targetUrls = /^https?:\/\//i.test(source)
    ? urls.filter((url) => url.toLowerCase() !== source.toLowerCase())
    : urls;
  const displayUrls = targetUrls.length > 0 ? targetUrls : urls;
  const targetText = humanList(displayUrls.slice(0, 3));
  const overflow = displayUrls.length > 3 ? ` and ${displayUrls.length - 3} more` : "";
  return source
    ? `${source} links to ${targetText}${overflow}`
    : `Links to ${targetText}${overflow}`;
};

const requestedEvidenceStructuredText = (
  card: GraphRAGDocumentEvidenceCard,
  rawEvidence: string,
): string => {
  const kind = requestedEvidenceCardKind(card);
  if (kind === "charts") return requestedChartEvidenceText(card, rawEvidence);
  if (kind === "tables") return requestedTableEvidenceText(card, rawEvidence);
  if (kind === "form_fields") return requestedFormEvidenceText(card, rawEvidence);
  if (kind === "styled_text") {
    return requestedStyledTextEvidenceText(card, rawEvidence);
  }
  if (kind === "connectors") return requestedConnectorEvidenceText(card, rawEvidence);
  if (kind === "comments") return requestedCommentEvidenceText(card, rawEvidence);
  if (kind === "speaker_notes") return requestedSpeakerNotesEvidenceText(card, rawEvidence);
  if (kind === "images") return requestedImageEvidenceText(card, rawEvidence);
  if (kind === "media") return requestedMediaEvidenceText(card, rawEvidence);
  if (kind === "attachments") return requestedAttachmentEvidenceText(card, rawEvidence);
  if (kind === "links") return requestedLinkEvidenceText(card, rawEvidence);
  if (kind === "layout") return requestedLayoutEvidenceText(card, rawEvidence);
  return "";
};

const requestedEvidenceText = (card: GraphRAGDocumentEvidenceCard): string => {
  const rawEvidence =
    asString(card.evidence) ||
    asString(card.text) ||
    asString(card.citation_label);
  const structured = requestedEvidenceStructuredText(card, rawEvidence);
  if (structured) {
    return cleanGraphRAGTraceSampleText(
      normalizeRequestedEvidenceText(structured),
      320,
    );
  }

  const citationSummary = asString(card.citation_label)
    .split("|")
    .map((part) => part.trim())
    .find((part) => part && part !== card.title && !/\.(?:pdf|pptx?|docx?|xlsx?)\b/i.test(part));
  return requestedEvidenceFallbackText(
    citationSummary || rawEvidence,
  );
};

const collectRequestedEvidenceCandidates = (
  context: GraphRAGDocumentFallbackContext,
): RequestedEvidenceCandidate[] => {
  const terms = queryTerms(context.query);
  const requestedOrder = requestedEvidenceFamilyOrder(context.query);
  const cards = context.graphReasoning?.answer_plan?.evidence_cards || [];
  const seen = new Set<string>();

  return cards
    .filter(isRequestedEvidenceCard)
    .map((card, sourceIndex): RequestedEvidenceCandidate | null => {
      const kind = requestedEvidenceCardKind(card);
      const title = requestedEvidenceCardTitle(card);
      const evidence = requestedEvidenceText(card);
      if (!title || !evidence) return null;
      const key = `${kind}:${title}`.toLowerCase();
      if (seen.has(key)) return null;
      seen.add(key);
      return {
        kind,
        title,
        evidence,
        citation: requestedEvidenceCardCitation(card, context.sourceLabel),
        score: requestedEvidenceScore(card, terms),
        sourceIndex,
      };
    })
    .filter((candidate): candidate is RequestedEvidenceCandidate =>
      Boolean(candidate),
    )
    .sort((a, b) => {
      const leftOrder = requestedOrder.get(a.kind);
      const rightOrder = requestedOrder.get(b.kind);
      const leftRequested = leftOrder !== undefined;
      const rightRequested = rightOrder !== undefined;
      if (leftRequested || rightRequested) {
        return (
          (leftRequested ? leftOrder : Number.MAX_SAFE_INTEGER) -
            (rightRequested ? rightOrder : Number.MAX_SAFE_INTEGER) ||
          b.score - a.score ||
          a.sourceIndex - b.sourceIndex ||
          a.title.localeCompare(b.title)
        );
      }
      return b.score - a.score || a.sourceIndex - b.sourceIndex || a.title.localeCompare(b.title);
    });
};

const requestedEvidenceInlineCitation = (citation: string): string => {
  const parts = citation
    .split(",")
    .map((part) => part.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .filter((part) => !/\.(?:pdf|pptx?|docx?|xlsx?|csv|md|txt|json|xml)\b/i.test(part))
    .filter((part) => !/^selected document$/i.test(part));
  const primaryLocation =
    parts.find((part) => /^(?:slide|page|sheet)\s+\d+/i.test(part)) ||
    parts.find((part) => /^(?:table|chart|row|cell)\b/i.test(part)) ||
    parts[0] ||
    "";
  return primaryLocation;
};

const requestedEvidenceBullet = (
  candidate: RequestedEvidenceCandidate,
): string => {
  const evidence = stripTrailingPunctuation(candidate.evidence);
  const citation = requestedEvidenceInlineCitation(candidate.citation);
  const suffix = citation ? ` (${citation}).` : ".";
  return `- **${candidate.title}**: ${evidence}${suffix}`;
};

const repairTruncatedEvidenceLocations = (value: string): string => {
  const fullSlideTitles = new Map<string, string[]>();
  Array.from(value.matchAll(/\bslide\s+(\d+)\s*:\s*([^.;()\n]+)(?=[.;),]|\s*$)/gi))
    .forEach((match) => {
      const slide = match[1]?.trim();
      const title = normalizeRequestedEvidenceText(match[2] || "");
      if (!slide || !title || title.includes("...") || title.length < 5) return;
      const titles = fullSlideTitles.get(slide) || [];
      if (!titles.some((candidate) => candidate.toLowerCase() === title.toLowerCase())) {
        titles.push(title);
      }
      fullSlideTitles.set(slide, titles);
    });

  if (fullSlideTitles.size === 0) return value;

  return value.replace(
    /\bslide\s+(\d+)\s*:\s*([A-Za-z][A-Za-z0-9 -]{1,18})\.\.\./gi,
    (match, slide: string, clipped: string) => {
      const normalizedPrefix = clipped.replace(/\s+/g, " ").trim().toLowerCase();
      const fullTitle = (fullSlideTitles.get(slide) || [])
        .find((title) => title.toLowerCase().startsWith(normalizedPrefix));
      return fullTitle ? `slide ${slide}: ${fullTitle}` : match;
    },
  );
};

const buildRequestedEvidenceAnswer = (
  context: GraphRAGDocumentFallbackContext,
): string | null => {
  const candidates = collectRequestedEvidenceCandidates(context);
  if (candidates.length === 0) return null;
  const requestedFamilyCount = requestedEvidenceFamilyOrder(context.query).size;
  const answerLimit = Math.min(12, Math.max(6, requestedFamilyCount || 0));

  return repairTruncatedEvidenceLocations([
    "Extracted evidence:",
    ...candidates.slice(0, answerLimit).map(requestedEvidenceBullet),
  ].join("\n\n"));
};

const derivePathNodes = (relationships: FallbackCandidate[]): string[] => {
  const nodes: string[] = [];
  relationships.forEach((relationship) => {
    if (nodes.length === 0) nodes.push(relationship.source);
    if (nodes[nodes.length - 1] !== relationship.source) {
      nodes.push(relationship.source);
    }
    nodes.push(relationship.target);
  });
  return nodes.filter(Boolean);
};

const pathCandidate = (
  path: GraphRAGDocumentFallbackPath,
  context: GraphRAGDocumentFallbackContext,
): PathCandidate | null => {
  const relationships = (path.relationships || [])
    .map((relationship) =>
      relationshipCandidate(relationship, context.query, context.sourceLabel),
    )
    .filter((candidate): candidate is FallbackCandidate => Boolean(candidate));
  if (relationships.length === 0) return null;

  const nodes = (path.nodes || []).map(asString).filter(Boolean);
  const displayNodes = nodes.length > 0 ? nodes : derivePathNodes(relationships);
  const text =
    asString(path.display_text) ||
    asString(path.traversal_text) ||
    asString(path.text) ||
    formatRelationshipPath(relationships);
  const haystack = [
    text,
    asString(path.explanation),
    asString(path.answer_hint),
    displayNodes.join(" "),
    relationships.map((relationship) => relationship.statement).join(" "),
  ]
    .join(" ")
    .toLowerCase();
  const termScore = queryTerms(context.query).reduce(
    (total, term) => total + (haystack.includes(term) ? 10 : 0),
    0,
  );
  const numericScore = typeof path.score === "number" ? path.score * 10 : 0;

  return {
    nodes: displayNodes,
    relationships,
    text,
    explanation: cleanGraphRAGTraceSampleText(asString(path.explanation), 220),
    score: numericScore + termScore + relationships.length * 12,
  };
};

const collectPathCandidates = (
  context: GraphRAGDocumentFallbackContext,
): PathCandidate[] =>
  (context.graphReasoning?.paths || [])
    .map((path) => pathCandidate(path, context))
    .filter((candidate): candidate is PathCandidate => Boolean(candidate))
    .sort(
      (a, b) =>
        b.score - a.score ||
        b.relationships.length - a.relationships.length,
    );

function formatRelationshipPath(relationships: FallbackCandidate[]): string {
  if (relationships.length === 0) return "";
  const parts = [relationships[0].source];
  relationships.forEach((relationship) => {
    parts.push(`--[${relationship.relation}]-->`, relationship.target);
  });
  return parts.join(" ");
}

const normalizePathNodeKey = (value: string): string =>
  stripTrailingPunctuation(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();

const readablePathNodes = (
  nodes: string[],
  relationships: FallbackCandidate[],
): string[] => {
  const cleanNodes = nodes.map(stripTrailingPunctuation).filter(Boolean);
  if (cleanNodes.length > 1) return cleanNodes;
  if (relationships.length === 0) return [];
  return derivePathNodes(relationships).map(stripTrailingPunctuation).filter(Boolean);
};

const bridgeNodesForPath = (nodes: string[]): string[] => {
  const startKey = normalizePathNodeKey(nodes[0] || "");
  const endKey = normalizePathNodeKey(nodes[nodes.length - 1] || "");
  const seen = new Set<string>();
  return nodes
    .slice(1, -1)
    .map(stripTrailingPunctuation)
    .filter(Boolean)
    .filter((node) => {
      const key = normalizePathNodeKey(node);
      if (!key || key === startKey || key === endKey || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
};

const isDegeneratePathMeaning = (value: string): boolean => {
  const match = value.match(/^(.+?)\s+connects?\s+to\s+(.+?)\s+through\s+(.+?)\.?$/i);
  if (!match) return false;
  const keys = [
    normalizePathNodeKey(match[1] || ""),
    normalizePathNodeKey(match[2] || ""),
    ...String(match[3] || "")
      .split(/\s+to\s+/i)
      .map(normalizePathNodeKey),
  ].filter(Boolean);
  return keys.length > 2 && new Set(keys).size === 1;
};

const pathMeaning = (
  path: PathCandidate,
  nodes: string[],
  bridgeNodes: string[],
  plainLanguage: string,
): string => {
  const start = nodes[0] || "";
  const end = nodes[nodes.length - 1] || "";
  const startKey = normalizePathNodeKey(start);
  const endKey = normalizePathNodeKey(end);
  if (start && end && startKey && endKey && startKey !== endKey) {
    if (bridgeNodes.length > 0) {
      return `${start} connects to ${end} through ${bridgeNodes.join(" to ")}.`;
    }
    return `${start} connects to ${end} through a direct graph relationship.`;
  }

  const explanation = stripTrailingPunctuation(path.explanation || "");
  if (explanation && !isDegeneratePathMeaning(explanation)) {
    return `${explanation}.`;
  }

  const cleanPlainLanguage = stripTrailingPunctuation(plainLanguage);
  return cleanPlainLanguage ? `${cleanPlainLanguage}.` : "";
};

const textMentionsEntity = (text: string, entity: string): boolean => {
  const haystack = normalizePathNodeKey(text);
  const needle = normalizePathNodeKey(entity);
  if (!haystack || !needle) return false;
  if (haystack.includes(needle)) return true;
  const terms = needle
    .split(" ")
    .filter((term) => term.length > 2 && !STOP_WORDS.has(term));
  return terms.length > 1 && terms.every((term) => haystack.includes(term));
};

const relationshipSearchText = (relationship: FallbackCandidate): string =>
  [
    relationship.source,
    relationship.relation,
    relationship.target,
    relationship.statement,
    relationship.evidence,
    relationship.citation,
  ].join(" ");

const requestedPathEntities = (
  context: GraphRAGDocumentFallbackContext,
  paths: PathCandidate[],
  relationships: FallbackCandidate[],
): string[] => {
  const entities = uniqueStrings([
    ...paths.flatMap((path) => path.nodes),
    ...relationships.flatMap((relationship) => [
      relationship.source,
      relationship.target,
    ]),
  ])
    .map(stripTrailingPunctuation)
    .filter(isUsefulSummaryEntity)
    .filter((entity) => textMentionsEntity(context.query, entity));
  return entities
    .filter((entity) => {
      const entityKey = normalizePathNodeKey(entity);
      return !entities.some((other) => {
        const otherKey = normalizePathNodeKey(other);
        return (
          otherKey !== entityKey &&
          otherKey.includes(entityKey) &&
          otherKey.split(" ").length > entityKey.split(" ").length
        );
      });
    })
    .sort((a, b) => {
      const leftWords = normalizePathNodeKey(a).split(" ").filter(Boolean).length;
      const rightWords = normalizePathNodeKey(b).split(" ").filter(Boolean).length;
      return rightWords - leftWords || b.length - a.length || a.localeCompare(b);
    });
};

const sameRelationshipKey = (
  left: FallbackCandidate,
  right: FallbackCandidate,
): boolean =>
  normalizeEntityKey(left.source) === normalizeEntityKey(right.source) &&
  normalizeRelationKey(left.relationKey) === normalizeRelationKey(right.relationKey) &&
  normalizeEntityKey(left.target) === normalizeEntityKey(right.target);

const PROMPT_META_EVIDENCE_PATTERN =
  /^(?:question target|graph intent hints?|generated query|user query|the user explicitly asked|requested evidence|answer instruction|use this path)\b/i;

const supplementalEvidenceScore = (
  relationship: FallbackCandidate,
  entity: string,
): number => {
  const entityKey = normalizePathNodeKey(entity);
  const sourceKey = normalizePathNodeKey(relationship.source);
  const targetKey = normalizePathNodeKey(relationship.target);
  const evidenceKey = normalizePathNodeKey(relationship.evidence);
  const exactEndpointScore =
    Number(sourceKey === entityKey) * 240 + Number(targetKey === entityKey) * 180;
  const evidenceMentionsEntity = textMentionsEntity(relationship.evidence, entity);
  const evidenceStartsWithEntity = evidenceKey.startsWith(entityKey);
  const relationPenalty =
    /^(?:co_occurs(?:_with)?|mentioned?_with|mentions?|related(?:_to|_text)?|contains(?:_text)?)$/i.test(
      relationship.relationKey,
    )
      ? -50
      : 70;
  const promptMetaPenalty = PROMPT_META_EVIDENCE_PATTERN.test(relationship.evidence)
    ? -420
    : 0;
  return (
    exactEndpointScore +
    Number(evidenceMentionsEntity) * 70 +
    Number(evidenceStartsWithEntity) * 90 +
    relationPenalty +
    promptMetaPenalty +
  relationship.score
  );
};

const splitTaggedChunkEvidence = (text: string): string[] => {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return [];
  const tagged = normalized
    .split(/(?=\[[^\]]+\]\s+(?:page|slide|table|chart)\s+\d+\b)/i)
    .map((part) => part.trim())
    .filter(Boolean);
  if (tagged.length > 1) return tagged;
  return normalized
    .split(/(?<=[.!?])\s+/)
    .map((part) => part.trim())
    .filter(Boolean);
};

const cleanChunkEvidenceSentence = (text: string): string =>
  stripTrailingPunctuation(
    cleanStructuredRowMetadataTail(cleanGraphRAGTraceSampleText(text, 260))
      .replace(/^\[[^\]]+\]\s*/g, "")
      .replace(
        /^(?:page|slide|line|paragraph|row|tr|td|cell|p|h[1-6])\s+\d+(?:\s+(?:block|connector|chart|table|row|shape|control|cell)\s+\d+(?:\.\d+)?)?\s*:\s*/i,
        "",
      ),
  );

const bestChunkEvidenceForEntity = (
  context: GraphRAGDocumentFallbackContext,
  entity: string,
): { body: string; citation: string; score: number } | null => {
  const entityKey = normalizePathNodeKey(entity);
  return (context.chunks || [])
    .flatMap((chunk) => {
      const metadata = asRecord(chunk.metadata);
      const text =
        asString(metadata.semantic_text) ||
        asString(metadata.evidence_preview) ||
        asString(chunk.content);
      return splitTaggedChunkEvidence(text).map((sentence) => {
        const body = cleanChunkEvidenceSentence(sentence);
        if (!body || !textMentionsEntity(body, entity)) return null;
        const bodyKey = normalizePathNodeKey(body);
        const promptMetaPenalty = PROMPT_META_EVIDENCE_PATTERN.test(body)
          ? -420
          : 0;
        const score =
          Number(bodyKey.startsWith(entityKey)) * 220 +
          Number(bodyKey.includes(entityKey)) * 100 +
          promptMetaPenalty +
          Math.min(60, body.length / 3);
        return {
          body,
          citation: chunkCitation(chunk),
          score,
        };
      });
    })
    .filter(
      (candidate): candidate is { body: string; citation: string; score: number } =>
        Boolean(candidate),
    )
    .sort((a, b) => b.score - a.score)[0] || null;
};

const supplementalRequestedEvidence = (
  context: GraphRAGDocumentFallbackContext,
  selectedPath: PathCandidate,
  paths: PathCandidate[],
  relationships: FallbackCandidate[],
): SupplementalRequestedEvidence[] => {
  const selectedNodeKeys = new Set(
    readablePathNodes(selectedPath.nodes, selectedPath.relationships).map(
      normalizePathNodeKey,
    ),
  );
  const selectedRelationshipKeys = new Set(
    selectedPath.relationships.map((relationship) =>
      [
        normalizeEntityKey(relationship.source),
        normalizeRelationKey(relationship.relationKey),
        normalizeEntityKey(relationship.target),
      ].join("|"),
    ),
  );
  const missingEntities = requestedPathEntities(context, paths, relationships)
    .filter((entity) => !selectedNodeKeys.has(normalizePathNodeKey(entity)))
    .slice(0, 4);
  const chosen: SupplementalRequestedEvidence[] = [];

  missingEntities.forEach((entity) => {
    const relationshipMatch = relationships
      .filter((relationship) => textMentionsEntity(relationshipSearchText(relationship), entity))
      .filter((relationship) => {
        const key = [
          normalizeEntityKey(relationship.source),
          normalizeRelationKey(relationship.relationKey),
          normalizeEntityKey(relationship.target),
        ].join("|");
        return !selectedRelationshipKeys.has(key);
      })
      .filter((relationship) =>
        !chosen.some((existing) =>
          existing.relationship &&
          sameRelationshipKey(existing.relationship, relationship),
        ),
      )
      .sort((a, b) => {
        return (
          supplementalEvidenceScore(b, entity) -
            supplementalEvidenceScore(a, entity) ||
          b.score - a.score
        );
      })[0];
    const chunkMatch = bestChunkEvidenceForEntity(context, entity);
    const relationshipScore = relationshipMatch
      ? supplementalEvidenceScore(relationshipMatch, entity)
      : Number.NEGATIVE_INFINITY;
    if (chunkMatch && chunkMatch.score > relationshipScore) {
      chosen.push({
        entity,
        body: chunkMatch.body,
        citation: chunkMatch.citation,
      });
      return;
    }
    if (relationshipMatch) {
      const evidence = stripTrailingPunctuation(relationshipMatch.evidence);
      const fallback = stripTrailingPunctuation(relationshipMatch.statement);
      chosen.push({
        entity,
        relationship: relationshipMatch,
        body:
          evidence && !/(?:--|->|\bevidence=|\bprovenance=)/i.test(evidence)
            ? evidence
            : fallback,
        citation: relationshipMatch.citation,
      });
    }
  });

  return chosen.slice(0, 2);
};

const supplementalRequestedEvidenceBullet = (
  supplemental: SupplementalRequestedEvidence,
): string => {
  const { entity } = supplemental;
  const body = stripTrailingPunctuation(supplemental.body);
  const citation = supplemental.citation ? ` Source: ${supplemental.citation}.` : "";
  return `- **${stripTrailingPunctuation(entity)}**: ${body}.${citation}`;
};

const WEAK_EVIDENCE_RELATION_PATTERN =
  /^(?:appears?_with|co_occurs(?:_with)?|mentioned?_with|mentions?|related(?:_to|_text)?|contains(?:_text)?)$/i;

const answerReadyEvidenceText = (relationship: FallbackCandidate): string => {
  const evidence = stripTrailingPunctuation(relationship.evidence);
  if (!evidence) return "";
  if (PROMPT_META_EVIDENCE_PATTERN.test(evidence)) return "";
  if (/(?:--|->|\bevidence=|\bprovenance=)/i.test(evidence)) return "";
  return evidence;
};

const relationshipSummary = (relationship: FallbackCandidate): string =>
  WEAK_EVIDENCE_RELATION_PATTERN.test(relationship.relationKey)
    ? answerReadyEvidenceText(relationship) || stripTrailingPunctuation(relationship.statement)
    : stripTrailingPunctuation(relationship.statement);

const evidenceBullet = (relationship: FallbackCandidate): string => {
  const evidence = answerReadyEvidenceText(relationship) || stripTrailingPunctuation(relationship.statement);
  const citation = relationship.citation ? ` Source: ${relationship.citation}.` : "";
  if (WEAK_EVIDENCE_RELATION_PATTERN.test(relationship.relationKey) && evidence) {
    return `- ${evidence}.${citation}`;
  }
  return `- **${relationship.source} ${relationship.relation} ${relationship.target}**: ${evidence}.${citation}`;
};

const pathEvidenceBullet = (relationship: FallbackCandidate): string => {
  const statement = stripTrailingPunctuation(relationship.statement);
  const rawEvidence = answerReadyEvidenceText(relationship);
  if (WEAK_EVIDENCE_RELATION_PATTERN.test(relationship.relationKey) && rawEvidence) {
    const citation = relationship.citation ? ` Source: ${relationship.citation}.` : "";
    return `- ${rawEvidence}.${citation}`;
  }
  const evidence =
    rawEvidence
      ? ` ${rawEvidence}.`
      : "";
  const citation = relationship.citation ? ` Source: ${relationship.citation}.` : "";
  return `- **${statement}.**${evidence}${citation}`;
};

const SUMMARY_LOW_VALUE_RELATION_PATTERN =
  /^(?:created_with|links?_to|part_of|source_artifact|appears?_with|co_occurs(?:_with)?|mentioned?_with|mentions?|related_to)$/i;

const SUMMARY_STRUCTURAL_NODE_PATTERN =
  /\b(?:source brief|global footer|layout governance|source document|slide\s+\d+|https?:\/\/|\.pptx?|\.pdf|\.docx?|\.xlsx?|\.csv|\.md|\.html?)\b/i;

const SUMMARY_PROMPT_META_NODE_PATTERN =
  /\b(?:best path|document evidence|extracted evidence|graph paths?|key evidence|knowledge graph paths?|retrieved context|semantic relationships?|source proof|top relationships?)\b/i;

const isLowValueSummaryRelationship = (
  relationship: Pick<FallbackCandidate, "relationKey" | "source" | "target">,
): boolean => {
  const relationKey = normalizeRelationKey(relationship.relationKey);
  const sourceIsStructural = SUMMARY_STRUCTURAL_NODE_PATTERN.test(relationship.source);
  const targetIsStructural = SUMMARY_STRUCTURAL_NODE_PATTERN.test(relationship.target);
  const sourceIsPromptMeta = SUMMARY_PROMPT_META_NODE_PATTERN.test(relationship.source);
  const targetIsPromptMeta = SUMMARY_PROMPT_META_NODE_PATTERN.test(relationship.target);
  return (
    relationKey === "contains" ||
    relationKey.startsWith("contains_") ||
    relationKey === "has" ||
    relationKey.startsWith("has_") ||
    SUMMARY_LOW_VALUE_RELATION_PATTERN.test(relationKey) ||
    (sourceIsStructural && targetIsStructural) ||
    (sourceIsPromptMeta || targetIsPromptMeta) ||
    ((sourceIsStructural || targetIsStructural) &&
      /^(?:causes?|uses?|works?_at)$/i.test(relationKey))
  );
};

const collectSummaryEntities = (
  relationships: FallbackCandidate[],
  paths: PathCandidate[],
): EntitySummaryCandidate[] => {
  const scores = new Map<string, EntitySummaryCandidate>();
  const add = (name: string, score: number) => {
    const cleanName = stripTrailingPunctuation(name);
    if (!isUsefulSummaryEntity(cleanName)) return;
    const key = cleanName.toLowerCase();
    const current = scores.get(key);
    if (current) {
      current.score += score;
    } else {
      scores.set(key, { name: cleanName, score });
    }
  };

  relationships.forEach((relationship, index) => {
    const relationWeight = isLowValueSummaryRelationship(relationship)
      ? 1
      : 4;
    add(relationship.source, relationWeight + Math.max(0, 3 - index));
    add(relationship.target, relationWeight + Math.max(0, 3 - index));
  });

  paths.slice(0, 2).forEach((path) => {
    path.nodes.forEach((node, index) => {
      add(node, index === 0 || index === path.nodes.length - 1 ? 4 : 5);
    });
  });

  return Array.from(scores.values())
    .sort((a, b) => b.score - a.score || a.name.localeCompare(b.name))
    .slice(0, 6);
};

const summaryRelationshipText = (relationship: FallbackCandidate): string =>
  stripTrailingPunctuation(relationship.statement);

const summaryPathScore = (path: PathCandidate): number => {
  const semanticRelationships = path.relationships.filter(
    (relationship) => !isLowValueSummaryRelationship(relationship),
  ).length;
  const structuralRelationships = path.relationships.length - semanticRelationships;
  const structuralNodes = path.nodes.filter((node) =>
    SUMMARY_STRUCTURAL_NODE_PATTERN.test(node),
  ).length;
  if (semanticRelationships === 0) {
    return -structuralRelationships * 14 - structuralNodes * 8;
  }
  const numericScore = typeof path.score === "number" ? path.score : 0;
  return (
    semanticRelationships * 42 +
    numericScore -
    structuralRelationships * 14 -
    structuralNodes * 8
  );
};

const bestSemanticSummaryPath = (paths: PathCandidate[]): PathCandidate | null =>
  paths
    .map((path) => ({ path, score: summaryPathScore(path) }))
    .filter((candidate) => candidate.score > 0)
    .sort(
      (a, b) =>
        b.score - a.score ||
        b.path.relationships.length - a.path.relationships.length,
    )[0]?.path || null;

const pathRequestedEntitySearchText = (path: PathCandidate): string => {
  const nodes = readablePathNodes(path.nodes, path.relationships);
  return [
    nodes.join(" "),
    path.relationships.map(relationshipSearchText).join(" "),
    path.text,
    path.explanation,
  ].join(" ");
};

const pathRequestedEntityCoverage = (
  path: PathCandidate,
  requestedEntities: string[],
): number =>
  requestedEntities.filter((entity) =>
    textMentionsEntity(pathRequestedEntitySearchText(path), entity),
  ).length;

const pathRequestedEntityCoverageScore = (
  path: PathCandidate,
  requestedEntities: string[],
): number => {
  if (requestedEntities.length < 2) return 0;
  const nodes = readablePathNodes(path.nodes, path.relationships);
  const covered = requestedEntities.filter((entity) =>
    textMentionsEntity(pathRequestedEntitySearchText(path), entity),
  );
  const endpointBonus =
    Number(covered.some((entity) => nodes[0] && textMentionsEntity(nodes[0], entity))) * 20 +
    Number(covered.some((entity) => nodes[nodes.length - 1] && textMentionsEntity(nodes[nodes.length - 1], entity))) * 20;
  return (
    covered.length * 120 +
    endpointBonus +
    Math.min(40, path.relationships.length * 14) +
    Math.min(25, path.score / 10)
  );
};

const bestRequestedEntityPath = (
  context: GraphRAGDocumentFallbackContext,
  paths: PathCandidate[],
  relationships: FallbackCandidate[],
  requestedEntities = requestedPathEntities(context, paths, relationships),
): PathCandidate | null => {
  if (requestedEntities.length < 2) return null;
  return paths
    .map((path) => ({
      path,
      score: pathRequestedEntityCoverageScore(path, requestedEntities),
      coverage: pathRequestedEntityCoverage(path, requestedEntities),
    }))
    .filter((candidate) => candidate.coverage >= 2)
    .sort(
      (a, b) =>
        b.coverage - a.coverage ||
        b.score - a.score ||
        b.path.relationships.length - a.path.relationships.length,
    )[0]?.path || null;
};

const normalizeEntityKey = (value: string): string =>
  value.replace(/\s+/g, " ").trim().toLowerCase();

const filenameKey = (value: string): string =>
  value
    .split(/[\\/]/)
    .pop()
    ?.trim()
    .toLowerCase() || "";

const relationshipSourceMatches = (
  relationship: GraphRAGDocumentFallbackRelationship,
  sourceLabel?: string,
): boolean => {
  const expected = filenameKey(sourceLabel || "");
  const actual = filenameKey(explicitRelationshipSourceLabel(relationship));
  return Boolean(expected && actual && expected === actual);
};

const answerPlanPathNodes = (
  path?: GraphRAGDocumentFallbackPath,
): string[] => {
  const nodes = (path?.display_nodes || path?.nodes || [])
    .map(asString)
    .filter(Boolean);
  if (nodes.length > 1) return nodes;
  const source = asString(path?.source);
  const target = asString(path?.target);
  return [source, target].filter(Boolean);
};

const answerPlanRelationshipPool = (
  context: GraphRAGDocumentFallbackContext,
): GraphRAGDocumentFallbackRelationship[] => {
  const answerPlan = context.graphReasoning?.answer_plan;
  return [
    ...(answerPlan?.supporting_relationships || []),
    ...(answerPlan?.evidence_cards || []),
    ...(answerPlan?.hops || []),
    ...(answerPlan?.primary_path?.relationships || []),
    ...(context.graphReasoning?.direct_relationships || []),
    ...(context.relationships || []),
    ...(context.graphReasoning?.paths || []).flatMap((path) => path.relationships || []),
  ];
};

const answerPlanRelationshipScore = (
  relationship: GraphRAGDocumentFallbackRelationship,
  candidate: FallbackCandidate,
  context: GraphRAGDocumentFallbackContext,
): number => {
  const role = asString((relationship as GraphRAGDocumentEvidenceCard).role);
  const sourceMatchScore = relationshipSourceMatches(relationship, context.sourceLabel)
    ? 900
    : 0;
  const roleScore =
    role === "supporting_relationship"
      ? 220
      : role === "primary_path_hop"
        ? 160
        : 0;
  const semanticScore = isLowValueSummaryRelationship(candidate) ? -500 : 120;
  return sourceMatchScore + roleScore + semanticScore + candidate.score;
};

const bestAnswerPlanRelationshipForPair = (
  context: GraphRAGDocumentFallbackContext,
  source: string,
  target: string,
): FallbackCandidate | null => {
  const sourceKey = normalizeEntityKey(source);
  const targetKey = normalizeEntityKey(target);
  return answerPlanRelationshipPool(context)
    .map((relationship) => {
      const candidate = relationshipCandidate(
        relationship,
        context.query,
        context.sourceLabel,
      );
      if (!candidate) return null;
      if (
        normalizeEntityKey(candidate.source) !== sourceKey ||
        normalizeEntityKey(candidate.target) !== targetKey
      ) {
        return null;
      }
      return {
        candidate,
        score: answerPlanRelationshipScore(relationship, candidate, context),
      };
    })
    .filter(
      (item): item is { candidate: FallbackCandidate; score: number } =>
        Boolean(item),
    )
    .sort((a, b) => b.score - a.score)[0]?.candidate || null;
};

const answerPlanPathCandidate = (
  context: GraphRAGDocumentFallbackContext,
): PathCandidate | null => {
  const answerPlan = context.graphReasoning?.answer_plan;
  const primaryPath = answerPlan?.primary_path;
  if (!answerPlan || answerPlan.strategy !== "walk_strongest_path") return null;

  const nodes = answerPlanPathNodes(primaryPath);
  const orderedRelationships =
    nodes.length > 1
      ? nodes
          .slice(0, -1)
          .map((node, index) =>
            bestAnswerPlanRelationshipForPair(context, node, nodes[index + 1]),
          )
      : [];
  const completeOrderedRelationships = orderedRelationships.every(Boolean)
    ? (orderedRelationships as FallbackCandidate[])
    : [];
  const fallbackRelationships = (answerPlan.hops || primaryPath?.relationships || [])
    .map((relationship) =>
      relationshipCandidate(relationship, context.query, context.sourceLabel),
    )
    .filter((candidate): candidate is FallbackCandidate => Boolean(candidate));
  const relationships =
    completeOrderedRelationships.length > 0
      ? completeOrderedRelationships
      : fallbackRelationships;
  if (relationships.length === 0) return null;

  const displayNodes = nodes.length > 1 ? nodes : derivePathNodes(relationships);
  return {
    nodes: displayNodes,
    relationships,
    text:
      asString(primaryPath?.display_text) ||
      asString(primaryPath?.traversal_text) ||
      asString(primaryPath?.text) ||
      formatRelationshipPath(relationships),
    explanation: cleanGraphRAGTraceSampleText(
      asString(primaryPath?.explanation) || asString(primaryPath?.answer_hint),
      220,
    ),
    score: 10000 + relationships.length * 100,
  };
};

const splitCitationSource = (
  citation: string,
  fallbackSourceLabel?: string,
): { source: string; detail: string } => {
  const text = citation.replace(/\s+/g, " ").trim();
  if (!text) return { source: "", detail: "" };
  const sourceLabel = fallbackSourceLabel?.trim();
  if (sourceLabel && text.toLowerCase().startsWith(`${sourceLabel.toLowerCase()},`)) {
    return {
      source: sourceLabel,
      detail: text.slice(sourceLabel.length + 1).trim(),
    };
  }

  const fileMatch = text.match(/^([^,]+\.(?:pdf|pptx?|docx?|xlsx?|csv|md|html?|txt|json|xml))\s*,\s*(.+)$/i);
  if (fileMatch) {
    return {
      source: fileMatch[1].trim(),
      detail: fileMatch[2].trim(),
    };
  }

  return { source: "", detail: text };
};

const compactSummarySourceProof = (
  citations: string[],
  fallbackSourceLabel?: string,
): string => {
  const parts = uniqueStrings(citations).map((citation) =>
    splitCitationSource(citation, fallbackSourceLabel),
  );
  const sourceGroups = new Map<string, string[]>();
  const standalone: string[] = [];

  parts.forEach(({ source, detail }) => {
    if (!source) {
      if (detail) standalone.push(detail);
      return;
    }
    const details = sourceGroups.get(source) || [];
    if (detail) details.push(detail);
    sourceGroups.set(source, details);
  });

  const grouped = Array.from(sourceGroups.entries()).map(([source, details]) => {
    const cleanDetails = uniqueStrings(details);
    if (cleanDetails.length === 0) return source;
    const [primaryDetail, ...extraDetails] = cleanDetails;
    const extras = extraDetails
      .map((detail) =>
        detail.toLowerCase().startsWith(primaryDetail.toLowerCase())
          ? detail.slice(primaryDetail.length).replace(/^[\s,;:]+/, "").trim()
          : detail,
      )
      .filter(Boolean);
    return extras.length > 0
      ? `${source}, ${primaryDetail} (${extras.join("; ")})`
      : `${source}, ${primaryDetail}`;
  });

  return uniqueStrings([...grouped, ...standalone]).slice(0, 3).join("; ");
};

const asksForPerSourceEvidenceAnswer = (query: string): boolean => {
  const text = query.replace(/\s+/g, " ").trim().toLowerCase();
  const asksForSourceList =
    /\b(?:list\s+each|each|every|all|per[-\s]?(?:file|document|source)|filenames?|file names?)\b/.test(
      text,
    );
  const sourceScope =
    /\b(?:uploaded\s+)?(?:files?|documents?|sources?|filenames?)\b/.test(text);
  return asksForSourceList && sourceScope;
};

const sourceFromCitation = (
  citation: string,
  fallbackSourceLabel?: string,
): string => {
  const { source } = splitCitationSource(citation, fallbackSourceLabel);
  const cleanSource = displaySourceLabel(source);
  if (cleanSource && !isBatchSourceLabel(cleanSource)) return cleanSource;
  const embeddedFile = citation.match(
    /\b([^\\/;,\s]+\.(?:pdf|pptx?|docx?|xlsx?|csv|md|html?|txt|json|xml))\b/i,
  )?.[1];
  const cleanEmbedded = displaySourceLabel(embeddedFile);
  return cleanEmbedded && !isBatchSourceLabel(cleanEmbedded) ? cleanEmbedded : "";
};

const perSourceTermScore = (text: string, terms: string[]): number => {
  const haystack = text.toLowerCase();
  return terms.reduce(
    (total, term) => total + (haystack.includes(term) ? 8 : 0),
    0,
  );
};

const cleanStructuredRowMetadataTail = (value: string): string =>
  normalizeRequestedEvidenceText(value)
    .replace(/\s*;\s*(?:table|sheet|slide|chart|document|media)?\s*source\s+semantic_[a-z_]+\b[^;)]+/gi, "")
    .replace(/\s+\bvisibility\s+(?:visible|hidden)\b/gi, "")
    .replace(/\s+\bsemantic_[a-z_]+\b/gi, "")
    .replace(
      /\s+\b(?:sheet|row|line|paragraph|tr|td|cell)\s+\d+(?:\s+\w+\s+\d+)*\s*:\s*(?:Source|Relationship|Target|Evidence)\s*:.*$/i,
      "",
    )
    .replace(/\s+\b(?:Source|Relationship|Target|Evidence)\s*:\s*.*$/i, "")
    .trim();

const structuredRelationshipText = (
  source: string,
  relation: string,
  target: string,
  evidence: string,
): string => {
  const statement = sentenceCase(
    relationStatement(
      source,
      normalizeRelationKey(relation),
      formatGraphReferenceRelation(relation),
      target,
    ),
  );
  const cleanEvidence = stripTrailingPunctuation(
    cleanStructuredRowMetadataTail(evidence.replace(/\[[^\]]+\]/g, " ")),
  );
  return uniqueStrings([statement, cleanEvidence]).join(". ");
};

const compactStructuredChunkEvidence = (body: string): string => {
  const normalized = body.replace(/\s+/g, " ").trim();
  const assignmentMatch = normalized.match(
    /\bSource:\s*([^;|]+)[;|]\s*Relationship:\s*([^;|]+)[;|]\s*Target:\s*([^;|]+)[;|]\s*Evidence:\s*([^;]+)/i,
  );
  if (assignmentMatch) {
    return structuredRelationshipText(
      assignmentMatch[1] || "",
      assignmentMatch[2] || "",
      assignmentMatch[3] || "",
      assignmentMatch[4] || "",
    );
  }

  const markerValues = Array.from(
    normalized.matchAll(
      /\b(?:paragraph|row|tr|td|cell)\s+\d+:\s*([\s\S]*?)(?=\s+\b(?:paragraph|row|tr|td|cell)\s+\d+:|$)/gi,
    ),
    (match) =>
      stripTrailingPunctuation(
        String(match[1] || "")
          .replace(/\[[^\]]+\]/g, " ")
          .replace(/\s+/g, " "),
      ),
  ).filter(Boolean);
  const normalizedValues = markerValues.map((value) =>
    normalizeRelationKey(value),
  );
  const sourceIndex = normalizedValues.findIndex(
    (value, index) =>
      value === "source" &&
      normalizedValues[index + 1] === "relationship" &&
      normalizedValues[index + 2] === "target" &&
      normalizedValues[index + 3] === "evidence",
  );
  if (sourceIndex >= 0) {
    const source = markerValues[sourceIndex + 4] || "";
    const relation = markerValues[sourceIndex + 5] || "";
    const target = markerValues[sourceIndex + 6] || "";
    const evidence = markerValues[sourceIndex + 7] || "";
    if (source && relation && target) {
      return structuredRelationshipText(source, relation, target, evidence);
    }
  }

  return "";
};

const perSourceRelationshipCandidate = (
  relationship: FallbackCandidate,
  terms: string[],
): PerSourceEvidenceCandidate | null => {
  const source = sourceFromCitation(relationship.citation);
  if (!source) return null;
  const statement = stripTrailingPunctuation(relationship.statement);
  const directRelationship = DIRECT_RELATION_PATTERN.test(relationship.relationKey);
  const body =
    directRelationship
      ? statement
      : answerReadyEvidenceText(relationship) || statement;
  if (!body || PROMPT_META_EVIDENCE_PATTERN.test(body)) return null;
  const bodyWords = normalizePathNodeKey(body).split(/\s+/).filter(Boolean).length;
  const lowValueRelationship = isLowValueSummaryRelationship(relationship);
  const terseEvidencePenalty = bodyWords < 4 ? -180 : 0;
  const sourceTargetEchoPenalty =
    normalizePathNodeKey(body) === normalizePathNodeKey(relationship.source) ||
    normalizePathNodeKey(body) === normalizePathNodeKey(relationship.target)
      ? -120
      : 0;
  const relationScore = directRelationship
    ? 190
    : lowValueRelationship
      ? -130
    : WEAK_EVIDENCE_RELATION_PATTERN.test(relationship.relationKey)
      ? -30
      : 60;
  return {
    source,
    body,
    citation: relationship.citation,
    score:
      relationship.score +
      relationScore +
      terseEvidencePenalty +
      sourceTargetEchoPenalty +
      Math.min(70, bodyWords * 8) +
      perSourceTermScore(
        `${body} ${relationship.source} ${relationship.target}`,
        terms,
      ),
  };
};

const perSourceChunkCandidate = (
  chunk: GraphRAGDocumentFallbackChunk,
  terms: string[],
): PerSourceEvidenceCandidate | null => {
  const source = explicitChunkSourceLabel(chunk);
  if (!source || isBatchSourceLabel(source)) return null;
  const metadata = asRecord(chunk.metadata);
  const rawText =
    asString(metadata.semantic_text) ||
    asString(metadata.evidence_preview) ||
    asString(chunk.content);
  const ranked = splitTaggedChunkEvidence(rawText)
    .map((sentence) => {
      const compact = compactStructuredChunkEvidence(sentence);
      const body = compact || cleanChunkEvidenceSentence(sentence);
      return {
        body: body.replace(/\s*\[[^\]]+\]\s*/g, " ").replace(/\s+/g, " ").trim(),
        structured: Boolean(compact),
      };
    })
    .filter((item) => Boolean(item.body))
    .filter((item) => !PROMPT_META_EVIDENCE_PATTERN.test(item.body))
    .filter((item) => !isLowValueGraphRAGTraceSampleText(item.body))
    .map((item) => ({
      body: item.body,
      score:
        Number(item.structured) * 180 +
        perSourceTermScore(item.body, terms) +
        Math.min(50, item.body.length / 4),
    }))
    .sort((a, b) => b.score - a.score);
  const chosen = ranked[0];
  if (!chosen) return null;
  return {
    source,
    body: chosen.body,
    citation: chunkCitation(chunk),
    score: chosen.score,
  };
};

const collectPerSourceEvidence = (
  context: GraphRAGDocumentFallbackContext,
  relationships: FallbackCandidate[],
): PerSourceEvidenceCandidate[] => {
  if (!asksForPerSourceEvidenceAnswer(context.query)) return [];

  const terms = queryTerms(context.query);
  const sourceOrder = uniqueStrings([
    ...(context.chunks || []).map(explicitChunkSourceLabel),
    ...relationships.map((relationship) =>
      sourceFromCitation(relationship.citation, context.sourceLabel),
    ),
  ]).filter((source) => source && !isBatchSourceLabel(source));
  const bySource = new Map<string, PerSourceEvidenceCandidate>();
  const remember = (candidate: PerSourceEvidenceCandidate | null) => {
    if (!candidate) return;
    const key = candidate.source.toLowerCase();
    const existing = bySource.get(key);
    if (!existing || candidate.score > existing.score) {
      bySource.set(key, candidate);
    }
  };

  relationships.forEach((relationship) =>
    remember(perSourceRelationshipCandidate(relationship, terms)),
  );
  (context.chunks || []).forEach((chunk) =>
    remember(perSourceChunkCandidate(chunk, terms)),
  );

  const ordered = sourceOrder
    .map((source) => bySource.get(source.toLowerCase()))
    .filter((candidate): candidate is PerSourceEvidenceCandidate =>
      Boolean(candidate),
    );
  const extras = Array.from(bySource.values()).filter(
    (candidate) =>
      !ordered.some(
        (existing) =>
          existing.source.toLowerCase() === candidate.source.toLowerCase(),
      ),
  );
  return [...ordered, ...extras.sort((a, b) => b.score - a.score)].slice(0, 10);
};

const perSourceEvidenceBullet = (
  candidate: PerSourceEvidenceCandidate,
): string => {
  const body = stripTrailingPunctuation(candidate.body);
  const citation = candidate.citation ? ` Source: ${candidate.citation}.` : "";
  return `- **${candidate.source}**: ${body}.${citation}`;
};

const buildGraphRAGDocumentSummaryAnswer = (
  context: GraphRAGDocumentFallbackContext,
): string | null => {
  if (!isGraphRAGSummaryQuery(context.query)) return null;

  const paths = collectPathCandidates(context);
  const relationships = collectRelationshipCandidates(context);
  const strongerRelationships = relationships.filter(
    (relationship) => !isLowValueSummaryRelationship(relationship),
  );
  const topRelationships = (strongerRelationships.length > 0
    ? strongerRelationships
    : relationships).slice(0, 4);
  const bestPath = bestSemanticSummaryPath(paths);
  const entities = collectSummaryEntities(
    topRelationships.length > 0 ? topRelationships : relationships.slice(0, 6),
    bestPath ? [bestPath] : paths,
  );
  const requestedEvidence = collectRequestedEvidenceCandidates(context);
  const citations = uniqueStrings([
    ...topRelationships.map((relationship) => relationship.citation),
    ...requestedEvidence.map((candidate) => candidate.citation),
  ])
    .filter(Boolean)
    .slice(0, 3);

  if (
    entities.length === 0 &&
    topRelationships.length === 0 &&
    !bestPath &&
    requestedEvidence.length === 0
  ) {
    return null;
  }

  const topRelationshipText =
    topRelationships.length > 0
      ? topRelationships.map(summaryRelationshipText).join("; ")
      : requestedEvidence
        .slice(0, 3)
        .map((candidate) => `${candidate.title}: ${stripTrailingPunctuation(candidate.evidence)}`)
        .join("; ");
  const pathText = bestPath
    ? bestPath.nodes.length > 1
      ? bestPath.nodes.map(stripTrailingPunctuation).join(" to ")
      : formatRelationshipPath(bestPath.relationships)
    : topRelationships[0]
      ? `No meaningful multi-hop path surfaced; strongest direct evidence is ${summaryRelationshipText(topRelationships[0])}`
      : "";
  const proofText =
    citations.length > 0
      ? compactSummarySourceProof(citations, context.sourceLabel)
      : context.sourceLabel || "selected document";

  return [
    `- **Entities**: ${entities.length > 0 ? entities.map((entity) => entity.name).join(", ") : "No named entities surfaced in the retrieved context"}.`,
    `- **Top relationships**: ${topRelationshipText || "No graph relationships surfaced in the retrieved context"}.`,
    `- **Best path**: ${pathText || "No multi-hop path surfaced for this summary"}.`,
    `- **Source proof**: ${proofText}.`,
  ].join("\n\n");
};

export const buildGraphRAGDocumentEvidenceAnswer = (
  context: GraphRAGDocumentFallbackContext,
): string | null => {
  const relationships = collectRelationshipCandidates(context);
  const perSourceEvidence = collectPerSourceEvidence(context, relationships);
  if (perSourceEvidence.length > 0) {
    return [
      "Per-file evidence:",
      ...perSourceEvidence.map(perSourceEvidenceBullet),
    ].join("\n\n");
  }

  const summaryAnswer = buildGraphRAGDocumentSummaryAnswer(context);
  if (summaryAnswer) return summaryAnswer;

  const requestedEvidenceAnswer = asksForRequestedEvidenceAnswer(context.query)
    ? buildRequestedEvidenceAnswer(context)
    : null;
  if (requestedEvidenceAnswer) return requestedEvidenceAnswer;

  const answerPlanPath = answerPlanPathCandidate(context);
  const paths = collectPathCandidates(context);
  const requestedEntities = requestedPathEntities(context, paths, relationships);
  const requestedEntityPath = bestRequestedEntityPath(
    context,
    paths,
    relationships,
    requestedEntities,
  );
  const answerPlanCoverage =
    answerPlanPath && requestedEntities.length > 1
      ? pathRequestedEntityCoverage(answerPlanPath, requestedEntities)
      : 0;
  const requestedPathCoverage =
    requestedEntityPath && requestedEntities.length > 1
      ? pathRequestedEntityCoverage(requestedEntityPath, requestedEntities)
      : 0;
  const requestedPathShouldLead =
    Boolean(requestedEntityPath) &&
    Boolean(answerPlanPath) &&
    requestedPathCoverage > answerPlanCoverage &&
    requestedEntityPath!.relationships.length > answerPlanPath!.relationships.length;
  const bestPath =
    (requestedPathShouldLead ? requestedEntityPath : null) ||
    answerPlanPath ||
    requestedEntityPath ||
    bestSemanticSummaryPath(paths) ||
    paths.find((path) => path.relationships.length > 1) ||
    paths[0];

  if (bestPath) {
    const readableNodes = readablePathNodes(bestPath.nodes, bestPath.relationships);
    const bridgeNodes = bridgeNodesForPath(readableNodes);
    const plainLanguage = bestPath.relationships
      .map((relationship) => relationship.statement)
      .join("; ");
    const readablePath = readableNodes.join(" to ");
    const relationshipText = bestPath.relationships
      .map(relationshipSummary)
      .filter(Boolean)
      .join("; ");
    const meaning = pathMeaning(
      bestPath,
      readableNodes,
      bridgeNodes,
      plainLanguage,
    );
    const supplementalEvidence = supplementalRequestedEvidence(
      context,
      bestPath,
      paths,
      relationships,
    );
    return [
      readablePath ? `Path: **${readablePath}**.` : "",
      relationshipText ? `Relationships: ${relationshipText}.` : "",
      bridgeNodes.length > 0
        ? `Bridge: ${bridgeNodes.join(" to ")}.`
        : "",
      meaning ? `Meaning: ${meaning}` : "",
      supplementalEvidence.length > 0 ? "Related requested context:" : "",
      ...supplementalEvidence.map(supplementalRequestedEvidenceBullet),
      perSourceEvidence.length > 0 ? "Per-file evidence:" : "",
      ...perSourceEvidence.map(perSourceEvidenceBullet),
      perSourceEvidence.length === 0 ? "Source evidence:" : "",
      ...(perSourceEvidence.length === 0
        ? bestPath.relationships.map(pathEvidenceBullet)
        : []),
    ]
      .filter(Boolean)
      .join("\n\n");
  }

  const chosen = relationships[0];
  if (!chosen) return null;

  const supporting = relationships
    .filter((candidate) => candidate !== chosen)
    .slice(0, 2);
  return [
    `Direct relationship: **${chosen.statement}.**`,
    chosen.evidence ? `Evidence: ${chosen.evidence}` : "",
    chosen.citation ? `Source: ${chosen.citation}.` : "",
    perSourceEvidence.length > 0 ? "Per-file evidence:" : "",
    ...perSourceEvidence.map(perSourceEvidenceBullet),
    supporting.length > 0 ? "Related graph evidence:" : "",
    ...(perSourceEvidence.length === 0 ? supporting.map(evidenceBullet) : []),
  ]
    .filter(Boolean)
    .join("\n\n");
};

const chunkCitation = (
  chunk: GraphRAGDocumentFallbackChunk,
  fallbackSourceLabel?: string,
): string => {
  const metadata = asRecord(chunk.metadata);
  const source = preferredChunkSourceLabel(chunk, fallbackSourceLabel);
  const location = normalizeLocation(metadata);
  const chunkLabel =
    chunk.chunk_index !== undefined ? `chunk ${chunk.chunk_index}` : "";
  return [source, location || chunkLabel].filter(Boolean).join(", ");
};

const chunkText = (chunk: GraphRAGDocumentFallbackChunk): string =>
  cleanGraphRAGTraceSampleText(String(chunk.content || ""), 360);

const extractValidationModel = (text: string): string => {
  const match =
    text.match(/\bvalidation\s+model\s*:\s*([^\s()]+)/i) ||
    text.match(/\bmodel\s*:\s*([^\s()]+)/i);
  return stripTrailingPunctuation(match?.[1] || "");
};

const bestChunk = (
  context: GraphRAGDocumentFallbackContext,
): { text: string; citation: string; model?: string } | null => {
  const terms = queryTerms(context.query);
  const ranked = (context.chunks || [])
    .map((chunk) => {
      const text = chunkText(chunk);
      if (!text || isLowValueGraphRAGTraceSampleText(text)) return null;
      const model = extractValidationModel(text);
      const haystack = text.toLowerCase();
      const score =
        (model ? 30 : 0) +
        terms.reduce((total, term) => total + (haystack.includes(term) ? 6 : 0), 0);
      return {
        text,
        citation: chunkCitation(chunk, context.sourceLabel),
        model,
        score,
      };
    })
    .filter((item): item is ChunkCandidate =>
      Boolean(item),
    )
    .sort((a, b) => b.score - a.score);
  return ranked[0] || null;
};

export const buildGraphRAGDocumentFallbackAnswer = (
  context: GraphRAGDocumentFallbackContext,
  reason: "timeout" | "error" = "timeout",
): string => {
  const relationships = collectRelationshipCandidates(context);
  const chunk = bestChunk(context);
  const top = relationships[0];
  const perSourceEvidence = collectPerSourceEvidence(context, relationships);
  const asksForModel = /\bmodel\b/i.test(context.query);
  const asksForGate = /\bgates?\b/i.test(context.query);
  const sourceNote =
    reason === "timeout"
      ? "The model endpoint did not answer in time, so I answered from the extracted GraphRAG evidence."
      : "The model endpoint failed, so I answered from the extracted GraphRAG evidence.";

  if (perSourceEvidence.length > 0) {
    return [
      "Per-file evidence:",
      ...perSourceEvidence.map(perSourceEvidenceBullet),
      sourceNote,
    ].join("\n\n");
  }

  if (asksForModel && chunk?.model) {
    const evidence = chunk.text.includes(chunk.model)
      ? chunk.text
      : [chunk.text, top?.evidence].filter(Boolean).join(" ");
    return [
      `The document reports **${chunk.model}** as the validation model.`,
      evidence ? `Evidence: ${evidence}` : "",
      `Source: ${chunk.citation}.`,
      sourceNote,
    ]
      .filter(Boolean)
      .join("\n\n");
  }

  const directGate =
    asksForGate && relationships.find((candidate) => candidate.relationKey === "gates");
  const chosen = directGate || top;
  if (chosen) {
    const lead =
      asksForGate && chosen.relationKey === "gates"
        ? `${stripTrailingPunctuation(chosen.source)} gates ${stripTrailingPunctuation(chosen.target)}.`
        : `${chosen.statement}.`;
    const supporting = relationships
      .filter((candidate) => candidate !== chosen)
      .slice(0, 2)
      .map((candidate) => `${candidate.statement}.`)
      .join(" ");
    return [
      lead,
      supporting ? `Supporting context: ${supporting}` : "",
      chosen.evidence ? `Evidence: ${chosen.evidence}` : "",
      `Source: ${chosen.citation}.`,
      sourceNote,
    ]
      .filter(Boolean)
      .join("\n\n");
  }

  if (chunk) {
    return [
      chunk.text,
      `Source: ${chunk.citation}.`,
      sourceNote,
    ].join("\n\n");
  }

  return [
    "I found the document context, but it did not include enough extracted evidence to answer this question confidently.",
    sourceNote,
  ].join("\n\n");
};
