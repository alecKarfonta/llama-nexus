export type ChatReasoningLevel = "low" | "medium" | "high" | "none";

type PromptMessage = {
  role: string;
  content: unknown;
};

const DEFAULT_SYSTEM_INSTRUCTION =
  "You are a helpful AI assistant. Answer directly and do not include chain-of-thought, hidden reasoning, scratch work, or a Thinking Process section.";
const REASONING_PATTERN = /Reasoning:\s*\w+/i;
const GRAPH_REASONING_CONTRACT_TITLE = "GraphRAG answer contract:";
const GRAPH_CONTEXT_MARKERS = [
  /--- Knowledge Graph Context ---/i,
  /=== Graph Answer Plan ===/i,
  /=== Graph Reasoning Summary ===/i,
  /=== Graph Paths ===/i,
  /=== Bridge Entities ===/i,
  /=== Relevant Graph Relationships ===/i,
  /=== Requested Evidence Coverage ===/i,
  /=== Local Pipeline Graph Overview ===/i,
  /=== Pipeline Interaction Handoff ===/i,
  /\bEvidence checklist:/i,
  /\bGraph paths:/i,
  /\bGraph relationships:/i,
  /\bBridge entities:/i,
  /--\[[^\]]+\]--?>/,
];

const EXPLICIT_GRAPH_QUERY_PATTERN =
  /\b(?:graph|relationship|relationships|relation|edge|edges|path|paths|hop|hops|bridge|bridges|through|via|connect|connected|connects|connection|connections|dependency|dependencies|lineage|connector|connectors|diagram|visual|chart|charts|slide|slides|table|tables|shape|shapes|visual group|shape group|grouped visual)\b/i;

const DOCUMENT_EVIDENCE_QUERY_PATTERN =
  /\b(?:document|file|source|sources|passage|passages|quote|quotes|cite|citation|citations|evidence|summari[sz]e|report|reports|reported|result|results|what does|what is|which|where)\b/i;

const stringifyPromptContent = (content: unknown): string => {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";

  return content
    .map((part) => {
      if (!part || typeof part !== "object") return "";
      const record = part as Record<string, unknown>;
      return typeof record.text === "string" ? record.text : "";
    })
    .filter(Boolean)
    .join("\n");
};

const latestUserQuery = (messages: PromptMessage[]): string => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user") return stringifyPromptContent(message.content);
  }
  return "";
};

export const isExplicitGraphRAGQuestion = (queryText: string): boolean =>
  EXPLICIT_GRAPH_QUERY_PATTERN.test(queryText || "");

const isDocumentFirstGraphRAGQuestion = (queryText: string): boolean =>
  DOCUMENT_EVIDENCE_QUERY_PATTERN.test(queryText || "") &&
  !isExplicitGraphRAGQuestion(queryText);

const graphQuestionModeInstruction = (queryText: string): string => {
  if (isExplicitGraphRAGQuestion(queryText)) {
    return [
      "Question mode: explicit graph reasoning.",
      "- It is appropriate to lead with graph relationships, graph paths, and hop labels when they directly answer the user's graph/path/relationship question.",
      "- Still cite source-document provenance for every graph claim when it is available.",
    ].join("\n");
  }

  if (isDocumentFirstGraphRAGQuestion(queryText)) {
    return [
      "Question mode: document answer.",
      "- Citation, evidence, source, and support words alone do not make this a graph/path question.",
      "- Lead with the answer from the selected document, retrieved passage, or source-document evidence in plain language.",
      "- Do not lead with or create headings named Direct Graph Relationship, Direct Relationship, Supporting Graph Path, Graph Path, or Relevant Graph Relationships.",
      "- If graph relationships clarify the answer, mention them only after the source citation as brief related context.",
    ].join("\n");
  }

  return [
    "Question mode: source-first answer.",
    "- Lead with source-document evidence unless the user explicitly asks for graph/path/relationship reasoning.",
    "- Use graph relationships as brief supporting context after the document answer.",
  ].join("\n");
};

const graphReasoningInstruction = (queryText: string): string => [
  GRAPH_REASONING_CONTRACT_TITLE,
  graphQuestionModeInstruction(queryText),
  "- For ordinary document questions, answer from quoted retrieved passages or source-document evidence first. Use graph relationships as supporting context only when they clarify the source evidence.",
  "- Use graph paths, relationship chains, and hop-by-hop labels first only when the user explicitly asks for a graph path, relationship, connection, bridge, edge, dependency, source lineage, visual connector, or how entities are connected.",
  "- If Relevant Graph Relationships directly answer an explicit graph, visual, relationship, graph-citation, or graph-support question, answer from those direct relationships before using Graph Answer Plan or Graph Paths.",
  "- If a Graph Answer Plan is present, follow its strategy and ordered hops for explicit graph/path questions unless a direct relationship answers the user's specific graph/source/visual/support question more precisely.",
  "- If the plan includes an Evidence checklist, use it as the citation order for graph hops and supporting relationships.",
  "- If a Pipeline Interaction Handoff is present, use its graph anchors, lanes, starter prompts, retrieval rehearsals, and citation scopes silently to choose evidence; do not describe the handoff mechanics unless the user asks for diagnostics.",
  "- Treat graph paths and relationships as structured evidence, not as ordinary retrieved snippets.",
  "- Prefer a direct graph relationship over an indirect graph path when both mention the user's source and target entities. Only describe the connection as indirect when no direct relationship is listed.",
  "- For grouped visual evidence, name the group/shape/connector relationship exactly when listed, such as groups_visual, shape_group, or connector provenance. Do not replace it with a broader multi-hop path.",
  "- When the user asks which chart, table, slide, visual, or source evidence supports a relationship, name the specific cited evidence label from context when present, such as slide title, table row/cell, chart series/value, shape, or connector.",
  "- When the user asks for extracted evidence families such as charts, tables, form fields, OCR, or styled text, address each requested family explicitly; if a Requested Evidence Coverage section is present, treat it as cited source-document evidence and do not say a listed family is missing.",
  "- Only say a requested evidence family was not found when neither Requested Evidence Coverage nor graph relationships nor retrieved passages list evidence for that family.",
  "- Distinguish relationship scope explicitly: say whether the evidence is a direct graph relationship, an indirect graph path, or source-document evidence supporting a claim.",
  "- For connection questions, include a concise Graph path sentence before the hop-by-hop explanation, naming whether it is direct or indirect.",
  "- Treat source document/source evidence nodes as provenance bridges. Do not describe them as domain entities or as the substance of the answer.",
  "- When a Graph Reasoning Summary names a strongest path, answer from that path first only for explicit path/connection questions; for ordinary source, citation, or document-summary questions, cite the retrieved passage or source-document evidence first.",
  "- If no direct relationship exists but an indirect path does, do not answer as though nothing was found. Explain the path and say it is indirect.",
  "- For explicit connection, graph location, graph source-lineage, impact, or how/why relationship questions, walk the relationship chain hop by hop and name relationship labels.",
  "- Cite human-readable provenance attached to the path or relationship, including file, row, page, slide, table, chart, connector, or chunk when present.",
  "- For upload trace summaries, keep the answer compact enough to complete: lead with the top entities, top relationships, and best path/citation examples instead of exhaustive lists.",
  "- Keep final answers source-first and plain-language. Do not expose internal retrieval details such as run ids, pipeline_run_id values, graph_path_reasoning modes, chunk match score, path score, token match reasons, retrieval rehearsals, graph ranking scores, or a `Pipeline Trace` section unless the user explicitly asks for diagnostics.",
  "- Never print raw numeric graph scores such as `score=0.788` or `79% score` in the answer. Use qualitative confidence only when it helps the user.",
  "- When scores are present in context, translate them into qualitative confidence only if needed; explain confidence through source agreement such as extracted connectors, slide text, chart values, table rows, or cited passages, not through semantic matching or token matching. Leave raw scoring details to the source review UI.",
  "- If the graph summary says no connecting path was found, say that specifically before falling back to retrieved passages or extracted entities.",
  "- Do not invent graph edges. If evidence conflicts, prefer higher-scored paths and call out uncertainty.",
].join("\n");

const appendSegment = (base: string, segment: string): string => {
  if (!segment) return base;
  if (!base) return segment;
  if (base.endsWith("\n") || segment.startsWith("\n"))
    return `${base}${segment}`;
  return `${base}\n${segment}`;
};

const applyReasoningInstruction = (
  content: string,
  reasoningLevel: ChatReasoningLevel,
): string => {
  if (reasoningLevel === "none" || reasoningLevel === "low") return content;

  const reasoningInstruction = `Reasoning: ${reasoningLevel}`;
  if (REASONING_PATTERN.test(content)) {
    return content.replace(REASONING_PATTERN, reasoningInstruction);
  }

  return appendSegment(content, reasoningInstruction);
};

const hasGraphKnowledgeContext = (contextText: string): boolean =>
  Boolean(contextText) &&
  GRAPH_CONTEXT_MARKERS.some((pattern) => pattern.test(contextText));

const applyGraphReasoningInstruction = (
  content: string,
  contextText: string,
  queryText: string,
): string => {
  if (
    !hasGraphKnowledgeContext(contextText) ||
    content.includes(GRAPH_REASONING_CONTRACT_TITLE)
  ) {
    return content;
  }

  return appendSegment(content, graphReasoningInstruction(queryText));
};

export const applyKnowledgeSystemPrompt = <T extends PromptMessage>(
  messages: T[],
  options: {
    contextText?: string;
    reasoningLevel: ChatReasoningLevel;
    baseInstruction?: string;
    userQuery?: string;
  },
): T[] => {
  const contextText = options.contextText || "";
  const baseInstruction = options.baseInstruction || DEFAULT_SYSTEM_INSTRUCTION;
  const queryText = options.userQuery || latestUserQuery(messages);
  const needsSystemPrompt =
    Boolean(contextText) || options.reasoningLevel !== "none";

  if (!needsSystemPrompt) {
    return messages;
  }

  const buildContent = (existingContent?: unknown): string => {
    const baseContent =
      typeof existingContent === "string" && existingContent.trim()
        ? existingContent
        : baseInstruction;
    const withGraphReasoning = applyGraphReasoningInstruction(
      baseContent,
      contextText,
      queryText,
    );
    const withContext = appendSegment(withGraphReasoning, contextText);
    return applyReasoningInstruction(withContext, options.reasoningLevel);
  };

  if (messages.length > 0 && messages[0].role === "system") {
    return [
      {
        ...messages[0],
        content: buildContent(messages[0].content),
      },
      ...messages.slice(1),
    ];
  }

  return [
    {
      role: "system",
      content: buildContent(baseInstruction),
    } as T,
    ...messages,
  ];
};
