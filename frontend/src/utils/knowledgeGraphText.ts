export const normalizeGraphText = (value: unknown) =>
  typeof value === 'string' ? value.replace(/\s+/g, ' ').trim() : '';

const decodeGraphHtmlEntities = (value: string) =>
  value.replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (match, entity: string) => {
    const normalized = entity.toLowerCase();
    const namedEntities: Record<string, string> = {
      amp: '&',
      apos: "'",
      gt: '>',
      lt: '<',
      nbsp: ' ',
      quot: '"',
    };
    const decodeCodePoint = (codePoint: number) =>
      Number.isFinite(codePoint) && codePoint >= 0 && codePoint <= 0x10ffff
        ? String.fromCodePoint(codePoint)
        : match;

    if (namedEntities[normalized]) return namedEntities[normalized];

    if (normalized.startsWith('#x')) {
      const codePoint = Number.parseInt(normalized.slice(2), 16);
      return decodeCodePoint(codePoint);
    }

    if (normalized.startsWith('#')) {
      const codePoint = Number.parseInt(normalized.slice(1), 10);
      return decodeCodePoint(codePoint);
    }

    return match;
  });

export const normalizeGraphDisplayText = (value: unknown) =>
  normalizeGraphText(value)
    .replace(/<\s*(br|\/p|\/li|\/div)\s*\/?>/gi, ' ')
    .replace(/<\/?[a-z][^>]*>/gi, ' ')
    .replace(/&(?:#x?[0-9a-f]+|[a-z]+);/gi, (match) => decodeGraphHtmlEntities(match))
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/\*+/g, '')
    .replace(/\s+/g, ' ')
    .trim();

export const normalizeGraphSearchResultText = (value: unknown) =>
  normalizeGraphDisplayText(value)
    .replace(/[\s|+*_=-]{8,}/g, ' ')
    .replace(/\s*\|\s*/g, ' / ')
    .replace(/(?:\/\s*){2,}/g, '/ ')
    .replace(/^\s*\/\s*|\s*\/\s*$/g, '')
    .replace(/\s+/g, ' ')
    .trim();

export const getGraphTextSearchSnippet = (
  text: string,
  query: string,
  before = 72,
  after = 120
) => {
  if (!query) return '';

  const normalizedText = normalizeGraphSearchResultText(text);
  const index = normalizedText.toLowerCase().indexOf(query.toLowerCase());
  if (index < 0) return '';

  const start = Math.max(0, index - before);
  const end = Math.min(normalizedText.length, index + query.length + after);

  return `${start > 0 ? '...' : ''}${normalizedText.slice(start, end)}${end < normalizedText.length ? '...' : ''}`;
};

export const isGraphEvidenceExcerpt = (value?: string | null) => {
  const text = normalizeGraphDisplayText(value || '').trim();
  return Boolean(text && (text.startsWith('...') || /^[a-z),.;:]/.test(text)));
};

export const trimLeadingGraphEvidenceExcerptFragment = (value: string) => {
  const text = normalizeGraphDisplayText(value).replace(/^\.\.\.\s*/, '').trim();
  const parentheticalRecovery = text.match(/^[a-z][^.!?:]{0,56}\)\s+([A-Z0-9].*)$/);
  if (parentheticalRecovery?.[1]) return parentheticalRecovery[1].trim();

  const labelRecovery = text.match(/^[a-z][a-z0-9/_. -]{0,32}:\s+([A-Z0-9].*)$/);
  if (labelRecovery?.[1]) return labelRecovery[1].trim();

  const dashRecovery = text.match(/^[a-z][a-z0-9/_. ]{0,36}\s+-\s+([A-Z0-9].*)$/);
  if (dashRecovery?.[1]) return dashRecovery[1].trim();

  const leadingStubRecovery = text.match(/^[a-z]{1,4}\s+([A-Z][\w/.-].*)$/);
  if (leadingStubRecovery?.[1]) return leadingStubRecovery[1].trim();

  return text.replace(/^[,.;:)\]\s]+/, '').trimStart();
};

export const formatGraphEvidenceSnippetText = (value?: string | null) => {
  const text = normalizeGraphDisplayText(value || '').trim();
  if (!text || !isGraphEvidenceExcerpt(text)) return text;
  const excerpt = trimLeadingGraphEvidenceExcerptFragment(text);
  return excerpt ? `... ${excerpt}` : text;
};
