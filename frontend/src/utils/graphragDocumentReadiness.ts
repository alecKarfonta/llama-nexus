import { isLowValueGraphRAGTraceSampleText } from './graphragTraceSamples';

type ReadinessTone = 'success' | 'warning' | 'running' | 'neutral';

export interface GraphRAGReadinessLaneInput {
  key: string;
  label: string;
  status?: string;
  sampleCount?: number;
  sampleCountLabel?: string;
  sampleTitle?: string;
  sampleBody?: string;
  warningCount?: number;
  durationMs?: number;
}

export interface GraphRAGReadinessEvidenceInput {
  label: string;
  value?: number;
}

export interface GraphRAGReadinessCard {
  key: string;
  label: string;
  title: string;
  detail: string;
  countLabel: string;
  statusLabel: string;
  tone: ReadinessTone;
  warningCount: number;
}

export interface GraphRAGDocumentReadiness {
  headline: string;
  detail: string;
  statusLabel: string;
  tone: ReadinessTone;
  chips: string[];
  cards: GraphRAGReadinessCard[];
  primaryCard: GraphRAGReadinessCard | null;
}

const COMPLETE_STATUSES = new Set(['complete', 'completed', 'success', 'ready']);
const RUNNING_STATUSES = new Set(['running', 'processing', 'pending', 'submitted']);
const WARNING_STATUSES = new Set(['warning', 'partial', 'skipped', 'preview_only', 'not_configured']);
const ERROR_STATUSES = new Set(['failed', 'error', 'unavailable', 'unhealthy', 'misconfigured']);

const LANE_LABELS: Record<string, string> = {
  content: 'Text parsed',
  chunks: 'References ready',
  relationships: 'Relationships ready',
  paths: 'Paths indexed',
  chat: 'Chat ready',
};

const LANE_PRIORITIES: Record<string, number> = {
  relationships: 1,
  chunks: 2,
  content: 3,
  paths: 4,
  chat: 5,
};

const compactText = (value: string, limit = 96): string => {
  const text = value.replace(/\s+/g, ' ').trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(1, limit - 1)).trim()}...`;
};

const formatNumber = (value: number): string =>
  new Intl.NumberFormat('en-US').format(Math.max(0, Math.round(value)));

const statusTone = (status?: string, warningCount = 0): ReadinessTone => {
  const normalized = String(status || '').toLowerCase();
  if (warningCount > 0 || WARNING_STATUSES.has(normalized) || ERROR_STATUSES.has(normalized)) {
    return 'warning';
  }
  if (RUNNING_STATUSES.has(normalized)) return 'running';
  if (COMPLETE_STATUSES.has(normalized)) return 'success';
  return 'neutral';
};

const statusLabel = (status?: string): string => {
  const normalized = String(status || '').replace(/[_-]+/g, ' ').trim();
  if (!normalized) return 'ready';
  return normalized.replace(/\b\w/g, letter => letter.toUpperCase());
};

const cardTitleForLane = (lane: GraphRAGReadinessLaneInput): string => {
  const count = Math.max(0, Math.round(lane.sampleCount || 0));
  const sampleTitle = compactText(lane.sampleTitle || '');
  if (
    sampleTitle &&
    !isLowValueGraphRAGTraceSampleText(`${sampleTitle} ${lane.sampleBody || ''}`)
  ) {
    return sampleTitle;
  }
  if (lane.key === 'relationships' && count > 0) return `${formatNumber(count)} relationships`;
  if (lane.key === 'chunks' && count > 0) return `${formatNumber(count)} reference chunks`;
  if (lane.key === 'content' && count > 0) return `${formatNumber(count)} text items`;
  if (lane.key === 'paths' && count > 0) return `${formatNumber(count)} graph paths`;
  if (lane.key === 'chat' && count > 0) return `${formatNumber(count)} chat signals`;
  return compactText(lane.sampleTitle || lane.label || 'Ready');
};

export const buildGraphRAGDocumentReadiness = ({
  lanes,
  evidence,
  questionCount = 0,
}: {
  lanes: GraphRAGReadinessLaneInput[];
  evidence?: GraphRAGReadinessEvidenceInput[];
  questionCount?: number;
}): GraphRAGDocumentReadiness => {
  const cards = [...lanes]
    .filter(lane => lane.sampleCount || lane.sampleTitle || lane.status)
    .sort((left, right) =>
      (LANE_PRIORITIES[left.key] || 99) - (LANE_PRIORITIES[right.key] || 99) ||
      left.label.localeCompare(right.label),
    )
    .map((lane): GraphRAGReadinessCard => ({
      key: lane.key,
      label: LANE_LABELS[lane.key] || lane.label,
      title: cardTitleForLane(lane),
      detail: compactText(lane.sampleBody || lane.sampleTitle || lane.sampleCountLabel || '', 140),
      countLabel: lane.sampleCountLabel || (lane.sampleCount ? `${formatNumber(lane.sampleCount)} evidence items` : statusLabel(lane.status)),
      statusLabel: lane.warningCount ? `${lane.warningCount} issue${lane.warningCount === 1 ? '' : 's'}` : statusLabel(lane.status),
      tone: statusTone(lane.status, lane.warningCount),
      warningCount: lane.warningCount || 0,
    }));

  const readyCount = cards.filter(card => card.tone === 'success').length;
  const warningCount = cards.reduce((total, card) => total + card.warningCount + (card.tone === 'warning' && card.warningCount === 0 ? 1 : 0), 0);
  const runningCount = cards.filter(card => card.tone === 'running').length;
  const sampleTotal = lanes.reduce((total, lane) => total + Math.max(0, Math.round(lane.sampleCount || 0)), 0);
  const topEvidence = [...(evidence || [])]
    .filter(item => item.label)
    .sort((left, right) => (right.value || 0) - (left.value || 0) || left.label.localeCompare(right.label))
    .slice(0, 2)
    .map(item => item.value ? `${formatNumber(item.value)} ${item.label}` : item.label);

  const chips = [
    ...topEvidence,
    questionCount > 0 ? `${formatNumber(questionCount)} suggested questions` : '',
  ].filter(Boolean);

  const headline = warningCount > 0
    ? 'Import needs review'
    : runningCount > 0
      ? 'Import still running'
      : readyCount > 0
        ? 'Ready for chat'
        : 'Import evidence';

  const detail = [
    cards.length > 0 ? `${formatNumber(readyCount)}/${formatNumber(cards.length)} checks` : '',
    sampleTotal > 0 ? `${formatNumber(sampleTotal)} items` : '',
    warningCount > 0 ? `${formatNumber(warningCount)} issue${warningCount === 1 ? '' : 's'}` : '',
  ].filter(Boolean).join(' · ') || 'No evidence items yet';

  const primaryCard =
    cards.find(card =>
      !isLowValueGraphRAGTraceSampleText(`${card.title} ${card.detail}`),
    ) || cards[0] || null;

  return {
    headline,
    detail,
    statusLabel: cards.length > 0 ? `${formatNumber(readyCount)}/${formatNumber(cards.length)} ready` : 'No evidence',
    tone: warningCount > 0 ? 'warning' : runningCount > 0 ? 'running' : readyCount > 0 ? 'success' : 'neutral',
    chips,
    cards,
    primaryCard,
  };
};
