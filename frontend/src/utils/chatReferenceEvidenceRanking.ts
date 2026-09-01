export interface ChatReferenceRelationshipLike {
  source?: unknown;
  relation?: unknown;
  relationship_type?: unknown;
  target?: unknown;
}

const normalizeRelationshipKeyPart = (value: unknown): string =>
  String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ");

const normalizeRelationshipRelationKey = (value: unknown): string =>
  normalizeRelationshipKeyPart(value).replace(/\s+/g, "_");

const HIGH_SIGNAL_RELATION_KEYS = new Set([
  "approves",
  "blocks",
  "decision",
  "depends",
  "drives",
  "gate",
  "gates",
  "influences",
  "requires",
  "supports",
]);

const HIGH_SIGNAL_TARGET_RELATION_PATTERN =
  /^(?:gates?|requires?|supports?|blocks?|drives?)\s+/i;

const isHighSignalDirectRelationship = (
  relationship: ChatReferenceRelationshipLike,
): boolean => {
  const relationKey = normalizeRelationshipRelationKey(
    relationship.relation || relationship.relationship_type,
  );
  if (HIGH_SIGNAL_RELATION_KEYS.has(relationKey)) return true;
  return (
    relationKey === "relation" &&
    HIGH_SIGNAL_TARGET_RELATION_PATTERN.test(
      normalizeRelationshipKeyPart(relationship.target),
    )
  );
};

export const chatReferenceRelationshipKey = (
  relationship: ChatReferenceRelationshipLike,
): string => {
  const relation = relationship.relation || relationship.relationship_type;

  return [
    normalizeRelationshipKeyPart(relationship.source),
    normalizeRelationshipKeyPart(relation),
    normalizeRelationshipKeyPart(relationship.target),
  ].join("|");
};

export const selectAnswerEvidenceDirectRelationships = <
  T extends ChatReferenceRelationshipLike,
>(
  relationships: T[],
  evidenceCards: ChatReferenceRelationshipLike[],
  limit = 10,
): T[] => {
  const relationshipLimit = Math.max(limit, 0);
  const availableRelationships = relationships.filter(Boolean);
  const evidenceKeys = new Set(
    evidenceCards
      .filter((card) => card?.source || card?.target)
      .map(chatReferenceRelationshipKey)
      .filter((key) => key !== "||"),
  );

  if (evidenceKeys.size === 0) return availableRelationships.slice(0, relationshipLimit);

  const selectedKeys = new Set<string>();
  const selected = availableRelationships.filter((relationship) => {
    const key = chatReferenceRelationshipKey(relationship);
    const matchesEvidenceCard = evidenceKeys.has(key);
    if (matchesEvidenceCard) selectedKeys.add(key);
    return matchesEvidenceCard;
  });
  const supplemental = availableRelationships.filter((relationship) => {
    const key = chatReferenceRelationshipKey(relationship);
    return !selectedKeys.has(key) && isHighSignalDirectRelationship(relationship);
  });

  return [...selected, ...supplemental].slice(0, relationshipLimit);
};
