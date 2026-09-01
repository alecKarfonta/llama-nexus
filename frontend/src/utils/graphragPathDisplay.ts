import {
  cleanGraphReferenceLabel,
  isDocumentLikeGraphLabel,
} from "./graphragReferenceDisplay";

const uniqueStrings = (values: Array<string | undefined | null>): string[] => {
  const seen = new Set<string>();
  const output: string[] = [];

  values.forEach((value) => {
    const normalized = String(value || "").trim();
    const key = normalized.toLowerCase();
    if (!normalized || seen.has(key)) return;
    seen.add(key);
    output.push(normalized);
  });

  return output;
};

export const displayPathNodes = (nodes?: string[]): string[] => {
  const cleanedNodes = uniqueStrings(
    (nodes || []).map((node) => cleanGraphReferenceLabel(node, "Node")),
  );
  if (cleanedNodes.length <= 2) return cleanedNodes;

  return cleanedNodes.reduce<string[]>((displayNodes, node, index) => {
    if (!isDocumentLikeGraphLabel(node)) return [...displayNodes, node];

    const bridgeLabel =
      index > 0 && index < cleanedNodes.length - 1
        ? "source evidence"
        : "source document";

    return displayNodes[displayNodes.length - 1] === bridgeLabel
      ? displayNodes
      : [...displayNodes, bridgeLabel];
  }, []);
};
