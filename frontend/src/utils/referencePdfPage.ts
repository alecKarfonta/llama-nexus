export interface ReferencePageLocation {
  label: string;
  value: string;
  category?: string;
}

export interface ReferencePageTarget {
  title?: string;
  subtitle?: string;
  body?: string;
  sourceLocations?: ReferencePageLocation[];
  locationHints?: string[];
}

const positivePageNumber = (value: number): number | null =>
  Number.isFinite(value) && value >= 1 ? Math.floor(value) : null;

const pageNumberFromText = (value: string): number | null => {
  const text = value.replace(/\s+/g, " ").trim();
  if (!text) return null;

  const pagePattern = /\b(?:page|p\.?)\s*[#:=/-]?\s*(\d+)\b/i;
  const pageMatch = text.match(pagePattern);
  if (pageMatch) return positivePageNumber(Number(pageMatch[1]));

  const keyValueMatch = text.match(/\bpage(?:_number)?\s*=\s*(\d+)\b/i);
  if (keyValueMatch) return positivePageNumber(Number(keyValueMatch[1]));

  return null;
};

export const pageNumberFromReference = (
  reference: ReferencePageTarget,
): number | null => {
  for (const location of reference.sourceLocations || []) {
    const label = `${location.label || ""} ${location.category || ""}`.toLowerCase();
    const value = String(location.value || "").trim();
    if (/\bpage(?:_number)?\b/.test(label)) {
      const directNumber = Number(value);
      const pageNumber = positivePageNumber(directNumber);
      if (pageNumber) return pageNumber;
    }
    if (/\bpage[_\s-]*index\b/.test(label)) {
      const indexNumber = Number(value);
      if (Number.isFinite(indexNumber) && indexNumber >= 0) {
        return Math.floor(indexNumber) + 1;
      }
    }

    const locationPage = pageNumberFromText(
      `${location.label} ${location.value} ${location.category || ""}`,
    );
    if (locationPage) return locationPage;
  }

  const values = [
    reference.title,
    reference.subtitle,
    reference.body,
    ...(reference.locationHints || []),
  ];
  for (const value of values) {
    const pageNumber = pageNumberFromText(String(value || ""));
    if (pageNumber) return pageNumber;
  }
  return null;
};

export const pageNumbersFromReferences = (
  references: ReferencePageTarget[],
  limit = 12,
): number[] => {
  const pages = new Set<number>();
  for (const reference of references) {
    const pageNumber = pageNumberFromReference(reference);
    if (pageNumber) pages.add(pageNumber);
  }
  return Array.from(pages)
    .sort((left, right) => left - right)
    .slice(0, limit);
};
