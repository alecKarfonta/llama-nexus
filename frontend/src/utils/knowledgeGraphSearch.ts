export const blendGraphSearchResults = <T>(
  primaryResults: T[],
  secondaryResults: T[],
  limit = 80
) => {
  const blended: T[] = [];
  let primaryIndex = 0;
  let secondaryIndex = 0;

  while (
    blended.length < limit &&
    (primaryIndex < primaryResults.length || secondaryIndex < secondaryResults.length)
  ) {
    if (primaryIndex < primaryResults.length) {
      blended.push(primaryResults[primaryIndex]);
      primaryIndex += 1;
      if (blended.length >= limit) break;
    }

    if (secondaryIndex < secondaryResults.length) {
      blended.push(secondaryResults[secondaryIndex]);
      secondaryIndex += 1;
    }
  }

  return blended;
};
