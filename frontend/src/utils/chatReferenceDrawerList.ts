export const DEFAULT_REFERENCE_DRAWER_LIMIT = 5

export interface ReferenceDrawerRowsOptions {
    showAll?: boolean
    search?: string
    kindFilter?: string
    limit?: number
}

export interface ReferenceDrawerRows<T> {
    rows: T[]
    hiddenCount: number
    capped: boolean
}

export const selectReferenceDrawerRows = <T>(
    references: T[],
    options: ReferenceDrawerRowsOptions = {},
): ReferenceDrawerRows<T> => {
    const limit = Math.max(0, options.limit ?? DEFAULT_REFERENCE_DRAWER_LIMIT)
    const searchActive = Boolean(options.search?.trim())
    const filterActive = Boolean(options.kindFilter && options.kindFilter !== 'all')
    const capped = Boolean(
        !options.showAll &&
        !searchActive &&
        !filterActive &&
        references.length > limit,
    )
    const rows = capped ? references.slice(0, limit) : references

    return {
        rows,
        hiddenCount: Math.max(0, references.length - rows.length),
        capped,
    }
}
