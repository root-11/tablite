type Mask* = enum
    INVALID = -1
    UNUSED = 0
    VALID = 1

proc unusedMaskSearch*(arr: var seq[Mask]): int =
    # Partitioned search for faster unused mask retrieval
    const stepSize = 50_000
    let len = arr.len

    if arr[^1] != Mask.UNUSED:
        # page is full
        return arr.len

    if arr[0] == MASK.UNUSED:
        # page is completely empty
        return 0

    var i = 0

    while i < len:
        let nextIndex = min(i + stepSize, arr.len)
        let lastIndex = nextIndex - 1

        if arr[lastIndex] != Mask.UNUSED:
            # if the last element in the step is used, we can skip `stepSize`
            i = nextIndex
            continue

        # last element is unused, therefore we need to check
        for j in i..lastIndex:
            if arr[j] == Mask.UNUSED:
                return j

        i = nextIndex

    return 0