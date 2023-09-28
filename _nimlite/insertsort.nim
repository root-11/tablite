proc insertSort*[T](a: var openarray[T], comparator: (proc (x: T, y: T): bool {.inline.})): void {.inline.} =
    # our array is likely to be nearly sorted or already sorted, therefore the complexity is better than bubble sort
    for i in 1 .. a.high:
        let value = a[i]
        var j = i
        while j > 0 and comparator(value, a[j-1]):
            a[j] = a[j-1]
            dec j
        a[j] = value