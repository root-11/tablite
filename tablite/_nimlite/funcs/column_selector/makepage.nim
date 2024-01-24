import std/[enumerate, unicode, tables, times]
import ../../numpy
import ../../pytypes
from mask import Mask

proc canBeNone*(page: BaseNDArray): bool =
    var canBeNone {.noinit.}: bool

    case page.kind:
    of K_STRING: canBeNone = true
    of K_OBJECT:
        let types = page.getPageTypes()

        if K_NONETYPE in types:
            if likely(types[K_NONETYPE] > 0):
                # there is at least one none in the page, we need to check for it
                canBeNone = true

        if not canBeNone and K_STRING in types:
            if likely(types[K_STRING] > 0):
                # there can be at least one empty string in the page, we need to check forit
                canBeNone = true
    else:
        canBeNone = false

    return canBeNone

template makePage*[T: typed](dt: typedesc[T], page: BaseNDArray, mask: var seq[Mask], reason_lst: var seq[string], conv: proc, allow_empty: bool, original_name: string, desired_name: string, desired_type: KindObjectND): BaseNDArray =
    template getTypeUserName(t: KindObjectND): string =
        case t:
        of K_BOOLEAN: "bool"
        of K_INT: "int"
        of K_FLOAT: "float"
        of K_STRING: "str"
        of K_DATE: "date"
        of K_TIME: "time"
        of K_DATETIME: "datetime"
        of K_NONETYPE: "NoneType"

    template createCastErrorReason(v: string, kind: KindObjectND): string =
        "Cannot cast '" & v & "' from '" & original_name & "[" & getTypeUserName(kind) & "]' to '" & desired_name & "[" & getTypeUserName(desired_type) & "]'."

    when page is ObjectNDArray:
        template createNoneErrorReason(): string =
            "'" & original_name & "' cannot be empty in '" & desired_name & "[" & getTypeUserName(desired_type) & "]'."

    when page is UnicodeNDArray or page is ObjectNDArray:
        template createEmptyErrorReason(): string =
            "'" & original_name & "' cannot be empty string in '" & desired_name & "[" & getTypeUserName(desired_type) & "]'."

    when page is ObjectNDArray:
        # this is an object page, check for strings and nones
        let canBeNone = page.canBeNone
    else:
        var inTypeKind {.noinit.}: KindObjectND

        case page.kind:
        of K_BOOLEAN: inTypeKind = KindObjectND.K_BOOLEAN
        of K_INT8, K_INT16, K_INT32, K_INT64: inTypeKind = KindObjectND.K_INT
        of K_FLOAT32, K_FLOAT64: inTypeKind = KindObjectND.K_FLOAT
        of K_DATE: inTypeKind = KindObjectND.K_DATE
        of K_DATETIME: inTypeKind = KindObjectND.K_DATETIME
        of K_STRING: inTypeKind = KindObjectND.K_STRING
        of K_OBJECT: discard # we're never this type

    when T is UnicodeNDArray:
        # string arrays are first saved into separate sequences, after we collect them we will copy their memory to single buffer
        var longest = 1
        var runeBuf = newSeqOfCap[seq[Rune]](page.len)
    else:
        when T is ObjectNDArray:
            var dtypes = {
                KindObjectND.K_NONETYPE: 0,
                desired_type: 0
            }.toTable

        # we know the maximum size of the sequence based on the page size as we cannot manifest new values
        var buf = newSeqOfCap[T.baseType](page.len)

    when page is UnicodeNDArray or page is ObjectNDArray:
        template addEmpty(i: int): void =
            when T is UnicodeNDArray:
                runeBuf.add(newSeq[Rune](0))
            else:
                when T is ObjectNDArray:
                    buf.add(PY_None)
                    dtypes[KindObjectND.K_NONETYPE] = dtypes[KindObjectND.K_NONETYPE] + 1
            mask[i] = Mask.VALID

    for (i, v) in enumerate(page.pgIter):
        # 0. Check if already invalid, if so, skip
        if mask[i] == Mask.INVALID:
            continue

        # 1. Emptiness test
        when page is UnicodeNDArray:
            # original page is made of strings, check for empty strings
            if unlikely(v.len == 0):
                # empty string
                if not allow_empty:
                    mask[i] = Mask.INVALID
                    reason_lst[i] = createEmptyErrorReason()
                else:
                    addEmpty(i)
                continue
        else:
            when page is ObjectNDArray:
                # original page is made of python objects, check for empty strings and nones
                if unlikely(canBeNone):
                    # there can be empties in the page
                    case v.kind:
                    of K_STRING:
                        if PY_String(v).value.len == 0:
                            # empty string
                            if not allow_empty:
                                mask[i] = Mask.INVALID
                                reason_lst[i] = createEmptyErrorReason()
                            else:
                                addEmpty(i)
                            continue
                    of K_NONETYPE:
                        if not allow_empty:
                            # empty object
                            mask[i] = Mask.INVALID
                            reason_lst[i] = createNoneErrorReason()
                        else:
                            addEmpty(i)
                        continue
                    else:
                        # other types are never empty
                        discard

        # 2. Attempt type conversion
        try:
            when T is UnicodeNDArray:
                # our result is a string page, so we need to know the maximum length and convert to runes
                let str = conv(v)
                let res = str.toRunes

                longest = max(longest, res.len)
                runeBuf.add(res)
            else:
                buf.add(conv(v))

            when T is ObjectNDArray:
                dtypes[desired_type] = dtypes[desired_type] + 1
            mask[i] = Mask.VALID
        except ValueError:
            # 2b. we couldn't cast the type, generate error string
            when page is ObjectNDArray:
                let inTypeKind = v.kind
                let strRepr = v.toRepr
            else:
                when page is DateNDArray:
                    let strRepr = v.format(fmtDate)
                else:
                    when page is DateTimeNDArray:
                        let strRepr = v.format(fmtDateTime)
                    else:
                        let strRepr = $v

            reason_lst[i] = createCastErrorReason(strRepr, inTypeKind)
            mask[i] = Mask.INVALID

    # 3. Dump the page
    when T is UnicodeNDArray:
        # we need to turn the individual strings into contigous block
        let strCount = runeBuf.len
        var buf = newSeq[Rune](longest * strCount)

        for (i, str) in enumerate(runeBuf):
            buf[i * longest].addr.copyMem(addr str[0], str.len * sizeof(Rune))

        let res = T(shape: @[strCount], buf: buf, size: longest, kind: T.pageKind)
    else:
        when T is ObjectNDArray:
            let res = T(shape: @[buf.len], dtypes: dtypes, buf: buf, kind: T.pageKind)
        else:
            let res = T(shape: @[buf.len], buf: buf, kind: T.pageKind)

    BaseNDArray res
