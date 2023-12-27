import std/[sugar, sequtils, unicode, enumerate]
import numpy, pickling, ranking, infertypes, encfile, csvparse, pytypes

type PageType = enum
    PG_UNSET,
    PG_UNICODE,
    PG_INT32_SIMPLE, PG_INT32_US, PG_INT32_EU,
    PG_FLOAT32_SIMPLE, PG_FLOAT32_US, PG_FLOAT32_EU,
    PG_BOOL,
    PG_OBJECT,
    PG_DATE,
    PG_DATETIME
    PG_DATE_SHORT

var none_str = ""

proc collectPageInfo*(
        obj: var ReaderObj, fh: var BaseEncodedFile,
        guess_dtypes: bool, n_pages: int, row_count: int,
        import_fields: var seq[uint]
    ): (uint, seq[uint], seq[Rank]) =
    var ranks: seq[Rank]
    var longest_str = collect(newSeqOfCap(n_pages)):
        for _ in 0..n_pages-1:
            1u
    
    var n_rows: uint = 0

    if guess_dtypes:
        ranks = collect(newSeqOfCap(n_pages)):
            for _ in 0..n_pages-1:
                newRank()
    else:
        ranks = newSeq[Rank](0)

    

    for (row_idx, fields, field_count) in obj.parseCSV(fh):
        if row_count >= 0 and row_idx >= (uint row_count):
            break

        var fidx = -1

        for idx in 0..field_count-1:
            if not ((uint idx) in import_fields):
                continue

            inc fidx

            # let fidx = uint idx
            let field = fields[idx]

            if fidx < 0 or fidx >= n_pages:
                raise newException(Exception, "what")

            if not guess_dtypes:
                longest_str[fidx] = max(uint field.runeLen, longest_str[fidx])
            else:
                let rank = addr ranks[fidx]
                let dt = rank[].updateRank(field.unsafeAddr)

                if dt == DataTypes.DT_STRING:
                    longest_str[fidx] = max(uint field.runeLen, longest_str[fidx])

        for idx in (fidx+1)..n_pages-1:
            # fill missing fields with nones
            longest_str[idx] = max(uint none_str.len, longest_str[idx])

            if guess_dtypes:
                discard ranks[idx].updateRank(addr none_str)

        inc n_rows

    return (n_rows, longest_str, ranks)

proc isAnyInt(dt: PageType): bool {.inline.} =
    return dt == PageType.PG_INT32_SIMPLE or dt == PageType.PG_INT32_US or dt == PageType.PG_INT32_EU

proc isAnyInt(dt: DataTypes): bool {.inline.} =
    return dt == DataTypes.DT_INT_SIMPLE or dt == DataTypes.DT_INT_US or dt == DataTypes.DT_INT_EU

proc isAnyFloat(dt: PageType): bool {.inline.} =
    return dt == PageType.PG_FLOAT32_SIMPLE or dt == PageType.PG_FLOAT32_US or dt == PageType.PG_FLOAT32_EU

proc isAnyFloat(dt: DataTypes): bool {.inline.} =
    return dt == DataTypes.DT_FLOAT_SIMPLE or dt == DataTypes.DT_FLOAT_US or dt == DataTypes.DT_FLOAT_EU

proc dumpPageHeader*(
        destinations: var seq[string],
        n_pages: int, n_rows: uint, guess_dtypes: bool,
        longest_str: var seq[uint], ranks: var seq[Rank]
    ): (seq[File], seq[PageType], uint32) =
    let page_file_handlers = collect(newSeqOfCap(n_pages)):
        for p in destinations:
            open(p, fmWrite)

    var column_dtypes = newSeq[PageType](n_pages)
    var binput: uint32 = 0

    if not guess_dtypes:
        for idx, (fh, i) in enumerate(zip(page_file_handlers, longest_str)):
            column_dtypes[idx] = PageType.PG_UNICODE
            fh.writeNumpyHeader("<U" & $i, n_rows)
    else:
        for i in 0..n_pages-1:
            let fh = page_file_handlers[i]
            let rank = addr ranks[i]
            var dtype = column_dtypes[i]

            rank[].sortRanks(false) # sort accounting for strings, so that if string is primary type, everything overlaps to string

            for it in rank[].iter():
                let dt = it[0]
                let count = it[1]

                if count == 0:
                    break

                if dtype == PageType.PG_UNSET:
                    case dt:
                        of DataTypes.DT_INT_SIMPLE: dtype = PageType.PG_INT32_SIMPLE
                        of DataTypes.DT_INT_US: dtype = PageType.PG_INT32_US
                        of DataTypes.DT_INT_EU: dtype = PageType.PG_INT32_EU
                        of DataTypes.DT_FLOAT_SIMPLE: dtype = PageType.PG_FLOAT32_SIMPLE
                        of DataTypes.DT_FLOAT_US: dtype = PageType.PG_FLOAT32_US
                        of DataTypes.DT_FLOAT_EU: dtype = PageType.PG_FLOAT32_EU
                        of DataTypes.DT_STRING:
                            dtype = PageType.PG_UNICODE
                            break # if the first type is string, everying is a subset of string
                        of DataTypes.DT_DATETIME, DataTypes.DT_DATETIME_US:
                            dtype = PageType.PG_DATETIME
                        of DataTypes.DT_DATE, DataTypes.DT_DATE_US:
                            dtype = PageType.PG_DATE
                        of DataTypes.DT_DATE_SHORT:
                            dtype = PageType.PG_DATE_SHORT
                        else: dtype = PageType.PG_OBJECT
                    continue


                # check overlapping types
                if isAnyFloat(dtype) and isAnyInt(dt) or dt == DataTypes.DT_DATE_SHORT: discard # float overlaps ints
                elif isAnyInt(dtype) and isAnyFloat(dt): # int is a subset of float, change to float
                    if dt == DataTypes.DT_FLOAT_SIMPLE:
                        dtype = PageType.PG_FLOAT32_SIMPLE
                    elif dt == DataTypes.DT_FLOAT_US:
                        dtype = PageType.PG_FLOAT32_US
                    elif dt == DataTypes.DT_FLOAT_EU:
                        dtype = PageType.PG_FLOAT32_EU
                    else:
                        raise newException(Exception, "invalid")
                elif isAnyInt(dtype) and dt == DataTypes.DT_DATE_SHORT: discard # int is a subset of int, change to float
                elif dtype == PageType.PG_DATE_SHORT:
                    case dt:
                        of DataTypes.DT_FLOAT_SIMPLE: dtype = PageType.PG_FLOAT32_SIMPLE
                        of DataTypes.DT_FLOAT_US: dtype = PageType.PG_FLOAT32_US
                        of DataTypes.DT_FLOAT_EU: dtype = PageType.PG_FLOAT32_EU
                        of DataTypes.DT_INT_SIMPLE: dtype = PageType.PG_INT32_SIMPLE
                        of DataTypes.DT_INT_US: dtype = PageType.PG_INT32_US
                        of DataTypes.DT_INT_EU: dtype = PageType.PG_INT32_EU
                        else: discard
                else: dtype = PageType.PG_OBJECT # types cannot overlap

            case dtype:
                of PageType.PG_UNICODE: fh.writeNumpyHeader("<U" & $ longest_str[i], n_rows)
                of PageType.PG_INT32_SIMPLE, PageType.PG_INT32_US, PageType.PG_INT32_EU: fh.writeNumpyHeader("<i8", n_rows)
                of PageType.PG_FLOAT32_SIMPLE, PageType.PG_FLOAT32_US, PageType.PG_FLOAT32_EU: fh.writeNumpyHeader("<f8", n_rows)
                of PageType.PG_BOOL: fh.writeNumpyHeader("|b1", n_rows)
                of PageType.PG_OBJECT, PageType.PG_DATE, PageType.PG_DATETIME, PageType.PG_DATE_SHORT:
                    dtype = PageType.PG_OBJECT
                    fh.writeNumpyHeader("|O", n_rows)
                    rank[].sortRanks(true) # this is an object type, put string backs to the end
                else: raise newException(Exception, "invalid")

            column_dtypes[i] = dtype

        for idx in 0..n_pages-1:
            var fh = page_file_handlers[idx]
            let dt = column_dtypes[idx]
            if dt == PageType.PG_OBJECT:
                fh.writePickleStart(binput, n_rows)

    return (page_file_handlers, column_dtypes, binput)

proc inferLocaleInt(rank: var Rank, str: var string): int {.inline.} =
    for r_addr in rank.iter():
        let dt = r_addr[0]
        try:
            case dt:
                of DataTypes.DT_INT_SIMPLE: return str.unsafeAddr.inferInt(true, false)
                of DataTypes.DT_INT_US: return str.unsafeAddr.inferInt(false, true)
                of DataTypes.DT_INT_EU: return str.unsafeAddr.inferInt(false, false)
                else: discard
        except ValueError:
            discard

    raise newException(ValueError, "Not an int")

proc inferLocaleFloat(rank: var Rank, str: var string): float {.inline.} =
    for r_addr in rank.iter():
        let dt = r_addr[0]
        try:
            case dt:
                of DataTypes.DT_FLOAT_SIMPLE: return str.unsafeAddr.inferFloat(true, false)
                of DataTypes.DT_FLOAT_US: return str.unsafeAddr.inferFloat(false, true)
                of DataTypes.DT_FLOAT_EU: return str.unsafeAddr.inferFloat(false, false)
                else: discard
        except ValueError:
            discard

    raise newException(ValueError, "Not a float")


proc dumpPageBody*(
        obj: var ReaderObj, fh: var BaseEncodedFile,
        guess_dtypes: bool, n_pages: int, row_count: int,
        import_fields: var seq[uint],
        page_file_handlers: var seq[File],
        longest_str: var seq[uint], ranks: var seq[Rank], column_dtypes: var seq[PageType],
        binput: var uint32
    ): void =
    for (row_idx, fields, field_count) in obj.parseCSV(fh):
        if row_count >= 0 and row_idx >= (uint row_count):
            break

        var fidx = -1

        for idx in 0..field_count-1:
            if not ((uint idx) in import_fields):
                continue

            inc fidx

            var str = fields[idx]
            var fh = page_file_handlers[fidx]

            if not guess_dtypes:
                fh.writeNumpyUnicode(str, longest_str[fidx])
            else:
                let dt = column_dtypes[fidx]
                var rank = ranks[fidx]

                case dt:
                    of PageType.PG_UNICODE: fh.writeNumpyUnicode(str, longest_str[fidx])
                    of PageType.PG_INT32_SIMPLE, PageType.PG_INT32_US, PageType.PG_INT32_EU:
                        fh.writeNumpyInt(inferLocaleInt(rank, str))
                    of PageType.PG_FLOAT32_SIMPLE, PageType.PG_FLOAT32_US, PageType.PG_FLOAT32_EU:
                        fh.writeNumpyFloat(inferLocaleFloat(rank, str))
                    of PageType.PG_BOOL: fh.writeNumpyBool(str)
                    of PageType.PG_OBJECT:
                        for r_addr in rank.iter():
                            let dt = r_addr[0]
                            try:
                                case dt:
                                    of DataTypes.DT_INT_SIMPLE:
                                        fh.writePicklePyObj(str.unsafeAddr.inferInt(true, false), binput)
                                    of DataTypes.DT_INT_US:
                                        fh.writePicklePyObj(str.unsafeAddr.inferInt(false, true), binput)
                                    of DataTypes.DT_INT_EU:
                                        fh.writePicklePyObj(str.unsafeAddr.inferInt(false, false), binput)
                                    of DataTypes.DT_FLOAT_SIMPLE:
                                        fh.writePicklePyObj(str.unsafeAddr.inferFloat(true, false), binput)
                                    of DataTypes.DT_FLOAT_US:
                                        fh.writePicklePyObj(str.unsafeAddr.inferFloat(false, true), binput)
                                    of DataTypes.DT_FLOAT_EU:
                                        fh.writePicklePyObj(str.unsafeAddr.inferFloat(false, false), binput)
                                    of DataTypes.DT_BOOL:
                                        fh.writePicklePyObj(str.unsafeAddr.inferBool(), binput)
                                    of DataTypes.DT_DATE:
                                        fh.writePicklePyObj(str.unsafeAddr.inferDate(false, false), binput)
                                    of DataTypes.DT_DATE_SHORT:
                                        fh.writePicklePyObj(str.unsafeAddr.inferDate(true, false), binput)
                                    of DataTypes.DT_DATE_US:
                                        fh.writePicklePyObj(str.unsafeAddr.inferDate(false, true), binput)
                                    of DataTypes.DT_TIME:
                                        fh.writePicklePyObj(str.unsafeAddr.inferTime(), binput)
                                    of DataTypes.DT_DATETIME:
                                        fh.writePicklePyObj(str.unsafeAddr.inferDatetime(false), binput)
                                    of DataTypes.DT_DATETIME_US:
                                        fh.writePicklePyObj(str.unsafeAddr.inferDatetime(true), binput)
                                    of DataTypes.DT_STRING:
                                        fh.writePicklePyObj(str, binput)
                                    of DataTypes.DT_NONE:
                                        fh.writePicklePyObj(str.unsafeAddr.inferNone, binput)
                                    of DataTypes.DT_MAX_ELEMENTS:
                                        raise newException(Exception, "not a type")
                            except ValueError:
                                continue
                            break
                    else: raise newException(Exception, "invalid: " & $dt)

        for idx in (fidx+1)..n_pages-1:
            var fh = page_file_handlers[idx]

            if not guess_dtypes:
                fh.writeNumpyUnicode(none_str, longest_str[idx])
            else:
                let dt = column_dtypes[idx]

                case dt:
                    of PageType.PG_UNICODE:
                        fh.writeNumpyUnicode(none_str, longest_str[idx])
                    else:
                        fh.writePicklePyObj(PY_None, binput)


proc dumpPageFooter*(
    n_pages: int, n_rows: uint,
    page_file_handlers: var seq[File],
    column_dtypes: var seq[PageType],
    binput: var uint32
): void =
    for idx in 0..n_pages-1:
        var fh = page_file_handlers[idx]
        let dt = column_dtypes[idx]
        if dt == PageType.PG_OBJECT:
            fh.writePickleFinish(binput, n_rows)
