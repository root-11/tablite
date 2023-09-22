import std/[sugar, sequtils, unicode, enumerate]
import numpy, pickling, ranking, infertypes, encfile, csvparse

type PageType = enum
    PG_UNSET,
    PG_UNICODE,
    PG_INT32,
    PG_FLOAT32,
    PG_BOOL,
    PG_OBJECT,
    PG_DATE,
    PG_DATETIME
    PG_DATE_SHORT

proc collectPageInfo*(
        obj: ptr ReaderObj, fh: ptr BaseEncodedFile,
        guess_dtypes: bool, n_pages: int, row_count: int,
        import_fields: ptr seq[uint]
    ): (uint, seq[uint], seq[Rank]) =
    var ranks {.noinit.}: seq[Rank]
    var longest_str = newSeq[uint](n_pages)
    var n_rows: uint = 0
    
    if guess_dtypes:
        ranks = collect(newSeqOfCap(n_pages)):
            for _ in 0..n_pages-1:
                newRank()

    for (row_idx, fields, field_count) in obj[].parseCSV(fh[]):
        if row_count >= 0 and row_idx >= (uint row_count):
            break
            
        var fidx = -1

        for idx in 0..field_count-1:
            if not ((uint idx) in import_fields[]):
                continue

            inc fidx

            # let fidx = uint idx
            let field = fields[idx]

            if not guess_dtypes:
                longest_str[fidx] = max(uint field.runeLen, longest_str[fidx])
            else:
                let rank = addr ranks[fidx]
                let dt = rank[].updateRank(field.unsafeAddr)

                if dt == DataTypes.DT_STRING:
                    longest_str[fidx] = max(uint field.runeLen, longest_str[fidx])

        inc n_rows

    return (n_rows, longest_str, ranks)

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
            # echo $rank[]

            for it in rank[].iter():
                let dt = it[0]
                let count = it[1]

                if count == 0:
                    break

                if dtype == PageType.PG_UNSET:
                    case dt:
                        of DataTypes.DT_INT: dtype = PageType.PG_INT32
                        of DataTypes.DT_FLOAT: dtype = PageType.PG_FLOAT32
                        of DataTypes.DT_STRING:
                            dtype = PageType.PG_UNICODE
                            break   # if the first type is string, everying is a subset of string
                        of DataTypes.DT_DATETIME, DataTypes.DT_DATETIME_US:
                            dtype = PageType.PG_DATETIME
                        of DataTypes.DT_DATE, DataTypes.DT_DATE_US:
                            dtype = PageType.PG_DATE
                        of DataTypes.DT_DATE_SHORT:
                            dtype = PageType.PG_DATE_SHORT
                        else: dtype = PageType.PG_OBJECT
                    continue

                # check overlapping types
                if dtype == PageType.PG_FLOAT32 and dt in [DataTypes.DT_INT, DataTypes.DT_DATE_SHORT]: discard                         # float overlaps ints
                elif dtype == PageType.PG_INT32 and dt == DataTypes.DT_FLOAT: dtype = PageType.PG_FLOAT32   # int is a subset of int, change to float
                elif dtype == PageType.PG_INT32 and dt == DataTypes.DT_DATE_SHORT: dtype = PageType.PG_INT32   # int is a subset of int, change to float
                elif dtype == PageType.PG_DATE_SHORT:
                    if dt == DataTypes.DT_FLOAT: dtype = PageType.PG_FLOAT32
                    elif dt == DataTypes.DT_INT: dtype = PageType.PG_INT32
                else: dtype = PageType.PG_OBJECT                                                            # types cannot overlap

            case dtype:
                of PageType.PG_UNICODE: fh.writeNumpyHeader("<U" & $ longest_str[i], n_rows)
                of PageType.PG_INT32: fh.writeNumpyHeader("<i8", n_rows)
                of PageType.PG_FLOAT32: fh.writeNumpyHeader("<f8", n_rows)
                of PageType.PG_BOOL: fh.writeNumpyHeader("|b1", n_rows)
                of PageType.PG_OBJECT, PageType.PG_DATE, PageType.PG_DATETIME, PageType.PG_DATE_SHORT:
                    dtype = PageType.PG_OBJECT
                    fh.writeNumpyHeader("|O", n_rows)
                    rank[].sortRanks(true) # this is an object type, put string backs to the end
                else: raise newException(Exception, "invalid")

            column_dtypes[i] = dtype

        for idx in 0..n_pages-1:
            let fh = page_file_handlers[idx].unsafeAddr
            let dt = column_dtypes[idx]
            if dt == PageType.PG_OBJECT:
                fh.writePickleStart(binput, n_rows)

    return (page_file_handlers, column_dtypes, binput)

proc dumpPageBody*(
        obj: ptr ReaderObj, fh: ptr BaseEncodedFile,
        guess_dtypes: bool, n_pages: int, row_count: int,
        import_fields: ptr seq[uint],
        page_file_handlers: var seq[File],
        longest_str: var seq[uint], ranks: var seq[Rank], column_dtypes: var seq[PageType],
        binput: var uint32
    ): void =
    for (row_idx, fields, field_count) in obj[].parseCSV(fh[]):
        if row_count >= 0 and row_idx >= (uint row_count):
            break

        var fidx = -1

        for idx in 0..field_count-1:
            if not ((uint idx) in import_fields[]):
                continue

            inc fidx

            var str = fields[idx]
            # let fidx = uint idx
            var fh = page_file_handlers[fidx].unsafeAddr

            if not guess_dtypes:
                fh.writeNumpyUnicode(str, longest_str[fidx])
            else:
                let dt = column_dtypes[fidx]

                case dt:
                    of PageType.PG_UNICODE: fh.writeNumpyUnicode(str, longest_str[fidx])
                    of PageType.PG_INT32: fh.writeNumpyInt(str)
                    of PageType.PG_FLOAT32: fh.writeNumpyFloat(str)
                    of PageType.PG_BOOL: fh.writeNumpyBool(str)
                    of PageType.PG_OBJECT: 
                        var rank = ranks[idx]
                        for r_addr in rank.iter():
                            let dt = r_addr[0]
                            try:
                                case dt:
                                    of DataTypes.DT_INT:
                                        fh.writePicklePyObj(str.unsafeAddr.inferInt(), binput)
                                    of DataTypes.DT_FLOAT:
                                        fh.writePicklePyObj(str.unsafeAddr.inferFloat(), binput)
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
                            except ValueError as e:
                                continue
                            break
                    else: raise newException(Exception, "invalid: " & $dt)

proc dumpPageFooter*(
    n_pages: int, n_rows: uint,
    page_file_handlers: var seq[File],
    column_dtypes: var seq[PageType],
    binput: var uint32
): void =
    for idx in 0..n_pages-1:
        let fh = page_file_handlers[idx].unsafeAddr
        let dt = column_dtypes[idx]
        if dt == PageType.PG_OBJECT:
            fh.writePickleFinish(binput, n_rows)