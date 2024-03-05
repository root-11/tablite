import nimpy as nimpy
import std/[sugar, sequtils, unicode, enumerate, tables]
import encfile, csvparse
import ../../[numpy, pickling, ranking, infertypes, pytypes, utils]

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

var noneStr = ""

proc collectPageInfo*(
        obj: var ReaderObj, fh: var BaseEncodedFile,
        guessDtypes: bool, nPages: int, rowCount: int,
        importFields: var seq[uint], skipEmpty: bool
    ): (uint, seq[uint], seq[Rank]) =
    var ranks: seq[Rank]
    var longestStr = collect(newSeqOfCap(nPages)):
        for _ in 0..<nPages:
            1u

    var nRows: uint = 0

    if guessDtypes:
        ranks = collect(newSeqOfCap(nPages)):
            for _ in 0..<nPages:
                newRank()
    else:
        ranks = newSeq[Rank](0)

    for (idxRow, fields, fieldCount) in obj.parseCSV(fh):
        if skipEmpty and fields.allFieldsEmpty(fieldCount):
                continue

        if rowCount >= 0 and idxRow >= (uint rowCount):
            break

        var fidx = -1

        for idx in 0..<fieldCount:
            if not ((uint idx) in importFields):
                continue

            inc fidx

            # let fidx = uint idx
            let field = fields[idx]

            if fidx < 0 or fidx >= nPages:
                raise newException(Exception, "what")

            if not guessDtypes:
                longestStr[fidx] = max(uint field.runeLen, longestStr[fidx])
            else:
                let rank = addr ranks[fidx]
                let dt = rank[].updateRank(field.addr)

                if dt == DataTypes.DT_STRING:
                    longestStr[fidx] = max(uint field.runeLen, longestStr[fidx])

        for idx in (fidx+1)..<nPages:
            # fill missing fields with nones
            longestStr[idx] = max(uint noneStr.len, longestStr[idx])

            if guessDtypes:
                discard ranks[idx].updateRank(addr noneStr)

        inc nRows

    return (nRows, longestStr, ranks)

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
        nPages: int, nRows: uint, guessDtypes: bool,
        longestStr: var seq[uint], ranks: var seq[Rank]
    ): (seq[File], seq[PageType], uint32) =
    let pageFileHandlers = collect(newSeqOfCap(nPages)):
        for p in destinations:
            open(p, fmWrite)

    var columnDtypes = newSeq[PageType](nPages)
    var binput: uint32 = 0

    if not guessDtypes:
        for idx, (fh, i) in enumerate(zip(pageFileHandlers, longestStr)):
            columnDtypes[idx] = PageType.PG_UNICODE
            fh.writeNumpyHeader(endiannessMark & "U" & $i, nRows)
    else:
        for i in 0..<nPages:
            let fh = pageFileHandlers[i]
            let rank = addr ranks[i]
            var dtype = columnDtypes[i]

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
                of PageType.PG_UNICODE: fh.writeNumpyHeader(endiannessMark & "U" & $ longestStr[i], nRows)
                of PageType.PG_INT32_SIMPLE, PageType.PG_INT32_US, PageType.PG_INT32_EU: fh.writeNumpyHeader(endiannessMark & "i8", nRows)
                of PageType.PG_FLOAT32_SIMPLE, PageType.PG_FLOAT32_US, PageType.PG_FLOAT32_EU: fh.writeNumpyHeader(endiannessMark & "f8", nRows)
                of PageType.PG_BOOL: fh.writeNumpyHeader("|b1", nRows)
                of PageType.PG_OBJECT, PageType.PG_DATE, PageType.PG_DATETIME, PageType.PG_DATE_SHORT:
                    dtype = PageType.PG_OBJECT
                    fh.writeNumpyHeader("|O", nRows)
                    rank[].sortRanks(true) # this is an object type, put string backs to the end
                else: raise newException(Exception, "invalid")

            columnDtypes[i] = dtype

        for idx in 0..<nPages:
            var fh = pageFileHandlers[idx]
            let dt = columnDtypes[idx]
            if dt == PageType.PG_OBJECT:
                fh.writePickleStart(binput, nRows)

    return (pageFileHandlers, columnDtypes, binput)

proc inferLocaleInt(rank: var Rank, str: var string): int {.inline.} =
    for ptrRank in rank.iter():
        let dt = ptrRank[0]
        try:
            case dt:
                of DataTypes.DT_INT_SIMPLE: return str.addr.inferInt(true, false)
                of DataTypes.DT_INT_US: return str.addr.inferInt(false, true)
                of DataTypes.DT_INT_EU: return str.addr.inferInt(false, false)
                else: discard
        except ValueError:
            discard

    raise newException(ValueError, "Not an int")

proc inferLocaleFloat(rank: var Rank, str: var string): float {.inline.} =
    for ptrRank in rank.iter():
        let dt = ptrRank[0]
        try:
            case dt:
                of DataTypes.DT_FLOAT_SIMPLE: return str.addr.inferFloat(true, false)
                of DataTypes.DT_FLOAT_US: return str.addr.inferFloat(false, true)
                of DataTypes.DT_FLOAT_EU: return str.addr.inferFloat(false, false)
                else: discard
        except ValueError:
            discard

    raise newException(ValueError, "Not a float")


proc dumpPageBody*(
        obj: var ReaderObj, fh: var BaseEncodedFile,
        guessDtypes: bool, nPages: int, rowCount: int, skipEmpty: bool,
        importFields: var seq[uint],
        pageFileHandlers: var seq[File],
        longestStr: var seq[uint], ranks: var seq[Rank], columnDtypes: var seq[PageType],
        binput: var uint32
    ): (seq[Table[KindObjectND, int]], seq[int]) =
    var bodyLens = newSeq[int](nPages)
    var typeCounts = collect:
        for _ in 0..<nPages:
            initTable[KindObjectND, int]()

    template addType(dtypes: ptr Table[KindObjectND, int], dt: KindObjectND, i: int): void =
        if not (dt in dtypes[]):
            dtypes[][dt] = 0

        dtypes[][dt] = dtypes[][dt] + 1
        bodyLens[i] = bodyLens[i] + 1

    for (idxRow, fields, fieldCount) in obj.parseCSV(fh):
        if skipEmpty and fields.allFieldsEmpty(fieldCount):
                continue

        if rowCount >= 0 and idxRow >= (uint rowCount):
            break

        var fidx = -1

        for idx in 0..<fieldCount:
            if not ((uint idx) in importFields):
                continue

            inc fidx

            var str = fields[idx]
            var fh = pageFileHandlers[fidx]
            var dtypes = addr typeCounts[fidx]

            if not guessDtypes:
                fh.writeNumpyUnicode(str, longestStr[fidx])
                dtypes.addType(K_STRING, fidx)
            else:
                let dt = columnDtypes[fidx]
                var rank = ranks[fidx]

                case dt:
                    of PageType.PG_UNICODE:
                        fh.writeNumpyUnicode(str, longestStr[fidx])
                        dtypes.addType(K_STRING, fidx)
                    of PageType.PG_INT32_SIMPLE, PageType.PG_INT32_US, PageType.PG_INT32_EU:
                        fh.writeNumpyInt(inferLocaleInt(rank, str))
                        dtypes.addType(K_INT, fidx)
                    of PageType.PG_FLOAT32_SIMPLE, PageType.PG_FLOAT32_US, PageType.PG_FLOAT32_EU:
                        fh.writeNumpyFloat(inferLocaleFloat(rank, str))
                        dtypes.addType(K_FLOAT, fidx)
                    of PageType.PG_BOOL:
                        fh.writeNumpyBool(str)
                        dtypes.addType(K_BOOLEAN, fidx)
                    of PageType.PG_OBJECT:
                        for ptrRank in rank.iter():
                            let dt = ptrRank[0]
                            try:
                                case dt:
                                    of DataTypes.DT_INT_SIMPLE:
                                        fh.writePicklePyObj(str.addr.inferInt(true, false), binput)
                                        dtypes.addType(K_INT, fidx)
                                    of DataTypes.DT_INT_US:
                                        fh.writePicklePyObj(str.addr.inferInt(false, true), binput)
                                        dtypes.addType(K_INT, fidx)
                                    of DataTypes.DT_INT_EU:
                                        fh.writePicklePyObj(str.addr.inferInt(false, false), binput)
                                        dtypes.addType(K_INT, fidx)
                                    of DataTypes.DT_FLOAT_SIMPLE:
                                        fh.writePicklePyObj(str.addr.inferFloat(true, false), binput)
                                        dtypes.addType(K_FLOAT, fidx)
                                    of DataTypes.DT_FLOAT_US:
                                        fh.writePicklePyObj(str.addr.inferFloat(false, true), binput)
                                        dtypes.addType(K_FLOAT, fidx)
                                    of DataTypes.DT_FLOAT_EU:
                                        fh.writePicklePyObj(str.addr.inferFloat(false, false), binput)
                                        dtypes.addType(K_FLOAT, fidx)
                                    of DataTypes.DT_BOOL:
                                        fh.writePicklePyObj(str.addr.inferBool(), binput)
                                        dtypes.addType(K_BOOLEAN, fidx)
                                    of DataTypes.DT_DATE:
                                        fh.writePicklePyObj(str.addr.inferDate(false, false), binput)
                                        dtypes.addType(K_DATE, fidx)
                                    of DataTypes.DT_DATE_SHORT:
                                        fh.writePicklePyObj(str.addr.inferDate(true, false), binput)
                                        dtypes.addType(K_DATE, fidx)
                                    of DataTypes.DT_DATE_US:
                                        fh.writePicklePyObj(str.addr.inferDate(false, true), binput)
                                        dtypes.addType(K_DATE, fidx)
                                    of DataTypes.DT_TIME:
                                        fh.writePicklePyObj(str.addr.inferTime(), binput)
                                        dtypes.addType(K_TIME, fidx)
                                    of DataTypes.DT_DATETIME:
                                        fh.writePicklePyObj(str.addr.inferDatetime(false), binput)
                                        dtypes.addType(K_DATETIME, fidx)
                                    of DataTypes.DT_DATETIME_US:
                                        fh.writePicklePyObj(str.addr.inferDatetime(true), binput)
                                        dtypes.addType(K_DATETIME, fidx)
                                    of DataTypes.DT_STRING:
                                        fh.writePicklePyObj(str, binput)
                                        dtypes.addType(K_STRING, fidx)
                                    of DataTypes.DT_NONE:
                                        fh.writePicklePyObj(str.addr.inferNone, binput)
                                        dtypes.addType(K_NONETYPE, fidx)
                                    of DataTypes.DT_MAX_ELEMENTS:
                                        raise newException(Exception, "not a type")
                            except ValueError:
                                continue
                            break
                    else: raise newException(Exception, "invalid: " & $dt)

        for idx in (fidx+1)..<nPages:
            var fh = pageFileHandlers[idx]
            var dtypes = addr typeCounts[idx]

            if not guessDtypes:
                fh.writeNumpyUnicode(noneStr, longestStr[idx])
                dtypes.addType(K_STRING, idx)
            else:
                let dt = columnDtypes[idx]

                case dt:
                    of PageType.PG_UNICODE:
                        fh.writeNumpyUnicode(noneStr, longestStr[idx])
                        dtypes.addType(K_STRING, idx)
                    else:
                        fh.writePicklePyObj(PY_None, binput)
                        dtypes.addType(K_NONETYPE, idx)

    return (typeCounts, bodyLens)


proc dumpPageFooter*(
    nPages: int, nRows: uint,
    pageFileHandlers: var seq[File],
    columnDtypes: var seq[PageType],
    binput: var uint32
): void =
    for idx in 0..<nPages:
        var fh = pageFileHandlers[idx]
        let dt = columnDtypes[idx]
        if dt == PageType.PG_OBJECT:
            fh.writePickleFinish(binput, nRows)