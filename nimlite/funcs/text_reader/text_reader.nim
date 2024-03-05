import nimpy as nimpy
import std/[os, enumerate, sugar, tables, json, options, strutils, paths]
import encfile, csvparse, table, ../../utils, paging, taskargs
from ../../numpy import newPyPage
from ../../ranking import Rank

proc collectPageInfoTask*(task: TaskArgs): (uint, seq[uint], seq[Rank]) =
    var dialect = task.dialect
    var encoding = task.encoding
    var destinations = task.destinations
    var path = task.path
    var guessDtypes = task.guessDtypes
    var rowCount = task.rowCount
    var rowOffset = task.rowOffset
    var importFields = task.importFields
    var skipEmpty = task.skipEmpty
    var obj = newReaderObj(dialect)

    var fh = newFile(path, encoding)
    let nPages = destinations.len

    try:
        fh.setFilePos(int64 rowOffset, fspSet)

        var (nRows, longestStr, ranks) = collectPageInfo(
            obj = obj,
            fh = fh,
            guessDtypes = guessDtypes,
            nPages = nPages,
            rowCount = rowCount,
            importFields = importFields,
            skipEmpty = skipEmpty
        )

        return (nRows, longestStr, ranks)

    finally:
        fh.close()


type PageInfo* = (uint, seq[uint], seq[Rank])

proc textReaderTask*(task: TaskArgs, page_info: PageInfo): seq[nimpy.PyObject] =
    var dialect = task.dialect
    var encoding = task.encoding
    var destinations = task.destinations
    var path = task.path
    var guessDtypes = task.guessDtypes
    var rowCount = task.rowCount
    var rowOffset = task.rowOffset
    var importFields = task.importFields
    var skipEmpty = task.skipEmpty
    var obj = newReaderObj(dialect)

    var fh = newFile(path, encoding)
    let nPages = destinations.len

    try:
        fh.setFilePos(int64 rowOffset, fspSet)

        var (nRows, longestStr, ranks) = pageInfo
        var (pageFileHandlers, columnDtypes, binput) = dumpPageHeader(
            destinations = destinations,
            nPages = nPages,
            nRows = nRows,
            guessDtypes = guessDtypes,
            longestStr = longestStr,
            ranks = ranks,
        )

        try:
            fh.setFilePos(int64 rowOffset, fspSet)

            let (pgTypes, pgLens) = dumpPageBody(
                obj = obj,
                fh = fh,
                guessDtypes = guessDtypes,
                nPages = nPages,
                rowCount = rowCount,
                skipEmpty = skipEmpty,
                importFields = importFields,
                pageFileHandlers = pageFileHandlers,
                longestStr = longestStr,
                ranks = ranks,
                columnDtypes = columnDtypes,
                binput = binput
            )

            dumpPageFooter(
                nPages = nPages,
                nRows = nRows,
                pageFileHandlers = pageFileHandlers,
                columnDtypes = columnDtypes,
                binput = binput
            )

            let elems = collect:
                for i in 0..<nPages:
                    let path = Path(destinations[i])
                    let workdir = string path.parentDir.parentDir
                    let id = string path.extractFilename.changeFileExt("")
                    let dtypes = pgTypes[i]
                    let len = pgLens[i]

                    newPyPage(id, workdir, len, dtypes)

            return elems
        finally:
            for f in pageFileHandlers:
                f.close()

    finally:
        fh.close()

proc getHeaders*(path: string, encoding: FileEncoding, dia: Dialect, skipEmpty: SkipEmpty, headerRowIndex: uint, lineCount: int): seq[seq[string]] =
    let fh = newFile(path, encoding)
    var obj = newReaderObj(dia)

    try:
        var totalLines: int = 0
        var linesToSkip = headerRowIndex
        var headers = newSeqOfCap[seq[string]](lineCount)

        for (idxRow, fields, fieldCount) in obj.parseCSV(fh):
            if skipEmpty.checkSkipEmpty(fields, fieldCount):
                continue

            if linesToSkip > 0:
                dec linesToSkip
                continue
            
            inc totalLines

            var row = newSeq[string](fieldCount)
            for i in 0..<fieldCount:
                row[i] = fields[i]

            headers.add(row)

            if totalLines >= lineCount: # we want one extra, because the first one is most likely labels
                break

        return headers
    finally:
        fh.close()

proc importTextFile*(
    pid: string, path: string, encoding: FileEncoding, dia: Dialect,
    columns: Option[seq[string]], firstRowHasHeaders: bool, headerRowIndex: uint,
    pageSize: uint, guessDtypes: bool, skipEmpty: SkipEmpty,
    start: Option[int] = none[int](), limit: Option[int] = none[int]()
): TabliteTable =

    echo "Collecting tasks: '" & path & "'"

    let optStart = (if start.isSome: start.get else: 0)
    let optLimit = (if limit.isSome: limit.get else: -1)
    let (newlineOffsets, newlines) = findNewlines(path, encoding, dia)
    let dirname = pid & "/pages"

    if not dirExists(dirname):
        createDir(dirname)

    if newlines > 0 and newlines > headerRowIndex:
        let firstLine = readColumns(path, encoding, dia, newlineOffsets[headerRowIndex], skipEmpty)

        var fields = newSeq[string](0)

        if firstRowHasHeaders:
            for n in firstLine:
                #[
                    deal with duplicate headers,
                    we can make them unique immediatly,
                    because even if user uses selects which columns to import we wouldn't know which to choose and default to first one
                    meanwhile this will deal with duplicates when all columns are improted
                ]#
                fields.add(uniqueName(n, fields))
        else:
            fields = collect(newSeqOfCap(firstLine.len)):
                for i in 0..<firstLine.len:
                    $i

        var impColumns = newSeq[string](0)

        if columns.isSome:
            var missing = newSeq[string]()
            for column in columns.get:
                if not (column in fields):
                    missing.add("'" & column & "'")
            if missing.len > 0:
                let field_list = collect(newSeqOfCap(fields.len)):
                    for f in fields:
                        "'" & f & "'"
                raise newException(IOError, "Missing columns: [" & missing.join(", ") & "]" & " | Available columns: (" & $field_list.len & ")[" & field_list.join(", ") & "]")
            impColumns = columns.get
        else:
            impColumns = fields

        var fieldRelation = collect(initOrderedTable()):
            for ix, name in enumerate(fields):
                if name in impColumns:
                    {uint ix: name}

        echo fieldRelation

        let importFields = collect: (for k in fieldRelation.keys: k)
        let importFieldNames = collect: (for v in fieldRelation.values: v)

        var fieldRelationInv = collect(initOrderedTable()):
            for (ix, name) in fieldRelation.pairs:
                {name: ix}

        var pageList = collect(initOrderedTable()):
            for (ix, name) in fieldRelation.pairs:
                {ix: newSeq[string]()}

        var nameList = newSeq[string]()
        var tableColumns = collect(initOrderedTable()):
            for name in impColumns:
                let unq = uniqueName(name, nameList)

                nameList.add(unq)

                {unq: fieldRelationInv[name]}

        let offsetRow = (if firstRowHasHeaders: 1 else: 0) + int headerRowIndex

        var pageIdx: uint32 = 1
        var rowIdx: uint = uint optStart + offsetRow
        var taskList = newSeq[TabliteTask]()
        let maxLine = (if optLimit >= 0: min(newlines, uint (optLimit + optStart) + offsetRow) else: newlines)

        echo "Dumping tasks: '" & path & "'"

        while rowIdx < maxLine:
            let pageCount = fieldRelation.len
            let nextLine = min(rowIdx + pageSize, maxLine)
            let rowCount = nextLine - rowIdx
            var pages = newSeq[string](pageCount)

            for idx in 0..<pageCount:
                var pagepath = dirname & "/" & $pageIdx & ".npy"

                if not pid.endsWith("tablite/nim"):
                    while fileExists(pagepath):
                        inc pageIdx
                        pagepath = dirname & "/" & $pageIdx & ".npy"

                let fieldIdx = importFields[idx]

                pageList[fieldIdx].add(pagepath)
                pages[idx] = pagepath

                inc pageIdx

            taskList.add(newTabliteTask(pages, newlineOffsets[rowIdx], rowCount))

            rowIdx = nextLine

        let tasks = newTabliteTasks(
            path = path,
            encoding = $encoding,
            dialect = dia,
            tasks = taskList,
            importFields = importFields,
            importFieldNames = importFieldNames,
            pageSize = pageSize,
            guessDtypes = guessDtypes,
            skipEmpty = $skipEmpty
        )
        let columns = collect(newSeqOfCap(tableColumns.len)):
            for (column_name, page_index) in tableColumns.pairs:
                newTabliteColumn(column_name, pageList[page_index])

        let table = newTabliteTable(tasks, columns)

        return table
    else:
        raise newException(IOError, "end of file")