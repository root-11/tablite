import std/[tables, paths, sequtils, times]
from std/sugar import collect
from std/os import createDir
from std/strutils import toLower
import nimpy as nimpy
import infos
from ../../pytypes import KindObjectND
import ../../numpy
import ../../pymodules
from ../../utils import uniqueName

proc toPageType(name: string): KindObjectND =
    case name.toLower():
    of "int": return KindObjectND.K_INT
    of "float": return KindObjectND.K_FLOAT
    of "bool": return KindObjectND.K_BOOLEAN
    of "str": return KindObjectND.K_STRING
    of "date": return KindObjectND.K_DATE
    of "time": return KindObjectND.K_TIME
    of "datetime": return KindObjectND.K_DATETIME
    else: raise newException(FieldDefect, "unsupported page type: '" & name & "'")

proc collectColumnSelectInfo*(table: nimpy.PyObject, cols: nimpy.PyObject, dirPid: string): (
    Table[string, seq[string]], int, Table[string, bool], OrderedTable[string, DesiredColumnInfo], seq[string], seq[string], seq[ColInfo], seq[ColInfo], seq[string], string
) =
    let startTime = cpuTime()

    var desiredColumnMap = initOrderedTable[string, DesiredColumnInfo]()
    var collisions = initTable[string, int]()

    let dirpage = Path(dirPid) / Path("pages")
    createDir(string dirpage)

    let phase1Time = cpuTime()

    ######################################################
    # 1. Figure out what user needs (types, column names)
    ######################################################
    for c in cols: # now lets iterate over all given columns
        # this is our old name
        let nameInp = c["column"].to(string)
        var rename = c.get("rename", builtins().None)

        if rename.isNone() or not builtins().isinstance(rename, builtins().str).to(bool) or builtins().len(rename).to(int) == 0:
            rename = builtins().None
        else:
            let nameOutStripped = rename.strip()
            if builtins().len(rename).to(int) > 0 and builtins().len(nameOutStripped).to(int) == 0:
                raise newException(ValueError, "Validating 'column_select' failed, '" & nameInp & "' cannot be whitespace.")

            rename = nameOutStripped

        var nameOut = if rename.isNone(): nameInp else: rename.to(string)

        if nameOut in collisions: # check if the new name has any collision, add suffix if necessary
            collisions[nameOut] = collisions[nameOut] + 1
            nameOut = nameOut & "_" & $(collisions[nameOut] - 1)
        else:
            collisions[nameOut] = 1

        let desiredType = c.get("type", builtins().None)

        desiredColumnMap[nameOut] = DesiredColumnInfo( # collect the information about the column, fill in any defaults
            originalName: nameInp,
            `type`: if desiredType.isNone(): K_NONETYPE else: toPageType(desiredType.to(string)),
            allowEmpty: c.get("allow_empty", builtins().False).to(bool)
        )

    let phase2Time = cpuTime()

    ######################################################
    # 2. Converting types to user specified
    ######################################################
    let columns = collect(initTable()):
        for pyColName in table.columns:
            let colName = pyColName.to(string)
            let pyColPages = table[colName].pages
            let pages = collect:
                for pyPage in pyColPages:
                    builtins().str(pyPage.path.absolute()).to(string)

            {colName: pages}

    let phase3Time = cpuTime()

    let columnNames = collect: (for k in columns.keys: k)
    var layout_set = newSeq[(int, int)]()

    for pages in columns.values():
        let pgCount = pages.len
        let elCount = getColumnLen(pages)
        let layout = (elCount, pgCount)

        if layout in layout_set:
            continue

        if layout_set.len != 0:
            raise newException(RangeDefect, "Data layout mismatch, pages must be consistent")

        layout_set.add(layout)

    let phase4Time = cpuTime()

    let pageCount = layout_set[0][1]

    # Registry of data
    var passedColumnData = newSeq[string]()
    var failedColumnData = newSeq[string]()

    var cols = initTable[string, seq[string]]()

    var resColsPass = newSeqOfCap[ColInfo](pageCount-1)
    var resColsFail = newSeqOfCap[ColInfo](pageCount-1)

    for _ in 0..<pageCount:
        resColsPass.add(initTable[string, ColSliceInfo]())
        resColsFail.add(initTable[string, ColSliceInfo]())

    var isCorrectType = initTable[string, bool]()

    let phase5Time = cpuTime()

    proc genpage(dirpid: string): ColSliceInfo {.inline.} = (dir_pid, tabliteBase().SimplePage.next_id(dir_pid).to(int))

    for (desiredNameNonUnique, colDesired) in desiredColumnMap.pairs():
        let keys = toSeq(passedColumnData)
        let desiredName = uniqueName(desiredNameNonUnique, keys)
        let thisCol = columns[colDesired.originalName]

        cols[desiredName] = thisCol

        passedColumnData.add(desiredName)

        var colDtypes = toSeq(thisCol.getColumnTypes().keys)
        var needsToIterate = false

        if K_NONETYPE in colDtypes:
            if not colDesired.allowEmpty:
                needsToIterate = true
            else:
                if colDesired.`type` == K_STRING:
                    # none strings are cast to empty string, therefore we need to iterate for consistency
                    needsToIterate = true
                else:
                    colDtypes.delete(colDtypes.find(K_NONETYPE))

        if not needsToIterate and colDtypes.len > 0:
            if colDtypes.len > 1:
                # multiple times always needs to cast
                needsToIterate = true
            else:
                let activeType = colDtypes[0]

                if activeType != colDesired.`type`:
                    # not same type, need to cast
                    needsToIterate = true
                elif activeType == K_STRING and not colDesired.allowEmpty:
                    # we may have same type but we still need to filter empty strings
                    needsToIterate = true

        isCorrectType[desiredName] = not needsToIterate

        for i in 0..<pageCount:
            resColsPass[i][desiredName] = genpage(dir_pid)

    let phase6Time = cpuTime()

    for desiredName in columns.keys:
        failedColumnData.add(desiredName)

        for i in 0..<pageCount:
            resColsFail[i][desiredName] = genpage(dir_pid)

    let rejectReasonName = uniqueName("reject_reason", columnNames)

    for i in 0..<pageCount:
        resColsFail[i][rejectReasonName] = genpage(dir_pid)

    failedColumnData.add(rejectReasonName)

    let finalTime = cpuTime()

    echo ">>>totalTime: " & $(finalTime - startTime)
    echo ">>>init: " & $(phase1Time - startTime)
    echo ">>>phase1: " & $(phase2Time - phase1Time)
    echo ">>>phase2: " & $(phase3Time - phase2Time)
    echo ">>>phase3: " & $(phase4Time - phase3Time)
    echo ">>>phase4: " & $(phase5Time - phase4Time)
    echo ">>>phase5: " & $(phase6Time - phase5Time)
    echo ">>>phase6: " & $(finalTime - phase6Time)

    return (columns, pageCount, isCorrectType, desiredColumnMap, passedColumnData, failedColumnData, resColsPass, resColsFail, columnNames, rejectReasonName)
