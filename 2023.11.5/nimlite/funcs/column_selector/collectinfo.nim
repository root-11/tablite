import std/[tables, paths, sequtils]
from std/sugar import collect
from std/os import createDir
from std/strutils import toLower
import nimpy as nimpy
import infos
from ../../pytypes import KindObjectND
import ../../numpy
import ../../pymodules
from ../../utils import uniqueName
from ../../nimpyext import `!`

type CollectColumnSelectInfoResult = object
    columns*: Table[string, seq[string]]
    pageCount*: int
    isCorrectType*: Table[string, bool]
    desiredColumnMap*: OrderedTable[string, DesiredColumnInfo]
    originalPagesMap*: Table[string, seq[nimpy.PyObject]]
    passedColumnData*: seq[string]
    failedColumnData*: seq[string]
    resColsPass*: seq[ColInfo]
    resColsFail*: seq[ColInfo]
    columnNames*: seq[string]
    rejectReasonName*: string

proc toPyObj*(self: CollectColumnSelectInfoResult): nimpy.PyObject =
    let pyObj = modules().builtins.classes.DictClass!()

    pyObj["columns"] = self.columns
    pyObj["page_count"] = self.pageCount
    pyObj["is_correct_type"] = self.isCorrectType
    pyObj["desired_column_map"] = self.desiredColumnMap.toPyObj()
    pyObj["original_pages_map"] = self.originalPagesMap
    pyObj["passed_column_data"] = self.passedColumnData
    pyObj["failed_column_data"] = self.failedColumnData
    pyObj["res_cols_pass"] = self.resColsPass
    pyObj["res_cols_fail"] = self.resColsFail
    pyObj["column_names"] = self.columnNames
    pyObj["reject_reason_name"] = self.rejectReasonName

    return pyObj

proc fromPyObjToCollectColumnSelectInfos*(pyObj: nimpy.PyObject): CollectColumnSelectInfoResult =
    let columns = pyObj["columns"].to(Table[string, seq[string]])
    let pageCount = pyObj["page_count"].to(int)
    let isCorrectType = pyObj["is_correct_type"].to(Table[string, bool])
    let desiredColumnMap = pyObj["desired_column_map"].to(OrderedTable[string, DesiredColumnInfo])
    let originalPagesMap = pyObj["original_pages_map"].to(Table[string, seq[nimpy.PyObject]])
    let passedColumnData = pyObj["passed_column_data"].to(seq[string])
    let failedColumnData = pyObj["failed_column_data"].to(seq[string])
    let resColsPass = pyObj["res_cols_pass"].to(seq[ColInfo])
    let resColsFail = pyObj["res_cols_fail"].to(seq[ColInfo])
    let columnNames = pyObj["column_names"].to(seq[string])
    let rejectReasonName = pyObj["reject_reason_name"].to(string)

    return CollectColumnSelectInfoResult(
        columns: columns,
        pageCount: pageCount,
        isCorrectType: isCorrectType,
        desiredColumnMap: desiredColumnMap,
        originalPagesMap: originalPagesMap,
        passedColumnData: passedColumnData,
        failedColumnData: failedColumnData,
        resColsPass: resColsPass,
        resColsFail: resColsFail,
        columnNames: columnNames,
        rejectReasonName: rejectReasonName,
    )

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

proc collectColumnSelectInfo*(table: nimpy.PyObject, cols: nimpy.PyObject, dirPid: string, pbar: nimpy.PyObject): CollectColumnSelectInfoResult =
    var desiredColumnMap = initOrderedTable[string, DesiredColumnInfo]()
    var originalPagesMap = initTable[string, seq[nimpy.PyObject]]()
    var collisions = initTable[string, int]()

    let dirpage = Path(dirPid) / Path("pages")
    createDir(string dirpage)

    ######################################################
    # 1. Figure out what user needs (types, column names)
    ######################################################
    for c in cols: # now lets iterate over all given columns
        # this is our old name
        let nameInp = c["column"].to(string)
        var rename = c.get("rename", nil)

        if rename.isNone() or not modules().isinstance(rename, modules().builtins.classes.StrClass) or modules().getLen(rename) == 0:
            rename = nil
        else:
            let nameOutStripped = rename.strip()
            if modules().getLen(rename) > 0 and modules().getLen(nameOutStripped) == 0:
                raise newException(ValueError, "Validating 'column_select' failed, '" & nameInp & "' cannot be whitespace.")

            rename = nameOutStripped

        var nameOut = if rename.isNone(): nameInp else: rename.to(string)

        if nameOut in collisions: # check if the new name has any collision, add suffix if necessary
            collisions[nameOut] = collisions[nameOut] + 1
            nameOut = nameOut & "_" & $(collisions[nameOut] - 1)
        else:
            collisions[nameOut] = 1

        let desiredType = c.get("type", nil)

        desiredColumnMap[nameOut] = DesiredColumnInfo( # collect the information about the column, fill in any defaults
            originalName: nameInp,
            `type`: if desiredType.isNone(): K_NONETYPE else: toPageType(desiredType.to(string)),
            allowEmpty: c.get("allow_empty", false).to(bool)
        )

    discard pbar.update(3)
    discard pbar.display()

    ######################################################
    # 2. Converting types to user specified
    ######################################################
    # Registry of data
    var passedColumnData = newSeq[string]()
    var failedColumnData = newSeq[string]()
    var columns = initTable[string, seq[string]]()

    for pyColName in table.columns:
        let colName = pyColName.to(string)
        let pyColPages = table[colName].pages
        let pyPageCount = modules().getLen(pyColPages)
        var pagesPaths = newSeqOfCap[string](pyPageCount)
        var pagesObjs = newSeqOfCap[nimpy.PyObject](pyPageCount)

        for pyPage in pyColPages:
            pagesPaths.add(modules().toStr(pyPage.path.absolute()))
            pagesObjs.add(pyPage)

        failedColumnData.add(colName)

        columns[colName] = pagesPaths
        originalPagesMap[colName] = pagesObjs

    let columnNames = collect: (for k in columns.keys: k)
    var layoutSet = newSeq[(int, int)]()

    for pages in columns.values():
        let pgCount = pages.len
        let elCount = getColumnLen(pages)
        let layout = (elCount, pgCount)

        if layout in layoutSet:
            continue

        if layoutSet.len != 0:
            raise newException(RangeDefect, "Data layout mismatch, pages must be consistent")

        layoutSet.add(layout)

    let pageCount = layoutSet[0][1]

    var cols = initTable[string, seq[string]]()

    var resColsPass = newSeqOfCap[ColInfo](max(pageCount - 1, 0))
    var resColsFail = newSeqOfCap[ColInfo](max(pageCount - 1, 0))

    for _ in 0..<pageCount:
        resColsPass.add(initTable[string, ColSliceInfo]())
        resColsFail.add(initTable[string, ColSliceInfo]())

    var isCorrectType = initTable[string, bool]()

    proc genpage(dirpid: string): ColSliceInfo {.inline.} = (dir_pid, modules().tablite.modules.base.classes.SimplePageClass.next_id(dir_pid).to(string))

    discard pbar.update(5)
    discard pbar.display()

    let colStepSize = (40 / desiredColumnMap.len - 1)

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

        discard pbar.update(colStepSize)
        discard pbar.display()

    for desiredName in columns.keys:
        for i in 0..<pageCount:
            resColsFail[i][desiredName] = genpage(dir_pid)

    let rejectReasonName = uniqueName("reject_reason", columnNames)

    for i in 0..<pageCount:
        resColsFail[i][rejectReasonName] = genpage(dir_pid)

    failedColumnData.insert(rejectReasonName, 0)

    discard pbar.update(2)
    discard pbar.display()

    return CollectColumnSelectInfoResult(
        columns: columns,
        pageCount: pageCount,
        isCorrectType: isCorrectType,
        desiredColumnMap: desiredColumnMap,
        originalPagesMap: originalPagesMap,
        passedColumnData: passedColumnData,
        failedColumnData: failedColumnData,
        resColsPass: resColsPass,
        resColsFail: resColsFail,
        columnNames: columnNames,
        rejectReasonName: rejectReasonName
    )
