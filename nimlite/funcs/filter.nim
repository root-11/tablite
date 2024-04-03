import nimpy
import std/[os, sugar, enumerate, strutils, tables, sequtils, options, times, macros, paths]
import ../[pymodules, utils, pytypes, numpy, dateutils, nimpyext]

const FILTER_KEYS = ["column1", "column2", "criteria", "value1", "value2"]
const FILTER_OPS = [">", ">=", "==", "<", "<=", "!=", "in"]
const FILTER_TYPES = ["all", "any"]

type FilterMethods = enum
    FM_GT
    FM_GE
    FM_EQ
    FM_LT
    FM_LE
    FM_NE
    FM_IN

type FilterType = enum
    FT_ALL
    FT_ANY

type ExpressionValue = (Option[string], Option[PY_ObjectND])
type Expression = (ExpressionValue, FilterMethods, ExpressionValue)


iterator pageZipper[T](iters: Table[string, seq[T]]): seq[T] =
    var allIters = newSeq[iterator(): T]()

    proc makeIterable(iterable: seq[T]): auto =
        return iterator(): auto =
            for v in iterable:
                yield v

    var res: seq[T] = @[]
    var finished = false

    for it in iters.values:
        let i = makeIterable(it)

        allIters.add(i)

        res.add(i())
        finished = finished or finished(i)

    while not finished:
        yield res

        res = newSeqOfCap[T](allIters.len)

        for i in allIters:
            res.add(i())
            finished = finished or finished(i)

iterator iterateRows(exprColumns: seq[string], tablePages: Table[string, seq[string]]): seq[PY_ObjectND] =
    var allIters = newSeq[iterator(): PY_ObjectND]()
    var res: seq[PY_ObjectND] = @[]
    var finished = false

    proc makeIterable(column: seq[string]): auto =
        return iterator(): auto =
            for v in iterateColumn[PY_ObjectND](column):
                yield v

    for column in exprColumns:
        let i = makeIterable(tablePages[column])

        allIters.add(i)

        res.add(i())
        finished = finished or finished(i)

    while not finished:
        yield res

        res = newSeqOfCap[PY_ObjectND](allIters.len)

        for i in allIters:
            res.add(i())
            finished = finished or finished(i)

iterator iteratePages(paths: seq[string]): seq[PY_ObjectND] =
    let pages = collect: (for p in paths: readNumpy(p))

    var allIters = newSeq[iterator(): PY_ObjectND]()
    var res: seq[PY_ObjectND] = @[]
    var finished = false

    proc makeIterable(page: BaseNDArray): auto =
        return iterator(): auto =
            for v in page.iterateObjectPage:
                yield v

    for pg in pages:
        let i = makeIterable(pg)

        allIters.add(i)

        res.add(i())
        finished = finished or finished(i)

    while not finished:
        yield res

        res = newSeqOfCap[PY_ObjectND](allIters.len)

        for i in allIters:
            res.add(i())
            finished = finished or finished(i)

proc extractValue(row: seq[PY_ObjectND], exprCols: seq[string], value: ExpressionValue): PY_ObjectND {.inline.} =
    let (col, val) = value

    if val.isSome:
        return val.get

    let idx = exprCols.find(col.get)

    return row[idx]

proc checkExpression(row: seq[PY_ObjectND], exprCols: seq[string], xpr: Expression): bool {.inline.} =
    let (leftXpr, criteria, rightXpr) = xpr
    let left = row.extractValue(exprCols, leftXpr)
    let right = row.extractValue(exprCols, rightXpr)
    let expressed = (
        case criteria:
        of FM_EQ: left == right
        of FM_NE: left != right
        of FM_GT: left > right
        of FM_GE: left >= right
        of FM_LT: left < right
        of FM_LE: left <= right
        of FM_IN: left in right
    )

    # echo left, " ", criteria, " ", right, " ? ", expressed

    return expressed

proc checkExpressions(row: seq[PY_ObjectND], exprCols: seq[string], expressions: seq[Expression], filterType: FilterType): bool {.inline.} =
    case filterType:
    of FT_ANY: any(expressions, xpr => row.checkExpression(exprCols, xpr))
    of FT_ALL: all(expressions, xpr => row.checkExpression(exprCols, xpr))

proc filter(table: nimpy.PyObject, pyExpressions: seq[nimpy.PyObject], filterTypeName: string): (nimpy.PyObject, nimpy.PyObject) =
    let m = modules()
    let builtins = m.builtins
    let tablite = m.tablite
    let base = tablite.modules.base
    let Config = tablite.modules.config.classes.Config
    let TableClass = builtins.getType(table)

    let filterType = (
        case filterTypeName.toLower():
        of "any": FT_ANY
        of "all": FT_ALL
        else: raise newException(ValueError, "invalid filter type '" & filterTypeName & "' expected " & $FILTER_TYPES)
    )

    if pyExpressions.len == 0:
        return (table, TableClass!())

    let columns = collect: (for c in table.columns.keys(): c.to(string))

    if columns.len == 0:
        return (table, TableClass!())

    var exprCols = newSeq[string]()

    var tablePages = initTable[string, seq[string]]()
    var passTablePages = initOrderedTable[string, nimpy.PyObject]()
    var failTablePages = initOrderedTable[string, nimpy.PyObject]()

    for c in columns:
        let pyCol = table[c]
        tablePages[c] = base.collectPages(pyCol)
        passTablePages[c] = base.classes.ColumnClass!(pyCol.path)
        failTablePages[c] = base.classes.ColumnClass!(pyCol.path)

    template addParam(columnName: string, valueName: string, paramName: string): auto =
        var res {.noInit.}: ExpressionValue

        if columnName.contains(expression):
            if valueName.contains(expression):
                raise newException(ValueError, "filter can only take 1 " & paramName & " expr element, got 2")

            let c = expression[columnName].to(string)

            if c notin columns:
                raise newException(ValueError, "no such column '" & $c & "'in " & $columns)

            if c notin exprCols:
                exprCols.add(c)

            res = (some(c), none[PY_ObjectND]())
        elif not valueName.contains(expression):
            raise newException(ValueError, "no " & paramName & " parameter")
        else:
            let pyVal = expression[valueName]
            let pyType = builtins.getTypeName(pyVal)
            let obj: PY_ObjectND = (
                case pyType
                of "int": newPY_Object(pyVal.to(int))
                of "float": newPY_Object(pyVal.to(float))
                of "bool": newPY_Object(pyVal.to(bool))
                of "str": newPY_Object(pyVal.to(string))
                of "datetime": newPY_Object(pyDateTime2NimDateTime(pyVal), K_DATETIME)
                of "date": newPY_Object(pyDate2NimDateTime(pyVal), K_DATE)
                of "time": newPY_Object(pyTime2NimDuration(pyVal))
                else: implement("invalid object type: " & pyType)
            )
            res = (none[string](), some(obj))

        res

    var expressions = newSeq[Expression]()

    for expression in pyExpressions:
        if not builtins.isinstance(expression, builtins.classes.DictClass):
            raise newException(KeyError, "expression must be a dict: " & $expression)

        if not builtins.getLen(expression) == 3:
            raise newException(ValueError, "expression must be of len 3: " & $builtins.getLen(expression))

        let invalidKeys = collect:
            for pyKey in expression.keys():
                let key = pyKey.to(string)

                if key in FILTER_KEYS:
                    continue

                key

        if invalidKeys.len > 0:
            raise newException(ValueError, "got unknown keys " & $invalidKeys & " expected " & $FILTER_KEYS)

        let criteria = expression["criteria"].to(string)

        let crit: FilterMethods = (
            case criteria
            of ">": FM_GT
            of ">=": FM_GE
            of "==": FM_EQ
            of "<": FM_LT
            of "<=": FM_LE
            of "!=": FM_NE
            of "in": FM_IN
            else: raise newException(ValueError, "invalid criteria '" & criteria & "' expected " & $FILTER_OPS)
        )

        let lOpt = addParam("column1", "value1", "left")
        let rOpt = addParam("column2", "value2", "right")

        expressions.add((lOpt, crit, rOpt))

    let pageSize = Config.PAGE_SIZE.to(int)
    let workdir = builtins.toStr(Config.workdir)
    let pidir = Config.pid.to(string)
    let basedir = Path(workdir) / Path(pidir)
    let pagedir = basedir / Path("pages")

    createDir(string pagedir)

    var bitmask = newSeq[bool](pageSize)
    var bitNum = 0
    var offset = 0

    template dumpPage(columns: seq[string], passColumn: nimpy.PyObject, failColumn: nimpy.PyObject): void =
        var firstPage = 0
        var currentOffset = 0
        
        while true:
            var len = getPageLen(columns[firstPage])

            if offset < currentOffset + len:
                break

            inc firstPage
            currentOffset = currentOffset + len

        var maskOffset = 0

        let indiceOffset = offset - currentOffset

        while maskOffset < bitNum:
            let page = readNumpy(columns[firstPage])

            let len = page.len
            let sliceMax = min((bitNum - maskOffset), len)
            let sliceLen = sliceMax - maskOffset
            let slice = maskOffset..<sliceMax

            var validIndices = newSeqOfCap[int](sliceLen - (sliceLen shr 2))
            var invalidIndices = newSeqOfCap[int](sliceLen shr 2)

            for (i, m) in enumerate(bitmask[slice]):
                if m: validIndices.add(i + indiceOffset)
                else: invalidIndices.add(i + indiceOffset)

            let passPid = base.classes.SimplePageClass.next_id(string basedir).to(string)
            let failPid = base.classes.SimplePageClass.next_id(string basedir).to(string)

            let passPath = string(pagedir / Path(passPid & ".npy"))
            let failPath = string(pagedir / Path(failPid & ".npy"))

            let passPage = page[validIndices]
            let failPage = page[invalidIndices]

            passPage.save(passPath)
            failPage.save(failPath)

            let passPagePy = newPyPage(passPage, string basedir, passPid)
            let failPagePy = newPyPage(failPage, string basedir, failPid)

            discard passColumn.pages.append(passPagePy)
            discard failColumn.pages.append(failPagePy)

            maskOffset = maskOffset + sliceLen
            inc firstPage

    template dumpPages(tablePages: Table[string, seq[string]]): void =
        for (key, col) in tablePages.pairs():
            col.dumpPage(passTablePages[key], failTablePages[key])

    for (i, row) in enumerate(exprCols.iterateRows(tablePages)):
        bitmask[bitNum] = row.checkExpressions(exprCols, expressions, filterType)

        inc bitNum

        if bitNum >= pageSize:
            tablePages.dumpPages()
            offset = offset + bitNum
            bitNum = 0

    if bitNum > 0:
        tablePages.dumpPages()

    template makeTable(T: nimpy.PyObject, columns: OrderedTable[string, nimpy.PyObject]): nimpy.PyObject =
        let tbl = T!()

        for (k, v) in columns.pairs:
            tbl[k] = v

        tbl

    let passTable = makeTable(TableClass, passTablePages)
    let failTable = makeTable(TableClass, failTablePages)

    return (passTable, failTable)

let m = modules()
let Config = m.tablite.modules.config.classes.Config

# Config.PAGE_SIZE = 2

let table = m.tablite.classes.TableClass!({
    "a": @[1, 2, 3, 4],
    "b": @[10, 20, 30, 40],
    "c": @[4, 4, 4, 4]
}.toTable)
let pyExpressions = @[
    m.builtins.classes.DictClass!(column1: "a", criteria: ">=", value2: 2),
    # m.builtins.classes.DictClass!(column1: "b", criteria: "==", value2: 20),
    m.builtins.classes.DictClass!(column1: "a", criteria: "==", column2: "c"),
]

Config.PAGE_SIZE = 2

let (tblPass, tblFail) = filter(table, pyExpressions, "all")

discard tblPass.show()
discard tblFail.show()
