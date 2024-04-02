import nimpy
# from zipper import zipper
import std/[sugar, enumerate, strutils, tables, sequtils, options]
import ../[pymodules, utils, pytypes, numpy, dateutils]

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

proc isEqual[T](a, b: T): bool {.inline.} = a == b
proc castSame[T](a, b: T): (T(a), T(b))

proc checkGT(row: seq[PY_ObjectND], exprCols: seq[string], left: PY_ObjectND, right: PY_ObjectND): bool {.inline.} =
    implement("checkGT")

proc checkGE(row: seq[PY_ObjectND], exprCols: seq[string], left: PY_ObjectND, right: PY_ObjectND): bool {.inline.} =
    implement("checkGE")

proc checkEQ(row: seq[PY_ObjectND], exprCols: seq[string], left: PY_ObjectND, right: PY_ObjectND): bool {.inline.} =
    if left.kind != right.kind:
        case left.kind:
        of K_INT:
            if right.kind == K_FLOAT:
                # int and float are same-y, use equals oper
                return float(PY_Int(left).value) == PY_Float(right).value
        of K_FLOAT:
            if right.kind == K_INT:
                # int and float are same-y, use equals oper
                return PY_Float(left).value == float (PY_Int(right).value)
        of K_DATETIME:
            if right.kind == K_DATE:
                # date and datetime are same-y, use equals oper
                return PY_DateTime(left).value == PY_Date(right).value
        of K_DATE:
            if right.kind == K_DATETIME:
                # date and datetime are same-y, use equals oper
                return PY_Date(left).value == PY_DateTime(right).value
        else: discard

        return false

    return (
        case left.kind:
        of K_NONETYPE: true
        of K_BOOLEAN: isEqual[PY_Boolean](left, right)
        of K_INT: isEqual[PY_Int](left, right)
        of K_FLOAT: isEqual[PY_Float](left, right)
        of K_STRING: isEqual[PY_String](left, right)
        of K_DATE: isEqual[PY_Date](left, right)
        of K_TIME: isEqual[PY_Time](left, right)
        of K_DATETIME: isEqual[PY_DateTime](left, right)
    )

proc checkLT(row: seq[PY_ObjectND], exprCols: seq[string], left: PY_ObjectND, right: PY_ObjectND): bool {.inline.} =
    implement("checkLT")

proc checkLE(row: seq[PY_ObjectND], exprCols: seq[string], left: PY_ObjectND, right: PY_ObjectND): bool {.inline.} =
    implement("checkLE")

proc checkNE(row: seq[PY_ObjectND], exprCols: seq[string], left: PY_ObjectND, right: PY_ObjectND): bool {.inline.} =
    return not row.checkEQ(exprCols, left, right)

proc checkIN(row: seq[PY_ObjectND], exprCols: seq[string], left: PY_ObjectND, right: PY_ObjectND): bool {.inline.} =
    implement("checkIN")

proc checkExpression(row: seq[PY_ObjectND], exprCols: seq[string], xpr: Expression): bool {.inline.} =
    let (leftXpr, criteria, rightXpr) = xpr
    let left = row.extractValue(exprCols, leftXpr)
    let right = row.extractValue(exprCols, rightXpr)

    return (
        case criteria:
        of FM_GT: row.checkGT(exprCols, left, right)
        of FM_GE: row.checkGE(exprCols, left, right)
        of FM_EQ: row.checkEQ(exprCols, left, right)
        of FM_LT: row.checkLT(exprCols, left, right)
        of FM_LE: row.checkLE(exprCols, left, right)
        of FM_NE: row.checkNE(exprCols, left, right)
        of FM_IN: row.checkIN(exprCols, left, right)
    )

proc checkExpressions(row: seq[PY_ObjectND], exprCols: seq[string], expressions: seq[Expression], filterType: FilterType): bool {.inline.} =
    case filterType:
    of FT_ANY: any(expressions, xpr => row.checkExpression(exprCols, xpr))
    of FT_ALL: all(expressions, xpr => row.checkExpression(exprCols, xpr))

proc filter(table: nimpy.PyObject, pyExpressions: seq[nimpy.PyObject], filterTypeName: string): nimpy.PyObject =
    let m = modules()
    let builtins = m.builtins
    let tablite = m.tablite
    let base = tablite.modules.base
    let Config = tablite.modules.config.classes.Config

    let filterType = (
        case filterTypeName.toLower():
        of "any": FT_ANY
        of "all": FT_ALL
        else: raise newException(ValueError, "invalid filter type '" & filterTypeName & "' expected " & $FILTER_TYPES)
    )

    if pyExpressions.len == 0:
        return table

    let columns = collect: (for c in table.columns.keys(): c.to(string))

    if columns.len == 0:
        return table

    var expressionPages = initTable[string, seq[string]]()
    
    template addParam(columnName: string, valueName: string, paramName: string): auto =
        var res {.noInit.}: ExpressionValue

        if columnName in expression:
            if valueName in expression:
                raise newException(ValueError, "filter can only take 1 " & paramName & " expr element, got 2")

            let c = expression[columnName].to(string)

            if c notin columns:
                raise newException(ValueError, "no such column '" & $c & "'in " & $columns)

            if c notin expressionPages:
                expressionPages[c] = base.collectPages(table[c])

            res = (some(c), none[PY_ObjectND]())
        elif valueName notin expression:
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

                if key notin FILTER_KEYS:
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
    let pageCount = base.collectPages(table[columns[0]]).len

    let exprCols = toSeq(expressionPages.keys)
    for pagePaths in expressionPages.pageZipper():
        var bitmask = newSeqOfCap[bool](pageSize)
        var bitNum = 0

        for (i, row) in enumerate(pagePaths.iteratePages()):

            bitmask[bitNum] = row.checkExpressions(exprCols, expressions, filterType)

            inc bitNum



var table = {
    "0": @[0, 1, 2],
    "1": @[3, 4, 5],
    "3": @[6, 7, 8],
}.toTable

# for a  in table.values:
#     discard a

for a in pageZipper(table):
    echo a
