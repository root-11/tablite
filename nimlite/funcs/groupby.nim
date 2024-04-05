import nimpy
import std/[math, tables, strutils, strformat, sequtils, enumerate, sugar, options, algorithm]
import ../[pytypes, numpy, pymodules, nimpyext]
import ./imputation

type Accumulator* = enum
    Max, Min, Sum, First, Last, Product, Count,
    CountUnique, Average, StandardDeviation, Median, Mode

proc str2Accumulator*(str: string): Accumulator =
    return (
        case str.toLower():
        of "max": Max
        of "min": Min
        of "sum": Sum
        of "first": First
        of "last": Last
        of "product": Product
        of "count": Count
        of "countunique", "count_unique": CountUnique
        of "avg", "average": Average
        of "stdev", "standarddeviation": StandardDeviation
        of "median": Median
        of "mode": Mode
        else: raise newException(ValueError, &"Unrecognized groupby accumulator - {str}.")
    )

# =============================================================
type GroupByFunction = ref object of RootObj
    val: Option[PY_ObjectND]

method `value=`*(self: GroupByFunction, value: Option[PY_ObjectND]) {. base .}=
    self.val = value
method value*(self: GroupByFunction): Option[PY_ObjectND] {. base .}=
    return self.val

method update(self: GroupByFunction, value: Option[PY_ObjectND]) {.base.} =
    raise newException(Defect, "not implemented.")

proc constructGroupByFunction[T: GroupByFunction](self: T): T =
    result = self

proc newGroupByFunction(): GroupByFunction =
    result = GroupByFunction().constructGroupByFunction()
# =============================================================

# =============================================================
type Limit = ref object of GroupByFunction

proc constructLimit[T: Limit](self: T): T =
    result = self.constructGroupByFunction()

proc newLimit(acc: Accumulator): Limit =
    result = Limit().constructLimit()

method run(self: Limit, value: Option[PY_ObjectND]) {.base.} =
    raise newException(FieldDefect, "not implemented")

method update(self: Limit, value: Option[PY_ObjectND]) =
    if value.isNone():
        discard
    elif self.value.isNone():
        self.value = value
    else:
        self.run(value)
# =============================================================

# =============================================================
type GroupbyMax = ref object of Limit

proc constructGroupbyMax[T: GroupbyMax](self: T): T =
    result = self.constructLimit()

proc newGroupbyMax(): GroupbyMax =
    result = GroupbyMax().constructGroupbyMax()

method run(self: GroupbyMax, value: Option[PY_ObjectND]) =
    var v = self.value.get()
    var vv = value.get()
    if v.kind == vv.kind:
        if vv > v:
            self.value = some(vv)
    else:
        raise newException(Defect, "cannot find max between mixed types.")
# =============================================================

# =============================================================
type GroupbyMin = ref object of Limit

proc constructGroupbyMin[T: GroupbyMin](self: T): T =
    result = self.constructLimit()

proc newGroupbyMin(): GroupbyMin =
    result = GroupbyMin().constructGroupbyMin()

method run(self: GroupbyMin, value: Option[PY_ObjectND]) =
    var v = self.value.get()
    var vv = value.get()
    if v.kind == vv.kind:
        if vv < v:
            self.value = some(vv)
    else:
        raise newException(Defect, "cannot find min between mixed types.")
# =============================================================

# =============================================================
type GroupBySum = ref object of GroupByFunction

proc constructGroupBySum[T: GroupBySum](self: T): T =
    result = self.constructGroupByFunction()
    result.value = some(newPY_Object(0.0))

proc newGroupBySum(): GroupBySum =
    result = GroupBySum().constructGroupBySum()

method update(self: GroupBySum, value: Option[PY_ObjectND]) =
    var unSupportedTypes = @[K_DATE, K_TIME, K_DATETIME, K_STRING]
    if value.isNone() or value.get().kind in unSupportedTypes:
        raise newException(ValueError, &"Sum of {value.get().kind} doesn't make sense.")

    var v: float = PY_Float(self.value.get()).value
    var vv: float
    if value.get().kind == K_INT:
        vv = float PY_Int(value.get()).value
    elif value.get().kind == K_FLOAT:
        vv = PY_Float(value.get()).value
    self.value = some(newPY_Object(v + vv))
# =============================================================

# =============================================================
type GroupByProduct = ref object of GroupByFunction

proc constructGroupByProduct[T: GroupByProduct](self: T): T =
    result = self.constructGroupByFunction()
    result.value = some(newPY_Object(1.0))

proc newGroupByProduct(): GroupByProduct =
    result = GroupByProduct().constructGroupByProduct()

method update(self: GroupByProduct, value: Option[PY_ObjectND]) =
    var unSupportedTypes = @[K_DATE, K_TIME, K_DATETIME, K_STRING]
    if value.isNone() or value.get().kind in unSupportedTypes:
        raise newException(ValueError, &"Product of {value.get().kind} doesn't make sense.")

    var v: float = PY_Float(self.value.get()).value
    var vv: float
    if value.get().kind == K_INT:
        vv = float PY_Int(value.get()).value
    elif value.get().kind == K_FLOAT:
        vv = PY_Float(value.get()).value
    self.value = some(newPY_Object(v * vv))
# =============================================================

# =============================================================
type GroupByFirst = ref object of GroupByFunction
    isSet: bool = false

proc constructGroupByFirst[T: GroupByFirst](self: T): T =
    result = self.constructGroupByFunction()

proc newGroupByFirst(): GroupByFirst =
    result = GroupByFirst().constructGroupByFirst()

method update(self: GroupByFirst, value: Option[PY_ObjectND]) =
    if not self.isSet:
        self.value = value
        self.isSet = true
# =============================================================

# =============================================================
type GroupByLast = ref object of GroupByFunction

proc constructGroupByLast[T: GroupByLast](self: T): T =
    result = self.constructGroupByFunction()
    result.value = some(newPY_Object())

proc newGroupByLast(): GroupByLast =
    result = GroupByLast().constructGroupByLast()

method update(self: GroupByLast, value: Option[PY_ObjectND]) =
    self.value = value
# =============================================================

# =============================================================
type GroupByCount = ref object of GroupByFunction

proc constructGroupByCount[T: GroupByCount](self: T): T =
    result = self.constructGroupByFunction()
    result.value = some(newPY_Object(0))

proc newGroupByCount(): GroupByCount =
    result = GroupByCount().constructGroupByCount()

method update(self: GroupByCount, value: Option[PY_ObjectND]) =
    var v: int = int PY_Int(self.value.get()).value
    self.value = some(newPY_Object(v + 1))
# =============================================================

# =============================================================
type GroupByCountUnique = ref object of GroupByFunction
    items: seq[PY_ObjectND] = @[]

proc constructGroupByCountUnique[T: GroupByCountUnique](self: T): T =
    result = self.constructGroupByFunction()
    result.value = some(newPY_Object(0))

proc newGroupByCountUnique(): GroupByCountUnique =
    result = GroupByCountUnique().constructGroupByCountUnique()

method update(self: GroupByCountUnique, value: Option[PY_ObjectND]) =
    if value.get() notin self.items:
        self.items.add(value.get())
    self.value = some(newPY_Object(len(self.items)))
# =============================================================

# =============================================================
type GroupByAverage = ref object of GroupByFunction
    sum: float = 0
    count: int = 0

proc constructGroupByAverage[T: GroupByAverage](self: T): T =
    result = self.constructGroupByFunction()
    result.value = some(newPY_Object(0.0))

proc newGroupByAverage(): GroupByAverage =
    result = GroupByAverage().constructGroupByAverage()

method update(self: GroupByAverage, value: Option[PY_ObjectND]) =
    var unSupportedTypes = @[K_DATE, K_TIME, K_DATETIME, K_STRING]
    if value.isNone() or value.get().kind in unSupportedTypes:
        raise newException(ValueError, &"Average of {value.get().kind} doesn't make sense.")
    var v: float
    if value.get().kind == K_INT:
        v = float PY_Int(value.get()).value
    else:
        v = PY_Float(value.get()).value
    self.sum += v
    self.count += 1
    self.value = some(newPY_Object(self.sum / float self.count))
# =============================================================

# =============================================================
type GroupByStandardDeviation = ref object of GroupByFunction
    count: int = 0
    mean: float = 0.0
    c: float = 0.0
#[
    Uses J.P. Welfords (1962) algorithm.
    For details see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
]#
proc constructGroupByStandardDeviation[T: GroupByStandardDeviation](self: T): T =
    result = self.constructGroupByFunction()
    result.value = some(newPY_Object(0.0))

proc newGroupByStandardDeviation(): GroupByStandardDeviation =
    result = GroupByStandardDeviation().constructGroupByStandardDeviation()

method value*(self: GroupByStandardDeviation): Option[PY_ObjectND] =
    if self.count <= 1:
        return some(newPY_Object(0.0))
    var variance = self.c / float(self.count - 1)
    return some(newPY_Object(pow(variance, (1/2))))

method update(self: GroupByStandardDeviation, value: Option[PY_ObjectND]) =
    var unSupportedTypes = @[K_DATE, K_TIME, K_DATETIME, K_STRING]
    if value.isNone() or value.get().kind in unSupportedTypes:
        raise newException(ValueError, &"Std.dev. of {value.get().kind} doesn't make sense.")
    self.count += 1
    var v: float
    if value.get().kind == K_INT:
        v = float(PY_Int(value.get()).value)
    else:
        v = PY_Float(value.get()).value
    var dt: float = v - self.mean
    self.mean += dt / float(self.count)
    self.c += dt * (v - self.mean)
# =============================================================

# =============================================================
const PARITY_TABLE = {
    K_NONETYPE: 0,
    K_BOOLEAN: 1,
    K_INT: 2,
    K_FLOAT: 2,
    K_TIME: 3,
    K_DATE: 4,
    K_DATETIME: 5,
}.toTable()

type Histogram = ref object of GroupByFunction
    hist*: OrderedTable[PY_ObjectND, int] = initOrderedTable[PY_ObjectND, int]()

proc constructHistogram[T: Histogram](self: T): T =
    result = self.constructGroupByFunction()

proc newHistogram(): Histogram =
    result = Histogram().constructHistogram()

proc cmpNonText(this, other: (PY_ObjectND, int)): int =
    let typ = system.cmp[int](PARITY_TABLE[this[0].kind], PARITY_TABLE[other[0].kind])
    if typ == 0:
        let val = system.cmp[float](this[0].toFloat(), other[0].toFloat())
        if val == 0:
            return system.cmp[int](this[1], other[1])
        return val
    return typ
    
proc cmpText(this, other: (PY_String, int)): int =
    let val = system.cmp[string](this[0].value, other[0].value)
    if val == 0:
        return system.cmp[int](this[1], other[1])
    return val

method sortedHistogram(self: Histogram): seq[(PY_ObjectND, int)] {.base.} =
    var text: seq[(PY_String, int)] = @[]
    var nonText: seq[(PY_ObjectND, int)] = @[]

    for (k, v) in self.hist.pairs:
        if k.kind == K_STRING:
            text.add((PY_String(k), v))
        else:
            nonText.add((k, v))
    
    nonText.sort(cmpNonText)
    text.sort(cmpText)

    var res: seq[(PY_ObjectND, int)] = @[]
    for (v, c) in nonText:
        res.add((v, c))
    for (v, c) in text:
        res.add((v, c))
    return res

method update(self: Histogram, value: Option[PY_ObjectND]) =
    var v = value.get()
    if self.hist.hasKey(v):
        self.hist[v] += 1
    else:
        self.hist[v] = 1
# =============================================================

# =============================================================
type GroupByMedian = ref object of Histogram

proc constructGroupByMedian[T: GroupByMedian](self: T): T =
    result = self.constructHistogram()

proc newGroupByMedian(): GroupByMedian =
    result = GroupByMedian().constructGroupByMedian()

template castToFloat(n: PY_ObjectND): float =
    if n.kind == K_INT:
        float PY_Int(n).value
    else:
        PY_Float(n).value

method value*(self: GroupByMedian): Option[PY_ObjectND] =
    var keys = len(self.hist)
    if keys == 1:
        for k in self.hist.keys():
            return some(k)
    elif keys mod 2 == 0:
        var 
            A: PY_ObjectND
            B: PY_ObjectND
            total: int = 0
            counts: seq[int] = collect:
                for v in self.hist.values:
                    v
            midpoint: float = sum(counts) / 2
        for (k, v) in self.sortedHistogram():
            total += v
            A = B
            B = k
            if float(total) > midpoint:
                var s = @[K_INT, K_FLOAT]
                if A.kind notin s or B.kind notin s:
                    raise newException(ValueError, "Can't find median of non numbers.")
                return some(newPY_Object((castToFloat(A) + castToFloat(B)) / 2.0))
    else:
        var 
            counts: seq[int] = collect:
                for v in self.hist.values:
                    v
            midpoint: float = sum(counts) / 2
            total = 0
        for (k, v) in self.sortedHistogram():
            total += v
            if float(total) > midpoint:
                return some(k)
# =============================================================

# =============================================================
type GroupByMode = ref object of Histogram

proc constructGroupByMode[T: GroupByMode](self: T): T =
    result = self.constructHistogram()

proc newGroupByMode(): GroupByMode =
    result = GroupByMode().constructGroupByMode()

proc cmpNonText(this, other: (int, PY_ObjectND)): int =
    let cnt = system.cmp[int](this[0], other[0])
    if cnt == 0:
        let typ = system.cmp[int](PARITY_TABLE[this[1].kind], PARITY_TABLE[other[1].kind])
        if typ == 0:
            return system.cmp[float](this[1].toFloat(), other[1].toFloat())
        return typ
    return cnt
    
proc cmpText(this, other: (int, PY_String)): int =
    let cnt = system.cmp[int](this[0], other[0])
    if cnt == 0:
        return system.cmp[string](this[1].value, other[1].value)
    return cnt

proc sortedHistogramReversed(self: Histogram, order: SortOrder = SortOrder.Ascending): seq[(int, PY_ObjectND)] =
    var text: seq[(int, PY_String)] = @[]
    var nonText: seq[(int, PY_ObjectND)] = @[]

    for (k, cnt) in self.hist.pairs:
        if k.kind == K_STRING:
            text.add((cnt, PY_String(k)))
        else:
            nonText.add((cnt, k))
    
    nonText.sort(cmpNonText, order)
    text.sort(cmpText, order)

    var res: seq[(int, PY_ObjectND)] = @[]
    for (cnt, v) in nonText:
        res.add((cnt, v))
    for (cnt, v) in text:
        res.add((cnt, v))
    return res

method value*(self: GroupByMode): Option[PY_ObjectND] =
    var hist = self.sortedHistogramReversed(SortOrder.Descending)
    var (_, mostFreq) = hist[0]
    return some(mostFreq)
# =============================================================


# =============================================================
# =============================================================
# =============================================================

proc getGroupByFunction(acc: Accumulator): GroupByFunction =
    return (
        case acc:
        of Accumulator.Max: newGroupbyMax()
        of Accumulator.Min: newGroupbyMin()
        of Accumulator.Sum: newGroupBySum()
        of Accumulator.Product: newGroupByProduct()
        of Accumulator.First: newGroupByFirst()
        of Accumulator.Last: newGroupByLast()
        of Accumulator.Count: newGroupByCount()
        of Accumulator.CountUnique: newGroupByCountUnique()
        of Accumulator.Average: newGroupByAverage()
        of Accumulator.StandardDeviation: newGroupByStandardDeviation()
        of Accumulator.Median: newGroupByMedian()
        of Accumulator.Mode: newGroupByMode()
    )

proc getPages(indices: seq[seq[PY_ObjectND]], columnIndex: int): seq[nimpy.PyObject] =
    let
        m = modules()
        tabliteConfig = m.tablite.modules.config.classes.Config
        pageSize = tabliteConfig.PAGE_SIZE.to(int)
        pageCount = int ceil(len(indices) / pageSize)
        wpid = tabliteConfig.pid.to(string)
        tablitDir = m.builtins.toStr(tabliteConfig.workdir)
        workdir = &"{tablitDir}/{wpid}"

    var
        s: seq[PY_ObjectND] = newSeq[PY_ObjectND](pageSize)
        pages: seq[nimpy.PyObject] = newSeqOfCap[nimpy.PyObject](pageCount)
        ix = 0

    proc save(values: seq[PY_ObjectND], pageLen: int) =
        let pid = m.tablite.modules.base.classes.SimplePageClass.next_id(workdir).to(string)
        var ndarr = newNDArray(values[0 ..< pageLen])
        ndarr.save(workdir, pid)
        var page = newPyPage(ndarr, workdir, pid)
        pages.add(page)

    for arr in indices:
        s[ix] = arr[columnIndex]
        inc ix
        if ix == pageSize: # create page
            save(s, ix)
            ix = 0
    if ix > 0:
        save(s, ix)

    return pages

proc getPages(values: seq[PY_ObjectND]): seq[nimpy.PyObject] =
    let
        m = modules()
        tabliteConfig = m.tablite.modules.config.classes.Config
        pageSize = tabliteConfig.PAGE_SIZE.to(int)
        pageCount = int ceil(len(values) / pageSize)
        wpid = tabliteConfig.pid.to(string)
        tablitDir = m.builtins.toStr(tabliteConfig.workdir)
        workdir = &"{tablitDir}/{wpid}"

    var
        s: seq[PY_ObjectND] = newSeq[PY_ObjectND](pageSize)
        pages: seq[nimpy.PyObject] = newSeqOfCap[nimpy.PyObject](pageCount)
        ix = 0

    proc save(values: seq[PY_ObjectND], pageLen: int) =
        let pid = m.tablite.modules.base.classes.SimplePageClass.next_id(workdir).to(string)
        var ndarr = newNDArray(values[0 ..< pageLen])
        ndarr.save(workdir, pid)
        var page = newPyPage(ndarr, workdir, pid)
        pages.add(page)

    for v in values:
        s[ix] = v
        inc ix
        if ix == pageSize: # create page
            save(s, ix)
            ix = 0
    if ix > 0:
        save(s, ix)

    return pages

iterator pageZipper[T](iters: OrderedTable[string, seq[T]]): seq[T] =
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

proc groupby*(T: nimpy.PyObject, keys: seq[string], functions: seq[(string, Accumulator)], tqdm: nimpy.PyObject = modules().tqdm.classes.TqdmClass): nimpy.PyObject =
    let
        m = modules()
        tabliteBase = m.tablite.modules.base
        tabliteConfig = m.tablite.modules.config.classes.Config
        pid = tabliteConfig.pid.to(string)
        workDir = m.builtins.toStr(tabliteConfig.workdir)
        pidDir = &"{workDir}/{pid}"

    if (len(keys) + len(functions)) == 0:
        raise newException(ValueError, "No keys or functions?")

    var unique_keys: seq[string] = @[]
    for k in keys:
        if k notin unique_keys:
            unique_keys.add(k)
        else:
            raise newException(ValueError, "duplicate keys found across rows and columns.")

    # only keys will produce unique values for each key group.
    if len(keys) > 0 and len(functions) == 0:
        var indexes: seq[seq[PY_ObjectND]] = collect:
            for a in T.index(keys).keys():
                a

        var newTable = m.tablite.classes.TableClass!()
        for (i, k) in enumerate(keys):
            var pages = getPages(indexes, i)
            var column = tabliteBase.classes.ColumnClass!(pidDir)
            for p in pages:
                discard column.pages.append(p)
            newTable[keys[i]] = column
        return newTable

    # grouping is required...
    # 1. Aggregate data.
    var columnNames: seq[string] = keys
    for (cn, acc) in functions:
        if cn notin columnNames:
            columnNames.add(cn)

    # var relevantT = T.slice(columnNames)
    var columnsPaths: OrderedTable[string, seq[string]] = collect(initOrderedTable()):
        for cn in columnNames:
            {cn: tabliteBase.collectPages(T[cn])}
    var TqdmClass = if tqdm.isNone: m.tqdm.classes.TqdmClass else: tqdm
    var pbar = TqdmClass!(desc: &"groupby", total: len(columnsPaths[toSeq(columnsPaths.keys)[0]]))
    var aggregationFuncs = initOrderedTable[seq[PY_ObjectND], seq[(string, GroupByFunction)]]()
    for pagesZipped in pageZipper(columnsPaths):
        for row in iteratePages(pagesZipped):
            var d = collect(initOrderedTable()):
                for (i, v) in enumerate(row):
                    {columnNames[i]: v}
            var key: seq[PY_ObjectND] = collect:
                for k in keys:
                    d[k]
            var aggFuncs: seq[(string, GroupByFunction)]
            if aggregationFuncs.hasKey(key):
                aggFuncs = aggregationFuncs[key]
            else:
                aggregationFuncs[key] = collect:
                    for (cn, acc) in functions:
                        (cn, getGroupByFunction(acc))
                aggFuncs = aggregationFuncs[key]
            for (cn, fun) in aggFuncs:
                fun.update(some(d[cn]))
        discard pbar.update(1)
    
    var keysFuncCols = keys
    for (cn, acc) in functions:
        keysFuncCols.add(cn)
    var cols: seq[seq[PY_ObjectND]] = collect:
        for _ in keysFuncCols:
            newSeq[PY_ObjectND]()
    for (keyTup, funcs) in aggregationFuncs.pairs:
        for (ix, keyVal) in enumerate(keyTup):
            cols[ix].add(keyVal)
        var start: int = len(keys)
        for (ix, p) in enumerate(start, funcs):
            var f: GroupByFunction = p[1]
            cols[ix].add(f.value.get())
    
    var newNames: seq[string] = keys
    for (cn, f) in functions:
        newNames.add(&"{f}({cn})")
    var newTable = m.tablite.classes.TableClass!()
    
    for (cn, data) in zip(newNames, cols):
        var column = tabliteBase.classes.ColumnClass!(pidDir)
        var pages = getPages(data)
        for p in pages:
            discard column.pages.append(p)
        newTable[cn] = column
    discard pbar.close()
    return newTable
