import nimpy
import std/[tables, sugar, options, times, algorithm, enumerate, hashes, sequtils, strformat, math]
import ../[pytypes, numpy, pymodules, utils, dateutils, nimpyext]

const PARITY_TABLE = {
    K_NONETYPE: 0,
    K_BOOLEAN: 1,
    K_INT: 2,
    K_FLOAT: 2,
    K_TIME: 3,
    K_DATE: 4,
    K_DATETIME: 5,
}.toTable()

proc clone*[T](arr: ptr seq[T]): seq[T] =
    let l = arr[].len
    var clone = newSeq[T](l)
    clone[0].addr.copyMem(addr arr[0], l * sizeof(T))
    result = clone

proc uniqueColumnValues(pagePaths: seq[string]): seq[PY_ObjectND] =
    var uniqueVals: seq[PY_ObjectND] = @[]
    for v in iterateColumn[PY_ObjectND](pagePaths):
        if not(v in uniqueVals):
            uniqueVals.add(v)
    result = uniqueVals

method toFloat*(self: PY_ObjectND): float {.base, inline.} = implement("PY_ObjectND.`toFloat` must be implemented by inheriting class: " & $self.kind)
method toFloat*(self: PY_NoneType): float = -Inf
method toFloat*(self: PY_Boolean): float = float(self.value)
method toFloat*(self: PY_Int): float = float(self.value)
method toFloat*(self: PY_Float): float = self.value
method toFloat*(self: PY_Date): float = self.value.toTime().toUnixFloat()
method toFloat*(self: PY_Time): float = self.value.duration2Seconds()
method toFloat*(self: PY_DateTime): float = self.value.toTime().toUnixFloat()

proc cmpNonText(this, other: PY_ObjectND): int =
    let r = system.cmp[int](PARITY_TABLE[this.kind], PARITY_TABLE[other.kind])

    if r == 0:
        return system.cmp[float](this.toFloat(), other.toFloat())

    return r

proc cmpText(this, other: PY_String): int = system.cmp[string](this.value, other.value)

proc unixSort*(values: seq[PY_ObjectND]): OrderedTable[PY_ObjectND, int] =
    var text: seq[PY_String] = @[]
    var nonText: seq[PY_ObjectND] = @[]

    for v in values:
        if v.kind == K_STRING:
            text.add(PY_String(v))
        else:
            nonText.add(v)

    nonText.sort(cmpNonText)
    text.sort(cmpText)

    var d = initOrderedTable[PY_ObjectND, int]()
    for ix, v in enumerate(nonText):
        d[v] = ix

    var l = len(nonText)
    for ix, v in enumerate(text):
        d[v] = ix + l

    return d

proc cmpTuples(this, other: (int, PY_ObjectND)): int = system.cmp[int](this[0], other[0])

proc cmpSeqs(this, other: (seq[int], seq[PY_ObjectND])): int =
    for (t, o) in zip(this[0], other[0]):
        var r = system.cmp[int](t, o)
        if r != 0:
            return r
    return 0

iterator getRowObjects(table: nimpy.PyObject, columns: seq[string], indices: seq[int]): (int, int, int, seq[(string, PY_ObjectND)]) =
    let base = modules().tablite.modules.base
    let columnsPages: seq[seq[string]] = collect: (for name in columns: base.collectPages(table[name]))
    let pageCount = len(columnsPages[0])

    var ix = 0
    var cnt = 0
    for pageIndex in 0 ..< pageCount:
        let pageLength = getPageLen(columnsPages[0][pageIndex])
        var pages: Option[seq[BaseNDArray]] = none[seq[BaseNDArray]]()
        while ix < len(indices):
            if indices[ix] >= cnt + pageLength:
                break
            if unlikely(isNone(pages)):
                var p = collect: (for paths in columnsPages: readNumpy(paths[pageIndex]))
                pages = some(p)
            let row = collect: (for (i, p) in enumerate(pages.get()): (columns[
                    i], getItemAsObject(p, indices[ix] - cnt)))
            yield (pageIndex, indices[ix], indices[ix] - cnt, row)
            inc ix
        cnt = cnt + pageLength

proc savePages(sliceData: seq[seq[PY_ObjectND]], columns: seq[nimpy.PyObject], pageIndex: int) =
    let m = modules()
    let tabliteConfig = m.tablite.modules.config.classes.Config
    let wpid = tabliteConfig.pid.to(string)
    let tablitDir = m.builtins.toStr(tabliteConfig.workdir)
    let workdir = &"{tablitDir}/{wpid}"

    for (vals, col) in zip(sliceData, columns):
        let pid = m.tablite.modules.base.classes.SimplePageClass.next_id(
                workdir).to(string)
        var arr = newNDArray(vals)
        arr.save(workdir, pid)
        var page = newPyPage(arr, workdir, pid)
        col.pages[pageIndex] = page

proc nearestNeighbourImputation*(T: nimpy.PyObject, sources: seq[string],
        missing: seq[PY_ObjectND], targets: seq[string],
        tqdm: nimpy.PyObject = nil, pbarInp: nimpy.PyObject = nil): nimpy.PyObject =
    let
        m = modules()
        tabliteBase = m.tablite.modules.base
        tabliteConf = m.tablite.modules.config.classes.Config
        pid: string = tabliteConf.pid.to(string)
        workDir: string = m.toStr(tabliteConf.workdir)
        pidDir: string = &"{workDir}/{pid}"

    var normIndex = initOrderedTable[string, Table[PY_ObjectND, float]]()
    var normalisedValues = initOrderedTable[string, seq[float]]()

    for name in sources:
        let pagePaths: seq[string] = tabliteBase.collectPages(T[name])
        let uniqueValues: seq[PY_ObjectND] = uniqueColumnValues(pagePaths)
        let sortedUniqueValues = unixSort(uniqueValues)
        var collectedUniqueValues = collect: (for (k, v) in
                sortedUniqueValues.pairs(): (v, k))
        collectedUniqueValues.sort(cmpTuples)
        let sortedKeys = collect: (for (_, k) in collectedUniqueValues: k)
        var d = initTable[PY_ObjectND, float]()
        let notMissingValues = collect:
            for v in sortedKeys:
                if v notin missing:
                    v
        let n = len(notMissingValues)
        for ix, v in enumerate(sortedKeys):
            d[v] = (if v notin missing: ix/n else: Inf)
        var arr = collect: (for v in iterateColumn[PY_ObjectND](pagePaths): d[v])
        normalisedValues[name] = arr
        normIndex[name] = d

    let missingValueIndexTablite: TableIndices = T.index(targets)
    let missingValueIndex = collect(initOrderedTable()): # strip out all that do not have missings.
        for (k, v) in missingValueIndexTablite.pairs():
            for m in missing:
                if m in k:
                    {k: v}


    var pbar: nimpy.PyObject
    if pbarInp.isNone:
        let missingValsCounts = collect: (for v in missing_value_index.values(): len(v))
        let totalSteps = sum(missingValsCounts)
        let TqdmClass = if tqdm.isNone: m.tqdm.classes.TqdmClass else: tqdm

        pbar = TqdmClass!(desc: &"imputation.nearest_neighbour", total: totalSteps)
    else:
        pbar = pbarInp

    var ranks: seq[PY_ObjectND] = @[]
    var newOrder = initTable[seq[int], seq[PY_ObjectND]]()

    for k in missingValueIndex.keys():
        for kk in k:
            if kk notin ranks:
                ranks.add(kk)

    let itemOrder = unixSort(ranks)
    for k in missingValueIndex.keys():
        var arr = newSeq[int]()
        for i in k:
            arr.add(itemOrder[i])
        newOrder[arr] = k

    var sortedNewOrder = toSeq(newOrder.pairs())
    sortedNewOrder.sort(cmpSeqs, SortOrder.Descending) # Fewest None's are at the front of the list.

    var targetsColumnsPages: seq[seq[string]] = collect:
        for name in targets:
            tabliteBase.collectPages(T[name])

    var targetsPYColumns: seq[nimpy.PyObject]
    var TableClass = m.builtins.getType(T)
    var newTable = TableClass!()
    for columnName in T.columns:
        if m.toStr(columnName) in targets:
            var c = tabliteBase.classes.ColumnClass!(pidDir)
            for p in T[columnName].pages:
                discard c.pages.append(p) # copy over all pages
            targetsPYColumns.add(c)
            newTable[columnName] = c
        else:
            newTable[columnName] = T[columnName]

    var targetsPages: Option[seq[seq[PY_ObjectND]]] = none[seq[seq[
            PY_ObjectND]]]()
    var lastPageIndex = -1
    var sparseMap = collect(initOrderedTable()):
        for t in targets:
            {t: initOrderedTable[int, PY_ObjectND]()}

    for (_, key) in sortedNewOrder:
        for (pageIndex, rowIdx, pageIdx, row) in getRowObjects(T, sources, missingValueIndex[key]):
            var errMap = newSeq[float](m.getLen(T))
            for (n, v) in row:
                let normValue = normIndex[n][v]
                if normValue != Inf:
                    for (ix, e) in enumerate(zip(errMap, normalisedValues[n])):
                        var (e1, e2) = e
                        errMap[ix] = e1 + abs(normValue - e2)
            let minErr = min(errMap)
            let ix = errMap.find(minErr)
            if lastPageIndex != pageIndex:
                if lastPageIndex != -1:
                    savePages(targetsPages.get(), targetsPYColumns, lastPageIndex)
                var p = collect:
                    for paths in targetsColumnsPages:
                        collect:
                            for v in readNumpy(paths[pageIndex]).iterateObjectPage():
                                v
                targetsPages = some(p)
                lastPageIndex = pageIndex
            var newTable = targetsPages.get().addr

            for (i, name) in enumerate(targets):
                var currentValue = newTable[i][pageIdx]
                if currentValue notin missing: # no need to replace anything.
                    continue

                var sv: PY_ObjectND
                if ix notin sparseMap[name]:
                    sv = getItemAsObject(targetsColumnsPages[i], ix)
                    sparseMap[name][ix] = sv
                else:
                    sv = sparseMap[name][ix]
                if sv notin missing: # can confidently impute.
                    newTable[i][pageIdx] = sv
                    sparseMap[name][rowIdx] = sv
                else: # replacement is required, but ix points to another missing value.
                # we therefore have to search after the next best match:
                    var tmpErrMap = clone(addr errMap)
                    for _ in 0 ..< len(errMap):
                        let tmpMinErr = min(tmpErrMap)
                        let tmpIx = tmpErrMap.find(tmpMinErr)

                        var sv {.noinit.}: PY_ObjectND

                        if tmpIx notin sparseMap[name]:
                            sv = getItemAsObject(targetsColumnsPages[i], tmpIx)
                            sparseMap[name][tmpIx] = sv
                        else:
                            sv = sparseMap[name][tmpIx]

                        if rowIdx == tmpIx or sv in missing:
                            tmpErrMap[tmpIx] = Inf
                            continue
                        else:
                            newTable[i][pageIdx] = sv
                            sparseMap[name][rowIdx] = sv
                            break
            discard pbar.update(1)

    if lastPageIndex != -1:
        savePages(targetsPages.get(), targetsPYColumns, lastPageIndex)

    discard pbar.close()
    return newTable
