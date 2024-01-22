import std/[os, tables, sugar, sets, sequtils, strutils, cpuinfo, paths, enumerate, unicode, times, macros]
import nimpy as nimpy
from nimpyext import `!`
import utils
import std/options as opt
from std/math import clamp
import pymodules as pymodules
import pytypes
import numpy
import infertypes as infer
import typetraits
import dateutils

type ColSliceInfo = (Path, int)
type ColInfo = Table[string, ColSliceInfo]
type DesiredColumnInfo = object
    original_name: string
    `type`: KindObjectND
    allow_empty: bool
type Mask = enum
    INVALID = -1
    UNUSED = 0
    VALID = 1

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

template makePage[T: typed](dt: typedesc[T], page: BaseNDArray, mask: var seq[Mask], reason_lst: var seq[string], conv: proc, allow_empty: bool, original_name: string, desired_name: string, desired_type: KindObjectND): T =
    template getTypeUserName(t: KindObjectND): string =
        case t:
        of K_BOOLEAN: "bool"
        of K_INT: "int"
        of K_FLOAT: "float"
        of K_STRING: "str"
        of K_DATE: "date"
        of K_TIME: "time"
        of K_DATETIME: "datetime"
        of K_NONETYPE: "NoneType"
    
    template createCastErrorReason(v: string, kind: KindObjectND): string =
        "Cannot cast '" & v & "' from '" & original_name & "[" & getTypeUserName(kind) & "]' to '" & desired_name & "[" & getTypeUserName(desired_type) & "]'."
    
    template createNoneErrorReason(): string =
        "'" & original_name & "' cannot be empty in '" & desired_name & "[" & getTypeUserName(desired_type) & "]'."

    template createEmptyErrorReason(): string =
        "'" & original_name & "' cannot be empty string in '" & desired_name & "[" & getTypeUserName(desired_type) & "]'."


    when T is UnicodeNDArray:
        var canBeNone: bool
        var isObjectPage: bool

        # only string and object types can have nones
        case page.kind:
        of K_UNICODE: canBeNone = true
        of K_OBJECT:
            let types = page.getPageTypes()

            if K_NONETYPE in types:
                if likely(types[K_NONETYPE] > 0):
                    canBeNone = true

            isObjectPage = true
        else:
            canBeNone = false
            isObjectPage = false

        var longest = 1
        let strings = collect:
            for (i, v) in enumerate(page.pgIter):
                var res: seq[Rune]

                if unlikely(canBeNone):
                    if not isObjectPage:
                        when v is string:
                            if v.len == 0:
                                mask[i] = Mask.INVALID
                                reason_lst[i] = createEmptyErrorReason()
                                continue
                    else:
                        when v is PY_ObjectND:
                            case v.kind:
                            of K_STRING:
                                if PY_String(v).value.len == 0:
                                    mask[i] = Mask.INVALID
                                    reason_lst[i] = createEmptyErrorReason()
                                    continue
                            of K_NONETYPE:
                                mask[i] = Mask.INVALID
                                reason_lst[i] = createNoneErrorReason()
                                continue
                            else:
                                discard

                try:
                    let str = conv(v)
                    res = str.toRunes

                    longest = max(longest, res.len)
                    mask[i] = Mask.VALID
                except ValueError:
                    mask[i] = Mask.INVALID

                    var kind {.noinit.}: KindObjectND
                    var strRepr {.noinit.}: string

                    when v is PY_ObjectND:
                        kind = v.kind
                        strRepr = v.toRepr
                    else:
                        strRepr = $v
                        implement("other")

                    reason_lst[i] = createCastErrorReason(strRepr, kind)

                    res = newSeq[Rune]()
                    continue

                res

        let buf = newSeq[Rune](longest * page.len)
        let shape = @[int(buf.len / longest)]

        for (i, str) in enumerate(strings):
            buf[i * longest].addr.copyMem(addr str[0], str.len * sizeof(Rune))

        T(shape: shape, buf: buf, size: longest, kind: K_UNICODE)
    else:
        let buf = collect:
            for (i, v) in enumerate(page.pgIter):
                var res = dt.default()

                when v is PY_ObjectND:
                    case v.kind:
                    of K_NONETYPE:
                        if not allow_empty:
                            reason_lst[i] = createNoneErrorReason()
                            mask[i] = Mask.INVALID
                            continue
                    of K_STRING:
                        if not allow_empty and PY_String(v).value.len == 0:
                            reason_lst[i] = createNoneErrorReason()
                            mask[i] = Mask.INVALID
                            continue
                    else:
                        discard

                try:
                    res = conv(v)
                    mask[i] = Mask.VALID
                except ValueError:
                    mask[i] = Mask.INVALID

                    var kind {.noinit.}: KindObjectND
                    var strRepr {.noinit.}: string

                    when v is PY_ObjectND:
                        kind = v.kind
                        strRepr = v.toRepr
                    else:
                        strRepr = $v
                        implement("other")

                    reason_lst[i] = createCastErrorReason(strRepr, kind)
                    continue

                res
        let shape = @[buf.len]

        T(shape: shape, buf: buf, kind: T.pageKind)

type
    ToDate = object
    ToDateTime = object
    FromDate = object
    FromDateTime = object
    ToTime = object

proc newMkCaster(caster: NimNode): NimNode =
    expectKind(caster, nnkLambda)
    expectLen(caster.params, 2)

    let castR = caster.params[0]
    let castT = caster.params[1][1]
    let nameCastT = caster.params[1][0]

    let descT = newNimNode(nnkBracketExpr)
    let paramsT = newIdentDefs(newIdentNode("T"), descT)

    var trueCastR: NimNode
    var trueCastT: NimNode

    if castR.strVal in [ToDate.name, ToDateTime.name]:
        trueCastR = newIdentNode(DateTime.name)
    elif castR.strVal == ToTime.name:
        trueCastR = newIdentNode(PY_Time.name)
    elif castR.strVal in [bool.name, int.name, float.name, string.name]:
        trueCastR = castR
    else:
        raise newException(FieldDefect, "Uncastable return type '" & $castR.strVal & "'")

    if castT.strVal in [FromDate.name, FromDateTime.name]:
        trueCastT = newIdentDefs(nameCastT, newIdentNode(DateTime.name))
    elif castT.strVal in [bool.name, int.name, float.name, string.name, PY_Time.name, PY_ObjectND.name]:
        trueCastT = newIdentDefs(nameCastT, castT)
    else:
        raise newException(FieldDefect, "Uncastable value type '" & $castT.strVal & "'")

    descT.add(newIdentNode("typedesc"))
    descT.add(castT)

    let descR = newNimNode(nnkBracketExpr)
    let paramsR = newIdentDefs(newIdentNode("R"), descR)

    descR.add(newIdentNode("typedesc"))
    descR.add(castR)

    let subProc = newProc(params = [trueCastR, trueCastT], body = caster.body, procType = nnkLambda)
    let makerProc = newProc(newIdentNode("fnCast"), params = [newNimNode(nnkProcTy), paramsT, paramsR], body = subProc)

    return makerProc

macro mkCaster(caster: untyped) = newMkCaster(caster)
macro mkCasters(converters: untyped) =
    expectKind(converters, nnkStmtList)
    expectKind(converters[0], nnkLambda)

    let params = converters[0].params
    let body = converters[0].body
    let identity = params[1]

    expectKind(identity, nnkIdentDefs)

    let nodes = newNimNode(nnkStmtList)

    for cvtr in body:
        expectKind(cvtr, nnkAsgn)
        expectKind(cvtr[0], nnkIdent)

        let toType = cvtr[0]
        let toBody = cvtr[1]
        let toFunc = newProc(params = [toType, identity], body = toBody, procType = nnkLambda)

        nodes.add(newMkCaster(toFunc))

    return nodes

mkCasters:
    proc (v: bool) =
        bool = v
        int = int v
        float = float v
        string = (if v: "True" else: "False")
        ToDate = days2Date(int v)
        ToDateTime = delta2Date(seconds = int v)
        ToTime = secondsToPY_Time(float v)

mkCasters:
    proc (v: int) =
        bool = v >= 1
        int = v
        float = float v
        string = $v
        ToDate = v.days2Date
        ToDateTime = delta2Date(seconds = v)
        ToTime = secondsToPY_Time(float v)

mkCasters:
    proc (v: float) =
        bool = v >= 1
        int = int v
        float = v
        string = $v
        ToDate = days2Date(int v)
        ToDateTime = seconds2Date(v)
        ToTime = secondsToPY_Time(v)

mkCasters:
    proc(v: FromDate) =
        bool = v.toTime.time2Duration.inSeconds >= 1
        int = v.toTime.time2Duration.inSeconds
        float = float v.toTime.time2Duration.inSeconds
        string = v.format(fmtDate)
        ToDate = v
        ToDateTime = v
        ToTime = v.newPY_Time

mkCasters:
    proc (v: PY_Time) =
        bool = v.value.inSeconds >= 1
        int = v.value.inSeconds
        float = v.value.duration2Seconds
        string = $v.value.duration2Time
        ToDate = v.value.duration2Date
        ToDateTime = v.value.duration2Date
        ToTime = v

mkCasters:
    proc (v: FromDateTime) =
        bool = v.toTime.time2Duration.inSeconds >= 1
        int = v.toTime.time2Duration.inSeconds
        float = v.toTime.time2Duration.duration2Seconds
        string = v.format(fmtDateTime)
        ToDate = v.datetime2Date
        ToDateTime = v
        ToTime = v.newPY_Time

mkCasters:
    proc (v: string) =
        bool = infer.inferBool(addr v)
        int = infer.inferInt(addr v)
        float = infer.inferFloat(addr v)
        string = v
        ToDate = infer.inferDate(addr v).value
        ToDateTime = infer.inferDatetime(addr v).value
        ToTime = infer.inferTime(addr v)

template obj2primCast[R](T1: typedesc, T2: typedesc, TR: typedesc[R], v: PY_ObjectND) = return T2.fnCast(TR)(T1(v).value)
template obj2prim(v: PY_ObjectND) =
    case v.kind:
    of K_BOOLEAN: PY_Boolean.obj2primCast(bool, R, v)
    of K_INT: PY_Int.obj2primCast(int, R, v)
    of K_FLOAT: PY_Float.obj2primCast(float, R, v)
    of K_STRING: PY_String.obj2primCast(string, R, v)
    of K_DATE: PY_Date.obj2primCast(FromDate, R, v)
    of K_TIME: PY_Time.fnCast(R)(PY_Time(v))
    of K_DATETIME: PY_Date.obj2primCast(FromDateTime, R, v)
    of K_NONETYPE: raise newException(ValueError, "cannot cast")

mkCasters:
    proc(v: PY_ObjectND) =
        bool = v.obj2prim()
        int = v.obj2prim()
        float = v.obj2prim()
        string = v.obj2prim()
        ToDate = v.obj2prim()
        ToDateTime = v.obj2prim()
        ToTime = v.obj2prim()

macro mkPageCaster(nBaseType: typedesc, overrides: untyped) =
    expectKind(nBaseType, nnkSym)

    const tPageGenerics = {
        bool.name: @[BooleanNDArray.name],
        int.name: @[Int8NDArray.name, Int16NDArray.name, Int32NDArray.name, Int64NDArray.name],
        float.name: @[Float32NDArray.name, Float64NDArray.name],
        FromDate.name: @[DateNDArray.name],
        FromDateTime.name: @[DateTimeNDArray.name],
        string.name: @[UnicodeNDArray.name],
        PY_ObjectND.name: @[ObjectNDArray.name],
    }.toTable

    let pageGenerics = tPageGenerics[nBaseType.strVal]
    let nPageGenerics = collect: (for ch in pageGenerics: newIdentNode(ch))

    const tCasterPage = {
        bool.name: BooleanNDArray.name,
        int.name: Int64NDArray.name,
        float.name: Float64NDArray.name,
        string.name: UnicodeNDArray.name,
        ToDate.name: DateNDArray.name,
        ToDateTime.name: DateTimeNDArray.name,
        ToTime.name: ObjectNDArray.name
    }.toTable

    const allCastTypes = [bool.name, int.name, float.name, string.name, ToDate.name, ToDateTime.name, ToTime.name]

    let nMaskSeq = newNimNode(nnkBracketExpr).add(bindSym("seq"), bindSym(Mask.name))
    let nMaskVarTy = newNimNode(nnkVarTy).add(nMaskSeq)
    let nMaskDef = newIdentDefs(newIdentNode("mask"), nMaskVarTy)

    let nReasonListSeq = newNimNode(nnkBracketExpr).add(bindSym("seq"), bindSym(string.name))
    let nReasonListVarTy = newNimNode(nnkVarTy).add(nReasonListSeq)
    let nReasonListDef = newIdentDefs(newIdentNode("reason_lst"), nReasonListVarTy)

    let nAllowEmpty = newIdentDefs(newIdentNode("allow_empty"), bindSym(bool.name))
    let nOriginaName = newIdentDefs(newIdentNode("original_name"), bindSym(string.name))
    let nDesiredName = newIdentDefs(newIdentNode("desired_name"), bindSym(string.name))
    let nDesiredType = newIdentDefs(newIdentNode("desired_type"), bindSym(KindObjectND.name))

    let nProcNodes = newNimNode(nnkStmtList)

    let nGeneric = newNimNode(nnkGenericParams)
    var nGenTypes {.noinit.}: NimNode

    if nPageGenerics.len > 1:
        var nNextT = nPageGenerics[^1]

        for i in countdown(nPageGenerics.len - 2, 1):
            nNextT = infix(nPageGenerics[i], "|", nNextT)

        nGenTypes = infix(nPageGenerics[0], "|", nNextT)

    else:
        nGenTypes = nPageGenerics[0]

    let nGenIdet = newIdentDefs(newIdentNode("T"), nGenTypes)

    nGeneric.add(nGenIdet)

    for newCaster in allCastTypes:
        var nPageReturnType {.noinit.}: NimNode
        var nBody {.noinit.}: NimNode

        let tCasterPage = tCasterPage[newCaster]
        let nPageTypeDesc = newNimNode(nnkBracketExpr).add(newIdentNode("typedesc"), newIdentNode(newCaster))
        let nArgReturnType = newIdentDefs(newIdentNode("R"), nPageTypeDesc)
        let nInPage = newIdentDefs(newIdentNode("page"), newIdentNode("T"))

        if tCasterPage in pageGenerics:
            nBody = overrides
            nPageReturnType = (if nPageGenerics.len > 1: newIdentNode("T") else: newIdentNode(tCasterPage))
        else:
            nPageReturnType = newIdentNode(tCasterPage)
            nBody = newNimNode(nnkStmtList)

            let nCallCasterFn = newDotExpr(nBaseType, newIdentNode("fnCast"))
            let nCallCaster = newCall(nCallCasterFn, newIdentNode("R"))

            let nCallMkPageFn = newDotExpr(nPageReturnType, newIdentNode("makePage"))
            let nCallArgs = [newIdentNode("page"), newIdentNode("mask"), newIdentNode("reason_lst"), nCallCaster, newIdentNode("allow_empty"), newIdentNode("original_name"), newIdentNode("desired_name"), newIdentNode("desired_type")]
            let nCallMkPage = newCall(nCallMkPageFn, nCallArgs)

            nBody.add(nCallMkPage)

        let nArgs = [nPageReturnType, nArgReturnType, nInPage, nMaskDef, nReasonListDef, nAllowEmpty, nOriginaName, nDesiredName, nDesiredType]
        let nNewProc = newProc(newIdentNode("castType"), nArgs, nBody)

        nProcNodes.add(nNewProc)
        nNewProc[2] = nGeneric

        # echo nNewProc.repr

    return nProcNodes

mkPageCaster(bool): page
mkPageCaster(int): page
mkPageCaster(float): page
mkPageCaster(FromDate): page
mkPageCaster(FromDateTime): page
mkPageCaster(string): (if allow_empty: page else: UnicodeNDArray.makePage(page, mask, reason_lst, string.fnCast(R), allow_empty, original_name, desired_name, desired_type))
mkPageCaster(PY_ObjectND): ObjectNDArray.makePage(page, mask, reason_lst, PY_ObjectND.fnCast(R), allow_empty, original_name, desired_name, desired_type)

template convertBasicPage[T](page: T, desired_type: KindObjectND, mask: var seq[Mask], reason_lst: var seq[string], allow_empty: bool, original_name: string, desired_name: string): BaseNDArray =
    case desired_type:
    of K_BOOLEAN: bool.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_INT: int.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_FLOAT: float.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_STRING: string.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_DATE: ToDate.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_TIME: ToTime.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_DATETIME: ToDateTime.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    else: corrupted()

proc unusedMaskSearch(arr: var seq[Mask]): int =
    # Partitioned search for faster unused mask retrieval
    const stepSize = 50_000
    let len = arr.len

    if arr[^1] != Mask.UNUSED:
        # page is full
        return arr.len

    if arr[0] == MASK.UNUSED:
        # page is completely empty
        return 0

    var i = 0

    while i < len:
        let nextIndex = min(i + stepSize, arr.len)
        let lastIndex = nextIndex - 1

        if arr[lastIndex] != Mask.UNUSED:
            # if the last element in the step is used, we can skip `stepSize`
            i = nextIndex
            continue

        # last element is unused, therefore we need to check
        for j in i..lastIndex:
            if arr[j] == Mask.UNUSED:
                return j

        i = nextIndex

    return 0

proc putPage(page: BaseNDArray, infos: var Table[string, nimpy.PyObject], colName: string, col: ColSliceInfo): void {.inline.} =
    let (dir, pid) = col

    infos[colName] = newPyPage(pid, string dir, page.len, page.getPageTypes())

proc finalizeSlice(indices: var seq[int], column_names: seq[string], infos: var Table[string, nimpy.PyObject], cast_paths: var Table[string, (Path, Path, bool)], pages: var seq[(string, nimpy.PyObject)], result_info: ColInfo): void =
    if indices.len == 0:
        return

    for col_name in column_names:
        let (src_path, dst_path, is_tmp) = cast_paths[col_name]
        var cast_data = readNumpy(string src_path)

        if cast_data.len != indices.len:
            cast_data = cast_data[indices]
            cast_data.putPage(infos, col_name, result_info[col_name])
            cast_data.save(string dst_path)
        elif src_path != dst_path and is_tmp:
            moveFile(string src_path, string dst_path)

        pages.add((col_name, infos[col_name]))

proc toColSliceInfo(path: Path): ColSliceInfo =
    let workdir = path.parentDir.parentDir
    let pid = parseInt(string path.extractFilename.changeFileExt(""))

    return (workdir, pid)

proc doSliceConvert(dir_pid: Path, page_size: int, columns: Table[string, string], reject_reason_name: string, res_pass: ColInfo, res_fail: ColInfo, desired_column_map: OrderedTable[string, DesiredColumnInfo], column_names: seq[string], is_correct_type: Table[string, bool]): (seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)]) =
    var cast_paths_pass = initTable[string, (Path, Path, bool)]()
    var cast_paths_fail = initTable[string, (Path, Path, bool)]()
    var page_infos_pass = initTable[string, nimpy.PyObject]()
    var page_infos_fail = initTable[string, nimpy.PyObject]()
    var pages_pass = newSeq[(string, nimpy.PyObject)]()
    var pages_fail = newSeq[(string, nimpy.PyObject)]()

    try:
        let page_paths = collect(initTable()):
            for (key, path) in columns.pairs:
                {key: path}

        let workdir = dir_pid / Path("processing")

        createDir(string workdir)

        var valid_mask = newSeq[Mask](page_size)
        var reason_lst = newSeq[string](page_size)

        for (k, v) in page_paths.pairs:
            let (wd, pid) = res_fail[k]
            cast_paths_fail[k] = (Path v, wd / Path("pages") / Path($pid & ".npy"), false)

        let (rj_wd, rj_pid) = res_fail[reject_reason_name]
        let reject_reason_path = rj_wd / Path("pages") / Path($rj_pid & ".npy")
        cast_paths_fail[reject_reason_name] = (reject_reason_path, reject_reason_path, false)

        for (desired_name, desired_info) in desired_column_map.pairs:
            let original_name = desired_info.original_name
            let original_path = Path page_paths[original_name]
            let sz_data = getPageLen(string original_path)
            let original_data = readNumpy(string original_path)

            assert valid_mask.len >= sz_data, "Invalid mask size"

            let already_cast = is_correct_type[desired_name]

            original_data.putPage(page_infos_fail, original_name, original_path.toColSliceInfo)

            if already_cast:
                # we already know the type, just set the mask
                for i in 0..<sz_data:
                    if valid_mask[i] == INVALID:
                        continue

                    valid_mask[i] = VALID

                cast_paths_pass[desired_name] = (original_path, original_path, false)
                original_data.putPage(page_infos_pass, desired_name, original_path.toColSliceInfo)
                continue

            var cast_path: Path
            var path_exists = true

            while path_exists:
                cast_path = workdir / Path(generate_random_string(5) & ".npy")
                path_exists = fileExists(string cast_path)

            let (workdir, pid) = res_pass[desired_name]
            let pagedir = workdir / Path("pages")
            let dst_path = pagedir / Path($pid & ".npy")

            cast_paths_pass[desired_name] = (cast_path, dst_path, true)

            let desired_type = desired_info.`type`
            let allow_empty = desired_info.allow_empty

            var converted_page: BaseNDArray

            template castPage(T: typedesc) = T(original_data).convertBasicPage(
                desired_type, valid_mask, reason_lst, allow_empty,
                original_name, desired_name
            )

            case original_data.kind:
            of K_BOOLEAN: converted_page = BooleanNDArray.castPage
            of K_INT8: converted_page = Int8NDArray.castPage
            of K_INT16: converted_page = Int16NDArray.castPage
            of K_INT32: converted_page = Int32NDArray.castPage
            of K_INT64: converted_page = Int64NDArray.castPage
            of K_FLOAT32: converted_page = Float32NDArray.castPage
            of K_FLOAT64: converted_page = Float64NDArray.castPage
            of K_UNICODE: converted_page = UnicodeNDArray.castPage
            of K_DATE: converted_page = DateNDArray.castPage
            of K_DATETIME: converted_page = DateTimeNDArray.castPage
            of K_OBJECT: converted_page = ObjectNDArray.castPage

            converted_page.putPage(page_infos_pass, desired_name, res_pass[desired_name])
            converted_page.save(string cast_path)

        var mask_slice = 0..<unusedMaskSearch(valid_mask)

        valid_mask = valid_mask[mask_slice]

        var invalid_indices = newSeqOfCap[int](valid_mask.len shr 2) # quarter seems okay
        var valid_indices = newSeqOfCap[int](valid_mask.len - (valid_mask.len shr 2))

        reason_lst = collect:
            for (i, m) in enumerate(valid_mask):
                if m != Mask.INVALID:
                    valid_indices.add(i)
                    continue

                invalid_indices.add(i)
                reason_lst[i]

        valid_indices.finalizeSlice(toSeq(desired_column_map.keys), page_infos_pass, cast_paths_pass, pages_pass, res_pass)
        invalid_indices.finalizeSlice(toSeq(columns.keys), page_infos_fail, cast_paths_fail, pages_fail, res_fail)

        if reason_lst.len > 0:
            let (dirpid, pid) = res_fail[reject_reason_name]
            let pathpid = string (dirpid / Path("pages") / Path($pid & ".npy"))
            let page = newNDArray(reason_lst)

            page.save(pathpid)
            page.putPage(page_infos_fail, reject_reason_name, res_fail[reject_reason_name])

            pages_fail.add((reject_reason_name, page_infos_fail[reject_reason_name]))

    finally:
        for (cast_path, _, is_tmp) in cast_paths_pass.values:
            if not is_tmp:
                continue
            discard tryRemoveFile(string cast_path)

    return (pages_pass, pages_fail)

proc doSliceConvertPy(): string {.exportpy.} =
    return "slice converted"

proc columnSelect(table: nimpy.PyObject, cols: nimpy.PyObject, tqdm: nimpy.PyObject, dir_pid: Path, TaskManager: nimpy.PyObject): (nimpy.PyObject, nimpy.PyObject) =
    var desired_column_map = initOrderedTable[string, DesiredColumnInfo]()
    var collisions = initTable[string, int]()

    let dirpage = dir_pid / Path("pages")
    createDir(string dirpage)

    ######################################################
    # 1. Figure out what user needs (types, column names)
    ######################################################
    for c in cols: # now lets iterate over all given columns
        # this is our old name
        let name_inp = c["column"].to(string)
        var rename = c.get("rename", builtins().None)

        if rename.isNone() or not builtins().isinstance(rename, builtins().str).to(bool) or builtins().len(rename).to(int) == 0:
            rename = builtins().None
        else:
            let name_out_stripped = rename.strip()
            if builtins().len(rename).to(int) > 0 and builtins().len(name_out_stripped).to(int) == 0:
                raise newException(ValueError, "Validating 'column_select' failed, '" & name_inp & "' cannot be whitespace.")

            rename = name_out_stripped

        var name_out = if rename.isNone(): name_inp else: rename.to(string)

        if name_out in collisions: # check if the new name has any collision, add suffix if necessary
            collisions[name_out] = collisions[name_out] + 1
            name_out = name_out & "_" & $(collisions[name_out] - 1)
        else:
            collisions[name_out] = 1

        let desired_type = c.get("type", builtins().None)

        desired_column_map[name_out] = DesiredColumnInfo( # collect the information about the column, fill in any defaults
            original_name: name_inp,
            `type`: if desired_type.isNone(): K_NONETYPE else: toPageType(desired_type.to(string)),
            allow_empty: c.get("allow_empty", builtins().False).to(bool)
        )

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

    let column_names = collect: (for k in columns.keys: k)
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

    let page_count = layout_set[0][1]

    # Registry of data
    var passed_column_data = newSeq[string]()
    var failed_column_data = newSeq[string]()

    var cols = initTable[string, seq[string]]()

    var res_cols_pass = newSeqOfCap[ColInfo](page_count-1)
    var res_cols_fail = newSeqOfCap[ColInfo](page_count-1)

    for _ in 0..<page_count:
        res_cols_pass.add(initTable[string, ColSliceInfo]())
        res_cols_fail.add(initTable[string, ColSliceInfo]())

    var is_correct_type = initTable[string, bool]()

    proc genpage(dirpid: Path): ColSliceInfo {.inline.} = (dir_pid, tabliteBase().SimplePage.next_id(string dir_pid).to(int))

    for (desired_name_non_unique, desired_columns) in desired_column_map.pairs():
        let keys = toSeq(passed_column_data)
        let desired_name = unique_name(desired_name_non_unique, keys)
        let this_col = columns[desired_columns.original_name]

        cols[desired_name] = this_col

        passed_column_data.add(desired_name)

        var col_dtypes = toSeq(this_col.getColumnTypes().keys)
        var needs_to_iterate = false

        if K_NONETYPE in col_dtypes:
            if not desired_columns.allow_empty:
                needs_to_iterate = true
            else:
                col_dtypes.delete(col_dtypes.find(K_NONETYPE))

        if not needs_to_iterate and col_dtypes.len > 0:
            if col_dtypes.len > 1:
                # multiple times always needs to cast
                needs_to_iterate = true
            else:
                let active_type = col_dtypes[0]

                if active_type != desired_columns.`type`:
                    # not same type, need to cast
                    needs_to_iterate = true
                elif active_type == K_STRING and not desired_columns.allow_empty:
                    # we may have same type but we still need to filter empty strings
                    needs_to_iterate = true

        is_correct_type[desired_name] = not needs_to_iterate

        for i in 0..<page_count:
            res_cols_pass[i][desired_name] = genpage(dir_pid)

    for desired_name in columns.keys:
        failed_column_data.add(desired_name)

        for i in 0..<page_count:
            res_cols_fail[i][desired_name] = genpage(dir_pid)

    let reject_reason_name = unique_name("reject_reason", column_names)

    for i in 0..<page_count:
        res_cols_fail[i][reject_reason_name] = genpage(dir_pid)

    failed_column_data.add(reject_reason_name)

    if toSeq(is_correct_type.values).all(proc (x: bool): bool = x):
        let tbl_pass_columns = collect(initTable()):
            for (desired_name, desired_info) in desired_column_map.pairs():
                {desired_name: table[desired_info.original_name]}

        let tbl_fail_columns = collect(initTable()):
            for desired_name in failed_column_data:
                {desired_name: newSeq[nimpy.PyObject]()}

        let tbl_pass = tablite().Table(columns = tbl_pass_columns)
        let tbl_fail = tablite().Table(columns = tbl_fail_columns)

        return (tbl_pass, tbl_fail)

    template ordered2PyDict(keys: seq[string]): nimpy.PyObject =
        let dict = pymodules.builtins().dict()

        for k in keys:
            dict[k] = newSeq[nimpy.PyObject]()

        dict

    var tbl_pass = tablite().Table(columns = passed_column_data.ordered2PyDict())
    var tbl_fail = tablite().Table(columns = failed_column_data.ordered2PyDict())

    var task_list_inp = collect:
        for i in 0..<page_count:
            let el = collect(initTable()):
                for (name, column) in columns.pairs:
                    {name: column[i]}
            (el, res_cols_pass[i], res_cols_fail[i])

    let cpu_count = clamp(task_list_inp.len, 1, countProcessors())
    let Config = tabliteConfig().Config
    var is_mp: bool

    if Config.MULTIPROCESSING_MODE.to(string) == Config.FORCE.to(string):
        is_mp = true
    elif Config.MULTIPROCESSING_MODE.to(string) == Config.FALSE.to(string):
        is_mp = false
    elif Config.MULTIPROCESSING_MODE.to(string) == Config.AUTO.to(string):
        let is_multithreaded = cpu_count > 1
        let is_multipage = page_count > 1

        is_mp = is_multithreaded and is_multipage

    var page_size = Config.PAGE_SIZE.to(int)
    var pbar = tqdm!(total: task_list_inp.len, desc: "column select")
    var converted = newSeqOfCap[(seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)])](task_list_inp.len)

    if is_mp:
        echo appType
        when appType != "lib":
            implement("cannot execute in app mode")
        else:
            template colInfoToPy(tbl: ColInfo): Table[string, (string, int)] =
                collect(initTable()):
                    for (cn, ct) in tbl.pairs:
                        let (cp, ci) = ct
                        {cn: (string cp, ci)}

            let tasks = collect:
                for (columns, res_pass, res_fail) in task_list_inp:
                    let pyResPass = colInfoToPy(res_pass)
                    let pyResFail = colInfoToPy(res_fail)
                    let ipkl = dill().dumps(doSliceConvertPy)

                    mplite().Task(tabliteBase().capsuleIntermediate, ipkl, string dir_pid, page_size, columns, reject_reason_name, pyResPass, pyResFail, desired_column_map, column_names, is_correct_type)

            # echo tasks

            let tm = TaskManager!(cpu_count=cpu_count)
            
            discard tm.start()

            let results = tm.execute(tasks)#, pbar=pbar)

            discard tm.stop()

            echo results
            implement("columnSelect.convert.mp")
    else:
        for (columns, res_pass, res_fail) in task_list_inp:
            converted.add(doSliceConvert(dir_pid, page_size, columns, reject_reason_name, res_pass, res_fail, desired_column_map, column_names, is_correct_type))

        discard pbar.update(1)

    proc extendTable(table: var nimpy.PyObject, columns: seq[(string, nimpy.PyObject)]): void {.inline.} =
        for (col_name, pg) in columns:
            let col = table[col_name]

            discard col.pages.append(pg) # can't col.extend because nim is dumb :)

    for (pg_pass, pg_fail) in converted:
        tbl_pass.extendTable(pg_pass)
        tbl_fail.extendTable(pg_fail)

    return (tbl_pass, tbl_fail)

when isMainModule and appType != "lib":
    proc newColumnSelectorInfo(column: string, `type`: string, allow_empty: bool, rename: opt.Option[string]): nimpy.PyObject =
        let pyDict = builtins().dict(
            column = column,
            type = `type`,
            allow_empty = allow_empty
        )

        if rename.isNone():
            pyDict["rename"] = nil
        else:
            pyDict["rename"] = rename.get()

        return pyDict

    let workdir = Path(pymodules.builtins().str(pymodules.tabliteConfig().Config.workdir).to(string))
    let pid = "nim"
    let pagedir = workdir / Path(pid) / Path("pages")

    createDir(string pagedir)

    pymodules.tabliteConfig().Config.pid = pid
    pymodules.tabliteConfig().Config.PAGE_SIZE = 2
    # pymodules.tabliteConfig().Config.MULTIPROCESSING_MODE = pymodules.tabliteConfig().Config.FALSE

    # let columns = pymodules.builtins().dict({"A ": @[nimValueToPy(0), nimValueToPy(nil), nimValueToPy(10), nimValueToPy(200)]}.toTable)
    # let columns = pymodules.builtins().dict({"A ": @[1, 22, 333]}.toTable)
    # let columns = pymodules.builtins().dict({"A ": @["1", "22", "333", "", "abc"]}.toTable)
    let columns = pymodules.builtins().dict({"A ": @[nimValueToPy("1"), nimValueToPy("222"), nimValueToPy("333"), nimValueToPy(nil), nimValueToPy("abc")]}.toTable)
    let table = pymodules.tablite().Table(columns = columns)

    discard table.show(dtype = true)

    let select_cols = builtins().list(@[
        newColumnSelectorInfo("A ", "int", true, opt.none[string]()),
        newColumnSelectorInfo("A ", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("A ", "bool", false, opt.none[string]()),
            # newColumnSelectorInfo("A ", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("A ", "date", false, opt.none[string]()),
            # newColumnSelectorInfo("A ", "datetime", false, opt.none[string]()),
            # newColumnSelectorInfo("A ", "time", false, opt.none[string]()),
    ])

    let (select_pass, select_fail) = table.columnSelect(
        select_cols,
        nimpy.pyImport("tqdm").tqdm,
        dir_pid = workdir / Path(pid),
        Taskmanager = mplite().TaskManager
    )

    discard select_pass.show(dtype = true)
    discard select_fail.show(dtype = true)

    implement("keep pages")
