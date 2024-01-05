import std/[os, tables, sugar, sets, sequtils, strutils, cpuinfo, paths, enumerate, unicode, times, macros]
import nimpy as nimpy
from nimpyext import `!`
import utils
import std/options as opt
from std/math import clamp
import pymodules as pymodules
import pytypes
import numpy
from unpickling import PageTypes
import typetraits
import dateutils

type ColSliceInfo = (Path, int)
type ColInfo = Table[string, ColSliceInfo]
type DesiredColumnInfo = object
    original_name: string
    `type`: PageTypes
    allow_empty: bool
type Mask = enum
    INVALID = -1
    UNUSED = 0
    VALID = 1

proc toPageType(name: string): PageTypes =
    case name.toLower():
    of "int": return PageTypes.DT_INT
    of "float": return PageTypes.DT_FLOAT
    of "bool": return PageTypes.DT_BOOL
    of "str": return PageTypes.DT_STRING
    of "date": return PageTypes.DT_DATE
    of "time": return PageTypes.DT_TIME
    of "datetime": return PageTypes.DT_DATETIME
    else: raise newException(FieldDefect, "unsupported page type: '" & name & "'")

template makePage[T: typed](dt: typedesc[T], page: typed, mask: var seq[Mask], reason_lst: var seq[string], conv: proc): T =
    when T is UnicodeNDArray:
        var longest = 1
        let strings = collect:
            for (i, v) in enumerate(page.buf):
                var res: seq[Rune]
                try:
                    let str = conv(v)
                    res = str.toRunes

                    longest = max(longest, res.len)
                    mask[i] = Mask.VALID
                except ValueError:
                    mask[i] = Mask.INVALID
                    reason_lst[i] = "Cannot cast"

                    res = newSeq[Rune]()

                res

        let buf = newSeq[Rune](longest * page.len)

        for (i, str) in enumerate(strings):
            buf[i * longest].addr.copyMem(addr str[0], str.len * sizeof(Rune))

        T(shape: page.shape, buf: buf, size: longest)
    else:
        let buf = collect:
            for (i, v) in enumerate(page.buf):
                var res = dt.default()

                try:
                    res = conv(v)
                    mask[i] = Mask.VALID
                except ValueError:
                    mask[i] = Mask.INVALID
                    reason_lst[i] = "Cannot cast"

                res

        T(shape: page.shape, buf: buf)

type ToDate = object
type ToDateTime = object
type ToTime = object
type CastablePrimitives = bool | int | float | string | ToDate | ToDateTime
type CastedPrimitives = bool | int | float | string | DateTime

macro mkCaster(caster: untyped) =
    expectKind(caster, nnkLambda)
    expectLen(caster.params, 2)

    let castR = caster.params[0]
    let castT = caster.params[1][1]

    let descT = newNimNode(nnkBracketExpr)
    let paramsT = newIdentDefs(newIdentNode("T"), descT)

    var trueCastR: NimNode

    if castR.strVal in [ToDate.name, ToDateTime.name]:
        trueCastR = newIdentNode(DateTime.name)
    elif castR.strVal == ToTime.name:
        trueCastR = newIdentNode(PY_Time.name)
    elif castR.strVal in [bool.name, int.name, float.name, string.name]:
        trueCastR = castR
    else:
        raise newException(FieldDefect, "Uncastable type '" & $castR.strVal & "'")

    descT.add(newIdentNode("typedesc"))
    descT.add(castT)

    let descR = newNimNode(nnkBracketExpr)
    let paramsR = newIdentDefs(newIdentNode("R"), descR)

    descR.add(newIdentNode("typedesc"))
    descR.add(castR)

    let subProc = newProc(params=[trueCastR, caster.params[1]], body=caster.body, procType=nnkLambda)
    let makerProc = newProc(newIdentNode("fnCast"), params=[newNimNode(nnkProcTy), paramsT, paramsR], body=subProc)

    return makerProc

macro tdesc(_: typedesc[BooleanNDArray]): typedesc[bool] = bindSym("bool")
macro tdesc(_: typedesc[Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray]): typedesc[int] = bindSym("int")
macro tdesc(_: typedesc[Float32NDArray | Float64NDArray]): typedesc[float] = bindSym("float")

macro pdesc(_: typedesc[bool]): typedesc[BooleanNDArray] = bindSym("BooleanNDArray")
macro pdesc(_: typedesc[int]): typedesc[Int64NDArray] = bindSym("Int64NDArray")
macro pdesc(_: typedesc[float]): typedesc[Float64NDArray] = bindSym("Float64NDArray")
macro pdesc(_: typedesc[ToDate]): typedesc[DateNDArray] = bindSym("DateNDArray")
macro pdesc(_: typedesc[ToDateTime]): typedesc[DateTimeNDArray] = bindSym("DateTimeNDArray")

mkCaster proc(v: bool): bool = v
mkCaster proc(v: bool): int = int v
mkCaster proc(v: bool): float = float v
mkCaster proc(v: bool): string = (if v: "True" else: "False")
mkCaster proc(v: bool): ToDate = days2Date(int v)
mkCaster proc(v: bool): ToDateTime = delta2Date(seconds=int v)
mkCaster proc(v: bool): ToTime = implement("bool2time")

mkCaster proc(v: int): bool = v >= 1
mkCaster proc(v: int): int = v
mkCaster proc(v: int): float = float v
mkCaster proc(v: int): string = $v
mkCaster proc(v: int): ToDate = v.days2Date
mkCaster proc(v: int): ToDateTime = delta2Date(seconds=v)
mkCaster proc(v: int): ToTime = implement("bool2time")

mkCaster proc(v: float): bool = v >= 1
mkCaster proc(v: float): int = int v
mkCaster proc(v: float): float = v
mkCaster proc(v: float): string = $v
mkCaster proc(v: float): ToDate = days2Date(int v)
mkCaster proc(v: float): ToDateTime = implement("float2datetime")
mkCaster proc(v: float): ToTime = implement("bool2time")

# TODO: turn into macro
proc castType[T: BooleanNDArray](R: typedesc[bool], page: T, mask: var seq[Mask], reason_lst: var seq[string]): BooleanNDArray = page
proc castType[T: BooleanNDArray](R: typedesc[int], page: T, mask: var seq[Mask], reason_lst: var seq[string]): Int64NDArray = Int64NDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: BooleanNDArray](R: typedesc[float], page: T, mask: var seq[Mask], reason_lst: var seq[string]): Float64NDArray = Float64NDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: BooleanNDArray](R: typedesc[string], page: T, mask: var seq[Mask], reason_lst: var seq[string]): UnicodeNDArray = UnicodeNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: BooleanNDArray](R: typedesc[ToDate], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateNDArray = DateNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: BooleanNDArray](R: typedesc[ToDateTime], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateTimeNDArray = DateTimeNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: BooleanNDArray](R: typedesc[ToTime], page: T, mask: var seq[Mask], reason_lst: var seq[string]): ObjectNDArray = ObjectNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))

proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](R: typedesc[bool], page: T, mask: var seq[Mask], reason_lst: var seq[string]): BooleanNDArray = BooleanNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](R: typedesc[int], page: T, mask: var seq[Mask], reason_lst: var seq[string]): T = page
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](R: typedesc[float], page: T, mask: var seq[Mask], reason_lst: var seq[string]): Float64NDArray = Float64NDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](R: typedesc[string], page: T, mask: var seq[Mask], reason_lst: var seq[string]): UnicodeNDArray = UnicodeNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](R: typedesc[ToDate], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateNDArray = DateNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](R: typedesc[ToDateTime], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateTimeNDArray = DateTimeNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](R: typedesc[ToTime], page: T, mask: var seq[Mask], reason_lst: var seq[string]): ObjectNDArray = ObjectNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))

proc castType[T: Float32NDArray | Float64NDArray](R: typedesc[bool], page: T, mask: var seq[Mask], reason_lst: var seq[string]): BooleanNDArray = BooleanNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Float32NDArray | Float64NDArray](R: typedesc[int], page: T, mask: var seq[Mask], reason_lst: var seq[string]): Int64NDArray = Int64NDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Float32NDArray | Float64NDArray](R: typedesc[float], page: T, mask: var seq[Mask], reason_lst: var seq[string]): T = page
proc castType[T: Float32NDArray | Float64NDArray](R: typedesc[string], page: T, mask: var seq[Mask], reason_lst: var seq[string]): UnicodeNDArray = UnicodeNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Float32NDArray | Float64NDArray](R: typedesc[ToDate], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateNDArray = DateNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Float32NDArray | Float64NDArray](R: typedesc[ToDateTime], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateTimeNDArray = DateTimeNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))
proc castType[T: Float32NDArray | Float64NDArray](R: typedesc[ToTime], page: T, mask: var seq[Mask], reason_lst: var seq[string]): ObjectNDArray = ObjectNDArray.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))

# template castType[T](R: typedesc, page: T, mask: var seq[Mask], reason_lst: var seq[string]) =
#     let pgType = R.pdesc
#     pgType.makePage(page, mask, reason_lst, T.tdesc.fnCast(R))

template convertBasicPage[T](page: T, desired: PageTypes, mask: var seq[Mask], reason_lst: var seq[string]): BaseNDArray =
    case desired:
    of DT_BOOL: bool.castType(page, mask, reason_lst)
    of DT_INT: int.castType(page, mask, reason_lst)
    of DT_FLOAT: float.castType(page, mask, reason_lst)
    of DT_STRING: string.castType(page, mask, reason_lst)
    of DT_DATE: ToDate.castType(page, mask, reason_lst)
    of DT_TIME: ToTime.castType(page, mask, reason_lst)
    of DT_DATETIME: ToDateTime.castType(page, mask, reason_lst)
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
        let nextIndex = i + stepSize
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

proc finalizeSlice(indices: var seq[int], column_names: seq[string], infos: var Table[string, nimpy.PyObject], cast_paths: var Table[string, (Path, bool)], pages: var seq[(string, nimpy.PyObject)], result_info: ColInfo): void =
    if indices.len == 0:
        return

    for col_name in column_names:
        let (cast_path, is_tmp) = cast_paths[col_name]

        if not is_tmp:
            pages.add((col_name, infos[col_name]))
            continue

        var cast_data = readNumpy(string cast_path)
        let (dirpid, pid) = result_info[col_name]
        let dirpage = dirpid / Path("pages")

        createDir(string dirpage)

        let pathpid = string (dirpage / Path($pid & ".npy"))

        if cast_data.len != indices.len:
            cast_data = cast_data[indices]

            cast_data.save(pathpid)
        else:
            moveFile(string cast_path, pathpid)

        pages.add((col_name, infos[col_name]))

proc putPage(page: BaseNDArray, infos: var Table[string, nimpy.PyObject], colName: string, col: ColSliceInfo): void {.inline.} =
    let (dir, pid) = col
    let pg = newPyPage(pid, string dir, page.len, page.getPageTypes())

    infos[colName] = pg

proc toColSliceInfo(path: Path): ColSliceInfo =
    let workdir = path.parentDir.parentDir
    let pid = parseInt(string path.extractFilename.changeFileExt(""))

    return (workdir, pid)

proc doSliceConvert(dir_pid: Path, page_size: int, columns: Table[string, string], reject_reason_name: string, res_pass: ColInfo, res_fail: ColInfo, desired_column_map: OrderedTable[string, DesiredColumnInfo], column_names: seq[string], is_correct_type: Table[string, bool]): (seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)]) =
    var cast_paths = initTable[string, (Path, bool)]()
    var page_infos = initTable[string, nimpy.PyObject]()
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

        for (desired_name, desired_info) in desired_column_map.pairs:
            let original_name = desired_info.original_name
            let original_path = page_paths[original_name]
            let sz_data = getPageLen(original_path)

            assert valid_mask.len >= sz_data, "Invalid mask size"

            let already_cast = is_correct_type[desired_name]

            if already_cast:
                # we already know the type, just set the mask
                for i in 0..<sz_data:
                    if valid_mask[i] == INVALID:
                        continue

                    valid_mask[i] = VALID

                cast_paths[desired_name] = (Path(original_path), false)
                continue

            var cast_path: Path
            var path_exists = true

            while path_exists:
                cast_path = workdir / Path(generate_random_string(5) & ".npy")
                path_exists = fileExists(string cast_path)

            cast_paths[desired_name] = (cast_path, true)

            let original_data = readNumpy(original_path)

            let desired_type = desired_info.`type`
            let allow_empty = desired_info.allow_empty

            var converted_page: BaseNDArray

            original_data.putPage(page_infos, original_name, Path(original_path).toColSliceInfo())

            if original_data of ObjectNDArray:
                # may contain mixed types or nones
                discard
            else:
                if original_data of BooleanNDArray:
                    converted_page = BooleanNDArray(original_data).convertBasicPage(desired_type, valid_mask, reason_lst)
                elif original_data of Int8NDArray:
                    converted_page = Int8NDArray(original_data).convertBasicPage(desired_type, valid_mask, reason_lst)
                elif original_data of Int16NDArray:
                    converted_page = Int16NDArray(original_data).convertBasicPage(desired_type, valid_mask, reason_lst)
                elif original_data of Int32NDArray:
                    converted_page = Int32NDArray(original_data).convertBasicPage(desired_type, valid_mask, reason_lst)
                elif original_data of Int64NDArray:
                    converted_page = Int64NDArray(original_data).convertBasicPage(desired_type, valid_mask, reason_lst)
                elif original_data of Float32NDArray:
                    converted_page = Float32NDArray(original_data).convertBasicPage(desired_type, valid_mask, reason_lst)
                elif original_data of Float64NDArray:
                    converted_page = Float64NDArray(original_data).convertBasicPage(desired_type, valid_mask, reason_lst)
                elif original_data of UnicodeNDArray:
                    implement("doSliceConvert.UnicodeNDArray")
                elif original_data of DateNDArray:
                    implement("doSliceConvert.DateNDArray")
                elif original_data of DateTimeNDArray:
                    implement("doSliceConvert.DateTimeNDArray")
                else:
                    corrupted()

            converted_page.putPage(page_infos, desired_name, res_pass[desired_name])
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

        valid_indices.finalizeSlice(toSeq(desired_column_map.keys), page_infos, cast_paths, pages_pass, res_pass)
        invalid_indices.finalizeSlice(toSeq(columns.keys), page_infos, cast_paths, pages_fail, res_fail)

        if reason_lst.len > 0:
            let (dirpid, pid) = res_fail[reject_reason_name]
            let pathpid = string (dirpid / Path($pid & ".npy"))
            newNDArray(reason_lst).save(pathpid)

            pages_fail.add((reject_reason_name, page_infos[reject_reason_name]))

    finally:
        for (cast_path, is_tmp) in cast_paths.values:
            if not is_tmp:
                continue
            discard tryRemoveFile(string cast_path)

    return (pages_pass, pages_fail)

proc columnSelect(table: nimpy.PyObject, cols: nimpy.PyObject, tqdm: nimpy.PyObject, dir_pid: Path): (nimpy.PyObject, nimpy.PyObject) =
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
            `type`: if desired_type.isNone(): PageTypes.DT_NONE else: toPageType(desired_type.to(string)),
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

        if PageTypes.DT_NONE in col_dtypes:
            if not desired_columns.allow_empty:
                needs_to_iterate = true
            else:
                col_dtypes.delete(col_dtypes.find(PageTypes.DT_NONE))

        if not needs_to_iterate and col_dtypes.len > 0:
            if col_dtypes.len > 1:
                needs_to_iterate = true
            else:
                let active_type = col_dtypes[0]

                if active_type != desired_columns.`type`:
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
        var tbl_pass_columns = collect(initTable()):
            for (desired_name, desired_info) in desired_column_map.pairs():
                {desired_name: table[desired_info.original_name]}

        let tbl_pass = tablite().Table(columns = tbl_pass_columns)
        let tbl_fail = tablite().Table(columns = failed_column_data)

        return (tbl_pass, tbl_fail)

    template ordered2PyDict(keys: seq[string]): nimpy.PyObject =
        let dict = pymodules.builtins().dict()

        for k in keys:
            dict[k] = newSeq[nimpy.PyObject]()

        dict

    var tbl_pass = tablite().Table(columns = passed_column_data.ordered2PyDict())
    var tbl_fail = tablite().Table(columns = failed_column_data.ordered2PyDict())

    echo ">>>tbl_pass.keys: " & $passed_column_data

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

    let columns = pymodules.builtins().dict({"A ": @[0, 10, 200]}.toTable)
    let table = pymodules.tablite().Table(columns = columns)

    let select_cols = builtins().list(@[
        newColumnSelectorInfo("A ", "int", false, opt.none[string]()),
        newColumnSelectorInfo("A ", "float", false, opt.none[string]()),
        newColumnSelectorInfo("A ", "str", false, opt.none[string]()),
        newColumnSelectorInfo("A ", "date", false, opt.none[string]()),
        newColumnSelectorInfo("A ", "datetime", false, opt.none[string]()),
        newColumnSelectorInfo("A ", "time", false, opt.none[string]()),
    ])

    let (select_pass, select_fail) = table.columnSelect(
        select_cols,
        nimpy.pyImport("tqdm").tqdm,
        dir_pid = workdir / Path(pid)
    )

    discard select_pass.show()
    discard select_fail.show()

