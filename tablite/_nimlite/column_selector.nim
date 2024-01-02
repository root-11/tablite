import std/[os, tables, sugar, sets, sequtils, strutils, cpuinfo, paths, enumerate, unicode]
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


type ColInfo = Table[string, (Path, int)]
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
                    res = conv(v).toRunes

                    longest = max(longest, res.len)
                    mask[i] = Mask.VALID
                except ValueError:
                    mask[i] = Mask.INVALID
                    reason_lst[i] = "Cannot cast"

                    res = newSeq[Rune]()

                res

        let buf = newSeq[Rune](longest * page.len)

        for (i, str) in enumerate(strings):
            buf[i * longest].addr.copyMem(addr str[0], str.len)

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

proc castType(_: typedesc[PY_Boolean], page: BooleanNDArray, mask: var seq[Mask], reason_lst: var seq[string]): BooleanNDArray = page
proc castType(_: typedesc[PY_Int], page: BooleanNDArray, mask: var seq[Mask], reason_lst: var seq[string]): Int64NDArray = implement("bool2int")
proc castType(_: typedesc[PY_Float], page: BooleanNDArray, mask: var seq[Mask], reason_lst: var seq[string]): Float64NDArray = implement("bool2float")
proc castType(_: typedesc[PY_String], page: BooleanNDArray, mask: var seq[Mask], reason_lst: var seq[string]): UnicodeNDArray = implement("bool2str")
proc castType(_: typedesc[PY_Date], page: BooleanNDArray, mask: var seq[Mask], reason_lst: var seq[string]): DateNDArray = implement("bool2date")
proc castType(_: typedesc[PY_Time], page: BooleanNDArray, mask: var seq[Mask], reason_lst: var seq[string]): ObjectNDArray = implement("bool2time")
proc castType(_: typedesc[PY_DateTime], page: BooleanNDArray, mask: var seq[Mask], reason_lst: var seq[string]): DateTimeNDArray = implement("bool2datetime")

proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](_: typedesc[PY_Boolean], page: T, mask: var seq[Mask], reason_lst: var seq[string]): BooleanNDArray = BooleanNDArray.makePage(page, mask, reason_lst, proc(v: int): bool = v >= 1)
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](_: typedesc[PY_Int], page: T, mask: var seq[Mask], reason_lst: var seq[string]): T = page
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](_: typedesc[PY_Float], page: T, mask: var seq[Mask], reason_lst: var seq[string]): Float64NDArray = Float64NDArray.makePage(page, mask, reason_lst, proc(v: int): float = float v)
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](_: typedesc[PY_String], page: T, mask: var seq[Mask], reason_lst: var seq[string]): UnicodeNDArray = UnicodeNDArray.makePage(page, mask, reason_lst, proc(v: int): string = $v)
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](_: typedesc[PY_Date], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateNDArray = implement("int2date")
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](_: typedesc[PY_Time], page: T, mask: var seq[Mask], reason_lst: var seq[string]): ObjectNDArray = implement("int2time")
proc castType[T: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray](_: typedesc[PY_DateTime], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateTimeNDArray = implement("int2datetime")

proc castType[T: Float32NDArray | Float64NDArray](_: typedesc[PY_Boolean], page: T, mask: var seq[Mask], reason_lst: var seq[string]): BooleanNDArray = implement("float2bool")
proc castType[T: Float32NDArray | Float64NDArray](_: typedesc[PY_Int], page: T, mask: var seq[Mask], reason_lst: var seq[string]): Int64NDArray = implement("float2int")
proc castType[T: Float32NDArray | Float64NDArray](_: typedesc[PY_Float], page: T, mask: var seq[Mask], reason_lst: var seq[string]): T = page
proc castType[T: Float32NDArray | Float64NDArray](_: typedesc[PY_String], page: T, mask: var seq[Mask], reason_lst: var seq[string]): UnicodeNDArray = implement("float2str")
proc castType[T: Float32NDArray | Float64NDArray](_: typedesc[PY_Date], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateNDArray = implement("float2date")
proc castType[T: Float32NDArray | Float64NDArray](_: typedesc[PY_Time], page: T, mask: var seq[Mask], reason_lst: var seq[string]): ObjectNDArray = implement("float2time")
proc castType[T: Float32NDArray | Float64NDArray](_: typedesc[PY_DateTime], page: T, mask: var seq[Mask], reason_lst: var seq[string]): DateTimeNDArray = implement("float2datetime")

template convertBasicPage[T](page: T, desired: PageTypes, mask: var seq[Mask], reason_lst: var seq[string]): BaseNDArray =
    case desired:
    of DT_BOOL: PY_Boolean.castType(page, mask, reason_lst)
    of DT_INT: PY_Int.castType(page, mask, reason_lst)
    of DT_FLOAT: PY_Float.castType(page, mask, reason_lst)
    of DT_STRING: PY_String.castType(page, mask, reason_lst)
    of DT_DATE: PY_Date.castType(page, mask, reason_lst)
    of DT_TIME: PY_Time.castType(page, mask, reason_lst)
    of DT_DATETIME: PY_DateTime.castType(page, mask, reason_lst)
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

proc doSliceConvert(dir_pid: Path, page_size: int, columns: Table[string, string], reject_reason_name: string, res_pass: ColInfo, res_fail: ColInfo, desired_column_map: Table[string, DesiredColumnInfo], column_names: seq[string], is_correct_type: Table[string, bool]): (seq[(string, string)], seq[(string, string)]) =
    var cast_paths = initTable[string, (Path, bool)]()
    var pages_pass = newSeq[(string, string)]()
    var pages_fail = newSeq[(string, string)]()

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

            converted_page.save(string cast_path)


        var mask_slice = 0..<unusedMaskSearch(valid_mask)
        
        valid_mask = valid_mask[mask_slice]
        reason_lst = reason_lst[mask_slice]

       
    finally:
        discard

    implement("doSliceConvert")

    return (pages_pass, pages_fail)

proc columnSelect(table: nimpy.PyObject, cols: nimpy.PyObject, tqdm: nimpy.PyObject, dir_pid: Path): (nimpy.PyObject, nimpy.PyObject) =
    var desired_column_map = initTable[string, DesiredColumnInfo]()
    var collisions = initTable[string, int]()

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
    var passed_column_data = initTable[string, seq[string]]()
    var failed_column_data = initTable[string, seq[string]]()

    var cols = initTable[string, seq[string]]()

    var res_cols_pass = newSeqOfCap[ColInfo](page_count-1)
    var res_cols_fail = newSeqOfCap[ColInfo](page_count-1)

    for _ in 0..<page_count:
        res_cols_pass.add(initTable[string, (Path, int)]())
        res_cols_fail.add(initTable[string, (Path, int)]())

    var is_correct_type = initTable[string, bool]()

    template genpage(dirpid: Path): (Path, int) = (dir_pid, tabliteBase().SimplePage.next_id(string dir_pid).to(int))

    for (desired_name_non_unique, desired_columns) in desired_column_map.pairs():
        let keys = passed_column_data.toSeqKeys()
        let desired_name = unique_name(desired_name_non_unique, keys)
        let this_col = columns[desired_columns.original_name]

        cols[desired_name] = this_col

        passed_column_data[desired_name] = @[]

        var col_dtypes = this_col.getColumnTypes().toSeqKeys()
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
            res_cols_pass[i][desired_name] = genpage(dirpid)

    for desired_name in desired_column_map.keys:
        failed_column_data[desired_name] = @[]

        for i in 0..<page_count:
            res_cols_fail[i][desired_name] = genpage(dirpid)

    let reject_reason_name = unique_name("reject_reason", column_names)

    for i in 0..<page_count:
        res_cols_fail[i][reject_reason_name] = genpage(dirpid)

    failed_column_data[reject_reason_name] = @[]

    if is_correct_type.toSeqValues().all(proc (x: bool): bool = x):
        var tbl_pass_columns = collect(initTable()):
            for (desired_name, desired_info) in desired_column_map.pairs():
                {desired_name: table[desired_info.original_name]}

        let tbl_pass = tablite().Table(columns = tbl_pass_columns)
        let tbl_fail = tablite().Table(columns = failed_column_data)

        return (tbl_pass, tbl_fail)

    var tbl_pass = tablite().Table(columns = passed_column_data)
    var tbl_fail = tablite().Table(columns = failed_column_data)

    var task_list_inp = collect:
        for i in 0..<page_count:
            let el = collect(initTable()):
                for (name, column) in columns.pairs:
                    {name: column[i]}
            (el, res_cols_pass[0], res_cols_fail[0])

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
    var result = newSeqOfCap[(seq[(string, string)], seq[(string, string)])](task_list_inp.len)

    if is_mp:
        implement("columnSelect.convert.mp")
    else:
        for (columns, res_pass, res_fail) in task_list_inp:
            result.add(doSliceConvert(dir_pid, page_size, columns, reject_reason_name, res_pass, res_fail, desired_column_map, column_names, is_correct_type))

        discard pbar.update(1)

    implement("columnSelect.convert")

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

    discard pyImport("sys").path.extend(@[
        "/home/ratchet/envs/callisto/lib/python3.10/site-packages",
        "/home/ratchet/Documents/dematic/tablite",
        "/home/ratchet/Documents/dematic/mplite"
    ])

    let columns = pymodules.builtins().dict({"A ": @[0, 1, 2]}.toTable)
    let table = pymodules.tablite().Table(columns = columns)

    let select_cols = builtins().list(@[
        newColumnSelectorInfo("A ", "float", false, opt.none[string]()),
        newColumnSelectorInfo("A ", "str", false, opt.none[string]())
    ])

    let (select_pass, select_fail) = table.columnSelect(
        select_cols,
        nimpy.pyImport("tqdm").tqdm,
        dir_pid = Path("/media/ratchet/hdd/tablite/nim")
    )

    echo select_pass.show()
