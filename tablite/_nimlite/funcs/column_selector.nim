
import column_selector/sliceconv
import column_selector/infos
import column_selector/collectinfo

export ColInfo
export toPyObj
export collectColumnSelectInfo
export doSliceConvert
export fromPyObjToDesiredInfos

when isMainModule and appType != "lib":

    import std/[os, tables, sugar, sets, sequtils, paths, macros]
    import nimpy as nimpy
    from ../nimpyext import `!`
    import std/options as opt
    import ../pymodules as pymodules
    import ../numpy
    import typetraits

    proc columnSelect(table: nimpy.PyObject, cols: nimpy.PyObject, tqdm: nimpy.PyObject, dir_pid: Path, TaskManager: nimpy.PyObject): (nimpy.PyObject, nimpy.PyObject) =
        # this is nim-only implementation, the library build doesn't need it because we need TaskManager to be used for slices
        var (columns, page_count, is_correct_type, desired_column_map, passed_column_data, failed_column_data, res_cols_pass, res_cols_fail, column_names, reject_reason_name) = collectColumnSelectInfo(table, cols, string dir_pid)

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

        var page_size = tabliteConfig().Config.PAGE_SIZE.to(int)
        var pbar = tqdm!(total: task_list_inp.len, desc: "column select")
        var converted = newSeqOfCap[(seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)])](task_list_inp.len)

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
    # pymodules.tabliteConfig().Config.PAGE_SIZE = 2
    # pymodules.tabliteConfig().Config.MULTIPROCESSING_MODE = pymodules.tabliteConfig().Config.FALSE

    # let columns = pymodules.builtins().dict({"A ": @[nimValueToPy(0), nimValueToPy(nil), nimValueToPy(10), nimValueToPy(200)]}.toTable)
    # let columns = pymodules.builtins().dict({"A ": @[1, 22, 333]}.toTable)
    # let columns = pymodules.builtins().dict({"A ": @["1", "22", "333", "", "abc"]}.toTable)
    # let columns = pymodules.builtins().dict({"A ": @[nimValueToPy("1"), nimValueToPy("222"), nimValueToPy("333"), nimValueToPy(nil), nimValueToPy("abc")]}.toTable)
    let columns = pymodules.builtins().dict({"A ": @[nimValueToPy(1), nimValueToPy(2.0), nimValueToPy("333"), nimValueToPy("abc")]}.toTable)
    # let columns = pymodules.builtins().dict({"A": @[nimValueToPy("0"), nimValueToPy(nil), nimValueToPy("2")], "B": @[nimValueToPy("3"), nimValueToPy(nil), nimValueToPy("4")]}.toTable)
    # let columns = pymodules.builtins().dict({"str": @["1", "0"]})
    # let columns = pymodules.builtins().dict({"float": @[1.0, 0.0]})
    # let columns = pymodules.builtins().dict({"date": @[
    #     datetime().date(2000, 1, 1),
    #     datetime().date(2000, 1, 2),
    # ]})
    # let columns = pymodules.builtins().dict({"str": @[nimValueToPy("abc"), nimValueToPy("efg"), nimValueToPy(nil)]}.toTable)
    let table = pymodules.tablite().Table(columns = columns)

    discard table.show(dtype = true)

    let select_cols = builtins().list(@[
        # newColumnSelectorInfo("A ", "int", true, opt.none[string]()),
        newColumnSelectorInfo("A ", "float", true, opt.none[string]()),
            # newColumnSelectorInfo("A ", "float", false, opt.none[string]()),
                # newColumnSelectorInfo("A ", "bool", false, opt.none[string]()),
                # newColumnSelectorInfo("A ", "str", false, opt.none[string]()),
                # newColumnSelectorInfo("A ", "date", false, opt.none[string]()),
                # newColumnSelectorInfo("A ", "datetime", false, opt.none[string]()),
                # newColumnSelectorInfo("A ", "time", false, opt.none[string]()),
        # newColumnSelectorInfo("A", "int", true, opt.none[string]()),
        # newColumnSelectorInfo("B", "str", true, opt.none[string]()),

        # newColumnSelectorInfo("str", "bool", false, opt.some("bool")),
        # newColumnSelectorInfo("str", "int", false, opt.some("int")),
        # newColumnSelectorInfo("str", "float", false, opt.some("float")),
        # newColumnSelectorInfo("str", "str", false, opt.some("str")),

        # newColumnSelectorInfo("float", "bool", false, opt.some("bool")),
        # newColumnSelectorInfo("float", "int", false, opt.some("int")),
        # newColumnSelectorInfo("float", "float", false, opt.some("float")),
        # newColumnSelectorInfo("float", "str", false, opt.some("str")),
        # newColumnSelectorInfo("float", "date", false, opt.some("date")),
        # newColumnSelectorInfo("float", "time", false, opt.some("time")),
        # newColumnSelectorInfo("float", "datetime", false, opt.some("datetime")),

        # newColumnSelectorInfo("date", "bool", false, opt.some("bool")),
        # newColumnSelectorInfo("date", "int", false, opt.some("int")),
        # newColumnSelectorInfo("date", "float", false, opt.some("float")),
        # newColumnSelectorInfo("date", "str", false, opt.some("str")),
        # newColumnSelectorInfo("date", "date", false, opt.some("date")),
        # newColumnSelectorInfo("date", "time", false, opt.some("time")),
        # newColumnSelectorInfo("date", "datetime", false, opt.some("datetime")),

        # newColumnSelectorInfo("str", "str", true, opt.some("str")),
    ])

    let (select_pass, select_fail) = table.columnSelect(
        select_cols,
        nimpy.pyImport("tqdm").tqdm,
        dir_pid = workdir / Path(pid),
        Taskmanager = mplite().TaskManager
    )

    discard select_pass.show(dtype = true)
    discard select_fail.show(dtype = true)