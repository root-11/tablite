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
    import nimpy
    from ../nimpyext import `!`
    import std/options as opt
    import ../pymodules
    import ../numpy
    import typetraits

    proc columnSelect(table: nimpy.PyObject, cols: nimpy.PyObject, tqdm: nimpy.PyObject, dir_pid: Path, TaskManager: nimpy.PyObject): (nimpy.PyObject, nimpy.PyObject) =
        # this is nim-only implementation, the library build doesn't need it because we need TaskManager to be used for slices
        let TableClass = modules().getType(table)
        var pbar = tqdm!(total: 100, desc: "column select")
        let colInfoResult = collectColumnSelectInfo(table, cols, string dir_pid, pbar)

        if toSeq(colInfoResult.isCorrectType.values).all(proc (x: bool): bool = x):
            let tblPassColumns = collect(initTable()):
                for (desiredName, desiredInfo) in colInfoResult.desiredColumnMap.pairs():
                    {desiredName: table[desiredInfo.originalName]}

            let tblFailColumns = collect(initTable()):
                for desiredName in colInfoResult.failedColumnData:
                    {desiredName: newSeq[nimpy.PyObject]()}

            let tblPass = TableClass!(columns: tblPassColumns)
            let tblFail = TableClass!(columns: tblFailColumns)

            return (tblPass, tblFail)

        template ordered2PyDict(keys: seq[string]): nimpy.PyObject =
            let dict = modules().builtins.classes.DictClass!()

            for k in keys:
                dict[k] = newSeq[nimpy.PyObject]()

            dict

        var tblPass = TableClass!(columns = colInfoResult.passedColumnData.ordered2PyDict())
        var tblFail = TableClass!(columns = colInfoResult.failedColumnData.ordered2PyDict())

        var taskListInp = collect:
            for i in 0..<colInfoResult.pageCount:
                let el = collect(initTable()):
                    for (name, column) in colInfoResult.columns.pairs:
                        {name: (column[i], colInfoResult.originalPagesMap[name][i])}
                (el, colInfoResult.resColsPass[i], colInfoResult.resColsFail[i])

        let tabliteConfig = modules().tablite.modules.config.classes.Config
        var pageSize = tabliteConfig.PAGE_SIZE.to(int)
        var converted = newSeqOfCap[(seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)])](taskListInp.len)
        var pbarStep = 45 / max(taskListInp.len - 1, 1)

        for (columns, resPass, resFail) in taskListInp:
            converted.add(doSliceConvert(dir_pid, pageSize, columns, colInfoResult.rejectReasonName, resPass, resFail, colInfoResult.desiredColumnMap, colInfoResult.columnNames, colInfoResult.isCorrectType))

            discard pbar.update(pbarStep)

        proc extendTable(table: var nimpy.PyObject, columns: seq[(string, nimpy.PyObject)]): void {.inline.} =
            for (col_name, pg) in columns:
                let col = table[col_name]

                discard col.pages.append(pg) # can't col.extend because nim is dumb :)

        for (pg_pass, pg_fail) in converted:
            tblPass.extendTable(pg_pass)
            tblFail.extendTable(pg_fail)

        discard pbar.update(pbar.total.to(float) - pbar.n.to(float))
        discard pbar.close()

        return (tblPass, tblFail)

    proc newColumnSelectorInfo(column: string, `type`: string, allow_empty: bool, rename: opt.Option[string]): nimpy.PyObject =
        let pyDict = modules().builtins.classes.DictClass!(
            column: column,
            type: `type`,
            allow_empty: allow_empty
        )

        if rename.isNone():
            pyDict["rename"] = nil
        else:
            pyDict["rename"] = rename.get()

        return pyDict

    let tabliteConfig = modules().tablite.modules.config.classes.Config
    let workdir = Path(modules().toStr(tabliteConfig.workdir))
    let pid = "nim"
    let pagedir = workdir / Path(pid) / Path("pages")

    createDir(string pagedir)

    tabliteConfig.pid = pid
    # tabliteConfig.pageSize = 2
    # tabliteConfig.MULTIPROCESSING_MODE = tabliteConfig.FALSE

    # let columns = modules().builtins.classes.DictClass!({"A ": @[nimValueToPy(0), nimValueToPy(nil), nimValueToPy(10), nimValueToPy(200)]}.toTable)
    # let columns = modules().builtins.classes.DictClass!({"A ": @[1, 22, 333]}.toTable)
    # let columns = modules().builtins.classes.DictClass!({"A": @[0, 1]}.toTable)
    # let columns = modules().builtins.classes.DictClass!({"A ": @["1", "22", "333", "", "abc"]}.toTable)
    # let columns = modules().builtins.classes.DictClass!({"A": @["a", "1", "c"], "B": @["d", "e", "f"]}.toTable)
    # let columns = modules().builtins.classes.DictClass!({"A ": @[nimValueToPy("1"), nimValueToPy("222"), nimValueToPy("333"), nimValueToPy(nil), nimValueToPy("abc")]}.toTable)
    # let columns = modules().builtins.classes.DictClass!({"A ": @[nimValueToPy(1), nimValueToPy(2.0), nimValueToPy("333"), nimValueToPy("abc")]}.toTable)
    # let columns = modules().builtins.classes.DictClass!({"A": @[nimValueToPy(111111), nimValueToPy(222222), nimValueToPy(333333)], "B": @[nimValueToPy(0), nimValueToPy(nil), nimValueToPy(2)]}.toTable)
    # let columns = modules().builtins.classes.DictClass!({"A": @[nimValueToPy("0"), nimValueToPy(nil), nimValueToPy("2")], "B": @[nimValueToPy("3"), nimValueToPy(nil), nimValueToPy("4")]}.toTable)
    # let columns = modules().builtins.classes.DictClass!({"str": @["1", "0"]})
    # let columns = modules().builtins.classes.DictClass!({"float": @[1.0, 0.0]})
    # let columns = modules().builtins.classes.DictClass!({"date": @[
    #     datetime().date(2000, 1, 1),
    #     datetime().date(2000, 1, 2),
    # ]})
    let columns = modules().builtins.classes.DictClass!({"date": @[
        modules().datetime.classes.DateClass!(2000, 1, 1),
        modules().datetime.classes.DateClass!(2000, 1, 2),
    ]})
    # let columns = pymodules.builtins().dict({"str": @[nimValueToPy("abc"), nimValueToPy("efg"), nimValueToPy(nil)]}.toTable)
    let table = modules().tablite.classes.TableClass!(columns = columns)
    let dirdata = os.getEnv("DATA_DIR", ".")
    # let table = modules().tablite.fromFile(dirdata & "/gesaber_data_10k.csv")
    # let table = modules().tablite.classes.TableClass.load("/media/ratchet/hdd/tablite/filter_0_false.tpz")

    # discard table.show(dtype = true)

    let select_cols = modules().builtins.classes.ListClass!(@[
        # newColumnSelectorInfo("A ",, "int", true, opt.none[string]()),
        # newColumnSelectorInfo("A ",, "float", true, opt.none[string]()),
            # newColumnSelectorInfo("A ",, "float", false, opt.none[string]()),
                # newColumnSelectorInfo("A ", "bool", false, opt.none[string]()),
                # newColumnSelectorInfo("A ",, "str", false, opt.none[string]()),
                # newColumnSelectorInfo("A ", "date", false, opt.none[string]()),
                # newColumnSelectorInfo("A ", "datetime", false, opt.none[string]()),
                # newColumnSelectorInfo("A ", "time", false, opt.none[string]()),
            # newColumnSelectorInfo("A",, "int", true, opt.none[string]()),
            # newColumnSelectorInfo("B",, "str", true, opt.none[string]()),

            # newColumnSelectorInfo("str", "bool", false, opt.some("bool")),
            # newColumnSelectorInfo("str",, "int", false, opt.some("int")),
            # newColumnSelectorInfo("str",, "float", false, opt.some("float")),
            # newColumnSelectorInfo("str",, "str", false, opt.some("str")),

            # newColumnSelectorInfo("float", "bool", false, opt.some("bool")),
            # newColumnSelectorInfo("float",, "int", false, opt.some("int")),
            # newColumnSelectorInfo("float",, "float", false, opt.some("float")),
            # newColumnSelectorInfo("float",, "str", false, opt.some("str")),
            # newColumnSelectorInfo("float", "date", false, opt.some("date")),
            # newColumnSelectorInfo("float", "time", false, opt.some("time")),
            # newColumnSelectorInfo("float", "datetime", false, opt.some("datetime")),

            # newColumnSelectorInfo("date", "bool", false, opt.some("bool")),
            # newColumnSelectorInfo("date",, "int", false, opt.some("int")),
            # newColumnSelectorInfo("date",, "float", false, opt.some("float")),
            # newColumnSelectorInfo("date",, "str", false, opt.some("str")),
            # newColumnSelectorInfo("date", "date", false, opt.some("date")),
            # newColumnSelectorInfo("date", "time", false, opt.some("time")),
            # newColumnSelectorInfo("date", "datetime", false, opt.some("datetime")),

            # newColumnSelectorInfo("str",, "str", true, opt.some("str")),

            # newColumnSelectorInfo("A",, "str", false, opt.none[string]()),
            # newColumnSelectorInfo("B",, "int", false, opt.none[string]()),

            # newColumnSelectorInfo("A", "int", false, opt.none[string]()),
            
            # newColumnSelectorInfo("date", "bool", false, opt.some("bool")),
            # newColumnSelectorInfo("date", "int", false, opt.some("int")),
            # newColumnSelectorInfo("date", "float", false, opt.some("float")),
            # newColumnSelectorInfo("date", "str", false, opt.some("str")),
            # newColumnSelectorInfo("date", "date", false, opt.some("date")),
            newColumnSelectorInfo("date", "time", false, opt.some("time")),
            newColumnSelectorInfo("date", "datetime", false, opt.some("datetime")),

            # newColumnSelectorInfo("sale_date", "datetime", false, opt.none[string]()),
            # newColumnSelectorInfo("cust_nbr", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Order_Number", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("prod_slbl", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("cases", "int", false, opt.none[string]()),

            # newColumnSelectorInfo("Article code", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Article Description", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Department", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Department Name", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("MC", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("MC Name", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Season", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Season Name", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Source", "int", false, opt.none[string]()),
            # newColumnSelectorInfo("Source Name", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("X-site artl status", "int", false, opt.none[string]()),
            # newColumnSelectorInfo("X-site artl status desc", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Fragile?", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Inner Type (Current)", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Inner Name (Current)", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Inner Type (STO)", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Units per Case (Current)", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("Units per Case (STO)", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("Units per pallet\\nLATEST", "float", false, opt.some("Units per pallet")),
            # newColumnSelectorInfo("Case L (m)", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("Case W (m)", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("Case H (m)", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("Case Vol (m3)", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("Case Gross Weight (KG)", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("Inner L (m)", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("Inner Weight", "float", false, opt.none[string]()),
            # newColumnSelectorInfo("STOs", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Sum of STO Qty", "str", false, opt.none[string]()),
            # newColumnSelectorInfo("Pallet Ti", "int", false, opt.none[string]()),
            # newColumnSelectorInfo("Pallet Hi", "int", false, opt.none[string]()),
            # newColumnSelectorInfo("LAY", "str", false, opt.none[string]()),
    ])

    

    let (select_pass, select_fail) = table.columnSelect(
        select_cols,
        nimpy.pyImport("tqdm").tqdm,
        dir_pid = workdir / Path(pid),
        Taskmanager = modules().mplite.classes.TaskManager
    )

    discard select_pass.show(dtype = true)
    discard select_fail.show(dtype = true)