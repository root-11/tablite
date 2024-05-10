import nimpy
import std/[unittest, tables, sugar]
import ../nimlite/funcs/filter
import ../nimlite/[pymodules, nimpyext]


proc valueFilter1*(): auto =
    let m = modules()
    let table = m.tablite.classes.TableClass!({
        "a": @[1, 2, 3, 4],
        "b": @[10, 20, 30, 40],
        "c": @[4, 4, 4, 4]
    }.toTable)

    let pyExpressions = @[
        m.builtins.classes.DictClass!(column1: "a", criteria: ">=", value2: 2),
    ]

    return filter(table, pyExpressions, "all", nil)

proc valueFilter2*(): auto =
    let m = modules()

    let table = m.tablite.classes.TableClass!({
        "a": @[1, 2, 3, 4],
        "b": @[10, 20, 30, 40],
        "c": @[4, 4, 4, 4]
    }.toTable)
    let pyExpressions = @[
        m.builtins.classes.DictClass!(column1: "a", criteria: ">=", value2: 2),
        m.builtins.classes.DictClass!(column1: "a", criteria: "==", column2: "c"),
    ]

    return filter(table, pyExpressions, "all", nil)

proc valueFilter3*(): auto =
    let m = modules()

    let table = m.tablite.classes.TableClass!({
        "a": @[1, 2, 3, 4],
        "b": @[10, 20, 30, 40],
        "c": @[4, 4, 4, 4]
    }.toTable)
    let pyExpressions = @[
        m.builtins.classes.DictClass!(column1: "a", criteria: ">=", value2: 2),
        m.builtins.classes.DictClass!(column1: "a", criteria: "==", column2: "c"),
    ]

    return filter(table, pyExpressions, "all", nil)

proc valueFragmentation*(inpPagesSize: int, outPagesSize: int): auto =
    let m = modules()
    let Config = m.tablite.modules.config.classes.Config

    Config.PAGE_SIZE = inpPagesSize

    let table = m.tablite.classes.TableClass!({
        "a": @[1, 2, 3, 4],
        "b": @[10, 20, 30, 40],
        "c": @[4, 4, 4, 4]
    }.toTable)

    Config.PAGE_SIZE = outPagesSize

    let pyExpressions = @[
        m.builtins.classes.DictClass!(column1: "a", criteria: ">=", value2: 2),
    ]

    return filter(table, pyExpressions, "all", nil)

proc valueFragmentationCheck(inpPagesSize: int, outPagesSize: int): void =
    let m = modules()
    let (tblPass, tblFail) = valueFragmentation(inpPagesSize, outPagesSize)

    check m.builtins.getLen(tblPass) == 3
    check m.builtins.getLen(tblFail) == 1

    let vaPass = collect: (for v in tblPass["a"]: v.to(int))
    let vbPass = collect: (for v in tblPass["b"]: v.to(int))
    let vcPass = collect: (for v in tblPass["c"]: v.to(int))

    let vaFail = collect: (for v in tblFail["a"]: v.to(int))
    let vbFail = collect: (for v in tblFail["b"]: v.to(int))
    let vcFail = collect: (for v in tblFail["c"]: v.to(int))

    check vaPass == @[2, 3, 4]
    check vbPass == @[20, 30, 40]
    check vcPass == @[4, 4, 4]

    check vaFail == @[1]
    check vbFail == @[10]
    check vcFail == @[4]

when not defined(DEV_BUILD):
    test "value filter":
        let m = modules()
        let (tblPass, tblFail) = valueFilter1()

        check m.builtins.getLen(tblPass) == 3
        check m.builtins.getLen(tblFail) == 1

        let vaPass = collect: (for v in tblPass["a"]: v.to(int))
        let vbPass = collect: (for v in tblPass["b"]: v.to(int))
        let vcPass = collect: (for v in tblPass["c"]: v.to(int))

        let vaFail = collect: (for v in tblFail["a"]: v.to(int))
        let vbFail = collect: (for v in tblFail["b"]: v.to(int))
        let vcFail = collect: (for v in tblFail["c"]: v.to(int))

        check vaPass == @[2, 3, 4]
        check vbPass == @[20, 30, 40]
        check vcPass == @[4, 4, 4]

        check vaFail == @[1]
        check vbFail == @[10]
        check vcFail == @[4]

    test "value filter fragmented 2->2":
        valueFragmentationCheck(2, 2)

    test "value filter fragmented 2->1M":
        valueFragmentationCheck(2, 1_000_000)

    test "value filter fragmented 1M->2":
        valueFragmentationCheck(1_000_000, 2)

    test "value filter fragmented 3->2":
        valueFragmentationCheck(3, 2)

    test "value filter fragmented 3->1M":
        valueFragmentationCheck(3, 1_000_000)
