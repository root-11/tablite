import nimpy
import std/[unittest, tables, sugar]
import ../nimlite/funcs/imputation
import ../nimlite/[pymodules, nimpyext, pytypes]

test "imp1":
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(0), nimValueToPy(1), nimValueToPy(nil), nimValueToPy(3), nimValueToPy(0)]
    columns["B"] = @[nimValueToPy(4), nimValueToPy(5), nimValueToPy(6), nimValueToPy(7), nimValueToPy(4)]

    let table = modules().tablite.classes.TableClass!(columns)
    let r = nearestNeighbourImputation(table, @["A", "B"], @[PY_ObjectND(PY_None)], @["A"])

    let impA = collect: (for v in r["A"]: v.to(int))
    let impB = collect: (for v in r["B"]: v.to(int))
    
    check len(impA) == 5
    check len(impB) == 5

    check @[0, 1, 1, 3, 0] == impA
    check @[4, 5, 6, 7, 4] == impB

test "imp1 - multipage":
    modules().tablite.modules.config.classes.Config.PAGE_SIZE = 1
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(0), nimValueToPy(1), nimValueToPy(nil), nimValueToPy(3), nimValueToPy(0)]
    columns["B"] = @[nimValueToPy(4), nimValueToPy(5), nimValueToPy(6), nimValueToPy(7), nimValueToPy(4)]

    let table = modules().tablite.classes.TableClass!(columns)
    let r = nearestNeighbourImputation(table, @["A", "B"], @[PY_ObjectND(PY_None)], @["A"])

    let impA = collect: (for v in r["A"]: v.to(int))
    let impB = collect: (for v in r["B"]: v.to(int))
    
    check len(impA) == 5
    check len(impB) == 5

    check @[0, 1, 1, 3, 0] == impA
    check @[4, 5, 6, 7, 4] == impB

test "imp2":
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(0), nimValueToPy(1), nimValueToPy(nil), nimValueToPy(3), nimValueToPy(0)]
    columns["B"] = @[nimValueToPy("4"), nimValueToPy(5), nimValueToPy(6), nimValueToPy(7), nimValueToPy(4)]

    let table = modules().tablite.classes.TableClass!(columns)
    let r = nearestNeighbourImputation(table, @["A", "B"], @[PY_ObjectND(PY_None)], @["A"])

    let impA = collect: (for v in r["A"]: v.to(int))
    let impB = collect: (for v in r["B"]: v)
    
    check len(impA) == 5
    check len(impB) == 5

    check @[0, 1, 3, 3, 0] == impA
    check impB[0].to(string) == "4"
    check impB[1].to(int) == 5
    check impB[2].to(int) == 6
    check impB[3].to(int) == 7
    check impB[4].to(int) == 4

test "imp3":
    let columns = modules().builtins.classes.DictClass!()

    columns["a"] = @[1, 1, 5, 5, 6, 6]
    columns["b"] = @[2, 2, 5, 5, 6, -1]
    columns["c"] = @[nimValueToPy(3), nimValueToPy(nil), nimValueToPy(5), nimValueToPy("NULL"), nimValueToPy(6), nimValueToPy(6)]

    let table = modules().tablite.classes.TableClass!(columns)
    let r = nearestNeighbourImputation(table, @["a", "b", "c"], @[PY_ObjectND(PY_None), newPY_Object("NULL"), newPY_Object(-1)], @["b", "c"])

    let impA = collect: (for v in r["a"]: v.to(int))
    let impB = collect: (for v in r["b"]: v.to(int))
    let impC = collect: (for v in r["c"]: v.to(int))
    
    check len(impA) == 6
    check len(impB) == 6
    check len(impC) == 6

    check @[1, 1, 5, 5, 6, 6] == impA
    check @[2, 2, 5, 5, 6, 6] == impB
    check @[3, 3, 5, 5, 6, 6] == impC

test "imp4":
    let columns = modules().builtins.classes.DictClass!()

    columns["a"] = @[nimValueToPy(nil), nimValueToPy(1), nimValueToPy(2), nimValueToPy(3)]
    columns["b"] = @[nimValueToPy(0), nimValueToPy(nil), nimValueToPy(2), nimValueToPy(3)]
    columns["c"] = @[nimValueToPy(0), nimValueToPy(1), nimValueToPy(nil), nimValueToPy(3)]
    columns["d"] = @[nimValueToPy(0), nimValueToPy(1), nimValueToPy(2), nimValueToPy(nil)]

    let table = modules().tablite.classes.TableClass!(columns)
    let r = nearestNeighbourImputation(table, @["a", "b", "c", "d"], @[PY_ObjectND(PY_None)], @["a", "b", "c", "d"])

    let impA = collect: (for v in r["a"]: v)
    let impB = collect: (for v in r["b"]: v.to(int))
    let impC = collect: (for v in r["c"]: v.to(int))
    let impD = collect: (for v in r["d"]: v.to(int))
    
    check len(impA) == 4
    check len(impB) == 4
    check len(impC) == 4
    check len(impD) == 4

    check impA[0].isNone
    check impA[1].to(int) == 1
    check impA[2].to(int) == 2
    check impA[3].to(int) == 3

    check @[0, 0, 2, 3] == impB
    check @[0, 1, 0, 3] == impC
    check @[0, 1, 2, 0] == impD