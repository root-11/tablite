import nimpy
import std/[unittest, sugar, sequtils]
import ../nimlite/funcs/groupby
import ../nimlite/[pymodules, nimpyext]

template paginated(fn): void =
    let m = modules()

    m.tablite.modules.config.classes.Config.PAGE_SIZE = 1
    fn()
    m.tablite.modules.config.classes.Config.PAGE_SIZE = 1_000_000

proc testGroupOneKey(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[]) # None, 2, 4
    let col = collect: (for v in r["A"]: v)

    check col[0].isNone
    check col[1].to(int) == 2
    check col[2].to(int) == 4
test "groupby one key": testGroupOneKey()
test "groupby one key - paginated": paginated(testGroupOneKey)

proc testGroupBothKeys(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A", "B"], functions = @[]) # just like original
    let colA = collect: (for v in r["A"]: v)
    let colB = collect: (for v in r["B"]: v.to(int))

    check colA[0].isNone
    check colA[1].to(int) == 2
    check colA[2].to(int) == 2
    check colA[3].to(int) == 4
    check colA[4].isNone

    check @[2, 3, 4, 7, 6] == colB
test "groupby all keys": testGroupBothKeys()
test "groupby all keys - paginated": paginated(testGroupBothKeys)

proc testGroupMin(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A", "B"], functions = @[("A", Accumulator.Min)]) # Min(A) None, 2, 2, 4, None
    let colA = collect: (for v in r["A"]: v)
    let colB = collect: (for v in r["B"]: v.to(int))
    let colGrp = collect: (for v in r["Min(A)"]: v)

    check colA[0].isNone
    check colA[1].to(int) == 2
    check colA[2].to(int) == 2
    check colA[3].to(int) == 4
    check colA[4].isNone

    check @[2, 3, 4, 7, 6] == colB

    check colGrp[0].isNone
    check colGrp[1].to(int) == 2
    check colGrp[2].to(int) == 2
    check colGrp[3].to(int) == 4
    check colGrp[4].isNone
test "groupby min": testGroupMin()
test "groupby min - paginated": paginated(testGroupMin)

proc testGroupMax(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A", "B"], functions = @[("A", Accumulator.Max)]) # Max(A) None, 2, 2, 4, None
    let colA = collect: (for v in r["A"]: v)
    let colB = collect: (for v in r["B"]: v.to(int))
    let colGrp = collect: (for v in r["Max(A)"]: v)

    check colA[0].isNone
    check colA[1].to(int) == 2
    check colA[2].to(int) == 2
    check colA[3].to(int) == 4
    check colA[4].isNone

    check @[2, 3, 4, 7, 6] == colB

    check colGrp[0].isNone
    check colGrp[1].to(int) == 2
    check colGrp[2].to(int) == 2
    check colGrp[3].to(int) == 4
    check colGrp[4].isNone
test "groupby max": testGroupMax()
test "groupby max - paginated": paginated(testGroupMax)

proc testGroupSum(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.Sum)]) # 8, 7, 7
    let colGrp = collect: (for v in r["Sum(B)"]: v.to(float))

    check @[8.0, 7.0, 7.0] == colGrp
test "groupby sum": testGroupSum()
test "groupby sum - paginated": paginated(testGroupSum)

proc testGroupProduct(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.Product)]) # 12, 12, 7
    let colGrp = collect: (for v in r["Product(B)"]: v.to(float))

    check @[12.0, 12.0, 7.0] == colGrp
test "groupby product": testGroupProduct()
test "groupby product - paginated": paginated(testGroupProduct)

proc testGroupFirst(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.First)]) # 2, 3, 7
    let colGrp = collect: (for v in r["First(B)"]: v.to(int))

    check @[2, 3, 7] == colGrp
test "groupby first": testGroupFirst()
test "groupby first - paginated": paginated(testGroupFirst)

proc testGroupLast(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.Last)]) # 6, 4, 7
    let colGrp = collect: (for v in r["Last(B)"]: v.to(int))

    check @[6, 4, 7] == colGrp
test "groupby last": testGroupLast()
test "groupby last - paginated": paginated(testGroupLast)

proc testGroupCount(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.Count)]) # 2, 2, 1
    let colGrp = collect: (for v in r["Count(B)"]: v.to(int))

    check @[2, 2, 1] == colGrp
test "groupby count": testGroupCount()
test "groupby count - paginated": paginated(testGroupCount)

proc testGroupUnique(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.CountUnique)]) # 2, 2, 1
    let colGrp = collect: (for v in r["CountUnique(B)"]: v.to(int))

    check @[2, 2, 1] == colGrp
test "groupby count unique": testGroupUnique()
test "groupby count unique - paginated": paginated(testGroupUnique)

proc testGroupAvg(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.Average)]) # 4, 3.5, 7
    let colGrp = collect: (for v in r["Average(B)"]: v.to(float))

    check @[4.0, 3.5, 7.0] == colGrp
test "groupby average": testGroupAvg()
test "groupby average - paginated": paginated(testGroupAvg)

proc testGroupSTD(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.StandardDeviation)]) # 2.8284, 0.7071, 0.0
    let colGrp = collect: (for v in r["StandardDeviation(B)"]: v.to(float))

    for (a, b) in zip(@[2.82842712474619, 0.7071067811865476, 0.0], colGrp):
        check a - b < 1e-4
test "groupby stdev": testGroupSTD()
test "groupby stdev - paginated": paginated(testGroupSTD)

proc testGroupMedian(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.Median)]) # 4, 3.5, 7
    let colGrp = collect: (for v in r["Median(B)"]: v.to(float))

    check @[4.0, 3.5, 7.0] == colGrp
test "groupby median": testGroupMedian()
test "groupby median - paginated": paginated(testGroupMedian)

proc testGroupMode(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["A"] = @[nimValueToPy(nil), nimValueToPy(2), nimValueToPy(2), nimValueToPy(4), nimValueToPy(nil)]
    columns["B"] = @[nimValueToPy(2), nimValueToPy(3), nimValueToPy(4), nimValueToPy(7), nimValueToPy(6)]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["A"], functions = @[("B", Accumulator.Mode)]) # 6, 4, 7
    let colGrp = collect: (for v in r["Mode(B)"]: v.to(float))

    check @[6.0, 4.0, 7.0] == colGrp
test "groupby mode": testGroupMode()
test "groupby mode - paginated": paginated(testGroupMode)


proc testAll(): void =
    let m = modules()
    let columns = modules().builtins.classes.DictClass!()

    columns["a"] = @[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    columns["b"] = @[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    columns["c"] = @[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    columns["d"] = @[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    columns["e"] = @[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    columns["f"] = @[1, 4, 5, 10, 13, 1, 4, 7, 10, 13]
    columns["g"] = @[0, 1, 8, 27, 64, 0, 1, 8, 27, 64]

    let table = m.tablite.classes.TableClass!(columns = columns)
    let r = table.groupby(keys = @["a", "b"], functions = @[
        ("f", Accumulator.Max),
        ("f", Accumulator.Min),
        ("f", Accumulator.Sum),
        ("f", Accumulator.Product),
        ("f", Accumulator.First),
        ("f", Accumulator.Last),
        ("f", Accumulator.Count),
        ("f", Accumulator.CountUnique),
        ("f", Accumulator.Average),
        ("f", Accumulator.StandardDeviation),
        ("a", Accumulator.StandardDeviation),
        ("f", Accumulator.Median),
        ("f", Accumulator.Mode),
        ("g", Accumulator.Median),
    ])

    check m.getLen(r.columns) == 16
test "groupby all": testAll()
test "groupby all - paginated": paginated(testAll)
