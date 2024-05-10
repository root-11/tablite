import nimpy
import std/[unittest, tables, sugar, sequtils, times]
import ../nimlite/[pymodules, nimpyext, numpy, pytypes]

test "scalar":
    check readNumpy("tests/data/pages/scalar.npy").len == 110234

test "repaginate":
    let m = modules()
    let Config = m.tablite.modules.config.classes.Config

    Config.PAGE_SIZE = 2

    let inpElems = @["1", "22", "333", "4444", "55555", "666666", "7777777"]
    let columns = m.builtins.classes.DictClass!({"A": inpElems}.toTable)
    let table = m.tablite.classes.TableClass!(columns = columns)
    let pages = collect: (for p in table["A"].pages: m.toStr(p.path))

    check pages.len == 4

    Config.PAGE_SIZE = 1_000_000

    let newPages = repaginate(pages)

    check newPages.len == 1

    let page = UnicodeNDArray(readNumpy(m.toStr(newPages[0].path)))
    let outElems = collect: (for i in 0..<page.len: page[i])

    check inpElems == outElems

test "iterateColumn":
    let m = modules()
    let inpElems = @["1", "22", "333", "4444", "55555", "666666", "7777777"]
    let columns = m.builtins.classes.DictClass!({"A": inpElems}.toTable)
    let table = m.tablite.classes.TableClass!(columns = columns)

    check inpElems == toSeq(iterateColumn[string](table["A"]))

test "iterateColumn as object":
    let m = modules()
    let inpElems = @["1", "22", "333", "4444", "55555", "666666", "7777777"]
    let columns = m.builtins.classes.DictClass!({"A": inpElems}.toTable)
    let table = m.tablite.classes.TableClass!(columns = columns)

    let outElems = collect:
        for v in iterateColumn[PY_ObjectND](table["A"]):
            PY_String(v).value

    check inpElems == outElems

test "test ndarray creation":
    check newNDArray[DateNDArray](@[now().utc]).kind == K_DATE
    check newNDArray[DateTimeNDArray](@[now().utc]).kind == K_DATETIME
    check newNDArray(@[false, false, true]).kind == K_BOOLEAN
    check newNDArray(@[1, 2, 3]).kind == K_INT64
    check newNDArray(@[1.0, 2.0, 3.0]).kind == K_FLOAT64
    check newNDArray(@["a", "bb", "ccc"]).kind == K_STRING
    check newNDArray(@[newPY_Object()]).kind == K_OBJECT
