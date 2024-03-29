import nimpy
import std/[sugar, macros, enumerate]
import ../[pymodules, utils]

const FILTER_KEYS = ["column1", "column2", "criteria", "value1", "value2"]
const FILTER_OPS = [">", ">=", "==", "<", "<=", "!=", "in"]
const FILTER_TYPES = ["all", "any"]

proc filter(table: nimpy.PyObject, expressions: seq[nimpy.PyObject], filterType: string): nimpy.PyObject =
    let m = modules()
    let builtins = m.builtins
    let tablite = m.tablite
    let base = tablite.modules.base
    let Config = tablite.modules.config.classes.Config

    if filterType notin FILTER_TYPES:
        raise newException(ValueError, "invalid filter type '" & filterType & "' expected " & $FILTER_TYPES)

    if expressions.len == 0:
        return table

    let columns = collect: (for c in table.columns.keys(): c.to(string))

    if columns.len == 0:
        return table

    template checkParam(columnName: string, valueName: string, paramName: string) =
        if columnName in expression:
            if valueName in expression:
                raise newException(ValueError, "filter can only take 1 " & paramName & " expr element, got 2")

            let c = expression[columnName].to(string)

            if c notin columns:
                raise newException(ValueError, "no such column '" & $c & "'in " & $columns)
        elif valueName notin expression:
            raise newException(ValueError, "no left parameter")

    for expression in expressions:
        if not builtins.isinstance(expression, builtins.classes.DictClass):
            raise newException(KeyError, "expression must be a dict: " & $expression)

        if not builtins.getLen(expression) == 3:
            raise newException(ValueError, "expression must be of len 3: " & $builtins.getLen(expression))

        let invalidKeys = collect:
            for pyKey in expression.keys():
                let key = pyKey.to(string)

                if key notin FILTER_KEYS:
                    continue

                key

        if invalidKeys.len > 0:
            raise newException(ValueError, "got unknown keys " & $invalidKeys & " expected " & $FILTER_KEYS)

        let criteria = expression["criteria"].to(string)

        if criteria notin FILTER_OPS:
            raise newException(ValueError, "invalid criteria '" & criteria & "' expected " & $FILTER_OPS)

        checkParam("column1", "value1", "left")
        checkParam("column2", "value2", "right")

    let pageCount = base.collectPages(table[columns[0]]).len



proc myIter(c: int = 5): iterator: int =
    return iterator: int =
        for i in 0..<c:
            yield i

proc myIterStr(): auto =
    return iterator: auto =
        yield "a"
        yield "b"
        yield "c"

proc collectIters(itIdents: openArray[NimNode]): NimNode {.compileTime.} =
    let nNextIterTuple = newNimNode(nnkTupleConstr)

    for nId in itIdents:
        nNextIterTuple.add(newCall(nId))

    return nNextIterTuple

proc createResultStatement(itIdents: openArray[NimNode]): NimNode {.compileTime.} =
    let nNextIter = newNimNode(nnkVarSection)
    let nNextIterDefs = newNimNode(nnkIdentDefs).add(newIdentNode("results"), newEmptyNode())

    nNextIterDefs.add(collectIters(itIdents))
    nNextIter.add(nNextIterDefs)

    return nNextIter

proc collectFinished(itIdents: openArray[NimNode]): NimNode {.compileTime.} =
    var nLastElement = newCall(bindSym("finished"), itIdents[0])

    for i in 1..<itIdents.len:
        let nOther = newCall(bindSym("finished"), itIdents[i])

        nLastElement = infix(nLastElement, "or", nOther)

    return nLastElement

proc createFinishedStatement(itIdents: openArray[NimNode]): NimNode {.compileTime.} =
    let nNextIter = newNimNode(nnkVarSection)
    let nNextIterDefs = newNimNode(nnkIdentDefs).add(newIdentNode("finished"), newEmptyNode())

    nNextIterDefs.add(collectFinished(itIdents))
    nNextIter.add(nNextIterDefs)

    return nNextIter

proc createWhileLoop(itIdents: openArray[NimNode]): NimNode {.compileTime.} =
    let nIdRes = newIdentNode("results")
    let nIdFinished = newIdentNode("finished")
    let nWhileStmt = newNimNode(nnkWhileStmt)
    let nNotFinished = prefix(nIdFinished, "not")
    let nStmtList = newNimNode(nnkStmtList)
    let nYieldStmt = newNimNode(nnkYieldStmt).add(nIdRes)

    nWhileStmt.add(nNotFinished)
    nWhileStmt.add(nStmtList)
    nStmtList.add(nYieldStmt)
    nStmtList.add(newNimNode(nnkAsgn).add(nIdRes, collectIters(itIdents)))
    nStmtList.add(newNimNode(nnkAsgn).add(nIdFinished, collectFinished(itIdents)))

    return nWhileStmt

proc rewriteAsIterator(nNode: NimNode): NimNode {.compileTime.} =
    discard

proc collectZipperArgs(nSqs: NimNode): seq[NimNode] {.compileTime.} =
    expectKind(nSqs, nnkBracket)

    let nYieldTuple = newNimNode(nnkTupleConstr)

    var itProcArgs = @[nYieldTuple]

    echo "-------------------"
    for (i, nSq) in enumerate(nSqs):
        let nIterProcType = getType(nSq)
        let iterableType = $nIterProcType[0]

        echo nIterProcType.treeRepr

        expectKind(nIterProcType, nnkBracketExpr)

        case iterableType:
        of "proc": discard
        of "seq", "array": 
            discard rewriteAsIterator(iterableType, nSq)
        else: raise newException(ValueError, "unsupported iterable")


        # case nIterProcType[0]
        # expectIdent(nIterProcType[0], "proc")

        echo nIterProcType[0]


    return itProcArgs

macro zipper(sqs: varargs[typed]): untyped =
    expectKind(sqs, nnkBracket)

    discard collectZipperArgs(sqs)

    # let nYieldTuple = newNimNode(nnkTupleConstr)

    # var itProcArgs = @[nYieldTuple]
    # var itIdents = newSeqOfCap[NimNode](sqs.len)

    # let nBody = newNimNode(nnkStmtList)

    # for (i, sq) in enumerate(sqs):
    #     let nIterProcType = getType(sq)
    #     echo getType(sq).treeRepr

    #     expectKind(nIterProcType, nnkBracketExpr)
    #     expectIdent(nIterProcType[0], "proc")

    #     let nIterType = nIterProcType[1]
    #     let nIdent = newIdentNode("it" & $i)
    #     let nIdentSq = newIdentNode("iit" & $i)


    #     let nIdentDef = newNimNode(nnkIdentDefs)
    #     let nIteratorTy = newNimNode(nnkIteratorTy)
    #     let nIteratorPars = newNimNode(nnkFormalParams)


    #     nBody.add(
    #         newNimNode(nnkVarSection).add(
    #             newNimNode(nnkIdentDefs).add(
    #                 nIdent, newEmptyNode(), nIdentSq
    #         )
    #     )
    #     )

    #     nIteratorPars.add(nIterType)
    #     nIteratorTy.add(nIteratorPars, newEmptyNode())
    #     nIdentDef.add(nIdentSq, nIteratorTy, newEmptyNode())

    #     nYieldTuple.add(nIterType)
    #     itProcArgs.add(nIdentDef)

    #     itIdents.add(nIdent)

    # nBody.add(createResultStatement(itIdents))
    # nBody.add(createFinishedStatement(itIdents))
    # nBody.add(createWhileLoop(itIdents))

    # let nProcName = newEmptyNode()
    # let nProc = newProc(nProcName, itProcArgs, nBody, nnkIteratorDef)

    # let nCall = newCall(newPar(nProc))

    # for i in 0..<sqs.len:
    #     nCall.add(sqs[i])

    # # echo "nProc: \n" & nProc.repr
    # # echo "nCall: \n" & nCall.treeRepr
    # echo nCall.repr

    # return nCall

    implement("hi")

for tpl in zipper(myIter(), myIterStr(), @[1, 2, 3], [4, 5, 6]):
    # discard
    echo tpl
