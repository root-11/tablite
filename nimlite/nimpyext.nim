import std/macros
import nimpy {.all.}

# Maybe I can get this to be part of nimpy lib (https://github.com/yglukhov/nimpy/issues/299)

template toBangArg(inNode: NimNode, plainArgs: NimNode, kwArgs: NimNode) =
    var node: NimNode

    if inNode.kind == nnkPar:
        node = inNode[0]
    else:
        node = inNode

    if node.kind == nnkHiddenStdConv and node[0].kind == nnkEmpty:
        discard
    elif node.kind == nnkExprEqExpr or node.kind == nnkAsgn or node.kind == nnkExprColonExpr:
        kwArgs.add(newTree(nnkPar,
          newCall("cstring", newLit($node[0])),
          newCall("toPyObjectArgument", node[1])))
    else:
        plainArgs.add(newCall("toPyObjectArgument", node))

macro `!`*(o: PyObject, args: varargs[untyped]): PyObject =
    expectKind(o, nnkSym)

    let plainArgs = newTree(nnkBracket)
    let kwArgs = newTree(nnkBracket)

    expectKind(args, nnkArgList)
    expectLen(args, 1)

    if args[0].kind == nnkTupleConstr:
        for arg in args[0]:
            toBangArg(arg, plainArgs, kwArgs)
    else:
        toBangArg(args[0], plainArgs, kwArgs)

    result = newCall(bindSym"newPyObjectConsumingRef",
      newCall(bindSym"callObjectAux", newDotExpr(o, newIdentNode("privateRawPyObj")), plainArgs, kwArgs))
