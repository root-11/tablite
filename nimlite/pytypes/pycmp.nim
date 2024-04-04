import std/[macros, typetraits, times, strutils]

import ./pyobjs
from ../utils import implement

template isSameType(first: PY_ObjectND, second: PY_ObjectND): bool = first.kind == second.kind


proc addDefaultMethod(nMethodName: NimNode): NimNode {.compileTime.} =
    let nBody = newNimNode(nnkStmtList)
        .add(newCall(
            newIdentNode("implement"),
            infix(
                newLit(PY_ObjectND.name & "." & nMethodName.repr & " must be implemented by inheriting class: "),
                "&",
                prefix(newDotExpr(newIdentNode("self"), newIdentNode("kind")), "$")
            )
        ))

    let nParams = [
        bindSym("bool"),
        newIdentDefs(newIdentNode("self"), newIdentNode(PY_ObjectND.name)),
        newIdentDefs(newIdentNode("other"), newIdentNode(PY_ObjectND.name))
    ]

    let nPragmas = newNimNode(nnkPragma)
        .add(newIdentNode("base"))
        .add(newIdentNode("inline"))

    return newProc(postfix(nMethodName, "*"), nParams, nBody, nnkMethodDef, nPragmas)

proc addSimpleMethod(nMethodName: NimNode, typeName: string): NimNode {.compileTime.} =
    let nBody = newNimNode(nnkStmtList).add(
        infix(
            newCall(
                newDotExpr(newIdentNode("self"), newIdentNode("isSameType")),
                newIdentNode("other")
        ),
        "and",
        newCall(
            nMethodName,
            newDotExpr(newIdentNode("self"), newIdentNode("value")),
            newDotExpr(newCall(newIdentNode(typeName), newIdentNode("other")), newIdentNode("value"))
        )))

    let nParams = [
        bindSym("bool"),
        newIdentDefs(newIdentNode("self"), newIdentNode(typeName)),
        newIdentDefs(newIdentNode("other"), newIdentNode(PY_ObjectND.name))
    ]

    return newProc(postfix(nMethodName, "*"), nParams, nBody, nnkMethodDef)

proc addNumericMethod(nMethodName: NimNode, typeName1: string, typeName2: string): NimNode {.compileTime.} =
    var otherKind: string
    var nSameySelf: NimNode
    var nSameyOther: NimNode

    case typeName2:
    of PY_Int.name:
        otherKind = $K_INT
        nSameySelf = newDotExpr(newIdentNode("self"), newIdentNode("value"))
        nSameyOther = newCall(bindSym(float.name), newDotExpr(newCall(newIdentNode(typeName2), newIdentNode("other")), newIdentNode("value")))
    of PY_Float.name:
        otherKind = $K_FLOAT
        nSameySelf = newCall(bindSym(float.name), newDotExpr(newIdentNode("self"), newIdentNode("value")))
        nSameyOther = newDotExpr(newCall(newIdentNode(typeName2), newIdentNode("other")), newIdentNode("value"))
    else: raise newException(ValueError, "invalid type: " & $typeName2)

    let nBody = newNimNode(nnkStmtList).add(
        newNimNode(nnkIfStmt).add(
            newNimNode(nnkElifBranch).add(
                newCall(
                    newDotExpr(newIdentNode("self"), newIdentNode("isSameType")),
                    newIdentNode("other")
        ),
        newNimNode(nnkStmtList).add(
            newNimNode(nnkReturnStmt).add(
                newCall(
                    nMethodName,
                    newDotExpr(newIdentNode("self"), newIdentNode("value")),
                    newDotExpr(newCall(newIdentNode(typeName1), newIdentNode("other")), newIdentNode("value")))))
            ),
            newNimNode(nnkElifBranch).add(
                infix(newDotExpr(newIdentNode("other"), newIdentNode("kind")), "==", newIdentNode(otherKind)),
                newNimNode(nnkStmtList).add(
                    newNimNode(nnkReturnStmt).add(
                        newCall(
                            nMethodName,
                            nSameySelf,
                            nSameyOther)))
                ),
            newNimNode(nnkElse).add(
                newNimNode(nnkReturnStmt).add(bindSym("false"))))
    )

    let nParams = [
        bindSym("bool"),
        newIdentDefs(newIdentNode("self"), newIdentNode(typeName1)),
        newIdentDefs(newIdentNode("other"), newIdentNode(PY_ObjectND.name))
    ]

    return newProc(postfix(nMethodName, "*"), nParams, nBody, nnkMethodDef)



proc addDateMethod(nMethodName: NimNode, typeName1: string, typeName2: string): NimNode {.compileTime.} =
    let otherKind = (
        case typeName2:
        of PY_Date.name: $KindObjectND.K_DATE
        of PY_DateTime.name: $KindObjectND.K_DATETIME
        else: raise newException(ValueError, "invalid type: " & $typeName2)
    )

    let nBody = newNimNode(nnkStmtList).add(
        newNimNode(nnkIfStmt).add(
            newNimNode(nnkElifBranch).add(
                newCall(
                    newDotExpr(newIdentNode("self"), newIdentNode("isSameType")),
                    newIdentNode("other")
        ),
        newNimNode(nnkStmtList).add(
            newNimNode(nnkReturnStmt).add(
                newCall(
                    nMethodName,
                    newDotExpr(newIdentNode("self"), newIdentNode("value")),
                    newDotExpr(newCall(newIdentNode(typeName1), newIdentNode("other")), newIdentNode("value"))
            )))),
            newNimNode(nnkElifBranch).add(
                infix(newDotExpr(newIdentNode("other"), newIdentNode("kind")), "==", newIdentNode(otherKind)),
                newNimNode(nnkStmtList).add(
                    newNimNode(nnkReturnStmt).add(
                        newCall(
                            nMethodName,
                            newDotExpr(newIdentNode("self"), newIdentNode("value")),
                            newDotExpr(newCall(newIdentNode(typeName2), newIdentNode("other")), newIdentNode("value")))))
                ),
            newNimNode(nnkElse).add(newNimNode(nnkReturnStmt).add(bindSym("false"))))
    )

    let nParams = [
        bindSym("bool"),
        newIdentDefs(newIdentNode("self"), newIdentNode(typeName1)),
        newIdentDefs(newIdentNode("other"), newIdentNode(PY_ObjectND.name))
    ]

    return newProc(postfix(nMethodName, "*"), nParams, nBody, nnkMethodDef)

proc addNoneCode(nMethodName: NimNode, nNoneCode: NimNode): NimNode {.compileTime.} =
    let nParams = [
        bindSym("bool"),
        newIdentDefs(newIdentNode("self"), newIdentNode(PY_NoneType.name)),
        newIdentDefs(newIdentNode("other"), newIdentNode(PY_ObjectND.name))
    ]

    return newProc(postfix(nMethodName, "*"), nParams, nNoneCode, nnkMethodDef)

macro addCmpMethod(nMethodName: untyped, nNoneCode: untyped): untyped =
    let nMethods = newNimNode(nnkStmtList)

    nMethods.add(addDefaultMethod(nMethodName))
    nMethods.add(addNoneCode(nMethodName, nNoneCode))
    nMethods.add(addSimpleMethod(nMethodName, PY_Boolean.name))
    nMethods.add(addNumericMethod(nMethodName, PY_Int.name, PY_Float.name))
    nMethods.add(addNumericMethod(nMethodName, PY_Float.name, PY_Int.name))
    nMethods.add(addSimpleMethod(nMethodName, PY_String.name))
    nMethods.add(addDateMethod(nMethodName, PY_Date.name, PY_DateTime.name))
    nMethods.add(addSimpleMethod(nMethodName, PY_Time.name))
    nMethods.add(addDateMethod(nMethodName, PY_DateTime.name, PY_Date.name))

    return nMethods


addCmpMethod(`==`): isSameType(self, other)
addCmpMethod(`!=`): not isSameType(self, other)
addCmpMethod(`>`): false
addCmpMethod(`>=`): false
addCmpMethod(`<`): false
addCmpMethod(`<=`): false

method contains*(self: PY_ObjectND, other: PY_ObjectND): bool {.base, inline.} = implement("PY_ObjectND.`in` must be implemented by inheriting class: " & $self.kind)
method contains*(self: PY_NoneType, other: PY_ObjectND): bool = self.isSameType(other) or self.toRepr() in other.toRepr()
method contains*(self: PY_Boolean, other: PY_ObjectND): bool = self.toRepr() in other.toRepr()
method contains*(self: PY_Int, other: PY_ObjectND): bool = self.toRepr() in other.toRepr()
method contains*(self: PY_Float, other: PY_ObjectND): bool = self.toRepr() in other.toRepr()
method contains*(self: PY_String, other: PY_ObjectND): bool = self.toRepr() in other.toRepr()
method contains*(self: PY_Date, other: PY_ObjectND): bool = self.toRepr() in other.toRepr()
method contains*(self: PY_Time, other: PY_ObjectND): bool = self.toRepr() in other.toRepr()
method contains*(self: PY_DateTime, other: PY_ObjectND): bool = self.toRepr() in other.toRepr()
