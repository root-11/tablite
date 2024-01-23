import std/[macros, tables]
from std/typetraits import name
from std/sugar import collect
import ../../numpy
from ../../pytypes import PY_ObjectND, KindObjectND
import casters
from mask import Mask
from makepage import makePage
from ../../utils import corrupted

macro mkPageCaster(nBaseType: typedesc, overrides: untyped) =
    expectKind(nBaseType, nnkSym)

    const tPageGenerics = {
        bool.name: @[BooleanNDArray.name],
        int.name: @[Int8NDArray.name, Int16NDArray.name, Int32NDArray.name, Int64NDArray.name],
        float.name: @[Float32NDArray.name, Float64NDArray.name],
        FromDate.name: @[DateNDArray.name],
        FromDateTime.name: @[DateTimeNDArray.name],
        string.name: @[UnicodeNDArray.name],
        PY_ObjectND.name: @[ObjectNDArray.name],
    }.toTable

    let pageGenerics = tPageGenerics[nBaseType.strVal]
    let nPageGenerics = collect: (for ch in pageGenerics: newIdentNode(ch))

    const tCasterPage = {
        bool.name: BooleanNDArray.name,
        int.name: Int64NDArray.name,
        float.name: Float64NDArray.name,
        string.name: UnicodeNDArray.name,
        ToDate.name: DateNDArray.name,
        ToDateTime.name: DateTimeNDArray.name,
        ToTime.name: ObjectNDArray.name
    }.toTable

    const allCastTypes = [bool.name, int.name, float.name, string.name, ToDate.name, ToDateTime.name, ToTime.name]

    let nMaskSeq = newNimNode(nnkBracketExpr).add(bindSym("seq"), bindSym(Mask.name))
    let nMaskVarTy = newNimNode(nnkVarTy).add(nMaskSeq)
    let nMaskDef = newIdentDefs(newIdentNode("mask"), nMaskVarTy)

    let nReasonListSeq = newNimNode(nnkBracketExpr).add(bindSym("seq"), bindSym(string.name))
    let nReasonListVarTy = newNimNode(nnkVarTy).add(nReasonListSeq)
    let nReasonListDef = newIdentDefs(newIdentNode("reason_lst"), nReasonListVarTy)

    let nAllowEmpty = newIdentDefs(newIdentNode("allow_empty"), bindSym(bool.name))
    let nOriginaName = newIdentDefs(newIdentNode("original_name"), bindSym(string.name))
    let nDesiredName = newIdentDefs(newIdentNode("desired_name"), bindSym(string.name))
    let nDesiredType = newIdentDefs(newIdentNode("desired_type"), bindSym(KindObjectND.name))

    let nProcNodes = newNimNode(nnkStmtList)

    let nGeneric = newNimNode(nnkGenericParams)
    var nGenTypes {.noinit.}: NimNode

    if nPageGenerics.len > 1:
        var nNextT = nPageGenerics[^1]

        for i in countdown(nPageGenerics.len - 2, 1):
            nNextT = infix(nPageGenerics[i], "|", nNextT)

        nGenTypes = infix(nPageGenerics[0], "|", nNextT)

    else:
        nGenTypes = nPageGenerics[0]

    let nGenIdet = newIdentDefs(newIdentNode("T"), nGenTypes)

    nGeneric.add(nGenIdet)

    for newCaster in allCastTypes:
        var nPageReturnType {.noinit.}: NimNode
        var nBody {.noinit.}: NimNode

        let tCasterPage = tCasterPage[newCaster]
        let nPageTypeDesc = newNimNode(nnkBracketExpr).add(newIdentNode("typedesc"), newIdentNode(newCaster))
        let nArgReturnType = newIdentDefs(newIdentNode("R"), nPageTypeDesc)
        let nInPage = newIdentDefs(newIdentNode("page"), newIdentNode("T"))

        if tCasterPage in pageGenerics:
            nBody = overrides
            nPageReturnType = (if nPageGenerics.len > 1: newIdentNode("T") else: newIdentNode(tCasterPage))
        else:
            nPageReturnType = newIdentNode(tCasterPage)
            nBody = newNimNode(nnkStmtList)

            let nIfStmt = newNimNode(nnkIfStmt)
            let nElifBranch = newNimNode(nnkElifBranch)
            let nAllowPrefix = newNimNode(nnkPrefix)
            let nElse = newNimNode(nnkElse)

            template statements(nPageReturnType: NimNode, fnName: string): NimNode =
                let nCallCasterFn = newDotExpr(nBaseType, newIdentNode(fnName))
                let nCallCaster = newCall(nCallCasterFn, newIdentNode("R"))

                let nCallMkPageFn = newDotExpr(nPageReturnType, newIdentNode("makePage"))
                let nCallArgs = [newIdentNode("page"), newIdentNode("mask"), newIdentNode("reason_lst"), nCallCaster, newIdentNode("allow_empty"), newIdentNode("original_name"), newIdentNode("desired_name"), newIdentNode("desired_type")]
                let nCallMkPage = newCall(nCallMkPageFn, nCallArgs)

                nCallMkPage

            nAllowPrefix.add(newIdentNode("not"))
            nAllowPrefix.add(newIdentNode("allow_empty"))

            nElifBranch.add(nAllowPrefix)

            if nPageReturnType.strVal != ObjectNDArray.name:
                nElifBranch.add(statements(nPageReturnType, "fnCast"))
                nElse.add(statements(newIdentNode(ObjectNDArray.name), "fnPyCast"))
                nIfStmt.add(nElifBranch)
                nIfStmt.add(nElse)
                nBody.add(nIfStmt)
            else:
                nBody.add(statements(nPageReturnType, "fnCast"))

        let nArgs = [newIdentNode(BaseNDArray.name), nArgReturnType, nInPage, nMaskDef, nReasonListDef, nAllowEmpty, nOriginaName, nDesiredName, nDesiredType]
        let nNewProc = newProc(postfix(newIdentNode("castType"), "*"), nArgs, nBody)

        nProcNodes.add(nNewProc)
        nNewProc[2] = nGeneric

    return nProcNodes

mkPageCaster(bool): page
mkPageCaster(int): page
mkPageCaster(float): page
mkPageCaster(FromDate): page
mkPageCaster(FromDateTime): page
mkPageCaster(string): (if allow_empty: page else: UnicodeNDArray.makePage(page, mask, reason_lst, string.fnCast(R), allow_empty, original_name, desired_name, desired_type))
mkPageCaster(PY_ObjectND): ObjectNDArray.makePage(page, mask, reason_lst, PY_ObjectND.fnCast(R), allow_empty, original_name, desired_name, desired_type)

template convertBasicPage*[T](page: T, desired_type: KindObjectND, mask: var seq[Mask], reason_lst: var seq[string], allow_empty: bool, original_name: string, desired_name: string): BaseNDArray =
    case desired_type:
    of K_BOOLEAN: bool.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_INT: int.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_FLOAT: float.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_STRING: string.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_DATE: ToDate.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_TIME: ToTime.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    of K_DATETIME: ToDateTime.castType(page, mask, reason_lst, allow_empty, original_name, desired_name, desired_type)
    else: corrupted()
