import std/[macros, tables]
from std/typetraits import name
from std/sugar import collect
import ../../numpy
from ../../pytypes import PY_ObjectND, KindObjectND
import casters
from mask import Mask
from makepage import makePage, canBeNone
from ../../utils import corrupted, implement

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
    let nReasonListDef = newIdentDefs(newIdentNode("reasonLst"), nReasonListVarTy)

    let nAllowEmpty = newIdentDefs(newIdentNode("allowEmpty"), bindSym(bool.name))
    let nOriginaName = newIdentDefs(newIdentNode("originalName"), bindSym(string.name))
    let nDesiredName = newIdentDefs(newIdentNode("desiredName"), bindSym(string.name))
    let nDesiredType = newIdentDefs(newIdentNode("desiredType"), bindSym(KindObjectND.name))

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
    let pragmas = newNimNode(nnkPragma).add(newIdentNode("inline"))

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
            let nAllowPrefix = prefix(newIdentNode("allowEmpty"), "not")
            let nCheckNone = prefix(newCall(newIdentNode("canBeNone"), newIdentNode("page")), "not")
            let nElse = newNimNode(nnkElse)

            template statements(nPageReturnType: NimNode, fnName: string): NimNode =
                let nCallCasterFn = newDotExpr(nBaseType, newIdentNode(fnName))
                let nCallCaster = newCall(nCallCasterFn, newIdentNode("R"))

                let nCallMkPageFn = newDotExpr(nPageReturnType, newIdentNode("makePage"))
                let nCallArgs = [newIdentNode("page"), newIdentNode("mask"), newIdentNode("reasonLst"), nCallCaster, newIdentNode("allowEmpty"), newIdentNode("originalName"), newIdentNode("desiredName"), newIdentNode("desiredType")]
                let nCallMkPage = newCall(nCallMkPageFn, nCallArgs)

                nCallMkPage

            nElifBranch.add(infix(nAllowPrefix, "or", nCheckNone))

            if not (nPageReturnType.strVal in [ObjectNDArray.name, UnicodeNDArray.name]):
                nElifBranch.add(statements(nPageReturnType, "fnCast"))
                nElse.add(statements(newIdentNode(ObjectNDArray.name), "fnPyCast"))
                nIfStmt.add(nElifBranch)
                nIfStmt.add(nElse)
                nBody.add(nIfStmt)
            else:
                nBody.add(statements(nPageReturnType, "fnCast"))

        let nArgs = [newIdentNode(BaseNDArray.name), nArgReturnType, nInPage, nMaskDef, nReasonListDef, nAllowEmpty, nOriginaName, nDesiredName, nDesiredType]
        let nNewProc = newProc(postfix(newIdentNode("castType"), "*"), nArgs, nBody, pragmas=pragmas)

        nNewProc[2] = nGeneric
        nProcNodes.add(nNewProc)

    return nProcNodes

mkPageCaster(string):
    if not allowEmpty:
        # if we're casting to string and we don't allow empties, process per row
        UnicodeNDArray.makePage(page, mask, reasonLst, string.fnCast(R), allowEmpty, originalName, desiredName, desiredType)
    else:
        when page is ObjectNDArray:
            if page.canBeNone:
                # if input page can be nones, cast per row
                UnicodeNDArray.makePage(page, mask, reasonLst, string.fnCast(R), allowEmpty, originalName, desiredName, desiredType)
            else:
                # if input page can't have nones, return as is
                page
        else:
            # otherwise return the page
            page

mkPageCaster(bool): page
mkPageCaster(int): page
mkPageCaster(float): page
mkPageCaster(FromDate): page
mkPageCaster(FromDateTime): page
mkPageCaster(PY_ObjectND): ObjectNDArray.makePage(page, mask, reasonLst, PY_ObjectND.fnCast(R), allowEmpty, originalName, desiredName, desiredType)

template convertBasicPage*[T](page: T, desiredType: KindObjectND, mask: var seq[Mask], reasonLst: var seq[string], allowEmpty: bool, originalName: string, desiredName: string): BaseNDArray =
    case desiredType:
    of K_BOOLEAN: bool.castType(page, mask, reasonLst, allowEmpty, originalName, desiredName, desiredType)
    of K_INT: int.castType(page, mask, reasonLst, allowEmpty, originalName, desiredName, desiredType)
    of K_FLOAT: float.castType(page, mask, reasonLst, allowEmpty, originalName, desiredName, desiredType)
    of K_STRING: string.castType(page, mask, reasonLst, allowEmpty, originalName, desiredName, desiredType)
    of K_DATE: ToDate.castType(page, mask, reasonLst, allowEmpty, originalName, desiredName, desiredType)
    of K_TIME: ToTime.castType(page, mask, reasonLst, allowEmpty, originalName, desiredName, desiredType)
    of K_DATETIME: ToDateTime.castType(page, mask, reasonLst, allowEmpty, originalName, desiredName, desiredType)
    else: raise newException(FieldDefect, "uncastable type: " & $desiredType)
