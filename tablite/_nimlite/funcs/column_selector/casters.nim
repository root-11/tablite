import std/[macros, times]
from std/typetraits import name
import ../../pytypes
from ../../utils import implement
import ../../dateutils
import ../../infertypes as infer

type
    ToDate* = object
    ToDateTime* = object
    FromDate* = object
    FromDateTime* = object
    ToTime* = object

proc newMkCaster(caster: NimNode, isPyCaster: bool): NimNode =
    expectKind(caster, nnkLambda)
    expectLen(caster.params, 2)

    let castR = caster.params[0]
    let castT = caster.params[1][1]
    let nameCastT = caster.params[1][0]

    let descT = newNimNode(nnkBracketExpr)
    let paramsT = newIdentDefs(newIdentNode("T"), descT)

    var trueCastR: NimNode
    var trueCastT: NimNode

    if castR.strVal in [ToDate.name, ToDateTime.name]:
        trueCastR = newIdentNode(DateTime.name)
    elif castR.strVal == ToTime.name:
        trueCastR = newIdentNode(PY_Time.name)
    elif castR.strVal in [bool.name, int.name, float.name, string.name]:
        trueCastR = castR
    else:
        raise newException(FieldDefect, "Uncastable return type '" & $castR.strVal & "'")

    if castT.strVal in [FromDate.name, FromDateTime.name]:
        trueCastT = newIdentDefs(nameCastT, newIdentNode(DateTime.name))
    elif castT.strVal in [bool.name, int.name, float.name, string.name, PY_Time.name, PY_ObjectND.name]:
        trueCastT = newIdentDefs(nameCastT, castT)
    else:
        raise newException(FieldDefect, "Uncastable value type '" & $castT.strVal & "'")

    descT.add(newIdentNode("typedesc"))
    descT.add(castT)

    let descR = newNimNode(nnkBracketExpr)
    let paramsR = newIdentDefs(newIdentNode("R"), descR)

    descR.add(newIdentNode("typedesc"))
    descR.add(castR)

    if isPyCaster:
        let subProc = newProc(params = [newIdentNode(PY_ObjectND.name), trueCastT], body = caster.body, procType = nnkLambda)
        return newProc(postfix(newIdentNode("fnPyCast"), "*"), params = [newNimNode(nnkProcTy), paramsT, paramsR], body = subProc)
    else:
        let subProc = newProc(params = [trueCastR, trueCastT], body = caster.body, procType = nnkLambda)
        return newProc(postfix(newIdentNode("fnCast"), "*"), params = [newNimNode(nnkProcTy), paramsT, paramsR], body = subProc)

macro mkCasters(converters: untyped) =
    expectKind(converters, nnkStmtList)
    expectKind(converters[0], nnkLambda)

    let params = converters[0].params
    let body = converters[0].body
    let identity = params[1]

    expectKind(identity, nnkIdentDefs)

    let nodes = newNimNode(nnkStmtList)

    for cvtr in body:
        expectKind(cvtr, nnkAsgn)
        expectKind(cvtr[0], nnkIdent)

        let toType = cvtr[0]
        let toBody = cvtr[1]
        let toFunc = newProc(params = [toType, identity], body = toBody, procType = nnkLambda)

        nodes.add(newMkCaster(toFunc, false))

        case toType.strVal:
        of bool.name, int.name, float.name, string.name:
            let tBodyPy = newCall(newIdentNode("newPY_Object"), toBody)
            let toFuncPy = newProc(params = [toType, identity], body = tBodyPy, procType = nnkLambda)
            let nToPy = newMkCaster(toFuncPy, true)

            nodes.add(nToPy)
        of ToDate.name, ToDateTime.name, ToTime.name:
            var kind {.noinit.}: KindObjectND

            case toType.strVal:
            of ToDate.name: kind = KindObjectND.K_DATE
            of ToDateTime.name: kind = KindObjectND.K_DATETIME
            of ToTime.name: kind = KindObjectND.K_TIME

            let tBodyPy = newCall(newIdentNode("newPY_Object"), toBody, newIdentNode($kind))
            let toFuncPy = newProc(params = [toType, identity], body = tBodyPy, procType = nnkLambda)
            let nToPy = newMkCaster(toFuncPy, true)

            nodes.add(nToPy)
        else:
            implement(toType.strVal)

    return nodes

mkCasters:
    proc (v: bool) =
        bool = v
        int = int v
        float = float v
        string = (if v: "True" else: "False")
        ToDate = days2Date(int v)
        ToDateTime = delta2Date(seconds = int v)
        ToTime = secondsToPY_Time(float v)

mkCasters:
    proc (v: int) =
        bool = v >= 1
        int = v
        float = float v
        string = $v
        ToDate = v.days2Date
        ToDateTime = delta2Date(seconds = v)
        ToTime = secondsToPY_Time(float v)

mkCasters:
    proc (v: float) =
        bool = v >= 1
        int = int v
        float = v
        string = $v
        ToDate = days2Date(int v)
        ToDateTime = seconds2Date(v)
        ToTime = secondsToPY_Time(v)

mkCasters:
    proc(v: FromDate) =
        bool = v.toTime.time2Duration.inSeconds >= 1
        int = v.toTime.time2Duration.inSeconds
        float = float v.toTime.time2Duration.inSeconds
        string = v.format(fmtDate)
        ToDate = v
        ToDateTime = v
        ToTime = v.newPY_Time

mkCasters:
    proc (v: PY_Time) =
        bool = v.value.inSeconds >= 1
        int = v.value.inSeconds
        float = v.value.duration2Seconds
        string = $v.value.duration2Time
        ToDate = v.value.duration2Date
        ToDateTime = v.value.duration2Date
        ToTime = v

mkCasters:
    proc (v: FromDateTime) =
        bool = v.toTime.time2Duration.inSeconds >= 1
        int = v.toTime.time2Duration.inSeconds
        float = v.toTime.time2Duration.duration2Seconds
        string = v.format(fmtDateTime)
        ToDate = v.datetime2Date
        ToDateTime = v
        ToTime = v.newPY_Time

mkCasters:
    proc (v: string) =
        bool = infer.inferBool(addr v)
        int = int infer.inferFloat(addr v)
        float = infer.inferFloat(addr v)
        string = v
        ToDate = infer.inferDate(addr v).value
        ToDateTime = infer.inferDatetime(addr v).value
        ToTime = infer.inferTime(addr v)

template obj2prim(v: PY_ObjectND) =
    case v.kind:
    of K_BOOLEAN: bool.fnCast(R)(PY_Boolean(v).value)
    of K_INT: int.fnCast(R)(PY_Int(v).value)
    of K_FLOAT: float.fnCast(R)(PY_Float(v).value)
    of K_STRING: string.fnCast(R)(PY_String(v).value)
    of K_DATE: FromDate.fnCast(R)(PY_Date(v).value)
    of K_TIME: PY_Time.fnCast(R)(PY_Time(v))
    of K_DATETIME: FromDateTime.fnCast(R)(PY_Date(v).value)
    of K_NONETYPE: raise newException(ValueError, "cannot cast")

mkCasters:
    proc(v: PY_ObjectND) =
        bool = v.obj2prim()
        int = v.obj2prim()
        float = v.obj2prim()
        string = v.obj2prim()
        ToDate = v.obj2prim()
        ToDateTime = v.obj2prim()
        ToTime = v.obj2prim()