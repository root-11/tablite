import std/[tables]
import nimpy as nimpy
from std/sugar import collect
from ../../pymodules import builtins
from ../../pytypes import KindObjectND, str2ObjKind

type ColSliceInfo* = (string, string)
type ColInfo* = Table[string, ColSliceInfo]
type DesiredColumnInfo* = object
    originalName*: string
    `type`*: KindObjectND
    allowEmpty*: bool

proc newDesiredColumnInfo*(name: string, `type`: KindObjectND, allowEmpty: bool): DesiredColumnInfo =
    DesiredColumnInfo(
        originalName: name,
        `type`: `type`,
        allowEmpty: allowEmpty
    )

proc toPyObj*(infos: var OrderedTable[string, DesiredColumnInfo]): nimpy.PyObject =
    let elems = collect:
        for (name, info) in infos.pairs:
            (name, (info.originalName, $info.`type`, info.allowEmpty))

    let res = builtins().dict(elems)

    return res

proc fromPyObjToDesiredInfos*(pyInfos: nimpy.PyObject): OrderedTable[string, DesiredColumnInfo] =
    let res = collect(initOrderedTable()):
        for k in pyInfos:
            let pyInfo = pyInfos[k]
            let (pyName, pyType, pyAllowEmpty) = (pyInfo[0], pyInfo[1], pyInfo[2])
            let info = newDesiredColumnInfo(
                pyName.to(string),
                str2ObjKind(pyType.to(string)),
                pyAllowEmpty.to(bool)
            )

            {k.to(string): info}

    return res