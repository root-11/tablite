from std/tables import Table
from std/strutils import replace
from std/times import DateTime, Duration, initTimeFormat, format

from ../utils import implement
from ../dateutils import duration2Date

const fmtDate* = initTimeFormat("yyyy-MM-dd")
const fmtDateTime* = initTimeFormat("yyyy-MM-dd HH:mm:ss")
const fmtTime* = initTimeFormat("HH:mm:ss")

type KindObjectND* = enum
    K_NONETYPE,
    K_BOOLEAN,
    K_INT,
    K_FLOAT,
    K_STRING,
    K_DATE,
    K_TIME,
    K_DATETIME

type Shape* = seq[int]
type
    PyObj = object of RootObj
    PY_Object* = ref PyObj
type
    PyObjND {.requiresInit.} = object of PyObj
        kind*: KindObjectND
    PY_ObjectND* = ref PyObjND
type PY_Boolean* = ref object of PY_ObjectND
    value*: bool
type PY_Int* = ref object of PY_ObjectND
    value*: int
type PY_Float* = ref object of PY_ObjectND
    value*: float
type PY_String* = ref object of PY_ObjectND
    value*: string
type PY_Bytes* = ref object of PY_Object
    value*: seq[char]
type PY_NoneType* = ref object of PY_ObjectND
type PY_Date* = ref object of PY_ObjectND
    value*: DateTime
type PY_Time* = ref object of PY_ObjectND
    value*: Duration
type PY_DateTime* = ref object of PY_ObjectND
    value*: DateTime
type Py_Iterable* = ref object of PY_Object
    elems*: seq[PY_Object]
type Py_Tuple* = ref object of Py_Iterable
type Py_List* = ref object of Py_Iterable
type Py_Set* = ref object of Py_Iterable
type Py_Dict* = ref object of Py_Object
    elems*: Table[PY_Object, PY_Object]

let PY_None* = PY_NoneType(kind: KindObjectND.K_NONETYPE)

method toRepr*(self: PY_ObjectND): string {.base, inline.} = implement("PY_ObjectND.`$` must be implemented by inheriting class: " & $self.kind)
method toRepr*(self: PY_NoneType): string = "None"
method toRepr*(self: PY_Boolean): string = $self.value
method toRepr*(self: PY_Int): string = $self.value
method toRepr*(self: PY_Float): string = $self.value
method toRepr*(self: PY_String): string = self.value
method toRepr*(self: PY_Date): string = self.value.format(fmtDate)
method toRepr*(self: PY_Time): string = self.value.duration2Date.format(fmtTime)
method toRepr*(self: PY_DateTime): string = self.value.format(fmtDateTime)

method `$`*(self: PY_ObjectND): string {.base, inline.} = implement("PY_ObjectND.`$` must be implemented by inheriting class: " & $self.kind)
method `$`*(self: PY_NoneType): string = "None"
method `$`*(self: PY_Boolean): string = self.toRepr
method `$`*(self: PY_Int): string = self.toRepr & "i"
method `$`*(self: PY_Float): string = self.toRepr & "f"
method `$`*(self: PY_String): string = "\"" & self.toRepr.replace("\"", "\\\"") & "\""
method `$`*(self: PY_Date): string = "Date(" & self.toRepr & ")"
method `$`*(self: PY_Time): string = "Time(" & self.toRepr & ")"
method `$`*(self: PY_DateTime): string = "DateTime(" & self.toRepr & ")"
