from std/tables import Table
import std/times
import dateutils

const fmtDate* = initTimeFormat("yyyy-MM-dd")
const fmtDateTime* = initTimeFormat("yyyy-MM-dd HH:mm:ss")

type KindObjectND* = enum
    K_NONETYPE,
    K_BOOLEAN,
    K_INT,
    K_FLOAT,
    K_STRING,
    K_DATE,
    K_TIME,
    K_DATETIME

proc str2ObjKind*(val: string): KindObjectND =
    case val:
    of $K_NONETYPE: K_NONETYPE
    of $K_BOOLEAN: K_BOOLEAN
    of $K_INT: K_INT
    of $K_FLOAT: K_FLOAT
    of $K_STRING: K_STRING
    of $K_DATE: K_DATE
    of $K_TIME: K_TIME
    of $K_DATETIME: K_DATETIME
    else: raise newException(ValueError, "invalid object kind: " & val)

type Shape* = seq[int]
type PY_Object* = ref object of RootObj
type PY_ObjectND* {.requiresInit.} = ref object of PY_Object
    kind*: KindObjectND
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

proc toRepr*(self: PY_ObjectND): string {.inline.} =
    case self.kind:
    of K_NONETYPE: "None"
    of K_BOOLEAN: $PY_Boolean(self).value
    of K_INT: $PY_Int(self).value
    of K_FLOAT: $PY_Float(self).value
    of K_STRING: $PY_String(self).value
    of K_DATE: PY_Date(self).value.format(fmtDate)
    of K_TIME: $PY_Time(self).value
    of K_DATETIME: PY_DateTime(self).value.format(fmtDateTime)

proc newPY_Date*(year: uint16, month, day: uint8): PY_Date {.inline.} = PY_Date(value: date2NimDatetime(int year, int month, int day), kind: K_DATE)

proc newPY_DateTime*(date: PY_Date, time: PY_Time): PY_DateTime = PY_DateTime(value: date.value + time.value, kind: K_DATETIME)
proc newPY_DateTime*(year: uint16, month, day, hour, minute, second: uint8, microsecond: uint32): PY_DateTime {.inline.} =
    return PY_DateTime(
        value: datetime2NimDatetime(
            int year, int month, int day,
            int hour, int minute, int second, int microsecond
        ), kind: K_DATETIME
    )

proc secondsToPY_Time*(seconds: float): PY_Time = PY_Time(value: seconds2Duration(seconds), kind: K_TIME)

proc newPY_Time*(hour, minute, second: uint8, microsecond: uint32, tz_days, tz_seconds, tz_microseconds: int32): PY_Time {.inline.} =
    let dur = initDuration(days = tz_days, seconds = tz_seconds, microseconds = tz_microseconds)
    let time = time2NimDuration(int hour, int minute, int second, int microsecond).duration2Time
    let delta = time2Duration(time - dur)

    return PY_Time(value: delta, kind: K_TIME)

proc newPY_Time*(hour, minute, second: uint8, microsecond: uint32): PY_Time {.inline.} =
    return PY_Time(
        value: time2NimDuration(
            int hour, int minute, int second, int microsecond
        ), kind: K_TIME
    )


proc newPY_Time*(date: DateTime): PY_Time = PY_Time(value: date.toTime.time2Duration, kind: K_TIME)

proc `$`*(self: PY_ObjectND): string {.inline.} = "PY_ObjectND"
proc `$`*(self: PY_Date): string {.inline.} = "Date(" & self.toRepr & ")"
proc `$`*(self: PY_Time): string {.inline.} = "Time(" & self.toRepr & ")"
proc `$`*(self: PY_DateTime): string {.inline.} = "DateTime(" & self.toRepr & ")"

proc calcShapeElements*(shape: var Shape): int {.inline.} =
    var elements = 1

    for m in shape:
        elements = elements * m

    return elements

proc getYear*(self: ptr PY_Date | PY_Date | ptr PY_DateTime | PY_DateTime): YearRange = YearRange self.value.year()
proc getMonth*(self: ptr PY_Date | PY_Date | ptr PY_DateTime | PY_DateTime): Month = self.value.month()
proc getDay*(self: ptr PY_Date | PY_Date | ptr PY_DateTime | PY_DateTime): MonthdayRange = self.value.monthday()

proc getHour*(self: ptr PY_DateTime | PY_DateTime): HourRange = self.value.hour()
proc getMinute*(self: ptr PY_DateTime | PY_DateTime): MinuteRange = self.value.minute()
proc getSecond*(self: ptr PY_DateTime | PY_DateTime): SecondRange = self.value.second()
proc getMicrosecond*(self: ptr PY_DateTime | PY_DateTime): MicrosecondRange = MicrosecondRange (self.value.nanosecond() / 1000)

proc getHour*(self: ptr PY_Time | PY_Time): HourRange = HourRange self.value.inHours()
proc getMinute*(self: ptr PY_Time | PY_Time): MinuteRange =
    let inHours = self.value.inHours() * 60
    let inMinutes = self.value.inMinutes()

    return MinuteRange (inMinutes - inHours)

proc getSecond*(self: ptr PY_Time | PY_Time): SecondRange =
    let inMinutes = self.value.inMinutes() * 60
    let inSeconds = self.value.inSeconds()

    return SecondRange (inSeconds - inMinutes)

proc getMicrosecond*(self: ptr PY_Time | PY_Time): MicrosecondRange =
    let inSeconds = self.value.inSeconds() * 1_000_000
    let inMicroseconds = self.value.inMicroseconds()

    return MicrosecondRange (inMicroseconds - inSeconds)

proc newPY_Object*(): PY_ObjectND {.inline.} = PY_None
proc newPY_Object*(v: bool): PY_ObjectND {.inline.} = PY_Boolean(value: v, kind: K_BOOLEAN)
proc newPY_Object*(v: int): PY_ObjectND {.inline.} = PY_Int(value: v, kind: K_INT)
proc newPY_Object*(v: float): PY_ObjectND {.inline.} = PY_Float(value: v, kind: K_FLOAT)
proc newPY_Object*(v: string): PY_ObjectND {.inline.} = PY_String(value: v, kind: K_STRING)
proc newPY_Object*(v: DateTime, k: KindObjectND): PY_ObjectND {.inline.} =
    case k:
    of K_DATE: return PY_Date(value: v, kind: k)
    of K_DATETIME: return PY_DateTime(value: v, kind: k)
    of K_TIME: return newPY_Time(v)
    else: raise newException(Exception, "invalid date type: " & $k)
