from std/tables import Table
import std/[times, strutils]
import dateutils
from ./utils import implement
import ./pytypes/[pyobjs, pycmp]

export pyobjs
export pycmp

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


# method `>`*(self: PY_ObjectND, other: PY_ObjectND): bool {.base, inline.} = implement("PY_ObjectND.`>` must be implemented by inheriting class: " & $self.kind)
# method `>`*(self: PY_NoneType, other: PY_ObjectND): bool = isSameType(self, other)
# method `>`*(self: PY_Boolean, other: PY_ObjectND): bool = self.isSameType(other) and PY_Boolean(other).value > self.value
# method `>`*(self: PY_Int, other: PY_ObjectND): bool = self.isSameType(other) and PY_Int(other).value > self.value or other.kind > K_FLOAT and PY_Float(other).value > float(self.value)
# method `>`*(self: PY_Float, other: PY_ObjectND): bool = self.isSameType(other) and PY_Float(other).value > self.value or other.kind > K_INT and float(PY_Int(other).value) > self.value
# method `>`*(self: PY_String, other: PY_ObjectND): bool = self.isSameType(other) and PY_String(other).value > self.value
# method `>`*(self: PY_Date, other: PY_ObjectND): bool = self.isSameType(other) and PY_Date(other).value > self.value or other.kind > K_DATETIME and PY_DateTime(other).value > self.value
# method `>`*(self: PY_Time, other: PY_ObjectND): bool = self.isSameType(other) and PY_Time(other).value > self.value
# method `>`*(self: PY_DateTime, other: PY_ObjectND): bool = self.isSameType(other) and PY_DateTime(other).value > self.value or other.kind > K_DATE and PY_Date(other).value > self.value


# method `==`*(self: PY_ObjectND, other: PY_ObjectND): bool {.base, inline.} = implement("PY_ObjectND.`==` must be implemented by inheriting class: " & $self.kind)
# method `==`*(self: PY_NoneType, other: PY_ObjectND): bool = isSameType(self, other)
# method `==`*(self: PY_Boolean, other: PY_ObjectND): bool = self.isSameType(other) and PY_Boolean(other).value == self.value
# method `==`*(self: PY_Int, other: PY_ObjectND): bool = self.isSameType(other) and PY_Int(other).value == self.value or other.kind == K_FLOAT and PY_Float(other).value == float(self.value)
# method `==`*(self: PY_Float, other: PY_ObjectND): bool = self.isSameType(other) and PY_Float(other).value == self.value or other.kind == K_INT and float(PY_Int(other).value) == self.value
# method `==`*(self: PY_String, other: PY_ObjectND): bool = self.isSameType(other) and PY_String(other).value == self.value
# method `==`*(self: PY_Date, other: PY_ObjectND): bool = self.isSameType(other) and PY_Date(other).value == self.value or other.kind == K_DATETIME and PY_DateTime(other).value == self.value
# method `==`*(self: PY_Time, other: PY_ObjectND): bool = self.isSameType(other) and PY_Time(other).value == self.value
# method `==`*(self: PY_DateTime, other: PY_ObjectND): bool = self.isSameType(other) and PY_DateTime(other).value == self.value or other.kind == K_DATE and PY_Date(other).value == self.value

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

proc getHour*(self: ptr PY_Time | PY_Time): HourRange = HourRange self.value.inHours() mod 24
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
proc newPY_Object*(v: Duration): PY_ObjectND {.inline.} = PY_Time(value: v, kind: K_TIME)
proc newPY_Object*(v: DateTime, k: KindObjectND): PY_ObjectND {.inline.} =
    case k:
    of K_DATE: return PY_Date(value: v, kind: k)
    of K_DATETIME: return PY_DateTime(value: v, kind: k)
    of K_TIME: return newPY_Time(v)
    else: raise newException(Exception, "invalid date type: " & $k)
