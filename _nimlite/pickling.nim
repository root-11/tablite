from std/endians import bigEndian16, bigEndian32, bigEndian64

const PKL_BINPUT = 'q'
const PKL_LONG_BINPUT = 'r'
const PKL_TUPLE1 = '\x85'
const PKL_TUPLE2 = '\x86'
const PKL_TUPLE3 = '\x87'
const PKL_TUPLE = 't'
const PKL_PROTO = '\x80'
const PKL_GLOBAL = 'c'
const PKL_BININT1 = 'K'
const PKL_BININT2 = 'M'
const PKL_BININT = 'J'
const PKL_SHORT_BINBYTES = 'C'
const PKL_REDUCE = 'R'
const PKL_MARK = '('
const PKL_BINUNICODE = 'X'
const PKL_NEWFALSE = '\x89'
const PKL_NEWTRUE = '\x88'
const PKL_NONE = 'N'
const PKL_BUILD = 'b'
const PKL_EMPTY_LIST = ']'
const PKL_STOP = '.'
const PKL_APPENDS = 'e'
const PKL_BINFLOAT = 'G'

type PY_NoneType* = object
let PY_None* = PY_NoneType()

type PY_Date* = object
    year: uint16
    month, day: uint8


type PY_Time* = object
    hour, minute, second: uint8
    microsecond: uint32
    has_tz: bool
    tz_days, tz_seconds, tz_microseconds: int32

type PY_DateTime* = object
    date: PY_Date
    time: PY_Time

proc newPyDate*(year: uint16, month, day: uint8): PY_Date =
    PY_Date(year: year, month: month, day: day)

proc newPyTime*(hour: uint8, minute: uint8, second: uint8, microsecond: uint32): PY_Time =
    return PY_Time(hour: hour, minute: minute, second: second, microsecond: microsecond)

proc newPyTime*(hour: uint8, minute: uint8, second: uint8, microsecond: uint32, tz_days: int32, tz_seconds: int32, tz_microseconds: int32): PY_Time =
    if tz_days == 0 and tz_seconds == 0:
        return newPyTime(hour, minute, second, microsecond)

    return PY_Time(
            hour: hour, minute: minute, second: second, microsecond: microsecond,
            has_tz: true,
            tz_days: tz_days, tz_seconds: tz_seconds, tz_microseconds: tz_microseconds
        )

proc newPyDateTime*(date: PY_Date, time: PY_Time): PY_DateTime =
    PY_DateTime(date: date, time: time)

proc writePickleBinput(fh: ptr File, binput: var uint32): void =
    if binput <= 0xff:
        fh[].write(PKL_BINPUT)
        discard fh[].writeBuffer(binput.unsafeAddr, 1)
        inc binput
        return

    fh[].write(PKL_LONG_BINPUT)
    discard fh[].writeBuffer(binput.unsafeAddr, 4)
    inc binput

proc writePickleGlobal(fh: ptr File, module_name: string, import_name: string): void =
    fh[].write(PKL_GLOBAL)

    fh[].write(module_name)
    fh[].write('\x0A')

    fh[].write(import_name)
    fh[].write('\x0A')

proc writePickleProto(fh: ptr File): void =
    fh[].write(PKL_PROTO)
    fh[].write("\3")

proc writePickleBinintGeneric[T: uint8|uint16|uint32](fh: ptr File, value: T): void =
    when T is uint8:
        fh[].write(PKL_BININT1)
        discard fh[].writeBuffer(value.unsafeAddr, 1)
    when T is uint16:
        fh[].write(PKL_BININT2)
        discard fh[].writeBuffer(value.unsafeAddr, 2)
    when T is uint32:
        fh[].write(PKL_BININT)
        discard fh[].writeBuffer(value.unsafeAddr, 4)

proc writePickleBinfloat(fh: ptr File, value: float): void =
    # pickle stores floats big-endian
    var f: float

    f.unsafeAddr.bigEndian64(value.unsafeAddr)

    fh[].write(PKL_BINFLOAT)
    discard fh[].writeBuffer(f.unsafeAddr, 8)

proc writePickleBinint[T: int|uint|int32|uint32](fh: ptr File, value: T): void =
    when T is int or T is int32:
        if value < 0:
            fh.writePickleBinintGeneric(uint32 value)
            return

    if value <= 0xff:
        fh.writePickleBinintGeneric(uint8 value)
        return

    if value <= 0xffff:
        fh.writePickleBinintGeneric(uint16 value)
        return

    fh.writePickleBinintGeneric(uint32 value)

proc writePickleShortbinbytes(fh: ptr File, value: string): void =
    fh[].write(PKL_SHORT_BINBYTES)

    let len = value.len()

    discard fh[].writeBuffer(len.unsafeAddr, 1)
    discard fh[].writeBuffer(value[0].unsafeAddr, value.len)

proc writePickleBinunicode(fh: ptr File, value: string): void =
    let len = uint32 value.len
    # try:

    fh[].write(PKL_BINUNICODE)
    discard fh[].writeBuffer(len.unsafeAddr, 4)
    fh[].write(value)
    # discard fh[].writeBuffer(value[0].unsafeAddr, len)
    # finally:
    #     echo "failed to dump: '" & value & "' of size " & $len

proc writePickleBoolean(fh: ptr File, value: bool): void =
    if value == true:
        fh[].write(PKL_NEWTRUE)
    else:
        fh[].write(PKL_NEWFALSE)

proc writePickleDateBody(fh: ptr File, value: ptr PY_Date, binput: var uint32): void =
    var year: uint16
    year.unsafeAddr.bigEndian16(value.year.unsafeAddr)

    discard fh[].writeBuffer(year.unsafeAddr, 2)
    discard fh[].writeBuffer(value.month.unsafeAddr, 1)
    discard fh[].writeBuffer(value.day.unsafeAddr, 1)

proc writePickleDate(fh: ptr File, value: PY_Date, binput: var uint32): void =
    fh.writePickleGlobal("datetime", "date")
    fh.writePickleBinput(binput)
    fh[].write(PKL_SHORT_BINBYTES)
    fh[].write('\4') # date has 4 bytes 2(y)-1(m)-1(d)

    fh.writePickleDateBody(value.unsafeAddr, binput)

    fh.writePickleBinput(binput)
    fh[].write(PKL_TUPLE1)
    fh.writePickleBinput(binput)
    fh[].write(PKL_REDUCE)
    fh.writePickleBinput(binput)

proc writePickleTimeBody(fh: ptr File, value: ptr PY_Time, binput: var uint32): void =
    var microsecond: uint32
    microsecond.unsafeAddr.bigEndian32(value.microsecond.unsafeAddr)

    var ptr_microseconds = cast[pointer](cast[int](microsecond.unsafeAddr) + 1)

    discard fh[].writeBuffer(value.hour.unsafeAddr, 1)
    discard fh[].writeBuffer(value.minute.unsafeAddr, 1)
    discard fh[].writeBuffer(value.second.unsafeAddr, 1)
    discard fh[].writeBuffer(ptr_microseconds, 3)
    fh.writePickleBinput(binput)

    if not value.has_tz:
        fh[].write(PKL_TUPLE1)
    else:
        fh.writePickleGlobal("datetime", "timezone")
        fh.writePickleBinput(binput)
        fh.writePickleGlobal("datetime", "timedelta")
        fh.writePickleBinput(binput)
        fh.writePickleBinint(value.tz_days)
        fh.writePickleBinint(value.tz_seconds)
        fh.writePickleBinint(value.tz_microseconds)
        fh[].write(PKL_TUPLE3)
        fh.writePickleBinput(binput)
        fh[].write(PKL_REDUCE)
        fh.writePickleBinput(binput)
        fh[].write(PKL_TUPLE1)
        fh.writePickleBinput(binput)
        fh[].write(PKL_REDUCE)
        fh.writePickleBinput(binput)
        fh[].write(PKL_TUPLE2)

    fh.writePickleBinput(binput)
    fh[].write(PKL_REDUCE)
    fh.writePickleBinput(binput)

proc writePickleTime(fh: ptr File, value: PY_Time, binput: var uint32): void =
    fh.writePickleGlobal("datetime", "time")
    fh.writePickleBinput(binput)
    fh[].write(PKL_SHORT_BINBYTES)
    fh[].write('\6')

    fh.writePickleTimeBody(value.unsafeAddr, binput)

proc writePickleDatetime(fh: ptr File, value: PY_DateTime, binput: var uint32): void =
    fh.writePickleGlobal("datetime", "datetime")
    fh.writePickleBinput(binput)
    fh[].write(PKL_SHORT_BINBYTES)
    fh[].write('\10')

    fh.writePickleDateBody(value.date.unsafeAddr, binput)
    fh.writePickleTimeBody(value.time.unsafeAddr, binput)

proc writePicklePyObj*[T: int|float|PY_NoneType|string|bool|PY_Date|PY_Time|PY_DateTime](fh: ptr File, value: T, binput: var uint32): void =
    when T is PY_NoneType:
        fh[].write(PKL_NONE)
        return
    when T is int:
        fh.writePickleBinint(value)
        return
    when T is float:
        fh.writePickleBinfloat(value)
        return
    when T is string:
        fh.writePickleBinunicode(value)
        return
    when T is bool:
        fh.writePickleBoolean(value)
        return
    when T is PY_Date:
        fh.writePickleDate(value, binput)
        return
    when T is PY_Time:
        fh.writePickleTime(value, binput)
        return
    when T is PY_DateTime:
        fh.writePickleDatetime(value, binput)
        return
    raise newException(Exception, "not implemented error: " & $value)

proc writePickleStart*(fh: ptr File, binput: var uint32, elem_count: uint): void =
    binput = 0

    fh.writePickleProto()
    fh.writePickleGlobal("numpy.core.multiarray", "_reconstruct")
    fh.writePickleBinput(binput)
    fh.writePickleGlobal("numpy", "ndarray")
    fh.writePickleBinput(binput)
    fh.writePickleBinint(0)
    fh[].write(PKL_TUPLE1)
    fh.writePickleBinput(binput)
    fh.writePickleShortbinbytes("b")
    fh.writePickleBinput(binput)
    fh[].write(PKL_TUPLE3)
    fh.writePickleBinput(binput)
    fh[].write(PKL_REDUCE)
    fh.writePickleBinput(binput)
    fh[].write(PKL_MARK)

    if true:
        fh.writePickleBinint(1)
        fh.writePickleBinint(elem_count)
        fh[].write(PKL_TUPLE1)
        fh.writePickleBinput(binput)
        fh.writePickleGlobal("numpy", "dtype")
        fh.writePickleBinput(binput)
        fh.writePickleBinunicode("O8")
        fh.writePickleBinput(binput)
        fh.writePickleBoolean(false)
        fh.writePickleBoolean(true)
        fh[].write(PKL_TUPLE3)
        fh.writePickleBinput(binput)
        fh[].write(PKL_REDUCE)
        fh.writePickleBinput(binput)
        fh[].write(PKL_MARK)

        if true:
            fh.writePickleBinint(3)
            fh.writePickleBinunicode("|")
            fh.writePickleBinput(binput)
            fh[].write(PKL_NONE)
            fh[].write(PKL_NONE)
            fh[].write(PKL_NONE)
            fh.writePickleBinint(-1)
            fh.writePickleBinint(-1)
            fh.writePickleBinint(63)
            fh[].write(PKL_TUPLE)

        fh.writePickleBinput(binput)
        fh[].write(PKL_BUILD)
        fh.writePickleBoolean(false)
        fh[].write(PKL_EMPTY_LIST)
        fh.writePickleBinput(binput)

        # now we dump objects

        if elem_count > 0:
            fh[].write(PKL_MARK)

        # fh[].write(PKL_APPENDS)

proc writePickleFinish*(fh: ptr File, binput: var uint32, elem_count: uint): void =
    if elem_count > 0:
        fh[].write(PKL_APPENDS)

    fh[].write(PKL_TUPLE)
    fh.writePickleBinput(binput)
    fh[].write(PKL_BUILD)
    fh[].write(PKL_STOP)
