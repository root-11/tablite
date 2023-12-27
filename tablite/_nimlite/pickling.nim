from std/endians import bigEndian16, bigEndian32, bigEndian64
import pickleproto, pytypes

# type PY_Date* = object
#     year: uint16
#     month, day: uint8


# type PY_Time* = object
#     hour, minute, second: uint8
#     microsecond: uint32
#     has_tz: bool
#     tz_days, tz_seconds, tz_microseconds: int32

# type PY_DateTime* = object
#     date: PY_Date
#     time: PY_Time

# proc newPyDate*(year: uint16, month, day: uint8): PY_Date =
#     PY_Date(year: year, month: month, day: day)

# proc newPyTime*(hour: uint8, minute: uint8, second: uint8, microsecond: uint32): PY_Time =
#     return PY_Time(hour: hour, minute: minute, second: second, microsecond: microsecond)

# proc newPyTime*(hour: uint8, minute: uint8, second: uint8, microsecond: uint32, tz_days: int32, tz_seconds: int32, tz_microseconds: int32): PY_Time =
#     if tz_days == 0 and tz_seconds == 0:
#         return newPyTime(hour, minute, second, microsecond)

#     return PY_Time(
#             hour: hour, minute: minute, second: second, microsecond: microsecond,
#             has_tz: true,
#             tz_days: tz_days, tz_seconds: tz_seconds, tz_microseconds: tz_microseconds
#         )

# proc newPyDateTime*(date: PY_Date, time: PY_Time): PY_DateTime =
#     PY_DateTime(date: date, time: time)

proc writePickleBinput(fh: var File, binput: var uint32): void {.inline.} =
    if binput <= 0xff:
        fh.write(PKL_BINPUT)
        discard fh.writeBuffer(binput.unsafeAddr, 1)
        inc binput
        return

    fh.write(PKL_LONG_BINPUT)
    discard fh.writeBuffer(binput.unsafeAddr, 4)
    inc binput

proc writePickleGlobal(fh: var File, module_name: string, import_name: string): void {.inline.} =
    fh.write(PKL_GLOBAL)

    fh.write(module_name)
    fh.write(PKL_STRING_TERM)

    fh.write(import_name)
    fh.write(PKL_STRING_TERM)

proc writePickleProto(fh: var File): void =
    fh.write(PKL_PROTO)
    fh.write(PKL_PROTO_VERSION)

proc writePickleBinintGeneric[T: uint8|uint16|uint32](fh: var File, value: T): void {.inline.} =
    when T is uint8:
        fh.write(PKL_BININT1)
        discard fh.writeBuffer(value.unsafeAddr, 1)
    when T is uint16:
        fh.write(PKL_BININT2)
        discard fh.writeBuffer(value.unsafeAddr, 2)
    when T is uint32:
        fh.write(PKL_BININT)
        discard fh.writeBuffer(value.unsafeAddr, 4)

proc writePickleBinfloat(fh: var File, value: float): void =
    # pickle stores floats big-endian
    var f: float

    f.unsafeAddr.bigEndian64(value.unsafeAddr)

    fh.write(PKL_BINFLOAT)
    discard fh.writeBuffer(f.unsafeAddr, 8)

proc writePickleBinint[T: int|uint|int32|uint32](fh: var File, value: T): void {.inline.} =
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

proc writePickleShortbinbytes(fh: var File, value: string): void {.inline.} =
    fh.write(PKL_SHORT_BINBYTES)

    let len = value.len()

    discard fh.writeBuffer(len.unsafeAddr, 1)
    discard fh.writeBuffer(value[0].unsafeAddr, value.len)

proc writePickleBinunicode(fh: var File, value: string): void {.inline.} =
    let len = uint32 value.len
    # try:

    fh.write(PKL_BINUNICODE)
    discard fh.writeBuffer(len.unsafeAddr, 4)
    fh.write(value)
    # discard fh.writeBuffer(value[0].unsafeAddr, len)
    # finally:
    #     echo "failed to dump: '" & value & "' of size " & $len

proc writePickleBoolean(fh: var File, value: bool): void {.inline.} =
    if value == true:
        fh.write(PKL_NEWTRUE)
    else:
        fh.write(PKL_NEWFALSE)

proc writePickleDateBody(fh: var File, value: ptr PY_Date | ptr PY_DateTime, binput: var uint32): void {.inline.} =
    var year = value.getYear()
    year.addr.bigEndian16(addr year)

    var month = value.getMonth()
    var day = value.getDay()

    discard fh.writeBuffer(addr year, 2)
    discard fh.writeBuffer(addr month, 1)
    discard fh.writeBuffer(addr day, 1)

proc writePickleDate(fh: var File, value: PY_Date, binput: var uint32): void {.inline.} =
    fh.writePickleGlobal("datetime", "date")
    fh.writePickleBinput(binput)
    fh.write(PKL_SHORT_BINBYTES)
    fh.write('\4') # date has 4 bytes 2(y)-1(m)-1(d)

    fh.writePickleDateBody(value.unsafeAddr, binput)

    fh.writePickleBinput(binput)
    fh.write(PKL_TUPLE1)
    fh.writePickleBinput(binput)
    fh.write(PKL_REDUCE)
    fh.writePickleBinput(binput)

proc writePickleTimeBody(fh: var File, value: ptr PY_DateTime | ptr PY_Time, binput: var uint32): void {.inline.} =
    var microsecond = uint32 value.getMicrosecond()
    microsecond.addr.bigEndian32(addr microsecond)

    var second = uint8 value.getSecond()
    var minute = uint8 value.getMinute()
    var hour = uint8 value.getHour()

    var ptr_microseconds = cast[pointer](cast[int](addr microsecond) + 1)

    discard fh.writeBuffer(addr hour , 1)
    discard fh.writeBuffer(addr minute , 1)
    discard fh.writeBuffer(addr second , 1)
    discard fh.writeBuffer(ptr_microseconds, 3)
    fh.writePickleBinput(binput)

    fh.write(PKL_TUPLE1)

    fh.writePickleBinput(binput)
    fh.write(PKL_REDUCE)
    fh.writePickleBinput(binput)

proc writePickleTime(fh: var File, value: PY_Time, binput: var uint32): void {.inline.} =
    fh.writePickleGlobal("datetime", "time")
    fh.writePickleBinput(binput)
    fh.write(PKL_SHORT_BINBYTES)
    fh.write('\6')

    fh.writePickleTimeBody(addr value, binput)

proc writePickleDatetime(fh: var File, value: PY_DateTime, binput: var uint32): void {.inline.} =
    fh.writePickleGlobal("datetime", "datetime")
    fh.writePickleBinput(binput)
    fh.write(PKL_SHORT_BINBYTES)
    fh.write('\10')

    fh.writePickleDateBody(addr value, binput)
    fh.writePickleTimeBody(addr value, binput)

proc writePicklePyObj*(fh: var File, value: PY_NoneType, binput: var uint32): void {.inline.} = fh.write(PKL_NONE)
proc writePicklePyObj*(fh: var File, value: int, binput: var uint32): void {.inline.} = fh.writePickleBinint(value)
proc writePicklePyObj*(fh: var File, value: float, binput: var uint32): void {.inline.} = fh.writePickleBinfloat(value)
proc writePicklePyObj*(fh: var File, value: string, binput: var uint32): void {.inline.} = fh.writePickleBinunicode(value)
proc writePicklePyObj*(fh: var File, value: bool, binput: var uint32): void {.inline.} = fh.writePickleBoolean(value)
proc writePicklePyObj*(fh: var File, value: PY_Date, binput: var uint32): void {.inline.} = fh.writePickleDate(value, binput)
proc writePicklePyObj*(fh: var File, value: PY_Time, binput: var uint32): void {.inline.} = fh.writePickleTime(value, binput)
proc writePicklePyObj*(fh: var File, value: PY_DateTime, binput: var uint32): void {.inline.} = fh.writePickleDatetime(value, binput)

proc writePickleStart*(fh: var File, binput: var uint32, elem_count: uint): void {.inline.} =
    binput = 0

    fh.writePickleProto()
    fh.writePickleGlobal("numpy.core.multiarray", "_reconstruct")
    fh.writePickleBinput(binput)
    fh.writePickleGlobal("numpy", "ndarray")
    fh.writePickleBinput(binput)
    fh.writePickleBinint(0)
    fh.write(PKL_TUPLE1)
    fh.writePickleBinput(binput)
    fh.writePickleShortbinbytes("b")
    fh.writePickleBinput(binput)
    fh.write(PKL_TUPLE3)
    fh.writePickleBinput(binput)
    fh.write(PKL_REDUCE)
    fh.writePickleBinput(binput)
    fh.write(PKL_MARK)

    if true:
        fh.writePickleBinint(1)
        fh.writePickleBinint(elem_count)
        fh.write(PKL_TUPLE1)
        fh.writePickleBinput(binput)
        fh.writePickleGlobal("numpy", "dtype")
        fh.writePickleBinput(binput)
        fh.writePickleBinunicode("O8")
        fh.writePickleBinput(binput)
        fh.writePickleBoolean(false)
        fh.writePickleBoolean(true)
        fh.write(PKL_TUPLE3)
        fh.writePickleBinput(binput)
        fh.write(PKL_REDUCE)
        fh.writePickleBinput(binput)
        fh.write(PKL_MARK)

        if true:
            fh.writePickleBinint(3)
            fh.writePickleBinunicode("|")
            fh.writePickleBinput(binput)
            fh.write(PKL_NONE)
            fh.write(PKL_NONE)
            fh.write(PKL_NONE)
            fh.writePickleBinint(-1)
            fh.writePickleBinint(-1)
            fh.writePickleBinint(63)
            fh.write(PKL_TUPLE)

        fh.writePickleBinput(binput)
        fh.write(PKL_BUILD)
        fh.writePickleBoolean(false)
        fh.write(PKL_EMPTY_LIST)
        fh.writePickleBinput(binput)

        # now we dump objects

        if elem_count > 0:
            fh.write(PKL_MARK)

        # fh.write(PKL_APPENDS)

proc writePickleFinish*(fh: var File, binput: var uint32, elem_count: uint): void {.inline.} =
    if elem_count > 0:
        fh.write(PKL_APPENDS)

    fh.write(PKL_TUPLE)
    fh.writePickleBinput(binput)
    fh.write(PKL_BUILD)
    fh.write(PKL_STOP)
