import std/[os, unicode, strutils, sugar, times, tables, enumerate, sequtils]
from std/macros import bindSym
from std/typetraits import name
import dateutils, pytypes, unpickling, utils
import pymodules as pymodules
import nimpy as nimpy, nimpy/raw_buffers
import pickling

const NUMPY_MAGIC = "\x93NUMPY"
const NUMPY_MAJOR = "\x01"
const NUMPY_MINOR = "\x00"
const NUMPY_MAGIC_LEN = NUMPY_MAGIC.len
const NUMPY_MAJOR_LEN = NUMPY_MAJOR.len
const NUMPY_MINOR_LEN = NUMPY_MINOR.len

type NDArrayTypeDescriptor = enum
    D_OBJECT
    D_UNICODE
    D_BOOLEAN
    D_INT
    D_FLOAT
    D_TIME
    D_DATE_DAYS
    D_DATETIME_SECONDS
    D_DATETIME_MILISECONDS
    D_DATETIME_MICROSECONDS

type KindNDArray* = enum
    K_BOOLEAN,
    K_INT8,
    K_INT16
    K_INT32,
    K_INT64,
    K_FLOAT32,
    K_FLOAT64,
    K_DATE,
    K_DATETIME,
    K_STRING,
    K_OBJECT

template gendtype[T](dt: typedesc[T], name: char): string = endiannessMark & $name & $sizeof(T)
template gendtypeStr(len: int): string = endiannessMark & "U" & $len

type NDArrayDescriptor = (Endianness, NDArrayTypeDescriptor, int)
type BaseNDArray* {.requiresInit.} = ref object of RootObj
    shape*: Shape
    kind*: KindNDArray

const HeaderBooleanNDArray* = "|b1"
const HeaderInt8NDArray* = gendtype(int8, 'i')
const HeaderInt16NDArray* = gendtype(int16, 'i')
const HeaderInt32NDArray* = gendtype(int32, 'i')
const HeaderInt64NDArray* = gendtype(int64, 'i')
const HeaderFloat32NDArray* = gendtype(float32, 'f')
const HeaderFloat64NDArray* = gendtype(float64, 'f')
const HeaderDateNDArray* = endiannessMark & "M8[D]"
const HeaderDateTimeNDArray* = endiannessMark & "M8[us]"
const HeaderObjectNDArray* = "|O8"

type BooleanNDArray* = ref object of BaseNDArray
    buf*: seq[bool]
    dtype* = HeaderBooleanNDArray

type Int8NDArray* = ref object of BaseNDArray
    buf*: seq[int8]
    dtype* = HeaderInt8NDArray
type Int16NDArray* = ref object of BaseNDArray
    buf*: seq[int16]
    dtype* = HeaderInt16NDArray
type Int32NDArray* = ref object of BaseNDArray
    buf*: seq[int32]
    dtype* = HeaderInt32NDArray
type Int64NDArray* = ref object of BaseNDArray
    buf*: seq[int64]
    dtype* = HeaderInt64NDArray

type Float32NDArray* = ref object of BaseNDArray
    buf*: seq[float32]
    dtype* = HeaderFloat32NDArray

type Float64NDArray* = ref object of BaseNDArray
    buf*: seq[float64]
    dtype* = HeaderFloat64NDArray

type DateNDArray* = ref object of BaseNDArray
    buf*: seq[DateTime]
    dtype* = HeaderDateNDArray

type DateTimeNDArray* = ref object of BaseNDArray
    buf*: seq[DateTime]
    dtype* = HeaderDateTimeNDArray

type UnicodeNDArray* = ref object of BaseNDArray
    buf*: seq[Rune]
    size*: int

type ObjectNDArray* {.requiresInit.} = ref object of BaseNDArray
    buf*: seq[PyObjectND]
    dtypes*: Table[KindObjectND, int]
    dtype* = HeaderObjectNDArray

template pageKind*(_: typedesc[BooleanNDArray]): KindNDArray = KindNDArray.K_BOOLEAN
template pageKind*(_: typedesc[Int8NDArray]): KindNDArray = KindNDArray.K_INT8
template pageKind*(_: typedesc[Int16NDArray]): KindNDArray = KindNDArray.K_INT16
template pageKind*(_: typedesc[Int32NDArray]): KindNDArray = KindNDArray.K_INT32
template pageKind*(_: typedesc[Int64NDArray]): KindNDArray = KindNDArray.K_INT64
template pageKind*(_: typedesc[Float32NDArray]): KindNDArray = KindNDArray.K_FLOAT32
template pageKind*(_: typedesc[Float64NDArray]): KindNDArray = KindNDArray.K_FLOAT64
template pageKind*(_: typedesc[DateNDArray]): KindNDArray = KindNDArray.K_DATE
template pageKind*(_: typedesc[DateTimeNDArray]): KindNDArray = KindNDArray.K_DATETIME
template pageKind*(_: typedesc[UnicodeNDArray]): KindNDArray = KindNDArray.K_STRING
template pageKind*(_: typedesc[ObjectNDArray]): KindNDArray = KindNDArray.K_OBJECT

template headerType*(T: typedesc[BaseNDArray]): string =
    case T.pageKind:
    of K_BOOLEAN: HeaderBooleanNDArray
    of K_INT8: HeaderInt8NDArray
    of K_INT16: HeaderInt16NDArray
    of K_INT32: HeaderInt32NDArray
    of K_INT64: HeaderInt64NDArray
    of K_FLOAT32: HeaderFloat32NDArray
    of K_FLOAT64: HeaderFloat64NDArray
    of K_DATE: HeaderDateNDArray
    of K_DATETIME: HeaderDateTimeNDArray
    of K_OBJECT: HeaderObjectNDArray
    of K_STRING: corrupted(IndexDefect)


iterator pgIter*(self: BooleanNDArray): bool =
    for el in self.buf:
        yield el

iterator pgIter*(self: Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray): int =
    for el in self.buf:
        yield int el

iterator pgIter*(self: Float32NDArray | Float64NDArray): float =
    for el in self.buf:
        yield float el

iterator pgIter*(self: DateNDArray | DateTimeNDArray): DateTime =
    for el in self.buf:
        yield el

iterator pgIter*(self: ObjectNDArray): PY_ObjectND =
    for el in self.buf:
        yield el

iterator pgIter*(self: UnicodeNDArray): string =
    let sz = self.size
    let buf = self.buf
    let len = buf.len
    let empty = Rune('\x00')
    var i = 0

    while i < len:
        let next = i + sz

        var str: string

        if buf[next - 1] != empty:
            # max string
            str = $buf[i..<next]
        elif buf[i] == empty:
            # empty string
            str = ""
        else:
            # locate end of string
            for j in countdown(next - 1, i):
                if buf[j] != empty:
                    str = $buf[i..j]
                    break

        yield str
        i = next

proc `[]`(self: UnicodeNDArray, slice: seq[int] | openArray[int]): UnicodeNDArray =
    let buf = newSeq[Rune](self.size * slice.len)

    for (i, j) in enumerate(slice):
        buf[i * self.size].addr.copyMem(addr self.buf[j * self.size], self.size)

    return UnicodeNDArray(shape: @[slice.len], size: self.size, buf: buf, kind: K_STRING)

proc `[]`(self: ObjectNDArray, slice: seq[int] | openArray[int]): BaseNDArray =
    var dtypes = initTable[KindObjectND, int]()
    let buf = collect:
        for i in slice:
            let el = self.buf[i]
            let dt = el.kind

            if unlikely(not (dt in dtypes)):
                dtypes[dt] = 0

            dtypes[dt] = dtypes[dt] + 1

            el

    let typeList = toSeq(dtypes.keys)
    let shape = @[buf.len]

    if typeList.len == 1:
        # we are single type, we can potentially simplify this page
        let baseType = typeList[0]

        case baseType:
        of K_BOOLEAN:
            let newBuf = collect: (for v in buf: PY_Boolean(v).value)
            return BooleanNDArray(shape: shape, buf: newBuf, kind: K_BOOLEAN)
        of K_INT:
            let newBuf = collect: (for v in buf: int64 PY_Int(v).value)
            return Int64NDArray(shape: shape, buf: newBuf, kind: K_INT64)
        of K_FLOAT:
            let newBuf = collect: (for v in buf: float64 PY_Float(v).value)
            return Float64NDArray(shape: shape, buf: newBuf, kind: K_FLOAT64)
        of K_DATE:
            let newBuf = collect: (for v in buf: PY_Date(v).value)
            return DateNDArray(shape: shape, buf: newBuf, kind: K_DATE)
        of K_DATETIME:
            let newBuf = collect: (for v in buf: PY_DateTime(v).value)
            return DateTimeNDArray(shape: shape, buf: newBuf, kind: K_DATETIME)
        of K_STRING:
            let newBuf = collect: (for v in buf: PY_String(v).value)
            return newBuf.newNDArray
        of K_NONETYPE, K_TIME: 
            # nones and times are always treated as objects
            discard

    return ObjectNDArray(shape: shape, buf: buf, kind: self.kind, dtypes: dtypes)

proc primitiveSlice[T: BooleanNDArray | Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray | Float32NDArray | Float64NDArray | DateNDArray | DateTimeNDArray](self: T, slice: seq[int] | openArray[int]): T =
    let buf = collect:
        for i in slice:
            self.buf[i]
    return T(shape: @[buf.len], buf: buf, kind: self.kind)

proc `[]`*[T: BaseNDArray](self: T, slice: seq[int] | openArray[int]): T =
    case self.kind:
    of K_BOOLEAN: return BooleanNDArray(self).primitiveSlice(slice)
    of K_INT8: return Int8NDArray(self).primitiveSlice(slice)
    of K_INT16: return Int16NDArray(self).primitiveSlice(slice)
    of K_INT32: return Int32NDArray(self).primitiveSlice(slice)
    of K_INT64: return Int64NDArray(self).primitiveSlice(slice)
    of K_FLOAT32: return Float32NDArray(self).primitiveSlice(slice)
    of K_FLOAT64: return Float64NDArray(self).primitiveSlice(slice)
    of K_DATE: return DateNDArray(self).primitiveSlice(slice)
    of K_DATETIME: return DateTimeNDArray(self).primitiveSlice(slice)
    of K_STRING: return UnicodeNDArray(self)[slice]
    of K_OBJECT: return ObjectNDArray(self)[slice]

proc dtype*(self: UnicodeNDArray): string = gendtypeStr(self.size)

macro baseType*(self: typedesc[BooleanNDArray]): typedesc[bool] = bindSym(bool.name)
macro baseType*(self: typedesc[Int8NDArray]): typedesc[int8] = bindSym(int8.name)
macro baseType*(self: typedesc[Int16NDArray]): typedesc[int16] = bindSym(int16.name)
macro baseType*(self: typedesc[Int32NDArray]): typedesc[int32] = bindSym(int32.name)
macro baseType*(self: typedesc[Int64NDArray]): typedesc[int64] = bindSym(int64.name)
macro baseType*(self: typedesc[Float32NDArray]): typedesc[float32] = bindSym(float32.name)
macro baseType*(self: typedesc[Float64NDArray]): typedesc[float64] = bindSym(float64.name)
macro baseType*(self: typedesc[DateNDArray]): typedesc[DateTime] = bindSym(DateTime.name)
macro baseType*(self: typedesc[DateTimeNDArray]): typedesc[DateTime] = bindSym(DateTime.name)
macro baseType*(self: typedesc[UnicodeNDArray]): typedesc[string] = bindSym(string.name)
macro baseType*(self: typedesc[ObjectNDArray]): typedesc[PY_ObjectND] = bindSym(PY_ObjectND.name)

template default*(self: typedesc[BooleanNDArray]): bool = false
template default*(self: typedesc[Int8NDArray]): int8 = 0
template default*(self: typedesc[Int16NDArray]): int16 = 0
template default*(self: typedesc[Int32NDArray]): int32 = 0
template default*(self: typedesc[Int64NDArray]): int64 = 0
template default*(self: typedesc[Float32NDArray]): float32 = 0
template default*(self: typedesc[Float64NDArray]): float64 = 0
template default*(self: typedesc[DateNDArray]): times.DateTime = days2Date(0)
template default*(self: typedesc[DateTimeNDArray]): times.DateTime = days2Date(0)
template default*(self: typedesc[UnicodeNDArray]): string = ""
template default*(self: typedesc[ObjectNDArray]): PY_ObjectND = PY_None

proc writeNumpyHeader*(fh: File, dtype: string, shape: uint): void =
    let header = "{'descr': '" & dtype & "', 'fortran_order': False, 'shape': (" & $shape & ",)}"
    let header_len = len(header)
    let padding = (64 - ((NUMPY_MAGIC_LEN + NUMPY_MAJOR_LEN + NUMPY_MINOR_LEN + 2 + header_len)) mod 64)

    let padding_header = uint16 (padding + header_len)

    fh.write(NUMPY_MAGIC)
    fh.write(NUMPY_MAJOR)
    fh.write(NUMPY_MINOR)

    discard fh.writeBuffer(padding_header.unsafeAddr, 2)

    fh.write(header)

    for i in 0..padding-2:
        fh.write(" ")
    fh.write("\n")

proc writeNumpyUnicode*(fh: var File, str: var string, unicode_len: uint): void {.inline.} =
    for rune in str.toRunes():
        var ch = uint32(rune)
        discard fh.writeBuffer(ch.unsafeAddr, 4)

    let dt = unicode_len - (uint str.runeLen)

    for i in 1..dt:
        fh.write("\x00\x00\x00\x00")

proc writeNumpyInt*(fh: var File, value: int): void {.inline.} =
    discard fh.writeBuffer(value.unsafeAddr, 8)

proc writeNumpyFloat*(fh: var File, value: float): void {.inline.} =
    discard fh.writeBuffer(value.unsafeAddr, 8)

proc writeNumpyBool*(fh: var File, str: var string): void {.inline.} =
    fh.write((if str.toLower() == "true": '\x01' else: '\x00'))

proc writeNumpyBool*(fh: var File, value: var bool): void {.inline.} =
    fh.write(if value: '\x01' else: '\x00')

proc repr(self: ObjectNDArray): string =
    let elems = collect: (for e in self.buf: $e)
    return "ObjectNDArray(buf: @[" & elems.join(", ") & "], shape: " & $self.shape & ")"

proc `$`*(self: BaseNDArray): string =
    case self.kind:
    of K_BOOLEAN: return repr(BooleanNDArray self)
    of K_INT64: return repr(Int64NDArray self)
    of K_FLOAT64: return repr(Float64NDArray self)
    of K_STRING: return repr(UnicodeNDArray self)
    of K_OBJECT: return repr(ObjectNDArray self)
    of K_DATE: return repr(DateNDArray self)
    of K_DATETIME: return repr(DateTimeNDArray self)
    else: implement("BaseNDArray.`$`" & $self.kind)


proc validateHeader(fh: File, buf: var array[NUMPY_MAGIC_LEN, uint8], header: string, header_len: int): void {.inline.} =
    if fh.readBytes(buf, 0, header_len) != header_len:
        corrupted()

    for idx in 0..<header_len:
        if buf[idx] != uint8 header[idx]:
            corrupted()

proc consumeHeaderString(header: var string, header_len: int, offset: var int): string =
    var start_index = -1
    var end_index = -1
    var is_inside = false

    while offset < header_len:
        if header[offset] == '\'':
            if is_inside:
                end_index = offset
                is_inside = false
                break
            else:
                start_index = offset
                is_inside = true

        offset = offset + 1

    if start_index == end_index or start_index < 0 or end_index < 0:
        corrupted()

    offset = offset + 1

    return header[start_index+1..end_index-1]

proc consumeBool(header: var string, header_len: int, offset: var int): bool =
    while offset < header_len:
        if header[offset].isAlphaNumeric():
            break

        offset = offset + 1

    var start_index = offset
    var end_index = -1

    while offset < header_len:
        if header[offset] == ',':
            end_index = offset - 1
            break

        offset = offset + 1

    var res_str = $header[start_index..end_index]
    var res: bool

    case res_str.toLower():
        of "true": res = true
        of "false": res = false
        else: corrupted()

    offset = offset + 1

    return res

proc consumeShape(header: var string, header_len: int, offset: var int): Shape =
    var shape = newSeq[int]()
    var start_index = -1
    var end_index = -1

    while offset < header_len:
        if header[offset] == ')':
            end_index = offset - 1
            break

        if header[offset] == '(':
            start_index = offset + 1

        offset = offset + 1

    if start_index == end_index or start_index < 0 or end_index < 0: corrupted()

    let shape_str_seq = header[start_index..end_index].split(',')
    let len_shape_str_seq = shape_str_seq.len

    for i in 0..len_shape_str_seq-1:
        let sh_str = shape_str_seq[i]

        if sh_str.len == 0:
            if i + 1 != len_shape_str_seq: corrupted()
            continue

        shape.add(parseInt(sh_str))

    return shape

proc consumeDescr(header: var string, header_len: int, offset: var int): NDArrayDescriptor =
    let descr = consumeHeaderString(header, header_len, offset)

    while offset < header_len:
        if header[offset] == ',':
            break

        offset = offset + 1

    offset = offset + 1

    var endianness: Endianness
    let valid_types = ['b', 'i', 'f', 'U', 'O', 'm', 'M']

    var type_offset: int = 1

    case descr[0]:
        of '|', '<': endianness = Endianness.littleEndian
        of '>':
            endianness = Endianness.bigEndian
        else:
            if not (descr[0] in valid_types): corrupted()

            type_offset = 0

    let type_string = descr[type_offset]

    if not (type_string in valid_types): corrupted()

    var size: int
    var descriptor: NDArrayTypeDescriptor
    var dt_descriptor: string

    if type_string == 'm' or type_string == 'M':
        if descr[type_offset + 1] != '8' or descr[type_offset + 2] != '[' or descr[^1] != ']':
            corrupted()

        dt_descriptor = descr[(type_offset + 3)..^2]
    elif type_string == 'O':
        if (type_offset + 1) != descr.len:
            if descr[type_offset + 1] != '8' or (type_offset + 2) != descr.len:
                corrupted()

    case type_string:
        of 'O':
            size = -1
            descriptor = NDArrayTypeDescriptor.D_OBJECT
        of 'm':
            case dt_descriptor:
            else: implement(descr)
        of 'M':
            case dt_descriptor:
            of "D":
                size = 8
                descriptor = NDArrayTypeDescriptor.D_DATE_DAYS
            of "us":
                size = 8
                descriptor = NDArrayTypeDescriptor.D_DATETIME_MICROSECONDS
            else: implement(descr)
        else:
            size = parseInt(descr[type_offset+1..descr.len-1])

            case type_string:
                of 'b':
                    if size != 1: corrupted()
                    descriptor = NDArrayTypeDescriptor.D_BOOLEAN
                of 'i':
                    if size != 8: implement("int size != 8")
                    descriptor = NDArrayTypeDescriptor.D_INT
                of 'f':
                    if size != 8: implement("float size != 8")
                    descriptor = NDArrayTypeDescriptor.D_FLOAT
                of 'U':
                    if size <= 0: corrupted()
                    descriptor = NDArrayTypeDescriptor.D_UNICODE
                else:
                    # never happens
                    corrupted()


    return (endianness, descriptor, size)

proc parseHeader(header: var string): (NDArrayDescriptor, bool, Shape) =
    var offset = 0
    var header_len = header.len
    var entry_consumed = false

    while offset < header_len:
        # consume entry token
        if header[offset] == '{':
            entry_consumed = true
            break

        offset = offset + 1

    if not entry_consumed:
        corrupted()

    offset = offset + 1

    var descr: NDArrayDescriptor
    var has_descr = false
    var order: bool
    var has_order = false
    var shape: Shape
    var has_shape = false

    for i in 0..2:
        let name = consumeHeaderString(header, header_len, offset)

        case name:
            of "descr":
                descr = consumeDescr(header, header_len, offset)
                has_descr = true
            of "fortran_order":
                order = consumeBool(header, header_len, offset)
                has_order = true
            of "shape":
                shape = consumeShape(header, header_len, offset)
                has_shape = true
            else:
                corrupted()

    if not has_descr or not has_order or not has_shape:
        corrupted()

    return (descr, order, shape)



template readPrimitiveBuffer[T: typed](fh: var File, shape: var Shape): seq[T] =
    let elements = calcShapeElements(shape)

    if elements == 0:
        newSeq[T](elements)
    else:
        let size_T = sizeof(T)

        var buf {.noinit.} = newSeq[T](elements)
        var buffer_size = elements * size_T

        if fh.readBuffer(addr buf[0], buffer_size) != buffer_size:
            corrupted()

        buf

proc newBooleanNDArray(fh: var File, shape: var Shape): BooleanNDArray =
    return BooleanNDArray(
        buf: readPrimitiveBuffer[bool](fh, shape),
        shape: shape,
        kind: K_BOOLEAN
    )

template newIntNDArray(fh: var File, endianness: Endianness, size: int, shape: var Shape) =
    case size:
        of 1: Int8NDArray(buf: readPrimitiveBuffer[int8](fh, shape), shape: shape, kind: K_INT8)
        of 2: Int16NDArray(buf: readPrimitiveBuffer[int16](fh, shape), shape: shape, kind: K_INT16)
        of 4: Int32NDArray(buf: readPrimitiveBuffer[int32](fh, shape), shape: shape, kind: K_INT32)
        of 8: Int64NDArray(buf: readPrimitiveBuffer[int64](fh, shape), shape: shape, kind: K_INT64)
        else: corrupted()

proc newDateArray_Days(fh: var File, endianness: Endianness, shape: var Shape): DateNDArray {.inline.} =
    let buf = collect: (for v in readPrimitiveBuffer[int64](fh, shape): days2Date(v))

    return DateNDArray(buf: buf, shape: shape, kind: K_DATE)

proc newDateTimeArray_Seconds(fh: var File, endianness: Endianness, shape: var Shape): DateTimeNDArray {.inline.} =
    let data = readPrimitiveBuffer[int64](fh, shape)
    let buf = collect: (for v in data: initTime(v, 0).utc())

    return DateTimeNDArray(buf: buf, shape: shape, kind: K_DATETIME)

proc newDateTimeArray_Miliseconds(fh: var File, endianness: Endianness, shape: var Shape): DateTimeNDArray {.inline.} =
    let data = readPrimitiveBuffer[int64](fh, shape)
    let buf = collect:
        for v in data:
            let (s, m) = divmod(v, 1000)
            initTime(s, m * 1000).utc()

    return DateTimeNDArray(buf: buf, shape: shape, kind: K_DATETIME)

proc newDateTimeArray_Microseconds(fh: var File, endianness: Endianness, shape: var Shape): DateTimeNDArray {.inline.} =
    let data = readPrimitiveBuffer[int64](fh, shape)
    let buf = collect:
        for v in data:
            let (s, u) = divmod(v, 1_000_000)
            initTime(s, u).utc()

    return DateTimeNDArray(buf: buf, shape: shape, kind: K_DATETIME)

template newFloatNDArray(fh: var File, endianness: Endianness, size: int, shape: var Shape) =
    case size:
        of 4: Float32NDArray(buf: readPrimitiveBuffer[float32](fh, shape), shape: shape, kind: K_FLOAT32)
        of 8: Float64NDArray(buf: readPrimitiveBuffer[float64](fh, shape), shape: shape, kind: K_FLOAT64)
        else: corrupted()

proc newUnicodeNDArray(fh: var File, endianness: Endianness, size: int, shape: var Shape): UnicodeNDArray =
    var elements = calcShapeElements(shape)
    var elem_size = elements * size
    var buf_size = elem_size * sizeof(Rune)
    var buf {.noinit.} = newSeq[Rune](elem_size)

    if fh.readBuffer(addr buf[0], buf_size) != buf_size:
        corrupted()

    return UnicodeNDArray(buf: buf, shape: shape, size: size, kind: K_STRING)

proc newObjectNDArray(fh: var File, endianness: Endianness, shape: var Shape): ObjectNDArray =
    var elements = calcShapeElements(shape)
    var (shape, buf, dtypes) = readPickledPage(fh, endianness, shape)

    if calcShapeElements(shape) != elements:
        corrupted()

    return ObjectNDArray(shape: shape, buf: buf, dtypes: dtypes, kind: K_OBJECT)

proc readPageInfo(fh: var File): (NDArrayDescriptor, bool, Shape) =
    var header_bytes: array[NUMPY_MAGIC_LEN, uint8]

    validateHeader(fh, header_bytes, NUMPY_MAGIC, NUMPY_MAGIC_LEN)
    validateHeader(fh, header_bytes, NUMPY_MAJOR, NUMPY_MAJOR_LEN)
    validateHeader(fh, header_bytes, NUMPY_MINOR, NUMPY_MINOR_LEN)

    var header_size: uint16

    if fh.readBuffer(addr header_size, 2) != 2:
        corrupted()

    var header = newString(header_size)

    if fh.readBuffer(addr header[0], int header_size) != int header_size:
        corrupted()

    var (descr, order, shape) = parseHeader(header)

    if order: implement("fortran_order")
    if shape.len != 1: implement("shape.len != 1")

    return (descr, order, shape)

proc readNumpy(fh: var File): BaseNDArray =
    var ((descrEndianness, descrType, descrSize), _, shape) = readPageInfo(fh)
    var page: BaseNDArray

    case descrType:
        of D_BOOLEAN: page = newBooleanNDArray(fh, shape)
        of D_INT: page = newIntNDArray(fh, descrEndianness, descrSize, shape)
        of D_FLOAT: page = newFloatNDArray(fh, descrEndianness, descrSize, shape)
        of D_UNICODE: page = newUnicodeNDArray(fh, descrEndianness, descrSize, shape)
        of D_OBJECT: page = newObjectNDArray(fh, descrEndianness, shape)
        of D_DATE_DAYS: page = newDateArray_Days(fh, descrEndianness, shape)
        of D_DATETIME_SECONDS: page = newDateTimeArray_Seconds(fh, descrEndianness, shape)
        of D_DATETIME_MILISECONDS: page = newDateTimeArray_Miliseconds(fh, descrEndianness, shape)
        of D_DATETIME_MICROSECONDS: page = newDateTimeArray_Microseconds(fh, descrEndianness, shape)
        else: implement($descrType)

    return page

proc readNumpy*(path: string): BaseNDArray =
    var fh = open(path, fmRead)
    let arr = readNumpy(fh)

    fh.close()

    return arr

proc toNumpyPrimitive(arrType: string, shape: var Shape, sizeof: int, buf: pointer): nimpy.PyObject =
    let elements = calcShapeElements(shape)
    let np = pyImport("numpy")
    var ndBuf: RawPyBuffer
    let ndArray = np.ndarray(shape, arrType)

    ndArray.getBuffer(ndBuf, PyBUF_WRITABLE or PyBUF_ND)

    ndBuf.buf.copyMem(buf, sizeof * elements)
    ndBuf.release()

    return ndArray

proc toNumpyPrimitive[T: bool | int8 | int16 | int32 | int64 | float32 | float64](shape: var Shape, buf: pointer): nimpy.PyObject =
    when T is bool:
        return toNumpyPrimitive("?", shape, sizeof(T), buf)
    else:
        let sz = sizeof(T)

        when T is int8 or T is int16 or T is int32 or T is int64:
            return toNumpyPrimitive("i" & $sz, shape, sz, buf)
        else:
            when T is float32 or T is float64:
                return toNumpyPrimitive("f" & $sz, shape, sz, buf)
            else:
                corrupted()

proc toPython(self: BooleanNDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[bool](self.shape, addr self.buf[0])

proc toPython(self: Int8NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[int8](self.shape, addr self.buf[0])
proc toPython(self: Int16NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[int16](self.shape, addr self.buf[0])
proc toPython(self: Int32NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[int32](self.shape, addr self.buf[0])
proc toPython(self: Int64NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[int64](self.shape, addr self.buf[0])

proc toPython(self: Float32NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[float32](self.shape, addr self.buf[0])
proc toPython(self: Float64NDArray): nimpy.PyObject {.inline.} = toNumpyPrimitive[float64](self.shape, addr self.buf[0])

proc toPython(self: UnicodeNDArray): nimpy.PyObject = toNumpyPrimitive(gendtypeStr(self.size), self.shape, self.size * sizeof(Rune), addr self.buf[0])

proc toPython(self: DateNDArray): nimpy.PyObject =
    var buf = collect:
        for el in self.buf:
            el.toTime.time2Duration.inDays

    return toNumpyPrimitive(self.dtype, self.shape, sizeof(int64), addr buf[0])

proc toPython(self: DateTimeNDArray): nimpy.PyObject =
    var buf = collect:
        for el in self.buf:
            el.toTime.time2Duration.inMicroseconds

    return toNumpyPrimitive(self.dtype, self.shape, sizeof(int64), addr buf[0])

proc toNimpy(self: PY_NoneType): nimpy.PyObject = pymodules.builtins().None
proc toNimpy(self: PY_Boolean): nimpy.PyObject = pymodules.builtins().bool(self.value)
proc toNimpy(self: PY_Int): nimpy.PyObject = pymodules.builtins().int(self.value)
proc toNimpy(self: PY_Float): nimpy.PyObject = pymodules.builtins().float(self.value)
proc toNimpy(self: PY_String): nimpy.PyObject = pymodules.builtins().str(self.value)
proc toNimpy(self: PY_Date): nimpy.PyObject =
    return pymodules.datetime().date(self.value.year, self.value.month, self.value.monthday)
proc toNimpy(self: PY_Time): nimpy.PyObject =
    let hour = self.getHour()
    let minute = self.getMinute()
    let second = self.getSecond()
    let microsecond = self.getMicrosecond()

    return pymodules.datetime().time(
        hour = hour, minute = minute, second = second, microsecond = microsecond
    )

proc toNimpy(self: PY_DateTime): nimpy.PyObject =
    return pymodules.datetime().datetime(
        self.value.year, self.value.month, self.value.monthday,
        self.value.hour, self.value.minute, self.value.second, int(self.value.nanosecond / 1000)
    )

proc toNimpy(self: PY_ObjectND): nimpy.PyObject =
    case self.kind:
    of K_NONETYPE: return PY_NoneType(self).toNimpy()
    of K_BOOLEAN: return PY_Boolean(self).toNimpy()
    of K_INT: return PY_Int(self).toNimpy()
    of K_FLOAT: return PY_Float(self).toNimpy()
    of K_STRING: return PY_String(self).toNimpy()
    of K_DATE: return PY_Date(self).toNimpy()
    of K_TIME: return PY_Time(self).toNimpy()
    of K_DATETIME: return PY_DateTime(self).toNimpy()

proc toPython(self: ObjectNDArray): nimpy.PyObject =
    let buf = collect:
        for el in self.buf:
            el.toNimpy()

    return numpy().array(buf)


proc toPython*(self: BaseNDArray): nimpy.PyObject =
    case self.kind:
    of K_BOOLEAN: return BooleanNDArray(self).toPython()
    of K_INT8: return Int8NDArray(self).toPython()
    of K_INT16: return Int16NDArray(self).toPython()
    of K_INT32: return Int32NDArray(self).toPython()
    of K_INT64: return Int64NDArray(self).toPython()
    of K_FLOAT32: return Float32NDArray(self).toPython()
    of K_FLOAT64: return Float64NDArray(self).toPython()
    of K_DATE: return DateNDArray(self).toPython()
    of K_DATETIME: return DateTimeNDArray(self).toPython()
    of K_STRING: return UnicodeNDArray(self).toPython()
    of K_OBJECT: return ObjectNDArray(self).toPython()

proc getPageLen*(fh: var File): int =
    var (_, _, shape) = readPageInfo(fh)

    return calcShapeElements(shape)

proc getPageLen*(path: string): int =
    var fh = open(path, fmRead)
    let len = getPageLen(fh)

    fh.close()

    return len

proc getColumnLen*(pages: openArray[string]): int =
    var acc = 0

    for p in pages:
        acc = acc + getPageLen(p)

    return acc

template toType(dtype: KindObjectND, shape: Shape): Table[KindObjectND, int] = {dtype: calcShapeElements(shape)}.toTable
proc getPageTypes*(self: BaseNDArray): Table[KindObjectND, int] =
    case self.kind:
    of K_BOOLEAN: K_BOOLEAN.toType(self.shape)
    of K_INT8, K_INT16, K_INT32, K_INT64: K_INT.toType(self.shape)
    of K_FLOAT32, K_FLOAT64: K_FLOAT.toType(self.shape)
    of K_STRING: K_STRING.toType(self.shape)
    of K_DATE: K_DATE.toType(self.shape)
    of K_DATETIME: K_DATETIME.toType(self.shape)
    of K_OBJECT: ObjectNDArray(self).dtypes

proc getPageTypes*(path: string): Table[KindObjectND, int] =
    var fh = open(path, fmRead)
    var ((descrEndianness, descrType, _), _, shape) = readPageInfo(fh)
    var dtypes: Table[KindObjectND, int]
    let elements = shape.calcShapeElements()

    case descrType:
    of D_BOOLEAN: dtypes = {KindObjectND.K_BOOLEAN: elements}.toTable
    of D_INT: dtypes = {K_INT: elements}.toTable
    of D_FLOAT: dtypes = {KindObjectND.K_FLOAT: elements}.toTable
    of D_UNICODE: dtypes = {KindObjectND.K_STRING: elements}.toTable
    of D_DATE_DAYS: dtypes = {KindObjectND.K_DATE: elements}.toTable
    of D_DATETIME_SECONDS, D_DATETIME_MILISECONDS, D_DATETIME_MICROSECONDS: dtypes = {KindObjectND.K_DATETIME: elements}.toTable
    of D_OBJECT:
        # we only need to read the object page to know the dtypes
        dtypes = newObjectNDArray(fh, descrEndianness, shape).dtypes
    else: implement($descrType)

    fh.close()

    return dtypes
proc getColumnTypes*(pages: openArray[string] | seq[string]): Table[KindObjectND, int] =
    var dtypes = initTable[KindObjectND, int]()

    for p in pages:
        for (t, v) in getPageTypes(p).pairs():
            if not (t in dtypes):
                dtypes[t] = 0
            dtypes[t] = dtypes[t] + v

    return dtypes

proc len*(self: BaseNDArray): int = calcShapeElements(self.shape)

proc save(self: BooleanNDArray | Int8NDArray | Int16NDArray | Int32NDArray | Int64NDArray | Float32NDArray | Float64NDArray | UnicodeNDArray, path: string): void =
    let dtype = self.dtype
    let buf = self.buf
    let elements = calcShapeElements(self.shape)
    let size = (if buf.len > 0: buf[0].sizeof else: 0)
    let fh = open(path, fmWrite)

    fh.writeNumpyHeader(dtype, uint elements)

    if buf.len > 0:
        when self is UnicodeNDArray:
            discard fh.writeBuffer(addr buf[0], size * elements * self.size)
        else:
            discard fh.writeBuffer(addr buf[0], size * elements)

    fh.close()

proc save[T: DateNDArray | DateTimeNDArray](self: T, path: string): void =
    let dtype = self.dtype
    let elements = calcShapeElements(self.shape)
    let size = 8
    let fh = open(path, fmWrite)
    var value: int64

    fh.writeNumpyHeader(dtype, uint elements)

    for el in self.buf:
        when T is DateNDArray:
            value = el.toTime().time2Duration.inDays
        else:
            when T is DateTimeNDArray:
                value = el.toTime().time2Duration.inMicroseconds
            else:
                corrupted(ObjectConversionDefect)

        discard fh.writeBuffer(addr value, size)

    fh.close()

template saveAsPrimitive(self: ObjectNDArray, path: string, headerType: string, dump: proc): void =
    let elements = uint calcShapeElements(self.shape)
    var fh = open(path, fmWrite)

    fh.writeNumpyHeader(headerType, elements)

    for v in self.pgIter:
        fh.dump(v)

    fh.close()

proc writeBool(fh: var File, v: PY_ObjectND): void {.inline.} = fh.writeNumpyBool(PY_Boolean(v).value)
proc writeInt(fh: var File, v: PY_ObjectND): void {.inline.} = fh.writeNumpyInt(PY_Int(v).value)
proc writeFloat(fh: var File, v: PY_ObjectND): void {.inline.} = fh.writeNumpyFloat(PY_Float(v).value)
proc writeDate(fh: var File, v: PY_ObjectND): void {.inline.} = fh.writeNumpyInt(PY_Date(v).value.toTime().time2Duration.inDays)
proc writeDateTime(fh: var File, v: PY_ObjectND): void {.inline.} = fh.writeNumpyInt(PY_DateTime(v).value.toTime().time2Duration.inMicroseconds)

proc saveAsUnicode(self: ObjectNDArray, path: string): void =
    var longest = 1

    for v in self.pgIter:
        longest = max(longest, PY_String(v).value.len)

    let elements = uint calcShapeElements(self.shape)
    var fh = open(path, fmWrite)
    
    fh.writeNumpyHeader(gendtypeStr(longest), elements)

    for v in self.pgIter:
        fh.writeNumpyUnicode(PY_String(v).value, uint longest)

    fh.close()

proc save(self: ObjectNDArray, path: string): void =
    var hasNones = K_NONETYPE in self.dtypes
    
    if not hasNones or self.dtypes[K_NONETYPE] == 0:
        # we have no nones, we may be able to save as primitive
        var colDtypes = toSeq(self.dtypes.keys)

        if hasNones:
            # cleanup
            colDtypes.delete(colDtypes.find(K_NONETYPE))

        if colDtypes.len == 1:
            # we only have left-over type
            let activeType = colDtypes[0]

            if self.dtypes[activeType] > 0:
                # save as primitive if possible
                case activeType:
                of K_BOOLEAN: self.saveAsPrimitive(path, BooleanNDArray.headerType, writeBool); return
                of K_INT: self.saveAsPrimitive(path, Int64NDArray.headerType, writeInt); return
                of K_FLOAT: self.saveAsPrimitive(path, Float64NDArray.headerType, writeFloat); return
                of K_STRING: self.saveAsUnicode(path); return
                of K_DATE: self.saveAsPrimitive(path, DateNDArray.headerType, writeDate); return
                of K_DATETIME: self.saveAsPrimitive(path, DateTimeNDArray.headerType, writeDateTime); return
                of K_NONETYPE: discard  # can't happen
                of K_TIME: discard      # time is always an object

    let dtype = self.dtype
    let elements = uint calcShapeElements(self.shape)
    var fh = open(path, fmWrite)

    var binput: uint32 = 0

    fh.writeNumpyHeader(dtype, elements)
    fh.writePickleStart(binput, elements)

    for el in self.buf:
        fh.writePicklePyObj(el, binput)

    fh.writePickleFinish(binput, elements)

    fh.close()

proc newNDArray*(arr: seq[string] | openArray[string] | iterator(): string): UnicodeNDArray =
    var longest = 1
    var page_len = 0
    let runes = collect:
        for str in arr:
            let res = str.toRunes
            longest = max(longest, res.len)
            page_len = page_len + 1
            res
    
    let shape = @[page_len]
    let buf = newSeq[Rune](longest * page_len)

    for (i, str) in enumerate(runes):
        buf[i * longest].addr.copyMem(addr str[0], str.len * sizeof(Rune))

    return UnicodeNDArray(shape: shape, buf: buf, size: longest, kind: K_STRING)

proc save*(self: BaseNDArray, path: string): void =
    case self.kind:
    of K_BOOLEAN: BooleanNDArray(self).save(path)
    of K_INT8: Int8NDArray(self).save(path)
    of K_INT16: Int16NDArray(self).save(path)
    of K_INT32: Int32NDArray(self).save(path)
    of K_INT64: Int64NDArray(self).save(path)
    of K_FLOAT32: Float32NDArray(self).save(path)
    of K_FLOAT64: Float64NDArray(self).save(path)
    of K_STRING: UnicodeNDArray(self).save(path)
    of K_DATE: DateNDArray(self).save(path)
    of K_DATETIME: DateTimeNDArray(self).save(path)
    of K_OBJECT: ObjectNDArray(self).save(path)

proc type2PyType(`type`: KindObjectND): nimpy.PyObject =
    case `type`:
    of K_BOOLEAN: return pymodules.builtins().getattr("bool")     # nim's word reservation behaviour is stupid
    of K_INT: return pymodules.builtins().getattr("int")       # ditto
    of K_FLOAT: return pymodules.builtins().getattr("float")   # ditto
    of K_STRING: return pymodules.builtins().str
    of K_NONETYPE: return pymodules.PyNoneClass
    of K_DATE: return pymodules.datetime().date
    of K_TIME: return pymodules.datetime().time
    of K_DATETIME: return pymodules.datetime().datetime

proc newPyPage*(id: int, path: string, len: int, dtypes: Table[KindObjectND, int]): nimpy.PyObject =
    let pyDtypes = collect(initTable()):
        for (dt, n) in dtypes.pairs:
            { dt.type2PyType: n }

    let pg = pymodules.tabliteBase().SimplePage(id, path, len, pyDtypes)

    return pg

when isMainModule and appType != "lib":
    var arr = readNumpy("./tests/data/pages/mixed.npy")

    echo arr

    let pyObj = pymodules.builtins().repr(arr.toPython())

    echo $pyObj
