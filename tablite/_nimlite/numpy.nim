import std/unicode
import std/strutils

const NUMPY_MAGIC = "\x93NUMPY"
const NUMPY_MAJOR = "\x01"
const NUMPY_MINOR = "\x00"
const NUMPY_MAGIC_LEN = NUMPY_MAGIC.len
const NUMPY_MAJOR_LEN = NUMPY_MAJOR.len
const NUMPY_MINOR_LEN = NUMPY_MINOR.len

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

type NDArrayTypeDescriptor = enum
    D_OBJECT
    D_UNICODE
    D_BOOLEAN
    D_INT
    D_FLOAT
    D_TIME
    D_DATETIME

type NDArrayDescriptor = (Endianness, NDArrayTypeDescriptor, int)
type BaseNDArray* = ref object of RootObj
    shape*: seq[int]

type BooleanNDArray* = ref object of BaseNDArray
    buf*: seq[bool]

type Int8NDArray* = ref object of BaseNDArray
    buf*: seq[int8]
type Int16NDArray* = ref object of BaseNDArray
    buf*: seq[int16]
type Int32NDArray* = ref object of BaseNDArray
    buf*: seq[int32]
type Int64NDArray* = ref object of BaseNDArray
    buf*: seq[int64]

type Float32NDArray* = ref object of BaseNDArray
    buf*: seq[float32]
type Float64NDArray* = ref object of BaseNDArray
    buf*: seq[float64]

type UnicodeNDArray* = ref object of BaseNDArray
    buf*: seq[char]
    size*: int

proc `$`*(self: BaseNDArray): string =
    if self of BooleanNDArray: return repr(cast[BooleanNDArray](self))
    if self of Int64NDArray: return repr(cast[Int64NDArray](self))
    if self of Float64NDArray: return repr(cast[Float64NDArray](self))
    if self of UnicodeNDArray: return repr(cast[UnicodeNDArray](self))
    else:
        raise newException(Exception, "'$' not implemented")


template corrupted(): void = raise newException(IOError, "file corrupted")

proc validateHeader(fh: File, buf: var array[NUMPY_MAGIC_LEN, uint8], header: string, header_len: int): void {.inline.} =
    if fh.readBytes(buf, 0, header_len) != header_len:
        corrupted()

    for idx in 0..header_len-1:
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

proc consumeShape(header: var string, header_len: int, offset: var int): seq[int] =
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

    if start_index == end_index or start_index < 0 or end_index < 0:
        corrupted()

    let shape_str_seq = header[start_index..end_index].split(',')
    let len_shape_str_seq = shape_str_seq.len

    for i in 0..len_shape_str_seq-1:
        let sh_str = shape_str_seq[i]

        if sh_str.len == 0:
            if i + 1 != len_shape_str_seq:
                corrupted()
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

    var typeOffset: int = 1

    case descr[0]:
        of '|', '<': endianness = Endianness.littleEndian
        of '>':
            endianness = Endianness.bigEndian
        else:
            if not (descr[0] in valid_types):
                corrupted()
            typeOffset = 0

    let typeString = descr[typeOffset]

    if not (typeString in valid_types):
        corrupted()

    var size: int
    var descriptor: NDArrayTypeDescriptor

    case typeString:
        of 'O':
            if (typeOffset + 1) != descr.len:
                corrupted()
            size = -1
            descriptor = NDArrayTypeDescriptor.D_OBJECT
        of 'm', 'M':
            raise newException(Exception, "not yet implemented")
        else:
            size = parseInt(descr[typeOffset+1..descr.len-1])

            case typeString:
                of 'b':
                    if size != 1:
                        corrupted()
                    descriptor = NDArrayTypeDescriptor.D_BOOLEAN
                of 'i':
                    if size != 8:
                        raise newException(Exception, "not yet implemented")
                    descriptor = NDArrayTypeDescriptor.D_INT
                of 'f':
                    if size != 8:
                        raise newException(Exception, "not yet implemented")
                    descriptor = NDArrayTypeDescriptor.D_FLOAT
                of 'U':
                    if size <= 0:
                        corrupted()
                    descriptor = NDArrayTypeDescriptor.D_UNICODE
                else:
                    # never happens
                    corrupted()


    return (endianness, descriptor, size)

proc parseHeader(header: var string): (NDArrayDescriptor, bool, seq[int]) =
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
    var shape: seq[int]
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

proc calcShapeElements(shape: var seq[int]): int {.inline.} =
    var elements = 1

    for m in shape:
        elements = elements * m

    return elements

template readPrimitiveBuffer[T: typed](fh: var File, shape: var seq[int]): seq[T] =
    var elements = calcShapeElements(shape)
    var buf {.noinit.} = newSeq[T](elements)
    var size_T = sizeof(T)
    var buffer_size = elements * size_T

    if fh.readBuffer(addr buf[0], buffer_size) != buffer_size:
        corrupted()

    buf

proc newBooleanNDArray(fh: var File, shape: var seq[int]): BooleanNDArray =
    return BooleanNDArray(
        buf: readPrimitiveBuffer[bool](fh, shape),
        shape: shape
    )

template newIntNDArray(fh: var File, endianness: Endianness, size: int, shape: var seq[int]) =
    case size:
        of 1: Int8NDArray(
            buf: readPrimitiveBuffer[int8](fh, shape),
            shape: shape
        )
        of 2: Int16NDArray(
            buf: readPrimitiveBuffer[int16](fh, shape),
            shape: shape
        )
        of 4: Int32NDArray(
            buf: readPrimitiveBuffer[int32](fh, shape),
            shape: shape
        )
        of 8: Int64NDArray(
            buf: readPrimitiveBuffer[int64](fh, shape),
            shape: shape
        )
        else: corrupted()

template newFloatNDArray(fh: var File, endianness: Endianness, size: int, shape: var seq[int]) =
    case size:
        of 4: Float32NDArray(
            buf: readPrimitiveBuffer[float32](fh, shape),
            shape: shape
        )
        of 8: Float64NDArray(
            buf: readPrimitiveBuffer[float64](fh, shape),
            shape: shape
        )
        else: corrupted()

proc newUnicodeNDArray(fh: var File, endianness: Endianness, size: int, shape: var seq[int]): UnicodeNDArray =
    var elements = calcShapeElements(shape)
    var elem_size = elements * size
    var buf {.noinit.} = newSeq[char](elem_size)
    var buffer_size = elem_size * 4

    if fh.readBuffer(addr buf[0], buffer_size) != buffer_size:
        corrupted()

    return UnicodeNDArray(buf: buf, shape: shape, size: size)

proc readNumpy(fh: var File): BaseNDArray =
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

    var ((descr_endianness, descr_type, descr_size), order, shape) = parseHeader(header)

    if order:
        raise newException(Exception, "'fortran_order' not implemented")
    if shape.len != 1:
        raise newException(Exception, "'shape' not implemented")

    var page: BaseNDArray

    case descr_type:
        of D_BOOLEAN: page = newBooleanNDArray(fh, shape)
        of D_INT: page = newIntNDArray(fh, descr_endianness, descr_size, shape)
        of D_FLOAT: page = newFloatNDArray(fh, descr_endianness, descr_size, shape)
        of D_UNICODE: page = newUnicodeNDArray(fh, descr_endianness, descr_size, shape)
        else:
            raise newException(Exception, "'" & $descr_type & "' not implemented")

    return page

proc readNumpy(path: string): BaseNDArray =
    var fh = open(path, fmRead)
    let arr = readNumpy(fh)

    fh.close()

    return arr


when isMainModule and appType != "lib":
    echo $readNumpy("/home/ratchet/Documents/dematic/tablite/tests/data/pages/boolean_nones.npy")
