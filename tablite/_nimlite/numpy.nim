import std/[tables, unicode, strutils, sugar, sequtils]
import  pickleproto

const NUMPY_MAGIC = "\x93NUMPY"
const NUMPY_MAJOR = "\x01"
const NUMPY_MINOR = "\x00"
const NUMPY_MAGIC_LEN = NUMPY_MAGIC.len
const NUMPY_MAJOR_LEN = NUMPY_MAJOR.len
const NUMPY_MINOR_LEN = NUMPY_MINOR.len

template corrupted(): void = raise newException(IOError, "file corrupted")
template implement(name: string = ""): void = raise newException(Exception, if name.len == 0: "not yet imlemented" else: "'" & name & "' not yet imlemented")

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

type PY_Object* = ref object of RootObj
type PY_ObjectND* = ref object of PY_Object
type PY_Boolean* = ref object of PY_ObjectND
    value*: bool
type PY_Int* = ref object of PY_ObjectND
    value*: int
type PY_Float* = ref object of PY_ObjectND
    value*: float
type PY_String* = ref object of PY_ObjectND
    value*: string
type PY_Bytes* = ref object of PY_ObjectND
    value*: seq[char]
type PY_None* = ref object of PY_ObjectND
type PY_Date* = ref object of PY_ObjectND
type PY_Time* = ref object of PY_ObjectND
type PY_DateTime* = ref object of PY_ObjectND
type Py_Iterable* = ref object of PY_Object
    elems*: seq[PY_Object]
type Py_Tuple* = ref object of Py_Iterable
type Py_List* = ref object of Py_Iterable
type Py_Set* = ref object of Py_Iterable
type Py_Dict* = ref object of Py_Object
    elems*: Table[PY_Object, PY_Object]

type PY_NpMultiArray* = ref object of Py_Object
    shape: seq[int]
    dtype: string
    elems: seq[PY_Object]
type PY_NpDType* = ref object of Py_Iterable
    dtype: string

type ObjectNDArray* = ref object of BaseNDArray
    buf*: seq[PyObjectND]

proc toString(self: PY_Object, depth: int = 0): string;

proc repr(self: ObjectNDArray): string =
    let elems = collect: (for e in self.buf: e.toString)
    return "ObjectNDArray(buf: @[" & elems.join(", ") & "], shape: " & $self.shape & ")"

proc `$`*(self: BaseNDArray): string =
    if self of BooleanNDArray: return repr(BooleanNDArray self)
    if self of Int64NDArray: return repr(Int64NDArray self)
    if self of Float64NDArray: return repr(Float64NDArray self)
    if self of UnicodeNDArray: return repr(UnicodeNDArray self)
    if self of ObjectNDArray: return repr(ObjectNDArray self)
    else:
        implement("BaseNDArray.`$`")


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

type IterPickle = iterator(): uint8
proc unpickleFile(fh: File, endianness: Endianness): IterPickle =
    const READ_BUF_SIZE = 2048
    var buf {.noinit.}: array[READ_BUF_SIZE, uint8]

    return iterator(): uint8 =
        while not unlikely(fh.endOfFile):
            let bytes_read = fh.readBuffer(addr buf, READ_BUF_SIZE)

            for i in 0..(bytes_read-1):
                yield buf[i]

proc readLine(iter: IterPickle): string =
    var res = ""

    while not iter.finished:
        let ch = char iter()

        if ch == '\n':
            break

        res = res & ch

    return res

# type PyReduceKind* = enum
#     Multiarray
#     Dtype

# type BasePickle = ref object of PY_Object
type ProtoPickle = ref object of PY_Int
type GlobalPickle = ref object of PY_Object
    module: string
    name: string
type BinPutPickle = ref object of PY_Object
    index: int



type TuplePickle = ref object of Py_Tuple
type BinBytesPickle = ref object of PY_Bytes

type BinIntPickle = ref object of PY_Int
type BinUnicodePickle = ref object of PY_String
type BooleanPickle = ref object of PY_Boolean
type NonePickle = ref object of PY_None
    
type ReducePickle = ref object of Py_Object
    value: PY_Object
#     case kind*: PyReduceKind
#     of Multiarray:
#         obj: ObjectNDArray
#     of Dtype:
#         nil

type MarkPickle = ref object of PY_Object
type BuildPickle = ref object of PY_Object
#     inst: BasePickle
#     state: BasePickle
# type EmptyListPickle = ref object of Py_List
# type EmptyDictPickle = ref object of Py_Dict
# type EmptySetPickle = ref object of Py_Set
# type EmptyTuplePickle = ref object of Py_Tuple
type AppendsPickle = ref object of Py_Iterable
type StopPickle = ref object of PY_Object
    value: PY_Object

const WHITESPACE_CHARACTERS = "    "

proc toString(self: ProtoPickle, depth: int): string =
    return "PROTO(protocol: " & $self.value & ")"
proc toString(self: GlobalPickle, depth: int): string =
    return "GLOBAL(module: '" & self.module & "', name: '" & self.name & "')"
proc toString(self: BinPutPickle, depth: int): string =
    return "BINPUT(index: " & $self.index & ")"
proc toString(self: BinIntPickle, depth: int): string =
    return "BININT" & "(value: " & $self.value & ")"
proc toString(self: TuplePickle, depth: int): string =
    let ws0 = repeat(WHITESPACE_CHARACTERS, depth)
    let ws1 = repeat(WHITESPACE_CHARACTERS, depth + 2)
    let ws2 = repeat(WHITESPACE_CHARACTERS, depth + 3)
    let elems = collect: (for e in self.elems: "\n" & ws2 & e.toString(depth * 2))
    return "TUPLE(" & (if self.elems.len > 3: "" else: $self.elems.len) & "\n" & ws1 & "elems: (" & elems.join(", ") & "\n" & ws1 & ")\n" & ws0 & ")"
proc toString(self: BinBytesPickle, depth: int): string =
    return "BINBYTES(bytes: b'" & self.value.join("") & "')"
proc toString(self: ReducePickle, depth: int): string =
    let ws0 = repeat(WHITESPACE_CHARACTERS, depth)
    let ws = repeat(WHITESPACE_CHARACTERS, depth + 1)

    return "REDUCE()"
proc toString(self: MarkPickle, depth: int): string =
    return "MARK()"
proc toString(self: BinUnicodePickle, depth: int): string =
    return "BINUNICODE(value: " & $self.value & ")"
proc toString(self: NonePickle, depth: int): string =
    return "NONE()"
proc toString(self: BooleanPickle, depth: int): string =
    return "BOOLEAN(value: " & $self.value & ")"
proc toString(self: BuildPickle, depth: int): string =
    let ws0 = repeat(WHITESPACE_CHARACTERS, depth)
    let ws = repeat(WHITESPACE_CHARACTERS, depth + 1)
    return "BUILD()"
proc toString(self: AppendsPickle, depth: int): string =
    let ws0 = repeat(WHITESPACE_CHARACTERS, depth)
    let ws1 = repeat(WHITESPACE_CHARACTERS, depth + 2)
    let ws2 = repeat(WHITESPACE_CHARACTERS, depth + 3)
    let elems = collect: (for e in self.elems: "\n" & ws2 & e.toString(depth * 2))
    return "APPENDS(\n" & ws1 & "elems: (" & elems.join(", ") & "\n" & ws1 & ")\n" & ws0 & ")"
proc toString(self: StopPickle, depth: int): string =
    let ws0 = repeat(WHITESPACE_CHARACTERS, depth)
    let ws = repeat(WHITESPACE_CHARACTERS, depth + 1)

    return "STOP(\n" & ws & "value: " & self.value.toString(depth + 1) & "\n" & ws0 & ")"
proc toString(self: Py_Dict, depth: int): string =
    return repr(self)
proc toString(self: Py_List, depth: int): string =
    let ws0 = repeat(WHITESPACE_CHARACTERS, depth)
    let ws1 = repeat(WHITESPACE_CHARACTERS, depth + 2)
    let ws2 = repeat(WHITESPACE_CHARACTERS, depth + 3)
    let elems = collect: (for e in self.elems: "\n" & ws2 & e.toString(depth * 2))
    return "Py_List(\n" & (if self.elems.len > 3: "" else: $self.elems.len) & "\n" & ws1 & "elems: (" & elems.join(", ") & "\n" & ws1 & ")\n" & ws0 & ")"
proc toString(self: Py_Set, depth: int): string =
    let ws0 = repeat(WHITESPACE_CHARACTERS, depth)
    let ws1 = repeat(WHITESPACE_CHARACTERS, depth + 2)
    let ws2 = repeat(WHITESPACE_CHARACTERS, depth + 3)
    let elems = collect: (for e in self.elems: "\n" & ws2 & e.toString(depth * 2))
    return "Py_Set(\n" & (if self.elems.len > 3: "" else: $self.elems.len) & "\n" & ws1 & "elems: (" & elems.join(", ") & "\n" & ws1 & ")\n" & ws0 & ")"
proc toString(self: Py_Tuple, depth: int): string =
    let ws0 = repeat(WHITESPACE_CHARACTERS, depth)
    let ws1 = repeat(WHITESPACE_CHARACTERS, depth + 2)
    let ws2 = repeat(WHITESPACE_CHARACTERS, depth + 3)
    let elems = collect: (for e in self.elems: "\n" & ws2 & e.toString(depth * 2))
    return "Py_Tuple(\n" & (if self.elems.len > 3: "" else: $self.elems.len) & "\n" & ws1 & "elems: (" & elems.join(", ") & "\n" & ws1 & ")\n" & ws0 & ")"


proc loadProto(iter: IterPickle): ProtoPickle =
    let v = iter()

    if v != uint8 PKL_PROTO_VERSION:
        corrupted()

    return ProtoPickle(value: int v)

type Stack = seq[PY_Object]
type MetaStack = seq[Stack]
type Memo = Table[int, PY_Object]

proc loadGlobal(iter: IterPickle, stack: var Stack): GlobalPickle =
    let module = iter.readLine()
    let name = iter.readLine()

    # TODO: validate class exists

    let value = GlobalPickle(module: module, name: name)

    stack.add(value)

    return value


proc loadBinput(iter: IterPickle, stack: var Stack, memo: var Memo): BinPutPickle =
    let i = int iter()

    if i < 0:
        corrupted()

    memo[i] = stack[^1] # last element

    return BinPutPickle(index: i)

template readIntOfSize(iter: IterPickle, sz: int): int =
    var arr: array[sz, uint8]

    for i in 0..(sz - 1):
        arr[i] = iter()

    if sz == 4:
        int cast[int32](arr)
    elif sz == 2:
        int int cast[int16](arr)
    elif sz == 1:
        int cast[int8](arr)
    else:
        corrupted()

proc loadBinInt(iter: IterPickle, stack: var Stack): BinIntPickle =
    let value = BinIntPickle(value: int iter.readIntOfSize(4))
    stack.add(value)
    return value

proc loadBinInt1(iter: IterPickle, stack: var Stack): BinIntPickle =
    let value = BinIntPickle(value: int int iter())
    stack.add(value)
    return value

proc loadBinInt2(iter: IterPickle, stack: var Stack): BinIntPickle =
    let value = BinIntPickle(value: int iter.readIntOfSize(2))
    stack.add(value)
    return value

proc loadTuple1(iter: IterPickle, stack: var Stack): TuplePickle =
    let elems = @[stack[^1]]
    let tpl = TuplePickle(elems: elems)

    # replace last stack element with 1-tuple
    stack[^1] = tpl

    return tpl

proc loadTuple2(iter: IterPickle, stack: var Stack): TuplePickle =
    let elems = @[stack[^2], stack[^1]]
    let tpl = TuplePickle(elems: elems)

    # replace last 2 stack elements with 2-tuple
    stack.delete(stack.len - 1)
    stack[^1] = tpl

    return tpl

proc loadTuple3(iter: IterPickle, stack: var Stack): TuplePickle =
    let elems = @[stack[^3], stack[^2], stack[^1]]
    let tpl = TuplePickle(elems: elems)

    # replace last 3 stack elements with 3-tuple
    stack.delete((stack.len - 2)..(stack.len - 1))
    stack[^1] = tpl

    return tpl

proc popMark(stack: var Stack, metastack: var MetaStack): Stack =
    let items = stack
    stack = metastack.pop()
    return items

proc loadTuple(iter: IterPickle, stack: var Stack, metastack: var MetaStack): TuplePickle =
    let value = TuplePickle(elems: popMark(stack, metastack))
    stack.add(value)
    return value

proc loadShortBinBytes(iter: IterPickle, stack: var Stack): BinBytesPickle =
    let sz = int iter()
    var res = newSeqOfCap[char](sz)

    for _ in 0..(sz - 1):
        res.add(char iter())

    let value = BinBytesPickle(value: res)

    stack.add(value)

    return value

proc loadBinUnicode(iter: IterPickle, stack: var Stack): BinUnicodePickle =
    let sz = iter.readIntOfSize(4)
    var res = ""

    for _ in 0..(sz - 1):
        res = res & char iter()

    let value = BinUnicodePickle(value: res)

    stack.add(value)

    return value


proc newReducePickle(fn: GlobalPickle, args: TuplePickle): PY_Object =
    if fn.module == "numpy.core.multiarray" and fn.name == "_reconstruct":
        if args.elems.len != 3:
            corrupted()
        return PY_NpMultiArray()
    elif fn.module == "numpy" and fn.name == "dtype":
        if args.elems.len != 3:
            corrupted()
        if not (args.elems[0] of BinUnicodePickle):
            corrupted()
        let dtype = (BinUnicodePickle args.elems[0]).value
        if dtype != "O8":
            corrupted()
        return PY_NpDType(dtype: dtype)
    else:
        implement("REDUCE[" & fn.module & " " & fn.name & "]: " & args.toString)
    
    implement("newReducePickle")

proc loadReduce(iter: IterPickle, stack: var Stack): PY_Object =
    let args = stack.pop()
    let fn = stack[^1]
    let reduce = newReducePickle(GlobalPickle fn, TuplePickle args)

    stack[^1] = reduce

    return reduce

proc loadMark(iter: IterPickle, stack: var Stack, metastack: var MetaStack): MarkPickle =
    metastack.add(stack)
    stack = newSeq[PY_Object]()

    return MarkPickle()

proc setState(self: PY_NpDType, state: TuplePickle): void =
    self.elems.add(state.elems)
proc setState(self: PY_NpMultiArray, state: TuplePickle): void =
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__setstate__.html
    const STATE_VER = 0
    const STATE_SHAPE = 1
    const STATE_DTYPE = 2
    # const STATE_IS_FORTRAN = 3
    const STATE_RAWDATA = 4

    let np_ver = (BinIntPickle state.elems[STATE_VER]).value

    if np_ver != 1:
        corrupted()

    let np_shape = collect: (for e in (TuplePickle state.elems[STATE_SHAPE]).elems: (BinIntPickle e).value)
    let np_dtype = PY_NpDType state.elems[STATE_DTYPE]
    let np_rawdata = Py_List state.elems[STATE_RAWDATA]

    self.dtype = np_dtype.dtype
    self.elems = np_rawdata.elems
    self.shape = np_shape

proc loadBuild(stack: var Stack): PY_Object =
    let state = TuplePickle stack.pop()
    let inst = stack[^1]

    if (inst of PY_NpDType):
        setState(PY_NpDType inst, state)
    elif (inst of PY_NpMultiArray):
        setState(PY_NpMultiArray inst, state)
    else:
        corrupted()

    return inst

proc loadAppends(stack: var Stack, metastack: var MetaStack): Py_Iterable =
    let elems = popMark(stack, metastack)
    let obj = Py_Iterable stack[^1]
    
    obj.elems.add(elems)

    return obj

proc loadStop(stack: var Stack): StopPickle =
    let stop = StopPickle(value: stack.pop())

    return stop

proc loadBoolean(bval: bool, stack: var Stack): BooleanPickle =
    let value = BooleanPickle(value: bval)

    stack.add(value)

    return value

proc loadNone(stack: var Stack): NonePickle =
    let value = NonePickle()

    stack.add(value)

    return value

proc loadEmptyContainer[T: Py_Iterable | Py_Dict](stack: var Stack): T =
    let value = T()

    stack.add(value)

    return value

template unpickle(iter: IterPickle, stack: var Stack, memo: var Memo, metastack: var MetaStack): PY_Object =
    let opcode = char iter()

    case opcode:
        of PKL_PROTO: iter.loadProto()
        of PKL_GLOBAL: iter.loadGlobal(stack)
        of PKL_BINPUT: iter.loadBinput(stack, memo)
        of PKL_BININT: iter.loadBinInt(stack)
        of PKL_BININT1: iter.loadBinInt1(stack)
        of PKL_BININT2: iter.loadBinInt2(stack)
        of PKL_TUPLE1: iter.loadTuple1(stack)
        of PKL_TUPLE2: iter.loadTuple2(stack)
        of PKL_TUPLE3: iter.loadTuple3(stack)
        of PKL_SHORT_BINBYTES: iter.loadShortBinBytes(stack)
        of PKL_REDUCE: iter.loadReduce(stack)
        of PKL_MARK: iter.loadMark(stack, metastack)
        of PKL_BINUNICODE: iter.loadBinUnicode(stack)
        of PKL_NEWTRUE: loadBoolean(true, stack)
        of PKL_NEWFALSE: loadBoolean(false, stack)
        of PKL_NONE: loadNone(stack)
        of PKL_TUPLE: iter.loadTuple(stack, metastack)
        of PKL_BUILD: loadBuild(stack)
        of PKL_EMPTY_LIST: loadEmptyContainer[Py_List](stack)
        of PKL_EMPTY_DICT: loadEmptyContainer[Py_Dict](stack)
        of PKL_EMPTY_SET: loadEmptyContainer[Py_Set](stack)
        of PKL_EMPTY_TUPLE: loadEmptyContainer[Py_Tuple](stack)
        of PKL_APPENDS: loadAppends(stack, metastack)
        of PKL_STOP: loadStop(stack)
        else: raise newException(Exception, "opcode not implemeted: '" & (if opcode in PrintableChars: $opcode else: "0x" & (uint8 opcode).toHex()) & "'")

proc toString(self: PY_Object, depth: int = 0): string =
    if self of ProtoPickle: return toString(ProtoPickle self, depth)
    if self of GlobalPickle: return toString(GlobalPickle self, depth)
    if self of BinPutPickle: return toString(BinPutPickle self, depth)
    if self of BinIntPickle: return toString(BinIntPickle self, depth)
    if self of TuplePickle: return toString(TuplePickle self, depth)
    if self of BinBytesPickle: return toString(BinBytesPickle self, depth)
    if self of ReducePickle: return toString(ReducePickle self, depth)
    if self of MarkPickle: return toString(MarkPickle self, depth)
    if self of BinUnicodePickle: return toString(BinUnicodePickle self, depth)
    if self of NonePickle: return toString(NonePickle self, depth)
    if self of BooleanPickle: return toString(BooleanPickle self, depth)
    if self of BuildPickle: return toString(BuildPickle self, depth)
    if self of AppendsPickle: return toString(AppendsPickle self, depth)
    if self of StopPickle: return toString(StopPickle self, depth)
    if self of Py_Dict: return toString(Py_Dict self, depth)
    if self of Py_List: return toString(Py_List self, depth)
    if self of Py_Set: return toString(Py_Set self, depth)
    if self of Py_Tuple: return toString(Py_Tuple self, depth)
    if self of PY_NpDType: return repr(PY_NpDType self)
    if self of PY_NpMultiArray: return repr(PY_NpMultiArray self)

    return "^BasePickle"

proc toPage(arr: PY_NpMultiArray): ObjectNDArray =
    let buf = collect: (for e in arr.elems: PY_ObjectND e)
    return ObjectNDArray(shape: arr.shape, buf: buf)

proc construct(self: StopPickle): ObjectNDArray =
    if not (self.value of PY_NpMultiArray):
        corrupted()
    
    return (PY_NpMultiArray self.value).toPage()

proc newObjectNDArray(fh: var File, endianness: Endianness, shape: var seq[int]): ObjectNDArray =
    var elements = calcShapeElements(shape)
    let iter = unpickleFile(fh, endianness)
    var stack: Stack = newSeq[PY_Object]()
    var metastack: MetaStack = newSeq[Stack]()
    var memo: Memo = initTable[int, PY_Object]()

    var stop: StopPickle
    var hasStop = false

    while unlikely(not iter.finished):
        let opcode = iter.unpickle(stack, memo, metastack)

        if opcode of StopPickle:
            hasStop = true
            stop = StopPickle opcode
            break

    if unlikely(not hasStop):
        corrupted()

    let page = stop.construct()

    if calcShapeElements(page.shape) != elements:
        corrupted()

    return page

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
        of D_OBJECT: page = newObjectNDArray(fh, descr_endianness, shape)
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
