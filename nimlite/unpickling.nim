import std/[sequtils, strutils, sugar, tables, endians, enumerate]
from nimpy import `.()`, toPyObjectArgument, `$`, `.`, to
from nimpy/raw_buffers import RawPyBuffer, PyBUF_WRITABLE, PyBUF_ND, getBuffer, release
import pytypes, pickleproto, utils
import pymodules as pymodules

type IterPickle = iterator(): uint8
type Stack = seq[PY_Object]
type MetaStack = seq[Stack]
type Memo = Table[uint, PY_Object]
type ObjectPage = (Shape, seq[PY_ObjectND], Table[KindObjectND, int])

type ProtoPickle = ref object of PY_Object
    value: int
type GlobalPickle = ref object of PY_Object
    module: string
    name: string
type BinPutPickle = ref object of PY_Object
    index: uint

type TuplePickle = ref object of Py_Tuple
type BinBytesPickle = ref object of PY_Bytes

type BinIntPickle = ref object of PY_Int
type BinFloatPickle = ref object of PY_Float
type BinUnicodePickle = ref object of PY_String
type BooleanPickle = ref object of PY_Boolean
type NonePickle = ref object of PY_NoneType
    
type ReducePickle = ref object of Py_Object
    value: PY_Object

type MarkPickle = ref object of PY_Object
type AppendPickle = ref object of PY_Object
type BuildPickle = ref object of PY_Object
type AppendsPickle = ref object of Py_Iterable
type StopPickle = ref object of PY_Object
    value: PY_Object

type PY_NpMultiArray* = ref object of Py_Object
    shape: Shape
    dtype: string
    elems: seq[PY_Object]
type PY_NpDType* = ref object of Py_Iterable
    dtype: string

proc toString(self: PY_Object, depth: int = 0): string {.inline.};
proc `$`*(self: PY_Object): string {.inline.} = return self.toString()

proc unpickleFile*(fh: File, endianness: Endianness): IterPickle =
    const READ_BUF_SIZE = 1_000_000 # 1MB chunks
    var buf {.noinit.}: array[READ_BUF_SIZE, uint8]

    return iterator(): uint8 =
        while not unlikely(fh.endOfFile):
            let bytesRead = fh.readBuffer(addr buf, READ_BUF_SIZE)

            for i in 0..<bytesRead:
                yield buf[i]

proc readLine(iter: var IterPickle): string =
    var res = ""

    while not iter.finished:
        let ch = char iter()

        if ch == '\n':
            break

        res = res & ch

    return res


const WHITESPACE_CHARACTERS = "    "

proc toString(self: ProtoPickle, depth: int): string =
    return "PROTO(value: " & $self.value & ")"
proc toString(self: GlobalPickle, depth: int): string =
    return "GLOBAL(module: '" & self.module & "', name: '" & self.name & "')"
proc toString(self: BinPutPickle, depth: int): string =
    return "BINPUT(index: " & $self.index & ")"
proc toString(self: BinIntPickle, depth: int): string =
    return "BININT" & "(value: " & $self.value & ")"
proc toString(self: BinFloatPickle, depth: int): string =
    return "BinFloatPickle" & "(value: " & $self.value & ")"
proc toString(self: TuplePickle, depth: int): string =
    let ws0 = repeat(WHITESPACE_CHARACTERS, depth)
    let ws1 = repeat(WHITESPACE_CHARACTERS, depth + 2)
    let ws2 = repeat(WHITESPACE_CHARACTERS, depth + 3)
    let elems = collect: (for e in self.elems: "\n" & ws2 & e.toString(depth * 2))
    return "TUPLE(" & (if self.elems.len > 3: "" else: $self.elems.len) & "\n" & ws1 & "elems: (" & elems.join(", ") & "\n" & ws1 & ")\n" & ws0 & ")"
proc toString(self: BinBytesPickle, depth: int): string =
    return "BINBYTES(value: b'" & self.value.join("") & "')"
proc toString(self: ReducePickle, depth: int): string =
    return "REDUCE()"
proc toString(self: MarkPickle, depth: int): string =
    return "MARK()"
proc toString(self: BinUnicodePickle, depth: int): string =
    return "BINUNICODE(value: '" & $self.value & "')"
proc toString(self: NonePickle, depth: int): string =
    return "NONE()"
proc toString(self: BooleanPickle, depth: int): string =
    return "BOOLEAN(value: " & $self.value & ")"
proc toString(self: BuildPickle, depth: int): string =
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

proc toString(self: PY_Object, depth: int = 0): string =
    if self of ProtoPickle: return toString(ProtoPickle self, depth)
    if self of GlobalPickle: return toString(GlobalPickle self, depth)
    if self of BinPutPickle: return toString(BinPutPickle self, depth)
    if self of BinIntPickle: return toString(BinIntPickle self, depth)
    if self of BinFloatPickle: return toString(BinFloatPickle self, depth)
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
    if self of PY_Date: return $PY_Date(self)
    if self of PY_DateTime: return $PY_DateTime(self)
    if self of PY_Time: return $PY_Time(self)

    return "^PY_Object"

template readOfSize(iter: var IterPickle, sz: int): openArray[uint8] =
    var arr: array[sz, uint8]

    for i in 0..<sz:
        arr[i] = iter()

    arr

template readIntOfSize(iter: var IterPickle, sz: int): int =
    when sz == 4: int cast[int32](iter.readOfSize(sz))
    elif sz == 2: int cast[uint16](iter.readOfSize(sz))
    elif sz == 1: int cast[uint8](iter.readOfSize(sz))
    else:
        raise newException(IOError, "invalid int size: " & $sz)

proc loadProto(iter: var IterPickle): ProtoPickle {.inline.} =
    let v = iter()

    if v != uint8 PKL_PROTO_VERSION:
        raise newException(IOError, "invalid protocol version: " & $v)

    return ProtoPickle(value: int v)


proc loadGlobal(iter: var IterPickle, stack: var Stack): GlobalPickle {.inline.} =
    let module = iter.readLine()
    let name = iter.readLine()

    let value = GlobalPickle(module: module, name: name)

    stack.add(value)

    return value


proc loadBinput(iter: var IterPickle, stack: var Stack, memo: var Memo): BinPutPickle {.inline.} =
    let i = uint iter()

    memo[i] = stack[^1] # last element

    return BinPutPickle(index: i)

proc loadLongBinput(iter: var IterPickle, stack: var Stack, memo: var Memo): BinPutPickle {.inline.} =
    let i = uint iter.readIntOfSize(4)

    memo[i] = stack[^1] # last element

    return BinPutPickle(index: i)

proc loadBinGet(iter: var IterPickle, stack: var Stack, memo: var Memo): PY_Object {.inline.} =
    let idx = uint iter()
    let obj = memo[idx]

    stack.add(obj)

    return obj

proc loadLongBinGet(iter: var IterPickle, stack: var Stack, memo: var Memo): PY_Object {.inline.} =
    let idx = uint iter.readIntOfSize(4)
    let obj = memo[idx]

    stack.add(obj)

    return obj

proc loadBinFloat(iter: var IterPickle, stack: var Stack): BinFloatPickle {.inline.} =
    var arr: array[8, uint8]

    for i in countdown(7, 0):
        arr[i] = iter()

    let flt = float cast[float64](arr)
    let value = BinFloatPickle(value: flt, kind: K_FLOAT)

    stack.add(value)

    return value

proc loadBinInt(iter: var IterPickle, stack: var Stack): BinIntPickle {.inline.} =
    let value = BinIntPickle(value: int iter.readIntOfSize(4), kind: K_INT)
    stack.add(value)
    return value

proc loadBinInt1(iter: var IterPickle, stack: var Stack): BinIntPickle {.inline.} =
    let value = BinIntPickle(value: int iter(), kind: K_INT)
    stack.add(value)
    return value

proc loadBinInt2(iter: var IterPickle, stack: var Stack): BinIntPickle {.inline.} =
    let value = BinIntPickle(value: int iter.readIntOfSize(2), kind: K_INT)
    stack.add(value)
    return value

proc loadTuple1(iter: var IterPickle, stack: var Stack): TuplePickle {.inline.} =
    let elems = @[stack[^1]]
    let tpl = TuplePickle(elems: elems)

    # replace last stack element with 1-tuple
    stack[^1] = tpl

    return tpl

proc loadTuple2(iter: var IterPickle, stack: var Stack): TuplePickle {.inline.} =
    let elems = @[stack[^2], stack[^1]]
    let tpl = TuplePickle(elems: elems)

    # replace last 2 stack elements with 2-tuple
    stack.delete(stack.len - 1)
    stack[^1] = tpl

    return tpl

proc loadTuple3(iter: var IterPickle, stack: var Stack): TuplePickle {.inline.} =
    let elems = @[stack[^3], stack[^2], stack[^1]]
    let tpl = TuplePickle(elems: elems)

    # replace last 3 stack elements with 3-tuple
    stack.delete((stack.len - 2)..(stack.len - 1))
    stack[^1] = tpl

    return tpl

proc popMark(stack: var Stack, metastack: var MetaStack): Stack {.inline.} =
    let items = stack
    stack = metastack.pop()
    return items

proc loadTuple(iter: var IterPickle, stack: var Stack, metastack: var MetaStack): TuplePickle {.inline.} =
    let value = TuplePickle(elems: popMark(stack, metastack))
    stack.add(value)
    return value

proc loadShortBinBytes(iter: var IterPickle, stack: var Stack): BinBytesPickle {.inline.} =
    let sz = int iter()
    var res = newSeqOfCap[char](sz)

    for _ in 0..<sz:
        res.add(char iter())

    let value = BinBytesPickle(value: res)

    stack.add(value)

    return value

proc loadBinUnicode(iter: var IterPickle, stack: var Stack): BinUnicodePickle {.inline.} =
    let sz = iter.readIntOfSize(4)
    var res = ""

    for _ in 0..<sz:
        res = res & char iter()

    let value = BinUnicodePickle(value: res, kind: K_STRING)

    stack.add(value)

    return value


proc newReducePickle(fn: var GlobalPickle, args: var TuplePickle): PY_Object =
    if fn.module == "numpy.core.multiarray" and fn.name == "_reconstruct":
        if args.elems.len != 3:
            raise newException(IOError, "invalid arguments for numpy.core.multiarray")
        return PY_NpMultiArray()
    elif fn.module == "numpy" and fn.name == "dtype":
        if args.elems.len != 3 or not (args.elems[0] of BinUnicodePickle):
            raise newException(IOError, "invalid arguments for numpy.dtype")
        let dtype = (BinUnicodePickle args.elems[0]).value

        return PY_NpDType(dtype: dtype)
    elif fn.module == "datetime" and fn.name == "date":
        if args.elems.len != 1 or not (args.elems[0] of BinBytesPickle):
            raise newException(IOError, "invalid arguments for datetime.date")
        let bytes = (BinBytesPickle args.elems[0]).value
        var year: uint16
        var month, day: uint8

        swapEndian16(addr year, addr bytes[0])
        copyMem(addr month, addr bytes[2], 1)
        copyMem(addr day, addr bytes[3], 1)

        return newPY_Date(year, month, day)
    elif fn.module == "datetime" and fn.name == "datetime":
        if args.elems.len != 1 or not (args.elems[0] of BinBytesPickle):
            raise newException(IOError, "invalid arguments for datetime.datetime")

        let bytes = (BinBytesPickle args.elems[0]).value
        var year: uint16
        var month, day: uint8
        var hour, minute, second: uint8
        var microsecond: uint32

        swapEndian16(addr year, addr bytes[0])
        copyMem(addr month, addr bytes[2], 1)
        copyMem(addr day, addr bytes[3], 1)
        copyMem(addr hour, addr bytes[4], 1)
        copyMem(addr minute, addr bytes[5], 1)
        copyMem(addr second, addr bytes[6], 1)
        copyMem(addr microsecond, addr bytes[7], 3)
        swapEndian32(addr microsecond, addr microsecond)

        return newPY_DateTime(
            year, month, day,
            hour, minute, second, microsecond
        )
    elif fn.module == "datetime" and fn.name == "time":
        if args.elems.len != 1 or not (args.elems[0] of BinBytesPickle):
            raise newException(IOError, "invalid arguments for datetime.time")

        let bytes = (BinBytesPickle args.elems[0]).value
        var hour, minute, second: uint8
        var microsecond: uint32

        copyMem(addr hour, addr bytes[0], 1)
        copyMem(addr minute, addr bytes[1], 1)
        copyMem(addr second, addr bytes[2], 1)
        copyMem(addr microsecond, addr bytes[3], 3)
        swapEndian32(addr microsecond, addr microsecond)

        return newPY_Time(hour, minute, second, microsecond)
    elif fn.module == "numpy.core.multiarray" and fn.name == "scalar":
        # mega slow but sometimes we get terrible pages
        let dtypeName = PY_NpDType(args.elems[0]).dtype
        let bytes = BinBytesPickle(args.elems[1]).value
        let byteCount = bytes.len
        let np = pymodules.modules().numpy.module
        let dtypePy = np.dtype(dtypeName)

        let bytesPy = np.empty(byteCount, dtype="bytes") # can't do it we builtins.bytes() because it creates readonly buffer
        var buf: RawPyBuffer
        
        bytesPy.getBuffer(buf, PyBUF_WRITABLE or PyBUF_ND)

        if byteCount > 0:
            # copy memory
            buf.buf.copyMem(addr bytes[0], byteCount)

        buf.release()

        let pyBytes = bytesPy.data.tobytes()
        let valPy = np.core.multiarray.scalar(dtypePy, pyBytes).tolist()
        let typeName = pymodules.modules().builtins.getTypeName(valPy)

        case typeName: # construct the nim native python object
        of "float": return newPY_Object(valPy.to(float))
        of "int": return newPY_Object(valPy.to(int))
        of "bool": return newPY_Object(valPy.to(bool))
        else: raise newException(IOError, "not supported type: " & typeName)
    else:
        implement("REDUCE[" & fn.module & " " & fn.name & "]: " & args.toString)

proc loadReduce(iter: var IterPickle, stack: var Stack): PY_Object {.inline.} =
    var args = stack.pop()
    var fn = stack[^1]
    let reduce = newReducePickle(GlobalPickle fn, TuplePickle args)

    stack[^1] = reduce

    return reduce

proc loadMark(iter: var IterPickle, stack: var Stack, metastack: var MetaStack): MarkPickle {.inline.} =
    metastack.add(stack)
    stack = newSeq[PY_Object]()

    return MarkPickle()

proc setState(self: var PY_NpDType, state: var TuplePickle): void {.inline.} = self.elems.add(state.elems)
proc setState(self: var PY_NpMultiArray, state: var TuplePickle): void {.inline.} =
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.__setstate__.html
    const STATE_VER = 0
    const STATE_SHAPE = 1
    const STATE_DTYPE = 2
    # const STATE_IS_FORTRAN = 3
    const STATE_RAWDATA = 4

    let np_ver = (BinIntPickle state.elems[STATE_VER]).value

    if np_ver != 1:
        raise newException(IOError, "unsupported numpy version: " & $np_ver)

    let np_shape = collect: (for e in (TuplePickle state.elems[STATE_SHAPE]).elems: (BinIntPickle e).value)
    let np_dtype = PY_NpDType state.elems[STATE_DTYPE]
    let np_rawdata = Py_List state.elems[STATE_RAWDATA]

    self.dtype = np_dtype.dtype
    self.elems = np_rawdata.elems
    self.shape = np_shape

proc loadBuild(stack: var Stack): PY_Object {.inline.} =
    var state = TuplePickle stack.pop()
    var inst = stack[^1]

    if (inst of PY_NpDType): setState(PY_NpDType inst, state)
    elif (inst of PY_NpMultiArray): setState(PY_NpMultiArray inst, state)
    else: raise newException(IOError, "unsupported build type: " & inst.toString)

    return inst

proc loadAppend(stack: var Stack): AppendPickle {.inline.} =
    let value = stack.pop()
    let list = Py_Iterable stack[^1]

    list.elems.add(value)

    return AppendPickle()

proc loadAppends(stack: var Stack, metastack: var MetaStack): Py_Iterable {.inline.} =
    let elems = popMark(stack, metastack)
    let obj = Py_Iterable stack[^1]
    
    obj.elems.add(elems)

    return obj

proc loadStop(stack: var Stack): StopPickle {.inline.} =
    return StopPickle(value: stack.pop())

proc loadBoolean(bval: bool, stack: var Stack): BooleanPickle {.inline.} =
    let value = BooleanPickle(value: bval, kind: K_BOOLEAN)

    stack.add(value)

    return value

proc loadNone(stack: var Stack): NonePickle {.inline.} =
    let value = NonePickle(kind: K_NONETYPE)

    stack.add(value)

    return value

proc loadEmptyContainer[T: Py_Iterable | Py_Dict](stack: var Stack): T {.inline.} =
    let value = T()

    stack.add(value)

    return value

proc unpickle(iter: var IterPickle, stack: var Stack, memo: var Memo, metastack: var MetaStack): PY_Object {.inline.} =
    let opcode = char iter()

    case opcode:
    of PKL_PROTO: iter.loadProto()
    of PKL_GLOBAL: iter.loadGlobal(stack)
    of PKL_BINPUT: iter.loadBinput(stack, memo)
    of PKL_LONG_BINPUT: iter.loadLongBinput(stack, memo)
    of PKL_BININT: iter.loadBinInt(stack)
    of PKL_BININT1: iter.loadBinInt1(stack)
    of PKL_BININT2: iter.loadBinInt2(stack)
    of PKL_BINFLOAT: iter.loadBinFloat(stack)
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
    of PKL_APPEND: loadAppend(stack)
    of PKL_APPENDS: loadAppends(stack, metastack)
    of PKL_STOP: loadStop(stack)
    of PKL_BINGET: iter.loadBinGet(stack, memo)
    of PKL_LONG_BINGET: iter.loadLongBinGet(stack, memo)
    else: raise newException(IOError, "opcode not implemeted: '" & (if opcode in PrintableChars: $opcode else: "0x" & (uint8 opcode).toHex()) & "'")

proc getType(self: PY_ObjectND): KindObjectND =
    case self.kind:
    of K_NONETYPE: return K_NONETYPE
    of K_BOOLEAN: return K_BOOLEAN
    of K_INT: return K_INT
    of K_FLOAT: return K_FLOAT
    of K_STRING: return K_STRING
    of K_DATE: return K_DATE
    of K_TIME: return K_TIME
    of K_DATETIME: return K_DATETIME

proc toPage(arr: var PY_NpMultiArray): ObjectPage =
    var buf = newSeqOfCap[PY_ObjectND](arr.elems.len)
    var dtypes = initTable[KindObjectND, int]()

    for (i, e) in enumerate(arr.elems):
        let obj = PY_ObjectND(e)
        let dt = obj.getType()
        
        buf.add(obj)
        
        if not (dt in dtypes):
            dtypes[dt] = 0

        dtypes[dt] = dtypes[dt] + 1

    return (arr.shape, buf, dtypes)

proc construct(self: var StopPickle): ObjectPage {.inline.} =
    if not (self.value of PY_NpMultiArray):
        raise newException(IOError, "unsupported construct type: " & self.value.toString)
    
    return PY_NpMultiArray(self.value).toPage()

proc readPickledPage*(fh: var File, endianness: Endianness, shape: var Shape): ObjectPage =
    var iter = unpickleFile(fh, endianness)
    var stack: Stack = newSeq[PY_Object]()
    var metastack: MetaStack = newSeq[Stack]()
    var memo: Memo = initTable[uint, PY_Object]()

    var stop: StopPickle
    var hasStop = false

    while unlikely(not iter.finished):
        let opcode = iter.unpickle(stack, memo, metastack)

        if opcode of StopPickle:
            hasStop = true
            stop = StopPickle opcode
            break

    if unlikely(not hasStop):
        raise newException(IOError, "no stop instruction in pickle file")

    return stop.construct()