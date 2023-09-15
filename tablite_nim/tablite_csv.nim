# import faster_than_csv as csv

# type DataTypes = enum
#     INT, BOOLEAN, FLOAT,
#     STRING,
#     DATE, TIME, DATETIME,
#     MAX_ELEMENTS


# type Rank = object
#     items: array[DataTypes.MAX_ELEMENTS, int]
#     ranks: array[DataTypes.MAX_ELEMENTS, int]

# iterator iter(rank: Rank): DataTypes {.closure.} =
#     var x = 0
#     let max = int(DataTypes.MAX_ELEMENTS)
#     while x < max:
#         yield DataTypes(x)
#         inc x

# proc newRank(): Rank =
#     var items: array[int(DataTypes.MAX_ELEMENTS), int]

#     for i in 0..(int(DataTypes.MAX_ELEMENTS)-1):
#         items[i] = i

#     return Rank(items: items)

import argparse
import std/enumerate
import os, sugar, times, tables, sequtils, json, unicode, parseutils, encodings, bitops, osproc

type Encodings {.pure.} = enum ENC_UTF8, ENC_UTF16

type BaseEncodedFile = ref object of RootObj
    fh: File

type FileUTF8 = ref object of BaseEncodedFile
type FileUTF16 = ref object of BaseEncodedFile
    endianness: Endianness

proc endOfFile(f: BaseEncodedFile): bool = f.fh.endOfFile()
proc getFilePos(f: BaseEncodedFile): uint = uint f.fh.getFilePos()
proc setFilePos(f: BaseEncodedFile, pos: int64, relativeTo: FileSeekPos): void = f.fh.setFilePos(pos, relativeTo)
proc close(f: BaseEncodedFile): void = f.fh.close()

proc readLine(f: FileUTF8, str: var string): bool = f.fh.readLine(str)
proc readLine(f: FileUTF16, str: var string): bool = 
    var ch_arr {.noinit.}: array[2, uint8]
    var ch: uint16

    let newline_char: uint16 = 0x000a
    var wchar_seq {.noinit.} = newSeqOfCap[uint16](80)

    while unlikely(not f.endOfFile):
        if f.fh.readBuffer(addr ch_arr, 2) != ch_arr.len:
            raise newException(Exception, "malformed file")

        if f.endianness == bigEndian: # big if true
            (ch_arr[0], ch_arr[1]) = (ch_arr[1], ch_arr[0])

        ch = cast[uint16](ch_arr)

        if newline_char == ch:
            break

        wchar_seq.add(ch)

    var wstr {.noinit.} = newWideCString(wchar_seq.len)

    if wchar_seq.len > 0:
        copyMem(wstr[0].unsafeAddr, wchar_seq[0].unsafeAddr, wchar_seq.len * 2)
    else:
        return false

    str = $wstr

    return true

proc readLine(f: BaseEncodedFile, str: var string): bool = 
    if f of FileUTF8:
        return readLine(cast[FileUTF8](f), str)
    elif f of FileUTF16:
        return readLine(cast[FileUTF16](f), str)
    else:
        raise newException(Exception, "encoding not implemented")

proc newFileUTF16(filename: string): FileUTF16 =
    var fh = open(filename, fmRead)

    if fh.getFileSize() mod 2 != 0:
        raise newException(Exception, "invalid size")

    var bom_bytes: array[2, uint16]
    
    if fh.readBuffer(addr bom_bytes, bom_bytes.len) != bom_bytes.len:
        raise newException(Exception, "cannot find bom")

    var bom = cast[uint16](bom_bytes)
    var endianness: Endianness;

    if bom == 0xfeff:
        endianness = Endianness.littleEndian
    elif bom == 0xfffe:
        endianness = Endianness.bigEndian
    else:
        raise newException(Exception, "invalid bom")

    return FileUTF16(fh: fh, endianness: endianness)

proc newFile(filename: string, encoding: Encodings): BaseEncodedFile =
    var f: BaseEncodedFile
    case encoding:
        of ENC_UTF8:
            return FileUTF8(fh: open(filename, fmRead))
        of ENC_UTF16:
            return newFileUTF16(filename)
        else:
            raise newException(Exception, "encoding not implemented")


proc peek_char(fh: File): (bool, char) =
    if unlikely(fh.endOfFile()):
        return (false, ' ')

    let ox = fh.getFilePos()
    let peeked = fh.readChar()

    fh.setFilePos(ox, FileSeekPos.fspSet)

    return (true, peeked)

proc seek_next_line(fh: File): bool =
    while likely(not fh.endOfFile()):
        let ch = fh.readChar()

        if unlikely(ch == '\n'):
            return true

    return false

proc find_newlines(fh: BaseEncodedFile): (seq[uint], uint) =
    var newline_offsets = newSeq[uint](1)
    var total_lines: uint = 0
    var str: string

    newline_offsets[0] = fh.getFilePos()

    while likely(fh.readLine(str)):
        inc total_lines

        newline_offsets.add(fh.getFilePos())

    return (newline_offsets, total_lines)

proc find_newlines(path: string, encoding: Encodings): (seq[uint], uint) =
    let fh = newFile(path, encoding)
    try:
        return find_newlines(fh)
    finally:
        fh.close()

# proc find_newlines_cached(path: string, encoding: Encodings): (seq[int], int) =
#     let fh = newFile(path, encoding)
#     try:
#         var newline_offsets = newSeq[int](1)
#         var ring_buffer: array[8192, char]
#         var bufpos = 0
#         var bufsize = 0
#         var bufoffset = 0

#         var total_lines: int = 0

#         newline_offsets[0] = 0

#         while likely(not fh.endOfFile):
#             if bufpos > bufsize:
#                 bufpos = 0
#                 bufoffset = bufsize
#                 bufsize = fh.readBuffer(addr ring_buffer, ring_buffer.len)

#             let ch = ring_buffer[bufpos]
#             inc bufpos

#             if ch == '\n':
#                 inc total_lines
#                 newline_offsets.add(bufoffset + bufpos)

#         # while likely(fh.seek_next_line()):
#         #     inc total_lines

#         #     newline_offsets.add(int fh.getFilePos())

#         return (newline_offsets, total_lines)
#     finally:
#         fh.close()

# import streams, lexbase, times

import lists

type Quoting {.pure.} = enum
    QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONNUMERIC, QUOTE_NONE,
    QUOTE_STRINGS, QUOTE_NOTNULL

type ParserState {.pure.} = enum
    START_RECORD, START_FIELD, ESCAPED_CHAR, IN_FIELD,
    IN_QUOTED_FIELD, ESCAPE_IN_QUOTED_FIELD, QUOTE_IN_QUOTED_FIELD,
    EAT_CRNL, AFTER_ESCAPED_CRNL

type Dialect = object
    delimiter: char
    quotechar: char
    escapechar: char
    doublequote: bool
    quoting: Quoting
    skipinitialspace: bool
    lineterminator: char
    strict: bool

proc newDialect(delimiter: char = ',', quotechar: char = '"', escapechar: char = '\\', doublequote: bool = true, quoting: Quoting = QUOTE_MINIMAL, skipinitialspace: bool = false, lineterminator: char = '\n'): Dialect =
    Dialect(delimiter:delimiter, quotechar:quotechar, escapechar:escapechar, doublequote:doublequote, quoting:quoting, skipinitialspace:skipinitialspace, lineterminator:lineterminator)

const fields_count: uint = 128;
const field_limit: uint = 128 * 1024;

type BufType = array[field_limit, uint8]

type StringContainer = object
    buf: seq[BufType]
    szs: seq[uint]
    els: uint

proc newStringContainer(): StringContainer =
    StringContainer(
        buf: newSeq[BufType](fields_count),
        szs: newSeq[uint](fields_count)
    )

# type HeapString = object
#     str: string

# proc newHeapString(str: var string): ref HeapString =
#   var heapValue = new(type(HeapString))
#   heapValue[].str = str
#   heapValue

type ReaderObj = object
    numeric_field: bool
    line_num: uint
    dialect: Dialect

    field_len: uint
    field_size: uint
    field: seq[char]

    # container: StringContainer
    fields: seq[string]
    # fields: seq[ref HeapString]
    # fields: SinglyLinkedList[string]
    field_count: uint

var readerAlloc = newSeq[string](1024)
# var readerAlloc = newSeq[ptr string](1024)
# var readerAlloc = newSeq[ref HeapString](1024)
# var readerAlloc = initSinglyLinkedList()

# var arr = newSeq[array[field_limit, uint8]](1024)
# var container = newStringContainer()

proc newReaderObj(dialect: Dialect): ReaderObj =
    ReaderObj(dialect: dialect, fields: readerAlloc)
    # ReaderObj(dialect: dialect, container: container)

# proc add_field(self: var ReaderObj): void =
#     if self.field_len > 0:
#         copyMem(self.container[self.container.els].unsafeAddr, self.field[0].unsafeAddr, self.field_len)

#     self.container[self.container.els] = self.field_len
#     inc self.container.els

proc parse_grow_buff(self: var ReaderObj): bool =
    let field_size_new: uint = (if self.field_size > 0: 2u * self.field_size else: 4096u)
    
    self.field.setLen(field_size_new)
    self.field_size = field_size_new

    return true

proc parse_add_char(self: var ReaderObj, state: var ParserState, c: char): bool =
    if self.field_len >= field_limit:
        return false

    if unlikely(self.field_len == self.field_size and not self.parse_grow_buff()):
        return false

    self.field[self.field_len] = c
    inc self.field_len

    return true



proc parse_save_field(self: var ReaderObj): bool =
    # self.field_len = 0
    # return true
    # self.field_len = 0

    if self.numeric_field:
        self.numeric_field = false

        raise newException(Exception, "not yet implemented: parse_save_field numeric_field")

    var field {.noinit.} = newString(self.field_len)
    # var field = new(type(string))
    
    # var field = self.fields[self.field_count]
    # field.setLen(self.field_len)

    # echo "save field len: " & $self.field_len# & " field: " & $self.field

    if likely(self.field_len > 0):
        copyMem(field[0].unsafeAddr, self.field[0].unsafeAddr, self.field_len)

    # echo "save field: '" & field & "'"
    # self.fields.add(newSinglyLinkedNode(field))
    # self.fields[self.field_count] = newHeapString(field)
    # self.fields[self.field_count] = addr field
    if unlikely(self.field_count + 1 >= (uint self.field.high)):
        self.field.setLen(self.field.len() * 2)

    self.fields[self.field_count] = field

    inc self.field_count

    # echo "--- capacity: " & $self.container.buf.high

    # if (uint self.container.buf.high) < self.container.els:
    #     var arr {.noinit.}: BufType

    #     self.container.buf.add(arr)
    #     self.container.szs.add(0)

    # if self.field_len > 0:
    #     copyMem(self.container.buf[self.container.els].unsafeAddr, self.field[0].unsafeAddr, self.field_len)

    # self.container.szs[self.container.els] = self.field_len
    # inc self.container.els

    self.field_len = 0

    return true

let NOT_SET = uint32.high
let EOL = uint32.high - 1

proc parse_process_char(self: var ReaderObj, state: var ParserState, cc: uint32): bool =
    let dia = self.dialect
    var ch_code = cc
    var c = (if ch_code < EOL: char ch_code else: '\x00')

    case state:
        of START_RECORD, START_FIELD:
            if ch_code == EOL:
                return true

            if state == START_RECORD: # nim cannot fall through
                if unlikely(c in ['\n', '\r']):
                    state = EAT_CRNL
                else:
                    state = START_FIELD

            if unlikely(c in ['\n', '\r'] or unlikely(ch_code == EOL)):
                if unlikely(not self.parse_save_field()):
                    return false

                state = (if ch_code == EOL: START_RECORD else: EAT_CRNL)
            elif unlikely(c == dia.quotechar and dia.quoting != QUOTE_NONE):
                state = IN_QUOTED_FIELD
            elif unlikely(c == dia.escapechar):
                state = ESCAPED_CHAR
            elif unlikely(c == ' ' and dia.skipinitialspace):
                discard
            elif unlikely(c == dia.delimiter):
                if unlikely(not self.parse_save_field()):
                    return false
            else:
                if dia.quoting == QUOTE_NONNUMERIC:
                    self.numeric_field = true
                if unlikely(not self.parse_add_char(state, c)):
                    return false
                state = IN_FIELD
        of ESCAPED_CHAR:
            if c in ['\n', '\r']:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
                state = AFTER_ESCAPED_CRNL

            if ch_code == EOL:
                c = '\n'
                ch_code = uint32 c

            if unlikely(not self.parse_add_char(state, c)):
                return false

            state = IN_FIELD
        of AFTER_ESCAPED_CRNL, IN_FIELD:
            if state == AFTER_ESCAPED_CRNL and ch_code == EOL:
                return true # nim is stupid

            if unlikely(c in ['\n', '\r'] or unlikely(ch_code == EOL)):
                if unlikely(not self.parse_save_field()):
                    return false
                state = (if ch_code == EOL: START_RECORD else: EAT_CRNL)
            elif c == dia.escapechar:
                state = ESCAPED_CHAR
            elif c == dia.delimiter:
                if unlikely(not self.parse_save_field()):
                    return false
                state = START_FIELD
            else:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
        of IN_QUOTED_FIELD:
            if ch_code == EOL:
                discard
            elif c == dia.escapechar:
                state = ESCAPE_IN_QUOTED_FIELD
            elif c == dia.quotechar and dia.quoting != QUOTE_NONE:
                if dia.doublequote:
                    state = QUOTE_IN_QUOTED_FIELD
                else:
                    state = IN_FIELD
            else:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
        of ESCAPE_IN_QUOTED_FIELD:
            if ch_code == EOL:
                c = '\n'
                ch_code = uint32 c
            
            if unlikely(not self.parse_add_char(state, c)):
                return false

            state = IN_QUOTED_FIELD
        of QUOTE_IN_QUOTED_FIELD:
            if dia.quoting != QUOTE_NONE and c == dia.quotechar:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
                state = IN_QUOTED_FIELD
            elif c == dia.delimiter:
                if unlikely(not self.parse_save_field()):
                    return false
                state = START_FIELD
            elif c in ['\n', '\r'] or ch_code == EOL:
                if unlikely(not self.parse_save_field()):
                    return false
                state = (if ch_code == EOL: START_RECORD else: EAT_CRNL)
            elif not dia.strict:
                if unlikely(not self.parse_add_char(state, c)):
                    return false
                state = IN_FIELD
            else:
                return false
        of EAT_CRNL:
            if c in ['\n', '\r']:
                discard
            elif ch_code == EOL:
                state = START_RECORD
            else:
                return false

    return true

# import encodings

iterator parse_csv(self: var ReaderObj, fh: BaseEncodedFile): (uint, ptr seq[string], uint) =
    let dia = self.dialect

    var state: ParserState = START_RECORD
    var line_num: uint = 0
    var line = newStringOfCap(80)
    var pos: uint
    var linelen: uint;

    self.field_len = 0
    self.field_count = 0

    while likely(not fh.endOfFile):
        if not fh.readLine(line):
            break

        if self.field_len != 0 and state == IN_QUOTED_FIELD:
            if dia.strict:
                raise newException(Exception, "unexpected end of data")
            elif self.parse_save_field():
                break

        line.add('\n')

        
        linelen = uint line.len
        pos = 0

        # echo "line: '" & line & "'"
        # echo "state: " & $state

        # state = START_RECORD

        while pos < linelen:
            if unlikely(not self.parse_process_char(state, uint32 line[pos])):
                # echo $state & " " & line[pos] & " " & $obj.fields & " " & $obj.field_len
                raise newException(Exception, "illegal")
            
            inc pos

        # if unlikely(not obj.parse_process_char(state, uint32 '\n')):
        #     raise newException(Exception, "illegal")

        if unlikely(not self.parse_process_char(state, EOL)):
            raise newException(Exception, "illegal")

        # for i in 0..32:
        #     obj.fields.add(cunny_fields[i])

        # var addr_fields = addr obj.fields
        # yield it
        # obj.fields = initSinglyLinkedList[string]()
        # obj.fields.setLen(0)
        # yield (addr obj.fields, obj.field_count)
        # yield line_num
        yield (line_num, addr self.fields, self.field_count)

        self.field_count = 0
        # self.container.els = 0

        inc line_num

        # raise newException(Exception, "not yet implemented")

iterator parse_csv(self: var ReaderObj, path: string, encoding: Encodings): (uint, ptr seq[string], uint) =
    var fh = newFile(path, encoding)

    try:
        for it in self.parse_csv(fh):
            yield it
    finally:
        fh.close()


# iterator iter_fields(self: var ReaderObj): ptr string =
#     for i in 0..self.container.els - 1:
#         let sz = self.container.szs[i]
#         let pt = self.container.buf[i]

#         var st {.noinit.} = newString(sz)

#         if sz > 0:
#             copyMem(st[0].addr, pt[0].unsafeAddr, sz)

#         # echo $i & " -> " & $st

#         yield addr st

proc read_columns(path: string, encoding: Encodings, dialect: Dialect, row_offset: uint): seq[string] =
    let fh = newFile(path, encoding)
    var obj = newReaderObj(dialect)

    try:
        fh.setFilePos(int64 row_offset, fspSet)

        for (row_idx, fields, field_count) in obj.parse_csv(fh):
            return fields[0..field_count-1]
    finally:
        fh.close()

proc write_numpy_header(fh: File, dtype: string, shape: uint): void =
    const magic = "\x93NUMPY"
    const major = "\x01"
    const minor = "\x00"
    
    let header = "{'descr': '" & dtype & "', 'fortran_order': False, 'shape': (" & $shape & ",)}"
    let header_len = len(header)
    let padding = (64 - ((len(magic) + len(major) + len(minor) + 2 + header_len)) mod 64)
    
    var padding_bytes: array[2, uint8] # how the hell do we write a binary file in nim??! there's no resources, is there really no way other than casting to bytes?

    let padding_header = uint16 (padding + header_len)

    copyMem(padding_bytes[0].unsafeAddr, padding_header.unsafeAddr, padding_bytes.len)

    fh.write(magic)
    fh.write(major)
    fh.write(minor)

    discard fh.writeBytes(padding_bytes, 0, padding_bytes.len)

    fh.write(header)

    for i in 0..padding-2:
        fh.write(" ")
    fh.write("\n")

proc text_reader_task(
    path: string, encoding: Encodings, dialect: Dialect, 
    destinations: var seq[string], field_relation: var OrderedTable[uint, uint], 
    row_offset: uint, row_count: int): void =
    var obj = newReaderObj(dialect)
    let fh = newFile(path, encoding)
    
    let keys_field_relation = collect: (for k in field_relation.keys: k)

    try:
        fh.setFilePos(int64 row_offset, fspSet)

        let page_file_handlers = collect(newSeqOfCap(destinations.len)):
            for p in destinations:
                open(p, fmWrite)

        var longest_str = newSeq[uint](destinations.len)
        var n_rows: uint = 0

        for (row_idx, fields, field_count) in obj.parse_csv(fh):
            if row_count >= 0 and row_idx >= (uint row_count):
                break
                
            for idx in 0..field_count-1:
                if not ((uint idx) in keys_field_relation):
                    continue

                let fidx = field_relation[uint idx]
                let field = fields[idx]

                longest_str[fidx] = max(uint field.runeLen, longest_str[fidx])

            inc n_rows

        for (fh, i) in zip(page_file_handlers, longest_str):
            fh.write_numpy_header("<U" & $i, n_rows)

        fh.setFilePos(int64 row_offset, fspSet)

        var ch_arr {.noinit.}: array[4, uint8]

        for (row_idx, fields, field_count) in obj.parse_csv(fh):
            if row_count >= 0 and row_idx >= (uint row_count):
                break
                
            for idx in 0..field_count-1:
                if not ((uint idx) in keys_field_relation):
                    continue

                var str = fields[idx]
                let fidx = field_relation[uint idx]
                var fh = page_file_handlers[fidx]

                for rune in str.toRunes():
                    var ch = uint32(rune)
                    copyMem(addr ch_arr, ch.unsafeAddr, 4)
                    discard fh.writeBytes(ch_arr, 0, ch_arr.len)

                let dt = longest_str[fidx] - (uint str.runeLen)

                for i in 1..dt:
                    fh.write("\x00\x00\x00\x00")

        for f in page_file_handlers:
            f.close()

    finally:
        fh.close()

proc test_perf(path: string, encoding: Encodings, dialect: Dialect): void =
    

    var obj = newReaderObj(dialect)
    let fh = newFile(path, encoding)

    try:

        let d0 = getTime()

        # var elements = 0

        for (row_idx, fields, field_count) in obj.parse_csv(fh):
            discard
            # echo $fields[0..field_count-1]
            # echo $obj.container.szs
            # for s in obj.iter_fields():
            #     # echo "'" & s[] & "'"
            #     discard
            # var str_ptrs = strings[0..field_count-1]
            # echo $str_ptrs[0][]
            # echo $strings[0..field_count-1][]
            # inc elements

            # if elements mod 1_000_000 == 0:
            #     echo $elements & " | time: " & $(getTime() - d0)

        let d1 = getTime()

        echo $(d1 - d0)
    finally:
        fh.close()

proc import_file(path: string, encoding: Encodings, dia: Dialect, columns: ptr seq[string], execute: bool): void =
    echo "Collecting tasks: '" & path & "'"
    let (newline_offsets, newlines) = find_newlines(path, encoding)

    let dirname = "/media/ratchet/hdd/tablite/nim/page"

    if not dirExists(dirname):
        createDir(dirname)

    if newlines > 0:
        let fields = read_columns(path, encoding, dia, newline_offsets[0])

        var imp_columns: seq[string]

        if columns == nil:
            imp_columns = fields
        else:
            raise newException(Exception, "not implemented error:column selection")

        let new_fields = collect(initOrderedTable()):
            for ix, name in enumerate(fields):
                if name in imp_columns:
                    {uint ix: name}

        let inp_fields = collect(initOrderedTable()):
            for ix, name in new_fields.pairs:
                {ix: name}

        var field_relation = collect(initOrderedTable()):
            for i, c in enumerate(inp_fields.keys):
                {c: uint i}

        var page_idx: uint32 = 1
        var row_idx: uint = 1
        var page_size: uint = 1_000_000

        let path_task = dirname & "/tasks.txt"
        let ft = open(path_task, fmWrite)

        var delimiter = ""
        delimiter.addEscapedChar(dia.delimiter)
        var quotechar = ""
        quotechar.addEscapedChar(dia.quotechar)
        var escapechar = ""
        escapechar.addEscapedChar(dia.escapechar)
        var lineterminator = ""
        lineterminator.addEscapedChar(dia.lineterminator)

        echo "Dumping tasks: '" & path & "'"
        while row_idx < newlines:
            var pages = newSeq[string](fields.len)

            for idx in 0..fields.len - 1:
                pages[idx] = dirname & "/" & $page_idx & ".npy"
                inc page_idx

            ft.write("\"" & getAppFilename() & "\" ")

            case encoding:
                of ENC_UTF8:
                    ft.write("--encoding=" & "UTF8" & " ")
                of ENC_UTF16:
                    ft.write("--encoding" & "UTF16" & " ")

            ft.write("--delimiter=\"" & delimiter & "\" ")
            ft.write("--quotechar=\"" & quotechar & "\" ")
            ft.write("--escapechar=\"" & escapechar & "\" ")
            ft.write("--lineterminator=\"" & lineterminator & "\" ")
            ft.write("--doublequote=" & $dia.doublequote & " ")
            ft.write("--skipinitialspace=" & $dia.skipinitialspace & " ")
            ft.write("--quoting=" & $dia.quoting & " ")

            # ft.write("\"/home/ratchet/Documents/dematic/tablite/build/tablite_csv\" ")
            ft.write("task ")

            ft.write("--pages=\"" & pages.join(",") & "\" ")
            ft.write("--fields_keys=\"" & toSeq(field_relation.keys).join(",") & "\" ")
            ft.write("--fields_vals=\"" & toSeq(field_relation.values).join(",") & "\" ")

            ft.write("\"" & path & "\" ")
            ft.write($newline_offsets[row_idx] & " ")
            ft.write($page_size)

            ft.write("\n")

            # text_reader_task(path, encoding, dia, pages, field_relation, newline_offsets[1], -1)

            row_idx = row_idx + page_size

        ft.close()

        if execute:
            echo "Executing tasks: '" & path & "'"
            let args = @[
                "--progress",
                "-a",
                "\"" & path_task & "\""
            ]

            let para = "/usr/bin/parallel"

            let ret_code = execCmd(para & " " & args.join(" "))

            # let process = startProcess("/usr/bin/parallel", args=args)
            # let ret_code = process.waitForExit()

            if ret_code != 0:
                raise newException(Exception, "Process failed with errcode: " & $ret_code)

proc unescape_seq(str: string): string = # nim has no true unescape
    case str:
        of "\\n": return "\n"
        of "\\t": return "\t"

    return str

if isMainModule:
    var path_csv: string
    var encoding: Encodings
    var dialect: Dialect

    const boolean_true_choices = ["true", "yes", "t", "y"]
    # const boolean_false_choices = ["false", "no", "f", "n"]
    const boolean_choices = ["true", "false", "yes", "no", "t", "f", "y", "n"]

    var p = newParser:
        help("Imports tablite pages")
        option(
            "-e", "--encoding",
            help="file encoding",
            choices = @["UTF8", "UTF16"],
            default=some("UTF8")
        )

        option("--delimiter", help="text delimiter", default=some(","))
        option("--quotechar", help="text quotechar", default=some("\""))
        option("--escapechar", help="text escapechar", default=some("\\"))
        option("--lineterminator", help="text lineterminator", default=some("\\n"))

        option(
            "--doublequote",
            help="text doublequote",
            choices = @boolean_choices,
            default=some("true")
        )

        option(
            "--skipinitialspace",
            help="text skipinitialspace",
            choices = @boolean_choices,
            default=some("false")
        )

        option(
            "--quoting",
            help="text quoting",
            choices = @[
                "QUOTE_MINIMAL",
                "QUOTE_ALL",
                "QUOTE_NONNUMERIC",
                "QUOTE_NONE",
                "QUOTE_STRINGS",
                "QUOTE_NOTNULL"
            ],
            default=some("QUOTE_MINIMAL")
        )

        command("import"):
            arg("path", help="file path")
            arg("execute", help="execute immediatly")
            run:
                discard
                # echo opts.parentOpts.encoding
        command("task"):
            option("--pages", help="task pages", required = true)
            option("--fields_keys", help="field keys", required = true)
            option("--fields_vals", help="field vals", required = true)

            arg("path", help="file path")
            arg("offset", help="file offset")
            arg("count", help="line count")
            run:
                discard
        run:
            var delimiter = opts.delimiter.unescape_seq()
            var quotechar = opts.quotechar.unescape_seq()
            var escapechar = opts.escapechar.unescape_seq()
            var lineterminator = opts.lineterminator.unescape_seq()

            if delimiter.len != 1: raise newException(IOError, "'delimiter' must be 1 character")
            if quotechar.len != 1: raise newException(IOError, "'quotechar' must be 1 character")
            if escapechar.len != 1: raise newException(IOError, "'escapechar' must be 1 character")
            if lineterminator.len != 1: raise newException(IOError, "'lineterminator' must be 1 character")

            dialect = newDialect(
                delimiter = delimiter[0],
                quotechar = quotechar[0],
                escapechar = escapechar[0],
                doublequote = opts.doublequote in boolean_true_choices,
                quoting = (
                    case opts.quoting.toUpper():
                        of "QUOTE_MINIMAL":
                            QUOTE_MINIMAL
                        of "QUOTE_ALL":
                            QUOTE_ALL
                        of "QUOTE_NONNUMERIC":
                            QUOTE_NONNUMERIC
                        of "QUOTE_NONE":
                            QUOTE_NONE
                        of "QUOTE_STRINGS":
                            QUOTE_STRINGS
                        of "QUOTE_NOTNULL":
                            QUOTE_NOTNULL
                        else:
                            raise newException(Exception, "invalid 'quoting'")
                ),
                skipinitialspace = opts.skipinitialspace in boolean_true_choices,
                lineterminator = lineterminator[0],
            )

            case opts.encoding.toUpper():
                of "UTF8": encoding = ENC_UTF8
                of "UTF16": encoding = ENC_UTF16
                else: raise newException(Exception, "invalid 'encoding'")

    let opts = p.parse()
    p.run()

    if opts.import.isNone and opts.task.isNone:
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/bad_empty.csv", ENC_UTF8)
        (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/gdocs1.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/gesaber_data.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_be.csv", ENC_UTF16)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_le.csv", ENC_UTF16)
        # dialect = newDialect()

        let d0 = getTime()
        import_file(path_csv, encoding, dialect, nil, true)
        let d1 = getTime()
        
        echo $(d1 - d0)
    else:
        if opts.import.isSome:
            let execute = opts.import.get.execute in boolean_true_choices
            let path_csv = opts.import.get.path
            echo "Importing: '" & path_csv & "'"
            
            let d0 = getTime()
            import_file(path_csv, encoding, dialect, nil, execute)
            let d1 = getTime()
            
            echo $(d1 - d0)

        if opts.task.isSome:
            let path = opts.task.get.path
            var pages = opts.task.get.pages.split(",")
            let fields_keys = opts.task.get.fields_keys.split(",")
            let fields_vals = opts.task.get.fields_vals.split(",")

            var field_relation = collect(initOrderedTable()):
                for (k, v) in zip(fields_keys, fields_vals):
                    {parseUInt(k): parseUInt(v)}

            let offset = parseUInt(opts.task.get.offset)
            let count = parseInt(opts.task.get.count)

            # text_reader_task(path, encoding, dia, pages, field_relation, offset, count)
            text_reader_task(path, encoding, dialect, pages, field_relation, offset, count)