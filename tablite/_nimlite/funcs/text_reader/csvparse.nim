import std/strutils
import encfile

# const NOT_SET = uint32.high
const EOL = uint32.high - 1
const field_limit: uint = 128 * 1024

type Quoting* {.pure.} = enum
    QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONNUMERIC, QUOTE_NONE,
    QUOTE_STRINGS, QUOTE_NOTNULL

type ParserState {.pure.} = enum
    START_RECORD, START_FIELD, ESCAPED_CHAR, IN_FIELD,
    IN_QUOTED_FIELD, ESCAPE_IN_QUOTED_FIELD, QUOTE_IN_QUOTED_FIELD,
    EAT_CRNL, AFTER_ESCAPED_CRNL

type Dialect* = object
    delimiter*: char
    quotechar*: char
    escapechar*: char
    doublequote*: bool
    quoting*: Quoting
    skipinitialspace*: bool
    skiptrailingspace*: bool
    lineterminator*: char
    strict*: bool


type ReaderObj* = object
    numeric_field: bool
    line_num: uint
    dialect: Dialect

    field_len: uint
    field_size: uint
    field: seq[char]

    fields: seq[string]
    field_count: uint

proc newDialect*(delimiter: char = ',', quotechar: char = '"', escapechar: char = '\\', doublequote: bool = true, quoting: Quoting = QUOTE_MINIMAL, skipinitialspace: bool = false, skiptrailingspace: bool = false, lineterminator: char = '\n'): Dialect =
    Dialect(delimiter: delimiter, quotechar: quotechar, escapechar: escapechar, doublequote: doublequote, quoting: quoting, skipinitialspace: skipinitialspace, skiptrailingspace: skiptrailingspace, lineterminator: lineterminator)

proc newReaderObj*(dialect: Dialect): ReaderObj =
    ReaderObj(dialect: dialect, fields: newSeq[string](1024))

proc parseGrowBuff(self: var ReaderObj): bool =
    let field_size_new: uint = (if self.field_size > 0: 2u * self.field_size else: 4096u)

    self.field.setLen(field_size_new)
    self.field_size = field_size_new

    return true

proc parseAddChar(self: var ReaderObj, state: var ParserState, c: char): bool =
    if self.field_len >= field_limit:
        return false

    if unlikely(self.field_len == self.field_size and not self.parseGrowBuff()):
        return false

    self.field[self.field_len] = c
    inc self.field_len

    return true

proc parseSaveField(self: var ReaderObj, dia: Dialect): bool =
    if self.numeric_field:
        self.numeric_field = false

        raise newException(Exception, "not yet implemented: parseSaveField numeric_field")

    var field = newString(self.field_len)

    if likely(self.field_len > 0):
        copyMem(field[0].unsafeAddr, self.field[0].unsafeAddr, self.field_len)

    if unlikely(self.field_count + 1 >= (uint self.field.high)):
        self.field.setLen(self.field.len() * 2)

    if dia.skiptrailingspace:
        field = field.strip(leading = false, trailing = true)

    if dia.quoting != Quoting.QUOTE_NONE:
        field = field.multiReplace(("\n", "\\n"), ("\t", "\\t"))

    self.fields[self.field_count] = field

    inc self.field_count

    self.field_len = 0

    return true

proc parseProcessChar(self: var ReaderObj, state: var ParserState, cc: uint32): bool =
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
                if unlikely(not self.parseSaveField(dia)):
                    return false

                state = (if ch_code == EOL: START_RECORD else: EAT_CRNL)
            elif unlikely(c == dia.quotechar and dia.quoting != QUOTE_NONE):
                state = IN_QUOTED_FIELD
            elif unlikely(c == dia.escapechar):
                state = ESCAPED_CHAR
            elif unlikely(c == ' ' and dia.skipinitialspace):
                discard
            elif unlikely(c == dia.delimiter):
                if unlikely(not self.parseSaveField(dia)):
                    return false
            else:
                if dia.quoting == QUOTE_NONNUMERIC:
                    self.numeric_field = true
                if unlikely(not self.parseAddChar(state, c)):
                    return false
                state = IN_FIELD
        of ESCAPED_CHAR:
            if c in ['\n', '\r']:
                if unlikely(not self.parseAddChar(state, c)):
                    return false
                state = AFTER_ESCAPED_CRNL

            if ch_code == EOL:
                c = '\n'
                ch_code = uint32 c

            if unlikely(not self.parseAddChar(state, c)):
                return false

            state = IN_FIELD
        of AFTER_ESCAPED_CRNL, IN_FIELD:
            if state == AFTER_ESCAPED_CRNL and ch_code == EOL:
                return true # nim is stupid

            if unlikely(c in ['\n', '\r'] or unlikely(ch_code == EOL)):
                if unlikely(not self.parseSaveField(dia)):
                    return false
                state = (if ch_code == EOL: START_RECORD else: EAT_CRNL)
            elif c == dia.escapechar:
                state = ESCAPED_CHAR
            elif c == dia.delimiter:
                if unlikely(not self.parseSaveField(dia)):
                    return false
                state = START_FIELD
            else:
                if unlikely(not self.parseAddChar(state, c)):
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
                if unlikely(not self.parseAddChar(state, c)):
                    return false
        of ESCAPE_IN_QUOTED_FIELD:
            if ch_code == EOL:
                c = '\n'
                ch_code = uint32 c

            if unlikely(not self.parseAddChar(state, c)):
                return false

            state = IN_QUOTED_FIELD
        of QUOTE_IN_QUOTED_FIELD:
            if dia.quoting != QUOTE_NONE and c == dia.quotechar:
                if unlikely(not self.parseAddChar(state, c)):
                    return false
                state = IN_QUOTED_FIELD
            elif c == dia.delimiter:
                if unlikely(not self.parseSaveField(dia)):
                    return false
                state = START_FIELD
            elif c in ['\n', '\r'] or ch_code == EOL:
                if unlikely(not self.parseSaveField(dia)):
                    return false
                state = (if ch_code == EOL: START_RECORD else: EAT_CRNL)
            elif not dia.strict:
                if unlikely(not self.parseAddChar(state, c)):
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

iterator parseCSV*(self: var ReaderObj, fh: BaseEncodedFile): (uint, ptr seq[string], uint) =
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
            if self.field_len != 0 and state == IN_QUOTED_FIELD:
                if dia.strict:
                    raise newException(Exception, "unexpected end of data")
                elif self.parseSaveField(dia):
                    break
            raise newException(IOError, "malformed")

        line.add('\n')

        linelen = uint line.len
        pos = 0

        while pos < linelen:
            if unlikely(not self.parseProcessChar(state, uint32 line[pos])):
                raise newException(Exception, "illegal")

            inc pos

        if unlikely(not self.parseProcessChar(state, EOL)):
            raise newException(Exception, "illegal")

        if state == START_RECORD:
            yield (line_num, addr self.fields, self.field_count)

            self.field_count = 0

            inc line_num

proc readColumns*(path: string, encoding: FileEncoding, dialect: Dialect, row_offset: uint): seq[string] =
    let fh = newFile(path, encoding)
    var obj = newReaderObj(dialect)

    try:
        fh.setFilePos(int64 row_offset, fspSet)

        for (row_idx, fields, field_count) in obj.parseCSV(fh):
            return fields[0..field_count-1]
    finally:
        fh.close()

iterator parseCSV*(self: var ReaderObj, path: string, encoding: FileEncoding): (uint, ptr seq[string], uint) =
    var fh = newFile(path, encoding)

    try:
        for it in self.parseCSV(fh):
            yield it
    finally:
        fh.close()

proc str2quoting*(quoting: string): Quoting =
    case quoting.toUpper():
        of $QUOTE_MINIMAL:
            return QUOTE_MINIMAL
        of $QUOTE_ALL:
            return QUOTE_ALL
        of $QUOTE_NONNUMERIC:
            return QUOTE_NONNUMERIC
        of $QUOTE_NONE:
            return QUOTE_NONE
        of $QUOTE_STRINGS:
            return QUOTE_STRINGS
        of $QUOTE_NOTNULL:
            return QUOTE_NOTNULL
        else:
            raise newException(Exception, "invalid quoting: " & quoting)

proc findNewlinesNoQualifier*(fh: BaseEncodedFile): (seq[uint], uint) =
    var newline_offsets = newSeq[uint](1)
    var total_lines: uint = 0
    var str: string

    newline_offsets[0] = fh.getFilePos()

    while likely(fh.readLine(str)):
        inc total_lines

        newline_offsets.add(fh.getFilePos())

    return (newline_offsets, total_lines)

proc findNewlinesQualifier*(fh: BaseEncodedFile, dia: Dialect): (seq[uint], uint) =
    var newline_offsets = newSeq[uint](1)
    var total_lines: uint = 0
    var obj = newReaderObj(dia)

    newline_offsets[0] = fh.getFilePos()

    for (row_idx, fields, field_count) in obj.parseCSV(fh):
        inc total_lines

        newline_offsets.add(fh.getFilePos())

    return (newline_offsets, total_lines)

proc findNewlines*(fh: BaseEncodedFile, dia: Dialect): (seq[uint], uint) {.inline.} =
    if dia.quoting == Quoting.QUOTE_NONE:
        return fh.findNewlinesNoQualifier()

    return fh.findNewlinesQualifier(dia)

proc findNewlines*(path: string, encoding: FileEncoding, dia: Dialect): (seq[uint], uint) {.inline.} =
    let fh = newFile(path, encoding)
    try:
        return fh.findNewlines(dia)
    finally:
        fh.close()
