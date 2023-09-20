import argparse
import std/enumerate
import os, math, sugar, times, tables, sequtils, json, unicode, parseutils, encodings, bitops, osproc, lists, endians

type DataTypes = enum
    DT_DATETIME, DT_DATE, DT_TIME,
    DT_INT, DT_BOOLEAN, DT_FLOAT,
    DT_STRING,
    DT_MAX_ELEMENTS

include encfile
include csvparse
include pickling
include infertypes
include numpy

type Rank = array[int(DataTypes.DT_MAX_ELEMENTS), (DataTypes, uint)]

iterator iter(rank: var Rank): ptr (DataTypes, uint) {.closure.} =
    var x = 0
    let max = int(DataTypes.DT_MAX_ELEMENTS)
    while x < max:
        yield rank[x].unsafeAddr
        inc x

proc newRank(): Rank =
    var ranks {.noinit.}: Rank

    for i in 0..(int(DataTypes.DT_MAX_ELEMENTS)-1):
        ranks[i] = (DataTypes(i), uint 0)

    return ranks


proc `<` (a: (DataTypes, uint), b: (DataTypes, uint)): bool = a[1] < b[1]
proc `>` (a: (DataTypes, uint), b: (DataTypes, uint)): bool = a[1] > b[1]

proc insert_sort[T](a: var openarray[T]) =
    # our array is likely to be nearly sorted or already sorted, therefore the complexity is better than bubble sort
    for i in 1 .. a.high:
        let value = a[i]
        var j = i
        while j > 0 and value > a[j-1]:
            a[j] = a[j-1]
            dec j
        a[j] = value

proc update_rank(rank: var Rank, str: ptr string): (bool, DataTypes) =
    var rank_dtype: DataTypes
    var index: int
    var rank_count: uint
    var is_none: bool = false

    for i, r_addr in enumerate(rank.iter()):
        try:
            case r_addr[0]:
                of DataTypes.DT_INT:
                    discard str.inferInt()
                of DataTypes.DT_FLOAT:
                    discard str.inferFloat()
                of DataTypes.DT_BOOLEAN:
                    discard str.inferBool()
                of DataTypes.DT_DATE:
                    discard str.inferDate()
                of DataTypes.DT_TIME:
                    discard str.inferTime()
                of DataTypes.DT_DATETIME:
                    discard str.inferDatetime()
                of DataTypes.DT_STRING:
                    if str[] in ["null", "Null", "NULL", "#N/A", "#n/a", "", "None"]:
                        is_none = true
                else:
                    raise newException(Exception, "invalid type")
        except ValueError as e:
            echo "rank failed: '" & $str[] & "' -> " & e.msg & "\n---\n" & e.getStackTrace() & "--------"
            continue

        rank_dtype = r_addr[0]
        rank_count = r_addr[1]
        index = i

        echo "selected " & $r_addr[0] & " for '" & $str[] & "'"

        break

    if is_none:
        return (true, rank_dtype)

    rank[index] = (rank_dtype, rank_count + 1)
    rank.insert_sort()

    return (false, rank_dtype)



proc text_reader_task(
    path: string, encoding: Encodings, dialect: Dialect, 
    destinations: var seq[string], field_relation: var OrderedTable[uint, uint], 
    row_offset: uint, row_count: int): void =
    var obj = newReaderObj(dialect)
    
    let fh = newFile(path, encoding)
    let keys_field_relation = collect: (for k in field_relation.keys: k)
    let guess_dtypes = true
    let n_pages = destinations.len
    
    var ranks: seq[Rank]
    
    if guess_dtypes:
        ranks = collect(newSeqOfCap(n_pages)):
            for _ in 0..n_pages-1:
                newRank()

    try:
        fh.setFilePos(int64 row_offset, fspSet)

        let page_file_handlers = collect(newSeqOfCap(n_pages)):
            for p in destinations:
                open(p, fmWrite)

        var longest_str = newSeq[uint](n_pages)
        var column_dtypes = newSeq[char](n_pages)
        var column_nones = newSeq[bool](n_pages)
        var n_rows: uint = 0
        var binput: uint32 = 0

        for (row_idx, fields, field_count) in obj.parseCSV(fh):
            if row_count >= 0 and row_idx >= (uint row_count):
                break
                
            for idx in 0..field_count-1:
                if not ((uint idx) in keys_field_relation):
                    continue

                let fidx = field_relation[uint idx]
                let field = fields[idx]

                if not guess_dtypes:
                    longest_str[fidx] = max(uint field.runeLen, longest_str[fidx])
                else:
                    let rank = addr ranks[fidx]
                    let (is_none, dt) = rank[].update_rank(field.unsafeAddr)

                    if dt == DataTypes.DT_STRING and not is_none:
                        longest_str[fidx] = max(uint field.runeLen, longest_str[fidx])

                    if is_none:
                        column_nones[fidx] = true

            inc n_rows

        echo $longest_str

        if not guess_dtypes:
            for idx, (fh, i) in enumerate(zip(page_file_handlers, longest_str)):
                column_dtypes[idx] = 'U'
                fh.writeNumpyHeader("<U" & $i, n_rows)
        else:
            for i in 0..n_pages-1:
                let fh = page_file_handlers[i]
                let rank = addr ranks[i]
                var dtype = column_dtypes[i]
                var nilish = column_nones[i]

                for it in rank[].iter():
                    let dt = it[0]
                    let count = it[1]
    
                    if count == 0:
                        break

                    if dtype == '\x00':
                        case dt:
                            of DataTypes.DT_INT: dtype = 'i'
                            of DataTypes.DT_FLOAT: dtype = 'f'
                            of DataTypes.DT_STRING: dtype = 'U'
                            of DataTypes.DT_BOOLEAN: dtype ='?'
                            else: dtype = 'O'
                        continue

                    if dtype == 'f' and dt == DataTypes.DT_INT: discard
                    elif dtype == 'i' and dt == DataTypes.DT_FLOAT: dtype = 'f'
                    else: dtype = 'O'
                
                if nilish:
                    fh.writeNumpyHeader("|O", n_rows)
                else:
                    case dtype:
                        of 'U': fh.writeNumpyHeader("<U" & $ longest_str[i], n_rows)
                        of 'i': fh.writeNumpyHeader("<i8", n_rows)
                        of 'f': fh.writeNumpyHeader("<f8", n_rows)
                        of '?': fh.writeNumpyHeader("|b1", n_rows)
                        of 'O': fh.writeNumpyHeader("|O", n_rows)
                        else: raise newException(Exception, "invalid")

                column_dtypes[i] = dtype

            for idx in 0..n_pages-1:
                let fh = page_file_handlers[idx].unsafeAddr
                let dt = column_dtypes[idx]
                let nilish = column_nones[idx]
                if dt == 'O' or nilish:
                    fh.writePickleStart(binput, n_rows)


        fh.setFilePos(int64 row_offset, fspSet)

        for (row_idx, fields, field_count) in obj.parseCSV(fh):
            if row_count >= 0 and row_idx >= (uint row_count):
                break
                
            for idx in 0..field_count-1:
                if not ((uint idx) in keys_field_relation):
                    continue

                var str = fields[idx]
                let fidx = field_relation[uint idx]
                var fh = page_file_handlers[fidx].unsafeAddr

                if not guess_dtypes:
                    for rune in str.toRunes():
                        var ch = uint32(rune)
                        discard fh[].writeBuffer(ch.unsafeAddr, 4)

                    let dt = longest_str[fidx] - (uint str.runeLen)

                    for i in 1..dt:
                        fh[].write("\x00\x00\x00\x00")
                else:
                    let dt = column_dtypes[idx]
                    let nilish = column_nones[idx]
                    var rank = ranks[idx]

                    case dt:
                        of 'U':
                            if not nilish:
                                for rune in str.toRunes():
                                    var ch = uint32(rune)
                                    discard fh[].writeBuffer(ch.unsafeAddr, 4)

                                let dt = longest_str[fidx] - (uint str.runeLen)

                                for i in 1..dt:
                                    fh[].write("\x00\x00\x00\x00")
                            else:
                                if str in ["null", "Null", "NULL", "#N/A", "#n/a", "", "None"]:
                                    fh.writePicklePyObj(PY_None, binput)
                                else:
                                    fh.writePicklePyObj(str, binput)
                        of 'i':
                            if not nilish:
                                let parsed = parseInt(str)
                                discard fh[].writeBuffer(parsed.unsafeAddr, 8)
                            else:
                                try:
                                    fh.writePicklePyObj(parseInt(str), binput)
                                except ValueError as e:
                                    echo "dump failed: '" & $str & "' -> " & e.msg & "\n---\n" & e.getStackTrace() & "--------"
                                    fh.writePicklePyObj(PY_None, binput)
                        of 'f':
                            if not nilish:
                                let parsed = parseFloat(str)
                                discard fh[].writeBuffer(parsed.unsafeAddr, 8)
                            else:
                                try:
                                    fh.writePicklePyObj(parseFloat(str), binput)
                                except ValueError as e:
                                    echo "dump failed: '" & $str & "' -> " & e.msg & "\n---\n" & e.getStackTrace() & "--------"
                                    fh.writePicklePyObj(PY_None, binput)
                        of '?': fh[].write((if str.toLower() == "true": '\x01' else: '\x00'))
                        of 'O': 
                            for r_addr in rank.iter():
                                let dt = r_addr[0]
                                try:
                                    case dt:
                                        of DataTypes.DT_INT:
                                            fh.writePicklePyObj(str.unsafeAddr.inferInt(), binput)
                                        of DataTypes.DT_FLOAT:
                                            fh.writePicklePyObj(str.unsafeAddr.inferFloat(), binput)
                                        of DataTypes.DT_BOOLEAN:
                                            fh.writePicklePyObj(str.unsafeAddr.inferBool(), binput)
                                        of DataTypes.DT_DATE:
                                            fh.writePicklePyObj(str.unsafeAddr.inferDate(), binput)
                                        of DataTypes.DT_TIME:
                                            fh.writePicklePyObj(str.unsafeAddr.inferTime(), binput)
                                        of DataTypes.DT_DATETIME:
                                            fh.writePicklePyObj(str.unsafeAddr.inferDatetime(), binput)
                                        of DataTypes.DT_STRING:
                                            if str in ["null", "Null", "NULL", "#N/A", "#n/a", "", "None"]:
                                                fh.writePicklePyObj(PY_None, binput)
                                            else:
                                                fh.writePicklePyObj(str, binput)
                                        else:
                                            raise newException(Exception, "invalid type")
                                except ValueError as e:
                                    echo "dump failed: '" & $str & "' -> " & e.msg & "\n---\n" & e.getStackTrace() & "--------"
                                    continue
                                break
                        else: raise newException(Exception, "invalid")

        for idx in 0..n_pages-1:
            let fh = page_file_handlers[idx].unsafeAddr
            let dt = column_dtypes[idx]
            let nilish = column_nones[idx]
            if dt == 'O' or nilish:
                fh.writePickleFinish(binput, n_rows)

        for f in page_file_handlers:
            f.close()

    finally:
        fh.close()

proc import_file(path: string, encoding: Encodings, dia: Dialect, columns: ptr seq[string], execute: bool, multiprocess: bool): void =
    echo "Collecting tasks: '" & path & "'"
    let (newline_offsets, newlines) = findNewlines(path, encoding)

    let dirname = "/media/ratchet/hdd/tablite/nim/page"

    if not dirExists(dirname):
        createDir(dirname)

    if newlines > 0:
        let fields = readColumns(path, encoding, dia, newline_offsets[0])

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

            if not multiprocess:
                text_reader_task(path, encoding, dia, pages, field_relation, newline_offsets[row_idx], int page_size)

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

            ft.write("task ")

            ft.write("--pages=\"" & pages.join(",") & "\" ")
            ft.write("--fields_keys=\"" & toSeq(field_relation.keys).join(",") & "\" ")
            ft.write("--fields_vals=\"" & toSeq(field_relation.values).join(",") & "\" ")

            ft.write("\"" & path & "\" ")
            ft.write($newline_offsets[row_idx] & " ")
            ft.write($page_size)

            ft.write("\n")

            row_idx = row_idx + page_size

        ft.close()

        if multiprocess and execute:
            echo "Executing tasks: '" & path & "'"
            let args = @[
                "--progress",
                "-a",
                "\"" & path_task & "\""
            ]

            let para = "/usr/bin/parallel"

            let ret_code = execCmd(para & " " & args.join(" "))

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
            arg("multiprocess", help="use multiprocessing")
            run:
                discard
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
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/gdocs1.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data.csv", ENC_UTF8)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/gesaber_data.csv", ENC_UTF8)
        (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_be.csv", ENC_UTF16)
        # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_le.csv", ENC_UTF16)

        let d0 = getTime()
        import_file(path_csv, encoding, dialect, nil, true, false)
        let d1 = getTime()
        
        echo $(d1 - d0)
    else:
        if opts.import.isSome:
            let execute = opts.import.get.execute in boolean_true_choices
            let multiprocess = opts.import.get.multiprocess in boolean_true_choices
            let path_csv = opts.import.get.path
            echo "Importing: '" & path_csv & "'"
            
            let d0 = getTime()
            import_file(path_csv, encoding, dialect, nil, execute, multiprocess)
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

            text_reader_task(path, encoding, dialect, pages, field_relation, offset, count)