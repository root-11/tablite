import std/[os, options, tables]
import encfile, table, csvparse

when isMainModule and appType == "lib":
    import std/[paths]
    import nimpy
    import pylayer, textreader, taskargs, pymodules
    
    include includes/column_selector

    
    proc text_reader_task(
        path: string,
        encoding: string,
        dia_delimiter: string,
        dia_quotechar: string,
        dia_escapechar: string,
        dia_doublequote: bool,
        dia_quoting: string,
        dia_skipinitialspace: bool,
        dia_skiptrailingspace: bool,
        dia_lineterminator: string,
        dia_strict: bool,
        guess_dtypes: bool,
        tsk_pages: seq[string],
        tsk_offset: uint,
        tsk_count: int,
        import_fields: seq[uint]
    ): void {.exportpy.} =
        try:
            toTaskArgs(
                path = path,
                encoding = encoding,
                dia_delimiter = dia_delimiter,
                dia_quotechar = dia_quotechar,
                dia_escapechar = dia_escapechar,
                dia_doublequote = dia_doublequote,
                dia_quoting = dia_quoting,
                dia_skipinitialspace = dia_skipinitialspace,
                dia_skiptrailingspace = dia_skiptrailingspace,
                dia_lineterminator = dia_lineterminator,
                dia_strict = dia_strict,
                guess_dtypes = guess_dtypes,
                tsk_pages = tsk_pages,
                tsk_offset = tsk_offset,
                tsk_count = uint tsk_count,
                import_fields = import_fields
            ).textReaderTask
        except Exception as e:
            echo $e.msg & "\n" & $e.getStackTrace
            raise e

    proc text_reader(
        pid: string, path: string, encoding: string,
        columns: PyObject, first_row_has_headers: bool, header_row_index: uint,
        start: PyObject, limit: PyObject,
        guess_datatypes: bool,
        newline: string, delimiter: string, text_qualifier: string,
        strip_leading_and_tailing_whitespace: bool,
        page_size: uint,
        quoting: string
    ): TabliteTable {.exportpy.} =
        try:
            var arg_cols = (if isNone(columns): none[seq[string]]() else: some(columns.to(seq[string])))
            var arg_encoding = str2Enc(encoding)
            var arg_start = (if isNone(start): none[int]() else: some(start.to(int)))
            var arg_limit = (if isNone(limit): none[int]() else: some(limit.to(int)))
            var arg_newline = (if newline.len == 1: newline[0] else: raise newException(Exception, "'newline' not a char"))
            var arg_delimiter = (if delimiter.len == 1: delimiter[0] else: raise newException(Exception, "'delimiter' not a char"))
            var arg_text_qualifier = (if text_qualifier.len == 1: text_qualifier[0] else: raise newException(Exception, "'text_qualifier' not a char"))
            var arg_quoting = str2quoting(quoting)

            let table = textReader(
                    pid = pid,
                    path = path,
                    encoding = arg_encoding,
                    columns = arg_cols,
                    first_row_has_headers = first_row_has_headers,
                    header_row_index = header_row_index,
                    start = arg_start,
                    limit = arg_limit,
                    guess_datatypes = guess_datatypes,
                    newline = arg_newline,
                    delimiter = arg_delimiter,
                    text_qualifier = arg_text_qualifier,
                    strip_leading_and_tailing_whitespace = strip_leading_and_tailing_whitespace,
                    page_size = page_size,
                    quoting = arg_quoting
            )

            return table
        except Exception as e:
            echo $e.msg & "\n" & $e.getStackTrace
            raise e

when isMainModule and appType != "lib":
    import argparse
    import std/[sugar, json]
    import utils, cli

    var path_csv: string
    var encoding: FileEncoding
    var dialect: Dialect
    var cols = none[seq[string]]()
    var guess_dtypes: bool
    var pid = "/media/ratchet/hdd/tablite/nim"
    var taskname = "task"
    var use_json = false
    var page_size = uint 1_000_000

    const boolean_true_choices = ["true", "yes", "t", "y"]
    # const boolean_false_choices = ["false", "no", "f", "n"]
    const boolean_choices = ["true", "false", "yes", "no", "t", "f", "y", "n"]

    var p = newParser:
        help("Imports tablite pages")
        option(
            "-e", "--encoding",
            help = "file encoding",
            default = some($ENC_UTF8)
        )

        option(
            "-dt", "--guess_dtypes",
            help = "gues datatypes",
            choices = @boolean_choices,
            default = some("true")
        )

        option("--delimiter", help = "text delimiter", default = some(","))
        option("--quotechar", help = "text quotechar", default = some("\""))
        option("--escapechar", help = "text escapechar", default = some("\\"))
        option("--lineterminator", help = "text lineterminator", default = some("\\n"))

        option(
            "--doublequote",
            help = "text doublequote",
            choices = @boolean_choices,
            default = some("true")
        )

        option(
            "--skipinitialspace",
            help = "text skipinitialspace",
            choices = @boolean_choices,
            default = some("false")
        )

        option(
            "--skiptrailingspace",
            help = "text skiptrailingspace",
            choices = @boolean_choices,
            default = some("false")
        )

        option(
            "--quoting",
            help = "text quoting",
            choices = @[
                $QUOTE_MINIMAL,
                $QUOTE_NONE,
            ],
            default = some($QUOTE_MINIMAL)
        )

        command("import"):
            arg("path", help = "file path")

            option("--pid", help = "result pid")
            option("--name", help = "table name")
            option("--page_size", help = "page size")
            option("--columns", help = "columns")

            option(
                "-e",
                "--execute",
                help = "execute immediatly",
                choices = @boolean_choices,
                default = some("true")
            )

            option(
                "-mp",
                "--multiprocess",
                help = "use multiprocessing",
                choices = @boolean_choices,
                default = some("true")
            )

            option(
                "--use_json",
                help = "save task as json",
                choices = @boolean_choices
            )

            option("--start", help = "row offset", default = some("0"))
            option("--limit", help = "row count to read", default = some("-1"))

            option("--first_row_has_headers", help = "file has headers", default = some("true"))
            option("--header_row_index", help = "header offset", default = some("0"))
            run:
                path_csv = opts.path

                if opts.name_opt.isSome:
                    taskname = opts.name

                if opts.page_size_opt.isSome:
                    page_size = uint parseInt(opts.page_size)

                if opts.columns_opt.isSome:
                    cols = some(parseJson(opts.columns).to(seq[string]))

                if opts.use_json_opt.isSome:
                    use_json = opts.use_json in boolean_true_choices
        command("task"):
            option("--pages", help = "task pages", required = true)
            option("--fields", help = "fields to import", required = true)

            arg("path", help = "file path")
            arg("offset", help = "file offset")
            arg("count", help = "line count")
            run:
                path_csv = opts.path
        run:
            var delimiter = opts.delimiter.unescapeSeq()
            var quotechar = opts.quotechar.unescapeSeq()
            var escapechar = opts.escapechar.unescapeSeq()
            var lineterminator = opts.lineterminator.unescapeSeq()

            if delimiter.len != 1: raise newException(IOError, "'delimiter' must be 1 character: '" & delimiter & "'")
            if quotechar.len != 1: raise newException(IOError, "'quotechar' must be 1 character: '" & quotechar & "'")
            if escapechar.len != 1: raise newException(IOError, "'escapechar' must be 1 character: '" & escapechar & "'")
            if lineterminator.len != 1: raise newException(IOError, "'lineterminator' must be 1 character: '" & lineterminator & "'")

            dialect = newDialect(
                delimiter = delimiter[0],
                quotechar = quotechar[0],
                escapechar = escapechar[0],
                doublequote = opts.doublequote in boolean_true_choices,
                quoting = str2quoting(opts.quoting),
                skipinitialspace = opts.skipinitialspace in boolean_true_choices,
                skiptrailingspace = opts.skiptrailingspace in boolean_true_choices,
                lineterminator = lineterminator[0],
            )

            encoding = str2Enc(opts.encoding)

            guess_dtypes = opts.guess_dtypes in boolean_true_choices

    let opts = p.parse()

    p.run()

    # let dirname = getCurrentDir()

    if opts.import.isNone and opts.task.isNone:
        when defined(DEV_BUILD):
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/split_lines.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dealz Poland v1.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/floats.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/bad_empty.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/book1.csv", str2Enc($ENC_UTF8))
            (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/detect_misalignment.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Ritual B2B orderlines updated.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Ritual B2B orderlines_small.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_test.csv", str2Enc($ENC_UTF16))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/win1250_test.csv", str2ConvEnc("Windows-1252"))

            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/book1.txt", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/gdocs1.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M_1col.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/gesaber_data.csv", str2Enc($ENC_UTF8))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_be.csv", str2Enc($ENC_UTF16))
            # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_le.csv", str2Enc($ENC_UTF16))

            # cols = some(@["\"Item\"", "\"Materiál\"", "\"Objem\"", "\"Jednotka objemu\"", "\"Free Inv Pcs\""])
            # dialect.quoting = Quoting.QUOTE_NONE
            # dialect.delimiter = ';'

            let multiprocess = false
            let execute = true
            let start = some[int](0)
            let limit = some[int](-1)
            let first_row_has_headers = true
            let header_row_index = uint 0

            guess_dtypes = true
            # cols = some(@["a", "c"])
            # page_size = 2

            echo "Running test version with no arguments"
            importFile(pid, taskname, path_csv, encoding, dialect, cols, first_row_has_headers, header_row_index, page_size, guess_dtypes, start, limit, multiprocess, execute, use_json)
        else:
            raise newException(Exception, "Must provide 'import' or 'task'")
    if opts.import.isSome and opts.task.isSome:
        raise newException(Exception, "cannot do this")
    
    else:
        if opts.import.isSome:
            let multiprocess = opts.import.get.multiprocess in boolean_true_choices
            let execute = opts.import.get.execute in boolean_true_choices
            let start = some(parseInt(opts.import.get.start))
            let limit = some(parseInt(opts.import.get.limit))
            let first_row_has_headers = opts.import.get.first_row_has_headers in boolean_true_choices
            let header_row_index = uint parseInt(opts.import.get.header_row_index)

            if opts.import.get.pid_opt.isSome:
                pid = opts.import.get.pid

            importFile(pid, taskname, path_csv, encoding, dialect, cols, first_row_has_headers, header_row_index, page_size, guess_dtypes, start, limit, multiprocess, execute, use_json)
        elif opts.task.isSome:
            let tdia = newTabliteDialect(dialect)
            let count = parseInt(opts.task.get.count)
            let ttask = newTabliteTask(opts.task.get.pages.split(","), uint parseInt(opts.task.get.offset), uint count)
            let fields = collect: (for k in opts.task.get.fields.split(","): uint parseInt(k))

            echo path_csv, " ", $encoding, " ", tdia, " ", ttask, " ", fields, " ", guess_dtypes

            runTask(path_csv, $encoding, tdia, ttask, fields, guess_dtypes)