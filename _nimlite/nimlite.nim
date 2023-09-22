import std/[options, strutils]
import nimpy, nimpy/py_lib
import encfile, table, csvparse, textreader, utils

let
    builtins = pyBuiltinsModule()
    PyNoneClass = builtins.None.getattr("__class__")

proc is_none(py: PyObject): bool =
    return builtins.isinstance(py, PyNoneClass).to(bool)

proc textReader(
    pid: string, path: string, encoding: Encodings,
    columns: Option[seq[string]], first_row_has_headers: bool, header_row_index: uint,
    start: Option[uint], limit: Option[uint],
    guess_datatypes: bool,
    newline: char, delimiter: char,
    text_qualifier: char, strip_leading_and_tailing_whitespace: bool,
    page_size: uint,
    guess_dtypes: bool
): TabliteTable =
    var dialect = newDialect(
        delimiter = delimiter,
        quotechar = text_qualifier,
        escapechar = '\\',
        doublequote = true,
        quoting = QUOTE_MINIMAL,
        skipinitialspace = strip_leading_and_tailing_whitespace,
        lineterminator = newline,
    )

    return importTextFile(
        pid=pid,
        path=path,
        encoding=encoding,
        dia=dialect,
        columns=columns,
        page_size=page_size,
        guess_dtypes=guess_dtypes
    )

proc text_reader_task(
    path: string,
    encoding: string,
    dia_delimiter: string,
    dia_quotechar: string,
    dia_escapechar: string,
    dia_doublequote: bool,
    dia_quoting: string,
    dia_skipinitialspace: bool,
    dia_lineterminator: string,
    dia_strict: bool,
    guess_dtypes: bool,
    tsk_pages: seq[string],
    tsk_offset: uint,
    import_fields: seq[uint],
    count: int,
): void {.exportpy.} =

    var delimiter = dia_delimiter.unescapeSeq()
    var quotechar = dia_quotechar.unescapeSeq()
    var escapechar = dia_escapechar.unescapeSeq()
    var lineterminator = dia_lineterminator.unescapeSeq()

    if delimiter.len != 1: raise newException(IOError, "'delimiter' must be 1 character")
    if quotechar.len != 1: raise newException(IOError, "'quotechar' must be 1 character")
    if escapechar.len != 1: raise newException(IOError, "'escapechar' must be 1 character")
    if lineterminator.len != 1: raise newException(IOError, "'lineterminator' must be 1 character")

    var dialect = newDialect(
        delimiter = delimiter[0],
        quotechar = quotechar[0],
        escapechar = escapechar[0],
        doublequote = dia_doublequote,
        quoting = (
            case dia_quoting.toUpper():
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
        skipinitialspace = dia_skipinitialspace,
        lineterminator = lineterminator[0],
    )

    var arg_encoding: Encodings

    case encoding.toUpper():
        of "ENC_UTF8": arg_encoding = ENC_UTF8
        of "ENC_UTF16": arg_encoding = ENC_UTF16
        else: raise newException(Exception, "invalid 'encoding'")

    var arg_pages = tsk_pages
    var arg_import_fields = import_fields.unsafeAddr

    textReaderTask(
        path=path,
        encoding=arg_encoding,
        dialect=dialect,
        guess_dtypes=guess_dtypes,
        destinations=arg_pages,
        import_fields=arg_import_fields,
        row_offset=tsk_offset,
        row_count=count
    )

proc text_reader(
    pid: string, path: string, encoding: string,
    columns: PyObject, first_row_has_headers: bool, header_row_index: uint,
    start: PyObject, limit: PyObject,
    guess_datatypes: bool,
    newline: string, delimiter: string, text_qualifier: string,
    strip_leading_and_tailing_whitespace: bool,
    page_size: uint,
    guess_dtypes: bool
): TabliteTable {.exportpy.} =
    var arg_cols = (if is_none(columns): none[seq[string]]() else: some(columns.to(seq[string])))
    var arg_encoding: Encodings

    case encoding:
        of "ENC_UTF8": arg_encoding = ENC_UTF8
        of "ENC_UTF16": arg_encoding = ENC_UTF16
        else: raise newException(IOError, "invalid encoding: " & encoding)

    var arg_start = (if is_none(start): none[uint]() else: some(start.to(uint)))
    var arg_limit = (if is_none(limit): none[uint]() else: some(limit.to(uint)))
    var arg_newline = (if newline.len == 1: newline[0] else: raise newException(Exception, "'newline' not a char"))
    var arg_delimiter = (if delimiter.len == 1: delimiter[0] else: raise newException(Exception, "'delimiter' not a char"))
    var arg_text_qualifier = (if text_qualifier.len == 1: text_qualifier[0] else: raise newException(Exception, "'text_qualifier' not a char"))

    return textReader(
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
            page_size=page_size,
            guess_dtypes=guess_dtypes
    )

if isMainModule:
    echo "Nimlite imported"

when isMainModule and appType != "lib":
    echo "not lib"

# import std/[os, enumerate, sugar, times, tables, sequtils, json, unicode, osproc, options]
# import argparse
# import encfile, csvparse, table, utils, textreader

# if isMainModule:
#     var path_csv: string
#     var encoding: Encodings
#     var dialect: Dialect
#     var cols = none[seq[string]]()

#     const boolean_true_choices = ["true", "yes", "t", "y"]
#     # const boolean_false_choices = ["false", "no", "f", "n"]
#     const boolean_choices = ["true", "false", "yes", "no", "t", "f", "y", "n"]

#     var p = newParser:
#         help("Imports tablite pages")
#         option(
#             "-e", "--encoding",
#             help="file encoding",
#             choices = @["UTF8", "UTF16"],
#             default=some("UTF8")
#         )

#         option("--delimiter", help="text delimiter", default=some(","))
#         option("--quotechar", help="text quotechar", default=some("\""))
#         option("--escapechar", help="text escapechar", default=some("\\"))
#         option("--lineterminator", help="text lineterminator", default=some("\\n"))

#         option(
#             "--doublequote",
#             help="text doublequote",
#             choices = @boolean_choices,
#             default=some("true")
#         )

#         option(
#             "--skipinitialspace",
#             help="text skipinitialspace",
#             choices = @boolean_choices,
#             default=some("false")
#         )

#         option(
#             "--quoting",
#             help="text quoting",
#             choices = @[
#                 "QUOTE_MINIMAL",
#                 "QUOTE_ALL",
#                 "QUOTE_NONNUMERIC",
#                 "QUOTE_NONE",
#                 "QUOTE_STRINGS",
#                 "QUOTE_NOTNULL"
#             ],
#             default=some("QUOTE_MINIMAL")
#         )

#         command("import"):
#             arg("path", help="file path")
#             arg("execute", help="execute immediatly")
#             arg("multiprocess", help="use multiprocessing")
#             run:
#                 discard
#         command("task"):
#             option("--pages", help="task pages", required = true)
#             option("--fields_keys", help="field keys", required = true)
#             option("--fields_vals", help="field vals", required = true)

#             arg("path", help="file path")
#             arg("offset", help="file offset")
#             arg("count", help="line count")
#             run:
#                 discard
#         run:
#             var delimiter = opts.delimiter.unescapeSeq()
#             var quotechar = opts.quotechar.unescapeSeq()
#             var escapechar = opts.escapechar.unescapeSeq()
#             var lineterminator = opts.lineterminator.unescapeSeq()

#             if delimiter.len != 1: raise newException(IOError, "'delimiter' must be 1 character")
#             if quotechar.len != 1: raise newException(IOError, "'quotechar' must be 1 character")
#             if escapechar.len != 1: raise newException(IOError, "'escapechar' must be 1 character")
#             if lineterminator.len != 1: raise newException(IOError, "'lineterminator' must be 1 character")

#             dialect = newDialect(
#                 delimiter = delimiter[0],
#                 quotechar = quotechar[0],
#                 escapechar = escapechar[0],
#                 doublequote = opts.doublequote in boolean_true_choices,
#                 quoting = (
#                     case opts.quoting.toUpper():
#                         of "QUOTE_MINIMAL":
#                             QUOTE_MINIMAL
#                         of "QUOTE_ALL":
#                             QUOTE_ALL
#                         of "QUOTE_NONNUMERIC":
#                             QUOTE_NONNUMERIC
#                         of "QUOTE_NONE":
#                             QUOTE_NONE
#                         of "QUOTE_STRINGS":
#                             QUOTE_STRINGS
#                         of "QUOTE_NOTNULL":
#                             QUOTE_NOTNULL
#                         else:
#                             raise newException(Exception, "invalid 'quoting'")
#                 ),
#                 skipinitialspace = opts.skipinitialspace in boolean_true_choices,
#                 lineterminator = lineterminator[0],
#             )

#             case opts.encoding.toUpper():
#                 of "UTF8": encoding = ENC_UTF8
#                 of "UTF16": encoding = ENC_UTF16
#                 else: raise newException(Exception, "invalid 'encoding'")

#     let opts = p.parse()
#     p.run()

#     if opts.import.isNone and opts.task.isNone:
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/bad_empty.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/gdocs1.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/Dematic YDC Order Data_1M_1col.csv", ENC_UTF8)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/callisto/tests/testing/data/gesaber_data.csv", ENC_UTF8)
#         (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_be.csv", ENC_UTF16)
#         # (path_csv, encoding) = ("/home/ratchet/Documents/dematic/tablite/tests/data/utf16_le.csv", ENC_UTF16)

#         let d0 = getTime()
#         # cols = some(@["B", "A", "B"])
#         echo $importTextFile(path_csv, encoding, dialect, cols, true, multiprocess=false)
#         let d1 = getTime()

#         echo $(d1 - d0)
#     else:
#         if opts.import.isSome:
#             let execute = opts.import.get.execute in boolean_true_choices
#             let multiprocess = opts.import.get.multiprocess in boolean_true_choices
#             let path_csv = opts.import.get.path
#             echo "Importing: '" & path_csv & "'"

#             let d0 = getTime()
#             discard importTextFile(path_csv, encoding, dialect, cols, execute, multiprocess)
#             let d1 = getTime()

#             echo $(d1 - d0)

#         if opts.task.isSome:
#             let path = opts.task.get.path
#             var pages = opts.task.get.pages.split(",")
#             let fields_keys = opts.task.get.fields_keys.split(",")
#             let fields_vals = opts.task.get.fields_vals.split(",")

#             var field_relation = collect(initOrderedTable()):
#                 for (k, v) in zip(fields_keys, fields_vals):
#                     {parseUInt(k): v}
#             let import_fields = collect: (for k in field_relation.keys: k)

#             let offset = parseUInt(opts.task.get.offset)
#             let count = parseInt(opts.task.get.count)

#             textReaderTask(path, encoding, dialect, pages, import_fields.unsafeAddr, offset, count)
