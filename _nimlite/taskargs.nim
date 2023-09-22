import std/strutils
import csvparse, encfile, utils

type TaskArgs* = object
    path*: string
    encoding*: Encodings
    dialect*: Dialect
    guess_dtypes*: bool
    destinations*: seq[string]
    import_fields*: seq[uint]
    row_offset*: uint
    row_count*: int

proc toTaskArgs*(
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
): TaskArgs =
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


    return TaskArgs(
        path: path,
        encoding: arg_encoding,
        dialect: dialect,
        guess_dtypes: guess_dtypes,
        destinations: tsk_pages,
        import_fields: import_fields,
        row_offset: tsk_offset,
        row_count: count
    )