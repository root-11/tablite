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
    tsk_count: uint,
    import_fields: seq[uint]
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
        quoting = str2quoting(dia_quoting),
        skipinitialspace = dia_skipinitialspace,
        lineterminator = lineterminator[0],
    )

    var arg_encoding = str2Enc(encoding)

    return TaskArgs(
        path: path,
        encoding: arg_encoding,
        dialect: dialect,
        guess_dtypes: guess_dtypes,
        destinations: tsk_pages,
        import_fields: import_fields,
        row_offset: tsk_offset,
        row_count: int tsk_count
    )