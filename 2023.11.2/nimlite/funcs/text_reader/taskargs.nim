import nimpy
import csvparse, encfile
import ../../utils

type TaskArgs* = object
    path*: string
    encoding*: FileEncoding
    dialect*: Dialect
    guessDtypes*: bool
    destinations*: seq[string]
    importFields*: seq[uint]
    rowOffset*: uint
    rowCount*: int
    skipEmpty*: SkipEmpty

proc toTaskArgs*(
    path: string,
    encoding: string,
    diaDelimiter: string,
    diaQuotechar: string,
    diaEscapechar: string,
    diaDoublequote: bool,
    diaQuoting: string,
    diaSkipInitialSpace: bool,
    diaSkipTrailingSpace: bool,
    diaLineTerminator: string,
    diaStrict: bool,
    guessDtypes: bool,
    tskPages: seq[string],
    tskOffset: uint,
    tskCount: uint,
    importFields: seq[uint],
    skipEmpty: string
): TaskArgs =
    var delimiter = diaDelimiter.unescapeSeq()
    var quotechar = diaQuotechar.unescapeSeq()
    var escapechar = diaEscapechar.unescapeSeq()
    var lineterminator = diaLineTerminator.unescapeSeq()

    if delimiter.len != 1: raise newException(IOError, "'delimiter' must be 1 character")
    if quotechar.len != 1: raise newException(IOError, "'quotechar' must be 1 character")
    if escapechar.len != 1: raise newException(IOError, "'escapechar' must be 1 character")
    if lineterminator.len != 1: raise newException(IOError, "'lineterminator' must be 1 character")

    var dialect = newDialect(
        delimiter = delimiter[0],
        quotechar = quotechar[0],
        escapechar = escapechar[0],
        doublequote = diaDoublequote,
        quoting = str2quoting(diaQuoting),
        skipinitialspace = diaSkipInitialSpace,
        skiptrailingspace = diaSkipTrailingSpace,
        lineterminator = lineterminator[0],
    )

    var arg_encoding = str2Enc(encoding)

    return TaskArgs(
        path: path,
        encoding: arg_encoding,
        dialect: dialect,
        guessDtypes: guessDtypes,
        destinations: tskPages,
        importFields: importFields,
        rowOffset: tskOffset,
        rowCount: int tskCount,
        skipEmpty: str2SkipEmpty(skipEmpty)
    )

proc toTaskArgs*(
        task_info: PyObject,
        task: PyObject,
    ): TaskArgs =
        return toTaskArgs(
            path = task_info["path"].to(string),
            encoding = task_info["encoding"].to(string),
            diaDelimiter = task_info["dialect"]["delimiter"].to(string),
            diaQuotechar = task_info["dialect"]["quotechar"].to(string),
            diaEscapechar = task_info["dialect"]["escapechar"].to(string),
            diaDoublequote = task_info["dialect"]["doublequote"].to(bool),
            diaQuoting = task_info["dialect"]["quoting"].to(string),
            diaSkipInitialSpace = task_info["dialect"]["skipinitialspace"].to(bool),
            diaSkipTrailingSpace = task_info["dialect"]["skiptrailingspace"].to(bool),
            diaLineTerminator = task_info["dialect"]["lineterminator"].to(string),
            diaStrict = task_info["dialect"]["strict"].to(bool),
            guessDtypes = task_info["guess_dtypes"].to(bool),
            tskPages = task["pages"].to(seq[string]),
            tskOffset = task["offset"].to(uint),
            tskCount = task["count"].to(uint),
            importFields = task_info["import_fields"].to(seq[uint]),
            skipEmpty = task_info["skip_empty"].to(string)
        )