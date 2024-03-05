import std/options
import encfile, table, csvparse, text_reader

template makeDialect(): Dialect =
    newDialect(
        delimiter = delimiter,
        quotechar = text_qualifier,
        escapechar = '\\',
        doublequote = true,
        quoting = quoting,
        skipinitialspace = stripLeadingAndTailingWhitespace,
        skiptrailingspace = stripLeadingAndTailingWhitespace,
        lineterminator = newline,
    )

proc textReader*(
    pid: string, path: string, encoding: FileEncoding,
    columns: Option[seq[string]], firstRowHasHeaders: bool, headerRowIndex: uint,
    start: Option[int], limit: Option[int],
    guessDatatypes: bool,
    newline: char, delimiter: char,
    textQualifier: char, stripLeadingAndTailingWhitespace: bool, skipEmpty: bool,
    pageSize: uint,
    quoting: Quoting
): TabliteTable =
    var dialect = makeDialect()

    return importTextFile(
        pid = pid,
        path = path,
        encoding = encoding,
        dia = dialect,
        columns = columns,
        firstRowHasHeaders = firstRowHasHeaders,
        headerRowIndex = headerRowIndex,
        pageSize = pageSize,
        guessDtypes = guessDatatypes,
        skipEmpty = skipEmpty,
        start = start,
        limit = limit
    )

proc getHeaders*(
    path: string, encoding: FileEncoding,
    headerRowIndex: uint, lineCount: int,
    newline: char, delimiter: char,
    textQualifier: char, stripLeadingAndTailingWhitespace: bool,
    quoting: Quoting
): seq[seq[string]] =
    let dialect = makeDialect()

    return getHeaders(path, encoding, dialect, headerRowIndex, lineCount)