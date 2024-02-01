import nimpy
import std/options
import encfile, table, csvparse, text_reader

proc textReader*(
    pid: string, path: string, encoding: FileEncoding,
    columns: Option[seq[string]], firstRowHasHeaders: bool, headerRowIndex: uint,
    start: Option[int], limit: Option[int],
    guessDatatypes: bool,
    newline: char, delimiter: char,
    textQualifier: char, stripLeadingAndTailingWhitespace: bool,
    pageSize: uint,
    quoting: Quoting
): TabliteTable =
    var dialect = newDialect(
        delimiter = delimiter,
        quotechar = text_qualifier,
        escapechar = '\\',
        doublequote = true,
        quoting = quoting,
        skipinitialspace = stripLeadingAndTailingWhitespace,
        skiptrailingspace = stripLeadingAndTailingWhitespace,
        lineterminator = newline,
    )

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
        start = start,
        limit = limit
    )

