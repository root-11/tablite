import nimpy
import std/options
import encfile, table, csvparse, textreader

let
    builtins = pyBuiltinsModule()
    PyNoneClass = builtins.None.getattr("__class__")

proc isNone*(py: PyObject): bool =
    return builtins.isinstance(py, PyNoneClass).to(bool)

proc textReader*(
    pid: string, path: string, encoding: Encodings,
    columns: Option[seq[string]], first_row_has_headers: bool, header_row_index: uint,
    start: Option[int], limit: Option[int],
    guess_datatypes: bool,
    newline: char, delimiter: char,
    text_qualifier: char, strip_leading_and_tailing_whitespace: bool,
    page_size: uint,
    guess_dtypes: bool,
    quoting: Quoting
): TabliteTable =
    var dialect = newDialect(
        delimiter = delimiter,
        quotechar = text_qualifier,
        escapechar = '\\',
        doublequote = true,
        quoting = quoting,
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
        guess_dtypes=guess_dtypes,
        start=start,
        limit=limit
    )

