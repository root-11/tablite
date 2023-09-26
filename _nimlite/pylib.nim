from encfile import strToEnc

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

    toTaskArgs(
        path=path,
        encoding=encoding,
        dia_delimiter=dia_delimiter,
        dia_quotechar=dia_quotechar,
        dia_escapechar=dia_escapechar,
        dia_doublequote=dia_doublequote,
        dia_quoting=dia_quoting,
        dia_skipinitialspace=dia_skipinitialspace,
        dia_lineterminator=dia_lineterminator,
        dia_strict=dia_strict,
        guess_dtypes=guess_dtypes,
        tsk_pages=tsk_pages,
        tsk_offset=tsk_offset,
        import_fields=import_fields,
        count=count
    ).textReaderTask

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
    var arg_cols = (if isNone(columns): none[seq[string]]() else: some(columns.to(seq[string])))
    var arg_encoding: str2Enc(encoding)

    var arg_start = (if isNone(start): none[uint]() else: some(start.to(uint)))
    var arg_limit = (if isNone(limit): none[uint]() else: some(limit.to(uint)))
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