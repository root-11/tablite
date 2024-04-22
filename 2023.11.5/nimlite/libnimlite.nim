#[
    Put all python export wrappers here.

    There is an issue with nimpy where it is impossible to have exports in other files no matter if you use import/include syntax.
    Therefore, we must unfortunately put all Python accessible API's in this file.

    Alternative would be split the API's to separate dynamic libraries bout that adds memory overhead because each of the libraries would duplicate the binding code.
]#


template isLib(): bool = isMainModule and appType == "lib"


when isLib:
    import nimpy
    import std/[os, options, tables, paths, sugar]
    import pymodules, ranking, dateutils, pytypes

    # --------      NUMPY      --------
    import numpy
    proc read_page(path: string): nimpy.PyObject {.exportpy.} = return readNumpy(path).toPython()
    proc repaginate(column: nimpy.PyObject): void {.exportpy.} =
        let pages = collect: (for p in column.pages: modules().toStr(p.path))
        let newPages = repaginate(pages)

        column.pages = newPages
    # --------      NUMPY      --------

    # -------- COLUMN SELECTOR --------
    import funcs/column_selector as column_selector

    proc collect_column_select_info*(table: nimpy.PyObject, cols: nimpy.PyObject, dir_pid: string, pbar: nimpy.PyObject): nimpy.PyObject {.exportpy.} =
        return column_selector.collectColumnSelectInfo(table, cols, dir_pid, pbar).toPyObj()

    proc do_slice_convert*(dir_pid: string, page_size: int, columns: Table[string, (string, nimpy.PyObject)], reject_reason_name: string, res_pass: column_selector.ColInfo, res_fail: column_selector.ColInfo, desired_column_map: nimpy.PyObject, column_names: seq[string], is_correct_type: Table[string, bool]): (seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)]) {.exportpy.} =
        return column_selector.doSliceConvert(Path(dir_pid), page_size, columns, reject_reason_name, res_pass, res_fail, desired_column_map.fromPyObjToDesiredInfos(), column_names, is_correct_type)
    # -------- COLUMN SELECTOR --------

    # --------   TEXT READER   --------
    import funcs/text_reader as text_reader

    proc collect_text_reader_page_info_task(
        task_info: nimpy.PyObject,
        task: nimpy.PyObject,
    ): (uint, seq[uint], seq[Rank]) {.exportpy.} =
        return text_reader.toTaskArgs(task_info, task).collectPageInfoTask

    proc text_reader_task(
        task_info: nimpy.PyObject,
        task: nimpy.PyObject,
        page_info: (uint, seq[uint], seq[Rank])
    ): seq[nimpy.PyObject] {.exportpy.} =
        return text_reader.toTaskArgs(task_info, task).textReaderTask(page_info)

    proc get_headers(
        path: string, encoding: string,
        newline: string, delimiter: string, text_qualifier: string,
        strip_leading_and_tailing_whitespace: bool,
        quoting: string,
        header_row_index: uint, linecount: int): seq[seq[string]] {.exportpy.} =
        var arg_encoding = str2Enc(encoding)
        var arg_newline = (if newline.len == 1: newline[0] else: raise newException(Exception, "'newline' not a char"))
        var arg_delimiter = (if delimiter.len == 1: delimiter[0] else: raise newException(Exception, "'delimiter' not a char"))
        var arg_text_qualifier = (if text_qualifier.len == 1: text_qualifier[0] else: raise newException(Exception, "'text_qualifier' not a char"))
        var arg_quoting = str2quoting(quoting)

        let headers = getHeaders(
            path = path,
            encoding = arg_encoding,
            headerRowIndex = header_row_index,
            lineCount = linecount,
            newline = arg_newline,
            delimiter = arg_delimiter,
            textQualifier = arg_text_qualifier,
            stripLeadingAndTailingWhitespace = strip_leading_and_tailing_whitespace,
            quoting = arg_quoting
        )

        return headers

    proc text_reader(
        pid: string, path: string, encoding: string,
        columns: nimpy.PyObject, first_row_has_headers: bool, header_row_index: uint,
        start: nimpy.PyObject, limit: nimpy.PyObject,
        guess_datatypes: bool,
        newline: string, delimiter: string, text_qualifier: string,
        strip_leading_and_tailing_whitespace: bool, skip_empty: SkipEmpty,
        page_size: uint,
        quoting: string
    ): text_reader.TabliteTable {.exportpy.} =
        let arg_cols = (if isNone(columns): none[seq[string]]() else: some(columns.to(seq[string])))
        let arg_encoding = str2Enc(encoding)
        let arg_start = (if isNone(start): none[int]() else: some(start.to(int)))
        let arg_limit = (if isNone(limit): none[int]() else: some(limit.to(int)))
        let arg_newline = (if newline.len == 1: newline[0] else: raise newException(Exception, "'newline' not a char"))
        let arg_delimiter = (if delimiter.len == 1: delimiter[0] else: raise newException(Exception, "'delimiter' not a char"))
        let arg_text_qualifier = (if text_qualifier.len == 1: text_qualifier[0] else: raise newException(Exception, "'text_qualifier' not a char"))
        let arg_quoting = str2quoting(quoting)

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
                quoting = arg_quoting,
                skip_empty = skip_empty
        )

        return table
    # --------   TEXT READER   --------

    # --------    FILTER    -----------
    import funcs/filter as ff
    proc filter(table: nimpy.PyObject, expressions: seq[nimpy.PyObject], `type`: string, tqdm: nimpy.PyObject, pbar: nimpy.PyObject): (nimpy.PyObject, nimpy.PyObject) {.exportpy.} =
        return ff.filter(table, expressions, `type`, tqdm, pbar)

    # --------    FILTER    -----------

    # --------  IMPUTATION  -----------
    import funcs/imputation
    proc nearest_neighbour(T: nimpy.PyObject, sources: seq[string], missing: seq[nimpy.PyObject], targets: seq[string], tqdm: nimpy.PyObject, pbar: nimpy.PyObject): nimpy.PyObject {.exportpy.} =
        var miss: seq[PY_ObjectND] = @[]
        for m in missing:
            case modules().builtins.getTypeName(m):
            of "NoneType":
                miss.add(PY_ObjectND(PY_None))
            of "str":
                miss.add(newPY_Object(m.to(string)))
            of "int":
                miss.add(newPY_Object(m.to(int)))
            of "float":
                miss.add(newPY_Object(m.to(float)))
            of "datetime":
                miss.add(newPY_Object(pyDateTime2NimDateTime(m), K_DATETIME))
            of "date":
                miss.add(newPY_Object(pyDate2NimDateTime(m), K_DATE))
            of "time":
                miss.add(newPY_Object(pyTime2NimDuration(m)))
            of "bool":
                miss.add(newPY_Object(m.to(bool)))
            else:
                raise newException(ValueError, "unrecognized type.")
        return nearestNeighbourImputation(T, sources, miss, targets, tqdm, pbar)
    # --------  IMPUTATION  -----------
    
    # --------   GROUPBY  -----------
    import funcs/groupby as gb
    proc groupby(T: nimpy.PyObject, keys: seq[string], functions: seq[(string, string)], tqdm: nimpy.PyObject, pbar: nimpy.PyObject): nimpy.PyObject {. exportpy .} =
        var funcs = collect:
            for (cn, fn) in functions:
                (cn, str2Accumulator(fn))
        return gb.groupby(T, keys, funcs, tqdm, pbar)
    # --------   GROUPBY  -----------