import std/[times, osproc, enumerate, options, strutils]
import taskargs, table, encfile, csvparse, textreader

proc runTask*(path: string, encoding: string, dialect: TabliteDialect, task: TabliteTask, import_fields: seq[uint], guess_dtypes: bool): void =
    toTaskArgs(
        path = path,
        encoding = encoding,
        dia_delimiter = dialect.delimiter,
        dia_quotechar = dialect.quotechar,
        dia_escapechar = dialect.escapechar,
        dia_doublequote = dialect.doublequote,
        dia_quoting = dialect.quoting,
        dia_skipinitialspace = dialect.skipinitialspace,
        dia_skiptrailingspace = dialect.skiptrailingspace,
        dia_lineterminator = dialect.lineterminator,
        dia_strict = dialect.strict,
        guess_dtypes = guess_dtypes,
        tsk_pages = task.pages,
        tsk_offset = task.offset,
        tsk_count = task.count,
        import_fields = import_fields
    ).textReaderTask

proc executeParallel*(path: string): void =
    echo "Executing tasks: '" & path & "'"
    let args = @[
        "--progress",
        "-a",
        "\"" & path & "\""
    ]

    let para = "/usr/bin/parallel"
    let ret_code = execCmd(para & " " & args.join(" "))

    if ret_code != 0:
        raise newException(Exception, "Process failed with errcode: " & $ret_code)

proc importFile*(
    pid: string, taskname: string, path: string, encoding: Encodings, dialect: Dialect,
    cols: Option[seq[string]], first_row_has_headers: bool, header_row_index: uint,
    page_size: uint, guess_dtypes: bool,
    start: Option[int], limit: Option[int],
    multiprocess: bool, execute: bool, use_json: bool
): void =
    if use_json and execute:
        raise newException(IOError, "Cannot use 'execute' with combination of 'use_json'")

    let d0 = getTime()

    let table = importTextFile(pid, path, encoding, dialect, cols, first_row_has_headers, header_row_index, page_size, guess_dtypes, start, limit)
    let task = table.task

    if multiprocess:
        if use_json:
            discard table.saveTable(pid, taskname)
        else:
            let task_path = task.saveTasks(pid, taskname)

            discard table.saveTable(pid, taskname)

            if execute:
                executeParallel(task_path)
    else:
        if execute:
            for i, column_task in enumerate(task.tasks):
                runTask(task.path, task.encoding, task.dialect, column_task, task.import_fields, task.guess_dtypes)
                echo "Dumped " & $(i+1) & "/" & $task.tasks.len

    let d1 = getTime()
    echo $(d1 - d0)
