import text_reader/csvparse
import text_reader/encfile
import text_reader/paging
import text_reader/pylayer
import text_reader/taskargs
import text_reader/text_reader
import text_reader/table

export paging
export pylayer
export taskargs
export text_reader
export table
export encfile
export csvparse

when isMainModule and appType != "lib":
    import nimpy
    import std/[paths, osproc, times, enumerate, strutils, options, os]
    import ../[numpy, pymodules, nimpyext]

    proc toTaskArgs*(path: string, encoding: string, dialect: TabliteDialect, task: TabliteTask, import_fields: seq[uint], guess_dtypes: bool, skip_empty: string): TaskArgs {.inline.} =
        return toTaskArgs(
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
            import_fields = import_fields,
            skip_empty = skip_empty
        )

    proc runTask*(taskArgs: TaskArgs, pageInfo: PageInfo): void =
        discard taskArgs.textReaderTask(pageInfo)

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
        pid: string, taskname: string, path: string, encoding: FileEncoding, dialect: Dialect,
        cols: Option[seq[string]], first_row_has_headers: bool, header_row_index: uint,
        page_size: uint, guess_dtypes: bool, skipEmpty: SkipEmpty,
        start: Option[int], limit: Option[int]
    ): PyObject =
        let d0 = getTime()

        let table = importTextFile(pid, path, encoding, dialect, cols, first_row_has_headers, header_row_index, page_size, guess_dtypes, skipEmpty, start, limit)
        let task = table.task

        for i, column_task in enumerate(task.tasks):
            let taskArgs = toTaskArgs(task.path, task.encoding, task.dialect, column_task, task.import_fields, task.guess_dtypes, $skipEmpty)
            let pageInfo = taskArgs.collectPageInfoTask()

            taskArgs.runTask(pageInfo)
            echo "Dumped " & $(i + 1) & "/" & $task.tasks.len

        let d1 = getTime()
        echo $(d1 - d0)
        
        let pyTable = modules().tablite.classes.TableClass!()

        for cInfo in table.columns:
            let pyColumn = modules().tablite.modules.base.classes.ColumnClass!(pid)
            pyTable[cInfo.name] = pyColumn
            
            for page in cInfo.pages:
                let pPage = Path(page)
                
                let workdir = string pPage.parentDir.parentDir
                let id = string pPage.extractFilename.changeFileExt("")
                
                let len = getPageLen(page)
                let dtypes = getPageTypes(page)
                let page = newPyPage(id, workdir, len, dtypes)

                discard pyColumn.pages.append(page)

        return pyTable

    let m = modules()
    let Config = m.tablite.modules.config.classes.Config
    let workdir = Path(m.toStr(Config.workdir))

    var path_csv: string
    var encoding = str2Enc($ENC_UTF8)
    var dialect: Dialect
    var cols = none[seq[string]]()
    var guess_dtypes = true
    var pid = string (workdir / Path("nim"))
    var taskname = "task"
    var page_size = uint Config.PAGE_SIZE.to(int)

    var delimiter = ','
    var quotechar = '"'
    var escapechar = '\\'
    var lineterminator = '\n'
    var doublequote = true
    var quoting = QUOTE_MINIMAL

    var skipinitialspace = false
    var skiptrailingspace = false
    var skipEmpty = SkipEmpty.ALL

    dialect = newDialect(
        delimiter = delimiter,
        quotechar = quotechar,
        escapechar = escapechar,
        doublequote = doublequote,
        quoting = quoting,
        skipinitialspace = skipinitialspace,
        skiptrailingspace = skiptrailingspace,
        lineterminator = lineterminator,
    )

    let dirdata = os.getEnv("DATA_DIR", ".")

    # (path_csv, encoding) = ("tests/data/split_lines.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = (dirdata & "/Dealz Poland v1.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = ("tests/data/floats.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = ("tests/data/bad_empty.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = ("tests/data/book1.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = ("tests/data/detect_misalignment.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = (dirdata & "/Ritual B2B orderlines updated.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = (dirdata & "/Ritual B2B orderlines_small.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = ("tests/data/utf16_test.csv", str2Enc($ENC_UTF16))
    # (path_csv, encoding) = ("tests/data/win1250_test.csv", str2ConvEnc("Windows-1252"))

    # (path_csv, encoding) = ("tests/data/book1.txt", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = ("tests/data/gdocs1.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = ("tests/data/with_empty_lines.csv", str2Enc($ENC_UTF8))
    (path_csv, encoding) = ("tests/data/with_escape.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = (dirdata & "/Dematic YDC Order Data.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = (dirdata & "/Dematic YDC Order Data_1M.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = (dirdata & "/Dematic YDC Order Data_1M_1col.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = (dirdata & "/gesaber_data.csv", str2Enc($ENC_UTF8))
    # (path_csv, encoding) = ("tests/data/utf16_be.csv", str2Enc($ENC_UTF16))
    # (path_csv, encoding) = ("tests/data/utf16_le.csv", str2Enc($ENC_UTF16))

    # cols = some(@["\"Item\"", "\"Materiál\"", "\"Objem\"", "\"Jednotka objemu\"", "\"Free Inv Pcs\""])
    # dialect.quoting = Quoting.QUOTE_NONE
    # dialect.delimiter = ';'

    let start = some[int](0)
    let limit = some[int](-1)
    let first_row_has_headers = true
    let header_row_index = uint 0

    guess_dtypes = true
    # cols = some(@["a", "c"])
    # page_size = 2

    let pyTable = importFile(pid, taskname, path_csv, encoding, dialect, cols, first_row_has_headers, header_row_index, page_size, guess_dtypes, skipEmpty, start, limit)

    discard pyTable.show(dtype=true)
