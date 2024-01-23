import nimpy as nimpy
import std/[os, enumerate, sugar, tables, json, options, strutils, paths]
import encfile, csvparse, table, ../../utils, paging, taskargs
from ../../numpy import newPyPage

proc textReaderTask*(task: TaskArgs): seq[nimpy.PyObject] =
    var dialect = task.dialect
    var encoding = task.encoding
    var destinations = task.destinations
    var path = task.path
    var guess_dtypes = task.guess_dtypes
    var row_count = task.row_count
    var row_offset = task.row_offset
    var import_fields = task.import_fields
    var obj = newReaderObj(dialect)

    var fh = newFile(path, encoding)
    let n_pages = destinations.len

    try:
        fh.setFilePos(int64 row_offset, fspSet)

        var (n_rows, longest_str, ranks) = collectPageInfo(
            obj = obj,
            fh = fh,
            guess_dtypes = guess_dtypes,
            n_pages = n_pages,
            row_count = row_count,
            import_fields = import_fields
        )

        var (page_file_handlers, column_dtypes, binput) = dumpPageHeader(
            destinations = destinations,
            n_pages = n_pages,
            n_rows = n_rows,
            guess_dtypes = guess_dtypes,
            longest_str = longest_str,
            ranks = ranks,
        )

        try:
            fh.setFilePos(int64 row_offset, fspSet)

            let (pgTypes, pgLens) = dumpPageBody(
                obj = obj,
                fh = fh,
                guess_dtypes = guess_dtypes,
                n_pages = n_pages,
                row_count = row_count,
                import_fields = import_fields,
                page_file_handlers = page_file_handlers,
                longest_str = longest_str,
                ranks = ranks,
                column_dtypes = column_dtypes,
                binput = binput
            )

            dumpPageFooter(
                n_pages = n_pages,
                n_rows = n_rows,
                page_file_handlers = page_file_handlers,
                column_dtypes = column_dtypes,
                binput = binput
            )

            let elems = collect:
                for i in 0..<n_pages:
                    let path = Path(destinations[i])
                    let workdir = string path.parentDir.parentDir
                    let id = parseInt(string path.extractFilename.changeFileExt(""))
                    let dtypes = pgTypes[i]
                    let len = pgLens[i]

                    newPyPage(id, workdir, len, dtypes)

            return elems
        finally:
            for f in page_file_handlers:
                f.close()

    finally:
        fh.close()

proc importTextFile*(
    pid: string, path: string, encoding: FileEncoding, dia: Dialect,
    columns: Option[seq[string]], first_row_has_headers: bool, header_row_index: uint,
    page_size: uint, guess_dtypes: bool,
    start: Option[int] = none[int](), limit: Option[int] = none[int]()
): TabliteTable =

    echo "Collecting tasks: '" & path & "'"

    let opt_start = (if start.isSome: start.get else: 0)
    let opt_limit = (if limit.isSome: limit.get else: -1)
    let (newline_offsets, newlines) = findNewlines(path, encoding, dia)
    let dirname = pid & "/pages"

    if not dirExists(dirname):
        createDir(dirname)

    if newlines > 0 and newlines > header_row_index:
        let first_line = readColumns(path, encoding, dia, newline_offsets[header_row_index])

        var fields = newSeq[string](0)

        if first_row_has_headers:
            fields = first_line
        else:
            fields = collect(newSeqOfCap(first_line.len)):
                for i in 0..first_line.len-1:
                    $i

        var imp_columns = newSeq[string](0)

        if columns.isSome:
            var missing = newSeq[string]()
            for column in columns.get:
                if not (column in fields):
                    missing.add("'" & column & "'")
            if missing.len > 0:
                let field_list = collect(newSeqOfCap(fields.len)):
                    for f in fields:
                        "'" & f & "'"
                raise newException(IOError, "Missing columns: [" & missing.join(", ") & "]" & " | Available columns: (" & $field_list.len & ")[" & field_list.join(", ") & "]")
            imp_columns = columns.get
        else:
            imp_columns = fields

        var field_relation = collect(initOrderedTable()):
            for ix, name in enumerate(fields):
                if name in imp_columns:
                    {uint ix: name}

        let import_fields = collect: (for k in field_relation.keys: k)
        let import_field_names = collect: (for v in field_relation.values: v)

        var field_relation_inv = collect(initOrderedTable()):
            for (ix, name) in field_relation.pairs:
                {name: ix}

        var page_list = collect(initOrderedTable()):
            for (ix, name) in field_relation.pairs:
                {ix: newSeq[string]()}

        var name_list = newSeq[string]()
        var table_columns = collect(initOrderedTable()):
            for name in imp_columns:
                let unq = uniqueName(name, name_list)

                name_list.add(unq)

                {unq: field_relation_inv[name]}

        let offset_row = (if first_row_has_headers: 1 else: 0) + int header_row_index

        var page_idx: uint32 = 1
        var row_idx: uint = uint opt_start + offset_row
        var task_list = newSeq[TabliteTask]()
        let max_line = (if opt_limit >= 0: min(newlines, uint (opt_limit + opt_start) + offset_row) else: newlines)

        echo "Dumping tasks: '" & path & "'"
        while row_idx < max_line:
            let page_count = field_relation.len
            let next_line = min(row_idx + page_size, max_line)
            let row_count = next_line - row_idx
            var pages = newSeq[string](page_count)

            for idx in 0..page_count - 1:
                var pagepath = dirname & "/" & $page_idx & ".npy"

                if not pid.endsWith("tablite/nim"):
                    while fileExists(pagepath):
                        inc page_idx
                        pagepath = dirname & "/" & $page_idx & ".npy"

                let field_idx = import_fields[idx]

                page_list[field_idx].add(pagepath)
                pages[idx] = pagepath

                inc page_idx

            task_list.add(newTabliteTask(pages, newline_offsets[row_idx], row_count))

            row_idx = next_line

        let tasks = newTabliteTasks(
            path = path,
            encoding = $encoding,
            dialect = dia,
            tasks = task_list,
            import_fields = import_fields,
            import_field_names = import_field_names,
            page_size = page_size,
            guess_dtypes = guess_dtypes
        )
        let columns = collect(newSeqOfCap(table_columns.len)):
            for (column_name, page_index) in table_columns.pairs:
                newTabliteColumn(column_name, page_list[page_index])

        let table = newTabliteTable(tasks, columns)

        return table
    else:
        raise newException(IOError, "end of file")