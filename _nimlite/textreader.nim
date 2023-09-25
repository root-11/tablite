import std/[os, enumerate, sugar, tables, json, options, strutils]
import encfile, csvparse, table, utils, paging, taskargs

proc textReaderTask*(task: TaskArgs): void =
    var dialect = task.dialect
    var encoding = task.encoding
    var destinations = task.destinations
    var path = task.path
    var guess_dtypes = task.guess_dtypes
    var row_count = task.row_count
    var row_offset = task.row_offset
    var import_fields = task.import_fields.unsafeAddr

    var obj = newReaderObj(dialect)
    
    let fh = newFile(path, encoding)
    let n_pages = destinations.len

    try:
        fh.setFilePos(int64 row_offset, fspSet)

        var (n_rows, longest_str, ranks) = collectPageInfo(
            obj=obj.unsafeAddr,
            fh=fh.unsafeAddr,
            guess_dtypes=guess_dtypes,
            n_pages=n_pages,
            row_count=row_count,
            import_fields=import_fields
        )

        var (page_file_handlers, column_dtypes, binput) = dumpPageHeader(
            destinations=destinations,
            n_pages=n_pages,
            n_rows=n_rows,
            guess_dtypes=guess_dtypes,
            longest_str=longest_str,
            ranks=ranks,
        )

        try:
            fh.setFilePos(int64 row_offset, fspSet)

            dumpPageBody(
                obj=obj.unsafeAddr,
                fh=fh.unsafeAddr,
                guess_dtypes=guess_dtypes,
                n_pages=n_pages,
                row_count=row_count,
                import_fields=import_fields,
                page_file_handlers=page_file_handlers,
                longest_str=longest_str,
                ranks=ranks,
                column_dtypes=column_dtypes,
                binput=binput
            )

            dumpPageFooter(
                n_pages=n_pages,
                n_rows=n_rows,
                page_file_handlers=page_file_handlers,
                column_dtypes=column_dtypes,
                binput=binput
            )
        finally:
            for f in page_file_handlers:
                f.close()

    finally:
        fh.close()

proc importTextFile*(
    pid: string, path: string, encoding: Encodings, dia: Dialect, 
    columns: Option[seq[string]],
    page_size: uint, guess_dtypes: bool): TabliteTable =
    echo "Collecting tasks: '" & path & "'"
    let (newline_offsets, newlines) = findNewlines(path, encoding)

    let dirname = pid & "/pages"

    if not dirExists(dirname):
        createDir(dirname)

    if newlines > 0:
        let fields = readColumns(path, encoding, dia, newline_offsets[0])

        var imp_columns {.noinit.}: seq[string]

        if columns.isSome:
            var missing = newSeq[string]()
            for column in columns.get:
                if not (column in fields):
                    missing.add("'" & column & "'")
            if missing.len > 0:
                raise newException(IOError, "Missing columns: [" & missing.join(", ") & "]")
            imp_columns = columns.get
        else:
            imp_columns = fields

        var field_relation = collect(initOrderedTable()):
            for ix, name in enumerate(fields):
                if name in imp_columns:
                    {uint ix: name}
        let import_fields = collect: (for k in field_relation.keys: k)

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

        var page_idx: uint32 = 1
        var row_idx: uint = 1
        var task_list = newSeq[TabliteTask]()

        echo "Dumping tasks: '" & path & "'"
        while row_idx < newlines:
            let page_count = field_relation.len
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

            task_list.add(newTabliteTask(pages, newline_offsets[row_idx]))
            # textReaderTask(path, encoding, dia, pages, import_fields.unsafeAddr, newline_offsets[row_idx], int page_size)
            
            row_idx = row_idx + page_size


        let tasks = newTabliteTasks(
            path=path,
            encoding= $encoding,
            dialect=dia,
            tasks=task_list,
            import_fields=import_fields,
            page_size=page_size,
            guess_dtypes=guess_dtypes
        )
        let columns = collect(newSeqOfCap(table_columns.len)):
            for (column_name, page_index) in table_columns.pairs:
                newTabliteColumn(column_name, page_list[page_index])

        let table = newTabliteTable(tasks, columns)

        return table
    else:
        raise newException(IOError, "end of file")