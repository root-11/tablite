import csvparse

type TabliteColumn* = object
    name*: string
    pages*: seq[string]

type TabliteTask* = object
    pages*: seq[string]
    offset*: uint

type TabliteDialect* = object
    delimiter*: string
    quotechar*: string
    escapechar*: string
    doublequote*: bool
    quoting*: string
    skipinitialspace*: bool
    lineterminator*: string
    strict*: bool

type TabliteTasks* = object
    path*: string
    encoding*: string
    dialect*: TabliteDialect
    tasks*: seq[TabliteTask]
    import_fields*: seq[uint]
    page_size*: uint
    guess_dtypes*: bool

type TabliteTable* = object
    task*: TabliteTasks
    columns*: seq[TabliteColumn]

proc newTabliteDialect*(dialect: Dialect): TabliteDialect =
    var delimiter = ""
    delimiter.addEscapedChar(dialect.delimiter)
    var quotechar = ""
    quotechar.addEscapedChar(dialect.quotechar)
    var escapechar = ""
    escapechar.addEscapedChar(dialect.escapechar)
    var lineterminator = ""
    lineterminator.addEscapedChar(dialect.lineterminator)

    TabliteDialect(
        delimiter: delimiter,
        quotechar: quotechar,
        escapechar: escapechar,
        doublequote: dialect.doublequote,
        quoting: $dialect.quoting,
        skipinitialspace: dialect.skipinitialspace,
        lineterminator: lineterminator,
        strict: dialect.strict,
    )


proc newTabliteTask*(pages: seq[string], offset: uint): TabliteTask =
    TabliteTask(pages: pages, offset: offset)

proc newTabliteTasks*(
    path: string, encoding: string, dialect: Dialect,
    tasks: seq[TabliteTask], import_fields: seq[uint], page_size: uint, guess_dtypes: bool): TabliteTasks =
    TabliteTasks(
        path: path,
        encoding: encoding,
        dialect: dialect.newTabliteDialect,
        tasks: tasks,
        import_fields: import_fields,
        page_size: page_size,
        guess_dtypes: guess_dtypes
    )

proc newTabliteColumn*(name: string, pages: seq[string]): TabliteColumn =
    TabliteColumn(name: name, pages: pages)

proc newTabliteTable*(task: TabliteTasks, columns: seq[TabliteColumn]): TabliteTable =
    TabliteTable(task: task, columns: columns)
