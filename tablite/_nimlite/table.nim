import std/[json]
import csvparse

type TabliteColumn* = object
    name*: string
    pages*: seq[string]

type TabliteTask* = object
    pages*: seq[string]
    offset*: uint
    count*: uint

type TabliteDialect* = object
    delimiter*: string
    quotechar*: string
    escapechar*: string
    doublequote*: bool
    quoting*: string
    skipinitialspace*: bool
    skiptrailingspace*: bool
    lineterminator*: string
    strict*: bool

type TabliteTasks* = object
    path*: string
    encoding*: string
    dialect*: TabliteDialect
    tasks*: seq[TabliteTask]
    import_fields*: seq[uint]
    import_field_names*: seq[string]
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
        skiptrailingspace: dialect.skiptrailingspace,
        lineterminator: lineterminator,
        strict: dialect.strict,
    )


proc newTabliteTask*(pages: seq[string], offset: uint, count: uint): TabliteTask =
    TabliteTask(pages: pages, offset: offset, count: count)

proc newTabliteTasks*(
    path: string, encoding: string, dialect: Dialect,
    tasks: seq[TabliteTask], import_fields: seq[uint], import_field_names: seq[string], page_size: uint, guess_dtypes: bool): TabliteTasks =
    TabliteTasks(
        path: path,
        encoding: encoding,
        dialect: dialect.newTabliteDialect,
        tasks: tasks,
        import_fields: import_fields,
        import_field_names: import_field_names,
        page_size: page_size,
        guess_dtypes: guess_dtypes
    )

proc newTabliteColumn*(name: string, pages: seq[string]): TabliteColumn =
    TabliteColumn(name: name, pages: pages)

proc newTabliteTable*(task: TabliteTasks, columns: seq[TabliteColumn]): TabliteTable =
    TabliteTable(task: task, columns: columns)

proc saveTable*(table: TabliteTable, pid: string, taskname: string): string =
    let table_path = pid & "/pages/" & taskname & ".json"
    let fh = open(table_path, fmWrite)

    let json = %* table

    fh.write($json)

    fh.close()

    return table_path

