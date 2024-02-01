import std/[os, strutils]
import csvparse, encfile, ../../utils, table

type TaskArgs* = object
    path*: string
    encoding*: FileEncoding
    dialect*: Dialect
    guessDtypes*: bool
    destinations*: seq[string]
    importFields*: seq[uint]
    rowOffset*: uint
    rowCount*: int

proc toTaskArgs*(
    path: string,
    encoding: string,
    diaDelimiter: string,
    diaQuotechar: string,
    diaEscapechar: string,
    diaDoublequote: bool,
    diaQuoting: string,
    diaSkipInitialSpace: bool,
    diaSkipTrailingSpace: bool,
    diaLineTerminator: string,
    diaStrict: bool,
    guessDtypes: bool,
    tskPages: seq[string],
    tskOffset: uint,
    tskCount: uint,
    importFields: seq[uint]
): TaskArgs =
    var delimiter = diaDelimiter.unescapeSeq()
    var quotechar = diaQuotechar.unescapeSeq()
    var escapechar = diaEscapechar.unescapeSeq()
    var lineterminator = diaLineTerminator.unescapeSeq()

    if delimiter.len != 1: raise newException(IOError, "'delimiter' must be 1 character")
    if quotechar.len != 1: raise newException(IOError, "'quotechar' must be 1 character")
    if escapechar.len != 1: raise newException(IOError, "'escapechar' must be 1 character")
    if lineterminator.len != 1: raise newException(IOError, "'lineterminator' must be 1 character")

    var dialect = newDialect(
        delimiter = delimiter[0],
        quotechar = quotechar[0],
        escapechar = escapechar[0],
        doublequote = diaDoublequote,
        quoting = str2quoting(diaQuoting),
        skipinitialspace = diaSkipInitialSpace,
        skiptrailingspace = diaSkipTrailingSpace,
        lineterminator = lineterminator[0],
    )

    var arg_encoding = str2Enc(encoding)

    return TaskArgs(
        path: path,
        encoding: arg_encoding,
        dialect: dialect,
        guessDtypes: guessDtypes,
        destinations: tskPages,
        importFields: importFields,
        rowOffset: tskOffset,
        rowCount: int tskCount
    )

proc saveTasks*(task: TabliteTasks, pid: string, taskname: string): string =
    let task_path = pid & "/pages/" & taskname & ".txt"
    let fh = open(task_path, fmWrite)

    for column_task in task.tasks:
        fh.write("\"" & getAppFilename() & "\" ")

        fh.write("--encoding=\"" & task.encoding & "\" ")
        fh.write("--guess_dtypes=" & $task.guessDtypes & " ")

        fh.write("--delimiter=\"" & task.dialect.delimiter & "\" ")
        fh.write("--quotechar=\"" & task.dialect.quotechar & "\" ")
        fh.write("--escapechar=\"" & task.dialect.escapechar & "\" ")
        fh.write("--lineterminator=\"" & task.dialect.lineterminator & "\" ")
        fh.write("--doublequote=" & $task.dialect.doublequote & " ")
        fh.write("--skipinitialspace=" & $task.dialect.skipinitialspace & " ")
        fh.write("--skiptrailingspace=" & $task.dialect.skiptrailingspace & " ")
        fh.write("--quoting=" & task.dialect.quoting & " ")

        fh.write("task ")

        fh.write("--pages=\"" & column_task.pages.join(",") & "\" ")
        fh.write("--fields=\"" & task.importFields.join(",") & "\" ")

        fh.write("\"" & task.path & "\" ")
        fh.write($column_task.offset & " ")
        fh.write(task.page_size)

        fh.write("\n")

    fh.close()

    return task_path