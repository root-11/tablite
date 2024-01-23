import std/[tables, paths, sequtils]
from std/sugar import collect
from std/os import createDir
from std/strutils import toLower
import nimpy as nimpy
import infos
from ../../pytypes import KindObjectND
import ../../numpy
import ../../pymodules
from ../../utils import uniqueName

proc toPageType(name: string): KindObjectND =
    case name.toLower():
    of "int": return KindObjectND.K_INT
    of "float": return KindObjectND.K_FLOAT
    of "bool": return KindObjectND.K_BOOLEAN
    of "str": return KindObjectND.K_STRING
    of "date": return KindObjectND.K_DATE
    of "time": return KindObjectND.K_TIME
    of "datetime": return KindObjectND.K_DATETIME
    else: raise newException(FieldDefect, "unsupported page type: '" & name & "'")

proc collectColumnSelectInfo*(table: nimpy.PyObject, cols: nimpy.PyObject, dir_pid: string): (
    Table[string, seq[string]], int, Table[string, bool], OrderedTable[string, DesiredColumnInfo], seq[string], seq[string], seq[ColInfo], seq[ColInfo], seq[string], string
) =
    var desired_column_map = initOrderedTable[string, DesiredColumnInfo]()
    var collisions = initTable[string, int]()

    let dirpage = Path(dir_pid) / Path("pages")
    createDir(string dirpage)

    ######################################################
    # 1. Figure out what user needs (types, column names)
    ######################################################
    for c in cols: # now lets iterate over all given columns
        # this is our old name
        let name_inp = c["column"].to(string)
        var rename = c.get("rename", builtins().None)

        if rename.isNone() or not builtins().isinstance(rename, builtins().str).to(bool) or builtins().len(rename).to(int) == 0:
            rename = builtins().None
        else:
            let name_out_stripped = rename.strip()
            if builtins().len(rename).to(int) > 0 and builtins().len(name_out_stripped).to(int) == 0:
                raise newException(ValueError, "Validating 'column_select' failed, '" & name_inp & "' cannot be whitespace.")

            rename = name_out_stripped

        var name_out = if rename.isNone(): name_inp else: rename.to(string)

        if name_out in collisions: # check if the new name has any collision, add suffix if necessary
            collisions[name_out] = collisions[name_out] + 1
            name_out = name_out & "_" & $(collisions[name_out] - 1)
        else:
            collisions[name_out] = 1

        let desired_type = c.get("type", builtins().None)

        desired_column_map[name_out] = DesiredColumnInfo( # collect the information about the column, fill in any defaults
            original_name: name_inp,
            `type`: if desired_type.isNone(): K_NONETYPE else: toPageType(desired_type.to(string)),
            allow_empty: c.get("allow_empty", builtins().False).to(bool)
        )

    ######################################################
    # 2. Converting types to user specified
    ######################################################
    let columns = collect(initTable()):
        for pyColName in table.columns:
            let colName = pyColName.to(string)
            let pyColPages = table[colName].pages
            let pages = collect:
                for pyPage in pyColPages:
                    builtins().str(pyPage.path.absolute()).to(string)

            {colName: pages}

    let column_names = collect: (for k in columns.keys: k)
    var layout_set = newSeq[(int, int)]()

    for pages in columns.values():
        let pgCount = pages.len
        let elCount = getColumnLen(pages)
        let layout = (elCount, pgCount)

        if layout in layout_set:
            continue

        if layout_set.len != 0:
            raise newException(RangeDefect, "Data layout mismatch, pages must be consistent")

        layout_set.add(layout)

    let page_count = layout_set[0][1]

    # Registry of data
    var passed_column_data = newSeq[string]()
    var failed_column_data = newSeq[string]()

    var cols = initTable[string, seq[string]]()

    var res_cols_pass = newSeqOfCap[ColInfo](page_count-1)
    var res_cols_fail = newSeqOfCap[ColInfo](page_count-1)

    for _ in 0..<page_count:
        res_cols_pass.add(initTable[string, ColSliceInfo]())
        res_cols_fail.add(initTable[string, ColSliceInfo]())

    var is_correct_type = initTable[string, bool]()

    proc genpage(dirpid: string): ColSliceInfo {.inline.} = (dir_pid, tabliteBase().SimplePage.next_id(dir_pid).to(int))

    for (desired_name_non_unique, desired_columns) in desired_column_map.pairs():
        let keys = toSeq(passed_column_data)
        let desired_name = uniqueName(desired_name_non_unique, keys)
        let this_col = columns[desired_columns.original_name]

        cols[desired_name] = this_col

        passed_column_data.add(desired_name)

        var col_dtypes = toSeq(this_col.getColumnTypes().keys)
        var needs_to_iterate = false

        if K_NONETYPE in col_dtypes:
            if not desired_columns.allow_empty:
                needs_to_iterate = true
            else:
                col_dtypes.delete(col_dtypes.find(K_NONETYPE))

        if not needs_to_iterate and col_dtypes.len > 0:
            if col_dtypes.len > 1:
                # multiple times always needs to cast
                needs_to_iterate = true
            else:
                let active_type = col_dtypes[0]

                if active_type != desired_columns.`type`:
                    # not same type, need to cast
                    needs_to_iterate = true
                elif active_type == K_STRING and not desired_columns.allow_empty:
                    # we may have same type but we still need to filter empty strings
                    needs_to_iterate = true

        is_correct_type[desired_name] = not needs_to_iterate

        for i in 0..<page_count:
            res_cols_pass[i][desired_name] = genpage(dir_pid)

    for desired_name in columns.keys:
        failed_column_data.add(desired_name)

        for i in 0..<page_count:
            res_cols_fail[i][desired_name] = genpage(dir_pid)

    let reject_reason_name = uniqueName("reject_reason", column_names)

    for i in 0..<page_count:
        res_cols_fail[i][reject_reason_name] = genpage(dir_pid)

    failed_column_data.add(reject_reason_name)

    return (columns, page_count, is_correct_type, desired_column_map, passed_column_data, failed_column_data, res_cols_pass, res_cols_fail, column_names, reject_reason_name)
