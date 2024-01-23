when isLib:
    import ../funcs/column_selector as column_selector

    proc collect_column_select_info*(table: PyObject, cols: PyObject, dir_pid: string): (
        Table[string, seq[string]], int, Table[string, bool], PyObject, seq[string], seq[string], seq[column_selector.ColInfo], seq[column_selector.ColInfo], seq[string], string
    ) {.exportpy.} =
        var (columns, page_count, is_correct_type, desired_column_map, passed_column_data, failed_column_data, res_cols_pass, res_cols_fail, column_names, reject_reason_name) = column_selector.collectColumnSelectInfo(table, cols, dir_pid)

        return (columns, page_count, is_correct_type, desired_column_map.toPyObj, passed_column_data, failed_column_data, res_cols_pass, res_cols_fail, column_names, reject_reason_name)

    proc do_slice_convert*(dir_pid: string, page_size: int, columns: Table[string, string], reject_reason_name: string, res_pass: column_selector.ColInfo, res_fail: column_selector.ColInfo, desired_column_map: PyObject, column_names: seq[string], is_correct_type: Table[string, bool]): (seq[(string, nimpy.PyObject)], seq[(string, nimpy.PyObject)]) {.exportpy.} =
        return column_selector.doSliceConvert(Path(dir_pid), page_size, columns, reject_reason_name, res_pass, res_fail, desired_column_map.fromPyObjToDesiredInfos, column_names, is_correct_type)
