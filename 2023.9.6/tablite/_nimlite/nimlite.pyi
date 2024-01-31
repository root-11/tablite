def text_reader_task(path,  encoding,  dia_delimiter,  dia_quotechar,  dia_escapechar,  dia_doublequote,  dia_quoting,  dia_skipinitialspace,  dia_skiptrailingspace,  dia_lineterminator,  dia_strict,  guess_dtypes,  tsk_pages,  tsk_offset,  tsk_count, import_fields):
    pass


def text_reader(pid, path, encoding, columns, first_row_has_headers, header_row_index, start, limit, guess_datatypes, newline, delimiter, text_qualifier, strip_leading_and_tailing_whitespace, page_size, quoting):
    pass


def collect_column_select_info(table, cols, dir_pid, pbar):
    pass


def do_slice_convert(dir_pid, page_size, columns, reject_reason_name, res_pass, res_fail, desired_column_map, column_names, is_correct_type):
    pass


def read_page(path):
    pass


def repaginate(column):
    pass
