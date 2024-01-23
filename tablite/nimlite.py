import sys
import psutil
import platform
from pathlib import Path
from tqdm import tqdm as _tqdm
from tablite.config import Config
from mplite import Task, TaskManager
from tablite.utils import load_numpy
from tablite.base import SimplePage, Column, pytype_from_iterable

if True:
    paths = sys.argv[:]
    if Config.USE_NIMPORTER:
        import nimporter

        nimporter.Nimporter.IGNORE_CACHE = True
    import tablite._nimlite.nimlite as nl

    sys.argv.clear()
    sys.argv.extend(paths)  # importing nim module messes with pythons launch arguments!!!

def text_reader_task(*, pid, path, encoding, dialect, task, import_fields, guess_dtypes):
    return nl.text_reader_task(
        path=path,
        encoding=encoding,
        dia_delimiter=dialect["delimiter"],
        dia_quotechar=dialect["quotechar"],
        dia_escapechar=dialect["escapechar"],
        dia_doublequote=dialect["doublequote"],
        dia_quoting=dialect["quoting"],
        dia_skipinitialspace=dialect["skipinitialspace"],
        dia_skiptrailingspace=dialect["skiptrailingspace"],
        dia_lineterminator=dialect["lineterminator"],
        dia_strict=dialect["strict"],
        tsk_pages=task["pages"],
        tsk_offset=task["offset"],
        tsk_count=task["count"],
        import_fields=import_fields,
        guess_dtypes=guess_dtypes
    )


def text_reader(
    T,
    pid, path,
    encoding="ENC_UTF8",
    *,
    first_row_has_headers=True, header_row_index=0,
    columns=None,
    start=None, limit=None,
    guess_datatypes=False,
    newline='\n', delimiter=',', text_qualifier='"',
    quoting, strip_leading_and_tailing_whitespace=True,
    tqdm=_tqdm
):
    assert isinstance(path, Path)
    assert isinstance(pid, Path)

    table = nl.text_reader(
        pid=str(pid),
        path=str(path),
        encoding=encoding,
        first_row_has_headers=first_row_has_headers, header_row_index=header_row_index,
        columns=columns,
        start=start, limit=limit,
        guess_datatypes=guess_datatypes,
        newline=newline, delimiter=delimiter, text_qualifier=text_qualifier,
        quoting=quoting,
        strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace,
        page_size=Config.PAGE_SIZE
    )

    task_info = table["task"]
    task_columns = table["columns"]

    ti_path = task_info["path"]
    ti_encoding = task_info["encoding"]
    ti_dialect = task_info["dialect"]
    ti_guess_dtypes = task_info["guess_dtypes"]
    ti_tasks = task_info["tasks"]
    ti_import_fields = task_info["import_fields"]
    ti_import_field_names = task_info["import_field_names"]

    is_windows = platform.system() == "Windows"
    use_logical = False if is_windows else True

    cpus = max(psutil.cpu_count(logical=use_logical), 1)

    tasks = [
        Task(
            text_reader_task,
            path=ti_path,
            encoding=ti_encoding,
            dialect=ti_dialect,
            task=t,
            guess_dtypes=ti_guess_dtypes,
            import_fields=ti_import_fields,
            pid=pid
        ) for t in ti_tasks
    ]

    is_sp = False

    if Config.MULTIPROCESSING_MODE == Config.FALSE:
        is_sp = True
    elif Config.MULTIPROCESSING_MODE == Config.AUTO and cpus <= 1 or len(tasks) <= 1:
        is_sp = True
    elif Config.MULTIPROCESSING_MODE == Config.FORCE:
        is_sp = False

    if is_sp:
        res = [
            task.f(*task.args, **task.kwargs)
            for task in tqdm(tasks, "importing file")
        ]
    else:
        with TaskManager(cpus) as tm:
            res = tm.execute(tasks, tqdm)

            if not all(isinstance(r, list) for r in res):
                raise Exception("failed")

    col_path = pid
    column_dict = {
        cols: Column(col_path)
        for cols in ti_import_field_names
    }

    for res_pages in res:
        col_map = {
            n: res_pages[i]
            for i, n in enumerate(ti_import_field_names)
        }

        for k, c in column_dict.items():
            c.pages.append(col_map[k])

    if columns is None:
        columns = [c["name"] for c in task_columns]

    table_dict = {
        a["name"]: column_dict[b]
        for a, b in zip(task_columns, columns)
    }

    table = T(columns=table_dict)

    return table


def wrap(str_):
    return '"' + str_.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n").replace("\t", "\\t") + '"'


def collect_cs_info(i: int, columns: dict, res_cols_pass: list, res_cols_fail: list):
    el = {
        k: column[i]
        for k, column in columns.items()
    }

    col_pass = res_cols_pass[i]
    col_fail = res_cols_fail[i]

    return el, col_pass, col_fail


def column_select(table, cols, tqdm=_tqdm, TaskManager=TaskManager):
    T = type(table)
    dir_pid = Config.workdir / Config.pid

    columns, page_count, is_correct_type, desired_column_map, passed_column_data, failed_column_data, res_cols_pass, res_cols_fail, column_names, reject_reason_name = nl.collect_column_select_info(table, cols, str(dir_pid))

    if all(is_correct_type.values()):
        tbl_pass_columns = {
            desired_name: table[desired_info[0]]
            for desired_name, desired_info in desired_column_map.items()
        }

        tbl_fail_columns = {
            desired_name: []
            for desired_name in failed_column_data
        }

        tbl_pass = T(columns=tbl_pass_columns)
        tbl_fail = T(columns=tbl_fail_columns)

        return (tbl_pass, tbl_fail)

    task_list_inp = (
        collect_cs_info(i, columns, res_cols_pass, res_cols_fail)
        for i in range(page_count)
    )

    page_size = Config.PAGE_SIZE

    tasks = (
        Task(
            nl.do_slice_convert, str(dir_pid), page_size, columns, reject_reason_name, res_pass, res_fail, desired_column_map, column_names, is_correct_type
        )
        for columns, res_pass, res_fail in task_list_inp
    )

    cpu_count = max(psutil.cpu_count(), 1)

    if Config.MULTIPROCESSING_MODE == Config.FORCE:
        is_mp = True
    elif Config.MULTIPROCESSING_MODE == Config.FALSE:
        is_mp = False
    elif Config.MULTIPROCESSING_MODE == Config.AUTO:
        is_multithreaded = cpu_count > 1
        is_multipage = page_count > 1

        is_mp = is_multithreaded and is_multipage

    tbl_pass = T({k: [] for k in passed_column_data})
    tbl_fail = T({k: [] for k in failed_column_data})

    converted = []
    pbar = tqdm(total=page_count, desc="column select")

    if is_mp:
        with TaskManager(cpu_count=cpu_count) as tm:
            res = tm.execute(list(tasks), pbar=pbar)

            if any(isinstance(r, str) for r in res):
                raise Exception("tasks failed")

            converted.extend(res)
    else:
        for task in tasks:
            res = task.execute()

            if isinstance(res, str):
                raise Exception(res)

            converted.append(res)
            pbar.update(1)

    def extend_table(table, columns):
        for (col_name, pg) in columns:
            table[col_name].pages.append(pg)

    for pg_pass, pg_fail in converted:
        extend_table(tbl_pass, pg_pass)
        extend_table(tbl_fail, pg_fail)

    return tbl_pass, tbl_fail
