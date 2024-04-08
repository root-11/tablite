import sys
import psutil
import platform
import numpy as np
from pathlib import Path
from tqdm import tqdm as _tqdm
from tablite.config import Config
from mplite import Task, TaskChain, TaskManager
from tablite.base import BaseTable, Column
from typing import TYPE_CHECKING, Literal, Type, TypeVar, TypedDict, Union, List, Tuple


if True:
    paths = sys.argv[:]
    if Config.USE_NIMPORTER:
        import nimporter

        nimporter.Nimporter.IGNORE_CACHE = True
    import nimlite.libnimlite as nl

    sys.argv.clear()
    sys.argv.extend(paths)  # importing nim module messes with pythons launch arguments!!!


K = TypeVar("K", bound=BaseTable)
ValidEncoders = Literal["ENC_UTF8", "ENC_UTF16", "ENC_WIN1250"]
ValidQuoting = Literal["QUOTE_MINIMAL", "QUOTE_ALL", "QUOTE_NONNUMERIC", "QUOTE_NONE", "QUOTE_STRINGS", "QUOTE_NOTNULL"]
ValidSkipEmpty = Literal["NONE", "ANY", "ALL"]
ColumnSelectorDict = TypedDict(
    "ColumnSelectorDict", {
        "column": str,
        "type": Literal["int", "float", "bool", "str", "date", "time", "datetime"],
        "allow_empty": Union[bool, None],
        "rename": Union[str, None]
    }
)

FilterCriteria = Literal[">", ">=", "==", "<", "<=", "!=", "in"]
FilterType = Literal["all", "any"]
FilterDict = TypedDict(
    "FilterDict", {
        "column1": str,
        "value1": Union[str, None],
        "criteria": FilterCriteria,
        "column2": str,
        "value2": Union[str, None],
    }
)



def get_headers(
    path: Union[str, Path],
    encoding: ValidEncoders ="ENC_UTF8",
    *,
    header_row_index: int=0,
    newline: str='\n', delimiter: str=',', text_qualifier: str='"',
    quoting: ValidQuoting, strip_leading_and_tailing_whitespace: bool=True,
    linecount: int = 10
) -> list[list[str]]:
    return nl.get_headers(
            path=str(path),
            encoding=encoding,
            newline=newline, delimiter=delimiter, text_qualifier=text_qualifier,
            strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace,
            header_row_index=header_row_index,
            quoting=quoting,
            linecount=linecount
        )

def text_reader(
    T: Type[K],
    pid: str, path: Union[str, Path],
    encoding: ValidEncoders ="ENC_UTF8",
    *,
    first_row_has_headers: bool=True, header_row_index: int=0,
    columns: List[Union[str, None]]=None,
    start: Union[str, None] = None, limit: Union[str, None]=None,
    guess_datatypes: bool =False,
    newline: str='\n', delimiter: str=',', text_qualifier: str='"',
    quoting: ValidQuoting, strip_leading_and_tailing_whitespace: bool=True, skip_empty: ValidSkipEmpty = "NONE",
    tqdm=_tqdm
) -> K:
    assert isinstance(path, Path)
    assert isinstance(pid, Path)
    with tqdm(total=10, desc=f"importing file") as pbar:
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
            skip_empty=skip_empty,
            page_size=Config.PAGE_SIZE
        )

        pbar.update(1)

        task_info = table["task"]
        task_columns = table["columns"]

        ti_tasks = task_info["tasks"]
        ti_import_field_names = task_info["import_field_names"]

        is_windows = platform.system() == "Windows"
        use_logical = False if is_windows else True

        cpus = max(psutil.cpu_count(logical=use_logical), 1)

        pbar_step = 4 / max(len(ti_tasks), 1)

        class WrapUpdate:
            def update(self, n):
                pbar.update(n * pbar_step)

        wrapped_pbar = WrapUpdate()

        def next_task(task: Task, page_info):
            wrapped_pbar.update(1)
            return Task(
                nl.text_reader_task,
                *task.args, **task.kwargs, page_info=page_info
            )

        tasks = [
            TaskChain(
                Task(
                    nl.collect_text_reader_page_info_task,
                    task=t,
                    task_info=task_info
                ), next_task=next_task
            ) for t in ti_tasks
        ]

        is_sp = False

        if Config.MULTIPROCESSING_MODE == Config.FALSE:
            is_sp = True
        elif Config.MULTIPROCESSING_MODE == Config.FORCE:
            is_sp = False
        elif Config.MULTIPROCESSING_MODE == Config.AUTO and cpus <= 1 or len(tasks) <= 1:
            is_sp = True

        if is_sp:
            res = []

            for task in tasks:
                page = task.execute()

                res.append(page)
        else:
            with TaskManager(cpus, error_mode="exception") as tm:
                res = tm.execute(tasks, pbar=wrapped_pbar)

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

        pbar.update(pbar.total - pbar.n)

        table = T(columns=table_dict)

    return table


def wrap(str_: str) -> str:
    return '"' + str_.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n").replace("\t", "\\t") + '"'


def _collect_cs_info(i: int, columns: dict, res_cols_pass: list, res_cols_fail: list, original_pages_map: list):
    el = {
        name: (column[i], original_pages_map[name][i])
        for name, column in columns.items()
    }

    col_pass = res_cols_pass[i]
    col_fail = res_cols_fail[i]

    return el, col_pass, col_fail


def column_select(table: K, cols: list[ColumnSelectorDict], tqdm=_tqdm, TaskManager=TaskManager) -> Tuple[K, K]:
    with tqdm(total=100, desc="column select", bar_format='{desc}: {percentage:.1f}%|{bar}{r_bar}') as pbar:
        T = type(table)
        dir_pid = Config.workdir / Config.pid

        col_infos = nl.collect_column_select_info(table, cols, str(dir_pid), pbar)

        columns = col_infos["columns"]
        page_count = col_infos["page_count"]
        is_correct_type = col_infos["is_correct_type"]
        desired_column_map = col_infos["desired_column_map"]
        original_pages_map = col_infos["original_pages_map"]
        passed_column_data = col_infos["passed_column_data"]
        failed_column_data = col_infos["failed_column_data"]
        res_cols_pass = col_infos["res_cols_pass"]
        res_cols_fail = col_infos["res_cols_fail"]
        column_names = col_infos["column_names"]
        reject_reason_name = col_infos["reject_reason_name"]

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
            _collect_cs_info(i, columns, res_cols_pass, res_cols_fail, original_pages_map)
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
        step_size = 45 / max(page_count, 1)

        if is_mp:
            class WrapUpdate:
                def update(self, n):
                    pbar.update(n * step_size)

            with TaskManager(min(cpu_count, page_count), error_mode="exception") as tm:
                res = tm.execute(list(tasks), pbar=WrapUpdate())

                converted.extend(res)
        else:
            for task in tasks:
                res = task.f(*task.args, **task.kwargs)

                converted.append(res)
                pbar.update(step_size)

        def extend_table(table, columns):
            for (col_name, pg) in columns:
                table[col_name].pages.append(pg)

        for pg_pass, pg_fail in converted:
            extend_table(tbl_pass, pg_pass)
            extend_table(tbl_fail, pg_fail)

        pbar.update(pbar.total - pbar.n)

        return tbl_pass, tbl_fail

def read_page(path: Union[str, Path]) -> np.ndarray:
    return nl.read_page(str(path))

def repaginate(column: Column):
    nl.repaginate(column)

def nearest_neighbour(T: BaseTable, sources: Union[list[str], None], missing: Union[list, None], targets: Union[list[str], None], tqdm=_tqdm):
    return nl.nearest_neighbour(T, sources, list(missing), targets, tqdm)

def groupby(T, keys, functions, tqdm=_tqdm):
    return nl.groupby(T, keys, functions, tqdm)

def filter(table: BaseTable, expressions: list[FilterDict], type: FilterType, tqdm = _tqdm):
    return nl.filter(table, expressions, type, tqdm)
