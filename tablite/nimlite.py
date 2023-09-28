import psutil
import platform
import nimporter
import numpy as np
from pathlib import Path
from typing import Literal
import _nimlite.nimlite as nl
from tqdm import tqdm as _tqdm
from tablite.config import Config
from mplite import Task, TaskManager
from tablite.base import Page, Column, pytype_from_iterable

class TmpPage(Page):
    def __init__(self, id, path, data) -> None:
        self.id = id
        self.path = path / "pages" / f"{self.id}.npy"
        self.len = len(data)
        _, py_dtype = pytype_from_iterable(data) if self.len > 0 else (None, object)
        self.dtype = py_dtype

        self._incr_refcount()

def text_reader_task(*, pid, path, encoding, dialect, task, import_fields, guess_dtypes):
    nl.text_reader_task(
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

    pages = []
    for p in (Path(p) for p in task["pages"]):
        id = int(p.name.replace(p.suffix, ""))
        arr = np.load(p, allow_pickle=True)
        page = TmpPage(id, pid, arr)
        pages.append(page)

    return pages

def text_reader(
        T,
        pid: str, path: str,
        encoding: Literal["ENC_UTF8"]|Literal["ENC_UTF16"]|Literal["ENC_WIN1250"] = "ENC_UTF8",
        *,
        first_row_has_headers: bool = True, header_row_index: int = 0,
        columns: list[str]|None = None,
        start: int|None = None, limit: int|None = None,
        guess_datatypes: bool = False,
        newline: str = '\n', delimiter: str = ',', text_qualifier: str = '"',
        quoting: str, strip_leading_and_tailing_whitespace: bool = True,
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
        guess_dtypes=guess_datatypes,
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
        cols["name"]: Column(col_path)
        for cols in task_columns
    }

    columns = list(column_dict.values())
    for res_pages in res:
        for c, p in zip(columns, res_pages):
            c.pages.append(p)

    table = T(columns=column_dict)

    return table