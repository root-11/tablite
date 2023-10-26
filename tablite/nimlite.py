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


class NimPage(SimplePage):
    def __init__(self, id, path, data) -> None:
        _len = len(data)
        _, _dtype = pytype_from_iterable(data) if _len > 0 else (None, object)

        super().__init__(id, path, _len, _dtype)


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
        try:
            id = int(p.name.replace(p.suffix, ""))
            arr = load_numpy(p)
            page = NimPage(id, pid, arr)
            pages.append(page)
        except:
            raise

    return pages


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
