import sys
import json
import psutil
import platform
import numpy as np
import subprocess as sp
from pathlib import Path
from tqdm import tqdm as _tqdm
from tablite.config import Config
from mplite import Task, TaskManager
from tablite.utils import generate_random_string
from tablite.base import Page, Column, pytype_from_iterable

IS_WINDOWS = platform.system() == "Windows"
USE_CLI_BACKEND = IS_WINDOWS

CLI_BACKEND_PATH = Path(__file__).parent.parent / f"_nimlite/nimlite{'.exe' if IS_WINDOWS else ''}"

if not USE_CLI_BACKEND:
    paths = sys.argv[:]
    if Config.USE_NIMPORTER:
        import nimporter

        nimporter.Nimporter.IGNORE_CACHE = True
    import _nimlite.nimlite as nl

    sys.argv.clear()
    sys.argv.extend(paths)  # importing nim module messes with pythons launch arguments!!!


class TmpPage(Page):
    def __init__(self, id, path, data) -> None:
        self.id = id
        self.path = path / "pages" / f"{self.id}.npy"
        self.len = len(data)
        _, py_dtype = pytype_from_iterable(data) if self.len > 0 else (None, object)
        self.dtype = py_dtype

        self._incr_refcount()


def text_reader_task(*, pid, path, encoding, dialect, task, import_fields, guess_dtypes):
    if not USE_CLI_BACKEND:
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
    else:
        args = [
            str(CLI_BACKEND_PATH),
            f"--encoding={encoding}",
            f"--guess_dtypes={'true' if guess_dtypes else 'false'}",
            f"--delimiter=\"{dialect['delimiter']}\"",
            f"--quotechar=\"{dialect['quotechar']}\"",
            f"--lineterminator=\"{dialect['lineterminator']}\"",
            f"--skipinitialspace={'true' if dialect['skipinitialspace'] else 'false'}",
            f"--skiptrailingspace={'true' if dialect['skiptrailingspace'] else 'false'}",
            f"--quoting={dialect['quoting']}",
            f"task",
            f"{wrap(str(path))}",
            f"{task['offset']}",
            f"{task['count']}",
            f"--pages={','.join(task['pages'])}",
            f"--fields={','.join((str(v) for v in import_fields))}"
        ]

        sp.run(" ".join(args), shell=True, check=True)

    pages = []
    for p in (Path(p) for p in task["pages"]):
        id = int(p.name.replace(p.suffix, ""))
        arr = np.load(p, allow_pickle=True)
        page = TmpPage(id, pid, arr)
        pages.append(page)

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

    if not USE_CLI_BACKEND:
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
    else:
        taskname = generate_random_string(5)

        while (pid / "pages" / taskname / ".json").exists():
            taskname = generate_random_string(5)

        args = [
            str(CLI_BACKEND_PATH),
            f"--encoding={encoding}",
            f"--guess_dtypes={'true' if guess_datatypes else 'false'}",
            f"--delimiter={wrap(delimiter)}",
            f"--quotechar={wrap(text_qualifier)}",
            f"--lineterminator={wrap(newline)}",
            f"--skipinitialspace={'true' if strip_leading_and_tailing_whitespace else 'false'}",
            f"--skiptrailingspace={'true' if strip_leading_and_tailing_whitespace else 'false'}",
            f"--quoting={quoting}",
            f"import",
            f"{wrap(str(path))}",
            f"--pid={wrap(str(pid))}",
            f"--first_row_has_headers={'true' if first_row_has_headers else 'false'}",
            f"--header_row_index={header_row_index}",
            f"--page_size={Config.PAGE_SIZE}",
            f"--execute=false",
            f"--use_json=true",
            f"--name={taskname}"
        ]

        if start is not None:
            args.append(f"--start={start}")

        if limit is not None:
            args.append(f"--limit={limit}")

        if columns is not None:
            assert isinstance(columns, list)
            assert all(isinstance(v, str) for v in columns)

            args.append(f"--columns='[{','.join([c for c in columns])}]'")

        sp.run(" ".join(args), shell=True, check=True)

        table_path = pid / "pages" / (taskname + ".json")

        with open(table_path) as f:
            table = json.loads(f.read())

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

    try:
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
    finally:
        if USE_CLI_BACKEND and table_path.exists():
            table_path.unlink()

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


def wrap(str_):
    return '"' + str_.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n").replace("\t", "\\t") + '"'
