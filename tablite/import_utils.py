import math
import psutil
from pathlib import Path
import pyexcel
import sys
import warnings
import logging

from datatypes import DataTypes
from config import Config
from file_reader_utils import TextEscape
from utils import type_check, unique_name, sub_cls_check
from base import Table

from tqdm import tqdm as _tqdm

logging.getLogger("lml").propagate = False
logging.getLogger("pyexcel_io").propagate = False
logging.getLogger("pyexcel").propagate = False


def from_pandas(T, df):
    """
    Creates Table using pd.to_dict('list')

    similar to:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
    >>> df
        a  b
        0  1  4
        1  2  5
        2  3  6
    >>> df.to_dict('list')
    {'a': [1, 2, 3], 'b': [4, 5, 6]}

    >>> t = Table.from_dict(df.to_dict('list))
    >>> t.show()
        +===+===+===+
        | # | a | b |
        |row|int|int|
        +---+---+---+
        | 0 |  1|  4|
        | 1 |  2|  5|
        | 2 |  3|  6|
        +===+===+===+
    """
    sub_cls_check(type(T), Table)
    return T(columns=df.to_dict("list"))  # noqa


def from_hdf5(T, path):
    """
    imports an exported hdf5 table.
    """
    sub_cls_check(type(T), Table)

    import h5py

    type_check(path, Path)
    t = T()
    with h5py.File(path, "r") as h5:
        for col_name in h5.keys():
            dset = h5[col_name]
            t[col_name] = dset[:]
    return t


def from_json(T, jsn):
    """
    Imports tables exported using .to_json
    """
    sub_cls_check(T, Table)
    import json

    type_check(jsn, bytes)
    return Table(columns=json.loads(jsn))


def excel_reader(T, path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table from excel

    **kwargs are excess arguments that are ignored.
    """
    sub_cls_check(T, Table)

    book = pyexcel.get_book(file_name=str(path))

    if sheet is None:  # help the user.
        raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in book]}")
    elif sheet not in {ws.name for ws in book}:
        raise ValueError(f"sheet not found: {sheet}")

    if not (isinstance(start, int) and start >= 0):
        raise ValueError("expected start as an integer >=0")
    if not (isinstance(limit, int) and limit > 0):
        raise ValueError("expected limit as integer > 0")

    # import a sheets
    for ws in book:
        if ws.name != sheet:
            continue
        else:
            break
    if ws.name != sheet:
        raise ValueError(f'sheet "{sheet}" not found:\n\tSheets: {[str(ws.name) for ws in book]}')

    if columns is None:
        if first_row_has_headers:
            columns = [i[0] for i in ws.columns()]
        else:
            columns = [str(i) for i in range(len(ws.columns()))]

    used_columns_names = set()
    t = T()
    for idx, col in enumerate(ws.columns()):
        if first_row_has_headers:
            header, start_row_pos = str(col[0]), max(1, start)
        else:
            header, start_row_pos = str(idx), max(0, start)

        if header not in columns:
            continue

        unique_column_name = unique_name(str(header), used_columns_names)
        used_columns_names.add(unique_column_name)

        t[unique_column_name] = [v for v in col[start_row_pos : start_row_pos + limit]]
    return t


def ods_reader(T, path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table from .ODS
    """
    sub_cls_check(T, Table)
    sheets = pyexcel.get_book_dict(file_name=str(path))

    if sheet is None or sheet not in sheets:
        raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in sheets]}")

    data = sheets[sheet]
    for _ in range(len(data)):  # remove empty lines at the end of the data.
        if "" == "".join(str(i) for i in data[-1]):
            data = data[:-1]
        else:
            break

    if not (isinstance(start, int) and start >= 0):
        raise ValueError("expected start as an integer >=0")
    if not (isinstance(limit, int) and limit > 0):
        raise ValueError("expected limit as integer > 0")

    t = T()

    used_columns_names = set()
    for ix, value in enumerate(data[0]):
        if first_row_has_headers:
            header, start_row_pos = str(value), 1
        else:
            header, start_row_pos = f"_{ix + 1}", 0

        if columns is not None:
            if header not in columns:
                continue

        unique_column_name = unique_name(str(header), used_columns_names)
        used_columns_names.add(unique_column_name)

        t[unique_column_name] = [row[ix] for row in data[start_row_pos : start_row_pos + limit] if len(row) > ix]
    return t


def text_reader_task(
    source,
    table_key,
    columns,
    newline,
    guess_datatypes,
    delimiter,
    text_qualifier,
    text_escape_openings,
    text_escape_closures,
    strip_leading_and_tailing_whitespace,
    encoding,
):
    """PARALLEL TASK FUNCTION
    reads columnsname + path[start:limit] into hdf5.

    source: csv or txt file
    destination: available filename

    columns: column names or indices to import

    newline: '\r\n' or '\n'
    delimiter: ',' ';' or '|'
    text_escape_openings: str: default: "({[
    text_escape_closures: str: default: ]})"
    strip_leading_and_tailing_whitespace: bool

    encoding: chardet encoding ('utf-8, 'ascii', ..., 'ISO-22022-CN')
    root: hdf5 root, cannot be the same as a column name.
    """
    if isinstance(source, str):
        source = Path(source)
    type_check(source, Path)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    type_check(table_key, str)
    type_check(columns, list)

    if not all(isinstance(name, str) for name in columns):
        raise TypeError("All column names were not str")

    # declare CSV dialect.
    text_escape = TextEscape(
        text_escape_openings,
        text_escape_closures,
        text_qualifier=text_qualifier,
        delimiter=delimiter,
        strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace,
    )

    _index_error_found = False
    with source.open("r", encoding=encoding) as fi:  # --READ
        for line in fi:
            line = line.rstrip(newline)
            break  # break on first
        headers = text_escape(line)
        indices = {name: headers.index(name) for name in columns}
        data = {h: [] for h in indices}
        for line in fi:  # 1 IOP --> RAM.
            fields = text_escape(line.rstrip("\n"))
            if fields == [""] or fields == []:
                break
            for header, index in indices.items():
                try:
                    data[header].append(fields[index])
                except IndexError:
                    data[header].append(None)

                    if _index_error_found is False:  # only show the error once.
                        warnings.warn(f"Column {header} was not found. None will appear as fill value.")
                        _index_error_found = True

    # -- WRITE
    columns_refs = {}
    for col_name, values in data.items():
        if guess_datatypes:
            values = DataTypes.guess(values)
        columns_refs[col_name] = mem.mp_write_column(values)
    mem.mp_write_table(table_key, columns=columns_refs)


def _text_reader_task_size(
    newlines,
    filesize,
    cpu_count,
    free_virtual_memory,
    working_overhead=40,
    memory_usage_ceiling=0.9,
    python_mem_w_imports=40e6,
):
    """
    This function seeks to find the optimal allocation of RAM and CPU as
    CSV reading with type detection is CPU intensive.

    newlines: int: number of lines in the file
    filesize: int: size of file in bytes
    cpu_count: int: number of logical cpus
    free_virtual_memory: int: free ram available in bytes.
    working_overhead: int: number of bytes required per processed charater.
    memory_usage_ceiling: float: percentage of memory allowed to be occupied during task execution.
    python_mem_w_imports: int: memory required to launch python in a subprocess and load all imports.

    returns:
        lines per task: int
        cpu_count: int >= 0.
            If cpu_count returned is zero it means that there
            isn't enough memory to launch a subprocess.

    """
    working_overhead = max(working_overhead, 1)
    bytes_per_line = math.ceil(filesize / newlines)
    total_workload = working_overhead * filesize

    reserved_memory = int(free_virtual_memory * memory_usage_ceiling)

    if (
        total_workload < reserved_memory and total_workload < 10_000_000
    ):  # < 10 Mb:  It's a small task: use current process.
        lines_per_task, cpu_count = newlines + 1, 0
    else:
        multicore = False
        if cpu_count >= 2:  # there are multiple vPUs
            for n_cpus in range(
                cpu_count, 1, -1
            ):  # count down from max to min number of cpus until the task fits into RAM.
                free_memory = reserved_memory - (n_cpus * python_mem_w_imports)
                free_memory_per_vcpu = int(free_memory / cpu_count)  # 8 gb/ 16vCPU = 500Mb/vCPU
                lines_per_task = math.ceil(
                    free_memory_per_vcpu / (bytes_per_line * working_overhead)
                )  # 500Mb/vCPU / (10 * 109 bytes / line ) = 458715 lines per task

                cpu_count = n_cpus
                if free_memory_per_vcpu > 10_000_000:  # 10Mb as minimum task size
                    multicore = True
                    break

        if not multicore:  # it's a large task and there is no memory for another python subprocess.
            # Use current process and divide the total workload to fit into free memory.
            lines_per_task = newlines // max(1, math.ceil(total_workload / reserved_memory))
            cpu_count = 0

    return lines_per_task, cpu_count


def text_reader(
    T,
    path,
    columns,
    header_line,
    first_row_has_headers,
    encoding,
    start,
    limit,
    newline,
    guess_datatypes,
    text_qualifier,
    strip_leading_and_tailing_whitespace,
    delimiter,
    text_escape_openings,
    text_escape_closures,
    tqdm=_tqdm,
    **kwargs,
):
    """
    reads any text file

    excess kwargs are ignored.
    """

    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    read_stage, process_stage, dump_stage, consolidation_stage = 20, 50, 20, 10

    pbar_fname = path.name

    if len(pbar_fname) > 20:
        pbar_fname = pbar_fname[0:10] + "..." + pbar_fname[-7:]

    assert sum([read_stage, process_stage, dump_stage, consolidation_stage]) == 100, "Must add to to a 100"

    file_length = path.stat().st_size  # 9,998,765,432 = 10Gb

    with tqdm(
        total=100,
        desc=f"importing: reading '{pbar_fname}' bytes",
        unit="%",
        bar_format="{desc}: {percentage:3.2f}%|{bar}| [{elapsed}<{remaining}]",
    ) as pbar:
        with path.open("r", encoding=encoding, errors="ignore") as fi:
            # task: find chunk ...
            # Here is the problem in a nutshell:
            # --------------------------------------------------------
            # text = "this is my \n text".encode('utf-16')
            # >>> text
            # b'\xff\xfet\x00h\x00i\x00s\x00 \x00i\x00s\x00 \x00m\x00y\x00 \x00\n\x00 \x00t\x00e\x00x\x00t\x00'
            # >>> newline = "\n".encode('utf-16')
            # >>> newline in text
            # False
            # >>> newline.decode('utf-16') in text.decode('utf-16')
            # True
            # --------------------------------------------------------
            # This means we can't read the encoded stream to check if in contains a particular character.
            # We will need to decode it.
            # furthermore fi.tell() will not tell us which character we a looking at.
            # so the only option left is to read the file and split it in workable chunks.
            if not (isinstance(start, int) and start >= 0):
                raise ValueError("expected start as an integer >= 0")
            if not (isinstance(limit, int) and limit > 0):
                raise ValueError("expected limit as an integer > 0")

            try:
                newlines = 0

                for block in fi:
                    newlines = newlines + 1

                    pbar.update((len(block) / file_length) * read_stage)

                pbar.desc = f"importing: processing '{pbar_fname}'"
                pbar.update(read_stage - pbar.n)

                fi.seek(0)
            except Exception as e:
                raise ValueError(f"file could not be read with encoding={encoding}\n{str(e)}")
            if newlines < 1:
                raise ValueError(f"Using {newline} to split file, revealed {newlines} lines in the file.")

            if newlines <= start + (1 if first_row_has_headers else 0):  # Then start > end: Return EMPTY TABLE.
                t = Table()
                t.add_columns(*columns)
                t.save = True
                return t

        cpu_count = max(psutil.cpu_count(logical=True), 1)  # there's always at least one core!
        free_mem = psutil.virtual_memory().available

        lines_per_task, cpu_count = _text_reader_task_size(
            newlines=newlines, filesize=file_length, cpu_count=cpu_count, free_virtual_memory=free_mem
        )

        task_config = {
            "source": None,  # populated during task creation
            "table_key": None,  # populated during task creation
            "columns": columns,
            "newline": newline,
            "guess_datatypes": guess_datatypes,
            "delimiter": delimiter,
            "text_qualifier": text_qualifier,
            "text_escape_openings": text_escape_openings,
            "text_escape_closures": text_escape_closures,
            "encoding": encoding,
            "strip_leading_and_tailing_whitespace": strip_leading_and_tailing_whitespace,
        }

        checks = [
            True,
            True,
            isinstance(columns, list),
            isinstance(newline, str),
            isinstance(delimiter, str),
            isinstance(text_qualifier, (str, type(None))),
            isinstance(text_escape_openings, str),
            isinstance(text_escape_closures, str),
            isinstance(encoding, str),
            isinstance(strip_leading_and_tailing_whitespace, bool),
        ]
        if not all(checks):  # create an informative error message
            L = []
            for cfg, chk in zip(task_config, checks):
                if not chk:
                    L.append(f"{cfg}:{task_config[cfg]}")
            L = "\n\t".join(L)
            raise ValueError("error in import config:\n{}")

        tasks = []
        with path.open("r", encoding=encoding, errors="ignore") as fi:
            parts = []
            assert header_line != "" and header_line.endswith(newline)

            for ix, line in enumerate(fi, start=(-1 if first_row_has_headers else 0)):
                if ix < start:
                    # ix is -1 if the first row has headers, but header_line already has the first line.
                    # ix is 0 if there are no headers, and if start is 0, the first row is added to parts.
                    continue
                if ix >= start + limit:
                    break

                parts.append(line)
                if ix != 0 and ix % lines_per_task == 0:
                    p = Config.TEMPDIR / (path.stem + path.suffix)
                    with p.open("w", encoding=Config.ENCODING) as fo:
                        parts.insert(0, header_line)
                        fo.write("".join(parts))
                    pbar.update((len(parts) / newlines) * process_stage)
                    parts.clear()
                    tasks.append(
                        Task(
                            text_reader_task,
                            **{
                                **task_config,
                                **{"source": str(p), "table_key": mem.new_id("/table"), "encoding": "utf-8"},
                            },
                        )
                    )

            if parts:  # any remaining parts at the end of the loop.
                p = Config.TEMPDIR / (path.stem + path.suffix)
                with p.open("w", encoding=Config.ENCODING) as fo:
                    parts.insert(0, header_line)
                    fo.write("".join(parts))
                pbar.update((len(parts) / newlines) * process_stage)
                parts.clear()
                task_config.update({"source": str(p), "table_key": mem.new_id("/table")})
                tasks.append(
                    Task(
                        text_reader_task,
                        **{**task_config, **{"source": str(p), "table_key": mem.new_id("/table"), "encoding": "utf-8"}},
                    )
                )

        pbar.desc = f"importing: saving '{pbar_fname}' to disk"
        pbar.update((read_stage + process_stage) - pbar.n)

        len_tasks = len(tasks)
        dump_size = dump_stage / len_tasks

        class PatchTqdm:  # we need to re-use the tqdm pbar, this will patch the tqdm to update existing pbar instead of creating a new one
            def update(self, n=1):
                pbar.update(n * dump_size)

        if cpu_count > 1:
            # execute the tasks with multiprocessing
            with TaskManager(cpu_count - 1) as tm:
                errors = tm.execute(tasks, pbar=PatchTqdm())  # I expects a list of None's if everything is ok.

                # clean up the tmp source files, before raising any exception.
                for task in tasks:
                    tmp = Path(task.kwargs["source"])
                    tmp.unlink()

                if any(errors):
                    raise Exception("\n".join(e for e in errors if e))
        else:  # execute the tasks in currently process.
            for task in tasks:
                assert isinstance(task, Task)
                task.execute()

                pbar.update(dump_size)

        pbar.desc = f"importing: consolidating '{pbar_fname}'"
        pbar.update((read_stage + process_stage + dump_stage) - pbar.n)

        consolidation_size = consolidation_stage / len_tasks

        # consolidate the task results
        t = None
        for task in tasks:
            tmp = Table.load(path=mem.path, key=task.kwargs["table_key"])
            if t is None:
                t = tmp.copy()
            else:
                t += tmp
            tmp.save = False  # allow deletion of subproc tables.

            pbar.update(consolidation_size)

        pbar.update(100 - pbar.n)

        t.save = True
        return t


def make_text_reader_config(
    path,
    columns=None,
    first_row_has_headers=True,
    encoding=None,
    start=0,
    limit=sys.maxsize,
    sheet=None,
    guess_datatypes=True,
    newline="\n",
    text_qualifier=None,
    delimiter=None,
    strip_leading_and_tailing_whitespace=True,
    text_escape_openings="",
    text_escape_closures="",
    tqdm=_tqdm,
    **kwargs,
):
    additional_configs["tqdm"] = tqdm if tqdm is not None else iter

    if path.stat().st_size == 0:
        return cls()  # NO DATA: EMPTY TABLE.

    if encoding is None:
        encoding = get_encoding(path)

    if delimiter is None:
        try:
            delimiter = get_delimiter(path, encoding)
        except ValueError:
            return Table()  # NO DELIMITER: EMPTY TABLE.

    line_reader = TextEscape(
        openings=text_escape_openings,
        closures=text_escape_closures,
        text_qualifier=text_qualifier,
        delimiter=delimiter,
        strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace,
    )

    if not isinstance(newline, str):
        raise TypeError("newline must be str")

    with path.open("r", encoding=encoding) as fi:
        for line in fi:
            if line == "":  # skip any empty line.
                continue
            else:
                header_line = line
                line = line.rstrip(newline)
                break

        fields = line_reader(line)
        if not fields:
            warnings.warn("file was empty: {path}")
            return Table()  # returning an empty table as there was no data.

        if len(set(fields)) != len(fields):  # then there's duplicate names.
            new_fields = []
            for name in fields:
                new_fields.append(unique_name(name, new_fields))
            header_line = delimiter.join(new_fields) + newline
            fields = new_fields

    if first_row_has_headers:
        if columns is None or columns == []:
            columns = fields[:]
        elif isinstance(columns, list):
            for name in columns:
                if name not in fields:
                    raise ValueError(f"column not found: {name}")
        else:
            # fmt: off
            raise TypeError(f"The available columns are {fields}.\na list of strings was expected but {type(columns)} was received.")  # noqa
            # fmt: on
    else:
        if columns is None:
            columns = [str(i) for i in range(len(fields))]
        elif isinstance(columns, list):
            valids = [str(i) for i in range(len(fields))]
            for index in columns:
                if str(index) not in valids:
                    raise ValueError(f"index {index} not in range({len(fields)})")
        else:
            # fmt: off
            raise TypeError(f"The available columns are {fields}.\na list of strings was expected but {type(columns)} was received.")  # noqa
            # fmt: on

        header_line = delimiter.join(columns) + newline

    if not isinstance(strip_leading_and_tailing_whitespace, bool):
        raise TypeError("expected strip_leading_and_tailing_whitespace as boolean")

    config = {
        "path": str(path),
        "import_as": import_as,
        "columns": columns,
        "header_line": header_line,
        "first_row_has_headers": first_row_has_headers,
        "guess_datatypes": guess_datatypes,
        "encoding": encoding,
        "start": start,
        "limit": limit,
        "newline": newline,
        "text_qualifier": text_qualifier,
        "strip_leading_and_tailing_whitespace": strip_leading_and_tailing_whitespace,
        "delimiter": delimiter,
        "text_escape_openings": text_escape_openings,
        "text_escape_closures": text_escape_closures,
        "filesize": path.stat().st_size,  # if file length changes - re-import.
    }
    return config



file_readers = {  # dict of file formats and functions used during Table.import_file
    "fods": excel_reader,
    "json": excel_reader,
    "html": excel_reader,
    "simple": excel_reader,
    "rst": excel_reader,
    "mediawiki": excel_reader,
    "xlsx": excel_reader,
    "xls": excel_reader,
    "xlsm": excel_reader,
    "csv": text_reader,
    "tsv": text_reader,
    "txt": text_reader,
    "ods": ods_reader,
}

valid_readers = ",".join(list(file_readers.keys()))

