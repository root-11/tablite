import math
import pathlib
import json
import sys
import difflib
import itertools
import operator
import warnings
import logging

from collections import defaultdict
from multiprocessing import shared_memory

import pyexcel
import pyperclip
from tqdm import tqdm as _tqdm
import numpy as np
import tablite.h5py as h5py
import psutil
from mplite import TaskManager, Task

import atexit


from tablite.memory_manager import MemoryManager, Page, Pages
from tablite.file_reader_utils import TextEscape, get_headers, get_encoding, get_delimiter
from tablite.utils import summary_statistics, unique_name, expression_interpreter
from tablite.utils import arg_to_slice
from tablite import sortation
from tablite.groupby_utils import GroupBy, GroupbyFunction
from tablite.config import SINGLE_PROCESSING_LIMIT, TEMPDIR, H5_ENCODING
from tablite.datatypes import DataTypes

PYTHON_EXIT = False  # exit handler so that Table.__del__ doesn't run into import error during exit.

def exiting():
    global PYTHON_EXIT
    PYTHON_EXIT = True


atexit.register(exiting)


logging.getLogger("lml").propagate = False
logging.getLogger("pyexcel_io").propagate = False
logging.getLogger("pyexcel").propagate = False

log = logging.getLogger(__name__)


mem = MemoryManager()


def excel_reader(path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table from excel

    **kwargs are excess arguments that are ignored.
    """
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
    t = Table(save=True)
    for idx, column in enumerate(ws.columns()):

        if first_row_has_headers:
            header, start_row_pos = str(column[0]), max(1, start)
        else:
            header, start_row_pos = str(idx), max(0, start)

        if header not in columns:
            continue

        unique_column_name = unique_name(str(header), used_columns_names)
        used_columns_names.add(unique_column_name)

        t[unique_column_name] = [v for v in column[start_row_pos : start_row_pos + limit]]
    return t


def ods_reader(path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table from .ODS
    """
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

    t = Table(save=True)

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
        source = pathlib.Path(source)
    if not isinstance(source, pathlib.Path):
        raise TypeError()
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if not isinstance(table_key, str):
        raise TypeError()

    if not isinstance(columns, list):
        raise TypeError
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
                lines_per_task = math.ceil(free_memory_per_vcpu / (
                    bytes_per_line * working_overhead
                ))  # 500Mb/vCPU / (10 * 109 bytes / line ) = 458715 lines per task

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

    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    read_stage, process_stage, dump_stage, consolidation_stage = 20, 50, 20, 10

    pbar_fname = path.name

    if len(pbar_fname) > 20:
        pbar_fname = pbar_fname[0:10] + "..." + pbar_fname[-7:]

    assert sum([read_stage, process_stage, dump_stage, consolidation_stage]) == 100, "Must add to to a 100"

    file_length = path.stat().st_size  # 9,998,765,432 = 10Gb

    with tqdm(total=100, desc=f"importing: reading '{pbar_fname}' bytes", unit="%", bar_format="{desc}: {percentage:3.2f}%|{bar}| [{elapsed}<{remaining}]") as pbar:
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
                    p = TEMPDIR / (path.stem + f"{ix}" + path.suffix)
                    with p.open("w", encoding=H5_ENCODING) as fo:
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
                p = TEMPDIR / (path.stem + f"{ix}" + path.suffix)
                with p.open("w", encoding=H5_ENCODING) as fo:
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

        class PatchTqdm: # we need to re-use the tqdm pbar, this will patch the tqdm to update existing pbar instead of creating a new one
            def update(self, n=1):
                pbar.update(n * dump_size)

        if cpu_count > 1:
            # execute the tasks with multiprocessing
            with TaskManager(cpu_count - 1) as tm:
                errors = tm.execute(tasks, pbar=PatchTqdm())  # I expects a list of None's if everything is ok.

                # clean up the tmp source files, before raising any exception.
                for task in tasks:
                    tmp = pathlib.Path(task.kwargs["source"])
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


class Table(object):
    def __init__(self, key=None, save=False, _create=True, config=None) -> None:
        if key is None:
            key = mem.new_id("/table")
        elif not isinstance(key, str):
            raise TypeError
        self.key = key

        self.group = f"/table/{self.key}"
        self._columns = {}  # references for virtual datasets that behave like lists.
        if _create:
            if config is not None:
                if isinstance(config, dict):
                    logging.info(
                        f"import config for {config['path']}:\n" + "\n".join(f"{k}:{v}" for k, v in config.items())
                    )
                    config = json.dumps(config)
                if not isinstance(config, str):
                    raise TypeError("expected config as utf-8 encoded json")
            mem.create_table(key=key, save=save, config=config)  # attrs. 'columns'
        self._saved = save

    @property
    def save(self):
        return self._saved

    @save.setter
    def save(self, value):
        """
        Makes the table persistent on disk in HDF5 storage.
        """
        if not isinstance(value, bool):
            raise TypeError(f"expected bool, got: {type(value)}")
        if self._saved != value:
            self._saved = value
            mem.set_saved_flag(self.group, value)

    def __del__(self):
        if PYTHON_EXIT:
            return

        try:
            for key in list(self._columns):
                del self[key]
            mem.delete_table(self.group)
        except KeyError:
            log.info("Table.__del__ suppressed.")

    def __str__(self):
        return f"Table({len(self._columns):,} columns, {len(self):,} rows)"

    def __repr__(self):
        return self.__str__()

    @property
    def columns(self):
        """
        returns list of column names.
        """
        return list(self._columns.keys())

    @property
    def rows(self):
        """
        enables iteration

        for row in Table.rows:
            print(row)
        """

        n_max = len(self)
        generators = []
        for name, mc in self._columns.items():
            if len(mc) < n_max:
                warnings.warn(f"Column {name} has length {len(mc)} / {n_max}. None will appear as fill value.")
            generators.append(itertools.chain(iter(mc), itertools.repeat(None, times=n_max - len(mc))))

        for _ in range(len(self)):
            yield [next(i) for i in generators]

    def __iter__(self):
        """
        Disabled. Users should use Table.rows or Table.columns

        Why? See [1,2,3] below.

        >>> import this
        The Zen of Python, by Tim Peters

        Beautiful is better than ugly.
        Explicit is better than implicit.                          <---- [1]
        Simple is better than complex.
        Complex is better than complicated.
        Flat is better than nested.
        Sparse is better than dense.
        Readability counts.                                        <---- [2]
        Special cases aren't special enough to break the rules.
        Although practicality beats purity.
        Errors should never pass silently.
        Unless explicitly silenced.
        In the face of ambiguity, refuse the temptation to guess.  <---- [3]
        There should be one-- and preferably only one --obvious way to do it.
        Although that way may not be obvious at first unless you're Dutch.
        Now is better than never.
        Although never is often better than *right* now.
        If the implementation is hard to explain, it's a bad idea.
        If the implementation is easy to explain, it may be a good idea.
        Namespaces are one honking great idea -- let's do more of those!
        """
        raise AttributeError("use Table.rows or Table.columns")

    def __len__(self):
        """
        returns length of table.
        """
        if self._columns:
            return max(len(c) for c in self._columns.values())
        return 0  # if there are no columns.

    def __setitem__(self, keys, values):
        """
        Args:
            keys (str, tuple of str's): keys
            values (Column or Iterable): values

        Examples:
            t = Table()
            t['a'] = [1,2,3]  - column 'a' contains values [1,2,3]
            t[('b','c')] = [ [4,5,6], [7,8,9] ]
            # column 'b' contains values [4,5,6]
            # column 'c' contains values [7,8,9]

        """
        if isinstance(keys, str):
            if isinstance(values, (tuple, list, np.ndarray)):
                if len(values) == 0:
                    values = None
                self._columns[keys] = column = Column(values)  # overwrite if exists.
                mem.create_column_reference(self.key, column_name=keys, column_key=column.key)
            elif isinstance(values, Column):
                col = self._columns.get(keys, None)
                if col is None:  # it's a column from another table.
                    self._columns[keys] = col = values.copy()
                elif values.key == col.key:  # it's update from += or similar
                    self._columns[keys] = values
                else:
                    raise TypeError("No method for this case.")
                mem.create_column_reference(self.key, column_name=keys, column_key=col.key)

            elif values is None:  # it's an empty dataset.
                self._columns[keys] = col = Column(values)
                mem.create_column_reference(self.key, column_name=keys, column_key=col.key)
            else:
                raise NotImplementedError(f"No method for values of type {type(values)}")
        elif isinstance(keys, tuple) and len(keys) == len(values):
            for key, value in zip(keys, values):
                self.__setitem__(key, value)
        else:
            raise NotImplementedError(f"No method for keys of type {type(keys)}")

    def __getitem__(self, *keys):
        """
        Enables selection of columns and rows
        Examples:

            table['a']   # selects column 'a'
            table[3]  # selects row 3 as a tuple.
            table[:10]   # selects first 10 rows from all columns
            table['a','b', slice(3,20,2)]  # selects a slice from columns 'a' and 'b'
            table['b', 'a', 'a', 'c', 2:20:3]  # selects column 'b' and 'c' and 'a' twice for a slice.

        returns values in same order as selection.
        """
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(keys) == 1 and all(isinstance(i, tuple) for i in keys):
            keys = keys[0]

        if len(keys) == 1:
            if isinstance(keys[0], int):
                keys = (slice(keys[0]),)

        slices = [i for i in keys if isinstance(i, slice)]
        if len(slices) > 1:
            raise KeyError(f"multiple slices is not accepted: {slices}")

        cols = [i for i in keys if not isinstance(i, slice)]
        if cols:
            key_errors = [cname not in self.columns for cname in cols]
            if any(key_errors):
                raise KeyError(f"keys not found: {key_errors}")
            if len(set(cols)) != len(cols):
                raise KeyError(f"duplicated keys in {cols}")
        else:  # e.g. tbl[:10]
            cols = self.columns

        if len(cols) == 1:  # e.g. tbl['a'] or tbl['a'][:10]
            col = self._columns[cols[0]]
            if slices:
                return col[slices[0]]
            else:
                return col
        elif slices:
            slc = slices[0]
            t = Table()
            for name in cols:
                t[name] = self._columns[name][slc]
            return t
        else:
            t = Table()
            for name in cols:
                t[name] = self._columns[name]
            return t

    def __delitem__(self, key):
        """
        del table['a']  removes column 'a'
        del table[-3:] removes last 3 rows from all columns.
        """
        if isinstance(key, str) and key in self._columns:
            col = self._columns[key]
            mem.delete_column_reference(self.group, key, col.key)
            del self._columns[key]  # dereference the Column
        elif isinstance(key, slice):
            for col in self._columns.values():
                del col[key]
        else:
            raise NotImplementedError()

    def copy(self):
        """
        Returns a copy of the table.
        """
        t = Table()
        for name, col in self._columns.items():
            t[name] = col
        return t

    def clear(self):
        """
        removes all rows and columns from a table.
        """
        for name in self.columns:
            self.__delitem__(name)

    def __eq__(self, __o: object) -> bool:
        """
        Determines if two tables have identical content.
        """
        if not isinstance(__o, Table):
            return False
        if id(self) == id(__o):
            return True
        if len(self) != len(__o):
            return False
        if len(self) == len(__o) == 0:
            return True
        if self.columns != __o.columns:
            return False
        for name, col in self._columns.items():
            if col != __o._columns[name]:
                return False
        return True

    def __add__(self, other):
        """
        enables concatenation for tables with the same column names.
        """
        c = self.copy()
        c += other
        return c

    def __iadd__(self, other):
        """
        enables extension with other tables with the same column names.
        """
        if not isinstance(other, Table):
            raise TypeError(f"no method for {type(other)}")
        if set(self.columns) != set(other.columns) or len(self.columns) != len(other.columns):
            raise ValueError("Columns names are not the same. Use table.stack instead.")
        for name, col in self._columns.items():
            col += other[name]
            mem.create_column_reference(self.key, column_name=name, column_key=col.key)
        return self

    def __mul__(self, other):
        """
        enables repetition of a table
        Example: Table_x_10 = table * 10
        """
        if not isinstance(other, int):
            raise TypeError(f"can't multiply Table with {type(other)}")
        t = self.copy()
        for col in t._columns.values():
            col *= other
        return t

    def __imul__(self, other):
        """
        extends a table N times onto using itself as source.
        """
        if not isinstance(other, int):
            raise TypeError(f"can't multiply Table with {type(other)}")

        for col in self._columns.values():
            col *= other
        return self

    @classmethod
    def reload_saved_tables(cls, path=None):
        """
        Loads saved tables from a hdf5 storage.

        The default storage locations is:
        >>> from tablite.config import HDF5_Config
        >>> print(Config.H5_STORAGE)

        To import without changing the default location use:
        tables = reload_saved_tables("c:/another/location.hdf5)
        """
        tables = []
        if path is None:
            path = mem.path
        unsaved = 0
        with h5py.File(path, "r+") as h5:
            if "/table" not in h5.keys():
                return []

            for table_key in h5["/table"].keys():
                dset = h5[f"/table/{table_key}"]
                if dset.attrs["saved"] is False:
                    unsaved += 1
                else:
                    t = Table.load(path, key=table_key)
                    tables.append(t)
        if unsaved:
            warnings.warn(f"Dropping {unsaved} tables from cache where save==False.")
        return tables

    @classmethod
    def load(cls, path, key):
        """
        Special classmethod to load saved tables stored in hdf5 storage.
        Used by reload_saved_tables
        """
        with h5py.File(path, "r+") as h5:
            group = f"/table/{key}"
            dset = h5[group]
            saved = dset.attrs["saved"]
            t = Table(key=key, save=saved, _create=False)
            columns = json.loads(dset.attrs["columns"])
            for col_name, column_key in columns.items():
                c = Column.load(key=column_key)
                col_dset = h5[f"/column/{column_key}"]
                c._len = col_dset.attrs["length"]
                t[col_name] = c
            return t

    @classmethod
    def reset_storage(cls):
        """Resets all stored tables."""
        mem.reset_storage()

    def add_rows(self, *args, **kwargs):
        """its more efficient to add many rows at once.

        supported cases:

        t = Table()
        t.add_columns('row','A','B','C')

        (1) t.add_rows(1, 1, 2, 3)  # individual values as args
        (2) t.add_rows([2, 1, 2, 3])  # list of values as args
        (3) t.add_rows((3, 1, 2, 3))  # tuple of values as args
        (4) t.add_rows(*(4, 1, 2, 3))  # unpacked tuple becomes arg like (1)
        (5) t.add_rows(row=5, A=1, B=2, C=3)   # kwargs
        (6) t.add_rows(**{'row': 6, 'A': 1, 'B': 2, 'C': 3})  # dict / json interpreted a kwargs
        (7) t.add_rows((7, 1, 2, 3), (8, 4, 5, 6))  # two (or more) tuples as args
        (8) t.add_rows([9, 1, 2, 3], [10, 4, 5, 6])  # two or more lists as rgs
        (9) t.add_rows({'row': 11, 'A': 1, 'B': 2, 'C': 3},
                       {'row': 12, 'A': 4, 'B': 5, 'C': 6})  # two (or more) dicts as args - roughly comma sep'd json.
        (10) t.add_rows( *[ {'row': 13, 'A': 1, 'B': 2, 'C': 3},
                            {'row': 14, 'A': 1, 'B': 2, 'C': 3} ])  # list of dicts as args
        (11) t.add_rows(row=[15,16], A=[1,1], B=[2,2], C=[3,3])  # kwargs with lists as values

        if both args and kwargs, then args are added first, followed by kwargs.
        """
        if args:
            if not all(isinstance(i, (list, tuple, dict)) for i in args):  # 1,4
                args = [args]

            if all(isinstance(i, (list, tuple, dict)) for i in args):  # 2,3,7,8
                # 1. turn the data into columns:
                names = self.columns
                d = {n: [] for n in self.columns}
                for arg in args:
                    if len(arg) != len(names):
                        raise ValueError(f"len({arg})== {len(arg)}, but there are {len(self.columns)} columns")

                    if isinstance(arg, dict):
                        for k, v in arg.items():  # 7,8
                            d[k].append(v)

                    elif isinstance(arg, (list, tuple)):  # 2,3
                        for n, v in zip(names, arg):
                            d[n].append(v)

                    else:
                        raise TypeError(f"{arg}?")
                # 2. extend the columns
                for n, values in d.items():
                    col = self.__getitem__(n)
                    col.extend(values)

        if kwargs:
            if isinstance(kwargs, dict):
                if all(isinstance(v, (list, tuple)) for v in kwargs.values()):
                    for k, v in kwargs.items():
                        col = self._columns[k]
                        col.extend(v)
                else:
                    for k, v in kwargs.items():
                        col = self._columns[k]
                        col.extend([v])
            else:
                raise ValueError(f"format not recognised: {kwargs}")

        return

    def add_columns(self, *names):
        """
        same as:
        for name in names:
            table[name] = None
        """
        for name in names:
            self.__setitem__(name, None)

    def add_column(self, name, data=None):
        """
        verbose alias for table[name] = data, that checks if name already exists
        """
        if not isinstance(name, str):
            raise TypeError()
        if name in self.columns:
            raise ValueError(f"{name} already in {self.columns}")

        self.__setitem__(name, data)

    def stack(self, other):
        """
        returns the joint stack of tables
        Example:

        | Table A|  +  | Table B| = |  Table AB |
        | A| B| C|     | A| B| D|   | A| B| C| -|
                                    | A| B| -| D|
        """
        if not isinstance(other, Table):
            raise TypeError(f"stack only works for Table, not {type(other)}")

        t = self.copy()
        for name, col2 in other._columns.items():
            if name in t.columns:
                t[name].extend(col2)
            elif len(self) > 0:
                t[name] = [None] * len(self)
            else:
                t[name] = col2

        for name, col in t._columns.items():
            if name not in other.columns:
                if len(other) > 0:
                    if len(self) > 0:
                        col.extend([None] * len(other))
                    else:
                        t[name] = [None] * len(other)
        return t

    def types(self):
        """
        returns nested dict of data types in the form:

            {column name: {python type class: number of instances }, }

        example:
        >>> t.types()
        {
            'A': {<class 'str'>: 7},
            'B': {<class 'int'>: 7}
        }
        """

        d = {}
        for name, col in self._columns.items():
            assert isinstance(col, Column)
            d[name] = col.types()
        return d

    def to_ascii(self, blanks=None, row_counts=None, split_after=None):
        """
        enables viewing in terminals
        returns the table as ascii string

        blanks: any stringable item.
        row_counts: declares the column with row counts, so it is presented as the first column.
        split_after: integer: inserts "..." to highlight split of rows
        """
        widths = {}
        column_types = {}
        names = list(self.columns)
        if not names:
            return "Empty table"
        column_lengths = set()
        for name, col in self._columns.items():
            types = col.types()
            if name == row_counts:
                column_types[name] = "row"
            elif len(types) == 1:
                dt, _ = types.popitem()
                column_types[name] = dt.__name__
            else:
                column_types[name] = "mixed"
            dots = len("...") if split_after is not None else 0
            widths[name] = max(
                len(column_types[name]),
                len(name),
                dots,
                len(str(None)) if len(col) != len(self) else 0,
                *[len(str(v)) if not isinstance(v, str) else len(str(v)) for v in col],
            )
            column_lengths.add(len(col))

        def adjust(v, length):
            if v is None:
                return str(blanks).ljust(length)
            elif isinstance(v, str):
                return v.ljust(length)
            else:
                return str(v).rjust(length)

        s = []
        s.append("+" + "+".join(["=" * widths[n] for n in names]) + "+")
        s.append("|" + "|".join([n.center(widths[n], " ") for n in names]) + "|")
        s.append("|" + "|".join([column_types[n].center(widths[n], " ") for n in names]) + "|")
        s.append("+" + "+".join(["-" * widths[n] for n in names]) + "+")
        for ix, row in enumerate(self.rows):
            s.append("|" + "|".join([adjust(v, widths[n]) for v, n in zip(row, names)]) + "|")
            if ix == split_after:
                s.append("|" + "|".join([adjust("...", widths[n]) for _, n in zip(row, names)]) + "|")

        s.append("+" + "+".join(["=" * widths[h] for h in names]) + "+")

        if len(column_lengths) != 1:
            s.append("Warning: Columns have different lengths. None is used as fill value.")

        return "\n".join(s)

    def show(self, *args, blanks=None):
        """
        param: args:
          - slice
        blanks: fill value for `None`
        """
        if not self.columns:
            print("Empty Table")
            return

        row_count_tags = ["#", "~", "*"]
        cols = set(self.columns)
        for n, tag in itertools.product(range(1, 6), row_count_tags):
            if n * tag not in cols:
                tag = n * tag
                break

        t = Table()
        split_after = None
        if args:
            for arg in args:
                if isinstance(arg, slice):
                    ro = range(*arg.indices(len(self)))
                    if len(ro) != 0:
                        t[tag] = [f"{i:,}" for i in ro]  # add rowcounts as first column.
                        for name, col in self._columns.items():
                            t[name] = col[arg]  # copy to match slices
                    else:
                        t.add_columns(*[tag] + self.columns)

        elif len(self) < 20:
            t[tag] = [f"{i:,}".rjust(2) for i in range(len(self))]  # add rowcounts to copy
            for name, col in self._columns.items():
                t[name] = col

        else:  # take first and last 7 rows.
            n = len(self)
            j = int(math.ceil(math.log10(n)) / 3) + len(str(n))
            split_after = 6
            t[tag] = [f"{i:,}".rjust(j) for i in range(7)] + [f"{i:,}".rjust(j) for i in range(n - 7, n)]
            for name, col in self._columns.items():
                t[name] = [i for i in col[:7]] + [i for i in col[-7:]]

        print(t.to_ascii(blanks=blanks, row_counts=tag, split_after=split_after))

    def _repr_html_(self):
        """Ipython display compatible format
        https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
        """
        start, end = "<div><table border=1>", "</table></div>"

        if not self.columns:
            return f"{start}<tr>Empty Table</tr>{end}"

        row_count_tags = ["#", "~", "*"]
        cols = set(self.columns)
        for n, tag in itertools.product(range(1, 6), row_count_tags):
            if n * tag not in cols:
                tag = n * tag
                break

        html = ["<tr>" + f"<th>{tag}</th>" + "".join(f"<th>{cn}</th>" for cn in self.columns) + "</tr>"]

        column_types = {}
        column_lengths = set()
        for name, col in self._columns.items():
            types = col.types()
            if len(types) == 1:
                dt, _ = types.popitem()
                column_types[name] = dt.__name__
            else:
                column_types[name] = "mixed"
            column_lengths.add(len(col))

        html.append(
            "<tr>" + "<th>row</th>" + "".join(f"<th>{column_types[name]}</th>" for name in self.columns) + "</tr>"
        )

        if len(self) < 20:
            for ix, row in enumerate(self.rows):
                html.append("<tr>" + f"<td>{ix}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
        else:
            t = Table()
            for name, col in self._columns.items():
                t[name] = [i for i in col[:7]] + [i for i in col[-7:]]

            c = len(self) - 7
            for ix, row in enumerate(t.rows):
                if ix < 7:
                    html.append("<tr>" + f"<td>{ix}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
                if ix == 7:
                    html.append("<tr>" + "<td>...</td>" + "".join("<td>...</td>" for _ in self._columns) + "</tr>")
                if ix >= 7:
                    html.append("<tr>" + f"<td>{c}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
                    c += 1

        warning = (
            "Warning: Columns have different lengths. None is used as fill value." if len(column_lengths) != 1 else ""
        )

        return start + "".join(html) + end + warning

    def index(self, *args):
        """
        param: *args: column names
        returns multikey index on the columns as d[(key tuple, )] = {index1, index2, ...}

        Examples:
        >>> table6 = Table()
        >>> table6['A'] = ['Alice', 'Bob', 'Bob', 'Ben', 'Charlie', 'Ben','Albert']
        >>> table6['B'] = ['Alison', 'Marley', 'Dylan', 'Affleck', 'Hepburn', 'Barnes', 'Einstein']

        >>> table6.index('A')  # single key.
        {('Alice',): {0},
         ('Bob',): {1, 2},
         ('Ben',): {3, 5},
         ('Charlie',): {4},
         ('Albert',): {6}})

        >>> table6.index('A', 'B')  # multiple keys.
        {('Alice', 'Alison'): {0},
         ('Bob', 'Marley'): {1},
         ('Bob', 'Dylan'): {2},
         ('Ben', 'Affleck'): {3},
         ('Charlie', 'Hepburn'): {4},
         ('Ben', 'Barnes'): {5},
         ('Albert', 'Einstein'): {6}})

        """
        idx = defaultdict(set)
        tbl = self.__getitem__(*args)
        g = tbl.rows if isinstance(tbl, Table) else iter(tbl)
        for ix, key in enumerate(g):
            if isinstance(key, list):
                key = tuple(key)
            else:
                key = (key,)
            idx[key].add(ix)
        return idx

    def copy_to_clipboard(self):
        """copy data from a Table into clipboard."""
        try:
            s = ["\t".join([f"{name}" for name in self.columns])]
            for row in self.rows:
                s.append("\t".join((str(i) for i in row)))
            s = "\n".join(s)
            pyperclip.copy(s)
        except MemoryError:
            raise MemoryError("Cannot copy to clipboard. Select slice instead.")

    @staticmethod
    def copy_from_clipboard():
        """copy data from clipboard into Table."""
        t = Table()
        txt = pyperclip.paste().split("\n")
        t.add_columns(*txt[0].split("\t"))

        for row in txt[1:]:
            data = row.split("\t")
            t.add_rows(data)
        return t

    @classmethod
    def from_dict(self, d):
        """
        creates new Table instance from dict

        Example:
        >>> from tablite import Table
        >>> t = Table.from_dict({'a':[1,2], 'b':[3,4]})
        >>> t
        Table(2 columns, 2 rows)
        >>> t.show()
        +===+===+===+
        | # | a | b |
        |row|int|int|
        +---+---+---+
        | 0 |  1|  3|
        | 1 |  2|  4|
        +===+===+===+
        >>>

        """
        t = Table()
        for k, v in d.items():
            if not isinstance(k, str):
                raise TypeError("expected keys as str")
            if not isinstance(v, (list, tuple)):
                raise TypeError("expected values as list or tuple")
            t[k] = v
        return t

    def to_dict(self, columns=None, slice_=None):
        """
        columns: list of column names. Default is None == all columns.
        slice_: slice. Default is None == all rows.

        Example:
        >>> t.show()
        +===+===+===+
        | # | a | b |
        |row|int|int|
        +---+---+---+
        | 0 |  1|  3|
        | 1 |  2|  4|
        +===+===+===+
        >>> t.to_dict()
        {'a':[1,2], 'b':[3,4]}

        """
        if slice_ is None:
            slice_ = slice(0, len(self))
        assert isinstance(slice_, slice)

        if columns is None:
            columns = self.columns
        if not isinstance(columns, list):
            raise TypeError("expected columns as list of strings")

        column_selection, own_cols = [], set(self.columns)
        for name in columns:
            if name in own_cols:
                column_selection.append(name)
            else:
                raise ValueError(f"column({name}) not found")

        cols = {}
        for name in column_selection:
            col = self._columns[name]
            row_slice = col[slice_]
            cols[name] = (
                row_slice.tolist() if not isinstance(row_slice, list) else row_slice
            )  # pure python objects. No numpy.
        return cols

    def as_json_serializable(self, row_count="row id", start_on=1, columns=None, slice_=None):
        """
        returns json friendly format.

        For data conversion rules see DataTypes.to_json
        """
        if slice_ is None:
            slice_ = slice(0, len(self))
        assert isinstance(slice_, slice)

        new = {"columns": {}, "total_rows": len(self)}
        if row_count is not None:
            new["columns"][row_count] = [i + start_on for i in range(*slice_.indices(len(self)))]

        d = self.to_dict(columns, slice_=slice_)
        for k, data in d.items():
            new_k = unique_name(k, new["columns"])  # used to avoid overwriting the `row id` key.
            new["columns"][new_k] = [DataTypes.to_json(v) for v in data]  # deal with non-json datatypes.
        return new

    def to_json(self, *args, **kwargs):
        return json.dumps(self.as_json_serializable(*args, **kwargs))

    @classmethod
    def from_json(cls, jsn):
        """
        Imports tables exported using .to_json
        """
        d = json.loads(jsn)
        t = Table()
        for name, data in d["columns"].items():
            if not isinstance(name, str):
                raise TypeError(f"expect {name} as a string")
            if not isinstance(data, list):
                raise TypeError(f"expected {data} as list")
            t[name] = data
        return t

    def to_hdf5(self, path, tqdm=_tqdm):
        """
        creates a copy of the table as hdf5
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        total = ":,".format(len(self.columns) * len(self))  # noqa
        print(f"writing {total} records to {path}")

        with h5py.File(path, "a") as f:
            with tqdm(total=len(self.columns), unit="columns") as pbar:
                n = 0
                for name, mc in self.columns.values():
                    f.create_dataset(name, data=mc[:])  # stored in hdf5 as '/name'
                    n += 1
                    pbar.update(n)
        print(f"writing {path} to HDF5 done")

    def to_pandas(self):
        """
        returns pandas.DataFrame
        """
        try:
            return pd.DataFrame(self.to_dict())  # noqa
        except ImportError:
            import pandas as pd  # noqa
        return pd.DataFrame(self.to_dict())  # noqa

    def from_pandas(self, df):
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
        return self.from_dict(df.to_dict("list"))  # noqa

    def from_hdf5(self, path):
        """
        imports an exported hdf5 table.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)

        t = Table()
        with h5py.File(path, "r") as h5:
            for col_name in h5.keys():
                dset = h5[col_name]
                t[col_name] = dset[:]
        return t

    def to_sql(self):
        """
        generates ANSI-92 compliant SQL.
        """
        prefix = "Table"
        create_table = """CREATE TABLE {}{} ({})"""
        columns = []
        for name, col in self._columns.items():
            dtype = col.types()
            if len(dtype) == 1:
                dtype, _ = dtype.popitem()
                if dtype is int:
                    dtype = "INTEGER"
                elif dtype is float:
                    dtype = "REAL"
                else:
                    dtype = "TEXT"
            else:
                dtype = "TEXT"
            definition = f"{name} {dtype}"
            columns.append(definition)

        create_table = create_table.format(prefix, self.key, ", ".join(columns))

        # return create_table
        row_inserts = []
        for row in self.rows:
            row_inserts.append(str(tuple([i if i is not None else "NULL" for i in row])))
        row_inserts = f"INSERT INTO {prefix}{self.key} VALUES " + ",".join(row_inserts)
        return "begin; {}; {}; commit;".format(create_table, row_inserts)

    def export(self, path):
        """
        exports table to path in format given by path suffix

        path: str or pathlib.Path

        for list of supported formats, see `exporters`
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"expected pathlib.Path, not {type(path)}")

        ext = path.suffix[1:]  # .xlsx --> xlsx

        if ext not in exporters:
            raise TypeError(f"{ext} not in list of supported formats\n{list(file_readers.keys())}")

        handler = exporters.get(ext)
        handler(table=self, path=path)

        log.info(f"exported {self.key} to {path}")
        print(f"exported {self.key} to {path}")

    @classmethod
    def head(cls, path, linecount=5, delimiter=None):
        """
        Gets the head of any supported file format.
        """
        return get_headers(path, linecount=linecount, delimiter=delimiter)

    @classmethod
    def import_file(
        cls,
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
    ):
        """
        reads path and imports 1 or more tables as hdf5

        REQUIRED
        --------
        path: pathlib.Path or str
            selection of filereader uses path.suffix.
            See `filereaders`.

        OPTIONAL
        --------
        columns:
            None: (default) All columns will be imported.
            List: only column names from list will be imported (if present in file)
                  e.g. ['A', 'B', 'C', 'D']

                  datatype is detected using Datatypes.guess(...)
                  You can try it out with:
                  >> from tablite.datatypes import DataTypes
                  >> DataTypes.guess(['001','100'])
                  [1,100]

                  if the format cannot be achieved the read type is kept.
            Excess column names are ignored.

            HINT: To the head of file use: Table.head(path)

        first_row_has_headers: boolean
            True: (default) first row is used as column names.
            False: integers are used as column names.

        encoding: str. Defaults to None (autodetect)

        start: the first line to be read (default: 0)

        limit: the number of lines to be read from start (default sys.maxint ~ 2**63)

        OPTIONAL FOR EXCEL AND ODS READERS
        ----------------------------------

        sheet: sheet name to import  (applicable to excel- and ods-reader only)
            e.g. 'sheet_1'
            sheets not found excess names are ignored.

        OPTIONAL FOR TEXT READERS
        -------------------------
        guess_datatype: bool
            True: (default) datatypes are guessed using DataTypes.guess(...)
            False: all data is imported as strings.

        newline: newline character (applicable to text_reader only)
            str: '\n' (default) or '\r\n'

        text_qualifier: character (applicable to text_reader only)
            None: No text qualifier is used.
            str: " or '

        delimiter: character (applicable to text_reader only)
            None: file suffix is used to determine field delimiter:
                .txt: "|"
                .csv: ",",
                .ssv: ";"
                .tsv: "\t" (tab)

        strip_leading_and_tailing_whitespace: bool:
            True: default

        text_escape_openings: (applicable to text_reader only)
            None: default
            str: list of characters such as ([{

        text_escape_closures: (applicable to text_reader only)
            None: default
            str: list of characters such as }])

        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"expected pathlib.Path, got {type(path)}")
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        if not isinstance(start, int) or not 0 <= start <= sys.maxsize:
            raise ValueError(f"start {start} not in range(0,{sys.maxsize})")

        if not isinstance(limit, int) or not 0 < limit <= sys.maxsize:
            raise ValueError(f"limit {limit} not in range(0,{sys.maxsize})")

        import_as = path.suffix
        if import_as.startswith("."):
            import_as = import_as[1:]

        reader = file_readers.get(import_as, None)
        if reader is None:
            L = "\n\t".join(list(file_readers.keys()))
            raise ValueError(
                f"{import_as} is not in list of supported reader. Here is the list of supported formats:{L}"
            )

        if not isinstance(first_row_has_headers, bool):
            raise TypeError("first_row_has_headers is not bool")

        additional_configs = {}
        if reader == text_reader:
            # here we inject tqdm, if tqdm is not provided, use generic iterator
            additional_configs["tqdm"] = tqdm if tqdm is not None else iter

            if path.stat().st_size == 0:
                return Table()  # NO DATA: EMPTY TABLE.

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

        if reader == excel_reader:
            # config = path, first_row_has_headers, sheet, columns, start, limit
            config = {
                "path": str(path),
                "first_row_has_headers": first_row_has_headers,
                "sheet": sheet,
                "columns": columns,
                "start": start,
                "limit": limit,
                "filesize": path.stat().st_size,  # if file length changes - re-import.
            }

        if reader == ods_reader:
            # path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize,
            config = {
                "path": str(path),
                "first_row_has_headers": first_row_has_headers,
                "sheet": sheet,
                "columns": columns,
                "start": start,
                "limit": limit,
                "filesize": path.stat().st_size,  # if file length changes - re-import.
            }

        # At this point the import seems valid.
        # Now we check if the file already has been imported.

        # publish the settings
        logging.info("import config:\n" + "\n".join(f"{k}:{v}" for k, v in config.items()))
        jsn_str = json.dumps(config)
        for table_key, jsnb in mem.get_imported_tables().items():
            if jsn_str == jsnb:
                return Table.load(mem.path, table_key)  # table already imported.
        # not returned yet? Then it's an import job:
        t = reader(**config, **additional_configs)
        mem.set_config(t.group, jsn_str)
        if t.save is False:
            raise AttributeError("filereader should set table.save = True to avoid repeated imports")
        return t

    def _filter(self, expression):
        """
        filters based on an expression, such as:

            "all((A==B, C!=4, 200<D))"

        which is interpreted using python's compiler to:

            def _f(A,B,C,D):
                return all((A==B, C!=4, 200<D))
        """
        if not isinstance(expression, str):
            raise TypeError
        try:
            _f = expression_interpreter(expression, self.columns)
        except Exception as e:
            raise ValueError(f"Expression could not be compiled: {expression}:\n{e}")

        req_columns = [i for i in self.columns if i in expression]
        bitmap = [bool(_f(*r)) for r in self.__getitem__(*req_columns).rows]
        inverse_bitmap = [not i for i in bitmap]

        if len(self) * len(self.columns) < SINGLE_PROCESSING_LIMIT:
            true, false = Table(), Table()
            for col_name in self.columns:
                data = self[col_name][:]
                true[col_name] = list(itertools.compress(data, bitmap))
                false[col_name] = list(itertools.compress(data, inverse_bitmap))
            return true, false
        else:
            mask = np.array(bitmap, dtype=bool)
            return self._mp_compress(mask), self._mp_compress(np.invert(mask))  # true, false

    def filter(self, expressions, filter_type="all", tqdm=_tqdm):
        """
        enables filtering across columns for multiple criteria.

        expressions:

            str: Expression that can be compiled and executed row by row.
                exampLe: "all((A==B and C!=4 and 200<D))"

            list of dicts: (example):

                L = [
                    {'column1':'A', 'criteria': "==", 'column2': 'B'},
                    {'column1':'C', 'criteria': "!=", "value2": '4'},
                    {'value1': 200, 'criteria': "<", column2: 'D' }
                ]

            accepted dictionary keys: 'column1', 'column2', 'criteria', 'value1', 'value2'

        filter_type: 'all' or 'any'
        """
        if isinstance(expressions, str):
            return self._filter(expressions)

        if not isinstance(expressions, list):
            raise TypeError

        for expression in expressions:
            if not isinstance(expression, dict):
                raise TypeError(f"invalid expression: {expression}")
            if not len(expression) == 3:
                raise ValueError(f"expected 3 items, got {expression}")
            x = {"column1", "column2", "criteria", "value1", "value2"}
            if not set(expression.keys()).issubset(x):
                raise ValueError(f"got unknown key: {set(expression.keys()).difference(x)}")
            if expression["criteria"] not in filter_ops:
                raise ValueError(f"criteria missing from {expression}")

            c1 = expression.get("column1", None)
            if c1 is not None and c1 not in self.columns:
                raise ValueError(f"no such column: {c1}")
            v1 = expression.get("value1", None)
            if v1 is not None and c1 is not None:
                raise ValueError("filter can only take 1 left expr element. Got 2.")

            c2 = expression.get("column2", None)
            if c2 is not None and c2 not in self.columns:
                raise ValueError(f"no such column: {c2}")
            v2 = expression.get("value2", None)
            if v2 is not None and c2 is not None:
                raise ValueError("filter can only take 1 right expression element. Got 2.")

        if not isinstance(filter_type, str):
            raise TypeError()
        if filter_type not in {"all", "any"}:
            raise ValueError(f"filter_type: {filter_type} not in ['all', 'any']")

        # the results are to be gathered here:
        arr = np.zeros(shape=(len(expressions), len(self)), dtype=bool)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        _ = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)

        # the task manager enables evaluation of a column per core,
        # which is assembled in the shared array.
        max_task_size = math.floor(
            SINGLE_PROCESSING_LIMIT / len(self.columns)
        )  # 1 million fields per core (best guess!)

        filter_tasks = []
        for ix, expression in enumerate(expressions):
            for step in range(0, len(self), max_task_size):
                config = {
                    "table_key": self.key,
                    "expression": expression,
                    "shm_name": shm.name,
                    "shm_index": ix,
                    "shm_shape": arr.shape,
                    "slice_": slice(step, min(step + max_task_size, len(self))),
                }
                task = Task(f=filter_evaluation_task, **config)
                filter_tasks.append(task)

        merge_tasks = []
        for step in range(0, len(self), max_task_size):
            config = {
                "table_key": self.key,
                "true_key": mem.new_id("/table"),
                "false_key": mem.new_id("/table"),
                "shm_name": shm.name,
                "shm_shape": arr.shape,
                "slice_": slice(step, min(step + max_task_size, len(self)), 1),
                "filter_type": filter_type,
            }
            task = Task(f=filter_merge_task, **config)
            merge_tasks.append(task)

        n_cpus = min(
            max(len(filter_tasks), len(merge_tasks)), psutil.cpu_count()
        )  # revise for case where memory footprint is limited to include zero subprocesses.

        with tqdm(total=len(filter_tasks) + len(merge_tasks), desc="filter") as pbar:
            with TaskManager(n_cpus) as tm:
                # EVALUATE
                errs = tm.execute(filter_tasks, pbar=pbar)
                # tm.execute returns the tasks with results, but we don't
                # really care as the result is in the result array.
                if any(errs):
                    raise Exception(errs)
                # MERGE RESULTS
                errs = tm.execute(merge_tasks, pbar=pbar)
                # tm.execute returns the tasks with results, but we don't
                # really care as the result is in the result array.
                if any(errs):
                    raise Exception(errs)

        true = Table()
        true.add_columns(*self.columns)
        false = true.copy()

        for task in merge_tasks:
            tmp_true = Table.load(mem.path, key=task.kwargs["true_key"])
            if len(tmp_true):
                true += tmp_true
            else:
                pass

            tmp_false = Table.load(mem.path, key=task.kwargs["false_key"])
            if len(tmp_false):
                false += tmp_false
            else:
                pass
        return true, false

    def sort_index(self, sort_mode="excel", tqdm=_tqdm, pbar=None, **kwargs):
        """
        helper for methods `sort` and `is_sorted`

        param: sort_mode: str: "alphanumeric", "unix", or, "excel" (default)
        param: **kwargs: sort criteria. See Table.sort()
        """
        logging.info("Table.sort_index running 1 core")  # This is single core code.

        if not isinstance(kwargs, dict):
            raise ValueError("Expected keyword arguments, did you forget the ** in front of your dict?")
        if not kwargs:
            kwargs = {c: False for c in self.columns}

        for k, v in kwargs.items():
            if k not in self.columns:
                raise ValueError(f"no column {k}")
            if not isinstance(v, bool):
                raise ValueError(f"{k} was mapped to {v} - a non-boolean")

        if sort_mode not in sortation.modes:
            raise ValueError(f"{sort_mode} not in list of sort_modes: {list(sortation.Sortable.modes.modes)}")

        rank = {i: tuple() for i in range(len(self))}  # create index and empty tuple for sortation.

        _pbar = tqdm(total=len(kwargs.items()), desc="creating sort index") if pbar is None else pbar

        for key, reverse in kwargs.items():
            col = self._columns[key][:]
            col = col.tolist() if isinstance(col, np.ndarray) else col
            ranks = sortation.rank(values=set(col), reverse=reverse, mode=sort_mode)
            assert isinstance(ranks, dict)
            for ix, v in enumerate(col):
                rank[ix] += (ranks[v],)  # add tuple

            _pbar.update(1)

        new_order = [(r, i) for i, r in rank.items()]  # tuples are listed and sort...
        rank.clear()  # free memory.
        new_order.sort()
        sorted_index = [i for _, i in new_order]  # new index is extracted.
        new_order.clear()
        return sorted_index

    def reindex(self, index):
        """
        index: list of integers that declare sort order.

        Examples:

            Table:  ['a','b','c','d','e','f','g','h']
            index:  [0,2,4,6]
            result: ['b','d','f','h']

            Table:  ['a','b','c','d','e','f','g','h']
            index:  [0,2,4,6,1,3,5,7]
            result: ['a','c','e','g','b','d','f','h']

        """
        if index is not None:
            if not isinstance(index, list):
                raise TypeError
            if max(index) >= len(self):
                raise IndexError("index out of range: max(index) > len(self)")
            if min(index) < -len(self):
                raise IndexError("index out of range: min(index) < -len(self)")
            if not all(isinstance(i, int) for i in index):
                raise TypeError

        if (
            len(self) * len(self.columns) < SINGLE_PROCESSING_LIMIT
        ):  # the task is so small that multiprocessing doesn't make sense.
            t = Table()
            for col_name, col in self._columns.items():  # this LOOP can be done with TaskManager
                data = list(col[:])
                t.add_column(col_name, data=[data[ix] for ix in index])
            return t

        else:
            arr = np.zeros(shape=(len(index),), dtype=np.int64)
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)  # the co_processors will read this.
            sort_index = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            sort_index[:] = index

            tasks = []
            columns_refs = {}
            for name in self.columns:
                col = self[name]
                columns_refs[name] = d_key = mem.new_id("/column")
                tasks.append(
                    Task(
                        indexing_task,
                        source_key=col.key,
                        destination_key=d_key,
                        shm_name_for_sort_index=shm.name,
                        shape=arr.shape,
                    )
                )

            with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
                errs = tm.execute(tasks)
                if any(errs):
                    msg = "\n".join(errs)
                    raise Exception(f"multiprocessing error:{msg}")

            table_key = mem.new_id("/table")
            mem.create_table(key=table_key, columns=columns_refs)

            shm.close()
            shm.unlink()
            t = Table.load(path=mem.path, key=table_key)
            return t

    def drop_duplicates(self, *args):
        """
        removes duplicate rows based on column names

        args: (optional) column_names
        if no args, all columns are used.
        """
        if not args:
            args = self.columns
        index = [min(v) for v in self.index(*args).values()]
        return self.reindex(index)

    def sort(self, sort_mode="excel", **kwargs):
        """Perform multi-pass sorting with precedence given order of column names.
        sort_mode: str: "alphanumeric", "unix", or, "excel"
        kwargs:
            keys: columns,
            values: 'reverse' as boolean.

        examples:
        Table.sort('A'=False) means sort by 'A' in ascending order.
        Table.sort('A'=True, 'B'=False) means sort 'A' in descending order, then (2nd priority)
        sort B in ascending order.
        """
        if (
            len(self) * len(self.columns) < SINGLE_PROCESSING_LIMIT
        ):  # the task is so small that multiprocessing doesn't make sense.
            sorted_index = self.sort_index(sort_mode=sort_mode, **kwargs)

            t = Table()
            for col_name, col in self._columns.items():
                data = list(col[:])
                t.add_column(col_name, data=[data[ix] for ix in sorted_index])
            return t
        else:
            arr = np.zeros(shape=(len(self),), dtype=np.int64)
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)  # the co_processors will read this.
            sort_index = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            sort_index[:] = self.sort_index(sort_mode=sort_mode, **kwargs)

            tasks = []
            columns_refs = {}
            for name in self.columns:
                col = self[name]
                columns_refs[name] = d_key = mem.new_id("/column")
                tasks.append(
                    Task(
                        indexing_task,
                        source_key=col.key,
                        destination_key=d_key,
                        shm_name_for_sort_index=shm.name,
                        shape=arr.shape,
                    )
                )

            with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
                errs = tm.execute(tasks)
                if any(errs):
                    msg = "\n".join(errs)
                    raise Exception(f"multiprocessing error:{msg}")

            table_key = mem.new_id("/table")
            mem.create_table(key=table_key, columns=columns_refs)

            shm.close()
            shm.unlink()
            t = Table.load(path=mem.path, key=table_key)
            return t

    def is_sorted(self, **kwargs):
        """Performs multi-pass sorting check with precedence given order of column names.
        **kwargs: optional: sort criteria. See Table.sort()
        :return bool
        """
        logging.info("Table.is_sorted running 1 core")  # TODO: This is single core code.
        sorted_index = self.sort_index(**kwargs)
        if any(ix != i for ix, i in enumerate(sorted_index)):
            return False
        return True

    def _mp_compress(self, mask):
        """
        helper for `any` and `all` that performs compression of the table self according to mask
        using multiprocessing.
        """
        arr = np.zeros(shape=(len(self),), dtype=bool)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)  # the co_processors will read this.
        compresssion_mask = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        compresssion_mask[:] = mask

        t = Table()
        tasks = []
        columns_refs = {}
        for name in self.columns:
            col = self[name]
            d_key = mem.new_id("/column")
            columns_refs[name] = d_key
            t = Task(compress_task, source_key=col.key, destination_key=d_key, shm_index_name=shm.name, shape=arr.shape)
            tasks.append(t)

        with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
            results = tm.execute(tasks)
            if any(r is not None for r in results):
                for r in results:
                    print(r)
                raise Exception("!")

        with h5py.File(mem.path, "r+") as h5:
            table_key = mem.new_id("/table")
            dset = h5.create_dataset(name=f"/table/{table_key}", dtype=h5py.Empty("f"))
            dset.attrs["columns"] = json.dumps(columns_refs)
            dset.attrs["saved"] = False

        shm.close()
        shm.unlink()
        t = Table.load(path=mem.path, key=table_key)
        return t

    def all(self, **kwargs):
        """
        returns Table for rows where ALL kwargs match
        :param kwargs: dictionary with headers and values / boolean callable

        Examples:

            t = Table()
            t['a'] = [1,2,3,4]
            t['b'] = [10,20,30,40]

            def f(x):
                return x == 4
            def g(x):
                return x < 20

            t2 = t.any( **{"a":f, "b":g})
            assert [r for r in t2.rows] == [[1, 10], [4, 40]]

            t2 = t.any(a=f,b=g)
            assert [r for r in t2.rows] == [[1, 10], [4, 40]]

            def h(x):
                return x>=2

            def i(x):
                return x<=30

            t2 = t.all(a=h,b=i)
            assert [r for r in t2.rows] == [[2,20], [3, 30]]


        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you forget to add the ** in front of your dict?")
        if not all(k in self.columns for k in kwargs):
            raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in self.columns]}")

        ixs = None
        for k, v in kwargs.items():
            col = self._columns[k][:]
            if ixs is None:  # first header generates base set.
                if callable(v):
                    ix2 = {ix for ix, i in enumerate(col) if v(i)}
                else:
                    ix2 = {ix for ix, i in enumerate(col) if v == i}

            else:  # remaining headers reduce the base set.
                if callable(v):
                    ix2 = {ix for ix in ixs if v(col[ix])}
                else:
                    ix2 = {ix for ix in ixs if v == col[ix]}

            if not isinstance(ixs, set):
                ixs = ix2
            else:
                ixs = ixs.intersection(ix2)

            if not ixs:  # There are no matches.
                break

        if len(self) * len(self.columns) < SINGLE_PROCESSING_LIMIT:
            t = Table()
            for col_name in self.columns:
                data = self[col_name][:]
                t[col_name] = [data[i] for i in ixs]
            return t
        else:
            mask = np.array([True if i in ixs else False for i in range(len(self))], dtype=bool)
            return self._mp_compress(mask)

    def drop(self, *args):
        """
        removes all rows where args are present.

        Exmaple:
        >>> t = Table()
        >>> t['A'] = [1,2,3,None]
        >>> t['B'] = [None,2,3,4]
        >>> t2 = t.drop(None)
        >>> t2['A'][:], t2['B'][:]
        ([2,3], [2,3])

        """
        if not args:
            raise ValueError("What to drop? None? np.nan? ")
        d = {n: lambda x: x not in set(args) for n in self.columns}
        return self.all(**d)

    def replace(self, target, replacement):
        """
        Finds and replaces all instances of `target` with `replacement` across all Columns

        See Column.replace(target, replacement) for replacement in specific columns.
        """
        for _, col in self._columns.items():
            col.replace(target, replacement)

    def any(self, **kwargs):
        """
        returns Table for rows where ANY kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you forget to add the ** in front of your dict?")

        ixs = set()
        for k, v in kwargs.items():
            col = self._columns[k][:]
            if callable(v):
                ix2 = {ix for ix, r in enumerate(col) if v(r)}
            else:
                ix2 = {ix for ix, r in enumerate(col) if v == r}
            ixs.update(ix2)

        if len(self) * len(self.columns) < SINGLE_PROCESSING_LIMIT:
            t = Table()
            for col_name in self.columns:
                data = self[col_name][:]
                t[col_name] = [data[i] for i in ixs]
            return t
        else:
            mask = np.array([i in ixs for i in range(len(self))], dtype=bool)
            return self._mp_compress(mask)

    def groupby(self, keys, functions, tqdm=_tqdm, pbar=None):  # TODO: This is single core code.
        """
        keys: column names for grouping.
        functions: [optional] list of column names and group functions (See GroupyBy class)
        returns: table

        Example:

        t = Table()
        t.add_column('A', data=[1, 1, 2, 2, 3, 3] * 2)
        t.add_column('B', data=[1, 2, 3, 4, 5, 6] * 2)
        t.add_column('C', data=[6, 5, 4, 3, 2, 1] * 2)

        t.show()
        # +=====+=====+=====+
        # |  A  |  B  |  C  |
        # | int | int | int |
        # +-----+-----+-----+
        # |    1|    1|    6|
        # |    1|    2|    5|
        # |    2|    3|    4|
        # |    2|    4|    3|
        # |    3|    5|    2|
        # |    3|    6|    1|
        # |    1|    1|    6|
        # |    1|    2|    5|
        # |    2|    3|    4|
        # |    2|    4|    3|
        # |    3|    5|    2|
        # |    3|    6|    1|
        # +=====+=====+=====+

        g = t.groupby(keys=['A', 'C'], functions=[('B', gb.sum)])
        g.show()
        # +===+===+===+======+
        # | # | A | C |Sum(B)|
        # |row|int|int| int  |
        # +---+---+---+------+
        # |0  |  1|  6|     2|
        # |1  |  1|  5|     4|
        # |2  |  2|  4|     6|
        # |3  |  2|  3|     8|
        # |4  |  3|  2|    10|
        # |5  |  3|  1|    12|
        # +===+===+===+======+

        Cheat sheet:

        # list of unique values
        >>> g1 = t.groupby(keys=['A'], functions=[])
        >>> g1['A'][:]
        [1,2,3]

        # alternatively:
        >>> t['A'].unique()
        [1,2,3]

        # list of unique values, grouped by longest combination.
        >>> g2 = t.groupby(keys=['A', 'B'], functions=[])
        >>> g2['A'][:], g2['B'][:]
        ([1,1,2,2,3,3], [1,2,3,4,5,6])

        # alternatively:
        >>> list(zip(*t.index('A', 'B').keys()))
        [(1,1,2,2,3,3) (1,2,3,4,5,6)]

        # A key (unique values) and count hereof.
        >>> g3 = t.groupby(keys=['A'], functions=[('A', gb.count)])
        >>> g3['A'][:], g3['Count(A)'][:]
        ([1,2,3], [4,4,4])

        # alternatively:
        >>> t['A'].histogram()
        ([1,2,3], [4,4,4])

        for more exmaples see:
            https://github.com/root-11/tablite/blob/master/tests/test_groupby.py

        """
        if not isinstance(keys, list):
            raise TypeError("expected keys as a list of column names")

        if not keys:
            raise ValueError("Keys missing.")

        if len(set(keys)) != len(keys):
            duplicates = [k for k in keys if keys.count(k) > 1]
            s = "" if len(duplicates) > 1 else "s"
            raise ValueError(f"duplicate key{s} found across rows and columns: {duplicates}")

        if not isinstance(functions, list):
            raise TypeError(f"Expected functions to be a list of tuples. Got {type(functions)}")

        if not keys + functions:
            raise ValueError("No keys or functions?")

        if not all(len(i) == 2 for i in functions):
            raise ValueError(f"Expected each tuple in functions to be of length 2. \nGot {functions}")

        if not all(isinstance(a, str) for a, _ in functions):
            L = [(a, type(a)) for a, _ in functions if not isinstance(a, str)]
            raise ValueError(f"Expected column names in functions to be strings. Found: {L}")

        if not all(issubclass(b, GroupbyFunction) and b in GroupBy.functions for _, b in functions):
            L = [b for _, b in functions if b not in GroupBy._functions]
            if len(L) == 1:
                singular = f"function {L[0]} is not in GroupBy.functions"
                raise ValueError(singular)
            else:
                plural = f"the functions {L} are not in GroupBy.functions"
                raise ValueError(plural)

        # only keys will produce unique values for each key group.
        if keys and not functions:
            cols = list(zip(*self.index(*keys)))
            result = Table()

            pbar = tqdm(total=len(keys), desc="groupby") if pbar is None else pbar

            for col_name, col in zip(keys, cols):
                result[col_name] = col

                pbar.update(1)
            return result

        # grouping is required...
        # 1. Aggregate data.
        aggregation_functions = defaultdict(dict)
        cols = keys + [col_name for col_name, _ in functions]
        seen, L = set(), []
        for c in cols:  # maintains order of appearance.
            if c not in seen:
                seen.add(c)
                L.append(c)

        # there's a table of values.
        data = self.__getitem__(*L)
        if isinstance(data, Column):
            tbl = Table()
            tbl[L[0]] = data
        else:
            tbl = data

        pbar = tqdm(desc="groupby", total=len(tbl)) if pbar is None else pbar

        for row in tbl.rows:
            d = {col_name: value for col_name, value in zip(L, row)}
            key = tuple([d[k] for k in keys])
            agg_functions = aggregation_functions.get(key)
            if not agg_functions:
                aggregation_functions[key] = agg_functions = [(col_name, f()) for col_name, f in functions]
            for col_name, f in agg_functions:
                f.update(d[col_name])

            pbar.update(1)

        # 2. make dense table.
        cols = [[] for _ in cols]
        for key_tuple, funcs in aggregation_functions.items():
            for ix, key_value in enumerate(key_tuple):
                cols[ix].append(key_value)
            for ix, (_, f) in enumerate(funcs, start=len(keys)):
                cols[ix].append(f.value)

        new_names = keys + [f"{f.__name__}({col_name})" for col_name, f in functions]
        result = Table()
        for ix, (col_name, data) in enumerate(zip(new_names, cols)):
            revised_name = unique_name(col_name, result.columns)
            result[revised_name] = data
        return result

    def pivot(self, rows, columns, functions, values_as_rows=True, tqdm=_tqdm, pbar=None):
        """
        param: rows: column names to keep as rows
        param: columns: column names to keep as columns
        param: functions: aggregation functions from the Groupby class as

        example:

        t.show()
        # +=====+=====+=====+
        # |  A  |  B  |  C  |
        # | int | int | int |
        # +-----+-----+-----+
        # |    1|    1|    6|
        # |    1|    2|    5|
        # |    2|    3|    4|
        # |    2|    4|    3|
        # |    3|    5|    2|
        # |    3|    6|    1|
        # |    1|    1|    6|
        # |    1|    2|    5|
        # |    2|    3|    4|
        # |    2|    4|    3|
        # |    3|    5|    2|
        # |    3|    6|    1|
        # +=====+=====+=====+

        t2 = t.pivot(rows=['C'], columns=['A'], functions=[('B', gb.sum)])
        t2.show()
        # +===+===+========+=====+=====+=====+
        # | # | C |function|(A=1)|(A=2)|(A=3)|
        # |row|int|  str   |mixed|mixed|mixed|
        # +---+---+--------+-----+-----+-----+
        # |0  |  6|Sum(B)  |    2|None |None |
        # |1  |  5|Sum(B)  |    4|None |None |
        # |2  |  4|Sum(B)  |None |    6|None |
        # |3  |  3|Sum(B)  |None |    8|None |
        # |4  |  2|Sum(B)  |None |None |   10|
        # |5  |  1|Sum(B)  |None |None |   12|
        # +===+===+========+=====+=====+=====+

        """
        if isinstance(rows, str):
            rows = [rows]
        if not all(isinstance(i, str) for i in rows):
            raise TypeError(
                f"Expected rows as a list of column names, not {[i for i in rows if not isinstance(i,str)]}"
            )

        if isinstance(columns, str):
            columns = [columns]
        if not all(isinstance(i, str) for i in columns):
            raise TypeError(
                f"Expected columns as a list of column names, not {[i for i in columns if not isinstance(i, str)]}"
            )

        if not isinstance(values_as_rows, bool):
            raise TypeError(f"expected sum_on_rows as boolean, not {type(values_as_rows)}")

        keys = rows + columns
        assert isinstance(keys, list)

        extra_steps = 2

        if pbar is None:
            total = extra_steps

            if len(functions) == 0:
                total = total + len(keys)
            else:
                total = total + len(self)

            pbar = tqdm(total=total, desc="pivot")

        grpby = self.groupby(keys, functions, tqdm=tqdm, pbar=pbar)

        if len(grpby) == 0:  # return empty table. This must be a test?
            pbar.update(extra_steps)
            return Table()

        # split keys to determine grid dimensions
        row_key_index = {}
        col_key_index = {}

        r = len(rows)
        c = len(columns)
        g = len(functions)

        records = defaultdict(dict)

        for row in grpby.rows:
            row_key = tuple(row[:r])
            col_key = tuple(row[r : r + c])
            func_key = tuple(row[r + c :])

            if row_key not in row_key_index:
                row_key_index[row_key] = len(row_key_index)  # Y

            if col_key not in col_key_index:
                col_key_index[col_key] = len(col_key_index)  # X

            rix = row_key_index[row_key]
            cix = col_key_index[col_key]
            if cix in records:
                if rix in records[cix]:
                    raise ValueError("this should be empty.")
            records[cix][rix] = func_key

        pbar.update(1)
        result = Table()

        if values_as_rows:  # ---> leads to more rows.
            # first create all columns left to right

            n = r + 1  # rows keys + 1 col for function values.
            cols = [[] for _ in range(n)]
            for row, ix in row_key_index.items():
                for (col_name, f) in functions:
                    cols[-1].append(f"{f.__name__}({col_name})")
                    for col_ix, v in enumerate(row):
                        cols[col_ix].append(v)

            for col_name, values in zip(rows + ["function"], cols):
                col_name = unique_name(col_name, result.columns)
                result[col_name] = values
            col_length = len(cols[0])
            cols.clear()

            # then populate the sparse matrix.
            for col_key, c in col_key_index.items():
                col_name = "(" + ",".join([f"{col_name}={value}" for col_name, value in zip(columns, col_key)]) + ")"
                col_name = unique_name(col_name, result.columns)
                L = [None for _ in range(col_length)]
                for r, funcs in records[c].items():
                    for ix, f in enumerate(funcs):
                        L[g * r + ix] = f
                result[col_name] = L

        else:  # ---> leads to more columns.
            n = r
            cols = [[] for _ in range(n)]
            for row in row_key_index:
                for col_ix, v in enumerate(row):
                    cols[col_ix].append(v)  # write key columns.

            for col_name, values in zip(rows, cols):
                result[col_name] = values

            col_length = len(row_key_index)

            # now populate the sparse matrix.
            for col_key, c in col_key_index.items():  # select column.
                cols, names = [], []

                for f, v in zip(functions, func_key):
                    agg_col, func = f
                    terms = ",".join([agg_col] + [f"{col_name}={value}" for col_name, value in zip(columns, col_key)])
                    col_name = f"{func.__name__}({terms})"
                    col_name = unique_name(col_name, result.columns)
                    names.append(col_name)
                    cols.append([None for _ in range(col_length)])
                for r, funcs in records[c].items():
                    for ix, f in enumerate(funcs):
                        cols[ix][r] = f
                for name, col in zip(names, cols):
                    result[name] = col

        pbar.update(1)

        return result

    def _join_type_check(self, other, left_keys, right_keys, left_columns, right_columns):
        if not isinstance(other, Table):
            raise TypeError(f"other expected other to be type Table, not {type(other)}")

        if not isinstance(left_keys, list) and all(isinstance(k, str) for k in left_keys):
            raise TypeError(f"Expected keys as list of strings, not {type(left_keys)}")
        if not isinstance(right_keys, list) and all(isinstance(k, str) for k in right_keys):
            raise TypeError(f"Expected keys as list of strings, not {type(right_keys)}")

        if any(key not in self.columns for key in left_keys):
            raise ValueError(f"left key(s) not found: {[k for k in left_keys if k not in self.columns]}")
        if any(key not in other.columns for key in right_keys):
            raise ValueError(f"right key(s) not found: {[k for k in right_keys if k not in other.columns]}")

        if len(left_keys) != len(right_keys):
            raise ValueError(f"Keys do not have same length: \n{left_keys}, \n{right_keys}")

        for L, R in zip(left_keys, right_keys):
            Lcol, Rcol = self[L], other[R]
            if not set(Lcol.types()).intersection(set(Rcol.types())):
                left_types = tuple(t.__name__ for t in list(Lcol.types().keys()))
                right_types = tuple(t.__name__ for t in list(Rcol.types().keys()))
                raise TypeError(f"Type mismatch: Left key '{L}' {left_types} will never match right keys {right_types}")

        if not isinstance(left_columns, list) or not left_columns:
            raise TypeError("left_columns (list of strings) are required")
        if any(column not in self.columns for column in left_columns):
            raise ValueError(f"Column not found: {[c for c in left_columns if c not in self.columns]}")

        if not isinstance(right_columns, list) or not right_columns:
            raise TypeError("right_columns (list or strings) are required")
        if any(column not in other.columns for column in right_columns):
            raise ValueError(f"Column not found: {[c for c in right_columns if c not in other.columns]}")
        # Input is now guaranteed to be valid.

    def join(self, other, left_keys, right_keys, left_columns, right_columns, kind="inner", tqdm=_tqdm, pbar=None):
        """
        short-cut for all join functions.
        kind: 'inner', 'left', 'outer', 'cross'
        """
        kinds = {
            "inner": self.inner_join,
            "left": self.left_join,
            "outer": self.outer_join,
            "cross": self.cross_join,
        }
        if kind not in kinds:
            raise ValueError(f"join type unknown: {kind}")
        f = kinds.get(kind, None)
        return f(other, left_keys, right_keys, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def _sp_join(self, other, LEFT, RIGHT, left_columns, right_columns, tqdm=_tqdm, pbar=None):
        """
        private helper for single processing join
        """
        result = Table()

        if pbar is None:
            total = len(left_columns) + len(right_columns)
            pbar = tqdm(total=total, desc="join")

        for col_name in left_columns:
            col_data = self[col_name][:]
            result[col_name] = [col_data[k] if k is not None else None for k in LEFT]
            pbar.update(1)
        for col_name in right_columns:
            col_data = other[col_name][:]
            revised_name = unique_name(col_name, result.columns)
            result[revised_name] = [col_data[k] if k is not None else None for k in RIGHT]
            pbar.update(1)
        return result

    def _mp_join(self, other, LEFT, RIGHT, left_columns, right_columns, tqdm=_tqdm, pbar=None):
        """
        private helper for multiprocessing join
        """
        left_arr = np.zeros(shape=(len(LEFT)), dtype=np.int64)
        left_shm = shared_memory.SharedMemory(create=True, size=left_arr.nbytes)  # the co_processors will read this.
        left_index = np.ndarray(left_arr.shape, dtype=left_arr.dtype, buffer=left_shm.buf)
        left_index[:] = LEFT

        right_arr = np.zeros(shape=(len(RIGHT)), dtype=np.int64)
        right_shm = shared_memory.SharedMemory(create=True, size=right_arr.nbytes)  # the co_processors will read this.
        right_index = np.ndarray(right_arr.shape, dtype=right_arr.dtype, buffer=right_shm.buf)
        right_index[:] = RIGHT

        tasks = []
        columns_refs = {}
        for name in left_columns:
            col = self[name]
            columns_refs[name] = d_key = mem.new_id("/column")
            tasks.append(
                Task(
                    indexing_task,
                    source_key=col.key,
                    destination_key=d_key,
                    shm_name_for_sort_index=left_shm.name,
                    shape=left_arr.shape,
                )
            )

        for name in right_columns:
            col = other[name]
            columns_refs[name] = d_key = mem.new_id("/column")
            tasks.append(
                Task(
                    indexing_task,
                    source_key=col.key,
                    destination_key=d_key,
                    shm_name_for_sort_index=right_shm.name,
                    shape=right_arr.shape,
                )
            )

        if pbar is None:
            total = len(left_columns) + len(right_columns)
            pbar = tqdm(total=total, desc="join")

        with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
            results = tm.execute(tasks, tqdm=tqdm, pbar=pbar)

            if any(i is not None for i in results):
                for err in results:
                    if err is not None:
                        print(err)
                raise Exception("multiprocessing error.")

        with h5py.File(mem.path, "r+") as h5:
            table_key = mem.new_id("/table")
            dset = h5.create_dataset(name=f"/table/{table_key}", dtype=h5py.Empty("f"))
            dset.attrs["columns"] = json.dumps(columns_refs)
            dset.attrs["saved"] = False

        left_shm.close()
        left_shm.unlink()
        right_shm.close()
        right_shm.unlink()

        t = Table.load(path=mem.path, key=table_key)
        return t

    def left_join(
        self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None
    ):  # TODO: This is single core code.
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param left_columns: list of left columns to retain, if None, all are retained.
        :param right_columns: list of right columns to retain, if None, all are retained.
        :return: new Table
        Example:
        SQL:   SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
        Tablite: left_join = numbers.left_join(
            letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter']
        )
        """
        if left_columns is None:
            left_columns = list(self.columns)
        if right_columns is None:
            right_columns = list(other.columns)

        self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        left_index = self.index(*left_keys)
        right_index = other.index(*right_keys)
        LEFT, RIGHT = [], []
        for left_key, left_ixs in left_index.items():
            right_ixs = right_index.get(left_key, (None,))
            for left_ix in left_ixs:
                for right_ix in right_ixs:
                    LEFT.append(left_ix)
                    RIGHT.append(right_ix)

        if len(LEFT) * len(left_columns + right_columns) < SINGLE_PROCESSING_LIMIT:
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        else:  # use multi processing
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def inner_join(
        self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None
    ):  # TODO: This is single core code.
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param left_columns: list of left columns to retain, if None, all are retained.
        :param right_columns: list of right columns to retain, if None, all are retained.
        :return: new Table
        Example:
        SQL:   SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
        Tablite: inner_join = numbers.inner_join(
            letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter']
            )
        """
        if left_columns is None:
            left_columns = list(self.columns)
        if right_columns is None:
            right_columns = list(other.columns)

        self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        left_index = self.index(*left_keys)
        right_index = other.index(*right_keys)
        LEFT, RIGHT = [], []
        for left_key, left_ixs in left_index.items():
            right_ixs = right_index.get(left_key, None)
            if right_ixs is None:
                continue
            for left_ix in left_ixs:
                for right_ix in right_ixs:
                    LEFT.append(left_ix)
                    RIGHT.append(right_ix)

        if len(LEFT) * len(left_columns + right_columns) < SINGLE_PROCESSING_LIMIT:
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        else:  # use multi processing
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def outer_join(
        self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None
    ):  # TODO: This is single core code.
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param left_columns: list of left columns to retain, if None, all are retained.
        :param right_columns: list of right columns to retain, if None, all are retained.
        :return: new Table
        Example:
        SQL:   SELECT number, letter FROM numbers OUTER JOIN letters ON numbers.colour == letters.color
        Tablite: outer_join = numbers.outer_join(
            letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter']
            )
        """
        if left_columns is None:
            left_columns = list(self.columns)
        if right_columns is None:
            right_columns = list(other.columns)

        self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        left_index = self.index(*left_keys)
        right_index = other.index(*right_keys)
        LEFT, RIGHT, RIGHT_UNUSED = [], [], set(right_index.keys())
        for left_key, left_ixs in left_index.items():
            right_ixs = right_index.get(left_key, (None,))
            for left_ix in left_ixs:
                for right_ix in right_ixs:
                    LEFT.append(left_ix)
                    RIGHT.append(right_ix)
                    RIGHT_UNUSED.discard(left_key)

        for right_key in RIGHT_UNUSED:
            for right_ix in right_index[right_key]:
                LEFT.append(None)
                RIGHT.append(right_ix)

        if len(LEFT) * len(left_columns + right_columns) < SINGLE_PROCESSING_LIMIT:
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        else:  # use multi processing
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def cross_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
        """
        CROSS JOIN returns the Cartesian product of rows from tables in the join.
        In other words, it will produce rows which combine each row from the first table
        with each row from the second table
        """
        if left_columns is None:
            left_columns = list(self.columns)
        if right_columns is None:
            right_columns = list(other.columns)

        self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        LEFT, RIGHT = zip(*itertools.product(range(len(self)), range(len(other))))
        if len(LEFT) < SINGLE_PROCESSING_LIMIT:
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        else:  # use multi processing
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def lookup(self, other, *criteria, all=True, tqdm=_tqdm):  # TODO: This is single core code.
        """function for looking up values in `other` according to criteria in ascending order.
        :param: other: Table sorted in ascending search order.
        :param: criteria: Each criteria must be a tuple with value comparisons in the form:
            (LEFT, OPERATOR, RIGHT)
        :param: all: boolean: True=ALL, False=Any

        OPERATOR must be a callable that returns a boolean
        LEFT must be a value that the OPERATOR can compare.
        RIGHT must be a value that the OPERATOR can compare.

        Examples:
              ('column A', "==", 'column B')  # comparison of two columns
              ('Date', "<", DataTypes.date(24,12) )  # value from column 'Date' is before 24/12.
              f = lambda L,R: all( ord(L) < ord(R) )  # uses custom function.
              ('text 1', f, 'text 2')
              value from column 'text 1' is compared with value from column 'text 2'
        """
        assert isinstance(self, Table)
        assert isinstance(other, Table)

        all = all
        any = not all

        def not_in(a, b):
            return not operator.contains(str(a), str(b))

        def _in(a, b):
            return operator.contains(str(a), str(b))

        ops = {
            "in": _in,
            "not in": not_in,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "!=": operator.ne,
            "==": operator.eq,
        }

        functions, left_criteria, right_criteria = [], set(), set()

        for left, op, right in criteria:
            left_criteria.add(left)
            right_criteria.add(right)
            if callable(op):
                pass  # it's a custom function.
            else:
                op = ops.get(op, None)
                if not callable(op):
                    raise ValueError(f"{op} not a recognised operator for comparison.")

            functions.append((op, left, right))
        left_columns = [n for n in left_criteria if n in self.columns]
        right_columns = [n for n in right_criteria if n in other.columns]

        results = []
        lru_cache = {}
        left = self.__getitem__(*left_columns)
        if isinstance(left, Column):
            tmp, left = left, Table()
            left[left_columns[0]] = tmp
        right = other.__getitem__(*right_columns)
        if isinstance(right, Column):
            tmp, right = right, Table()
            right[right_columns[0]] = tmp
        assert isinstance(left, Table)
        assert isinstance(right, Table)

        for row1 in tqdm(left.rows, total=self.__len__()):
            row1_tup = tuple(row1)
            row1d = {name: value for name, value in zip(left_columns, row1)}
            row1_hash = hash(row1_tup)

            match_found = True if row1_hash in lru_cache else False

            if not match_found:  # search.
                for row2ix, row2 in enumerate(right.rows):
                    row2d = {name: value for name, value in zip(right_columns, row2)}

                    evaluations = {op(row1d.get(left, left), row2d.get(right, right)) for op, left, right in functions}
                    # The evaluations above does a neat trick:
                    # as L is a dict, L.get(left, L) will return a value
                    # from the columns IF left is a column name. If it isn't
                    # the function will treat left as a value.
                    # The same applies to right.
                    A = all and (False not in evaluations)
                    B = any and True in evaluations
                    if A or B:
                        match_found = True
                        lru_cache[row1_hash] = row2ix
                        break

            if not match_found:  # no match found.
                lru_cache[row1_hash] = None

            results.append(lru_cache[row1_hash])

        result = self.copy()
        if len(self) * len(other.columns) < SINGLE_PROCESSING_LIMIT:
            for col_name in other.columns:
                col_data = other[col_name][:]
                revised_name = unique_name(col_name, result.columns)
                result[revised_name] = [col_data[k] if k is not None else None for k in results]
            return result
        else:
            # 1. create shared memory array.
            right_arr = np.zeros(shape=(len(results)), dtype=np.int64)
            right_shm = shared_memory.SharedMemory(
                create=True, size=right_arr.nbytes
            )  # the co_processors will read this.
            right_index = np.ndarray(right_arr.shape, dtype=right_arr.dtype, buffer=right_shm.buf)
            right_index[:] = results
            # 2. create tasks
            tasks = []
            columns_refs = {}

            for name in other.columns:
                col = other[name]
                columns_refs[name] = d_key = mem.new_id("/column")
                tasks.append(
                    Task(
                        indexing_task,
                        source_key=col.key,
                        destination_key=d_key,
                        shm_name_for_sort_index=right_shm.name,
                        shape=right_arr.shape,
                    )
                )

            # 3. let task manager handle the tasks
            with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
                errs = tm.execute(tasks)
                if any(errs):
                    raise Exception(f"multiprocessing error. {[e for e in errs if e]}")

            # 4. close the share memory and deallocate
            right_shm.close()
            right_shm.unlink()

            # 5. update the result table.
            with h5py.File(mem.path, "r+") as h5:
                dset = h5[f"/table/{result.key}"]
                columns = dset.attrs["columns"]
                columns.update(columns_refs)
                dset.attrs["columns"] = json.dumps(columns)
                dset.attrs["saved"] = False

            # 6. reload the result table
            t = Table.load(path=mem.path, key=result.key)
            return t

    def replace_missing_values(self, *args, **kwargs):
        raise AttributeError("See imputation")

    def imputation(self, targets, missing=None, method="carry forward", sources=None, tqdm=_tqdm):
        """
        In statistics, imputation is the process of replacing missing data with substituted values.

        See more: https://en.wikipedia.org/wiki/Imputation_(statistics)

        Args:
            table (Table): source table.

            targets (str or list of strings): column names to find and
                replace missing values

            missing (any): value to be replaced

            method (str): method to be used for replacement. Options:

                'carry forward':
                    takes the previous value, and carries forward into fields
                    where values are missing.
                    +: quick. Realistic on time series.
                    -: Can produce strange outliers.

                'mean':
                    calculates the column mean (exclude `missing`) and copies
                    the mean in as replacement.
                    +: quick
                    -: doesn't work on text. Causes data set to drift towards the mean.

                'mode':
                    calculates the column mode (exclude `missing`) and copies
                    the mean in as replacement.
                    +: quick
                    -: most frequent value becomes over-represented in the sample

                'nearest neighbour':
                    calculates normalised distance between items in source columns
                    selects nearest neighbour and copies value as replacement.
                    +: works for any datatype.
                    -: computationally intensive (e.g. slow)

            sources (list of strings): NEAREST NEIGHBOUR ONLY
                column names to be used during imputation.
                if None or empty, all columns will be used.

        Returns:
            table: table with replaced values.
        """
        if isinstance(targets, str) and targets not in self.columns:
            targets = [targets]
        if isinstance(targets, list):
            for name in targets:
                if not isinstance(name, str):
                    raise TypeError(f"expected str, not {type(name)}")
                if name not in self.columns:
                    raise ValueError(f"target item {name} not a column name in self.columns:\n{self.columns}")
        else:
            raise TypeError("Expected source as list of column names")

        if method == "nearest neighbour":
            if sources in (None, []):
                sources = self.columns
            if isinstance(sources, str):
                sources = [sources]
            if isinstance(sources, list):
                for name in sources:
                    if not isinstance(name, str):
                        raise TypeError(f"expected str, not {type(name)}")
                    if name not in self.columns:
                        raise ValueError(f"source item {name} not a column name in self.columns:\n{self.columns}")
            else:
                raise TypeError("Expected source as list of column names")

        methods = ["nearest neighbour", "mean", "mode", "carry forward"]

        if method == "carry forward":

            new = Table()
            for name in self.columns:
                if name in targets:
                    data = self[name][:]  # create copy
                    last_value = None
                    for ix, v in enumerate(data):
                        if v == missing:  # perform replacement
                            data[ix] = last_value
                        else:  # keep last value.
                            last_value = v
                    new[name] = data
                else:
                    new[name] = self[name]

            return new

        elif method in {"mean", "mode"}:

            new = Table()
            for name in self.columns:
                if name in targets:
                    col = self[name].copy()
                    assert isinstance(col, Column)
                    stats = col.statistics()
                    new_value = stats[method]
                    col.replace(target=missing, replacement=new_value)
                    new[name] = col
                else:
                    new[name] = self[name]  # no entropy, keep as is.

            return new

        elif method == "nearest neighbour":

            new = self.copy()
            norm_index = {}
            normalised_values = Table()
            for name in sources:
                values = self[name].unique().tolist()
                values = sortation.unix_sort(values, reverse=False)
                values = [(v, k) for k, v in values.items()]
                values.sort()
                values = [k for _, k in values]

                n = len([v for v in values if v != missing])
                d = {v: i / n if v != missing else math.inf for i, v in enumerate(values)}
                normalised_values[name] = [d[v] for v in self[name]]
                norm_index[name] = d
                values.clear()

            missing_value_index = self.index(*targets)
            missing_value_index = {
                k: v for k, v in missing_value_index.items() if missing in k
            }  # strip out all that do not have missings.
            ranks = set()
            for k, v in missing_value_index.items():
                ranks.update(set(k))
            item_order = sortation.unix_sort(list(ranks))
            new_order = {tuple(item_order[i] for i in k): k for k in missing_value_index.keys()}

            with tqdm(unit="missing values", total=sum(len(v) for v in missing_value_index.values())) as pbar:
                for _, key in sorted(new_order.items(), reverse=True):  # Fewest None's are at the front of the list.
                    for row_id in missing_value_index[key]:
                        err_map = [0.0 for _ in range(len(self))]
                        for n, v in self.to_dict(
                            columns=sources, slice_=slice(row_id, row_id + 1, 1)
                        ).items():  # self.to_dict doesn't go to disk as hence saves an IOP.
                            v = v[0]
                            norm_value = norm_index[n][v]
                            if norm_value != math.inf:
                                err_map = [e1 + abs(norm_value - e2) for e1, e2 in zip(err_map, normalised_values[n])]

                        min_err = min(err_map)
                        ix = err_map.index(min_err)

                        for name in targets:
                            current_value = new[name][row_id]
                            if current_value != missing:  # no need to replace anything.
                                continue
                            if new[name][ix] != missing:  # can confidently impute.
                                new[name][row_id] = new[name][ix]
                            else:  # replacement is required, but ix points to another missing value.
                                # we therefore have to search after the next best match:
                                tmp_err_map = err_map[:]
                                for _ in range(len(err_map)):
                                    tmp_min_err = min(tmp_err_map)
                                    tmp_ix = tmp_err_map.index(tmp_min_err)
                                    if row_id == tmp_ix:
                                        tmp_err_map[tmp_ix] = math.inf
                                        continue
                                    elif new[name][tmp_ix] == missing:
                                        tmp_err_map[tmp_ix] = math.inf
                                        continue
                                    else:
                                        new[name][row_id] = new[name][tmp_ix]
                                        break

                        pbar.update(1)
            return new

        else:
            raise ValueError(f"method {method} not recognised amonst known methods: {list(methods)})")

    def transpose(self, tqdm=_tqdm):
        if len(self.columns) == 0:
            return Table()

        rows = [[] for _ in range(len(self) + 1)]
        rows[0] = self.columns[1:]

        for x in tqdm(range(0, len(self)), desc="table transpose"):
            for y in rows[0]:
                value = self[y][x]
                rows[x + 1].append(value)

        unique_names = []
        table = Table()

        for column_name, values in zip((unique_name(str(c), unique_names) for c in ([self.columns[0]] + list(self[self.columns[0]]))), rows):
            unique_names.append(column_name)

            table[column_name] = values

        return table

    def pivot_transpose(self, columns, keep=None, column_name="transpose", value_name="value", tqdm=_tqdm):
        """Transpose a selection of columns to rows.

        Args:
            columns (list of column names): column names to transpose
            keep (list of column names): column names to keep (repeat)

        Returns:
            Table: with columns transposed to rows

        Example:
            transpose columns 1,2 and 3 and transpose the remaining columns, except `sum`.

        Input:

        | col1 | col2 | col3 | sun | mon | tue | ... | sat | sum  |
        |------|------|------|-----|-----|-----|-----|-----|------|
        | 1234 | 2345 | 3456 | 456 | 567 |     | ... |     | 1023 |
        | 1244 | 2445 | 4456 |     |   7 |     | ... |     |    7 |
        | ...  |      |      |     |     |     |     |     |      |

        t.transpose(keep=[col1, col2, col3], transpose=[sun,mon,tue,wed,thu,fri,sat])`

        Output:

        |col1| col2| col3| transpose| value|
        |----|-----|-----|----------|------|
        |1234| 2345| 3456| sun      |   456|
        |1234| 2345| 3456| mon      |   567|
        |1244| 2445| 4456| mon      |     7|

        """
        if not isinstance(columns, list):
            raise TypeError
        for i in columns:
            if not isinstance(i, str):
                raise TypeError
            if i not in self.columns:
                raise ValueError

        if keep is None:
            keep = []
        for i in keep:
            if not isinstance(i, str):
                raise TypeError
            if i not in self.columns:
                raise ValueError

        if column_name in keep + columns:
            column_name = unique_name(column_name, set_of_names=keep + columns)
        if value_name in keep + columns + [column_name]:
            value_name = unique_name(value_name, set_of_names=keep + columns)

        new = Table()
        new.add_columns(*keep + [column_name, value_name])
        news = {name: [] for name in new.columns}

        n = len(keep)
        
        with tqdm(total=len(self), desc="transpose") as pbar:
            for ix, row in enumerate(self.__getitem__(*keep + columns).rows, start=1):
                keeps = row[:n]
                transposes = row[n:]

                for name, value in zip(keep, keeps):
                    news[name].extend([value] * len(transposes))
                for name, value in zip(columns, transposes):
                    news[column_name].append(name)
                    news[value_name].append(value)

                if ix % SINGLE_PROCESSING_LIMIT == 0:
                    for name, values in news.items():
                        new[name].extend(values)
                        values.clear()

                pbar.update(1)

        for name, values in news.items():
            new[name].extend(values)
            values.clear()
        return new

    def diff(self, other, columns=None):
        """compares table self with table other

        Args:
            self (Table): Table
            other (Table): Table
            columns (List, optional): list of column names to include in comparison. Defaults to None.

        Returns:
            Table: diff of self and other with diff in columns 1st and 2nd.
        """
        if columns is None:
            columns = [name for name in self.columns if name in other.columns]
        elif isinstance(columns, list) and all(isinstance(i, str) for i in columns):
            for name in columns:
                if name not in self.columns:
                    raise ValueError(f"column '{name}' not found")
                if name not in other.columns:
                    raise ValueError(f"column '{name}' not found")
        else:
            raise TypeError("Expected list of column names")

        t1 = self.__getitem__(*columns)
        if isinstance(t1, Table):
            t1 = [tuple(r) for r in self.rows]
        else:
            t1 = list(self)
        t2 = other.__getitem__(*columns)
        if isinstance(t2, Table):
            t2 = [tuple(r) for r in other.rows]
        else:
            t2 = list(other)

        sm = difflib.SequenceMatcher(None, t1, t2)
        new = Table()
        first = unique_name("1st", columns)
        second = unique_name("2nd", columns)
        new.add_columns(*columns + [first, second])

        news = {n: [] for n in new.columns}

        for opc, t1a, t1b, t2a, t2b in sm.get_opcodes():
            if opc == "insert":
                for name, col in zip(columns, zip(*t2[t2a:t2b])):
                    news[name].extend(col)
                news[first] += ["-"] * (t2b - t2a)
                news[second] += ["+"] * (t2b - t2a)

            elif opc == "delete":
                for name, col in zip(columns, zip(*t1[t1a:t1b])):
                    news[name].extend(col)
                news[first] += ["+"] * (t1b - t1a)
                news[second] += ["-"] * (t1b - t1a)

            elif opc == "equal":
                for name, col in zip(columns, zip(*t2[t2a:t2b])):
                    news[name].extend(col)
                news[first] += ["="] * (t2b - t2a)
                news[second] += ["="] * (t2b - t2a)

            elif opc == "replace":
                for name, col in zip(columns, zip(*t2[t2a:t2b])):
                    news[name].extend(col)
                news[first] += ["r"] * (t2b - t2a)
                news[second] += ["r"] * (t2b - t2a)

            else:
                pass

            if len(news[first]) % 1_000_000 == 0:
                for name, L in news.items():
                    new[name].extend(L)
                    L.clear()

        for name, L in news.items():
            new[name].extend(L)
            L.clear()
        return new


class Column(object):
    def __init__(self, data=None, key=None) -> None:
        """
        data: list of values
        key: (default None) id used during Table.load to instantiate the column.
        """
        if key is None:
            self.key = mem.new_id("/column")
        else:
            self.key = key
        self.group = f"/column/{self.key}"
        if key is None:
            self._len = 0
            self.extend(data)

        else:
            length, pages = mem.load_column_attrs(self.group)
            self._len = length

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>({self._len} values | key={self.key})"

    def __repr__(self) -> str:
        return self.__str__()

    def types(self):
        """
        returns dict with datatype: frequency of occurrence
        """
        return mem.get_pages(self.group).get_types()

    @classmethod
    def load(cls, key):
        """
        loads a column using it hdf5 storage key.
        """
        return Column(key=key)

    def __iter__(self):
        return (v for v in self.__getitem__())

    def __getitem__(self, item=None):
        """The __getitem__ operator. Behaves like getitem on a list.

        Args:
            item (slice, optional): The slice. Defaults to None (all records)

        Returns:
            list: list of python types.
        """
        slc = arg_to_slice(item)

        result = mem.get_data(self.group, slc)

        if isinstance(item, int) and len(result) == 1:
            return result[0]
        else:
            if isinstance(result, np.ndarray):
                return result.tolist()
            else:
                return result

    def to_numpy(self, item=None):
        """
        returns nympy.ndarray

        *item: (slice): None (default) returns all records.
        """
        slc = arg_to_slice(item)
        return mem.get_data(self.group, slc)

    def clear(self):
        """
        clears the column. Like list().clear()
        """
        old_pages = mem.get_pages(self.group)
        self._len = mem.create_virtual_dataset(self.group, pages_before=old_pages, pages_after=[])

    def append(self, value):
        """
        addends value. Like list().append(value)

        Note: Slower than .extend( many values ) as each append is written to disk
        """
        self.__setitem__(key=slice(self._len, None, None), value=[value])

    def insert(self, index, value):
        """
        inserts values. Like list().insert(index, value)
        """
        old_pages = mem.get_pages(self.group)
        new_pages = old_pages[:]

        ix, start, _, page = old_pages.get_page_by_index(index)

        if mem.get_ref_count(page) == 1:
            new_page = page  # ref count match. Now let the page class do the insert.
            new_page.insert(index - start, value)
        else:
            data = page[:].tolist()
            data.insert(index - start, value)
            new_page = Page(data)  # copy the existing page so insert can be done below

        new_pages[ix] = new_page  # insert the changed page.
        self._len = mem.create_virtual_dataset(self.group, pages_before=old_pages, pages_after=new_pages)

    def extend(self, values):
        """
        extends the list. Like list().extend( many values )

        Note: Faster than .append as all values are written to disk at once.
        """
        self.__setitem__(slice(self._len, None, None), values)  # self._extend_from_column(values)

    def remove(self, value):
        """
        removes a single value.

        To remove all instances of `value` use .remove_all( value )
        """
        pages = mem.get_pages(self.group)
        for ix, page in enumerate(pages):
            if value not in page[:]:
                continue
            if mem.get_ref_count(page) == 1:
                page.remove(value)
                new_pages = pages[:]
            else:
                data = page[:]  # copy the data.
                data = data.tolist()
                data.remove(value)  # remove from the copy.
                new_page = page(data)  # create new page from copy
                new_pages = pages[:]
                new_pages[ix] = new_page  # register the newly copied page.
            self._len = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)
            return
        raise ValueError(f"value not found: {value}")

    def remove_all(self, value):
        """
        removes all values of `value`

        To remove only one instance of `value` use .remove ( value )
        """
        pages = mem.get_pages(self.group)
        new_pages = pages[:]
        for ix, page in enumerate(pages):
            if value not in page[:]:
                continue
            new_data = [v for v in page[:] if v != value]
            new_page = Page(new_data)
            new_pages[ix] = new_page
        self._len = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)

    def pop(self, index):
        """
        removes value at index. Like list().pop( index )
        """
        index = self._len + index if index < 0 else index
        if index > self._len:
            raise IndexError(f"can't reach index {index} when length is {self._len}")

        pages = mem.get_pages(self.group)
        ix, start, _, page = pages.get_page_by_index(index)
        if mem.get_ref_count(page) == 1:
            value = page.pop(index - start)
        else:
            data = page[:]
            value = data.pop(index - start)
            new_page = Page(data)
            new_pages = pages[:]
            new_pages[ix] = new_page
        shape = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)
        self._len = shape
        return value

    def __setitem__(self, key, value):
        """
        Column.__setitem__(key,value) behaves just like a list
        """
        if isinstance(key, int):
            if isinstance(value, (list, tuple)):
                raise TypeError(
                    f"your key is an integer, but your value is a {type(value)}. \
                        Did you mean to insert? F.x. [{key}:{key+1}] = {value} ?"
                )
            if -self._len - 1 < key < self._len:
                key = self._len + key if key < 0 else key
                pages = mem.get_pages(self.group)
                ix, start, _, page = pages.get_page_by_index(key)
                if mem.get_ref_count(page) == 1:
                    page[key - start] = value
                else:
                    data = page[:].tolist()
                    data[key - start] = value
                    new_page = Page(data)
                    new_pages = pages[:]
                    new_pages[ix] = new_page
                    self._len = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)
            else:
                raise IndexError("list assignment index out of range")

        elif isinstance(key, slice):
            start, stop, step = key.indices(self._len)
            if key.start is None and key.stop is None and key.step in (None, 1):
                # documentation: new = list(value)
                # example: L[:] = [1,2,3]
                before = mem.get_pages(self.group)
                if isinstance(value, Column):
                    after = mem.get_pages(value.group)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    new_page = Page(value)
                    after = Pages([new_page])
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.start is not None and key.stop is None and key.step is None:
                # documentation: new = old[:key.start] + list(value)
                # example: L[0:] = [1,2,3]
                before = mem.get_pages(self.group)
                before_slice = before.getslice(0, start)
                if value is None:  # path used by add_columns and t['c'] = None e.g. reset to empty table.
                    after = Pages()
                elif isinstance(value, Column):
                    after = before_slice + mem.get_pages(value.group)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    if not before_slice:
                        after = Pages((Page(value),))
                    else:
                        last_page = before_slice[-1]
                        if mem.get_ref_count(last_page) == 1:
                            before_copy = mem.get_pages(self.group)  # this is required because .extend is polymorphic
                            last_page.extend(value)
                            after = before_slice
                            before = before_copy  # to overcome polymorphism.
                        else:  # ref count > 1
                            new_page = Page(value)
                            after = before_slice + Pages([new_page])
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.stop is not None and key.start is None and key.step is None:
                # documentation: new = list(value) + old[key.stop:]
                # example: L[:3] = [1,2,3]
                before = mem.get_pages(self.group)
                before_slice = before.getslice(stop, self._len)
                if isinstance(value, Column):
                    after = mem.get_pages(value.group) + before_slice
                elif isinstance(value, (list, tuple, np.ndarray)):
                    new_page = Page(value)
                    after = Pages([new_page]) + before_slice
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.step is None and key.start is not None and key.stop is not None:  # L[3:5] = [1,2,3]
                # documentation: new = old[:start] + list(values) + old[stop:]

                stop = max(start, stop)  # one of python's archaic rules.

                before = mem.get_pages(self.group)
                A, B = before.getslice(0, start), before.getslice(stop, self._len)
                if isinstance(value, Column):
                    after = A + mem.get_pages(value.group) + B
                elif isinstance(value, (list, tuple, np.ndarray)):
                    if value:
                        new_page = Page(value)
                        after = (
                            A + Pages([new_page]) + B
                        )  # new = old._getslice_(0,start) + list(value) + old._getslice_(stop,len(self.items))
                    else:
                        after = A + B
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.step is not None:
                seq = range(start, stop, step)
                seq_size = len(seq)
                if len(value) > seq_size:
                    raise ValueError(
                        f"attempt to assign sequence of size {len(value)} to extended slice of size {seq_size}"
                    )

                # documentation: See also test_slice_rules.py/MyList for details
                before = mem.get_pages(self.group)
                new = mem.get_data(
                    self.group, slice(None)
                ).tolist()  # new = old[:]  # cheap shallow pointer copy in case anything goes wrong.
                for new_index, position in zip(range(len(value)), seq):
                    new[position] = value[new_index]
                # all went well. No exceptions. Now update self.
                after = Pages([Page(new)])  # This may seem redundant, but is in fact is good as the user may
                # be cleaning up the dataset, so that we end up with a simple datatype instead of mixed.
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            else:
                raise KeyError(f"bad key: {key}")
        else:
            raise TypeError(f"bad key: {key}")

    def __delitem__(self, key):
        if isinstance(key, int):
            if -self._len - 1 < key < self._len:
                before = mem.get_pages(self.group)
                after = before[:]
                ix, start, _, page = before.get_page_by_index(key)
                if mem.get_ref_count(page) == 1:
                    del page[key - start]
                else:
                    data = mem.get_data(page.group)
                    mask = np.ones(shape=data.shape)
                    new_data = np.compress(mask, data, axis=0)
                    after[ix] = Page(new_data)
            else:
                raise IndexError("list assignment index out of range")

            self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

        elif isinstance(key, slice):
            start, stop, step = key.indices(self._len)
            before = mem.get_pages(self.group)
            if key.start is None and key.stop is None and key.step in (None, 1):  # del L[:] == L.clear()
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=[])
            elif key.start is not None and key.stop == key.step is None:  # del L[0:]
                after = before.getslice(0, start)
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            elif key.stop is not None and key.start == key.step is None:  # del L[:3]
                after = before.getslice(stop, self._len)
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            elif key.step is None and key.start is not None and key.stop is not None:  # del L[3:5]
                after = before.getslice(0, start) + before.getslice(stop, self._len)
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            elif key.step is not None:
                before = mem.get_pages(self.group)
                data = mem.get_data(self.group, slice(None))
                mask = np.ones(shape=data.shape)
                for i in range(start, stop, step):
                    mask[i] = 0
                new = np.compress(mask, data, axis=0)
                # all went well. No exceptions.
                after = Pages([Page(new)])  # This may seem redundant, but is in fact is good as the user may
                # be cleaning up the dataset, so that we end up with a simple datatype instead of mixed.
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            else:
                raise TypeError(f"bad key: {key}")
        else:
            raise TypeError(f"bad key: {key}")

    def __len__(self):
        """
        returns number of entries in the Column. Like len(list())
        """
        return self._len

    def __eq__(self, other):
        if len(self) != len(other):  # quick cheap check.
            return False

        if isinstance(other, (list, tuple)):
            return all(a == b for a, b in zip(self[:], other))

        elif isinstance(other, Column):
            if mem.get_pages(self.group) == mem.get_pages(other.group):  # special case.
                return True
            return (self.to_numpy() == other.to_numpy()).all()

        elif isinstance(other, np.ndarray):
            return (self.to_numpy() == other).all()
        else:
            raise TypeError

    def copy(self):
        return Column(data=self)

    def __copy__(self):
        return self.copy()

    def index(self):
        """
        returns dict with { unique entry : list of indices }

        example:
        >>> c = Column(data=['a','b','a','c','b'])
        >>> c.index()
        {'a':[0,2], 'b': [1,4], 'c': [3]}

        """
        data = self.__getitem__()
        d = {k: [] for k in np.unique(data)}
        for ix, k in enumerate(data):
            d[k].append(ix)
        return d

    def unique(self):
        """
        returns unique list of values.

        example:
        >>> c = Column(data=['a','b','a','c','b'])
        >>> c.unqiue()
        ['a','b','c']
        """
        try:
            return np.unique(self.__getitem__())
        except TypeError:  # np arrays can't handle dtype='O':
            return np.array({i for i in self.__getitem__()})

    def histogram(self):
        """
        returns 2 arrays: unique elements and count of each element

        example:
        >>> c = Column(data=['a','b','a','c','b'])
        >>> c.unqiue()
        ['a','b','c'],[2,2,1]
        """
        try:
            uarray, carray = np.unique(self.__getitem__(), return_counts=True)
            uarray, carray = uarray.tolist(), carray.tolist()
        except TypeError:  # np arrays can't handle dtype='O':
            d = defaultdict(int)
            for i in self.__getitem__():
                d[i] += 1
            uarray, carray = [], []
            for k, v in d.items():
                uarray.append(k), carray.append(v)
        return uarray, carray

    def count(self, item):
        return sum(1 for i in self.__getitem__() if i == item)

    def replace(self, target, replacement):
        """
        replaces target with replacement
        """
        old_pages = mem.get_pages(self.group)
        new_pages = old_pages[:]

        for ix, page in enumerate(old_pages):
            if type(target) not in page.datatypes():
                continue  # quick scan.

            data = page[:].tolist()
            if target in data:
                data = [i if i != target else replacement for i in page[:]]
                new_pages[ix] = Page(data)
        self._len = mem.create_virtual_dataset(self.group, pages_before=old_pages, pages_after=new_pages)

    def statistics(self):
        """
        returns dict with:
        - min (int/float, length of str, date)
        - max (int/float, length of str, date)
        - mean (int/float, length of str, date)
        - median (int/float, length of str, date)
        - stdev (int/float, length of str, date)
        - mode (int/float, length of str, date)
        - distinct (int/float, length of str, date)
        - iqr (int/float, length of str, date)
        - sum (int/float, length of str, date)
        - histogram (see .histogram)
        """
        return summary_statistics(*self.histogram())

    def __add__(self, other):
        """
        Concatenates to Columns. Like list() + list()

        Example:
        >>> one,two = Column(data=[1,2]), Column(data=[3,4])
        >>> both = one+two
        >>> both[:]
        [1,2,3,4]
        """
        c = self.copy()
        c.extend(other)
        return c

    def __contains__(self, item):
        """
        determines if item is in the Column. Similar to 'x' in ['a','b','c']
        returns boolean
        """
        return item in self.__getitem__()

    def __iadd__(self, other):
        """
        Extends instance of Column with another Column

        Example:
        >>> one,two = Column(data=[1,2]), Column(data=[3,4])
        >>> one += two
        >>> one[:]
        [1,2,3,4]

        """
        self.extend(other)
        return self

    def __imul__(self, other):
        """
        Repeats instance of column N times. Like list() * N

        Example:
        >>> one = Column(data=[1,2])
        >>> one *= 5
        >>> one
        [1,2, 1,2, 1,2, 1,2, 1,2]

        """
        if not isinstance(other, int):
            raise TypeError(f"a column can be repeated an integer number of times, not {type(other)} number of times")
        old_pages = mem.get_pages(self.group)
        new_pages = old_pages * other
        self._len = mem.create_virtual_dataset(self.group, old_pages, new_pages)
        return self

    def __mul__(self, other):
        """
        Repeats instance of column N times. Like list() * N

        Example:
        >>> one = Column(data=[1,2])
        >>> five = one * 5
        >>> five
        [1,2, 1,2, 1,2, 1,2, 1,2]

        """
        if not isinstance(other, int):
            raise TypeError(f"a column can be repeated an integer number of times, not {type(other)} number of times")
        new = Column()
        old_pages = mem.get_pages(self.group)
        new_pages = old_pages * other
        new._len = mem.create_virtual_dataset(new.group, old_pages, new_pages)
        return new

    def __ne__(self, other):
        """
        compares two columns. Like list1 != list2
        """
        if len(self) != len(other):  # quick cheap check.
            return True

        if isinstance(other, (list, tuple)):
            return any(a != b for a, b in zip(self[:], other))

        if isinstance(other, Column):
            if mem.get_pages(self.group) == mem.get_pages(other.group):  # special case.
                return False
            return (self.to_numpy() != other.to_numpy()).any()

        elif isinstance(other, np.ndarray):
            return (self.to_numpy() != other).any()
        else:
            raise TypeError

    def __le__(self, other):
        raise NotImplementedError("vectorised operation A <= B is type-ambiguous")

    def __lt__(self, other):
        raise NotImplementedError("vectorised operation A < B is type-ambiguous")

    def __ge__(self, other):
        raise NotImplementedError("vectorised operation A >= B is type-ambiguous")

    def __gt__(self, other):
        raise NotImplementedError("vectorised operation A > B is type-ambiguous")


# -------------- MULTI PROCESSING TASKS -----------------
def _in(a, b):
    """
    enables filter function 'in'
    """
    return str(a) in str(b)


filter_ops = {
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "<": operator.lt,
    "<=": operator.le,
    "!=": operator.ne,
    "in": _in,
}

filter_ops_from_text = {"gt": ">", "gteq": ">=", "eq": "==", "lt": "<", "lteq": "<=", "neq": "!=", "in": _in}


def filter_evaluation_task(table_key, expression, shm_name, shm_index, shm_shape, slice_):
    """
    multiprocessing tasks for evaluating Table.filter
    """
    assert isinstance(table_key, str)  # 10 --> group = '/table/10'
    assert isinstance(expression, dict)
    assert len(expression) == 3
    assert isinstance(shm_name, str)
    assert isinstance(shm_index, int)
    assert isinstance(shm_shape, tuple)
    assert isinstance(slice_, slice)
    c1 = expression.get("column1", None)
    c2 = expression.get("column2", None)
    c = expression.get("criteria", None)
    assert c in filter_ops
    f = filter_ops.get(c)
    assert callable(f)
    v1 = expression.get("value1", None)
    v2 = expression.get("value2", None)

    columns = mem.mp_get_columns(table_key)
    if c1 is not None:
        column_key = columns[c1]
        dset_A = mem.get_data(f"/column/{column_key}", slice_)
    else:  # v1 is active:
        dset_A = np.array([v1] * (slice_.stop - slice_.start))

    if c2 is not None:
        column_key = columns[c2]
        dset_B = mem.get_data(f"/column/{column_key}", slice_)
    else:  # v2 is active:
        dset_B = np.array([v2] * (slice_.stop - slice_.start))

    existing_shm = shared_memory.SharedMemory(name=shm_name)  # connect
    result_array = np.ndarray(shm_shape, dtype=np.bool, buffer=existing_shm.buf)
    result_array[shm_index][slice_] = np.array([f(a, b) for a, b in zip(dset_A, dset_B)])  # Evaluate
    existing_shm.close()  # disconnect


def filter_merge_task(table_key, true_key, false_key, shm_name, shm_shape, slice_, filter_type):
    """
    multiprocessing task for merging data after the filter task has been completed.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name)  # connect
    result_array = np.ndarray(shm_shape, dtype=np.bool, buffer=existing_shm.buf)
    mask_source = result_array

    if filter_type == "any":
        true_mask = np.any(mask_source, axis=0)
    else:
        true_mask = np.all(mask_source, axis=0)
    true_mask = true_mask[slice_]
    false_mask = np.invert(true_mask)

    # 2. load source
    columns = mem.mp_get_columns(table_key)

    true_columns, false_columns = {}, {}
    for col_name, column_key in columns.items():
        col = Column(key=column_key)
        slize = col.to_numpy(slice_)
        true_values = slize[true_mask]
        if np.any(true_mask):
            true_columns[col_name] = mem.mp_write_column(true_values)
        false_values = slize[false_mask]
        if np.any(false_mask):
            false_columns[col_name] = mem.mp_write_column(false_values)

    mem.mp_write_table(true_key, true_columns)
    mem.mp_write_table(false_key, false_columns)

    existing_shm.close()  # disconnect


def indexing_task(source_key, destination_key, shm_name_for_sort_index, shape):
    """
    performs the creation of a column sorted by sort_index (shared memory object).
    source_key: column to read
    destination_key: column to write
    shm_name_for_sort_index: sort index' shm.name created by main.
    shape: shm array shape.

    *used by sort and all join functions.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name_for_sort_index)  # connect
    sort_index = np.ndarray(shape, dtype=np.int64, buffer=existing_shm.buf)

    data = mem.get_data(f"/column/{source_key}", slice(None))  # --- READ!
    values = [data[ix] for ix in sort_index]

    existing_shm.close()  # disconnect
    mem.mp_write_column(values, column_key=destination_key)  # --- WRITE!


def compress_task(source_key, destination_key, shm_index_name, shape):
    """
    compresses the source using boolean mask from shared memory

    source_key: column to read
    destination_key: column to write
    shm_name_for_sort_index: sort index' shm.name created by main.
    shape: shm array shape.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_index_name)  # connect
    mask = np.ndarray(shape, dtype=np.int64, buffer=existing_shm.buf)

    data = mem.get_data(f"/column/{source_key}", slice(None))  # --- READ!
    values = np.compress(mask, data)

    existing_shm.close()  # disconnect
    mem.mp_write_column(values, column_key=destination_key)  # --- WRITE!


# ---------- FILE WRITERS ------------
def _check_input(table, path):
    if not isinstance(table, Table):
        raise TypeError
    if not isinstance(path, pathlib.Path):
        raise TypeError


def excel_writer(table, path):
    """
    writer for excel files.

    This can create xlsx files beyond Excels.
    If you're using pyexcel to read the data, you'll see the data is there.
    If you're using Excel, Excel will stop loading after 1,048,576 rows.

    See pyexcel for more details:
    http://docs.pyexcel.org/
    """
    _check_input(table, path)

    def gen(table):  # local helper
        yield table.columns
        for row in table.rows:
            yield row

    data = list(gen(table))
    if path.suffix in [".xls", ".ods"]:
        data = [
            [str(v) if (isinstance(v, (int, float)) and abs(v) > 2**32 - 1) else DataTypes.to_json(v) for v in row]
            for row in data
        ]

    pyexcel.save_as(array=data, dest_file_name=str(path))


def text_writer(table, path, tqdm=_tqdm):
    """exports table to csv, tsv or txt dependening on path suffix.
    follows the JSON norm. text escape is ON for all strings.

    """
    _check_input(table, path)

    def txt(value):  # helper for text writer
        if isinstance(value, str):
            if not (value.startswith('"') and value.endswith('"')):
                return f'"{value}"'  # this must be escape: "the quick fox, jumped over the comma"
            else:
                return value  # this would for example be an empty string: ""
        else:
            return str(DataTypes.to_json(value))  # this handles datetimes, timedelta, etc.

    delimiters = {".csv": ",", ".tsv": "\t", ".txt": "|"}
    delimiter = delimiters.get(path.suffix)

    with path.open("w", encoding="utf-8") as fo:
        fo.write(delimiter.join(c for c in table.columns) + "\n")
        for row in tqdm(table.rows, total=len(table)):
            fo.write(delimiter.join(txt(c) for c in row) + "\n")


def sql_writer(table, path):
    _check_input(table, path)
    with path.open("w", encoding="utf-8") as fo:
        fo.write(table.to_sql())


def json_writer(table, path):
    _check_input(table, path)
    with path.open("w") as fo:
        fo.write(table.to_json())


def h5_writer(table, path):
    _check_input(table, path)
    table.to_hdf5(path)


def html_writer(table, path):
    _check_input(table, path)
    with path.open("w", encoding="utf-8") as fo:
        fo.write(table._repr_html_())


exporters = {  # the commented formats are not yet supported by the pyexcel plugins:
    # 'fods': excel_writer,
    "json": json_writer,
    "html": html_writer,
    # 'simple': excel_writer,
    # 'rst': excel_writer,
    # 'mediawiki': excel_writer,
    "xlsx": excel_writer,
    "xls": excel_writer,
    # 'xlsm': excel_writer,
    "csv": text_writer,
    "tsv": text_writer,
    "txt": text_writer,
    "ods": excel_writer,
    "sql": sql_writer,
    # 'hdf5': h5_writer,
    # 'h5': h5_writer
}
