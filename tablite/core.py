import json
import logging
import numpy as np
import math

# import pathlib
from pathlib import Path
import sys
import difflib
import itertools
import operator

from collections import defaultdict
from string import digits
from multiprocessing import shared_memory

import psutil
from tqdm import tqdm as _tqdm

from mplite import TaskManager, Task

from config import Config
from base import Table as BaseTable
from base import Column
from utils import unique_name, expression_interpreter, type_check
import import_utils
import export_utils

import sortation
from groupby_utils import GroupBy, GroupbyFunction

TIMEOUT_MS = 60 * 1000  # maximum msec tolerance waiting for OS to release hdf5 write lock


logging.getLogger("lml").propagate = False
logging.getLogger("pyexcel_io").propagate = False
logging.getLogger("pyexcel").propagate = False

log = logging.getLogger(__name__)
DIGITS = set(digits)


class Table(BaseTable):
    _pid_dir = None  # workdir / gettpid /

    def __init__(self, columns=None, headers=None, rows=None, _path=None) -> None:
        """creates Table

        Args:
            EITHER:
                columns (dict, optional): dict with column names as keys, values as lists.
                Example: t = Table(columns={"a": [1, 2], "b": [3, 4]})
            OR
                headers (list of strings, optional): list of column names.
                rows (list of tuples or lists, optional): values for columns
                Example: t = Table(headers=["a", "b"], rows=[[1,3], [2,4]])
        """
        super().__init__(_path)

    @classmethod
    def from_pandas(cls, df):
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
        return import_utils.from_pandas(cls, df)

    @classmethod
    def from_hdf5(cls, path):
        """
        imports an exported hdf5 table.
        """
        return import_utils.from_hdf5(cls, path)

    @classmethod
    def from_json(cls, jsn):
        """
        Imports tables exported using .to_json
        """
        return import_utils.from_json(cls, jsn)

    def to_hdf5(self, path):
        """
        creates a copy of the table as hdf5
        """
        export_utils.to_hdf5(self, path)

    def to_pandas(self):
        """
        returns pandas.DataFrame
        """
        return export_utils.to_pandas(self)

    def to_sql(self):
        """
        generates ANSI-92 compliant SQL.
        """
        return export_utils.to_sql(self)  # remove after update to test suite.

    def export(self, path):
        """
        exports table to path in format given by path suffix

        path: str or pathlib.Path

        for list of supported formats, see `exporters`
        """
        type_check(path, Path)

        ext = path.suffix[1:]  # .xlsx --> xlsx

        if ext not in export_utils.exporters:
            raise TypeError(f"{ext} not a supported formats:{export_utils.supported_formats}")

        handler = export_utils.exporters.get(ext)
        handler(table=self, path=path)

        log.info(f"exported {self} to {path}")
        print(f"exported {self} to {path}")

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
        reads path and imports 1 or more tables

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

            HINT: To get the head of file use:
            >>> from tablite.tools import head
            >>> head = head(path)

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
            path = Path(path)
        type_check(path, Path)

        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        if not isinstance(start, int) or not 0 <= start <= sys.maxsize:
            raise ValueError(f"start {start} not in range(0,{sys.maxsize})")

        if not isinstance(limit, int) or not 0 < limit <= sys.maxsize:
            raise ValueError(f"limit {limit} not in range(0,{sys.maxsize})")

        if not isinstance(first_row_has_headers, bool):
            raise TypeError("first_row_has_headers is not bool")

        import_as = path.suffix
        if import_as.startswith("."):
            import_as = import_as[1:]

        reader = import_utils.file_readers.get(import_as, None)
        if reader is None:
            raise ValueError(f"{import_as} is not in supported format: {import_utils.valid_readers}")

        additional_configs = {}
        if reader == import_utils.text_reader:
            # here we inject tqdm, if tqdm is not provided, use generic iterator
            # fmt:off
            config = import_utils.text_reader(path, columns, first_row_has_headers, encoding, start, limit, newline,
                                              guess_datatypes, text_qualifier, strip_leading_and_tailing_whitespace,
                                              delimiter, text_escape_openings, text_escape_closures, tqdm=tqdm)
            # fmt:on

        elif reader == import_utils.excel_reader:
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

        if reader == import_utils.ods_reader:
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

        # At this point the import config seems valid.
        # Now we check if the file already has been imported.

        # publish the settings
        logging.info("import config:\n" + "\n".join(f"{k}:{v}" for k, v in config.items()))
        return reader(cls, **config, **additional_configs)

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

        if len(self) * len(self.columns) < config.SINGLE_PROCESSING_LIMIT:
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

        if not isinstance(expressions, list) and not isinstance(expressions, tuple):
            raise TypeError

        if len(self) == 0:
            return self.copy(), self.copy()

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
        max_task_size = math.floor(Config.SINGLE_PROCESSING_LIMIT / len(self.columns))
        # 1 million fields per core (best guess!)

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
            if len(self) * (len(filter_tasks) + len(merge_tasks)) >= config.SINGLE_PROCESSING_LIMIT:
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
            else:
                for t in filter_tasks:
                    r = t.f(*t.args, **t.kwargs)
                    if r is not None:
                        raise r
                    pbar.update(1)

                for t in merge_tasks:
                    r = t.f(*t.args, **t.kwargs)
                    if r is not None:
                        raise r
                    pbar.update(1)

        cls = type(self)

        true = cls()
        true.add_columns(*self.columns)
        false = true.copy()

        for task in merge_tasks:
            tmp_true = cls.load(mem.path, key=task.kwargs["true_key"])
            if len(tmp_true):
                true += tmp_true
            else:
                pass

            tmp_false = cls.load(mem.path, key=task.kwargs["false_key"])
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
            len(self) * len(self.columns) < Config.SINGLE_PROCESSING_LIMIT
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
                    raise Exception("\n".join(filter(lambda x: x is not None, errs)))

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
            len(self) * len(self.columns) < Config.SINGLE_PROCESSING_LIMIT
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
                    raise Exception("\n".join(filter(lambda x: x is not None, errs)))

            table_key = mem.new_id("/table")
            mem.create_table(key=table_key, columns=columns_refs)

            shm.close()
            shm.unlink()
            t = type(self).load(path=mem.path, key=table_key)
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

        cls = type(self)

        t = cls.load(path=mem.path, key=table_key)
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

        if len(self) * len(self.columns) < config.SINGLE_PROCESSING_LIMIT:
            cls = type(self)
            t = cls()
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

        if len(self) * len(self.columns) < config.SINGLE_PROCESSING_LIMIT:
            cls = type(self)

            t = cls()
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
                for col_name, f in functions:
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

    def _jointype_check(self, other, left_keys, right_keys, left_columns, right_columns):
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
        TODO: better memory management when processes share column chunks (requires masking Nones)
        """
        LEFT_NONE_MASK, RIGHT_NONE_MASK = (_maskify(arr) for arr in (LEFT, RIGHT))

        left_arr, left_shm = _share_mem(LEFT, np.int64)
        right_arr, right_shm = _share_mem(RIGHT, np.int64)
        left_msk_arr, left_msk_shm = _share_mem(LEFT_NONE_MASK, np.bool8)
        right_msk_arr, right_msk_shm = _share_mem(RIGHT_NONE_MASK, np.bool8)

        final_len = len(LEFT)

        assert len(LEFT) == len(RIGHT)

        tasks = []
        columns_refs = {}

        rows_per_page = tcfg.H5_PAGE_SIZE

        for name in left_columns:
            col = self[name]
            container = columns_refs[name] = []

            offset = 0

            while offset < final_len or final_len == 0:  # create an empty page
                new_offset = min(offset + rows_per_page, final_len)
                slice_ = slice(offset, new_offset)
                d_key = mem.new_id("/column")
                container.append(d_key)
                tasks.append(
                    Task(
                        indexing_task,
                        source_key=col.key,
                        destination_key=d_key,
                        shm_name_for_sort_index=left_shm.name,
                        shm_name_for_sort_index_mask=left_msk_shm.name,
                        shape=left_arr.shape,
                        slice_=slice_,
                    )
                )

                offset = new_offset

                if final_len == 0:
                    break

        for name in right_columns:
            revised_name = unique_name(name, columns_refs.keys())
            col = other[name]
            container = columns_refs[revised_name] = []

            offset = 0

            while offset < final_len or final_len == 0:  # create an empty page
                new_offset = min(offset + rows_per_page, final_len)
                slice_ = slice(offset, new_offset)
                d_key = mem.new_id("/column")
                container.append(d_key)
                tasks.append(
                    Task(
                        indexing_task,
                        source_key=col.key,
                        destination_key=d_key,
                        shm_name_for_sort_index=right_shm.name,
                        shm_name_for_sort_index_mask=right_msk_shm.name,
                        shape=right_arr.shape,
                        slice_=slice_,
                    )
                )

                offset = new_offset

                if final_len == 0:
                    break

        if pbar is None:
            total = len(tasks)
            pbar = tqdm(total=total, desc="join")

        with TaskManager(cpu_count=min(psutil.cpu_count(), total)) as tm:
            results = tm.execute(tasks, tqdm=tqdm, pbar=pbar)

            if any(i is not None for i in results):
                raise Exception("\n".join(filter(lambda x: x is not None, results)))

        merged_column_refs = {k: mem.mp_merge_columns(v) for k, v in columns_refs.items()}

        with h5py.File(mem.path, "r+") as h5:
            table_key = mem.new_id("/table")
            dset = h5.create_dataset(name=f"/table/{table_key}", dtype=h5py.Empty("f"))
            dset.attrs["columns"] = json.dumps(merged_column_refs)
            dset.attrs["saved"] = False

        left_shm.close()
        left_shm.unlink()
        right_shm.close()
        right_shm.unlink()

        left_msk_shm.close()
        left_msk_shm.unlink()
        right_msk_shm.close()
        right_msk_shm.unlink()

        t = Table.load(path=mem.path, key=table_key)
        return t

    def left_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
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

        self._jointype_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        left_index = self.index(*left_keys)
        right_index = other.index(*right_keys)
        LEFT, RIGHT = [], []
        for left_key, left_ixs in left_index.items():
            right_ixs = right_index.get(left_key, (None,))
            for left_ix in left_ixs:
                for right_ix in right_ixs:
                    LEFT.append(left_ix)
                    RIGHT.append(right_ix)

        if tcfg.PROCESSING_PRIORITY == "sp":
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        elif tcfg.PROCESSING_PRIORITY == "mp":
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        else:
            if len(LEFT) * len(left_columns + right_columns) < config.SINGLE_PROCESSING_LIMIT:
                return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
            else:  # use multi processing
                return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def inner_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
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

        self._jointype_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

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

        if tcfg.PROCESSING_PRIORITY == "sp":
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        elif tcfg.PROCESSING_PRIORITY == "mp":
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        else:
            if len(LEFT) * len(left_columns + right_columns) < config.SINGLE_PROCESSING_LIMIT:
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

        self._jointype_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

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

        if tcfg.PROCESSING_PRIORITY == "sp":
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        elif tcfg.PROCESSING_PRIORITY == "mp":
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        else:
            if len(LEFT) * len(left_columns + right_columns) < config.SINGLE_PROCESSING_LIMIT:
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

        self._jointype_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        LEFT, RIGHT = zip(*itertools.product(range(len(self)), range(len(other))))

        if tcfg.PROCESSING_PRIORITY == "sp":
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        elif tcfg.PROCESSING_PRIORITY == "mp":
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
        else:
            if len(LEFT) < Config.SINGLE_PROCESSING_LIMIT:
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

        ops = lookup_ops

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

        if tcfg.PROCESSING_PRIORITY == "sp":
            return self._sp_lookup(other, result, results)
        elif tcfg.PROCESSING_PRIORITY == "mp":
            return self._mp_lookup(other, result, results)
        else:
            if len(self) * len(other.columns) < Config.SINGLE_PROCESSING_LIMIT:
                return self._sp_lookup(other, result, results)
            else:
                return self._mp_lookup(other, result, results)

    def _sp_lookup(self, other, result, results):
        for col_name in other.columns:
            col_data = other[col_name][:]
            revised_name = unique_name(col_name, result.columns)
            result[revised_name] = [col_data[k] if k is not None else None for k in results]
        return result

    def _mp_lookup(self, other, result, results):
        # 1. create shared memory array.
        RIGHT_NONE_MASK = _maskify(results)
        right_arr, right_shm = _share_mem(results, np.int64)
        _, right_msk_shm = _share_mem(RIGHT_NONE_MASK, np.bool8)

        # 2. create tasks
        tasks = []
        columns_refs = {}

        for name in other.columns:
            revised_name = unique_name(name, result.columns + list(columns_refs.keys()))
            col = other[name]
            columns_refs[revised_name] = d_key = mem.new_id("/column")
            tasks.append(
                Task(
                    indexing_task,
                    source_key=col.key,
                    destination_key=d_key,
                    shm_name_for_sort_index=right_shm.name,
                    shm_name_for_sort_index_mask=right_msk_shm.name,
                    shape=right_arr.shape,
                )
            )

        # 3. let task manager handle the tasks
        with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
            errs = tm.execute(tasks)
            if any(errs):
                raise Exception("\n".join(filter(lambda x: x is not None, errs)))

        # 4. update the result table.
        with h5py.File(mem.path, "r+") as h5:
            dset = h5[f"/table/{result.key}"]
            columns = json.loads(dset.attrs["columns"])
            columns.update(columns_refs)
            dset.attrs["columns"] = json.dumps(columns)
            dset.attrs["saved"] = False

        # 5. close the share memory and deallocate
        right_shm.close()
        right_shm.unlink()

        right_msk_shm.close()
        right_msk_shm.unlink()

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

        for column_name, values in zip(
            (unique_name(str(c), unique_names) for c in ([self.columns[0]] + list(self[self.columns[0]]))), rows
        ):
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

                if ix % config.SINGLE_PROCESSING_LIMIT == 0:
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


# -------------- MULTI PROCESSING TASKS -----------------


def not_in(a, b):
    return not operator.contains(str(a), str(b))


def _in(a, b):
    """
    enables filter function 'in'
    """
    return str(a) in str(b)
    return operator.contains(str(a), str(b))  # TODO : check which method is faster


lookup_ops = {
    "in": _in,
    "not in": not_in,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "!=": operator.ne,
    "==": operator.eq,
}


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
        slize = col.get_numpy(slice_)
        true_values = slize[true_mask]
        if np.any(true_mask):
            true_columns[col_name] = mem.mp_write_column(true_values)
        false_values = slize[false_mask]
        if np.any(false_mask):
            false_columns[col_name] = mem.mp_write_column(false_values)

    mem.mp_write_table(true_key, true_columns)
    mem.mp_write_table(false_key, false_columns)

    existing_shm.close()  # disconnect


def indexing_task(
    source_key, destination_key, shm_name_for_sort_index, shape, slice_=slice(None), shm_name_for_sort_index_mask=None
):
    """
    performs the creation of a column sorted by sort_index (shared memory object).
    source_key: column to read
    destination_key: column to write
    shm_name_for_sort_index: sort index' shm.name created by main.
    shm_name_for_sort_index_mask: sort index' shm.name created by main containing None mask.
    shape: shm array shape.

    *used by sort and all join functions.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name_for_sort_index)  # connect
    sort_index = np.ndarray(shape, dtype=np.int64, buffer=existing_shm.buf)

    sort_slice = sort_index[slice_]

    if shm_name_for_sort_index_mask is not None:
        existing_mask_shm = shared_memory.SharedMemory(name=shm_name_for_sort_index_mask)  # connect
        sort_index_mask = np.ndarray(shape, dtype=np.int8, buffer=existing_mask_shm.buf)
    else:
        sort_index_mask = None

    data = mem.get_data(f"/column/{source_key}", slice(None))  # --- READ!

    values = [None] * len(sort_slice)

    start_offset = 0 if slice_.start is None else slice_.start

    for i, (j, ix) in enumerate(enumerate(sort_slice, start_offset)):
        if sort_index_mask is not None and sort_index_mask[j] == 1:
            values[i] = None
        else:
            values[i] = data[ix]

    existing_shm.close()  # disconnect
    if sort_index_mask is not None:
        existing_mask_shm.close()
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


def _maskify(arr):
    none_mask = [False] * len(arr)  # Setting the default

    for i in range(len(arr)):
        if arr[i] is None:  # Check if our value is None
            none_mask[i] = True
            arr[i] = 0  # Remove None from the original array

    return none_mask


def _share_mem(inp_arr, dtype):
    len_ = len(inp_arr)
    size = np.dtype(dtype).itemsize * len_
    shape = (len_,)

    out_shm = shared_memory.SharedMemory(create=True, size=size)  # the co_processors will read this.
    out_arr_index = np.ndarray(shape, dtype=dtype, buffer=out_shm.buf)
    out_arr_index[:] = inp_arr

    return out_arr_index, out_shm
