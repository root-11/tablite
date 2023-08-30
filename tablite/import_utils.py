import numpy as np
import os
import math
import platform
import psutil
import csv
from pathlib import Path
import pyexcel
import sys
import warnings
import logging

from mplite import TaskManager, Task

from tablite.datatypes import DataTypes, list_to_np_array
from tablite.config import Config
from tablite.file_reader_utils import TextEscape, get_encoding, get_delimiter, ENCODING_GUESS_BYTES
from tablite.utils import type_check, unique_name
from tablite.base import Table, Page, Column

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
    if not issubclass(T, Table):
        raise TypeError("Expected subclass of Table")

    return T(columns=df.to_dict("list"))  # noqa


def from_hdf5(T, path, tqdm=_tqdm, pbar=None):
    """
    imports an exported hdf5 table.

    Note that some loss of type information is to be expected in columns of mixed type:
    >>> t.show(dtype=True)
    +===+===+=====+=====+====+=====+=====+===================+==========+========+===============+===+=========================+=====+===+
    | # | A |  B  |  C  | D  |  E  |  F  |         G         |    H     |   I    |       J       | K |            L            |  M  | O |
    |row|int|mixed|float|str |mixed| bool|      datetime     |   date   |  time  |   timedelta   |str|           int           |float|int|
    +---+---+-----+-----+----+-----+-----+-------------------+----------+--------+---------------+---+-------------------------+-----+---+
    | 0 | -1|None | -1.1|    |None |False|2023-06-09 09:12:06|2023-06-09|09:12:06| 1 day, 0:00:00|b  |-100000000000000000000000|  inf| 11|
    | 1 |  1|    1|  1.1|1000|1    | True|2023-06-09 09:12:06|2023-06-09|09:12:06|2 days, 0:06:40|嗨 | 100000000000000000000000| -inf|-11|
    +===+===+=====+=====+====+=====+=====+===================+==========+========+===============+===+=========================+=====+===+
    >>> t.to_hdf5(filename)
    >>> t2 = Table.from_hdf5(filename)
    >>> t2.show(dtype=True)
    +===+===+=====+=====+=====+=====+=====+===================+===================+========+===============+===+=========================+=====+===+
    | # | A |  B  |  C  |  D  |  E  |  F  |         G         |         H         |   I    |       J       | K |            L            |  M  | O |
    |row|int|mixed|float|mixed|mixed| bool|      datetime     |      datetime     |  time  |      str      |str|           int           |float|int|
    +---+---+-----+-----+-----+-----+-----+-------------------+-------------------+--------+---------------+---+-------------------------+-----+---+
    | 0 | -1|None | -1.1|None |None |False|2023-06-09 09:12:06|2023-06-09 00:00:00|09:12:06|1 day, 0:00:00 |b  |-100000000000000000000000|  inf| 11|
    | 1 |  1|    1|  1.1| 1000|    1| True|2023-06-09 09:12:06|2023-06-09 00:00:00|09:12:06|2 days, 0:06:40|嗨 | 100000000000000000000000| -inf|-11|
    +===+===+=====+=====+=====+=====+=====+===================+===================+========+===============+===+=========================+=====+===+
    """
    if not issubclass(T, Table):
        raise TypeError("Expected subclass of Table")
    import h5py

    type_check(path, Path)
    t = T()
    with h5py.File(path, "r") as h5:
        for col_name in h5.keys():
            dset = h5[col_name]
            arr = np.array(dset[:])
            if arr.dtype == object:
                arr = np.array(DataTypes.guess([v.decode("utf-8") for v in arr]))
            t[col_name] = arr
    return t


def from_json(T, jsn):
    """
    Imports tables exported using .to_json
    """
    if not issubclass(T, Table):
        raise TypeError("Expected subclass of Table")
    import json

    type_check(jsn, str)
    d = json.loads(jsn)
    return T(columns=d["columns"])


def from_html(T, path, tqdm=_tqdm, pbar=None):
    if not issubclass(T, Table):
        raise TypeError("Expected subclass of Table")
    type_check(path, Path)

    if pbar is None:
        total = path.stat().st_size
        pbar = tqdm(total=total, desc="from_html", disable=Config.TQDM_DISABLE)

    row_start, row_end = "<tr>", "</tr>"
    value_start, value_end = "<th>", "</th>"
    chunk = ""
    t = None  # will be T()
    start, end = 0, 0
    data = {}
    with path.open("r") as fi:
        while True:
            start = chunk.find(row_start, start)  # row tag start
            end = chunk.find(row_end, end)  # row tag end
            if start == -1 or end == -1:
                new = fi.read(100_000)
                pbar.update(len(new))
                if new == "":
                    break
                chunk += new
                continue
            # get indices from chunk
            row = chunk[start + len(row_start) : end]
            fields = [v.rstrip(value_end) for v in row.split(value_start)]
            if not data:
                headers = fields[:]
                data = {f: [] for f in headers}
                continue
            else:
                for field, header in zip(fields, headers):
                    data[header].append(field)

            chunk = chunk[end + len(row_end) :]

            if len(data[headers[0]]) == Config.PAGE_SIZE:
                if t is None:
                    t = T(columns=data)
                else:
                    for k, v in data.items():
                        t[k].extend(DataTypes.guess(v))
                data = {f: [] for f in headers}

    for k, v in data.items():
        t[k].extend(DataTypes.guess(v))
    return t


def excel_reader(T, path, first_row_has_headers=True, header_row_index=0, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table from excel

    **kwargs are excess arguments that are ignored.
    """
    if not issubclass(T, Table):
        raise TypeError("Expected subclass of Table")

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
            columns = [i[header_row_index] for i in ws.columns()]
        else:
            columns = [str(i) for i in range(len(ws.columns()))]

    used_columns_names = set()
    t = T()
    for idx, col in enumerate(ws.columns()):
        if first_row_has_headers:
            header, start_row_pos = str(col[header_row_index]), (max(1, start) + header_row_index)
        else:
            header, start_row_pos = str(idx), (max(0, start) + header_row_index)

        if header not in columns:
            continue

        unique_column_name = unique_name(str(header), used_columns_names)
        used_columns_names.add(unique_column_name)

        t[unique_column_name] = [v for v in col[start_row_pos : start_row_pos + limit]]
    return t


def ods_reader(T, path, first_row_has_headers=True, header_row_index=0, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table from .ODS
    """
    if not issubclass(T, Table):
        raise TypeError("Expected subclass of Table")

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
    for ix, value in enumerate(data[header_row_index]):
        if first_row_has_headers:
            header, start_row_pos = str(value), (1 + header_row_index)
        else:
            header, start_row_pos = f"_{ix + 1}", (0 + header_row_index)

        if columns is not None:
            if header not in columns:
                continue

        unique_column_name = unique_name(str(header), used_columns_names)
        used_columns_names.add(unique_column_name)

        t[unique_column_name] = [row[ix] for row in data[start_row_pos : start_row_pos + limit] if len(row) > ix]
    return t


class TRconfig(object):
    def __init__(
        self,
        source,
        destination,
        start,
        end,
        guess_datatypes,
        delimiter,
        text_qualifier,
        text_escape_openings,
        text_escape_closures,
        strip_leading_and_tailing_whitespace,
        encoding,
        newline_offsets,
        fields
    ) -> None:
        self.source = source
        self.destination = destination
        self.start = start
        self.end = end
        self.guess_datatypes = guess_datatypes
        self.delimiter = delimiter
        self.text_qualifier = text_qualifier
        self.text_escape_openings = text_escape_openings
        self.text_escape_closures = text_escape_closures
        self.strip_leading_and_tailing_whitespace = strip_leading_and_tailing_whitespace
        self.encoding = encoding
        self.newline_offsets = newline_offsets
        self.fields = fields
        type_check(start, int),
        type_check(end, int),
        type_check(delimiter, str),
        type_check(text_qualifier, (str, type(None))),
        type_check(text_escape_openings, str),
        type_check(text_escape_closures, str),
        type_check(encoding, str),
        type_check(strip_leading_and_tailing_whitespace, bool),
        type_check(newline_offsets, list)
        type_check(fields, dict)

    def copy(self):
        return TRconfig(**self.dict())

    def dict(self):
        return {k: v for k, v in self.__dict__.items() if not (k.startswith("_") or callable(v))}


def _create_numpy_header(dtype, shape, file_handler):
    magic = b"\x93NUMPY"
    major = b"\x01"
    minor = b"\x00"
    header = {
        "descr": dtype,
        "fortran_order": False,
        "shape": shape,
    }
    header_str = str(header).encode("ascii")
    header_len = len(header_str)
    padding = 64 - ((len(magic) + len(major) + len(minor) + 2 + header_len)) % 64
    file_handler.write(magic)
    file_handler.write(major)
    file_handler.write(minor)
    file_handler.write((header_len + padding).to_bytes(2, "little"))
    file_handler.write(header_str)
    file_handler.write(b" " * (padding - 1) + "\n".encode("ascii"))


def text_reader_task(
    source,
    destination,
    start,
    end,
    guess_datatypes,
    delimiter,
    text_qualifier,
    text_escape_openings,
    text_escape_closures,
    strip_leading_and_tailing_whitespace,
    encoding,
    newline_offsets,
    fields
):
    """PARALLEL TASK FUNCTION
    reads columnsname + path[start:limit] into hdf5.

    source: csv or txt file
    destination: filename for page.
    start: int: start of page.
    end: int: end of page.
    guess_datatypes: bool: if True datatypes will be inferred by datatypes.Datatypes.guess
    delimiter: ',' ';' or '|'
    text_qualifier: str: commonly \"
    text_escape_openings: str: default: "({[
    text_escape_closures: str: default: ]})"
    strip_leading_and_tailing_whitespace: bool
    encoding: chardet encoding ('utf-8, 'ascii', ..., 'ISO-22022-CN')
    """
    if isinstance(source, str):
        source = Path(source)
    type_check(source, Path)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")
    type_check(destination, list)

    # declare CSV dialect.
    delim = delimiter

    class Dialect(csv.Dialect):
        delimiter = delim
        quotechar = '"' if text_qualifier is None else text_qualifier
        escapechar = '\\'
        doublequote = True
        quoting = csv.QUOTE_MINIMAL
        skipinitialspace = False if strip_leading_and_tailing_whitespace is None else strip_leading_and_tailing_whitespace
        lineterminator = "\n"

    with source.open("r", encoding=encoding, errors="ignore") as fi:  # --READ
        fi.seek(newline_offsets[start])
        reader = csv.reader(fi, dialect=Dialect)

        page_file_handlers = [open(f, mode="wb") for f in destination]

        # identify longest str
        longest_str = [0 for _ in range(len(destination))]
        for row in (next(reader) for _ in range(end - start)):
            for idx, c in ((fields[idx], c) for idx, c in filter(lambda t: t[0] in fields, enumerate(row))):
                longest_str[idx] = max(longest_str[idx], len(c))

        column_formats = [f"<U{i}" for i in longest_str]
        for idx, cf in enumerate(column_formats):
            _create_numpy_header(cf, (end - start, ), page_file_handlers[idx])

        # write page arrays to files
        fi.seek(newline_offsets[start])
        for row in (next(reader) for _ in range(end - start)):
            for idx, c in ((fields[idx], c) for idx, c in filter(lambda t: t[0] in fields, enumerate(row))):
                cbytes = np.asarray(c, dtype=column_formats[idx]).tobytes()
                page_file_handlers[idx].write(cbytes)

        [phf.close() for phf in page_file_handlers]

def text_reader(
    T,
    path,
    columns,
    first_row_has_headers,
    header_row_index,
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
    if not issubclass(T, Table):
        raise TypeError("Expected subclass of Table")

    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.stat().st_size == 0:
        return T()  # NO DATA: EMPTY TABLE.

    if encoding is None:
        encoding = get_encoding(path, nbytes=ENCODING_GUESS_BYTES)

    if delimiter is None:
        try:
            delimiter = get_delimiter(path, encoding)
        except ValueError:
            return T()  # NO DELIMITER: EMPTY TABLE.

    read_stage, process_stage, dump_stage, consolidation_stage = 20, 10, 60, 10
    assert sum([read_stage, process_stage, dump_stage, consolidation_stage]) == 100, "Must add to to a 100"
    pbar_fname = path.name

    if len(pbar_fname) > 20:
        pbar_fname = pbar_fname[0:10] + "..." + pbar_fname[-7:]

    file_length = path.stat().st_size  # 9,998,765,432 = 10Gb

    if not (isinstance(start, int) and start >= 0):
        raise ValueError("expected start as an integer >= 0")
    if not (isinstance(limit, int) and limit > 0):
        raise ValueError("expected limit as an integer > 0")

    newline_offsets = [0]

    # fmt:off
    with tqdm(total=100, desc=f"importing: reading '{pbar_fname}' bytes", unit="%",
              bar_format="{desc}: {percentage:3.2f}%|{bar}| [{elapsed}<{remaining}]", disable=Config.TQDM_DISABLE) as pbar:
        # fmt:on
        with path.open("rb") as fi:
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
            try:
                newlines = 0
                dx, dy = 0, 0

                for block in fi:
                    dx = dx + len(block)
                    newline_offsets.append(dx)
                    pbar.update(((dx - dy) / file_length) * read_stage)
                    newlines = newlines + 1
                    dy = dx

                pbar.desc = f"importing: processing '{pbar_fname}'"
                pbar.update(read_stage - pbar.n)

                fi.seek(0)
            except Exception as e:
                raise ValueError(f"file could not be read with encoding={encoding}\n{str(e)}")
            if newlines < 1:
                raise ValueError(f"Using {newline} to split file, revealed {newlines} lines in the file.")

            if newlines <= start + header_row_index + (1 if first_row_has_headers else 0):  # Then start > end: Return EMPTY TABLE.
                return Table(columns={n : [] for n in columns})

        line_reader = TextEscape(
            openings=text_escape_openings,
            closures=text_escape_closures,
            text_qualifier=text_qualifier,
            delimiter=delimiter,
            strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace,
        )

        with path.open("r", encoding=encoding) as fi:
            fi.seek(newline_offsets[header_row_index])
            for line in fi:
                if line == "":  # skip any empty line or header offset.
                    continue
                else:
                    line = line.rstrip(newline)
                    fields = line_reader(line)
                    break
            if not fields:
                warnings.warn("file was empty: {path}")
                return T()  # returning an empty table as there was no data.

        if columns is None:
            columns = fields[:]
        else:
            type_check(columns, list)
            if set(fields) < set(columns):
                missing = [c for c in columns if c not in fields]
                raise ValueError(f"missing columns {missing}")

        if first_row_has_headers is False:
            new_fields = {}
            for ix, name in enumerate(fields):
                new_fields[ix] = f"{ix}"  # name starts on 0.
            fields = new_fields
        else:  # first_row_has_headers is True, but ...
            new_fields, seen = {}, set()
            for ix, name in enumerate(fields):
                if name in columns:  # I may have to reduce to match user selection of columns.
                    unseen_name = unique_name(name, seen)
                    new_fields[ix] = unseen_name
                    seen.add(unseen_name)
            fields = {ix: name for ix, name in new_fields.items() if name in columns}

        if not fields:
            if columns is not None:
                raise ValueError(f"Columns not found: {columns}")
            else:
                raise ValueError("No columns?")

        field_relation = {f: i for i, f in enumerate(fields.keys())}
        inv_field_relation = dict(zip(field_relation.values(), field_relation.keys()))

        task_config = TRconfig(
            source=str(path),
            destination=None,
            start=1,
            end=Config.PAGE_SIZE,
            guess_datatypes=guess_datatypes,
            delimiter=delimiter,
            text_qualifier=text_qualifier,
            text_escape_openings=text_escape_openings,
            text_escape_closures=text_escape_closures,
            strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace,
            encoding=encoding,
            newline_offsets=newline_offsets,
            fields=field_relation
        )

        # make sure the tempdir is ready.
        workdir = Path(Config.workdir) / f"pid-{os.getpid()}"
        if not workdir.exists():
            workdir.mkdir()
            (workdir / "pages").mkdir()

        tasks, configs = [], []

        begin = header_row_index + 1 if first_row_has_headers else 0
        # Creates task for n pages of size Config.PAGE_SIZE. Assigns a page index for each column.
        for start in range(begin, newlines + 1, Config.PAGE_SIZE):
            end = min(start + Config.PAGE_SIZE, newlines)

            cfg = task_config.copy()
            cfg.start = start
            cfg.end = end
            cfg.destination = [workdir / "pages" / f"{next(Page.ids)}.npy" for _ in range(len(fields))]
            tasks.append(Task(f=text_reader_task, **cfg.dict()))
            configs.append(cfg)

            start = end

        pbar.desc = f"importing: saving '{pbar_fname}' to disk"
        pbar.update((read_stage + process_stage) - pbar.n)

        len_tasks = len(tasks)
        dump_size = dump_stage / len_tasks

        # TODO: Move external.
        class PatchTqdm:  # we need to re-use the tqdm pbar, this will patch
            # the tqdm to update existing pbar instead of creating a new one
            def update(self, n=1):
                pbar.update(n * dump_size)

        """
            all modern processors have more than one thread per core intel has hyper-threading, amd has SMT
            we don't want to stop using potentially half of our cores on other OS'es because windows can't that many file handles
        """
        is_windows = platform.system() == "Windows"
        use_logical = False if is_windows else True

        cpus = max(psutil.cpu_count(logical=use_logical), 1)  # there's always at least one core.
        
        cpus_needed = min(len(tasks), cpus)  # 4 columns won't require 96 cpus ...!
        if cpus_needed < 2 or Config.MULTIPROCESSING_MODE == Config.FALSE:
            for task in tasks:
                err = task.execute()
                if err is not None:
                    raise Exception(err)
                pbar.update(dump_size)

        else:
            with TaskManager(cpus_needed) as tm:
                errors = tm.execute(tasks, pbar=PatchTqdm())  # I expects a list of None's if everything is ok.
                if any(errors):
                    raise Exception("\n".join(e for e in errors if e))

        pbar.desc = f"importing: consolidating '{pbar_fname}'"
        pbar.update((read_stage + process_stage + dump_stage) - pbar.n)

        consolidation_size = consolidation_stage / len_tasks

        # consolidate the task results
        t = T()
        for name in fields.values():
            t[name] = Column(t.path)
        for cfg in configs:
            for idx, npy in ((inv_field_relation[idx], npy) for idx, npy in enumerate(cfg.destination)):
                data = np.load(npy, allow_pickle=True, fix_imports=False)

                if guess_datatypes:
                    data = list_to_np_array(DataTypes.guess(data))

                t[fields[idx]].extend(data)
                os.remove(npy)
            pbar.update(consolidation_size)

        pbar.update(100 - pbar.n)
        return t


file_readers = {  # dict of file formats and functions used during Table.import_file
    "fods": excel_reader,
    "json": excel_reader,
    "html": from_html,
    "hdf5": from_hdf5,
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
