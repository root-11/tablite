import json
import pickle
from itertools import count
from collections import defaultdict
from datetime import datetime, date, time
from pathlib import Path
from random import choice
from string import ascii_lowercase


import zlib
from tempfile import gettempdir
import sqlite3
from sys import getsizeof

import zipfile
import tempfile
import xlrd
import pyexcel_ods

__all__ = ['DataTypes', 'StoredList', 'Column', 'Table', 'file_reader',
           'GroupBy', 'Max', 'Min', 'Sum', 'First', 'Last', 'Count',
           'CountUnique', 'Average', 'StandardDeviation', 'Median', 'Mode']


class DataTypes(object):
    # supported datatypes.
    int = int
    str = str
    float = float
    bool = bool
    date = date
    datetime = datetime
    time = time

    numeric_types = {int, float, date, time, datetime}
    digits = '1234567890'
    decimals = set('1234567890-+eE.')
    integers = set('1234567890-+')
    nones = {'null', 'Null', 'NULL', '#N/A', '#n/a', "", 'None', None}
    none_type = type(None)

    date_formats = {  # Note: Only recognised ISO8601 formats are accepted.
        "NNNN-NN-NN": lambda x: date(*(int(i) for i in x.split("-"))),
        "NNNN-N-NN": lambda x: date(*(int(i) for i in x.split("-"))),
        "NNNN-NN-N": lambda x: date(*(int(i) for i in x.split("-"))),
        "NNNN-N-N": lambda x: date(*(int(i) for i in x.split("-"))),
        "NN-NN-NNNN": lambda x: date(*[int(i) for i in x.split("-")][::-1]),
        "N-NN-NNNN": lambda x: date(*[int(i) for i in x.split("-")][::-1]),
        "NN-N-NNNN": lambda x: date(*[int(i) for i in x.split("-")][::-1]),
        "N-N-NNNN": lambda x: date(*[int(i) for i in x.split("-")][::-1]),
        "NNNN.NN.NN": lambda x: date(*(int(i) for i in x.split("."))),
        "NNNN.N.NN": lambda x: date(*(int(i) for i in x.split("."))),
        "NNNN.NN.N": lambda x: date(*(int(i) for i in x.split("."))),
        "NNNN.N.N": lambda x: date(*(int(i) for i in x.split("."))),
        "NN.NN.NNNN": lambda x: date(*[int(i) for i in x.split(".")][::-1]),
        "N.NN.NNNN": lambda x: date(*[int(i) for i in x.split(".")][::-1]),
        "NN.N.NNNN": lambda x: date(*[int(i) for i in x.split(".")][::-1]),
        "N.N.NNNN": lambda x: date(*[int(i) for i in x.split(".")][::-1]),
        "NNNN/NN/NN": lambda x: date(*(int(i) for i in x.split("/"))),
        "NNNN/N/NN": lambda x: date(*(int(i) for i in x.split("/"))),
        "NNNN/NN/N": lambda x: date(*(int(i) for i in x.split("/"))),
        "NNNN/N/N": lambda x: date(*(int(i) for i in x.split("/"))),
        "NN/NN/NNNN": lambda x: date(*[int(i) for i in x.split("/")][::-1]),
        "N/NN/NNNN": lambda x: date(*[int(i) for i in x.split("/")][::-1]),
        "NN/N/NNNN": lambda x: date(*[int(i) for i in x.split("/")][::-1]),
        "N/N/NNNN": lambda x: date(*[int(i) for i in x.split("/")][::-1]),
        "NNNN NN NN": lambda x: date(*(int(i) for i in x.split(" "))),
        "NNNN N NN": lambda x: date(*(int(i) for i in x.split(" "))),
        "NNNN NN N": lambda x: date(*(int(i) for i in x.split(" "))),
        "NNNN N N": lambda x: date(*(int(i) for i in x.split(" "))),
        "NN NN NNNN": lambda x: date(*[int(i) for i in x.split(" ")][::-1]),
        "N N NNNN": lambda x: date(*[int(i) for i in x.split(" ")][::-1]),
        "NN N NNNN": lambda x: date(*[int(i) for i in x.split(" ")][::-1]),
        "N NN NNNN": lambda x: date(*[int(i) for i in x.split(" ")][::-1]),
        "NNNNNNNN": lambda x: date(*(int(x[:4]), int(x[4:6]), int(x[6:]))),
    }

    datetime_formats = {
        # Note: Only recognised ISO8601 formats are accepted.

        # year first
        'NNNN-NN-NNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x),  # -T
        'NNNN-NN-NNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x),

        'NNNN-NN-NN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, T=" "),  # - space
        'NNNN-NN-NN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, T=" "),

        'NNNN/NN/NNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/'),  # / T
        'NNNN/NN/NNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/'),

        'NNNN/NN/NN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', T=" "),  # / space
        'NNNN/NN/NN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', T=" "),

        'NNNN NN NNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd=' '),  # space T
        'NNNN NN NNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd=' '),

        'NNNN NN NN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd=' ', T=" "),  # space
        'NNNN NN NN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd=' ', T=" "),

        'NNNN.NN.NNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.'),  # dot T
        'NNNN.NN.NNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.'),

        'NNNN.NN.NN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', T=" "),  # dot
        'NNNN.NN.NN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', T=" "),


        # day first
        'NN-NN-NNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='-', T=' ', day_first=True),  # - T
        'NN-NN-NNNNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='-', T=' ', day_first=True),

        'NN-NN-NNNN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='-', T=' ', day_first=True),  # - space
        'NN-NN-NNNN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='-', T=' ', day_first=True),

        'NN/NN/NNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),  # / T
        'NN/NN/NNNNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),

        'NN/NN/NNNN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', T=' ', day_first=True),  # / space
        'NN/NN/NNNN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', T=' ', day_first=True),

        'NN NN NNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),  # space T
        'NN NN NNNNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),

        'NN NN NNNN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),  # space
        'NN NN NNNN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='/', day_first=True),

        'NN.NN.NNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', day_first=True),  # space T
        'NN.NN.NNNNTNN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', day_first=True),

        'NN.NN.NNNN NN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', day_first=True),  # space
        'NN.NN.NNNN NN:NN': lambda x: DataTypes.pattern_to_datetime(x, ymd='.', day_first=True),

        # compact formats - type 1
        'NNNNNNNNTNNNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=1),
        'NNNNNNNNTNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=1),
        'NNNNNNNNTNN': lambda x: DataTypes.pattern_to_datetime(x, compact=1),
        # compact formats - type 2
        'NNNNNNNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=2),
        'NNNNNNNNNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=2),
        'NNNNNNNNNNNNNN': lambda x: DataTypes.pattern_to_datetime(x, compact=2),
        # compact formats - type 3
        'NNNNNNNNTNN:NN:NN': lambda x: DataTypes.pattern_to_datetime(x, compact=3),
    }

    @staticmethod
    def pattern_to_datetime(iso_string, ymd=None, T=None, compact=0, day_first=False):
        assert isinstance(iso_string, str)
        if compact:
            s = iso_string
            if compact == 1:  # has T
                slices = [(0, 4, "-"), (4, 6, "-"), (6, 8, "T"), (9, 11, ":"), (11, 13, ":"), (13, len(s), "")]
            elif compact == 2:  # has no T.
                slices = [(0, 4, "-"), (4, 6, "-"), (6, 8, "T"), (8, 10, ":"), (10, 12, ":"), (12, len(s), "")]
            elif compact == 3:  # has T and :
                slices = [(0, 4, "-"), (4, 6, "-"), (6, 8, "T"), (9, 11, ":"), (12, 14, ":"), (15, len(s), "")]
            else:
                raise TypeError
            iso_string = "".join([s[a:b] + c for a, b, c in slices if b <= len(s)])
            iso_string = iso_string.rstrip(":")

        if day_first:
            s = iso_string
            iso_string = "".join((s[6:10], "-", s[3:5], "-", s[0:2], s[10:]))

        if "," in iso_string:
            iso_string = iso_string.replace(",", ".")

        dot = iso_string[::-1].find('.')
        if 0 < dot < 10:
            ix = len(iso_string) - dot
            microsecond = int(float(f"0{iso_string[ix - 1:]}") * 10 ** 6)
            iso_string = iso_string[:len(iso_string) - dot] + str(microsecond).rjust(6, "0")
        if ymd:
            iso_string = iso_string.replace(ymd, '-', 2)
        if T:
            iso_string = iso_string.replace(T, "T")
        return datetime.fromisoformat(iso_string)

    @staticmethod
    def to_json(v):
        if v is None:
            return v
        elif v is False:  # using isinstance(v, bool): won't work as False also is int of zero.
            return str(v)
        elif v is True:
            return str(v)
        elif isinstance(v, int):
            return v
        elif isinstance(v, str):
            return v
        elif isinstance(v, float):
            return v
        elif isinstance(v, datetime):
            return v.isoformat()
        elif isinstance(v, time):
            return v.isoformat()
        elif isinstance(v, date):
            return v.isoformat()
        else:
            raise TypeError(f"The datatype {type(v)} is not supported.")

    @staticmethod
    def from_json(v, dtype):
        if v in DataTypes.nones:
            if dtype is str and v == "":
                return ""
            else:
                return None
        if dtype is int:
            return int(v)
        elif dtype is str:
            return str(v)
        elif dtype is float:
            return float(v)
        elif dtype is bool:
            if v == 'False':
                return False
            elif v == 'True':
                return True
            else:
                raise ValueError(v)
        elif dtype is date:
            return date.fromisoformat(v)
        elif dtype is datetime:
            return datetime.fromisoformat(v)
        elif dtype is time:
            return time.fromisoformat(v)
        else:
            raise TypeError(f"The datatype {str(dtype)} is not supported.")

    @staticmethod
    def infer(v, dtype):
        if v in DataTypes.nones:
            return None
        if dtype is int:
            return DataTypes._infer_int(v)
        elif dtype is str:
            return DataTypes._infer_str(v)
        elif dtype is float:
            return DataTypes._infer_float(v)
        elif dtype is bool:
            return DataTypes._infer_bool(v)
        elif dtype is date:
            return DataTypes._infer_date(v)
        elif dtype is datetime:
            return DataTypes._infer_datetime(v)
        elif dtype is time:
            return DataTypes._infer_time(v)
        else:
            raise TypeError(f"The datatype {str(dtype)} is not supported.")

    @staticmethod
    def _infer_bool(value):
        if isinstance(value, bool):
            return value
        elif isinstance(value, int):
            raise ValueError("it's an integer.")
        elif isinstance(value, float):
            raise ValueError("it's a float.")
        elif isinstance(value, str):
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            else:
                raise ValueError
        else:
            raise ValueError

    @staticmethod
    def _infer_int(value):
        if isinstance(value, bool):
            raise ValueError("it's a boolean")
        if isinstance(value, int):
            return value
        elif isinstance(value, float):
            if int(value) == value:
                return int(value)
            raise ValueError("it's a float")
        elif isinstance(value, str):
            value = value.replace('"', '')  # "1,234" --> 1,234
            value = value.replace(" ", "")  # 1 234 --> 1234
            value = value.replace(',', '')  # 1,234 --> 1234
            value_set = set(value)
            if value_set - DataTypes.integers:  # set comparison.
                raise ValueError
            try:
                return int(float(value))
            except Exception:
                raise ValueError(f"{value} is not an integer")
        else:
            raise ValueError

    @staticmethod
    def _infer_float(value):
        if isinstance(value, int):
            raise ValueError("it's an integer")
        if isinstance(value, float):
            return value
        elif isinstance(value, str):
            value = value.replace('"', '')
            dot_index, comma_index = value.find('.'), value.find(',')
            if dot_index == comma_index == -1:
                pass  # there are no dots or commas.
            elif 0 < dot_index < comma_index:  # 1.234,567
                value = value.replace('.', '')  # --> 1234,567
                value = value.replace(',', '.')  # --> 1234.567
            elif dot_index > comma_index > 0:  # 1,234.678
                value = value.replace(',', '')

            elif comma_index and dot_index == -1:
                value = value.replace(',', '.')
            else:
                pass

            value_set = set(value)

            if not value_set.issubset(DataTypes.decimals):
                raise TypeError

            # if it's a string, do also
            # check that reverse conversion is valid,
            # otherwise we have loss of precision. F.ex.:
            # int(0.532) --> 0
            try:
                float_value = float(value)
            except Exception:
                raise ValueError(f"{value} is not a float.")
            if value_set.intersection('Ee'):  # it's scientific notation.
                v = value.lower()
                if v.count('e') != 1:
                    raise ValueError("only 1 e in scientific notation")

                e = v.find('e')
                v_float_part = float(v[:e])
                v_exponent = int(v[e + 1:])
                return float(f"{v_float_part}e{v_exponent}")

            elif "." in str(float_value) and not "." in value_set:
                # when traversing through Datatype.types,
                # integer is presumed to have failed for the column,
                # so we ignore this and turn it into a float...
                reconstructed_input = str(int(float_value))

            elif "." in value:
                precision = len(value) - value.index(".") - 1
                formatter = '{0:.' + str(precision) + 'f}'
                reconstructed_input = formatter.format(float_value)

            else:
                reconstructed_input = str(float_value)

            if value.lower() != reconstructed_input:
                raise ValueError

            return float_value
        else:
            raise ValueError

    @staticmethod
    def _infer_date(value):
        if isinstance(value, date):
            return value
        elif isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                pattern = "".join(["N" if n in DataTypes.digits else n for n in value])
                f = DataTypes.date_formats.get(pattern, None)
                if f:
                    return f(value)
                else:
                    raise ValueError
        else:
            raise ValueError

    @staticmethod
    def _infer_datetime(value):
        if isinstance(value, datetime):
            return value
        elif isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                if '.' in value:
                    dot = value.find('.', 11)  # 11 = len("1999.12.12")
                elif ',' in value:
                    dot = value.find(',', 11)
                else:
                    dot = len(value)

                pattern = "".join(["N" if n in DataTypes.digits else n for n in value[:dot]])
                f = DataTypes.datetime_formats.get(pattern, None)
                if f:
                    return f(value)
                else:
                    raise ValueError
        else:
            raise ValueError

    @staticmethod
    def _infer_time(value):
        if isinstance(value, time):
            return value
        elif isinstance(value, str):
            return time.fromisoformat(value)
        else:
            raise ValueError

    @staticmethod
    def _infer_str(value):
        if isinstance(value, str):
            return value
        else:
            return str(value)

    # Order is very important!
    types = [datetime, date, time, int, bool, float, str]

    @staticmethod
    def infer_range_from_slice(slice_item, length):
        assert isinstance(slice_item, slice)
        assert isinstance(length, int)
        item = slice_item

        if all((item.start is None,
               item.stop is None,
               item.step is None)):
            return 0, length, 1

        if item.step is None or item.step > 0:  # forward traverse
            step = 1
            if item.start is None:
                start = 0
            elif item.start < 0:
                start = length + item.start
            else:
                start = item.start

            if item.stop is None or item.stop > length:
                stop = length
            elif item.stop < 0:
                stop = length + item.stop
            else:
                stop = item.stop

            if start >= stop:
                return None  # empty list.

            return start, stop, step

        elif item.step < 0:  # item.step < 0: backward traverse
            step = item.step
            if item.start is None:  # a[::-1]
                start = length
            elif item.start < 0:
                start = item.start + length
            else:
                start = item.start

            if item.stop is None:
                stop = 0
            elif item.stop < 0:
                stop = item.stop + length
            else:
                stop = item.stop

            if start < stop:  # example [2:4:-1] --> []
                return None  # empty list.

            return start, stop, step

        else:
            return None


class Record(object):
    """ Internal datastructure used by StoredList"""

    ids = count()

    def __init__(self, stored_list):
        assert isinstance(stored_list, StoredList)
        self.stored_list = stored_list
        self.id = next(self.ids)  # id is used for storage retrieval. Not sequence.
        self._len = 0
        self.buffer = []
        self.loaded = False
        self.changed = False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.changed:
            c = "changed"
        else:
            c = "saved"
        return f"{self.__class__.__name__} ({c}) ({len(self)} items) {self.buffer[:5]} ..."

    def __sizeof__(self):
        return getsizeof(self.buffer)

    def __bool__(self):
        return bool(self.loaded)

    def __len__(self):
        if self.buffer:
            self._len = len(self.buffer)
        return self._len

    def save(self):
        """ save the buffer to disk and clears the buffer"""
        if self.changed:
            self.changed = False
            self.loaded = False
            self._len = len(self.buffer)
            try:
                self.stored_list.write(self)
            except Exception:
                self.stored_list.update(self)
            self.buffer.clear()
        else:
            pass

    def dump(self):
        """ wipes the buffer without saving """
        self.loaded = False
        if self.changed:
            raise Exception("data loss!")
        self.buffer.clear()

    def load(self):
        """
        loads the buffer savely from disk, by saving any
        other buffered data first.

        The total buffer will thereby never exceed set limit.
        """
        if self.loaded:
            return
        elif self.changed:
            return
        else:
            # 1. first save the buffer of everyone else.
            for r in self.stored_list.records:
                if r.changed:
                    r.save()
            # 2. then load own buffer.
            self.loaded = True
            self.changed = False
            self.buffer = self.stored_list.read(self)

    def append(self, value):
        self.loaded = True
        if len(self.buffer) < self.stored_list.buffer_limit:
            self.buffer.append(value)
            self.changed = True
        else:
            self.save()
            r = Record(self.stored_list)
            self.stored_list.records.append(r)
            r.append(value)

    def extend(self, items):
        self.loaded = True
        self.changed = True

        max_len = self.stored_list.buffer_limit
        # first store what fits.
        space = max_len - len(self)
        part, items = items[:space], items[space:]
        self.buffer.extend(part)
        if len(self.buffer) == max_len:
            self.save()

        # second store remaining items (if any)
        while items:
            part, items = items[:max_len], items[max_len:]
            r = Record(self.stored_list)
            self.stored_list.records.append(r)
            r.extend(part)

    def count(self, item):
        self.load()
        v = self.buffer.count(item)
        return v

    def index(self, item):
        self.load()
        try:
            v = self.buffer.index(item)
        except ValueError:
            v = None
        return v

    def insert(self, index, item):
        self.load()

        if len(self) < self.stored_list.buffer_limit:
            self.changed = True
            self.buffer.insert(index, item)
        else:
            self.save()
            r = Record(self.stored_list)
            self.stored_list.records.append(r)
            r.append(item)

    def pop(self, index=None):
        self.load()
        self.changed = True
        if index is None:
            return self.buffer.pop()
        else:
            return self.buffer.pop(index)

    def __contains__(self, item):
        self.load()
        return item in self.buffer

    def remove(self, item):
        self.load()
        try:
            self.buffer.remove(item)
            self.changed = True
        except ValueError:
            self.dump()

    def reverse(self):
        self.load()
        self.changed = True
        self.buffer.reverse()
        v = self.buffer[:]
        self.save()
        return v

    def __iter__(self):
        self.load()
        for v in self.buffer:
            yield v

    def __getitem__(self, item):
        self.load()
        return self.buffer[item]

    def __setitem__(self, key, value):
        self.load()
        self.changed = True
        self.buffer[key] = value

    def sort(self, reverse=False):
        self.load()
        self.changed = True
        self.buffer.sort(reverse=reverse)

    def __delitem__(self, key):
        self.load()
        self.changed = True
        del self.buffer[key]


class Buffer(object):
    """ internal storage class of the StoredList

    If you want to store using remote connections or another format
    simply override the methods in this class.
    """
    sql_create = "CREATE TABLE records (id INTEGER PRIMARY KEY, data BLOB);"
    sql_journal_off = "PRAGMA journal_mode = OFF"
    sql_sync_off = "PRAGMA synchronous = OFF "

    sql_delete = "DELETE FROM records WHERE id = ?"
    sql_insert = "INSERT INTO records VALUES (?, ?);"
    sql_update = "UPDATE records SET data = ? WHERE id=?;"
    sql_select = "SELECT data FROM records WHERE id=?"

    def __init__(self):
        self.file = self.windows_tempfile()
        self._conn = sqlite3.connect(self.file)  # SQLite3 connection
        with self._conn as c:
            c.execute(self.sql_create)
            c.execute(self.sql_journal_off)
            c.execute(self.sql_sync_off)

    def __del__(self):
        try:
            self._conn.close()
        except Exception:
            self._conn.interrupt()
            self._conn.close()
        self.file.unlink()

    def windows_tempfile(self, prefix='tmp', suffix='.db'):
        """ generates a safe tempfile which windows can't handle. """
        safe_folder = Path(gettempdir())
        while 1:
            n = "".join(choice(ascii_lowercase) for _ in range(5))
            name = f"{prefix}{n}{suffix}"
            p = safe_folder / name
            if not p.exists():
                break
        return p

    def buffer_update(self, record):
        assert isinstance(record, Record)
        data = zlib.compress(pickle.dumps(record.buffer))
        with self._conn as c:
            c.execute(self.sql_update, (data, record.id))  # UPDATE

    def buffer_write(self, record):
        assert isinstance(record, Record)
        data = zlib.compress(pickle.dumps(record.buffer))
        with self._conn as c:
            c.execute(self.sql_insert, (record.id, data))  # INSERT

    def buffer_read(self, record):
        assert isinstance(record, Record)
        with self._conn as c:
            q = c.execute(self.sql_select, (record.id,))  # READ
            data = q.fetchone()[0]
        return pickle.loads(zlib.decompress(data))

    def buffer_delete(self, record):
        assert isinstance(record, Record)
        with self._conn as c:
            c.execute(self.sql_delete, (record.id,))  # DELETE
        record.buffer.clear()


class StoredList(object):
    """ A type that behaves like a list, but stores items on disk """
    def __init__(self, buffer_limit=20_000):
        self.storage = None
        self.records = []
        if not isinstance(buffer_limit, int):
            raise TypeError(f'buffer_limit must be int, not {type(buffer_limit)}')
        self._buffer_limit = buffer_limit

    # internal data management methods
    # --------------------------------
    def _load_from_list(self, other):
        """ helper to load variables from another StoredList"""
        assert isinstance(other, StoredList)
        self.storage = other.storage
        self.records = other.records
        self._buffer_limit = other._buffer_limit

    def update(self, record):
        assert isinstance(record, Record)
        assert isinstance(self.storage, Buffer)
        self.storage.buffer_update(record)

    def write(self, record):
        if self.storage is None:
            self.storage = Buffer()
        assert isinstance(record, Record)
        assert isinstance(self.storage, Buffer)
        self.storage.buffer_write(record)

    def read(self, record):
        if self.storage is None:
            self.storage = Buffer()
        assert isinstance(self.storage, Buffer)
        assert isinstance(record, Record)
        return self.storage.buffer_read(record)

    def delete(self, record):
        assert isinstance(record, Record)
        assert isinstance(self.storage, Buffer)
        self.storage.buffer_delete(record)

    @property
    def buffer_limit(self):
        return self._buffer_limit

    @buffer_limit.setter
    def buffer_limit(self, value):
        if not isinstance(value, int):
            raise TypeError
        if self._buffer_limit < value: # reduce requires reload.
            L = StoredList(buffer_limit=value)
            for v in self:
                L.append(v)
            self._load_from_list(L)
        else:
            self._buffer_limit = value  # increase is permissive.

    def _normal_index(self, index):
        if not isinstance(index, int):
            raise TypeError
        if 0 <= index < len(self):
            return index

        if index < 0:
            return max(0, len(self) + index)

        if index >= len(self):
            return len(self)

    # public methods of a list
    # ------------------------
    def __len__(self):
        """ Return len(self). """
        return sum(len(r) for r in self.records)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"{self.__class__.__name__} {len(self)}"

    def append(self, value):
        """ Append object to the end of the list. """
        if not self.records:
            r = Record(self)
            self.records.append(r)
        r = self.records[-1]
        r.append(value)

    def clear(self):
        """ Remove all items from list. """
        new_list = StoredList(buffer_limit=self._buffer_limit)
        self._load_from_list(new_list)

    def copy(self):
        """ Return a shallow copy of the list. """
        L = StoredList(buffer_limit=self._buffer_limit)
        for i in self:
            L.append(i)
        return L

    def count(self, item):
        """ Return number of occurrences of value. """
        counter = 0
        for r in self.records:
            counter += r.count(item)
        return counter

    def extend(self, items):
        """ Extend list by appending elements from the iterable. """
        if not self.records:
            r = Record(self)
            self.records.append(r)
        r = self.records[-1]
        r.extend(items)

    def index(self, item):
        """
        Return first index of value.

        Raises ValueError if the value is not present.
        """
        counter = 0
        for r in self.records:
            assert isinstance(r, Record)
            v = r.index(item)
            if v is not None:
                return counter + v
            counter += len(r)

        raise ValueError(f"{item} not found")

    def insert(self, index, item):
        """ Insert object before index. """
        index = self._normal_index(index)

        counter = 0
        for r in self.records:
            if counter <= index <= counter + len(r):
                if counter == 0:
                    r.insert(index, item)
                else:
                    r.insert(index % counter, item)
                break

    def pop(self, index=None):
        """
        Remove and return item at index (default last).

        Raises IndexError if list is empty or index is out of range.
        """
        if index is None:
            r = self.records[-1]
            return r.pop(None)

        counter = 0
        for r in self.records:
            if counter < index <= counter + len(r):
                return r.pop(index % counter)

    def remove(self, item):
        """
        Remove first occurrence of value.

        Raises ValueError if the value is not present.
        """
        counter = 0
        for r in self.records:
            if item in r:
                r.remove(item)
                counter += 1
                break
        if not counter:
            raise ValueError(f"{item} not found")

    def reverse(self):
        """ Reverse *IN PLACE*. """
        new_list = StoredList(buffer_limit=self._buffer_limit)
        for r in reversed(self.records):
            data = r.reverse()
            new_list.extend(data)

        self._load_from_list(new_list)

    def sort(self, reverse=False):
        """ Implements a hybrid of quicksort and merge sort.

        See more on https://en.wikipedia.org/wiki/External_sorting
        """
        for r in self.records:
            r.sort(reverse=reverse)
            r.save()

        wb = [r for r in self.records]
        if len(wb) == 1:
            C = StoredList(self._buffer_limit)
            r = wb.pop()
            C.extend(r)
            wb.append(C)
        else:
            while len(wb) > 1:
                A = wb.pop(0)
                A = iter(A)
                B = wb.pop(0)
                B = iter(B)

                C = StoredList(self._buffer_limit)
                a, b = next(A), next(B)

                while True:
                    if (reverse and a >= b) or (not reverse and a <= b):
                        C.append(a)
                        try:
                            a = next(A)
                        except StopIteration:
                            C.append(b)
                            C.extend(list(B))
                            break
                    else:
                        C.append(b)
                        try:
                            b = next(B)
                        except StopIteration:
                            C.append(a)
                            C.extend(list(A))
                            break

                wb.append(C)

        L = wb.pop()
        assert len(L) == len(self), (len(L), len(self))
        self._load_from_list(L)

    def __add__(self, other):
        """ Return self+value. """
        if isinstance(other, (StoredList, list)):
            new_list = StoredList(buffer_limit=self._buffer_limit)
            new_list.extend(other)
            return new_list
        else:
            raise TypeError

    def __contains__(self, item):
        """ Return key in self. """
        for r in self.records:
            if item in r:
                return True
        return False

    def __delitem__(self, index):
        """ Delete self[key]. """
        if index > len(self) or index < 0:
            raise IndexError

        counter = 0
        for r in self.records:
            if counter < index < counter + len(r):
                if counter == 0:
                    del r[index]
                else:
                    del r[index % counter]
                return

    def __eq__(self, other):
        """ Return self==value. """
        if not isinstance(other, (StoredList, list)):
            raise TypeError

        if len(self) != len(other):
            return False

        return not any(a != b for a, b in zip(self, other))

    def _get_item(self, index):
        assert isinstance(index, int)
        index = self._normal_index(index)
        counter = 0
        for r in self.records:
            if counter <= index < counter + len(r):
                if counter == 0:
                    return r[index]
                else:
                    return r[index % counter]
            else:
                counter += len(r)

    def __getitem__(self, item):
        """ x.__getitem__(y) <==> x[y] """
        if isinstance(item, int):
            return self._get_item(item)

        assert isinstance(item, slice)
        L = StoredList(buffer_limit=self._buffer_limit)
        slc = DataTypes.infer_range_from_slice(item, len(self))
        if slc is None:
            return L
        start, stop, step = slc

        if step > 0:
            counter = 0
            for r in self.records:
                if counter > stop:
                    break
                if counter + len(r) > start:
                    start_ix = max(0, start - counter)
                    start_ix += (start_ix - start) % step
                    end_ix = min(stop-counter, len(r))
                    data = r[start_ix:end_ix:step]
                    L.extend(data)

                counter += len(r)

        else:  # step < 0  # item.step < 0: backward traverse

            counter = len(self)
            for r in reversed(self.records):
                if counter < stop:
                    break
                if counter - len(r) > start:
                    start_ix = max(start - counter, 0)
                    start_ix -= (start_ix - start) % step
                    end_ix = None if item.stop is None else max(stop-counter, counter - len(r))
                    L.extend(r[start_ix:end_ix:step])

                counter -= len(r)

        return L

    def __ge__(self, other):
        """ Return self>=value. """
        for a, b in zip(self, other):
            if a < b:
                return False
        return True

    def __gt__(self, other):
        """ Return self>value. """
        for a, b in zip(self, other):
            if a <= b:
                return False
        return True

    def __iadd__(self, other):
        """ Implement self+=value. """
        if not isinstance(other, (StoredList, list)):
            raise TypeError
        self.extend(other)
        return self

    def __imul__(self, value):
        """ Implement self*=value. """
        if not isinstance(value, int):
            raise TypeError
        if value <= 0:
            raise ValueError
        elif value == 1:
            return self
        else:
            new_list = StoredList(self._buffer_limit)
            for i in range(value):
                new_list += self
            return new_list

    def __iter__(self):
        """ Implement iter(self). """
        for r in self.records:
            for v in r:
                yield v

    def __le__(self, other):
        """ Return self<=value. """
        for a, b in zip(self, other):
            if a > b:
                return False
        return True

    def __lt__(self, other):
        """ Return self<value. """
        for a, b in zip(self, other):
            if a >= b:
                return False
        return True

    def __mul__(self, value):
        """ Return self*value. """
        if not isinstance(value, int):
            raise TypeError
        if value <= 0:
            raise ValueError
        new_list = StoredList(buffer_limit=self._buffer_limit)
        for i in range(value):
            new_list += self
        return new_list

    def __ne__(self, other):
        """ Return self!=value. """
        if not isinstance(other, (StoredList, list)):
            raise TypeError

        if len(self) != len(other):
            return True

        return any(a != b for a, b in zip(self, other))

    def __reversed__(self):
        """ Return a reverse iterator over the list. """
        for r in reversed(self.records):
            data = r.reverse()
            for i in data:
                yield i

    def __rmul__(self, value):
        """ Return value*self. """
        return self.__mul__(value)

    def __setitem__(self, key, value):
        """ Set self[key] to value. """
        if not isinstance(key, int):
            raise TypeError
        key = self._normal_index(key)

        counter = 0
        for r in self.records:
            if counter <= key < len(r) + counter:
                r[key] = value
                break

    def __sizeof__(self):
        """ Return the size of the list in memory, in bytes. """
        return getsizeof(self.records)

    __hash__ = None

    def __copy__(self):
        new_list = StoredList(self._buffer_limit)
        for i in self:
            new_list.append(i)
        return new_list


class CommonColumn(object):
    """ A Stored list with the necessary metadata to imitate a Column """
    def __init__(self, header, datatype, allow_empty, *args):
        assert isinstance(header, str)
        self.header = header
        assert isinstance(datatype, type)
        assert hasattr(DataTypes, datatype.__name__)
        self.datatype = datatype
        assert isinstance(allow_empty, bool)
        self.allow_empty = allow_empty

    def __eq__(self, other):
        if not isinstance(other, (StoredColumn, Column)):
            a, b = self.__class__.__name__, other.__class__.__name__
            raise TypeError(f"cannot compare {a} with {b}")

        return all([
            self.header == other.header,
            self.datatype == other.datatype,
            self.allow_empty == other.allow_empty,
            len(self) == len(other),
            all(a == b for a, b in zip(self, other))
        ])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"Column({self.header},{self.datatype},{self.allow_empty}) # ({len(self)} rows)"

    def __copy__(self):
        return self.copy()

    def copy(self):
        return Column(self.header, self.datatype, self.allow_empty, data=self[:])

    def to_json(self):
        return json.dumps({
            'header': self.header,
            'datatype': self.datatype.__name__,
            'allow_empty': self.allow_empty,
            'data': json.dumps([DataTypes.to_json(v) for v in self])
        })

    @classmethod
    def from_json(cls, json_):
        j = json.loads(json_)
        j['datatype'] = dtype = getattr(DataTypes, j['datatype'])
        j['data'] = [DataTypes.from_json(v, dtype) for v in json.loads(j['data'])]
        return Column(**j)

    def type_check(self, value):
        """ helper that does nothing unless it raises an exception. """
        if value is None:
            if not self.allow_empty:
                raise ValueError("None is not permitted.")
            return
        if not isinstance(value, self.datatype):
            raise TypeError(f"{value} is not of type {self.datatype}")

    def append(self, __object) -> None:
        self.type_check(__object)
        super().append(__object)  # call handled by super on the sub-class.

    def replace(self, values) -> None:
        assert isinstance(values, list)
        if len(values) != len(self):
            raise ValueError("input is not of same length as column.")
        for v in values:
            self.type_check(v)
        self.clear()
        self.extend(values)

    def __setitem__(self, key, value):
        self.type_check(value)
        super().__setitem__(key, value)


# class Column(list):
class Column(CommonColumn, list):  # MRO: CC first, then list.
    """ A list with metadata for use in the Table class. """
    def __init__(self, header, datatype, allow_empty, data=None):
        CommonColumn.__init__(self, header, datatype, allow_empty)  # first init the cc attrs.
        list.__init__(self)  # then init the list attrs.

        if data:
            for v in data:
                self.append(v)  # append does the type check.


class StoredColumn(CommonColumn, StoredList):  # MRO: CC first, then StoredList.
    """ A Stored list with the necessary metadata to imitate a Column """
    def __init__(self, header, datatype, allow_empty, data=None):
        CommonColumn.__init__(self, header, datatype, allow_empty)  # first init the cc attrs.
        StoredList.__init__(self)  # then init the list attrs.

        if data:
            for v in data:
                self.append(v)  # append does the type check.


class Table(object):
    new_tables_use_disk = False

    """ The main workhorse for data processing. """
    def __init__(self, **kwargs):
        self.columns = {}
        self._use_disk = kwargs.pop('use_disk', self.new_tables_use_disk)
        self.metadata = {**kwargs}

    @property
    def use_disk(self):
        return self._use_disk

    @use_disk.setter
    def use_disk(self, value):
        if not isinstance(value, bool):
            raise TypeError(str(value))
        if self._use_disk == value:
            return

        self._use_disk = value
        if value is True:
            C = StoredColumn
        else:
            C = Column

        for col_name, column in self.columns.items():
            self.columns[col_name] = C(col_name, column.datatype, column.allow_empty, data=column)

    def __eq__(self, other):
        if not isinstance(other, Table):
            a, b = self.__class__.__name__, other.__class__.__name__
            raise TypeError(f"cannot compare {a} with {b}")
        if self.metadata != other.metadata:
            return False
        if any(a != b for a, b in zip(self.columns.values(), other.columns.values())):
            return False
        return True

    def __len__(self):
        """ returns length of longest column."""
        return max(len(c) for c in self.columns.values())

    def __bool__(self):
        return any(self.columns)

    def __copy__(self):
        t = Table(use_disk=self._use_disk)
        for col in self.columns.values():
            t.add_column(col.header, col.datatype, col.allow_empty, data=col[:])
        t.metadata = self.metadata.copy()
        return t

    def __repr__(self):
        m = self.metadata.copy()
        m['use_disk'] = self._use_disk
        kwargs = ", ".join(f"{k}={v}" for k,v in sorted(m.items()))
        return f"{self.__class__.__name__}({kwargs})"

    def __str__(self):
        variation = ""
        lengths = {k: len(v) for k, v in self.columns.items()}
        if len(set(lengths.values())) != 1:
            longest_col = max(lengths.values())
            variation = f"(except {', '.join([f'{k}({v})' for k, v in lengths.items() if v < longest_col])})"
        return f"{self.__class__.__name__}() # {len(self.columns)} columns x {len(self)} rows {variation}"

    def show(self, *items):
        """ shows the table.
        param: items: column names, slice.
        :returns None. Output is printed to stdout.
        """
        if any(not isinstance(i, (str, slice)) for i in items):
            raise SyntaxError(f"unexpected input: {[not isinstance(i, (str, slice)) for i in items]}")

        slices = [i for i in items if isinstance(i, slice)]
        if len(slices) > 2:
            raise SyntaxError("1 > slices")
        if not slices:
            slc = slice(0, len(self), None)
        else:
            slc = slices[0]
        assert isinstance(slc, slice)

        headers = [i for i in items if isinstance(i, str)]
        if any(h not in self.columns for h in headers):
            raise ValueError(f"column not found: {[h for h in headers if h not in self.columns]}")
        if not headers:
            headers = list(self.columns)

        # starting to produce output
        c_lens = {}
        for h in headers:
            col = self.columns[h]
            assert isinstance(col, (Column, StoredColumn))
            c_lens[h] = max(
                [len(col.header), len(str(col.datatype.__name__)), len(str(False))] + [len(str(v)) for v in col[slc]])

        def adjust(v, length):
            if v is None:
                return str(v).ljust(length)
            elif isinstance(v, str):
                return v.ljust(length)
            else:
                return str(v).rjust(length)

        print("+", "+".join(["=" * c_lens[h] for h in headers]), "+", sep="")
        print("|", "|".join([h.center(c_lens[h], " ") for h in headers]), "|", sep="")
        print("|", "|".join([self.columns[h].datatype.__name__.center(c_lens[h], " ") for h in headers]), "|", sep="")
        print("|", "|".join([str(self.columns[h].allow_empty).center(c_lens[h], " ") for h in headers]), "|", sep="")
        print("+", "+".join(["-" * c_lens[h] for h in headers]), "+", sep="")
        for row in self.filter(*tuple(headers) + (slc,)):
            print("|", "|".join([adjust(v, c_lens[h]) for v, h in zip(row, headers)]), "|", sep="")
        print("+", "+".join(["=" * c_lens[h] for h in headers]), "+", sep="")

    def copy(self):
        return self.__copy__()

    def to_json(self):
        return json.dumps({
            'metadata': self.metadata,
            'columns': [c.to_json() for c in self.columns.values()]
        })

    @classmethod
    def from_json(cls, json_):
        t = Table()
        data = json.loads(json_)
        t.metadata = data['metadata']
        for c in data['columns']:
            col = Column.from_json(c)
            col.header = t.check_for_duplicate_header(col.header)
            t.columns[col.header] = col
        return t

    @classmethod
    def from_file(cls, path, **kwargs):
        """ reads path and returns 1 or more tables.
        Use `list(Table.from_file(...))` to obtain all tables """
        for table in file_reader(path, **kwargs):
            yield table

    def check_for_duplicate_header(self, header):
        assert isinstance(header, str)
        if not header:
            header = 'None'
        new_header = header
        counter = count(start=1)
        while new_header in self.columns:
            new_header = f"{header}_{next(counter)}"  # valid attr names must be ascii.
        return new_header

    def add_column(self, header, datatype, allow_empty=False, data=None):
        assert isinstance(header, str)
        header = self.check_for_duplicate_header(header)
        if self._use_disk is False:
            self.columns[header] = Column(header, datatype, allow_empty, data=data)
        else:
            self.columns[header] = StoredColumn(header, datatype, allow_empty, data=data)

    def add_row(self, values):
        if not isinstance(values, tuple):
            raise TypeError(f"expected tuple, got {type(values)}")
        if len(values) != len(self.columns):
            raise ValueError(f"expected {len(self.columns)} values not {len(values)}: {values}")
        for value, col in zip(values, self.columns.values()):
            col.append(value)

    def __contains__(self, item):
        return item in self.columns

    def __iter__(self):
        raise AttributeError("use Table.rows or Table.columns")

    def _slice(self, item=None):
        """ transforms a slice into start,stop,step"""
        if not item:
            item = slice(None, len(self), None)
        else:
            assert isinstance(item, slice)

        if item.stop < 0:
            start = len(self) + item.stop
            stop = len(self)
            step = 1 if item.step is None else item.step
        else:
            start = 0 if item.start is None else item.start
            stop = item.stop
            step = 1 if item.step is None else item.step
        return start, stop, step

    def __getitem__(self, item):
        """ returns rows as a tuple """
        if isinstance(item, int):
            item = slice(item, item + 1, 1)
        if isinstance(item, slice):
            t = Table(use_disk=self._use_disk)
            for col in self.columns.values():
                t.add_column(col.header, col.datatype, col.allow_empty, col[item])
            return t
        else:
            return self.columns[item]

    def __setitem__(self, key, value):
        if key in self.columns and isinstance(value, list):
            c = self.columns[key]
            c.clear()
            for v in value:
                c.append(v)
        else:
            raise TypeError(f"Use add_column to add_column: {key}")

    def __delitem__(self, key):
        """ delete column as key """
        if key in self.columns:
            del self.columns[key]
        else:
            raise KeyError(f"key not found")

    def __setattr__(self, name, value):
        if isinstance(name, str) and hasattr(self, name):
            if name in self.columns and isinstance(value, list):
                col = self.columns[name]
                col.replace(value)
                return
        super().__setattr__(name, value)

    def compare(self, other):
        """ compares the metadata of two tables."""
        if not isinstance(other, Table):
            a, b = self.__class__.__name__, other.__class__.__name__
            raise TypeError(f"cannot compare type {b} with {a}")

        if self.metadata != other.metadata:
            raise ValueError("tables have different metadata.")
        for a, b in [[self, other], [other, self]]:  # check both dictionaries.
            for name, col in a.columns.items():
                if name not in b.columns:
                    raise ValueError(f"Column {name} not in other")
                col2 = b.columns[name]
                if col.datatype != col2.datatype:
                    raise ValueError(f"Column {name}.datatype different: {col.datatype}, {col2.datatype}")
                if col.allow_empty != col2.allow_empty:
                    raise ValueError(f"Column {name}.allow_empty is different")
        return True

    def __iadd__(self, other):
        """ enables Table_1 += Table_2 """
        self.compare(other)
        for h, col in self.columns.items():
            c2 = other.columns[h]
            col.extend(c2[:])
        return self

    def __add__(self, other):
        """ enables Table_3 = Table_1 + Table_2 """
        self.compare(other)
        cp = self.copy()
        for h, col in cp.columns.items():
            c2 = other.columns[h]
            col.extend(c2[:])
        return cp

    @property
    def rows(self):
        """ enables iteration

        for row in table.rows:
            print(row)

        """
        for ix in range(len(self)):
            item = tuple(c[ix] if ix < len(c) else None for c in self.columns.values())
            yield item

    def index(self, *args):
        """ Creates index on *args columns as d[(key tuple, )] = {index1, index2, ...} """
        idx = defaultdict(set)
        for ix, key in enumerate(self.filter(*args)):
            idx[key].add(ix)
        return idx

    def _sort_index(self, **kwargs):
        if not isinstance(kwargs, dict):
            raise ValueError("Expected keyword arguments")
        if not kwargs:
            kwargs = {c: False for c in self.columns}

        for k, v in kwargs.items():
            if k not in self.columns:
                raise ValueError(f"no column {k}")
            if not isinstance(v, bool):
                raise ValueError(f"{k} was mapped to {v} - a non-boolean")
        none_substitute = float('-inf')

        rank = {i: tuple() for i in range(len(self))}
        for key in kwargs:
            unique_values = {v: 0 for v in self.columns[key] if v is not None}
            for r, v in enumerate(sorted(unique_values, reverse=kwargs[key])):
                unique_values[v] = r
            for ix, v in enumerate(self.columns[key]):
                rank[ix] += (unique_values.get(v, none_substitute),)

        new_order = [(r, i) for i, r in rank.items()]  # tuples are listed and sort...
        new_order.sort()
        sorted_index = [i for r, i in new_order]  # new index is extracted.

        rank.clear()  # free memory.
        new_order.clear()

        return sorted_index

    def sort(self, **kwargs):
        """ Perform multi-pass sorting with precedence given order of column names.
        :param kwargs: keys: columns, values: 'reverse' as boolean.
        """
        sorted_index = self._sort_index(**kwargs)
        for col_name, col in self.columns.items():
            assert isinstance(col, (StoredColumn, Column))
            col.replace(values=[col[ix] for ix in sorted_index])

    def is_sorted(self, **kwargs):
        sorted_index = self._sort_index(**kwargs)
        if any(ix != i for ix, i in enumerate(sorted_index)):
            return False
        return True

    def filter(self, *items):
        """ enables iteration on a limited number of headers:

        >>> table.columns
        'a','b','c','d','e'

        for row in table.filter('b', 'a', 'a', 'c'):
            b,a,a,c = row ...

        returns values in same order as headers. """
        if any(not isinstance(i, (str, slice)) for i in items):
            raise SyntaxError(f"unexpected input: {[not isinstance(i, (str, slice)) for i in items]}")

        slices = [i for i in items if isinstance(i, slice)]
        if len(slices) > 2:
            raise SyntaxError("1 > slices")

        if not slices:
            slc = slice(None, len(self), None)
        else:
            slc = slices[0]
        assert isinstance(slc, slice)

        headers = [i for i in items if isinstance(i, str)]
        if any(h not in self.columns for h in headers):
            raise ValueError(f"column not found: {[h for h in headers if h not in self.columns]}")

        sss = DataTypes.infer_range_from_slice(slc, len(self))
        if sss is None:
            return

        L = [self.columns[h] for h in headers]
        for ix in range(*sss):
            item = tuple(c[ix] if ix < len(c) else None for c in L)
            yield item

    def all(self, **kwargs):
        """
        returns Table for rows where ALL kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you remember to add the ** in front of your dict?")
        if not all(k in self.columns for k in kwargs):
            raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in self.columns]}")

        ixs = None
        for k, v in kwargs.items():
            col = self.columns[k]
            if ixs is None:  # first header.
                if callable(v):
                    ix2 = {ix for ix, i in enumerate(col) if v(i)}
                else:
                    ix2 = {ix for ix, i in enumerate(col) if v == i}

            else:  # remaining headers.
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

        t = Table(use_disk=self._use_disk)
        for col in self.columns.values():
            t.add_column(col.header, col.datatype, col.allow_empty, data=[col[ix] for ix in ixs])
        return t

    def any(self, **kwargs):
        """
        returns Table for rows where ANY kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you remember to add the ** in front of your dict?")

        ixs = set()
        for k, v in kwargs.items():
            col = self.columns[k]
            if callable(v):
                ix2 = {ix for ix, r in enumerate(col) if v(r)}
            else:
                ix2 = {ix for ix, r in enumerate(col) if v == r}
            ixs.update(ix2)

        t = Table(use_disk=self._use_disk)
        for col in self.columns.values():
            t.add_column(col.header, col.datatype, col.allow_empty, data=[col[ix] for ix in ixs])
        return t

    def _join_type_check(self, other, keys, columns):
        if not isinstance(other, Table):
            raise TypeError(f"other expected other to be type Table, not {type(other)}")
        if not isinstance(keys, list) and all(isinstance(k, str) for k in keys):
            raise TypeError(f"Expected keys as list of strings, not {type(keys)}")
        union = list(self.columns) + list(other.columns)
        if not all(k in self.columns and k in other.columns for k in keys):
            raise ValueError(f"key(s) not found: {[k for k in keys if k not in union]}")
        if not all(k in union for k in columns):
            raise ValueError(f"column(s) not found: {[k for k in keys if k not in union]}")

    def left_join(self, other, keys, columns):
        """
        :param other: self, other = (left, right)
        :param keys: list of keys for the join
        :param columns: list of columns to retain
        :return: new table

        Example:
        SQL:   SELECT number, letter FROM left LEFT JOIN right on left.colour == right.colour
        Table: left_join = left_table.left_join(right_table, keys=['colour'], columns=['number', 'letter'])
        """
        self._join_type_check(other, keys, columns)  # raises if error

        left_join = Table(use_disk=self._use_disk)
        for col_name in columns:
            if col_name in self.columns:
                col = self.columns[col_name]
            elif col_name in other.columns:
                col = other.columns[col_name]
            else:
                raise ValueError(f"column name '{col_name}' not in any table.")
            left_join.add_column(col_name, col.datatype, allow_empty=True)

        left_ixs = range(len(self))
        right_idx = other.index(*keys)

        for left_ix in left_ixs:
            key = tuple(self[h][left_ix] for h in keys)
            right_ixs = right_idx.get(key, (None,))
            for right_ix in right_ixs:
                for col_name, column in left_join.columns.items():
                    if col_name in self:
                        column.append(self[col_name][left_ix])
                    elif col_name in other:
                        if right_ix is not None:
                            column.append(other[col_name][right_ix])
                        else:
                            column.append(None)
                    else:
                        raise Exception('bad logic')
        return left_join

    def inner_join(self, other, keys, columns):
        """
        :param other: table
        :param keys: list of keys
        :param columns: list of columns to retain
        :return: new Table

        Example:
        SQL:   SELECT number, letter FROM left INNER JOIN right ON left.colour == right.colour
        Table: inner_join = left_table.inner_join_with(right_table, keys=['colour'],  columns=['number','letter'])
        """
        self._join_type_check(other, keys, columns)  # raises if error

        inner_join = Table(use_disk=self._use_disk)
        for col_name in columns:
            if col_name in self.columns:
                col = self.columns[col_name]
            elif col_name in other.columns:
                col = other.columns[col_name]
            else:
                raise ValueError(f"column name '{col_name}' not in any table.")
            inner_join.add_column(col_name, col.datatype, allow_empty=True)

        key_union = set(self.filter(*keys)).intersection(set(other.filter(*keys)))

        left_ixs = self.index(*keys)
        right_ixs = other.index(*keys)

        for key in key_union:
            for left_ix in left_ixs.get(key, set()):
                for right_ix in right_ixs.get(key, set()):
                    for col_name, column in inner_join.columns.items():
                        if col_name in self:
                            column.append(self[col_name][left_ix])
                        elif col_name in other:
                            column.append(other[col_name][right_ix])
                        else:
                            raise Exception("bad logic.")
        return inner_join

    def outer_join(self, other, keys, columns):
        """
        :param other: table
        :param keys: list of keys
        :param columns: list of columns to retain
        :return: new Table

        Example:
        SQL:   SELECT number, letter FROM left OUTER JOIN right ON left.colour == right.colour
        Table: outer_join = left_table.outer_join(right_table, keys=['colour'], columns=['number','letter'])
        """
        self._join_type_check(other, keys, columns)  # raises if error

        outer_join = Table(use_disk=self._use_disk)
        for col_name in columns:
            if col_name in self.columns:
                col = self.columns[col_name]
            elif col_name in other.columns:
                col = other.columns[col_name]
            else:
                raise ValueError(f"column name '{col_name}' not in any table.")
            outer_join.add_column(col_name, col.datatype, allow_empty=True)

        left_ixs = range(len(self))
        right_idx = other.index(*keys)
        right_keyset = set(right_idx)

        for left_ix in left_ixs:
            key = tuple(self[h][left_ix] for h in keys)
            right_ixs = right_idx.get(key, (None,))
            right_keyset.discard(key)
            for right_ix in right_ixs:
                for col_name, column in outer_join.columns.items():
                    if col_name in self:
                        column.append(self[col_name][left_ix])
                    elif col_name in other:
                        if right_ix is not None:
                            column.append(other[col_name][right_ix])
                        else:
                            column.append(None)
                    else:
                        raise Exception('bad logic')

        for right_key in right_keyset:
            for right_ix in right_idx[right_key]:
                for col_name, column in outer_join.columns.items():
                    if col_name in self:
                        column.append(None)
                    elif col_name in other:
                        column.append(other[col_name][right_ix])
                    else:
                        raise Exception('bad logic')
        return outer_join

    def groupby(self, keys, functions):
        g = GroupBy(keys=keys, functions=functions)
        g += self
        return g


class GroupbyFunction(object):
    def __init__(self, datatype):
        hasattr(DataTypes, datatype.__name__)
        self.datatype = datatype


class Limit(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.value = None
        self.f = None

    def update(self, value):
        if value is None:
            pass
        elif self.value is None:
            self.value = value
        else:
            self.value = self.f((value, self.value))


class Max(Limit):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.f = max


class Min(Limit):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.f = min


class Sum(Limit):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.f = sum


class First(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.value = None

    def update(self, value):
        if self.value is None:
            if value is not None:
                self.value = value


class Last(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.value = None

    def update(self, value):
        if value is not None:
            self.value = value


class Count(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype=int)  # datatype will be int no matter what type is given.
        self.value = 0

    def update(self, value):
        if value is not None:
            self.value += 1


class CountUnique(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype=int)  # datatype will be int no matter what type is given.
        self.items = set()

    def update(self, value):
        if value is not None:
            self.items.add(value)
            self.value = len(self.items)


class Average(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype=float)  # datatype will be float no matter what type is given.
        self.sum = 0
        self.count = 0
        self.value = 0

    def update(self, value):
        if value is not None:
            self.sum += value
            self.count += 1
            self.value = self.sum / self.count


class StandardDeviation(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype=float)  # datatype will be float no matter what type is given.
        self.count = 0
        self.mean = 0
        self.c = 0.0

    def update(self, value):
        if value is not None:
            self.count += 1
            dt = value - self.mean
            self.mean += dt / self.count
            self.c += dt * (value - self.mean)

    @property
    def value(self):
        if self.count <= 1:
            return 0.0
        variance = self.c / (self.count - 1)
        return variance ** (1 / 2)


class Histogram(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.hist = defaultdict(int)

    def update(self, value):
        if value is not None:
            self.hist[value] += 1


class Median(Histogram):
    def __init__(self, datatype):
        super().__init__(datatype)

    @property
    def value(self):
        midpoint = sum(self.hist.values()) / 2
        total = 0
        for k, v in self.hist.items():
            total += v
            if total > midpoint:
                return k


class Mode(Histogram):
    def __init__(self, datatype):
        super().__init__(datatype)

    @property
    def value(self):
        L = [(v, k) for k, v in self.hist.items()]
        L.sort(reverse=True)
        frequency, most_frequent = L[0]  # top of the list.
        return most_frequent


class GroupBy(object):
    functions = [
        Max, Min, Sum, First, Last,
        Count, CountUnique,
        Average, StandardDeviation, Median, Mode
    ]
    function_names = {f.__name__: f for f in functions}

    def __init__(self, keys, functions):
        """
        :param keys: headers for grouping
        :param functions: list of headers and functions.
        :return: None.
        """
        assert isinstance(keys, list)
        assert len(set(keys)) == len(keys), "duplicate key found."
        self.keys = keys

        assert isinstance(functions, list)
        assert all(len(i) == 2 for i in functions)
        assert all(isinstance(a, str) and issubclass(b, GroupbyFunction) for a, b in functions)
        self.groupby_functions = functions  # list with header name and function name

        self.output = None   # class Table.
        self.required_headers = None  # headers for reading input.
        self.data = defaultdict(list)  # key: [list of groupby functions]
        self.function_classes = []  # initiated functions.

        # Order is preserved so that this is doable:
        # for header, function, function_instances in zip(self.groupby_functions, self.function_classes) ....

    def setup(self, table):
        """ helper to setup the group functions """
        self.output = Table()
        self.required_headers = self.keys + [h for h, fn in self.groupby_functions]

        for h in self.keys:
            col = table[h]
            self.output.add_column(header=h, datatype=col.datatype, allow_empty=False)  # add column for keys

        self.function_classes = []
        for h, fn in self.groupby_functions:
            col = table[h]
            assert isinstance(col, Column)
            f_instance = fn(col.datatype)
            assert isinstance(f_instance, GroupbyFunction)
            self.function_classes.append(f_instance)

            function_name = f"{fn.__name__}({h})"
            self.output.add_column(header=function_name, datatype=f_instance.datatype, allow_empty=True)  # add column for fn's.

    def __iadd__(self, other):
        """
        To view results use `for row in self.rows`
        To add more data use self += new data (Table)
        """
        assert isinstance(other, Table)
        if self.output is None:
            self.setup(other)
        else:
            self.output.compare(other)  # this will raise if there are problems

        for row in other.filter(*self.required_headers):
            d = {h: v for h, v in zip(self.required_headers, row)}
            key = tuple([d[k] for k in self.keys])
            functions = self.data.get(key)
            if not functions:
                functions = [fn.__class__(fn.datatype) for fn in self.function_classes]
                self.data[key] = functions

            for (h, fn), f in zip(self.groupby_functions, functions):
                f.update(d[h])
        return self

    def _generate_table(self):
        """ helper that generates the result for .table and .rows """
        for key, functions in self.data.items():
            row = key + tuple(fn.value for fn in functions)
            self.output.add_row(row)
        self.data.clear()  # hereby we only create the table once.
        self.output.sort(**{k: False for k in self.keys})

    @property
    def table(self):
        """ returns Table """
        if self.output is None:
            return None

        if self.data:
            self._generate_table()

        assert isinstance(self.output, Table)
        return self.output

    @property
    def rows(self):
        """ returns iterator for Groupby.rows """
        if self.output is None:
            return None

        if self.data:
            self._generate_table()

        assert isinstance(self.output, Table)
        for row in self.output.rows:
            yield row

    def pivot(self, *args):
        """ pivots the groupby so that `columns` become new columns.

        :param args: column names
        :return: New Table

        Example:
        t = Table()
        t.add_column('A', int, data=[1, 1, 2, 2, 3, 3] * 2)
        t.add_column('B', int, data=[1, 2, 3, 4, 5, 6] * 2)
        t.add_column('C', int, data=[6, 5, 4, 3, 2, 1] * 2)

        t.show()
        +=====+=====+=====+
        |  A  |  B  |  C  |
        | int | int | int |
        |False|False|False|
        +-----+-----+-----+
        |    1|    1|    6|
        |    1|    2|    5|
        |    2|    3|    4|
        |    2|    4|    3|
        |    3|    5|    2|
        |    3|    6|    1|
        |    1|    1|    6|
        |    1|    2|    5|
        |    2|    3|    4|
        |    2|    4|    3|
        |    3|    5|    2|
        |    3|    6|    1|
        +=====+=====+=====+

        g = t.groupby(keys=['A', 'C'], functions=[('B', Sum)])

        t2 = g.pivot('A')

        t2.show()
        +=====+==========+==========+==========+
        |  C  |Sum(B,A=1)|Sum(B,A=2)|Sum(B,A=3)|
        | int |   int    |   int    |   int    |
        |False|   True   |   True   |   True   |
        +-----+----------+----------+----------+
        |    5|         4|      None|      None|
        |    6|         2|      None|      None|
        |    3|      None|         8|      None|
        |    4|      None|         6|      None|
        |    1|      None|      None|        12|
        |    2|      None|      None|        10|
        +=====+==========+==========+==========+
        """
        columns = args
        if not all(isinstance(i, str) for i in args):
            raise TypeError(f"column name not str: {[i for i in columns if not isinstance(i,str)]}")

        if self.output is None:
            return None

        if self.data:
            self._generate_table()

        assert isinstance(self.output, Table)
        if any(i not in self.output.columns for i in columns):
            raise ValueError(f"column not found in groupby: {[i not in self.output.columns for i in columns]}")

        sort_order = {k: False for k in self.keys}
        if not self.output.is_sorted(**sort_order):
            self.output.sort(**sort_order)

        t = Table()
        for col_name, col in self.output.columns.items():  # add vertical groups.
            if col_name in self.keys and col_name not in columns:
                t.add_column(col_name, col.datatype, allow_empty=False)

        tup_length = 0
        for column_key in self.output.filter(*columns):  # add horizontal groups.
            col_name = ",".join(f"{h}={v}" for h, v in zip(columns, column_key))  # expressed "a=0,b=3" in column name "Sum(g, a=0,b=3)"

            for (header, function), function_instances in zip(self.groupby_functions, self.function_classes):
                new_column_name = f"{function.__name__}({header},{col_name})"
                if new_column_name not in t.columns:  # it's could be duplicate key value.
                    t.add_column(new_column_name, datatype=function_instances.datatype, allow_empty=True)
                    tup_length += 1
                else:
                    pass  # it's a duplicate.

        # add rows.
        key_index = {k: i for i, k in enumerate(self.output.columns)}
        old_v_keys = tuple(None for k in self.keys if k not in columns)

        for row in self.output.rows:
            v_keys = tuple(row[key_index[k]] for k in self.keys if k not in columns)
            if v_keys != old_v_keys:
                t.add_row(v_keys + tuple(None for i in range(tup_length)))
                old_v_keys = v_keys

            function_values = [v for h, v in zip(self.output.columns, row) if h not in self.keys]

            col_name = ",".join(f"{h}={row[key_index[h]]}" for h in columns)
            for (header, function), fi in zip(self.groupby_functions, function_values):
                column_key = f"{function.__name__}({header},{col_name})"
                t[column_key][-1] = fi

        return t


# reading and writing data.
# --------------------------
def split_by_sequence(text, sequence):
    """ helper to split text according to a split sequence. """
    chunks = tuple()
    for element in sequence:
        idx = text.find(element)
        if idx < 0:
            raise ValueError(f"'{element}' not in row")
        chunk, text = text[:idx], text[len(element) + idx:]
        chunks += (chunk,)
    chunks += (text,)  # the remaining text.
    return chunks


encodings = [
    'utf-32',
    'utf-16',
    'ascii',
    'utf-8',
    'windows-1252',
    'utf-7',
]


def detect_encoding(path):
    """ helper that automatically detects encoding from files. """
    assert isinstance(path, Path)
    for encoding in encodings:
        try:
            snippet = path.open('r', encoding=encoding).read(100)
            if snippet.startswith('ï»¿'):
                return 'utf-8-sig'
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            pass
    raise UnicodeDecodeError


def detect_seperator(path, encoding):
    """
    :param path: pathlib.Path objects
    :param encoding: file encoding.
    :return: 1 character.
    """
    # After reviewing the logic in the CSV sniffer, I concluded that all it
    # really does is to look for a non-text character. As the separator is
    # determined by the first line, which almost always is a line of headers,
    # the text characters will be utf-8,16 or ascii letters plus white space.
    # This leaves the characters ,;:| and \t as potential separators, with one
    # exception: files that use whitespace as separator. My logic is therefore
    # to (1) find the set of characters that intersect with ',;:|\t' which in
    # practice is a single character, unless (2) it is empty whereby it must
    # be whitespace.
    text = ""
    for line in path.open('r', encoding=encoding):  # pick the first line only.
        text = line
        break
    seps = {',', '\t', ';', ':', '|'}.intersection(text)
    if not seps:
        if " " in text:
            return " "
        else:
            raise ValueError("separator not detected")
    if len(seps) == 1:
        return seps.pop()
    else:
        frq = [(text.count(i), i) for i in seps]
        frq.sort(reverse=True)  # most frequent first.
        return frq[0][-1]


def text_reader(path, split_sequence=None, sep=None):
    """ txt, tab & csv reader """
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")

    # detect newline format
    windows = '\n'
    unix = '\r\n'

    encoding = detect_encoding(path)  # detect encoding

    if split_sequence is None and sep is None:  #
        sep = detect_seperator(path, encoding)

    t = Table()
    t.metadata['filename'] = path.name
    n_columns = None
    with path.open('r', encoding=encoding) as fi:
        for line in fi:
            end = windows if line.endswith(windows) else unix
            # this is more robust if the file was concatenated by a non-programmer, than doing it once only.

            line = line.rstrip(end)
            line = line.lstrip('\ufeff')  # utf-8-sig byte order mark.

            if split_sequence:
                values = split_by_sequence(line, split_sequence)
            elif line.count('"') >= 2 or line.count("'") >= 2:
                values = text_escape(line, sep=sep)
            else:
                values = tuple((i.lstrip().rstrip() for i in line.split(sep)))

            if not t.columns:
                for v in values:
                    header = v.rstrip(" ").lstrip(" ")
                    t.add_column(header, datatype=str, allow_empty=True)
                n_columns = len(values)
            else:
                while n_columns > len(values):  # this makes the reader more robust.
                    values += ('', )
                t.add_row(values)
    yield t


def text_escape(s, escape='"', sep=';'):
    """ escapes text marks using a depth measure. """
    assert isinstance(s, str)
    word, words = [], tuple()
    in_esc_seq = False
    for ix, c in enumerate(s):
        if c == escape:
            if in_esc_seq:
                if ix+1 != len(s) and s[ix + 1] != sep:
                    word.append(c)
                    continue  # it's a fake escape.
                in_esc_seq = False
            else:
                in_esc_seq = True
            if word:
                words += ("".join(word),)
                word.clear()
        elif c == sep and not in_esc_seq:
            if word:
                words += ("".join(word),)
                word.clear()
        else:
            word.append(c)

    if word:
        if word:
            words += ("".join(word),)
            word.clear()
    return words


def excel_reader(path):
    """  returns Table(s) from excel path """
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")
    sheets = xlrd.open_workbook(str(path), logfile='', on_demand=True)
    assert isinstance(sheets, xlrd.Book)

    for sheet in sheets.sheets():
        if sheet.nrows == sheet.ncols == 0:
            continue
        else:
            table = excel_sheet_reader(sheet)
            table.metadata['filename'] = path.name
            yield table


def excel_datetime(value):
    """ converts excels internal datetime numerics to date, time or datetime. """
    Y, M, D, h, m, s = xlrd.xldate_as_tuple(value, 0)
    if all((Y, M, D, h, m, s)):
        return f"{Y}-{M}-{D}T{h}-{m}-{s}"
    if all((Y, M, D)):
        return f"{Y}-{M}-{D}"
    if all((h, m, s)):
        return f"{h}:{m}:{s}"
    return value  # .. we tried...


excel_datatypes = {0: lambda x: None,  # empty string
                   1: lambda x: str(x),  # unicode string
                   2: lambda x: x,  # numeric int or float
                   3: lambda x: excel_datetime(x),  # datetime float
                   4: lambda x: True if x == 1 else False,  # boolean
                   5: lambda x: str(x)}  # error code


def excel_sheet_reader(sheet):
    """ returns Table from a spreadsheet sheet. """
    assert isinstance(sheet, xlrd.sheet.Sheet)
    t = Table()
    t.metadata['sheet_name'] = sheet.name

    for column_index in range(sheet.ncols):
        data = []
        for row_index in range(sheet.nrows):
            etype = sheet.cell_type(row_index, column_index)
            value = sheet.cell_value(row_index, column_index)
            data.append(excel_datatypes[etype](value))

        dtypes = {type(v) for v in data[1:]}
        allow_empty = True if None in dtypes else False
        dtypes.discard(None)

        if len(dtypes) == 1:
            header, dtype, data = str(data[0]), dtypes.pop(), data[1:]
        else:
            header, dtype, data = str(data[0]), str, [str(v) for v in data[1:]]
        t.add_column(header, dtype, allow_empty, data)
    return t


def ods_reader(path):
    """  returns Table from .ODS """
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")
    sheets = pyexcel_ods.get_data(str(path))

    for sheet_name, data in sheets.items():
        if data == [[], []]:  # no data.
            continue
        table = Table(filename=path.name)
        table.metadata['filename'] = path.name
        table.metadata['sheet_name'] = sheet_name
        for ix, column_name in enumerate(data[0]):
            dtypes = set(type(row[ix]) for row in data[1:] if len(row) > ix)
            allow_empty = None in dtypes
            dtypes.discard(None)
            if len(dtypes) == 1:
                dtype = dtypes.pop()
            elif dtypes == {float, int}:
                dtype = float
            else:
                dtype = str
            values = [dtype(row[ix]) for row in data[1:] if len(row) > ix]
            table.add_column(column_name, dtype, allow_empty, data=values)
        yield table


def zip_reader(path):
    """ reads zip files and unpacks anything it can read."""
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")

    with tempfile.TemporaryDirectory() as temp_dir_path:
        tempdir = Path(temp_dir_path)

        with zipfile.ZipFile(path, 'r') as zipf:

            for name in zipf.namelist():

                zipf.extract(name, temp_dir_path)

                p = tempdir / name
                try:
                    tables = file_reader(p)
                    for table in tables:
                        yield table
                except Exception as e:  # unknown file type.
                    print(f'reading {p} resulted in the error:')
                    print(str(e))
                    continue

                p.unlink()


def log_reader(path):
    """ returns Table from log files (txt)"""
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")
    for line in path.open()[:10]:
        print(repr(line))
    print("please declare separators. Blank return means 'done'.")
    split_sequence = []
    while True:
        response = input(">")
        if response == "":
            break
        print("got", repr(response))
        split_sequence.append(response)
    table = text_reader(path, split_sequence=split_sequence)
    return table


def find_format(table):
    """ common function for harmonizing formats AFTER import. """
    assert isinstance(table, Table)

    for col_name, column in table.columns.items():
        assert isinstance(column, (StoredColumn, Column))
        column.allow_empty = any(v in DataTypes.nones for v in column)

        values = [v for v in column if v not in DataTypes.nones]
        assert isinstance(column, (StoredColumn, Column))
        values.sort()

        works = []
        if not values:
            works.append((0, DataTypes.str))
        else:
            for dtype in DataTypes.types:  # try all datatypes.
                last_value = None
                c = 0
                for v in values:
                    if v != last_value:  # no need to repeat duplicates.
                        try:
                            DataTypes.infer(v, dtype)  # handles None gracefully.
                        except (ValueError, TypeError):
                            break
                        last_value = v
                    c += 1

                works.append((c, dtype))
                if c == len(values):
                    break  # we have a complete match for the simplest
                    # data format for all values. No need to do more work.

        for c, dtype in works:
            if c == len(values):
                values.clear()
                if table.use_disk:
                    c2 = StoredColumn
                else:
                    c2 = Column

                new_column = c2(column.header, dtype, column.allow_empty)
                for v in column:
                    new_column.append(DataTypes.infer(v, dtype) if v not in DataTypes.nones else None)
                column.clear()
                table.columns[col_name] = new_column
                break


readers = {
        'csv': [text_reader, {}],
        'tsv': [text_reader, {}],
        'txt': [text_reader, {}],
        'xls': [excel_reader, {}],
        'xlsx': [excel_reader, {}],
        'xlsm': [excel_reader, {}],
        'ods': [ods_reader, {}],
        'zip': [zip_reader, {}],
        'log': [log_reader, {'sep': False}]
    }


def file_reader(path, **kwargs):
    """
    :param path: pathlib.Path object with extension as:
        .csv, .tsv, .txt, .xls, .xlsx, .xlsm, .ods, .zip, .log

        .zip is automatically flattened

    :param kwargs: dictionary options:
        'sep': False or single character
        'split_sequence': list of characters

    :return: generator of Tables.
        to get the table in one line.

        >>> list(file_reader(abc.csv)[0]

        use the following for Excel and Zips:
        >>> for table in file_reader(filename):
                ...
    """
    assert isinstance(path, Path)
    extension = path.name.split(".")[-1]
    if extension not in readers:
        raise TypeError(f"Filetype for {path.name} not recognised.")
    reader, default_kwargs = readers[extension]
    kwargs = {**default_kwargs, **kwargs}

    for table in reader(path, **kwargs):
        assert isinstance(table, Table), "programmer returned something else than a Table"
        find_format(table)
        yield table



