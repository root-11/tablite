from tempfile import TemporaryFile
from zipfile import ZipFile
from datetime import date,time,datetime
import json
import sys

from pathlib import Path
from table import Table, Column, DataTypes


from collections import namedtuple


class Record(object):
    def __init__(self, start, len, max, min, sorted):
        self.start = start
        self.len = len
        self.min = min
        self.max = max
        self.sorted = sorted


class Records(object):
    def __init__(self):
        self.file = TemporaryFile()
        self.records = {}

    def add(self, column):
        assert isinstance(column, Column)

        if column[-1] >= column[0]:  # Z > A
            if any(column[i - 1] > column[i] for i in column[1:]):
                sorted = False
            else:
                sorted = True
        else:  # A > Z
            if any(column[i - 1] < column[i] for i in column[1:]):
                sorted = False
            else:
                sorted = True

        mi, ma = min(column), max(column)

        max_cid = max(self.records)
        new_cid = max_cid + 1
        r = self.records[max_cid]
        assert isinstance(r, Record)
        start = r.start + r.len

        self.records[new_cid] = Record(start, len(column), mi, ma, sorted)
        
        with ZipFile(self.file, 'w') as zipf:
            zipf.writestr(f'{new_cid}.TLL', json.dumps(column.to_json()))

    def __iter__(self):
        with ZipFile(self.file, 'r') as zipf:
            for name in sorted(zipf.namelist()):
                zips = zipf.read(name)
                data = json.loads(zips)
                column = Column.from_json(data)
                for v in column:
                    yield v



class List(object):
    def __init__(self):
        self._file = TemporaryFile()

        self._working_memory_limit = 2000  # bytes
        self._records = Records()        
        self._buffer = []

    def deletes(self, slice_id):
        file = TemporaryFile()
        with ZipFile(self._file, 'r') as fi:
            with ZipFile(file, 'w') as fo:
                for item in fi.infolist():
                    if item.filename == f"{slice_id}":
                        pass
                    else:
                        fo.writestr(item.filename, fi.read(item.filename))
        self._file = file

    def append(self, value):
        """ Append object to the end of the list. """
        self._cache.append(value)


    def clear(self, *args, **kwargs):
        """ Remove all items from list. """
        

    def copy(self, *args, **kwargs):
        """ Return a shallow copy of the list. """
        pass

    def count(self, *args, **kwargs):
        """ Return number of occurrences of value. """
        pass

    def extend(self, *args, **kwargs):
        """ Extend list by appending elements from the iterable. """
        pass

    def index(self, *args, **kwargs):
        """
        Return first index of value.

        Raises ValueError if the value is not present.
        """
        pass

    def insert(self, *args, **kwargs):
        """ Insert object before index. """
        pass

    def pop(self, *args, **kwargs):
        """
        Remove and return item at index (default last).

        Raises IndexError if list is empty or index is out of range.
        """
        pass

    def remove(self, *args, **kwargs):
        """
        Remove first occurrence of value.

        Raises ValueError if the value is not present.
        """
        pass

    def reverse(self, *args, **kwargs):
        """ Reverse *IN PLACE*. """
        pass

    def sort(self, *args, **kwargs):
        """ Stable sort *IN PLACE*. """
        pass

    def __add__(self, *args, **kwargs):
        """ Return self+value. """
        pass

    def __contains__(self, *args, **kwargs):
        """ Return key in self. """
        pass

    def __delitem__(self, *args, **kwargs):
        """ Delete self[key]. """
        pass

    def __eq__(self, *args, **kwargs):
        """ Return self==value. """
        pass

    def __getattribute__(self, *args, **kwargs):
        """ Return getattr(self, name). """
        pass

    def __getitem__(self, y):
        """ x.__getitem__(y) <==> x[y] """
        pass

    def __ge__(self, *args, **kwargs):
        """ Return self>=value. """
        pass

    def __gt__(self, *args, **kwargs):
        """ Return self>value. """
        pass

    def __iadd__(self, *args, **kwargs):
        """ Implement self+=value. """
        pass

    def __imul__(self, *args, **kwargs):
        """ Implement self*=value. """
        pass

    def __iter__(self, *args, **kwargs):
        """ Implement iter(self). """
        pass

    def __len__(self, *args, **kwargs):
        """ Return len(self). """
        pass

    def __le__(self, *args, **kwargs):
        """ Return self<=value. """
        pass

    def __lt__(self, *args, **kwargs):
        """ Return self<value. """
        pass

    def __mul__(self, *args, **kwargs):
        """ Return self*value. """
        pass

    @staticmethod  # known case of __new__
    def __new__(*args, **kwargs):
        """ Create and return a new object.  See help(type) for accurate signature. """
        pass

    def __ne__(self, *args, **kwargs):
        """ Return self!=value. """
        pass

    def __repr__(self, *args, **kwargs):
        """ Return repr(self). """
        pass

    def __reversed__(self, *args, **kwargs):
        """ Return a reverse iterator over the list. """
        pass

    def __rmul__(self, *args, **kwargs):
        """ Return value*self. """
        pass

    def __setitem__(self, *args, **kwargs):
        """ Set self[key] to value. """
        pass

    def __sizeof__(self, *args, **kwargs):
        """ Return the size of the list in memory, in bytes. """
        pass

    __hash__ = None

    def __str__(self):
        pass

    def __copy__(self):
        pass

    def to_json(self):
        pass

    @classmethod
    def from_json(cls, json_):


    def type_check(self, value):
        pass

    def replace(self, values) -> None:
        pass



def test_basic_column():
    # creating a column remains easy:
    c = Column('A', int, False)

    c.archive = True  # example using tempfile
    c.archive = Path(__file__).parent / 'this.table'  # example using filename
    c.working_memory_limit = 20_000 # kbyte

    # so does adding values:
    c.append(44)
    c.append(44)
    assert len(c) == 2

    # and converting to and from json
    d = c.to_json()
    c2 = Column.from_json(d)
    assert len(c2) == 2

    # comparing columns is easy:
    assert c == c2
    assert c != Column('A', str, False)