import math
import json
import pickle
import sqlite3
import zlib
from pathlib import Path
from random import choice
from string import ascii_lowercase
from sys import getsizeof
from tempfile import gettempdir

from itertools import count
from abc import ABC

from tablite.datatypes import DataTypes

# Queries for StoredList
sql_create = "CREATE TABLE records (id INTEGER PRIMARY KEY, data BLOB);"
sql_journal_off = "PRAGMA journal_mode = OFF"
sql_sync_off = "PRAGMA synchronous = OFF "
sql_delete = "DELETE FROM records WHERE id = ?"
sql_insert = "INSERT INTO records VALUES (?, ?);"
sql_update = "UPDATE records SET data = ? WHERE id=?;"
sql_select = "SELECT data FROM records WHERE id=?"


def tempfile(prefix='tmp', suffix='.db'):
    """ generates a safe tempfile which windows can't handle. """
    safe_folder = Path(gettempdir())
    while 1:
        n = "".join(choice(ascii_lowercase) for _ in range(5))
        name = f"{prefix}{n}{suffix}"
        p = safe_folder / name
        if not p.exists():
            break
    return p


class Page(object):
    ids = count()

    def __init__(self):
        self.pid = next(Page.ids)
        self.data = []
        self.len = 0
        self.loaded = True

    def __len__(self):
        if not self.loaded:
            return self.len
        else:
            return len(self.data)

    def __repr__(self):
        return f"Page({self.pid}) {'loaded' if self.loaded else 'stored'} ({len(self)} items)"


class StoredList(object):
    """
    Python list stored on disk using sqlite's fast mmap.
    """
    default_page_size = 20_000

    def __init__(self, data=None, page_size=None):
        if page_size is None:
            page_size = StoredList.default_page_size
        if not isinstance(page_size, int):
            raise TypeError
        self._page_size = page_size

        self.storage_file = tempfile()
        self._conn = sqlite3.connect(str(self.storage_file))  # SQLite3 connection
        with self._conn as c:
            c.execute(sql_create)
            c.execute(sql_journal_off)
            c.execute(sql_sync_off)

        self.pages = []
        self._current_page = None
        self._new_page()

        if data is not None:
            self.extend(data)

    @property
    def page_size(self):
        return self._page_size

    @page_size.setter
    def page_size(self, value):
        if not isinstance(value, int):
            raise TypeError
        elif value < 0:
            raise ValueError
        elif self._page_size == value:
            pass  # leave as is.
        elif value > self._page_size:  # leave the page size. It won't make a difference.
            self._page_size = value
        else:  # pages will have to be reduced.
            print(f"reducing page size from {self._page_size} to {value}")
            self._page_size = value

            old_pages = [page for page in self.pages]
            self._new_page()
            for page in old_pages:
                self._load_page(page)
                self.extend(page.data)  # extension runs with the new page size.
                self._delete_page(page)

    def _new_page(self):
        """ internal method that stores current page and creates a new empty page"""
        if self._current_page is not None:
            self._store_page(self._current_page)

        page = Page()
        self.pages.append(page)
        data = zlib.compress(pickle.dumps(page.data))
        with self._conn as c:
            c.execute(sql_insert, (page.pid, data))  # INSERT

        assert page.loaded is True
        self._current_page = page

    def _store_page(self, page):
        """ internal method that stores page data to disk.
        :param page: Page
        """
        assert isinstance(page, Page)
        if not page.loaded:
            return  # because it is already stored.

        data = zlib.compress(pickle.dumps(page.data))
        page.len = len(page.data)  # update page len.
        page.data.clear()

        with self._conn as c:
            c.execute(sql_update, (data, page.pid))  # UPDATE
        page.loaded = False

    def _load_page(self, page):
        """ internal method that loads the data from a page.
        :param page: Page
        """
        assert isinstance(page, Page)
        if self._current_page == page and page.loaded:
            return
        self._store_page(self._current_page)
        assert self._current_page.loaded is False

        with self._conn as c:
            q = c.execute(sql_select, (page.pid,))  # READ
            data = q.fetchone()[0]
        page.data = pickle.loads(zlib.decompress(data))
        page.loaded = True
        self._current_page = page

    def _delete_page(self, page):
        """ internal method that deletes a page of data. """
        assert isinstance(page, Page)
        with self._conn as c:
            c.execute(sql_delete, (page.pid,))  # DELETE
        page.data.clear()
        self.pages.remove(page)

        if not self.pages:  # there must always be one page available.
            self._new_page()

        if self._current_page == page:
            self._current_page = self.pages[-1]

    # PUBLIC METHODS.

    def __len__(self):
        return sum(len(p) for p in self.pages)

    def __iter__(self):
        for page in self.pages:
            self._load_page(page)
            for value in page.data:
                yield value

    def __reversed__(self):
        for page in reversed(self.pages):
            self._load_page(page)
            for value in reversed(page.data):
                yield value

    def __repr__(self):
        return f"StoredList(page_size={self._page_size}, data={len(self)})"

    def __str__(self):
        if len(self) > 20:
            a, b = [v for v in self[:5]], [v for v in self[-5:]]
            return f"StoredList(page_size={self._page_size}, data={a}...{b})"
        else:
            return f"StoredList(page_size={self._page_size}, data={list(self[:])})"

    def append(self, value):
        """ Append object to the end of the list. """
        assert isinstance(self._current_page, Page)
        if len(self._current_page) == self._page_size:
            self._new_page()
        self._current_page.data.append(value)

    def clear(self):
        """ Remove all items from list. """
        for page in self.pages[::-1]:
            self._delete_page(page)

    def copy(self):
        """ Return a shallow copy of the list. """
        SL = StoredList(page_size=self.page_size)
        for page in self.pages:
            self._load_page(page)
            SL.extend(page.data)
        return SL

    def count(self, item):
        """ Return number of occurrences of item. """
        return sum(1 for v in self if v == item)

    def extend(self, items):
        """ Extend list by appending elements from the iterable. """
        while items:
            space = self._page_size - len(self._current_page)
            data = items[:space]
            self._current_page.data.extend(data[:])
            items = items[space:]
            if items:
                self._new_page()

    def index(self, item):
        """
        Return first index of value.
        Raises ValueError if the value is not present.
        """
        for ix, v in enumerate(self):
            if v == item:
                return ix
        raise ValueError(f"{item} is not in list")

    def insert(self, index, item):
        """ Insert object before index. """
        if not isinstance(index, int):
            raise TypeError
        if abs(index) > len(self):
            raise IndexError("index out of range")
        if index < 0:
            index = len(self) + index

        c = 0
        page = None
        for page in self.pages[:]:
            assert isinstance(page, Page)
            if c <= index <= len(page) + c:
                break
            c += len(page)
        ix = index - c

        assert isinstance(page, Page)
        self._load_page(page)
        page.data.insert(ix, item)

        if len(page) < self._page_size:  # if there is space on the page...
            return
        else:  # split the data in half and insert a page.
            n = len(page) // 2
            page.data, new_page = page.data[:n], page.data[n:]
            page_ix = self.pages.index(page)
            self._new_page()
            self._current_page.data.extend(new_page)
            self.pages.remove(self._current_page)
            self.pages.insert(page_ix + 1, self._current_page)

    def pop(self, index=None):
        """
        Remove and return item at index (default last).

        Raises IndexError if list is empty or index is out of range.
        """
        if index is None:
            index = 0
        if not isinstance(index, int):
            raise TypeError
        if abs(index) > len(self):
            raise IndexError("list index out of range")
        if index < 0:
            index = len(self) + index

        c = 0
        for page in self.pages[:]:
            if c <= index < len(page) + c:
                self._load_page(page)
                ix = index - c
                value = page.data.pop[ix]
                if not page.data:
                    self._delete_page(page)
                return value
            else:
                c += len(page)

    def remove(self, item):
        """
        Remove first occurrence of value.

        Raises ValueError if the value is not present.
        """
        for page in self.pages[:]:
            self._load_page(page)
            if item in page.data:
                page.data.remove(item)
                return
            if not page.data:
                self._delete_page(page)

        raise ValueError(f"{item} not in list")

    def reverse(self):
        """ Reverse *IN PLACE*. """
        self.pages.reverse()
        for page in self.pages:
            self._load_page(page)
            assert isinstance(page, Page)
            page.data.reverse()
            self._store_page(page)

    def sort(self, key=None, reverse=False):
        """ Implements a hybrid of quicksort and merge sort.
        See more on https://en.wikipedia.org/wiki/External_sorting
        """
        if key is not None:
            raise NotImplementedError(f"key is not supported.")

        for page in self.pages:
            self._load_page(page)
            page.data.sort()

        working_buffer = [page for page in self.pages]
        if len(working_buffer) == 1:
            C = StoredList(self.page_size)
            page = working_buffer.pop()
            self._load_page(page)
            C.extend(page.data)
            working_buffer.append(C)
        else:
            while len(working_buffer) > 1:
                A = working_buffer.pop(0)
                A = iter(A)
                B = working_buffer.pop(0)
                B = iter(B)

                C = StoredList(self.page_size)
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
                working_buffer.append(C)

        L = working_buffer.pop(0)
        assert len(L) == len(self)
        old_pages = [page for page in self.pages]
        self._new_page()
        self.extend(L)
        for page in old_pages:
            self._delete_page(page)

    def __add__(self, other):
        """
        A = [1,2,3]
        B = [4,5,6]
        C = A+B
        C = [1,2,3,4,5,6]
        """
        if not isinstance(other, (StoredList, list)):
            raise TypeError
        SL = StoredList(self._page_size)
        SL.extend(self)
        SL.extend(other)
        return SL

    def __contains__(self, item):
        return any(item == i for i in self)

    def __delitem__(self, index):
        _ = self.pop(index)

    def __eq__(self, other):
        """ Return self==value. """
        if not isinstance(other, (StoredList, list)):
            raise TypeError

        if len(self) != len(other):
            return False

        if any(a != b for a, b in zip(self, other)):
            return False
        return True

    def __getitem__(self, item):
        if not isinstance(item, (slice, int)):
            raise TypeError

        if isinstance(item, int):
            if not isinstance(item, int):
                raise TypeError
            if abs(item) > len(self):
                raise IndexError("list index out of range")
            if item < 0:
                item = len(self) + item
            c = 0
            for page in self.pages:
                if c <= item <= len(page) + c:
                    ix = item - c
                    self._load_page(page)
                    return page.data[ix]  # <--- Exit for integer item
                c += len(page)

        assert isinstance(item, slice)
        start, stop, step = DataTypes.infer_range_from_slice(item, len(self))

        L = StoredList(page_size=self._page_size)
        if step > 0 and start > stop:
            return L
        if step < 0 and start < stop:
            return L

        if step > 0:
            A = 0
            for page in self.pages:
                B = len(page) + A
                if stop < A:
                    break
                if B < start:
                    A += len(page)
                    continue
                self._load_page(page)
                if start >= A:
                    start_ix = start - A
                else:  # A > start:
                    steps = math.ceil((A-start) / step)
                    start_ix = (start + (steps * step)) - A
                if stop < B:
                    stop_ix = stop - A
                else:
                    stop_ix = len(page)

                data = page.data[start_ix:stop_ix:step]
                L.extend(data)
                A += len(page)
        else:  # step < 0  # item.step < 0: backward traverse
            B = len(self)
            for page in reversed(self.pages):
                A = B - len(page)
                if start < A:
                    B -= len(page)
                    continue
                if B < stop:
                    break
                self._load_page(page)

                if start > B:
                    steps = abs(math.floor((start-B) / step))
                    start_ix = start + (steps * step) - A
                else:  # start <= B
                    start_ix = start-A

                if item.stop is None:
                    stop_ix = None
                else:  # stop - A:
                    stop_ix = max(stop - A, 0)

                data = page.data[start_ix:stop_ix:step]
                L.extend(data)
                B -= len(page)

        return L

    def __ge__(self, other):
        """ Return self>=value. """
        return all(a >= b for a, b in zip(self, other))

    def __gt__(self, other):
        """ Return self>value. """
        return all(a > b for a, b in zip(self, other))

    def __iadd__(self, other):
        """ Implement self+=value.
        >>> A = [1,2,3]
        >>> A += [4,5]
        >>> print(A)
        [1,2,3,4,5]
        """
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
            new_list = StoredList(self._page_size)
            for i in range(value):
                new_list.extend(self)
            return new_list

    def __le__(self, other):
        """ Return self<=value. """
        return all(a <= b for a, b in zip(self, other))

    def __lt__(self, other):
        """ Return self<value. """
        return all(a <= b for a, b in zip(self, other))

    def __mul__(self, value):
        """ Return self*value. """
        if not isinstance(value, int):
            raise TypeError
        if value <= 0:
            raise ValueError
        new_list = StoredList(page_size=self._page_size)
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

    def __rmul__(self, value):
        """ Return value*self. """
        return self.__mul__(value)

    def __setitem__(self, key, value):
        """ Set self[key] to value. """
        if not isinstance(key, int):
            raise TypeError
        c = 0
        for page in self.pages:
            if c < key < c + len(page):
                self._load_page(page)
                ix = key - c
                page.data[ix] = value
                return

    def __sizeof__(self):
        """ Return the size of the list in memory, in bytes. """
        return getsizeof(self.pages)

    def disk_size(self):
        """ returns the size of the stored file"""
        self._store_page(self._current_page)
        return self.storage_file.stat().st_size

    def __hash__(self):
        raise TypeError("unhashable type: List")

    def __copy__(self):
        SL = StoredList(self._page_size)
        for page in self.pages:
            self._load_page(page)
            SL.extend(page.data)
        return SL


class CommonColumn(ABC):
    def __init__(self, header, datatype, allow_empty, metadata=None, data=None):
        if not isinstance(header, str) and header != "":
            raise ValueError
        self.header = header
        if not isinstance(datatype, type):
            raise ValueError
        self.datatype = datatype
        if not isinstance(allow_empty, bool):
            raise TypeError
        self.allow_empty = allow_empty
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError
        self.metadata = metadata
        if data is not None:
            self._init(data)

    def _init(self, data):
        if isinstance(data, tuple):
            for v in data:
                self.append(v)
        elif isinstance(data, (list, StoredColumn, InMemoryColumn)):
            self.extend(data)
        elif data is not None:
            raise NotImplementedError(f"{type(data)} is not supported.")

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.header},{self.datatype},{self.allow_empty}) # ({len(self)} rows)"

    def __copy__(self):
        return self.copy()

    def copy(self):
        return self.__class__(self.header, self.datatype, self.allow_empty, metadata=self.metadata.copy(), data=self)

    def to_json(self):
        return json.dumps({
            'header': self.header,
            'datatype': self.datatype.__name__,
            'allow_empty': self.allow_empty,
            'metadata': self.metadata,
            'data': json.dumps([DataTypes.to_json(v) for v in self])
        })

    def type_check(self, value):
        """ helper that does nothing unless it raises an exception. """
        if value is None:
            if not self.allow_empty:
                raise ValueError("None is not permitted.")
            return
        if not isinstance(value, self.datatype):
            raise TypeError(f"{value} is not of type {self.datatype}")

    def __len__(self):
        raise NotImplementedError("subclasses must implement this method")

    def append(self, value):
        self.type_check(value)
        super().append(value)

    def replace(self, values) -> None:
        assert isinstance(values, list)
        if len(values) != len(self):
            raise ValueError("input is not of same length as column.")
        if not all(self.type_check(v) for v in values):
            raise TypeError(f"input contains non-{self.datatype.__name__}")
        self.clear()
        self.extend(values)

    def clear(self):
        raise NotImplementedError("subclasses must implement this method")

    def count(self, item):
        raise NotImplementedError("subclasses must implement this method")

    def extend(self, items):
        raise NotImplementedError("subclasses must implement this method")

    def index(self, item):
        raise NotImplementedError("subclasses must implement this method")

    def pop(self, index=None):
        raise NotImplementedError("subclasses must implement this method")

    def remove(self, item):
        raise NotImplementedError("subclasses must implement this method")

    def reverse(self):
        raise NotImplementedError("subclasses must implement this method")

    def sort(self, key=None, reverse=False):
        raise NotImplementedError("subclasses must implement this method")

    def __add__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __contains__(self, item):
        raise NotImplementedError("subclasses must implement this method")

    def __delitem__(self, index):
        raise NotImplementedError("subclasses must implement this method")

    def __eq__(self, other):
        return all([
            self.header == other.header,
            self.datatype == other.datatype,
            self.allow_empty == other.allow_empty,
            super(self).__eq__(other),
        ])

    def __getitem__(self, item):
        raise NotImplementedError("subclasses must implement this method")

    def __ge__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __gt__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __iadd__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __imul__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __iter__(self):
        raise NotImplementedError("subclasses must implement this method")

    def __le__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __lt__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __mul__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __ne__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __reversed__(self):
        raise NotImplementedError("subclasses must implement this method")

    def __rmul__(self, other):
        raise NotImplementedError("subclasses must implement this method")

    def __setitem__(self, key, value):
        self.type_check(value)
        super().__setitem__(key, value)

    def __sizeof__(self):
        raise NotImplementedError("subclasses must implement this method")

    def __hash__(self):
        raise NotImplementedError("subclasses must implement this method")


class StoredColumn(CommonColumn, StoredList):  # MRO: CC first, then SL.
    """This is a sqlite backed mmaped list with headers and metadata."""

    def __init__(self, header, datatype, allow_empty, data=None, metadata=None, page_size=StoredList.default_page_size):
        CommonColumn.__init__(self, header, datatype, allow_empty, metadata=metadata)
        StoredList.__init__(self, page_size=page_size)
        self._init(data)

    @classmethod
    def from_json(cls, json_):
        j = json.loads(json_)
        j['datatype'] = dtype = getattr(DataTypes, j['datatype'])
        j['data'] = [DataTypes.from_json(v, dtype) for v in json.loads(j['data'])]
        return StoredColumn(**j)


class InMemoryColumn(CommonColumn, list):  # MRO: CC first, then list.
    """This is a list with headers and metadata."""

    def __init__(self, header, datatype, allow_empty, data=None, metadata=None):
        CommonColumn.__init__(self, header, datatype, allow_empty, metadata=metadata)
        list.__init__(self)  # then init the list attrs.
        self._init(data)

    @classmethod
    def from_json(cls, json_):
        j = json.loads(json_)
        j['datatype'] = dtype = getattr(DataTypes, j['datatype'])
        j['data'] = [DataTypes.from_json(v, dtype) for v in json.loads(j['data'])]
        return InMemoryColumn(**j)
