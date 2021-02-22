import json
import pickle
import sqlite3
import zlib
from itertools import count
from pathlib import Path
from random import choice
from string import ascii_lowercase
from sys import getsizeof
from tempfile import gettempdir

from tablite.datatypes import DataTypes


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
        self.file = windows_tempfile()
        self._conn = sqlite3.connect(str(self.file))  # SQLite3 connection
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


def windows_tempfile(prefix='tmp', suffix='.db'):
    """ generates a safe tempfile which windows can't handle. """
    safe_folder = Path(gettempdir())
    while 1:
        n = "".join(choice(ascii_lowercase) for _ in range(5))
        name = f"{prefix}{n}{suffix}"
        p = safe_folder / name
        if not p.exists():
            break
    return p