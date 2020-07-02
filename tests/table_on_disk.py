import json
from pathlib import Path
from tempfile import TemporaryFile
from zipfile import ZipFile

from table import Table, Column, DataTypes


class Record(object):
    def __init__(self, name, datatype, items):
        self.name = name
        self.len = len(items)
        self.min = min(items)
        self.max = max(items)
        assert datatype in DataTypes.types
        self.datatype = datatype


    def update(self, items):
        assert all(isinstance(i, self.datatype) for i in items)
        self.len = len(items)
        self.min = min(items)
        self.max = max(items)

    def may_contain(self, item):
        dtype = type(item)
        if dtype is not self.datatype:
            return False
        if dtype in DataTypes.numeric_types:
            if self.min <= item <= self.max:
                return True
            else:
                return False
        return True


class StoredList(object):
    def __init__(self):
        self.file = TemporaryFile()
        self.records = []
        self.buffer = []
        self.buffer_max_len = float('inf')
        self.datatype = None

    # internal data management methods
    # --------------------------------

    def _buffer_check(self):
        if len(self.buffer) < self.buffer_max_len:
            return

        while len(self.buffer) >= self.buffer_max_len:
            items, self.buffer = self.buffer[:self.buffer_max_len], self.buffer[self.buffer_max_len:]

            datatypes = {type(t) for t in items}
            if len(datatypes) != 1:
                raise TypeError("One datatype only.")
            dtype = datatypes.pop()
            if self.datatype is None:
                self.datatype = dtype
            if dtype != self.datatype:
                raise TypeError(f"expected {self.datatype}, got {dtype}")

            self._write(items, record=None)

    def _write(self, items, record=None):
        if record is None:
            r = Record(len(self.records) + 1, self.datatype, items)
            self.records.append(r)
        assert isinstance(record, Record)
        with ZipFile(self.file, 'w') as zipf:
            json_str = json.dumps([DataTypes.to_json(v) for v in items])
            zipf.writestr(record.name, json_str)
            record.update(items)

    def _read(self, record):
        with ZipFile(self.file, 'r') as zipf:
            zips = zipf.extract(record.name)
            data = json.loads(Path(zips).read_bytes())
        return data

    def _delete(self, r):
        assert isinstance(r, Record)
        self.records.remove(r)
        empty_list = []
        self._write(empty_list, r)

    def _records_len(self):
        """ returns rows in stored records, excluding rows in buffer. """
        return sum(r.len for r in self.records)

    def __len__(self):
        """ Return len(self). """
        return self._records_len() + len(self.buffer)

    def _normal_index(self, index):
        if not isinstance(index, int):
            raise TypeError
        if 0 <= index < len(self):
            pass
        elif index < 0:
            index = 0 if len(self) + index < 0 else len(self) + index
        elif index >= len(self):
            index = len(self)
        else:
            raise Exception('bad logic')

        return index

    # public methods of a list
    # ------------------------

    def append(self, value):
        """ Append object to the end of the list. """
        self.buffer.append(value)
        self._buffer_check()

    def clear(self):
        """ Remove all items from list. """
        self.buffer.clear()
        self.file = TemporaryFile()
        self.records = {}
        self.buffer = []

    def copy(self, *args, **kwargs):
        """ Return a shallow copy of the list. """
        pass

    def count(self, item):
        """ Return number of occurrences of value. """
        if type(item) not in self.datatypes:
            return 0

        counter = 0
        for r in self.records:
            assert isinstance(r, Record)
            if r.may_contain(item):
                data = self._read(r)
                counter += data.count(item)
        counter += self.buffer.count(item)
        return counter

    def extend(self, items):
        """ Extend list by appending elements from the iterable. """
        self.buffer.extend(items)
        self._buffer_check()

    def index(self, item):
        """
        Return first index of value.

        Raises ValueError if the value is not present.
        """
        if type(item) not in self.datatypes:
            return None

        counter = 0
        for r in self.records:
            assert isinstance(r, Record)
            if r.may_contain(item):
                data = self._read(r)
                if item in data:
                    return counter + data.index(item)
            counter += r.len

        if item in self.buffer:
            counter = sum(r.len for r in self.records)
            return counter + self.buffer.index(item)

        raise ValueError(f"{item} not found")

    def insert(self, index, item):
        """ Insert object before index. """
        index = self._normal_index(index)

        if not isinstance(item, self.datatype):
            raise TypeError

        _records = sum(r.len for r in self.records)
        if index < _records:
            counter = 0
            for r in self.records:
                if counter <= index <= counter + r.len:
                    data = self._read(r)
                    ix = index - counter
                    data.insert(ix, item)
                    self._write(data, r)
                    return None
                counter += r.len

        elif _records <= index <= len(self):
            self.buffer.insert(index - _records, item)
            self._buffer_check()
        else:
            self.buffer.append(item)
            self._buffer_check()

    def pop(self, index):
        """
        Remove and return item at index (default last).

        Raises IndexError if list is empty or index is out of range.
        """
        if index is None:
            index = len(self)
        else:
            index = self._normal_index(index)

        if self._records_len() < index:
            return self.buffer.pop()

        counter = 0
        for r in self.records:
            if counter < index <= counter + r.len:
                data = self._read(r)
                row_index = index - counter
                pop_value = data.pop(row_index)
                self._write(data, r)
                return pop_value

        else:  # there's no buffer and we are popping the last value.
            r = self.records[-1]
            self.buffer = self._read(r)
            self._delete(r)
            return self.buffer.pop()

    def remove(self, item):
        """
        Remove first occurrence of value.

        Raises ValueError if the value is not present.
        """
        if not isinstance(item, self.datatype):
            raise TypeError

        for r in self.records:
            if r.may_contain(item):
                data = self._read(r)
                if item in data:
                    data.remove(item)
                    self._write(data, r)
                    return None

        if item in self.buffer:
            self.buffer.remove(item)
        else:
            raise ValueError(f"{item} not found")

    def reverse(self):
        """ Reverse *IN PLACE*. """
        self._write(self.buffer, record=None)
        self.buffer.clear()

        new_list = StoredList()
        for r in reversed(self.records):
            data = self._read(r)
            data.reverse()
            new_list.extend(data)

        self.file = new_list.file
        self.records = new_list.records
        self.buffer = new_list.buffer


    def sort(self, reverse=False):
        """ Stable sort *IN PLACE*. """
        self._write(self.buffer, record=None)
        self.buffer.clear()

        for r in self.records:
            data = self._read(r)
            data.sort(reverse=reverse)
            self._write(data, r)

        new_list = StoredList()
        while self.records:
            if reverse:
                limit_value = max(r.max for r in self.records)
            else:
                limit_value = min(r.min for r in self.records)

            for r in self.records:
                if r.min == limit_value:
                    data = self._read(r)
                    new_list.extend([r for r in data if r == limit_value])
                    data = [r for r in data if r != limit_value]
                    if not data:
                        self._delete(r)
                    else:
                        self._write(data)

        self.file = new_list.file
        self.records = new_list.records
        self.buffer = new_list.buffer

    def __add__(self, other):
        """ Return self+value. """
        new_list = StoredList()

        if isinstance(other, (StoredList, list)):
            new_list.extend(other)
            for i in self:
                new_list.append(i)
            return new_list
        else:
            raise TypeError

    def __contains__(self, item):
        """ Return key in self. """
        if item in self.buffer:
            return True
        if any(i == item for i in self):
            return True
        return False

    def __delitem__(self, index):
        """ Delete self[key]. """
        if index > len(self) or index < 0:
            raise IndexError
        _records = self._records_len()
        if index > _records:
            del self.buffer[index - _records]

        counter = 0
        for r in self.records:
            if counter < index < counter + r.len:
                data = self._read(r)
                del data[index - counter]
                self._write(data)
                break

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
        for r in self.records:
            data = self._read(r)
            for v in data:
                yield v
        for v in self.buffer:
            yield v

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