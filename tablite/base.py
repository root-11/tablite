import os
import io
import math
import yaml
import atexit
import shutil
import logging
import zipfile
import numpy as np
from pathlib import Path
from itertools import count, chain

from utils import type_check, intercept, np_type_unify
from config import Config

log = logging.getLogger(__name__)


file_registry = set()


def register(path):
    global file_registry
    file_registry.add(path)


def shutdown():
    for path in file_registry:
        if str(os.getpid()) in str(path):  # safety feature to prevent rm -rf /
            log.debug(f"shutdown: running rmtree({path})")
            shutil.rmtree(path)


atexit.register(shutdown)


class Page(object):
    _ids = count(start=1)

    def __init__(self, path, array) -> None:
        """
        Args:
            path (Path): working directory.
            array (np.array): data
        """
        self.id = next(self._ids)
        type_check(path, Path)
        self.path = path / "pages" / f"{self.id}.npy"

        type_check(array, np.ndarray)

        self.len = len(array)
        np.save(self.path, array, allow_pickle=True, fix_imports=False)
        log.debug(f"Page saved: {self.path}")

    def __len__(self):
        return self.len

    def __del__(self):
        # trigger explicitly during cleanup UNLESS it's a users explicit save.
        if self.path.exists():
            os.remove(self.path)
        log.debug(f"Page deleted: {self.path}")

    def get(self):
        return np.load(self.path, allow_pickle=True, fix_imports=False)


class Column(object):
    def __init__(self, path, value=None) -> None:
        """Create Column

        Args:
            path (Path): path of table.yml
            value (Iterable, optional): Data to store. Defaults to None.
        """
        self.path = path
        self.pages = []  # keeps pointers to instances of Page
        if value is not None:
            self.extend(value)

    def __len__(self):
        return sum(len(p) for p in self.pages)

    @staticmethod
    def _paginate(values, page_size=None):
        """
        Takes a numpy array and turns it into
        a list of numpy arrays of page_size or less.

        Args:
            values (np.ndarray): values
            page_size (int, optional): page size. Defaults to config.PAGE_SIZE.

        Returns:
            list of ndarrays
        """
        type_check(values, np.ndarray)
        if page_size is None:
            page_size = Config.PAGE_SIZE
        type_check(page_size, int)

        arrays = []
        n = int(math.ceil(len(values) / page_size)) * page_size
        start = 0
        for end in range(page_size, n + 1, page_size):
            x = np.array(values[start:end])
            arrays.append(x)
            start = end
        return arrays

    def extend(self, value):  # USER FUNCTION.
        type_check(value, np.ndarray)
        for array in self._paginate(value):
            self.pages.append(Page(path=self.path.parent, array=array))

    def getpages(self, item):
        # internal function
        if isinstance(item, int):
            item = slice(item, item + 1, 1)

        type_check(item, slice)
        is_reversed = False if (item.step is None or item.step > 0) else True

        length = len(self)
        scan_item = slice(*item.indices(length))
        range_item = range(*item.indices(length))

        pages = []
        start, end = 0, 0
        for page in self.pages:
            start, end = end, end + page.len
            if is_reversed:
                if start > scan_item.start:
                    break
                if end < scan_item.stop:
                    continue
            else:
                if start > scan_item.stop:
                    break
                if end < scan_item.start:
                    continue
            ro = intercept(range(start, end), range_item)
            if len(ro) == 0:
                continue
            elif len(ro) == page.len:  # share the whole immutable page
                pages.append(page)
            else:  # fetch the slice and filter it.
                search_slice = slice(ro.start - start, ro.stop - start, ro.step)
                np_arr = np.load(page.path, allow_pickle=True, fix_imports=False)
                match = np_arr[search_slice]
                pages.append(match)

        if is_reversed:
            pages.reverse()
            for ix, page in enumerate(pages):
                if isinstance(page, Page):
                    data = page.get()
                    pages[ix] = np.flip(data)
                else:
                    pages[ix] = np.flip(page)

        return pages

    def __getitem__(self, item):  # USER FUNCTION.
        """gets numpy array.

        Args:
            item (int OR slice): slice of column

        Returns:
            np.ndarray: results as numpy array.

        Remember:
        >>> R = np.array([0,1,2,3,4,5])
        >>> R[3]
        3
        >>> R[3:4]
        array([3])
        """
        result = []
        for element in self.getpages(item):
            if isinstance(element, Page):
                result.append(element.get())
            else:
                result.append(element)

        if result:
            arr = np_type_unify(result)
        else:
            arr = np.array([])

        if isinstance(item, int):
            if not arr:
                raise IndexError(f"index {item} is out of bounds for axis 0 with size {len(self)}")
            return arr[0]
        else:
            return arr

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self._setitem_integer_key(key, value)

        elif isinstance(key, slice):
            type_check(value, np.ndarray)

            if key.start is None and key.stop is None and key.step in (None, 1):
                self._setitem_replace_all(key, value)
            elif key.start is not None and key.stop is None and key.step in (None, 1):
                self._setitem_extend(key, value)
            elif key.stop is not None and key.start is None and key.step in (None, 1):
                self._setitem_prextend(key, value)
            elif key.step in (None, 1) and key.start is not None and key.stop is not None:
                self._setitem_insert(key, value)
            elif key.step not in (None, 1):
                self._setitem_update(key, value)
            else:
                raise KeyError(f"bad key: {key}")
        raise KeyError(f"bad key: {key}")

    def _setitem_integer_key(self, key, value):
        # documentation:
        # example: L[3] = 27
        if isinstance(value, (list, type, np.ndarray)):
            raise TypeError(
                f"your key is an integer, but your value is a {type(value)}. \
                    Did you mean to insert? E.g. [{key}:{key+1}] = {value} ?"
            )
        length = len(self)
        key = length + key if key < 0 else key
        if not (0 <= key < length):
            raise IndexError("list assignment index out of range")

        # now there's a positive key and a single value.
        start, end = 0, 0
        for index, page in enumerate(self.pages):
            start, end = end, end + page.len
            if start <= key < end:
                data = page.get()
                data[key - start] = value
                new_page = Page(self.path, data)
                self.pages[index] = new_page
                break

    def _setitem_replace_all(self, key, value):
        # documentation: new = list(value)
        # example: L[:] = [1,2,3]
        self.pages.clear()  # Page.__del__ will take care of removed pages.
        self.extend(value)  # Column.extend handles pagination.

    def _setitem_extend(self, key, value):
        # documentation: new = old[:key.start] + list(value)
        # example: L[0:] = [1,2,3]
        start, end = 0, 0
        for index, page in enumerate(self.pages):
            start, end = end, end + page.len
            if start <= key.start < end:  # find beginning
                data = page.get()
                keep = data[key.start - start :]
                new = np_type_unify([keep, value])
                self.pages = self.pages[:index]
                self.extend(new)
                break

    def _setitem_prextend(self, key, value):
        # documentation: new = list(value) + old[key.stop:]
        # example: L[:3] = [1,2,3]
        start, end = 0, 0
        for index, page in enumerate(self.pages):
            start, end = end, end + page.len
            if start <= key.stop < end:  # find beginning
                data = page.get()
                keep = data[: key.stop - start]  # keeping up to key.stop
                new = np_type_unify([value, keep])
                tail = self.pages[index:]  # keep pointers to pages.
                self.pages = []
                self.extend(new)  # handles pagination.
                self.pages.extend(tail)  # handles old pages.
                break

    def _setitem_insert(self, key, value):
        # documentation: new = old[:start] + list(values) + old[stop:]
        # L[3:5] = [1,2,3]
        key_start, key_stop, _ = key.indices(self._len)
        # create 3 partitions: A + B + C = head + new + tail

        result_head, result_tail = [], []
        # first partition:
        start, end = 0, 0
        for page in self.pages:
            start, end = end, end + page.len
            data = None
            if end <= key_start:
                result_head.append(page)

            if start <= key_start < end:  # end of head
                data = page.get()
                head = data[: key_start - start]

            if start <= key_stop < end:  # start of tail
                data = page.get() if data is None else data  # don't load again if on same page.
                tail = data[key_stop - start :]

            if key_stop < start:
                result_tail.append(page)

        middle = np_type_unify([head, value, tail])
        new_pages = self._paginate(middle)
        self.pages = result_head + new_pages + result_tail

    def _setitem_update(self, key, value):
        # documentation: See also test_slice_rules.py/MyList for details
        key_start, key_stop, key_step = key.indices(self._len)

        seq = range(key_start, key_stop, key_step)
        seq_size = len(seq)
        if len(value) > seq_size:
            raise ValueError(f"attempt to assign sequence of size {len(value)} to extended slice of size {seq_size}")

        # determine unchanged pages
        head, changed, tail = [], [], []
        start, end = 0, 0
        for page in self.pages:
            start, end = end, end + page.len

            if end <= key_start:
                head.append(page)
            elif start <= key_start < end:
                changed.append(page)
                starts_on = start
            elif start <= key_stop < end:
                changed.append(page)
            else:  # key_stop < start:
                tail.append(page)

        # determine changed pages.
        changed_pages = [p.get() for p in changed]
        dtypes = {arr.dtype for arr in (changed_pages + [value])}

        if len(dtypes) == 1:
            dtype = dtypes.pop()
        else:
            for ix, arr in enumerate(changed_pages):
                changed_pages[ix] = np.array(arr, dtype=object)
        new = np.concatenate(changed_pages, dtype=dtype)

        for index, position in zip(range(len(value)), seq):
            new[position] = value[index - starts_on]
        new_pages = self._paginate(new)
        # merge.
        self.pages = head + new_pages + tail

    def __delitem__(self, key):
        if isinstance(key, int):
            self._del_by_int(key)
        elif isinstance(key, slice):
            self._del_by_slice(key)
        else:
            raise KeyError(f"bad key: {key}")

    def _del_by_int(self, key):
        """del column[n]"""
        start, end = 0, 0
        for index, page in enumerate(self.pages):
            start, end = end, end + page.len
            if start <= key < end:
                data = page.get()
                new_data = np.delete(data, [key])
                new_page = Page(self.path, new_data)
                self.pages[index] = new_page

    def _del_by_slice(self, key):
        """del column[m:n:o]"""
        key_start, key_stop, key_step = key.indices(self._len)
        seq = range(key_start, key_stop, key_step)

        # determine change
        head, changed, tail = [], [], []
        start, end = 0, 0
        for page in self.pages:
            start, end = end, end + page.len
            if key_stop < start:
                head.append(page)
            elif start <= key_start < end:
                starts_on = start
                changed.append(page)
            elif start <= key_stop <= end:
                changed.append(page)
            else:  # key_stop < start:
                tail.append(page)

        # create np array
        changed_pages = [p.get() for p in changed]
        dtypes = {arr.dtype for arr in changed_pages}

        if len(dtypes) == 1:
            dtype = dtypes.pop()
        else:
            for ix, arr in enumerate(changed_pages):
                changed_pages[ix] = np.array(arr, dtype=object)
        new = np.concatenate(changed_pages, dtype=dtype)
        # create mask for np.delete.
        filter = [i - starts_on for i in seq]
        pruned = np.delete(new, filter)
        new_pages = self._paginate(pruned)
        self.pages = head + new_pages + tail

    def __iter__(self):  # USER FUNCTION.
        for page in self.pages:
            for value in page.get():
                yield value

    def __eq__(self, other):  # USER FUNCTION.
        if len(self) != len(other):  # quick cheap check.
            return False

        if isinstance(other, (list, tuple)):
            return all(a == b for a, b in zip(self[:], other))

        elif isinstance(other, Column):
            if self.pages == other.pages:  # special case.
                return True

            # are the pages of same size?
            if len(self.pages) == len(other.pages):
                if [p.len for p in self.pages] == [p.len for p in other.pages]:
                    for a, b in zip(self.pages, other.pages):
                        if not (a.get() == b.get()).all():
                            return False
                    return True
            # to bad. Element comparison it is then:
            for a, b in zip(iter(self), iter(other)):
                if a != b:
                    return False
            return True

        elif isinstance(other, np.ndarray):
            start, end = 0, 0
            for p in self.pages:
                start, end = end, end + p.len
                if not (p.get() == other[start:end]).all():
                    return False
            return True
        else:
            raise TypeError(f"Cannot compare {self.__class__} with {type(other)}")

    def __ne__(self, other):
        """
        compares two columns. Like list1 != list2
        """
        if len(self) != len(other):  # quick cheap check.
            return True

        if isinstance(other, (list, tuple)):
            return any(a != b for a, b in zip(self[:], other))

        elif isinstance(other, Column):
            if self.pages != other.pages:  # special case.
                return True

            # are the pages of same size?
            if len(self.pages) == len(other.pages):
                if [p.len for p in self.pages] == [p.len for p in other.pages]:
                    for a, b in zip(self.pages, other.pages):
                        if not (a.get() == b.get()).all():
                            return True
                    return False
            # to bad. Element comparison it is then:
            for a, b in zip(iter(self), iter(other)):
                if a != b:
                    return True
            return False

        elif isinstance(other, np.ndarray):
            start, end = 0, 0
            for p in self.pages:
                start, end = end, end + p.len
                if (p.get() != other[start:end]).any():
                    return True
            return False
        else:
            raise TypeError(f"Cannot compare {self.__class__} with {type(other)}")

    def copy(self):
        cp = Column(path=self.path)
        cp.pages = self.pages[:]
        return cp

    def __copy__(self):
        return self.copy()

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
        self.pages = self.pages[:] * other
        return self

    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError(f"a column can be repeated an integer number of times, not {type(other)} number of times")
        cp = self.copy()
        cp *= other
        return cp

    def remove_all(self, *values):
        """
        removes all values of `value`

        To remove only one instance of `value` use .remove
        """


class Table(object):
    _pid_dir = None  # workdir / gettpid /
    _ids = count()

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
        if _path is None:
            if self._pid_dir is None:
                self._pid_dir = Path(Config.workdir) / f"pid-{os.getpid()}"
                if not self._pid_dir.exists():
                    self._pid_dir.mkdir()
                    (self._pid_dir / "tables").mkdir()  # NOT USED.
                    (self._pid_dir / "pages").mkdir()
                    (self._pid_dir / "index").mkdir()  # NOT USED.
                register(self._pid_dir)

            _path = Path(self._pid_dir) / f"{next(self._ids)}.yml"
            # if it exists under the given PID it will be overwritten.
        type_check(_path, Path)
        self.path = _path  # filename used during multiprocessing.
        self.columns = {}  # maps colunn names to instances of Column.

        # user friendly features.
        if columns and any((headers, rows)):
            raise ValueError("Either columns as dict OR headers and rows. Not both.")

        if headers and rows:
            rotated = list(zip(*rows))
            columns = {k: v for k, v in zip(headers, rotated)}

        if columns:
            type_check(columns, dict)
            for k, v in columns.items():
                self.__setitem__(k, v)

    def __str__(self):  # USER FUNCTION.
        return f"{self.__class__.__name__}({len(self.columns):,} columns, {len(self):,} rows)"

    def __repr__(self):
        return self.__str__()

    def items(self):  # USER FUNCTION.
        """returns table as dict."""
        return {name: column[:].tolist() for name, column in self.columns.items()}.items()

    def __setitem__(self, key, value):  # USER FUNCTION
        """table behaves like a dict.
        Args:
            key (str): column name
            value (iterable): list, tuple or nd.array with values.

        As Table now accepts the keyword `columns` as a dict:
            t = Table(columns={'b':[4,5,6], 'c':[7,8,9]})
        and the header/data combinations:
            t = Table(header=['b','c'], data=[[4,5,6],[7,8,9]])

        it is no longer necessary to write:
            t = Table
            t['b'] = [4,5,6]
            t['c'] = [7,8,9]

        and the following assignment method is DEPRECATED:

            t = Table()
            t[('b','c')] = [ [4,5,6], [7,8,9] ]
            Which then produced the table with two columns:
            t['b'] == [4,5,6]
            t['c'] == [7,8,9]

        This has the side-benefit that tuples now can be used as headers.
        """
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        self.columns[key] = Column(self.path, value)

    def __getitem__(self, *keys):  # USER FUNCTION
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
            raise TypeError(f"Bad key: {keys}")
        if isinstance(keys[0], tuple):
            keys = tuple(list(chain(*keys)))

        integers = [i for i in keys if isinstance(i, int)]
        if len(integers) == len(keys) == 1:  # return a single tuple.
            keys = [slice(keys[0])]

        column_names = [i for i in keys if isinstance(i, str)]
        column_names = list(self.columns) if not column_names else column_names
        not_found = [name for name in column_names if name not in self.columns]
        if not_found:
            raise KeyError(f"keys not found: {', '.join(not_found)}")

        slices = [i for i in keys if isinstance(i, slice)]
        slc = slice(0, len(self)) if not slices else slices[0]

        if len(column_names) == 1:  # e.g. tbl['a'] or tbl['a'][:10]
            col = self.columns[column_names[0]]
            if slices:
                return col[slc]  # return slice from column as list of values
            else:
                return col  # return whole column

        elif len(integers) == 1:  # return a single tuple.
            row_no = integers[0]
            slc = slice(row_no, row_no + 1)
            return tuple(self.columns[name][slc].tolist()[0] for name in column_names)

        elif not slices:  # e.g. new table with N whole columns.
            t = self.__class__()
            for name in column_names:
                t.columns[name] = self.columns[name]  # link pointers, but make no copies.
            return t

        else:  # e.g. new table from selection of columns and slices.
            t = self.__class__()
            for name in column_names:
                column = self.columns[name]

                new_column = Column(t.path)  # create new Column.
                for item in column.getpages(slc):
                    if isinstance(item, np.ndarray):
                        new_column.extend(item)  # extend subslice (expensive)
                    elif isinstance(item, Page):
                        new_column.pages.append(item)  # extend page (cheap)
                    else:
                        raise TypeError(f"Bad item: {item}")

                # below:
                # set the new column directly on t.columns.
                # Do not use t[name] as that triggers __setitem__ again.
                t.columns[name] = new_column

            return t

    def __len__(self):  # USER FUNCTION.
        return max(len(c) for c in self.columns.values())

    def __eq__(self, other) -> bool:  # USER FUNCTION.
        """
        Determines if two tables have identical content.
        """
        if isinstance(other, dict):
            return self.items() == other.items()
        if not isinstance(other, Table):
            return False
        if id(self) == id(other):
            return True
        if len(self) != len(other):
            return False
        if len(self) == len(other) == 0:
            return True
        if self.columns.keys() != other.columns.keys():
            return False
        for name, col in self.columns.items():
            if col != other.columns[name]:
                return False
        return True

    def save(self, path):  # USER FUNCTION.
        """saves table to compressed tpz file.

        .tpz is a gzip archive with table metadata captured as table.yml
        and the necessary set of pages saved as .npy files.

        --------------------------------------
        %YAML 1.2                              yaml version
        temp = false                           temp identifier.
        columns:                               start of columns section.
            name: “列 1”                       name of column 1.
                pages: [p1b1, p1b2]            list of pages in column 1.
                length: [1_000_000, 834_312]   list of page-lengths
                types: [0,0]                   list of zeroes, so column 1 is a C-level data format.
            name: “列 2”                       name of column 2
                pages: [p2b1, p2b2]            list of pages in column 2.
                length: [1_000_000, 834_312]   list of page-lengths
                types: [p3b1, p3b2]            list of nonzero type codes, so column 2 is not a C-level data format.
        ----------------------------------------

        Args:
            path (Path): workdir / PID / tables / <int>.yml
            table (_type_): Table.
        """
        type_check(path, Path)
        if path.is_dir():
            raise TypeError(f"filename needed: {path}")
        if path.suffix != ".tpz":
            path += ".tpz"

        d = {"temp": False}
        cols = {}
        for name, col in self.columns.items():
            type_check(col, Column)
            cols[name] = {
                "pages": [p.path.name for p in col.pages],
                "length": [p.len for p in col.pages],
                "types": [0 for _ in col.pages],
            }
        d["columns"] = cols

        yml = yaml.safe_dump(d, sort_keys=False, allow_unicode=True, default_flow_style=None)

        with zipfile.ZipFile(path, "w") as f:  # raise if exists.
            log.debug(f"writing .tpz to {path} with\n{yml}")
            f.writestr("table.yml", yml)
            for name, col in self.columns.items():
                for page in col.pages:
                    with open(page.path, "rb", buffering=0) as raw_io:
                        f.writestr(page.path.name, raw_io.read())
                    log.debug(f"adding Page {page.path}")
            log.debug("write completed.")

    @classmethod
    def load(cls, path):  # USER FUNCTION.
        """loads a table from .tpz file.

        Args:
            path (Path): source file

        Returns:
            Table: table in read-only mode.
        """
        type_check(path, Path)
        log.debug(f"loading {path}")
        with zipfile.ZipFile(path, "r") as f:
            yml = f.read("table.yml")
            metadata = yaml.safe_load(yml)
            t = cls()
            for name, d in metadata["columns"].items():
                column = Column(t.path)
                for page in d["pages"]:
                    bytestream = io.BytesIO(f.read(page))
                    data = np.load(bytestream, allow_pickle=True, fix_imports=False)
                    column.extend(data)
                t.columns[name] = column
        return t

    def copy(self):
        cls = type(self)
        t = cls()
        for name, column in self.columns.items():
            new = Column(t.path)
            new.pages = column.pages[:]
            t.columns[name] = new
        return t


def test_basics():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t = Table(columns=data)
    assert t == data
    assert t.items() == data.items()

    a = t["A"]  # selects column 'a'
    assert isinstance(a, Column)
    b = t[3]  # selects row 3 as a tuple.
    assert isinstance(b, tuple)
    c = t[:10]  # selects first 10 rows from all columns
    assert isinstance(c, Table)
    assert len(c) == 10
    assert c.items() == {k: v[:10] for k, v in data.items()}.items()
    d = t["A", "B", slice(3, 20, 2)]  # selects a slice from columns 'a' and 'b'
    assert isinstance(d, Table)
    assert len(d) == 9
    assert d.items() == {k: v[3:20:2] for k, v in data.items() if k in ("A", "B")}.items()

    e = t["B", "A"]  # selects column 'b' and 'c' and 'a' twice for a slice.
    assert list(e.columns) == ["B", "A"], "order not maintained."

    x = t["A"]
    assert isinstance(x, Column)
    assert x == list(A)
    assert x == np.array(A)
    assert x == tuple(A)


def test_empty_table():
    t2 = Table()
    assert isinstance(t2, Table)
    t2["A"] = []
    assert len(t2) == 0
    c = t2["A"]
    assert isinstance(c, Column)
    assert len(c) == 0


def test_page_size():
    original_value = Config.PAGE_SIZE

    Config.PAGE_SIZE = 10
    assert Config.PAGE_SIZE == 10
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    # during __setitem__ Column.paginate will use config.
    t = Table(columns=data)
    assert t == data
    assert t.items() == data.items()

    x = t["A"]
    assert isinstance(x, Column)
    assert len(x.pages) == math.ceil(len(A) / Config.PAGE_SIZE)
    Config.PAGE_SIZE = 7
    t2 = Table(columns=data)
    assert t == t2
    x2 = t2["A"]
    assert isinstance(x2, Column)
    assert len(x2.pages) == math.ceil(len(A) / Config.PAGE_SIZE)

    Config.reset()
    assert Config.PAGE_SIZE == original_value


def test_cleaup():
    A = list(range(1, 10_000_000))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t = Table(columns=data)
    assert isinstance(t, Table)
    _folder = t._pid_dir
    _t_path = t.path

    del t
    import gc

    while gc.collect() > 0:
        pass

    assert _folder.exists()  # should be there until sigint.
    assert not _t_path.exists()  # should have been deleted


def save_and_load():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t = Table(columns=data)
    assert isinstance(t, Table)
    my_folder = Path(Config.workdir) / "data"
    my_folder.mkdir(exist_ok=True)
    my_file = my_folder / "my_first_file.tpz"
    t.save(my_file)
    assert my_file.exists()
    assert os.path.getsize(my_file) > 0

    del t
    import gc

    while gc.collect() > 0:
        pass

    assert my_file.exists()
    t2 = Table.load(my_file)

    t3 = Table(columns=data)
    assert t2 == t3
    assert t2.path.parent == t3.path.parent, "t2 must be loaded into same PID dir as t3"

    del t2
    while gc.collect() > 0:
        pass

    assert my_file.exists(), "deleting t2 MUST not affect the file saved by the user."
    t4 = Table.load(my_file)
    assert t4 == t3

    os.remove(my_file)
    os.rmdir(my_folder)


def test_copy():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t1 = Table(columns=data)
    assert isinstance(t1, Table)

    t2 = t1.copy()
    for name, t2column in t2.columns.items():
        t1column = t1.columns[name]
        assert id(t2column) != id(t1column)
        for p2, p1 in zip(t2column.pages, t1column.pages):
            assert id(p2) == id(p1), "columns can point to the same pages."


def test_speed():
    log.setLevel(logging.INFO)
    A = list(range(1, 10_000_000))
    B = [i * 10 for i in A]
    data = {"A": A, "B": B}
    t = Table(columns=data)
    import time
    import random

    loops = 100
    random.seed(42)
    start = time.time()
    for i in range(loops):
        a, b = random.randint(1, len(A)), random.randint(1, len(A))
        a, b = min(a, b), max(a, b)
        block = t["A"][a:b]  # pure numpy.
        assert len(block) == b - a
    end = time.time()
    print(f"numpy array: {end-start}")

    random.seed(42)
    start = time.time()
    for i in range(loops):
        a, b = random.randint(1, len(A)), random.randint(1, len(A))
        a, b = min(a, b), max(a, b)
        block = t["A"][a:b].tolist()  # python.
        assert len(block) == b - a
    end = time.time()
    print(f"python list: {end-start}")


def test_immutability_of_pages():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)
    change = t["A"][7:3:-1]
    t["A"][3:7] = change


def test_slice_functions():
    Config.PAGE_SIZE = 3
    t = Table(columns={"A": np.array([1, 2, 3, 4])})
    L = t["A"]
    assert list(L[:]) == [1, 2, 3, 4]  # slice(None,None,None)
    assert list(L[:0]) == []  # slice(None,0,None)
    assert list(L[0:]) == [1, 2, 3, 4]  # slice(0,None,None)

    assert list(L[:2]) == [1, 2]  # slice(None,2,None)

    assert list(L[-1:]) == [4]  # slice(-1,None,None)
    assert list(L[-1::1]) == [4]
    assert list(L[-1:None:1]) == [4]

    # assert list(L[-1:4:1]) == [4]  # slice(-1,4,1)
    # assert list(L[-1:0:-1]) == [4, 3, 2]  # slice(-1,0,-1) -->  slice(4,0,-1) --> for i in range(4,0,-1)
    # assert list(L[-1:0:1]) == []  # slice(-1,0,1) --> slice(4,0,-1) --> for i in range(4,0,1)
    # assert list(L[-3:-1:1]) == [2, 3]

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    t = Table(columns={"A": np.array(data)})
    L = t["A"]
    assert list(L[-1:0:-1]) == data[-1:0:-1]
    assert list(L[-1:0:1]) == data[-1:0:1]

    data = list(range(100))
    t2 = Table(columns={"A": np.array(data)})
    L = t2["A"]
    assert list(L[51:40:-1]) == data[51:40:-1]
    Config.reset()


def test_various():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)
    t *= 2
    assert len(t) == len(A) * 2
    x = t["x"]
    assert len(x) == len(A) * 2

    t2 = t.copy()
    t2 += t
    assert len(t2) == len(t) * 2
    assert len(t2["A"]) == len(t["A"]) * 2
    assert len(t2["B"]) == len(t["B"]) * 2
    assert len(t2["C"]) == len(t["C"]) * 2

    assert t != t2

    orphaned_column = t["A"].copy()
    orphaned_column_2 = orphaned_column * 2
    t3 = Table()
    t3["A"] = orphaned_column_2


if __name__ == "__main__":
    print("running unittest in main")
    """
    Joe: Why is this here? Shouldn't all tests be in /tests?

    Bjorn: Yes tests should be in /tests, but where you're refactoring
    it can be more efficient to maintain a list of tests locally in
    __main__ as it saves you from maintaining the imports.
    """
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s : %(message)s")
    console.setFormatter(formatter)
    log.addHandler(console)
    import time

    start = time.time()
    test_basics()
    test_empty_table()
    test_page_size()
    test_cleaup()
    save_and_load()
    test_copy()
    # test_speed()
    # test_immutability_of_pages()

    test_slice_functions()

    print(f"duration: {time.time()-start}")  # duration: 5.388719081878662 with 30M elements.
