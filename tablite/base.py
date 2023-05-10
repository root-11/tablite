import os
import math
import numpy as np
from pathlib import Path
from itertools import count, chain

from utils import type_check, intercept
import config


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

    def __len__(self):
        return self.len

    def delete(self):
        # trigger explicitly during cleanup UNLESS it's a users explicit save.
        os.remove(self.path)

    def get(self):
        return np.load(self.path, allow_pickle=True, fix_imports=False)


class Column(object):
    def __init__(self, path, value=None) -> None:
        self.path = path
        self.pages = []
        if value is not None:
            self.extend(value)

    def __len__(self):
        return sum(len(p) for p in self.pages)

    def extend_from_pages(self, page):
        self.pages.append(page)

    @staticmethod
    def _paginate(values, page_size=config.PAGE_SIZE):
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
        type_check(page_size, int)

        arrays = []
        n = 1 + int(math.ceil(len(values) // page_size) + 1) * page_size
        start = 0
        for end in range(page_size, n, page_size):
            x = np.array(values[start:end])
            arrays.append(x)
            start = end
        return arrays

    def from_pages(self, iterable):
        self.pages.extend(iterable)

    def extend(self, value):
        type_check(value, np.ndarray)
        for array in self._paginate(value):
            self.pages.append(Page(path=self.path.parent, array=array))

    def __getitem__(self, item):
        # user function.
        pages = self.getpages(item)
        result = []
        for element in pages:
            if isinstance(element, Page):
                result.append(element.get())
            else:
                result.append(element)

        dtypes = {arr.dtype: len(arr) for arr in result}
        if len(dtypes) == 1:
            dtype, _ = dtypes.popitem()
        else:
            for ix, arr in enumerate(result):
                result[ix] = np.array(arr, dtype=object)
        return np.concatenate(result, dtype=dtype)

    def getpages(self, item):
        # internal function
        type_check(item, slice)
        item = slice(*item.indices(len(self)))
        range_item = range(*item.indices(len(self)))
        pages = []
        start, end = 0, 0
        for page in self.pages:
            start, end = end, end + page.len
            if start > item.stop:
                break
            if end < item.start:
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
        return pages

    def __iter__(self):
        for page in self.pages:
            for value in page.get():
                yield value

    def __eq__(self, other):
        if len(self) != len(other):  # quick cheap check.
            return False

        if isinstance(other, (list, tuple)):
            return all(a == b for a, b in zip(self[:], other))

        elif isinstance(other, Column):
            if self.pages == other.pages:  # special case.
                return True
            for p1, p2 in zip(self.pages, other.pages):
                if not (p1.get() == p2.get()).all():
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
            raise TypeError


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
                self._pid_dir = Path(config.workdir) / f"pid-{os.getpid()}"
                if not self._pid_dir.exists():
                    self._pid_dir.mkdir()
                    (self._pid_dir / "tables").mkdir()
                    (self._pid_dir / "pages").mkdir()
                    (self._pid_dir / "index").mkdir()

            _path = Path(self._pid_dir) / f"{next(self._ids)}.yml"
            # if it exists under the given PID it will be overwritten.
        type_check(_path, Path)
        self.path = _path
        self.columns = {}

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

    def __str__(self):
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
                        new_column.extend_from_pages(item)  # extend page (cheap)
                    else:
                        raise TypeError(f"Bad item: {item}")

                # below:
                # set the new column directly on t.columns.
                # Do not use t[name] as that triggers __setitem__ again.
                t.columns[name] = new_column

            return t

    def __len__(self):
        return max(len(c) for c in self.columns.values())

    def __eq__(self, other) -> bool:
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


if __name__ == "__main__":
    print("running unittest in main")
    """
    Joe: Why is this here? Shouldn't all tests be in /tests?

    Bjorn: Yes tests should be in /tests, but where you're refactoring
    it can be more efficient to maintain a list of tests locally in
    __main__ as it saves you from maintaining the imports.
    """
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
