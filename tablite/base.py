import os
import io
import math
import yaml
import atexit
import shutil
import logging
import warnings
import zipfile
import numpy as np
from pathlib import Path
from itertools import count, chain, product, repeat
from collections import defaultdict

from tablite.datatypes import DataTypes, numpy_to_python, coerce_to_pytype
from tablite.utils import (
    type_check,
    intercept,
    np_type_unify,
    dict_to_rows,
    unique_name,
    summary_statistics,
)
from tablite.config import Config


log = logging.getLogger(__name__)


file_registry = set()


def register(path):
    """registers path in file_registry

    The method is used by Table during init when the working directory path
    is set, so that python can clean all temporary files up at exit.

    Args:
        path (Path): typically tmp/tablite-tmp/PID-{os.getpid()}
    """
    global file_registry
    file_registry.add(path)


def shutdown():
    """method to clean up temporary files triggered at shutdown."""
    for path in file_registry:
        if str(os.getpid()) in str(path):  # safety feature to prevent rm -rf /
            log.debug(f"shutdown: running rmtree({path})")
            shutil.rmtree(path)


atexit.register(shutdown)


class Page(object):
    ids = count(start=1)

    def __init__(self, path, array) -> None:
        """
        Args:
            path (Path): working directory.
            array (np.array): data
        """
        self.id = next(self.ids)
        type_check(path, Path)
        self.path = path / "pages" / f"{self.id}.npy"

        type_check(array, np.ndarray)

        self.len = len(array)
        np.save(self.path, array, allow_pickle=True, fix_imports=False)
        log.debug(f"Page saved: {self.path}")

    def __len__(self):
        return self.len

    def __hash__(self) -> int:
        return self.id

    def __del__(self):
        """When python's reference count for an object is 0, python uses
        it's garbage collector to remove the object and free the memory.
        As tablite tables have columns and columns have page and pages have
        data stored on disk, the space on disk must be freed up as well.
        This __del__ override assures the cleanup of stored data.
        """
        if self.path.exists():
            os.remove(self.path)
        log.debug(f"Page deleted: {self.path}")

    def get(self):
        """loads stored data

        Returns:
            np.ndarray: stored data.
        """
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
        page_size = Config.PAGE_SIZE if page_size is None else page_size
        type_check(page_size, int)

        arrays = []
        n = int(math.ceil(len(values) / page_size)) * page_size
        start = 0
        for end in range(page_size, n + 1, page_size):
            x = np.array(values[start:end])
            arrays.append(x)
            start = end
        return arrays

    def repaginate(self):
        """resizes pages to Config.PAGE_SIZE"""
        new_pages = []
        start, end = 0, 0
        for _ in range(0, len(self) + 1, Config.PAGE_SIZE):
            start, end = end, end + Config.PAGE_SIZE
            array = self[slice(start, end, step=1)]
            new_pages.extend(Page(self.path.parent, array))
        self.pages = new_pages

    def extend(self, value):  # USER FUNCTION.
        """extends the column.

        Args:
            value (np.ndarray): data
        """
        type_check(value, np.ndarray)
        for array in self._paginate(value):
            self.pages.append(Page(path=self.path.parent, array=array))

    def clear(self):
        """
        clears the column. Like list().clear()
        """
        self.pages.clear()

    def getpages(self, item):
        """public non-user function to identify any pages + slices
        of data to be retrieved given a slice (item)

        Args:
            item (int,slice): target slice of data

        Returns:
            list of pages/np.ndarrays.

        Example: [Page(1), Page(2), np.ndarray([4,5,6], int64)]
        This helps, for example when creating a copy, as the copy
        can reference the pages 1 and 2 and only need to store
        the np.ndarray that is unique to it.
        """
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
            if len(arr) == 0:
                raise IndexError(f"index {item} is out of bounds for axis 0 with size {len(self)}")
            return arr[0]
        else:
            return arr

    def __setitem__(self, key, value):  # USER FUNCTION.
        """sets values.

        Args:
            key (int,slice): selector
            value (any): values to insert

        Raises:
            KeyError: Following normal slicing rules
        """
        if isinstance(key, int):
            self._setitem_integer_key(key, value)

        elif isinstance(key, slice):
            if not isinstance(value, np.ndarray):
                value = np.array(value)
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
        else:
            raise KeyError(f"bad key: {key}")

    def _setitem_integer_key(self, key, value):  # PRIVATE FUNCTION
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
                new_page = Page(self.path.parent, data)
                self.pages[index] = new_page
                break

    def _setitem_replace_all(self, key, value):  # PRIVATE FUNCTION
        """handles the following case:
        new = list(value)
        example: L[:] = [1,2,3]
        """
        self.pages.clear()  # Page.__del__ will take care of removed pages.
        self.extend(value)  # Column.extend handles pagination.

    def _setitem_extend(self, key, value):  # PRIVATE FUNCTION
        """handles the following case:
        new = old[:key.start] + list(value)
        example: L[0:] = [1,2,3]
        """
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
        else:
            new = Page(self.path.parent, value)
            self.pages.append(new)

    def _setitem_prextend(self, key, value):  # PRIVATE FUNCTION
        """handles the following case:
        new = list(value) + old[key.stop:]
        example: L[:3] = [1,2,3]
        """
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

    def _setitem_insert(self, key, value):  # PRIVATE FUNCTION
        """handles the following case:
        new = old[:start] + list(values) + old[stop:]
        L[3:5] = [1,2,3]
        """
        key_start, key_stop, _ = key.indices(len(self))
        # create 3 partitions: A + B + C = head + new + tail

        unchanged_head, unchanged_tail = [], []
        # first partition:
        start, end = 0, 0
        for page in self.pages:
            start, end = end, end + page.len
            data = None
            if end <= key_start:
                unchanged_head.append(page)

            if start <= key_start < end:  # end of head
                data = page.get()
                head = data[: key_start - start]

            if start <= key_stop < end:  # start of tail
                data = page.get() if data is None else data  # don't load again if on same page.
                tail = data[key_stop - start :]

            if key_stop < start:
                unchanged_tail.append(page)

        new_middle = np_type_unify([head, value, tail])
        new_pages = [Page(self.path.parent, arr) for arr in self._paginate(new_middle)]
        self.pages = unchanged_head + new_pages + unchanged_tail

    def _setitem_update(self, key, value):
        """
        See test_slice_rules.py/MyList for detailed behaviour
        """
        key_start, key_stop, key_step = key.indices(len(self))

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
        new = np_type_unify(changed_pages)

        for index, val in zip(range(key_start, key_stop, key_step), value):
            new[index - starts_on] = val
        new_pages = [Page(self.path.parent, arr) for arr in self._paginate(new)]
        # merge.
        self.pages = head + new_pages + tail

    def __delitem__(self, key):  # USER FUNCTION
        """deletes items selected by key

        Args:
            key (int,slice): selector

        Raises:
            KeyError: following normal slicing rules.
        """
        if isinstance(key, int):
            self._del_by_int(key)
        elif isinstance(key, slice):
            self._del_by_slice(key)
        else:
            raise KeyError(f"bad key: {key}")

    def _del_by_int(self, key):  # PRIVATE FUNCTION
        """handles the following case:
        del column[n]
        """
        start, end = 0, 0
        for index, page in enumerate(self.pages):
            start, end = end, end + page.len
            if start <= key < end:
                data = page.get()
                new_data = np.delete(data, [key])
                new_page = Page(self.path, new_data)
                self.pages[index] = new_page

    def _del_by_slice(self, key):
        """handles the following case:
        del column[m:n:o]
        """
        key_start, key_stop, key_step = key.indices(len(self))
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
        new = np_type_unify(changed_pages)
        # create mask for np.delete.
        filter = [i - starts_on for i in seq]
        pruned = np.delete(new, filter)
        new_arrays = self._paginate(pruned)
        self.pages = head + [Page(self.path.parent, arr) for arr in new_arrays] + tail

    def __iter__(self):  # USER FUNCTION.
        for page in self.pages:
            data = page.get()
            for value in data:
                yield value

    def __eq__(self, other):  # USER FUNCTION.
        """
        compares two columns. Like list1 == list2
        """
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

    def __ne__(self, other):  # USER FUNCTION
        """
        compares two columns. Like list1 != list2
        """
        if len(self) != len(other):  # quick cheap check.
            return True

        if isinstance(other, (list, tuple)):
            return any(a != b for a, b in zip(self[:], other))

        elif isinstance(other, Column):
            if self.pages == other.pages:  # special case.
                return False

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
        """returns deep=copy of Column

        Returns:
            Column
        """
        cp = Column(path=self.path)
        cp.pages = self.pages[:]
        return cp

    def __copy__(self):
        """see copy"""
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
        if not (isinstance(other, int) and other > 0):
            raise TypeError(f"a column can be repeated an integer number of times, not {type(other)} number of times")
        self.pages = self.pages[:] * other
        return self

    def __mul__(self, other):
        """
        Repeats instance of column N times. Like list() * N

        Example:
        >>> one = Column(data=[1,2])
        >>> two = one * 5
        >>> two
        [1,2, 1,2, 1,2, 1,2, 1,2]

        """
        if not isinstance(other, int):
            raise TypeError(f"a column can be repeated an integer number of times, not {type(other)} number of times")
        cp = self.copy()
        cp *= other
        return cp

    def __iadd__(self, other):
        if isinstance(other, (list, tuple)):
            other = np.array(other)
            self.extend(other)
        elif isinstance(other, Column):
            self.pages.extend(other.pages[:])
        else:
            raise TypeError(f"{type(other)} not supported.")
        return self

    def __contains__(self, item):
        """
        determines if item is in the Column. Similar to 'x' in ['a','b','c']
        returns boolean
        """
        for page in self.pages:
            if item in page.get():  # x in np.ndarray([...]) uses np.any(arr, value)
                return True
        return False

    def remove_all(self, *values):
        """
        removes all values of `values`
        """
        type_check(values, tuple)
        if isinstance(values[0], tuple):
            values = values[0]
        to_remove = np.array(values)
        for index, page in enumerate(self.pages):
            data = page.get()
            bitmask = np.isin(data, to_remove)  # identify elements to remove.
            if bitmask.any():
                bitmask = np.invert(bitmask)  # turn bitmask around to keep.
                new_data = np.compress(bitmask, data)
                new_page = Page(self.path.parent, new_data)
                self.pages[index] = new_page

    def replace(self, mapping):
        """
        replaces values using mapping

        example:
        >>> t = Table(columns={'A': [1,2,3,4]})
        >>> t['A'].replace({2:20,4:40})
        >>> t[:]
        np.ndarray([1,20,3,40])
        """
        type_check(mapping, dict)
        to_replace = np.array(list(mapping.keys()))
        for index, page in enumerate(self.pages):
            data = page.get()
            bitmask = np.isin(data, to_replace)  # identify elements to replace.
            if bitmask.any():
                warray = np.compress(bitmask, data)
                for ix, v in enumerate(np.nditer(warray)):
                    warray[ix] = mapping[v.item()]
                data[bitmask] = warray
                self.pages[index] = Page(path=self.path.parent, array=data)

    def types(self):
        """
        returns dict with python datatypes: frequency of occurrence
        """
        d = defaultdict(int)
        for page in self.pages:
            data = page.get()
            if data.dtype == "O":
                for i in data:
                    dtype = coerce_to_pytype(i)
                    d[dtype] += 1
            else:
                sample = coerce_to_pytype(data[0])
                d[sample] += len(page)
        return dict(d)

    def index(self):
        """
        returns dict with { unique entry : list of indices }

        example:
        >>> c = Column(data=['a','b','a','c','b'])
        >>> c.index()
        {'a':[0,2], 'b': [1,4], 'c': [3]}

        """
        d = defaultdict(list)
        for ix, v in enumerate(self.__iter__()):
            d[v].append(ix)

    def unique(self):
        """
        returns unique list of values.

        example:
        >>> c = Column(data=['a','b','a','c','b'])
        >>> c.unqiue()
        ['a','b','c']
        """
        arrays = []
        for page in self.pages:
            try:  # when it works, numpy is fast...
                arrays.append(np.unique(page.get()))
            except TypeError:  # ...but np.unique cannot handle Nones.
                arrays.append(list(set(page.get())))
        union = np_type_unify(arrays)
        try:
            return np.unique(union)
        except TypeError:
            return np.array(list(set(union)))

    def histogram(self):
        """
        returns 2 arrays: unique elements and count of each element

        example:
        >>> c = Column(data=['a','b','a','c','b'])
        >>> c.histogram()
        {'a':2,'b':2,'c':1}
        """
        d = defaultdict(int)
        for page in self.pages:
            uarray, carray = np.unique(page.get(), return_counts=True)
            for i, c in zip(uarray, carray):
                d[i] += c
        return d

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
        htg = self.histogram()
        values, counts = list(htg.keys()), list(htg.values())
        return summary_statistics(values, counts)

    def count(self, item):
        result = 0
        for page in self.pages:
            result += np.nonzero(page.get() == item)[0].shape[0]
            # what happens here ---^ below:
            # arr = page.get()
            # >>> arr
            # array([1,2,3,4,3], int64)
            # >>> (arr == 3)
            # array([False, False,  True, False,  True])
            # >>> np.nonzero(arr==3)
            # (array([2,4], dtype=int64), )  <-- tuple!
            # >>> np.nonzero(page.get() == item)[0]
            # array([2,4])
            # >>> np.nonzero(page.get() == item)[0].shape
            # (2, )
            # >>> np.nonzero(page.get() == item)[0].shape[0]
            # 2
        return result


class Table(object):
    _pid_dir = None  # typically Path(Config.workdir) / f"pid-{os.getpid()}"
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
            # if path exists under the given PID it will be overwritten.
            # this can only happen if the process previously was SIGKILLed.
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

    def __delitem__(self, key):  # USER FUNCTION.
        """
        del table['a']  removes column 'a'
        del table[-3:] removes last 3 rows from all columns.
        """
        if isinstance(key, (int, slice)):
            for column in self.columns.values():
                del column[key]
        elif key in self.columns:
            del self.columns[key]
        else:
            raise KeyError(f"Key not found: {key}")

    def __setitem__(self, key, value):  # USER FUNCTION
        """table behaves like a dict.
        Args:
            key (str or hashable): column name
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
        if value is None:
            self.columns[key] = Column(self.path, value=None)
        elif isinstance(value, Column):
            self.columns[key] = value
        elif isinstance(value, (list, tuple, np.ndarray)):
            value = np.array(value)
            self.columns[key] = Column(self.path, value)
        else:
            raise TypeError(f"{type(value)} not supported.")

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
        if not self.columns:
            return 0
        return max(len(c) for c in self.columns.values())

    @property
    def rows(self):
        """
        enables row based iteration

        for row in Table.rows:
            print(row)
        """
        n_max = len(self)
        generators = []
        for name, column in self.columns.items():
            if len(column) < n_max:
                warnings.warn(f"Column {name} has length {len(column)} / {n_max}. None will appear as fill value.")
            generators.append(chain(iter(column), repeat(None, times=n_max - len(column))))

        for _ in range(len(self)):
            yield [next(i) for i in generators]

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
            if not (col == other.columns[name]):
                return False
        return True

    def clear(self):  # USER FUNCTION.
        """
        clears the table. Like dict().clear()
        """
        self.columns.clear()

    def save(self, path, compression_method=zipfile.ZIP_STORED, compression_level=None):  # USER FUNCTION.
        """saves table to compressed tpz file.

        Args:
            path (Path): workdir / PID / tables / <int>.yml
            compression_method: See zipfile compression methods. Default no compression.
            compression_level: See zipfile compression levels. Defaults to None.
            The defaults are the fastest mode of operation.

        .tpz is a gzip archive with table metadata captured as table.yml
        and the necessary set of pages saved as .npy files.

        The zip contains table.yml which provides an overview of the data:
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
                types: [p3b1, p3b2]            list of nonzero type codes, so column 2 is not a C
                                                 -level data format. The type codes are available
                                                 in datatypes.Datatypes as _type_codes
        ----------------------------------------

        """
        type_check(path, Path)
        if path.is_dir():
            raise TypeError(f"filename needed: {path}")
        if path.suffix != ".tpz":
            path += ".tpz"

        _page_counter = 0
        d = {"temp": False}
        cols = {}
        for name, col in self.columns.items():
            type_check(col, Column)
            cols[name] = {
                "pages": [p.path.name for p in col.pages],
                "length": [p.len for p in col.pages],
                "types": [0 for _ in col.pages],
            }
            _page_counter += len(col.pages)
        d["columns"] = cols

        yml = yaml.safe_dump(d, sort_keys=False, allow_unicode=True, default_flow_style=None)

        _file_counter = 0
        with zipfile.ZipFile(
            path, "w", compression=compression_method, compresslevel=compression_level
        ) as f:  # raise if exists.
            log.debug(f"writing .tpz to {path} with\n{yml}")
            f.writestr("table.yml", yml)
            for name, col in self.columns.items():
                for page in set(col.pages):  # set of pages! remember t *= 1000 repeats t 1000x
                    with open(page.path, "rb", buffering=0) as raw_io:
                        f.writestr(page.path.name, raw_io.read())
                    _file_counter += 1
                    log.debug(f"adding Page {page.path}")

            _fields = len(self) * len(self.columns)
            _avg = _fields // _page_counter
            log.debug(f"Wrote {_fields} on {_page_counter} pages in {_file_counter} files: {_avg} fields/page")

    @classmethod
    def load(cls, path):  # USER FUNCTION.
        """loads a table from .tpz file.
        See also Table.save for details on the file format.

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

    def __imul__(self, other):
        """Repeats instance of table N times.
        Like list: t = t * N

        Args:
            other (int): multiplier
        """
        if not (isinstance(other, int) and other > 0):
            raise TypeError(f"a table can be repeated an integer number of times, not {type(other)} number of times")
        for col in self.columns.values():
            col *= other
        return self

    def __mul__(self, other):
        """Repeat table N times.
        Like list: new = old * N

        Args:
            other (int): multiplier

        Returns:
            Table
        """
        new = self.copy()
        return new.__imul__(other)

    def __iadd__(self, other):
        """Concatenates tables with same column names.

        Liek list: table_1 += table_2

        Args:
            other (Table)

        Raises:
            ValueError: If column names don't match.

        Returns:
            None: self is updated.
        """
        type_check(other, Table)
        for name in self.columns.keys():
            if name not in other.columns:
                raise ValueError(f"{name} not in other")
        for name in other.columns.keys():
            if name not in self.columns:
                raise ValueError(f"{name} missing from self")

        for name, column in self.columns.items():
            other_col = other.columns.get(name, None)
            column.pages.extend(other_col.pages[:])
        return self

    def __add__(self, other):
        """Concatenates tables with same column names.

        Liek list: table_3 = table_1 + table_2

        Args:
            other (Table)

        Raises:
            ValueError: If column names don't match.

        Returns:
            Table
        """
        type_check(other, Table)
        cp = self.copy()
        cp += other
        return cp

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

                d = {n: [] for n in self.columns}
                for arg in args:
                    if len(arg) != len(self.columns):
                        raise ValueError(f"len({arg})== {len(arg)}, but there are {len(self.columns)} columns")

                    if isinstance(arg, dict):
                        for k, v in arg.items():  # 7,8
                            d[k].append(v)

                    elif isinstance(arg, (list, tuple)):  # 2,3
                        for n, v in zip(self.columns, arg):
                            d[n].append(v)

                    else:
                        raise TypeError(f"{arg}?")
                # 2. extend the columns
                for n, values in d.items():
                    col = self.columns[n]
                    col.extend(np.array(values))

        if kwargs:
            if isinstance(kwargs, dict):
                if all(isinstance(v, (list, tuple)) for v in kwargs.values()):
                    for k, v in kwargs.items():
                        col = self.columns[k]
                        col.extend(np.array(v))
                else:
                    for k, v in kwargs.items():
                        col = self.columns[k]
                        col.extend(np.array([v]))
            else:
                raise ValueError(f"format not recognised: {kwargs}")

        return

    def add_columns(self, *names):
        """Adds column names to table."""
        for name in names:
            self.columns[name] = Column(self.path)

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

        cp = self.copy()
        for name, col2 in other.columns.items():
            if name not in cp.columns:
                cp[name] = [None] * len(self)
            cp[name].pages.extend(col2.pages[:])

        for name in self.columns:
            if name not in other.columns:
                if len(cp) > 0:
                    cp[name].extend(np.array([None] * len(other)))
        return cp

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
        for name, col in self.columns.items():
            assert isinstance(col, Column)
            d[name] = col.types()
        return d

    def display_dict(self, *args, blanks=None, dtype=False):
        """
        param: args:
          - slice
        blanks: fill value for `None`
        dtype: add datatype for each column
        """
        if not self.columns:
            print("Empty Table")
            return

        def datatype(col):  # PRIVATE
            """creates label for column datatype."""
            types = col.types()
            if len(types) == 1:
                dt, _ = types.popitem()
                typ = dt.__name__
            else:
                typ = "mixed"
            return typ

        row_count_tags = ["#", "~", "*"]
        cols = set(self.columns)
        for n, tag in product(range(1, 6), row_count_tags):
            if n * tag not in cols:
                tag = n * tag
                break
        slc = slice(0, 20, 1) if len(self) <= 20 else None
        if args:
            for arg in args:
                if isinstance(arg, slice):
                    slc = arg
                    break

        n = len(self)
        if slc:
            row_no = list(range(*slc.indices(len(self))))
            data = {tag: [f"{i:,}".rjust(2) for i in row_no]}
            for name, col in self.columns.items():
                data[name] = list(chain(iter(col), repeat(blanks, times=n - len(col))))
        else:
            data = {}
            j = int(math.ceil(math.log10(n)) / 3) + len(str(n))
            row_no = [f"{i:,}".rjust(j) for i in range(7)] + ["..."] + [f"{i:,}".rjust(j) for i in range(n - 7, n)]
            data = {tag: row_no}

            for name, col in self.columns.items():
                if len(col) == n:
                    row = col[:7].tolist() + ["..."] + col[-7:].tolist()
                else:
                    empty = [blanks] * 7
                    head = (col[:7].tolist() + empty)[:7]
                    tail = (col[n - 7 :].tolist() + empty)[-7:]
                    row = head + ["..."] + tail
                data[name] = row

        if dtype:
            for name, values in data.items():
                if name in self.columns:
                    col = self.columns[name]
                    values.insert(0, datatype(col))
                else:
                    values.insert(0, "row")

        return data

    def to_ascii(self, *args, blanks=None, dtype=False):
        """returns ascii view of table as string.

        Args:
            blanks (str, optional): value for whitespace. Defaults to None.
            dtype (bool, optional): adds subheader with datatype for column. Defaults to False.
        """

        def adjust(v, length):  # PRIVATE FUNCTION
            """whitespace justifies field values based on datatype"""
            if v is None:
                return str(blanks).ljust(length)
            elif isinstance(v, str):
                return v.ljust(length)
            else:
                return str(v).rjust(length)

        if not self.columns:
            return str(self)

        d = {}
        for name, values in self.display_dict(*args, blanks=blanks, dtype=dtype).items():
            as_text = [str(v) for v in values] + [str(name)]
            width = max(len(i) for i in as_text)
            new_name = name.center(width, " ")
            if dtype:
                values[0] = values[0].center(width, " ")
            d[new_name] = [adjust(v, width) for v in values]

        rows = dict_to_rows(d)
        s = []
        s.append("+" + "+".join(["=" * len(n) for n in rows[0]]) + "+")
        s.append("|" + "|".join(rows[0]) + "|")  # column names
        start = 1
        if dtype:
            s.append("|" + "|".join(rows[1]) + "|")  # datatypes
            start = 2

        s.append("+" + "+".join(["-" * len(n) for n in rows[0]]) + "+")
        for row in rows[start:]:
            s.append("|" + "|".join(row) + "|")
        s.append("+" + "+".join(["=" * len(n) for n in rows[0]]) + "+")

        if len(set(len(c) for c in self.columns.values())) != 1:
            warning = f"Warning: Columns have different lengths. {blanks} is used as fill value."
            s.append(warning)

        return "\n".join(s)

    def show(self, *args, blanks=None, dtype=False):
        """prints ascii view of table.

        Args:
            blanks (str, optional): value for whitespace. Defaults to None.
            dtype (bool, optional): adds subheader with datatype for column. Defaults to False.
        """
        print(self.to_ascii(*args, blanks=blanks, dtype=dtype))

    def _repr_html_(self, *args, blanks=None, dtype=False):
        """Ipython display compatible format
        https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
        """
        start, end = "<div><table border=1>", "</table></div>"

        if not self.columns:
            return f"{start}<tr>Empty Table</tr>{end}"
        rows = dict_to_rows(self.display_dict(*args, blanks=blanks, dtype=dtype))
        html = "".join(["<tr>" + "".join(f"<th>{cn}</th>" for cn in row) + "</tr>" for row in rows])

        warning = ""
        if len(set(len(c) for c in self.columns.values())) != 1:
            warning = f"Warning: Columns have different lengths. {blanks} is used as fill value."

        return start + "".join(html) + end + warning

    def to_dict(self, columns=None, slice_=None):
        """
        columns: list of column names. Default is None == all columns.
        slice_: slice. Default is None == all rows.

        returns: dict with columns as keys and lists of values.

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
            columns = list(self.columns.keys())
        if not isinstance(columns, list):
            raise TypeError("expected columns as list of strings")

        return {name: list(self.columns[name][slice_]) for name in columns}

    def as_json_serializable(self, row_count="row id", start_on=1, columns=None, slice_=None):
        """provides a JSON compatible format of the table.

        Args:
            row_count (str, optional): Label for row counts. Defaults to "row id".
            start_on (int, optional): row counts starts by default on 1.
            columns (list of str, optional): Column names.
                Defaults to None which returns all columns.
            slice_ (slice, optional): selector. Defaults to None which returns [:]

        Returns:
            JSON serializable dict: All python datatypes have been converted to JSON compliant data.
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
        iterators = [iter(self.columns[c]) for c in args]
        for ix, key in enumerate(zip(*iterators)):
            key = tuple(numpy_to_python(k) for k in key)
            idx[key].add(ix)
        return idx


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
    assert not t != t2
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
    x = t["A"]
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

    orphaned_column.replace(mapping={2: 20, 3: 30, 4: 40})
    z = set(orphaned_column[:].tolist())
    assert {2, 3, 4}.isdisjoint(z)
    assert {20, 30, 40}.issubset(z)

    t4 = t + t2
    assert len(t4) == len(t) + len(t2)


def test_types():
    # SINGLE TYPES.
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    c = t["A"]
    assert c.types() == {int: len(A)}

    assert t.types() == {"A": {int: len(A)}, "B": {int: len(A)}, "C": {int: len(A)}}

    # MIXED DATATYPES
    A = list(range(5))
    B = list("abcde")
    data = {"A": A, "B": B}
    t = Table(columns=data)
    typ1 = t.types()
    expected = {"A": {int: 5}, "B": {str: 5}}
    assert typ1 == expected
    more = {"A": B, "B": A}
    t += Table(columns=more)
    typ2 = t.types()
    expected2 = {"A": {int: 5, str: 5}, "B": {str: 5, int: 5}}
    assert typ2 == expected2


def test_table_row_functions():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    t2 = Table()
    t2.add_column("A")
    t2.add_columns("B", "C")
    t2.add_rows(**{"A": [i + max(A) for i in A], "B": [i + max(A) for i in A], "C": [i + max(C) for i in C]})
    t3 = t2.stack(t)
    assert len(t3) == len(t2) + len(t)

    t4 = Table(columns={"B": [-1, -2], "D": [0, 1]})
    t5 = t2.stack(t4)
    assert len(t5) == len(t2) + len(t4)
    assert list(t5.columns) == ["A", "B", "C", "D"]


def test_remove_all():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    c = t["A"]
    c.remove_all(3)
    A.remove(3)
    assert list(c) == A
    c.remove_all(4, 5, 6)
    A = [i for i in A if i not in {4, 5, 6}]
    assert list(c) == A


def test_replace():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    c = t["A"]
    c.replace({3: 30, 4: 40})
    assert list(c) == [i if i not in {3, 4} else i * 10 for i in A]


def test_display_options():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)

    d = t.display_dict()
    assert d == {
        "#": [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"],
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "C": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    }

    txt = t.to_ascii()
    expected = """\
+==+==+===+====+
|# |A | B | C  |
+--+--+---+----+
| 0| 1| 10| 100|
| 1| 2| 20| 200|
| 2| 3| 30| 300|
| 3| 4| 40| 400|
| 4| 5| 50| 500|
| 5| 6| 60| 600|
| 6| 7| 70| 700|
| 7| 8| 80| 800|
| 8| 9| 90| 900|
| 9|10|100|1000|
+==+==+===+====+"""
    assert txt == expected
    txt2 = t.to_ascii(dtype=True)
    expected2 = """\
+===+===+===+====+
| # | A | B | C  |
|row|int|int|int |
+---+---+---+----+
| 0 |  1| 10| 100|
| 1 |  2| 20| 200|
| 2 |  3| 30| 300|
| 3 |  4| 40| 400|
| 4 |  5| 50| 500|
| 5 |  6| 60| 600|
| 6 |  7| 70| 700|
| 7 |  8| 80| 800|
| 8 |  9| 90| 900|
| 9 | 10|100|1000|
+===+===+===+====+"""
    assert txt2 == expected2

    html = t._repr_html_()

    expected3 = """<div><table border=1>\
<tr><th>#</th><th>A</th><th>B</th><th>C</th></tr>\
<tr><th> 0</th><th>1</th><th>10</th><th>100</th></tr>\
<tr><th> 1</th><th>2</th><th>20</th><th>200</th></tr>\
<tr><th> 2</th><th>3</th><th>30</th><th>300</th></tr>\
<tr><th> 3</th><th>4</th><th>40</th><th>400</th></tr>\
<tr><th> 4</th><th>5</th><th>50</th><th>500</th></tr>\
<tr><th> 5</th><th>6</th><th>60</th><th>600</th></tr>\
<tr><th> 6</th><th>7</th><th>70</th><th>700</th></tr>\
<tr><th> 7</th><th>8</th><th>80</th><th>800</th></tr>\
<tr><th> 8</th><th>9</th><th>90</th><th>900</th></tr>\
<tr><th> 9</th><th>10</th><th>100</th><th>1000</th></tr>\
</table></div>"""
    assert html == expected3

    A = list(range(1, 51))
    B = [i * 10 for i in A]
    C = [i * 1000 for i in B]
    data = {"A": A, "B": B, "C": C}
    t2 = Table(columns=data)
    d2 = t2.display_dict()
    assert d2 == {
        "#": [" 0", " 1", " 2", " 3", " 4", " 5", " 6", "...", "43", "44", "45", "46", "47", "48", "49"],
        "A": [1, 2, 3, 4, 5, 6, 7, "...", 44, 45, 46, 47, 48, 49, 50],
        "B": [10, 20, 30, 40, 50, 60, 70, "...", 440, 450, 460, 470, 480, 490, 500],
        "C": [
            10000,
            20000,
            30000,
            40000,
            50000,
            60000,
            70000,
            "...",
            440000,
            450000,
            460000,
            470000,
            480000,
            490000,
            500000,
        ],
    }


def test_index():
    A = [1, 1, 2, 2, 3, 3, 4]
    B = list("asdfghjkl"[: len(A)])
    data = {"A": A, "B": B}
    t = Table(columns=data)
    t.index("A")
    t.index("A", "B")


def test_to_dict():
    # to_dict and as_json_serializable
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = Table(columns=data)
    assert t.to_dict() == data


def test_unique():
    t = Table({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert np.all(t["A"].unique() == np.array([1, 2, 3, 4]))


def test_histogram():
    t = Table({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert t["A"].histogram() == {1: 3, 2: 3, 3: 2, 4: 1}


def test_count():
    t = Table({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert t["A"].count(2) == 3


def test_contains():
    t = Table({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert 3 in t["A"]
    assert 7 not in t["A"]


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
    test_various()
    test_types()
    test_table_row_functions()
    test_remove_all()
    test_replace()
    test_display_options()
    test_index()
    test_to_dict()
    test_unique()
    test_histogram()
    test_count()

    print(f"duration: {time.time()-start}")  # duration: 5.388719081878662 with 30M elements.
