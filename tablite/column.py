from itertools import Count
from pathlib import Path
from utils import type_check, summary_statistics
import shutil
import config
import math
import numpy as np


class Page(object):
    _id_counter = Count()

    def __init__(self, path, type_path=None) -> None:
        self.id = next(self._id_counter)
        type_check(path, Path)
        self.path = path

        if type_path is not None:
            type_check(type_path, Path)
        self.type_path = type_path
    
    def __getitem__(self, item):




    def delete(self):
        pass

    def 


class Column(object):
    def __init__(self, path, data=None) -> None:
        """
        path (Path): table.path
        data: list of values
        key: (default None) id used during Table.load to instantiate the column.
        """
        type_check(path, Path)
        self._pages = []  # [9,9,1,4,7,5,...]
        self._index = []  # page sizes [1000,1000,5000,....]
        self._types = []  # page type [0,0,2,5,8,6,...]  # simple,simple, complex
        self._datatypes = {}  # {'int': 6500, 'float':4499, 'None':1}
        self._len = max(self._index)
        self.path = path  # table.yml path.
v
        self.extend(data)

    @property
    def pages(self):
        return zip(self._pages, self._index, self._types)

    def __len__(self):
        """
        returns number of entries in the Column. Like len(list())
        """
        return self._len

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>({self._len} values | key={self.key})"

    def __repr__(self) -> str:
        return self.__str__()

    def types(self):
        """
        returns dict with datatype: frequency of occurrence
        """
        return {k: v for k, v in self._datatypes.items()}  # return copy so the state is safe.

    def __setitem__(self, keys, values):
        if isinstance(keys, int):
            if isinstance(values, (Column, list, tuple)):
                raise TypeError(
                    f"your key is an integer, but your value is a {type(values)}. \
                        Did you mean to insert? F.x. [{keys}:{keys+1}] = {values} ?"
                )
            if not (-self._len - 1 < keys < self._len):
                raise IndexError("list assignment index out of range")

            self._setitem_key(keys, values)

        elif isinstance(keys, slice):
            if isinstance(values, Column):
                self._setitem_column(keys, values)
            else:
                self._setitem_values(keys, values)
        else:
            raise TypeError(f"Bad key type: {type(keys)}")

    def __getitem__(self, *keys):
        raise NotImplementedError("subclasses must implement this.")

    def __delitem__(self, key):
        raise NotImplementedError("subclasses must implement this.")

    def append(self, x):
        """
        Add an item to the end of the list.
        Equivalent to a[len(a):] = [x].
        """
        self.__setitem__(key=slice(self._len, None, None), value=[value])

    def extend(self, iterable):
        """
        Extend the list by appending all the items from the iterable.
        Equivalent to a[len(a):] = iterable.
        """
        self.__setitem__(slice(self._len, None, None), iterable)

    def insert(self, i, x):
        """
        Insert an item at a given position.
        The first argument is the index of the element before which
        to insert, so a.insert(0, x) inserts at the front of the list,
        and a.insert(len(a), x) is equivalent to a.append(x).
        """
        pass

    def remove(self, x):
        """Remove the first item from the list whose value is equal to x."""
        pass

    def remove_all(self, x):
        """Remove the items from the list whose value is equal to x."""
        pass

    def pop(self, i):
        """Remove the item at the given position in the list, and return it."""
        pass

    def clear(self):
        """Remove all items from the list. Equivalent to del a[:]."""
        pass

    def index(x, start=0, end=-1):
        """
        Return zero-based index in the list of the first item whose value is equal to x.
        Raises a ValueError if there is no such item.
        """
        pass

    def count(self, x):
        """Return the number of times x appears in the list."""
        return sum(1 for i in self.__getitem__() if i == x)

    def sort(self, *, key=None, reverse=False):
        """
        Sort the items of the list in place (the arguments can be used
        for sort customization, see sorted() for their explanation).
        """
        pass

    def reverse(self):
        """Reverse the elements of the list in place."""
        pass

    def copy(self):
        """
        Return a shallow copy of the list. Equivalent to a[:].
        """
        return Column(data=self)

    def __copy__(self):
        return self.copy()

    def indices(self):  # WAS: Index
        """
        returns dict with { unique entry : list of indices }

        example:
        >>> c = Column(data=['a','b','a','c','b'])
        >>> c.index()
        {'a':[0,2], 'b': [1,4], 'c': [3]}

        """
        data = self.__getitem__()
        d = {k: [] for k in np.unique(data)}
        for ix, k in enumerate(data):
            d[k].append(ix)
        return d

    def unique(self):
        """
        returns unique list of values.

        example:
        >>> c = Column(data=['a','b','a','c','b'])
        >>> c.unqiue()
        ['a','b','c']
        """
        try:
            return np.unique(self.__getitem__())
        except TypeError:  # np arrays can't handle dtype='O':
            return np.array({i for i in self.__getitem__()})

    def histogram(self):
        """
        returns 2 arrays: unique elements and count of each element

        example:
        >>> c = Column(data=['a','b','a','c','b'])
        >>> c.unqiue()
        ['a','b','c'],[2,2,1]
        """
        try:
            uarray, carray = np.unique(self.__getitem__(), return_counts=True)
            uarray, carray = uarray.tolist(), carray.tolist()
        except TypeError:  # np arrays can't handle dtype='O':
            d = defaultdict(int)
            for i in self.__getitem__():
                d[i] += 1
            uarray, carray = [], []
            for k, v in d.items():
                uarray.append(k), carray.append(v)
        return uarray, carray

    def replace(self, target, replacement):
        """
        replaces target with replacement
        """
        pass

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
        return summary_statistics(*self.histogram())

    def __add__(self, other):
        """
        Concatenates to Columns. Like list() + list()

        Example:
        >>> one,two = Column(data=[1,2]), Column(data=[3,4])
        >>> both = one+two
        >>> both[:]
        [1,2,3,4]
        """
        c = self.copy()
        c.extend(other)
        return c

    def __contains__(self, item):
        """
        determines if item is in the Column. Similar to 'x' in ['a','b','c']
        returns boolean
        """
        return item in self.__getitem__()

    def __iadd__(self, other):
        """
        Extends instance of Column with another Column

        Example:
        >>> one,two = Column(data=[1,2]), Column(data=[3,4])
        >>> one += two
        >>> one[:]
        [1,2,3,4]

        """
        self.extend(other)
        return self

    def __eq__(self, other):
        pass

    def __le__(self, other):
        raise NotImplementedError("vectorised operation A <= B is type-ambiguous")

    def __lt__(self, other):
        raise NotImplementedError("vectorised operation A < B is type-ambiguous")

    def __ge__(self, other):
        raise NotImplementedError("vectorised operation A >= B is type-ambiguous")

    def __gt__(self, other):
        raise NotImplementedError("vectorised operation A > B is type-ambiguous")

    def _setitem_key(self, key, value):
        """ private method for Column[int] = value """
        assert isinstance(key, int)
        key = self._len + key if key < 0 else key  # deal with negative index.

        pages = mem.get_pages(self.group)
        ix, start, _, page = pages.get_page_by_index(key)
        if mem.get_ref_count(page) == 1:
            page[key - start] = value
        else:
            data = page[:].tolist()
            data[key - start] = value
            new_page = Page(data)
            new_pages = pages[:]
            new_pages[ix] = new_page
            self._len = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)

    def _setitem_values(self, key, value):
        start, stop, step = key.indices(self._len)
        if key.start is None and key.stop is None and key.step in (None, 1):
            # documentation: new = list(value)
            # example: L[:] = [1,2,3]
            pass
            # drop old pages
            arrays = paginate(value)

            # set new pages.



        elif key.start is not None and key.stop is None and key.step is None:
            # documentation: new = old[:key.start] + list(value)
            # example: L[0:] = [1,2,3]
            pass
        elif key.stop is not None and key.start is None and key.step is None:
            # documentation: new = list(value) + old[key.stop:]
            # example: L[:3] = [1,2,3]
            pass
        elif key.step is None and key.start is not None and key.stop is not None:  # L[3:5] = [1,2,3]
            # documentation: new = old[:start] + list(values) + old[stop:]
            pass
        elif key.step is not None:
            pass
        else:
            raise KeyError(f"bad key: {key}")

    def _setitem_column(self, keys, values):
        pass


def get(path_to_page):
    """gets data from page

    remember:
    - simple data is stored in 1 page.
    - complex data is stored in 2 pages
    """
    pass


def paginate(values, page_size=config.PAGE_SIZE):
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
