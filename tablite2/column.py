import pathlib
from itertools import count
from collections import defaultdict
import numpy as np
import h5py
from tablite2.settings import HDF5_IMPORT_ROOT, HDF5_CACHE_DIR, HDF5_CACHE_FILE, HDF5_PAGE_SIZE


class Page(object):  # HDF5 dataset.   I need pages to make multiprocessing work effectively.
    pids = count(1)
    def __init__(self, data, encoding='ascii') -> None:
        self.pid = next(Page.pids)
        self.path = pathlib.Path(HDF5_CACHE_DIR) / HDF5_CACHE_FILE
        self.group = f"{HDF5_IMPORT_ROOT}/page-{self.pid}"
        self.encoding = encoding
        self.links = set()
    def link(self, other):
        if not isinstance(other, Column):
            raise TypeError
        self.links.add(other.cid)
    def unlink(self, other):
        if not isinstance(other, Column):
            raise TypeError
        self.links.remove(other.cid)
    def __setitem__(self,keys,values):
        if len(self.links) > 1:
            raise AttributeError("can setitem. immutable.")
    def __getitem__(self,key):
        if isinstance(key, int):
            pass
        elif isinstance(key, slice):
            pass
        else:
            raise TypeError(key)


class Column(object):   # HDF5 virtual dataset.  I need columns to make table copy & inheritance work effectively.
    cids = count(1)
    def __init__(self, data=None, pytype=None) -> None:
        self.cid = next(Column.cids)
        self.path = pathlib.Path(HDF5_CACHE_DIR) / HDF5_CACHE_FILE
        self.group = f"{HDF5_IMPORT_ROOT}/column-{self.cid}"

        if data is None:  # wait until .extend is used.
            pass
        if isinstance(data, Column):  # just use the source route.
            pass
        elif isinstance(data, np.ndarray):  # break it into pages.
            HDF5_PAGE_SIZE
            pass
        else:
            pass
        self.pytype = pytype
        if data.dtype != pytype:
            pass
        self._len = None

    @property
    def address(self):
        return (self.path,self.group)
    def __len__(self):
        pass
    def __eq__(self, other) -> bool:
        return any(a!=b for a,b in zip(self,other))
    def __setitem__(self, key,value):
        """
        col[ix] == value
        col[slice] == ndarray
        """
        pass
    def __getitem__(self, key):
        """
        col[int] --> value
        col[slice] --> ndarray
        """
        with h5py.File(name=self.path, mode='r') as h5:
            return h5[self.group][key]  # TODO check for datatype.
    def append(self, value):
        raise AttributeError("no point. Use extend instead.")
    def extend(self, values):  # recreate virtual dataset.
        pass
    def index(self):
        with h5py.File(name=self.path, mode='r') as h5:
            d = {k:[] for k in np.unique(h5[self.group])}
            for ix,k in enumerate(h5[self.group]):
                d[k].append(ix)
        return d
    def unique(self):  # as self is a virtual dataset, this is easy
        with h5py.File(name=self.path, mode='r') as h5:
            return np.unique(h5[self.group])
    def histogram(self):  # it may be faster to do this in parallel, but for now the brute-force approach below works.
        with h5py.File(name=self.path, mode='r') as h5:
            uarray, carray = np.unique(h5[self.group], return_counts=True)
        return uarray, carray
    def __iter__(self):
        return self
    def __next__(self):
        with h5py.File(name=self.path, mode='r') as h5:
            for v in h5[self.group]:
                yield v  # TODO check for datatype.
    
