import pathlib
from itertools import count
from collections import defaultdict
import json
import numpy as np
import h5py
from tablite2.settings import HDF5_COLUMN_ROOT,HDF5_PAGE_ROOT, HDF5_CACHE_DIR, HDF5_CACHE_FILE, HDF5_PAGE_SIZE

# h5py modes:
# r	 Readonly, file must exist (default)
# r+ Read/write, file must exist
# w	 Create file, truncate if exists
# w- or x	Create file, fail if exists
# a	 Read/write if exists, create otherwise
    

class Column(object):   # HDF5 virtual dataset.  I need columns to make table copy & inheritance work effectively.
    ref_count = {}  # used to determine if the dataset of a page is mutuable.
    page_ids = count(1)
    column_ids = count(1)

    def __init__(self, data=None, pytype=None,key=None) -> None:        
        self.path = pathlib.Path(HDF5_CACHE_DIR) / HDF5_CACHE_FILE
        self.cid = next(Column.column_ids) if key is None else key
        self.group = f"{HDF5_COLUMN_ROOT}/column-{self.cid}"

        if key is not None:
            assert data is None, "key and data? No."
            with h5py.File(self.path, mode='r') as h5:
                if key in h5[HDF5_COLUMN_ROOT]:
                    dset = h5[self.group]
                    self._len = dset.len
                    self.pytype = dset.dtype
        else:
            self._len = 0
            self.pytype = pytype
            if data is not None:
                self.extend(data)

    @property
    def pages(self):
        with h5py.File(name=self.path,mode='r') as h5:
            dset = h5[self.group]
            pages = [(fname,grp) for _,fname,grp,_ in dset.virtual_sources()]  # https://docs.h5py.org/en/stable/high/dataset.html#h5py.Dataset.virtual_sources
            return pages   

    def __len__(self):
        with h5py.File(name=self.path,mode='r') as h5:
            dset = h5[self.group]
            return dset.size
    def __eq__(self, other) -> bool:
        return any(a!=b for a,b in zip(self,other))

    def __setitem__(self, key, value):
        """
        col[ix] == value
        col[slice] == ndarray
        """
        pass  # https://docs.h5py.org/en/stable/high/dataset.html#h5py.Dataset.resize
    def __getitem__(self, key):
        """
        col[int] --> value
        col[slice] --> ndarray
        """
        with h5py.File(name=self.path, mode='r') as h5:
            return h5[self.group][key]  # TODO check for datatype.
            # https://docs.h5py.org/en/stable/high/dataset.html#h5py.Dataset.astype

    def __delitem__(self, key):
        # if ref count == 0, delete the dataset.
        raise NotImplementedError
    
    def clear(self):
        with h5py.File(self.path, 'a') as h5:
            dset = h5[self.group]
            old_pages = [(path,group) for _,path,group,_ in dset.virtual_sources()]

            for page in old_pages:
                path,group = page   # not sure this will work with imports...?
                self.ref_count[page] -= 1
                if self.ref_count[page]==0:
                    del h5[group]

    def append(self, value):
        raise AttributeError("no point. Use extend instead.")  # extend  last page if ref_count ==1

    @classmethod
    def _add(cls, path, data, encoding='utf-8'):
        """
        Adds new data. type checks the data.
        """
        if not isinstance(data,np.ndarray):
            data = np.array(data)
        # type check
        if data.dtype == 'O':  # datatype was non-native to HDF5, so utf-8 encoded bytes must used
            uarray, carray = np.unique(data, return_counts=True)
            dtypes = {type(v):c for v,c in zip(uarray,carray)}
            data = [str(v, encoding=encoding) for v in data]
        else:
            dtypes = {data.dtype: len(data)}
        # store.
        page_id = next(cls.page_ids)
        group = f"{HDF5_PAGE_ROOT}/page-{page_id}"
        with h5py.File(path, 'a') as h5:
            dset = h5.create_dataset(name=group, data=data, maxshape=(None,), chunks=HDF5_PAGE_SIZE)
            dset.attrs['datatype'] = json.dumps(dtypes)
            dset.attrs['encoding'] = encoding
        return path,group

    def extend(self, values):  # recreate virtual dataset
        if isinstance(values, (tuple, list, np.ndarray)):
            # all original data is stored as an individual dataset.
            new_pages = [ self._add(self.path, values) ]
        if isinstance(values, Column):
            new_pages = values.pages  # list
        else:
            raise TypeError(values)

        with h5py.File(self.path, 'a') as h5:
            dset = h5[self.group]
            old_pages = [(path,group) for _,path,group,_ in dset.virtual_sources()]
            
            pages = old_pages + new_pages
            for name in pages:  # add ref count for new connection.
                self.ref_count[name]+=1
            for name in old_pages:  # remove duplicate ref count.
                self.ref_count[name]-=1
            # above: add first, then remove prevents ref count < 1.

            dtypes = {}
            for dn in pages:
                dset = h5[dn]
                dtypes[dset.dtype] += dset.len
            shape = sum(dtypes.values())
            L = [x for x in dtypes]
            L.sort(key=lambda x: x.itemsize, reverse=True)
            dtype = L[0]  # np.bytes

            layout = h5py.VirtualLayout(shape=(shape,), dtypes=dtype, maxshape=(None,))
            a, b = 0, 0
            for group in pages:
                dset = h5[group]
                b += dset.len
                vsource = h5py.VirtualSource(dset)
                layout[a:b] = vsource
                a = b
            h5.create_virtual_dataset(self.group, layout=layout)

    def __iter__(self):
        return self

    def __next__(self):
        with h5py.File(name=self.path, mode='r') as h5:
            for v in h5[self.group]:
                yield v  # TODO check for datatype.
    
    def copy(self):
        return Column(data=self)

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

