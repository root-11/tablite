import os
import pathlib
import json
import zipfile
import operator

import logging
logging.getLogger('lml').propagate = False
logging.getLogger('pyexcel_io').propagate = False
logging.getLogger('pyexcel').propagate = False

import pyexcel
import pyperclip
from tqdm import tqdm
import numpy as np
import h5py

from collections import defaultdict
from itertools import count, chain

# from tablite.datatypes import DataTypes
# from tablite.file_reader_utils import detect_encoding, detect_seperator, split_by_sequence, text_escape
# from tablite.groupby_utils import Max, Min, Sum, First, Last, Count, CountUnique, Average, StandardDeviation, Median, Mode, GroupbyFunction

# from tablite.columns import StoredColumn, InMemoryColumn
# from tablite.stored_list import tempfile

from tablite.memory_manager import MemoryManager

mem = MemoryManager()

class Table(object):
    ids = count(1)
    def __init__(self,key=None, save=False, _create=True) -> None:
        self.key = next(Table.ids) if key is None else key
        self.group = f"/table/{self.key}"
        self._columns = {}  # references for virtual datasets that behave like lists.
        if _create:
            mem.create_table(self.group, save)  # attrs. 'columns'
        self._saved = save
    
    @property
    def save(self):
        return self._saved

    @save.setter
    def save(self, value):
        if not isinstance(value, bool):
            raise TypeError(f'expected bool, got: {type(value)}')
        if self._saved != value:
            self._saved = value
            mem.set_saved_flag(self.group, value)

    def __del__(self):
        mem.delete_table(self.group)

    @property
    def columns(self):
        return list(self._columns.keys())

    @property
    def rows(self):
        """
        enables iteration

        for row in Table.rows:
            print(row)
        """
        generators = [iter(mc) for mc in self.columns.values()]
        for _ in range(len(self)):
            yield [next(i) for i in generators]

    def __len__(self):
        for v in self._columns.values():  
            return len(v)  # return on first key.
        return 0  # if there are no columns.

    def __setitem__(self, keys, values):
        if isinstance(keys, str): 
            if isinstance(values, (tuple,list)):
                self._columns[keys] = column = Column(values)  # overwrite if exists.
            elif isinstance(values, Column):
                self._columns[keys] = column = values.copy()
            else:
                raise NotImplemented
            mem.create_column_reference(self.key, column_name=keys, column_key=column.key)
        else:
            raise NotImplemented
    
    def __getitem__(self,keys):
        if isinstance(keys,str) and keys in self.columns:
            return self._columns[keys]

    def __delitem__(self, key):
        if isinstance(key, str) and key in self._columns:
            col = self._columns[key]
            mem.delete_column_reference(self.group, key, col.key)
            del self._columns[key]  # dereference the Column
        else:
            raise NotImplemented

    @classmethod
    def reload_saved_tables(cls,path=None):
        tables = []
        if path is None:
            path = mem.path
        with h5py.File(path, 'r') as h5:
            for table_key in h5["/table"].keys():
                t = Table.load(path, key=table_key)
                
                tables.append(t)
        return tables

    @classmethod
    def load(cls, path, key):
        with h5py.File(path, 'r') as h5:
            t = Table(key, _create=False)
            dset = h5[f"/table/{key}"]
            columns = json.loads(dset.attrs['columns'])
            for col_name, column_key in columns.items():
                t._columns[col_name] = Column.load(key=column_key)
            return t

    @classmethod
    def reset_storage(cls):
        mem.reset_storage()


class Column(object):
    ids = count(1)
    def __init__(self, data=None, key=None) -> None:
        self.key = f"{next(self.ids)}" if key is None else key
        self.group = f"/column/{self.key}"
        self._len = 0
        if data:
            self.extend(data)
    
    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>({self._len} | {self.key})"

    @classmethod
    def load(cls, key):
        return Column(key=key)
    
    def extend(self, data):
        if isinstance(data, (tuple, list, np.ndarray)):  # all original data is stored as an individual dataset.
            group = mem.create_page(data=data)
            new_pages = [ group ]
        elif isinstance(data, Column):
            new_pages = mem.get_pages(group=data.group)  # list of pages in hdf5.
        else:
            raise TypeError(data)

        shape = mem.create_virtual_dataset(self.group, new_pages)
        assert isinstance(shape,int) 
        self._len = shape

    def __getitem__(self, item=None):
        if item is None:
            item = slice(0,None,1)
        return mem.get_data(self.group, item)

    def __len__(self):
        return self._len

    def __eq__(self,other):
        if isinstance(other, (list,tuple)):
            B = np.array(other)
        elif isinstance(other, np.ndarray): 
            B = other
        elif isinstance(other, Column):
            B = mem.pages(other.group)
            A = mem.pages(self.group)
            if A == B:
                return True  # special case.
        else:
            raise TypeError
        A = self.__getitem__()
        return (A==B).all()

    def copy(self):
        c = Column()
        c.extend(self)
        return c
    
    def index(self):
        data = self.__getitem__()
        d = {k:[] for k in np.unique(data)}  
        for ix,k in enumerate(data):
            d[k].append(ix)
        return d

    def unique(self):  
        return np.unique(self.__getitem__())

    def histogram(self):  
        uarray, carray = np.unique(self.__getitem__(), return_counts=True)
        return uarray, carray
