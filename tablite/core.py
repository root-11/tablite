import os
import pathlib
import json
import zipfile
import operator
import warnings
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

from tablite.memory_manager import MemoryManager, Page

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
        for key in self.columns:
            del self[key]
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
                mem.create_column_reference(self.key, column_name=keys, column_key=column.key)
            elif isinstance(values, Column):
                col = self._columns.get(keys,None)
                if col is None:  # it's a column from another table.
                    self._columns[keys] = column = values.copy()
                    mem.create_column_reference(self.key, column_name=keys, column_key=column.key)
                elif values.key == col.key:  # it's update from += or similar
                    self._columns[keys] = values
                else:                    
                    raise NotImplemented()
            else:
                raise NotImplemented()
            
        else:
            raise NotImplemented()
    
    def __getitem__(self,keys):
        if isinstance(keys,str) and keys in self._columns:
            return self._columns[keys]

    def __delitem__(self, key):
        if isinstance(key, str) and key in self._columns:
            col = self._columns[key]
            mem.delete_column_reference(self.group, key, col.key)
            del self._columns[key]  # dereference the Column
        else:
            raise NotImplemented()()

    def __add__(self,other):
        raise NotImplemented()

    def __iadd__(self,other):
        raise NotImplemented()

    @classmethod
    def reload_saved_tables(cls,path=None):
        tables = []
        if path is None:
            path = mem.path
        unsaved = 0
        with h5py.File(path, 'r') as h5:
            if "/table" not in h5.keys():
                return []

            for table_key in h5["/table"].keys():
                dset = h5[f"/table/{table_key}"]
                if dset.attrs['saved'] is False:
                    unsaved += 1
                else:
                    t = Table.load(path, key=table_key)
                    tables.append(t)
        warnings.warn(f"Dropping {unsaved} tables from cache where save==False.")
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

    def add_rows(self, *args, **kwargs):
        """ its more efficient to add many rows at once. """
        raise NotImplementedError

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
    
    def __getitem__(self, item=None):
        if isinstance(item, int):
            slc = slice(item,item+1,1)
        if item is None:
            slc = slice(0,None,1)
        if isinstance(item, slice):
            slc = item
        if not isinstance(slc, slice):
            raise TypeError(f"expected slice or int, got {type(item)}")
        
        result = mem.get_data(self.group, slc)

        if isinstance(item, int) and len(result)==1:
            return result[0]
        else:
            return result

    def append(self, value):
        pages = mem.get_pages(self.group)
        all_pages = pages[:]
        if pages:
            last_page = pages[-1]
            if mem.get_ref_counts(last_page) == 1: 
                data = np.array([value])
                target_cls = last_page.page_class_type_from_np(data)
                if isinstance(last_page, target_cls):
                    last_page.append(data)
                else:  # new datatype for ref_count == 1, so we create a mixed type page.
                    data = np.array(last_page[:] + [value])
                    new_page = last_page.create(data)
                    all_pages[-1] = new_page
            else:
                new_page = last_page.create(data)
                all_pages.append(new_page)
        else:
            new_page = last_page.create(data)
            all_pages.append(new_page)
        
        shape = mem.create_virtual_dataset(self.group, old_pages=pages, new_pages=all_pages)
        self._len = shape
        
    def insert(self, index, value):
        old_pages = mem.get_pages(self.group)
        page = old_pages.get_page_by_index(index)
        if mem.get_ref_counts(page) == 1:
            data = np.array([value])
            target_cls = page.page_class_type_from_np(data)
            if isinstance(page, target_cls):
                page.insert(index, value)
                new_page = page
            else:
                data = np.array(page[:] + [value])
                new_page = page.create(data)
        else:
            data = np.array(page[:] + [value])
            new_page = page.create(data)

        # old_pages = mem.get_pages(self.group)
        ix = old_pages.index(page)
        new_pages = old_pages[:]
        new_pages[ix] = new_page
        
        shape = mem.create_virtual_dataset(self.group, old_pages=old_pages, new_pages=new_pages)
        self._len = shape
    
    def _extend_from_column(self, column):
        """ internal API for Column.extend(values) where values is of type Column """
        if not isinstance(column, Column):
            raise TypeError
        pages = mem.get_pages(self.group)
        other = mem.get_pages(column.group)
        all_pages = pages.extend(other)
        shape = mem.create_virtual_dataset(self.group, old_pages=pages, new_pages=all_pages)
        self._len = shape

    def _extend_from_values(self, values):
        """ internal API for Column.extend(values) where values is of type list, tuple or np.ndarray """
        if not isinstance(values, (list,tuple, np.ndarray)):
            raise TypeError
        if isinstance(values, np.ndarray):
            data = values
        else:
            data = np.array(values)

        pages = mem.get_pages(self.group)
        all_pages = pages[:]
        if pages:
            last_page = pages[-1]
            if mem.get_ref_counts(last_page) == 1: 
                
                target_cls = last_page.page_class_type_from_np(data)
                if isinstance(last_page, target_cls):
                    last_page.extend(data)
                else:  # new datatype for ref_count == 1, so we create a mixed type page.
                    data = np.array(last_page[:] + data)
                    new_page = last_page.create(data)
                    all_pages[-1] = new_page
            else:
                new_page = last_page.create(data)
                all_pages.append(new_page)
        else:
            new_page = Page.create(data)
            all_pages.append(new_page)
        
        shape = mem.create_virtual_dataset(self.group, old_pages=pages, new_pages=all_pages)
        self._len = shape

    def extend(self, values):
        if isinstance(values, Column):
            self._extend_from_column(values)
        elif isinstance(values, (list,tuple, np.ndarray)):
            self._extend_from_values(values)
        else:
            raise TypeError(f'Column cannot extend using {type(values)}')
        
    def remove(self, value):
        pages = mem.get_pages(self.group)
        for ix, page in enumerate(pages):
            if value not in page[:]:
                continue
            if mem.get_ref_count(page) == 1:
                page.remove(value)
                new_pages = pages[:]
            else:
                data = page[:]
                data = data.aslist()
                data.remove(value)
                new_page = page.create(data)
                new_pages = pages[:]
                new_pages[ix] = new_page
            shape = mem.create_virtual_dataset(self.group, old_pages=pages, new_pages=new_pages)
            self._len = shape
            break

    def remove_all(self, value):
        pages = mem.get_pages(self.group)
        new_pages = pages[:]
        for ix, page in enumerate(pages):
            if value not in page[:]:
                continue
            new_data = [v for v in page[:] if v != value]
            new_page = page.create(new_data)
            new_pages[ix] = new_page
        shape = mem.create_virtual_dataset(self.group, old_pages=pages, new_pages=new_pages)
        self._len = shape
        
    def pop(self, index):
        index = self._len + index if index < 0 else index

        pages = mem.get_pages(self.group)
        a,b = 0,0
        for ix, page in enumerate(pages):
            a = b
            b += len(page)
            if a <= index < b:
                if mem.get_ref_count(page) == 1:
                    value = page.pop(index-a)
                else:
                    data = page[:]
                    value = data.pop(index-a)
                    new_page = page.create(data)
                    new_pages = pages[:]
                    new_pages[ix] = new_page
                shape = mem.create_virtual_dataset(self.group, old_pages=pages, new_pages=new_pages)
                self._len = shape
                return value
        raise IndexError(f"index {index} out of bound")

    def _update_by_index(self, key, value):
        assert isinstance(key, int)
        

    def _update_by_slice(self, key, value):
        key_len = len(range(*key.indices(self._len)))
        value_len = len(value)
        if key.start == self._len and key.stop is None:
            pass  # just extend.
        elif key.start is None and key.stop == 0:
            pass  # insert a page up front.
        elif key_len == value_len: 
            pass  # its a 1-2-1 value replacement.
        elif key_len > value_len:
            pass  # it's a reduction
        elif key_len < value_len:
            pass  # it's an extension.
        else:
            raise NotImplementedError(f"{key}, {value} on {self._len}")

    def _update_by_column(self, key, value):
        key_len = len(range(*key.indices(self._len)))
        value_len = len(value)
        if key.start == self._len and key.stop is None:
            pass  # just extend.
        elif key.start is None and key.stop == 0:
            pass  # insert a page up front.
        elif key_len == value_len: 
            pass  # its a 1-2-1 value replacement.
        elif key_len > value_len:
            pass  # it's a reduction
        elif key_len < value_len:
            pass  # it's an extension.
        else:
            raise NotImplementedError(f"{key}, {value} on {self._len}")
        

    def __setitem__(self, key, value):
        """
        Column.__setitem__(key,value) delegates the operations to ---> :
        >>> L = [0, 10, 20, 3, 4, 5, 100]  # ---> create (__init__)
        >>> L[3] = 30                      # ---> update 
        [0, 10, 20, 30, 4, 5, 100]
        >>> L[4:5] = [40,50]               # ---> update many as slice has same length as values
        [0, 10, 20, 30, 40, 50, 5, 100]
        >>> L[-2:-1] = [60,70,80,90]                      # ---> 1 x update + insert
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        >>> L[len(L):] = [110]                            # ---> append
        [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        >>> del L[:3]                                     # ---> delete
        [30, 40, 50, 60, 70, 80, 90, 100, 110]
        """
        if isinstance(key, int):
            if abs(key) > self._len:
                raise IndexError("IndexError: list index out of range")
            if isinstance(value, (list,tuple)):
                raise TypeError(f"did you mean to insert? F.x. [{key}:{key+1}] = {value} ?")
            self._update_by_index(key,value)

        elif isinstance(key, slice):
            if isinstance(value, (list,tuple,np.ndarray)):
                self._update_by_slice(key,value)
            elif isinstance(value, Column):
                self._update_by_column(key,value)
            else:
                raise TypeError(f"No method for {type(value)}")
        else:
            raise TypeError(f"no method for key of type: {type(key)}")

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

    def __copy__(self):
        return self.copy()
    
    def index(self):
        data = self.__getitem__()
        d = {k:[] for k in np.unique(data)}  
        for ix,k in enumerate(data):
            d[k].append(ix)
        return d

    def unique(self):  
        return np.unique(self.__getitem__())

    def histogram(self):
        """ 
        returns 2 arrays: unique elements and count of each element 
        
        example:
        >>> for item, counts in zip(self.histogram()):
        >>>     print(item,counts)
        """
        uarray, carray = np.unique(self.__getitem__(), return_counts=True)
        return uarray, carray

    def __add__(self, other):
        c = self.copy()
        c.extend(other)
        return c
    
    def __contains__(self, item):
        return item in self.__getitem__()
    
    def __iadd__(self, other):
        self.extend(other)
        return self
    
    def __imul__(self, other):
        if not isinstance(other, int):
            raise TypeError(f"a column can be repeated an integer number of times, not {type(other)} number of times")
        for _ in range(other):
            self.extend(self)
        return self
    
    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError(f"a column can be repeated an integer number of times, not {type(other)} number of times")
        c = self.copy()
        for _ in range(1, other):
            c.extend(self)
        return c
    
    def __ne__(self, other):
        if len(self) != len(other):  # quick cheap check.
            return False
        if not isinstance(other, np.ndarray):
            other = np.array(other)
        return (self.__getitem__()!=other).any()  # speedy np c level comparison.
    
    def __le__(self,other):
        raise NotImplemented()
    
    def __lt__(self,other):
        raise NotImplemented()
    
    def __ge__(self,other):
        raise NotImplemented()

    def __gt__(self,other):
        raise NotImplemented()

