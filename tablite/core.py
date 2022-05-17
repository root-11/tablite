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
from tablite.utils import intercept

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
        generators = [iter(mc) for mc in self._columns.values()]
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
        else:
            raise KeyError(f"no such column: {keys}")

    def __delitem__(self, key):
        if isinstance(key, str) and key in self._columns:
            col = self._columns[key]
            mem.delete_column_reference(self.group, key, col.key)
            del self._columns[key]  # dereference the Column
        else:
            raise NotImplemented()()

    def copy(self):
        t = Table()
        for name, col in self._columns.items():
            t[name] = col
        return t

    def clear(self):
        for name in self.columns:
            self.__delitem__(name)

    def __add__(self,other):
        raise NotImplemented()

    def __iadd__(self,other):
        if not isinstance(other, Table):
            raise TypeError(f"no method for {type(other)}")
        if set(self.columns) != set(other.columns) or len(self.columns) != len(other.columns):
            raise ValueError("Columns names are not the same")
        for name,col in self._columns.items():
            other_col = other[name]
            col += other_col
        return self

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
                c = Column.load(key=column_key)
                ds2 = h5[f"/column/{column_key}"]
                c._len = ds2.len()
                t._columns[col_name] = c
            return t

    @classmethod
    def reset_storage(cls):
        mem.reset_storage()

    def add_rows(self, *args, **kwargs):
        """ its more efficient to add many rows at once. """
        raise NotImplementedError()

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

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def load(cls, key):
        return Column(key=key)
    
    def __iter__(self):
        return (v for v in self.__getitem__())

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
        data = np.array([value])
        self.__setitem__(key=slice(self._len,None,None), value=data)
        
    def insert(self, index, value):
        if isinstance(value, (list, tuple)):
            new_data = np.array(value)
        elif isinstance(value, np.ndarray):
            new_data = value
        else:
            new_data = np.array([value])

        target_cls = page.page_class_type_from_np(new_data)
        
        old_pages = mem.get_pages(self.group)
        new_pages = old_pages[:]

        ix, start, _, page = old_pages.get_page_by_index(index)

        if mem.get_ref_count(page) == 1 and isinstance(page, target_cls):
            new_page = page  # ref count and target class match. Now let the page class do the insert.
        else:
            new_page = Page.create(page[:])  # copy the existing page so insert can be done below

        new_page.insert(index - start, new_data)
        new_pages[ix] = new_page  # insert the changed page.
        
        shape = mem.create_virtual_dataset(self.group, pages_before=old_pages, pages_after=new_pages)
        self._len = shape
    
    def extend(self, values):
        self.__setitem__(slice(self._len,None,None), values)  #  self._extend_from_column(values)
        
    def remove(self, value):
        """ see also remove_all """
        pages = mem.get_pages(self.group)
        for ix, page in enumerate(pages):
            if value not in page[:]:
                continue
            if mem.get_ref_count(page) == 1:
                page.remove(value)
                new_pages = pages[:]
            else:
                data = page[:]  # copy the data.
                data = data.aslist()  
                data.remove(value)  # remove from the copy.
                new_page = page.create(data)  # create new page from copy
                new_pages = pages[:] 
                new_pages[ix] = new_page  # register the newly copied page.
            self._len = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)
            return  
        raise ValueError(f"value not found: {value}")

    def remove_all(self, value):
        """ see also remove """
        pages = mem.get_pages(self.group)
        new_pages = pages[:]
        for ix, page in enumerate(pages):
            if value not in page[:]:
                continue
            new_data = [v for v in page[:] if v != value]
            new_page = page.create(new_data)
            new_pages[ix] = new_page
        self._len = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)
        
    def pop(self, index):
        index = self._len + index if index < 0 else index
        if index > self._len:
            raise IndexError(f"can't reach index {index} when length is {self._len}")

        pages = mem.get_pages(self.group)
        ix,start,_, page = pages.get_page_by_index(index)
        if mem.get_ref_count(page) == 1:
            value = page.pop(index-start)
        else:
            data = page[:]
            value = data.pop(index-start)
            new_page = page.create(data)
            new_pages = pages[:]
            new_pages[ix] = new_page
        shape = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)
        self._len = shape
        return value

    def __setitem__(self, key, value):
        """
        Column.__setitem__(key,value) behaves just like a list
        """
        if isinstance(key, int):
            if isinstance(value, (list,tuple)):
                raise TypeError(f"your key is an integer, but your value is a {type(value)}. Did you mean to insert? F.x. [{key}:{key+1}] = {value} ?")
            if -self._len-1 < key < self._len:
                # self._update_by_index(key,value)
                data = np.array([value])
                target_cls = Page.page_class_type_from_np(data)
                pages = mem.get_pages(self.group)
                ix,start,_,page = pages.get_page_by_index(key)
                if mem.get_ref_count(page)==1 and isinstance(page, target_cls):
                    page[key-start] = data
                else:
                    data = page[:].tolist() 
                    data[key-start] = value
                    data = np.array(data)
                    new_page = Page.create(data)
                    new_pages = pages[:]
                    new_pages[ix] = new_page
                    self._len = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)
            else:
                raise IndexError("list assignment index out of range")

        elif isinstance(key, slice):
            start,stop,step = key.indices(self._len)
            if key.start == key.stop == None and key.step in (None,1):   # L[:] = [1,2,3]
                # self.items = list(value) -- kept as documentation reference for test_slice_rules.py | MyList
                if isinstance(value, Column):
                    before = mem.get_pages(self.group)
                    after = mem.get_pages(value.group)
                elif isinstance(value, (list,tuple,np.ndarray)):
                    data = np.array(value)
                    new_page = Page.create(data)
                    after = [new_page]
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.start != None and key.stop == key.step == None:   # L[0:] = [1,2,3]
                # self.items = self.items[:key.start] + list(value)  -- kept as documentation reference for test_slice_rules.py | MyList
                # self.items = self._getslice_(0,start) + list(value)  
                before = mem.get_pages(self.group) 
                before_slice = before.getslice(0,start)
                if isinstance(value, Column):
                    after = before_slice + mem.get_pages(value.group)
                elif isinstance(value, (list,tuple,np.ndarray)):
                    data = np.array(value)
                    
                    last_page = before_slice[-1]
                    target_cls = Page.page_class_type_from_np(data)  
                    if mem.get_ref_count(last_page) == 1:
                        if isinstance(last_page, target_cls):
                            last_page.extend(data)
                            after = before_slice
                        else:  # new datatype for ref_count == 1, so we create a mixed type page to avoid thousands of small pages.
                            data = np.array(last_page[:] + data)
                            new_page = Page.create(data)
                            after = before_slice[:-1] + [new_page]
                    else:  # ref count > 1
                        new_page = Page.create(data)
                        after = before_slice + [new_page]
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.stop != None and key.start == key.step == None:  # L[:3] = [1,2,3]
                # self.items = list(value) + self.items[key.stop:]  -- kept as documentation reference for test_slice_rules.py | MyList
                # self.items = list(value) + self._getslice_(stop, len(self.items))
                before = mem.get_pages(self.group)
                before_slice = before.getslice(stop, self._len)
                if isinstance(value, Column):
                    after = mem.get_pages(value.group) + before_slice
                elif isinstance(value, (list,tuple, np.ndarray)):
                    data = np.array(value)
                    new_page = Page.create(data)
                    after = [new_page] + before_slice
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
                
            elif key.step == None and key.start != None and key.stop != None:  # L[3:5] = [1,2,3]
                stop = max(start,stop)
                # -- kept as documentation reference for test_slice_rules.py | MyList
                # self.items = self.items[:start] + list(value) + self.items[stop:]
                # self.items = self._getslice_(0,start) + list(value) + self._getslice_(stop,len(self.items))
                before = mem.get_pages(self.group)
                A,B = before.getslice(0,start), before.getslice(stop, self._len)
                if isinstance(value, Column):
                    after = A + mem.get_pages(value.group) + B
                elif isinstance(value, (list,tuple, np.ndarray)):
                    data = np.array(value)
                    new_page = Page.create(data)
                    after = A + [new_page] + B
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.step != None:
                seq = range(start,stop,step)
                seq_size = len(seq)
                if len(value) > seq_size:
                    raise ValueError(f"attempt to assign sequence of size {len(value)} to extended slice of size {seq_size}")
                
                # -- kept as documentation reference for test_slice_rules.py | MyList
                # new = self.items[:]  # cheap shallow pointer copy in case anything goes wrong.
                # for new_index, position in zip(range(len(value)),seq):
                #     new[position] = value[new_index]
                # # all went well. No exceptions.
                # self.items = new

                before = mem.get_pages(self.group)
                new = mem.get_data(self.group)  
                for new_index, position in zip(range(len(value)), seq):
                    new[position] = value[new_index]
                # all went well. No exceptions.
                after = [Page.create(new)]  # This may seem redundant, but is in fact is good as the user may 
                # be cleaning up the dataset, so that we end up with a simple datatype instead of mixed.
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
        else:
            raise TypeError(f"bad key: {key}")

    def __len__(self):
        return self._len

    def __eq__(self, other):
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

