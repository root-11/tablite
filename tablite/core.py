import os
import io
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
        elif isinstance(keys, tuple) and len(keys) == len(values):
            for key, value in zip(keys,values):
                self.__setitem__(key,value)
        else:
            raise NotImplemented()
    
    def __getitem__(self,keys):
        if isinstance(keys,str) and keys in self._columns:
            return self._columns[keys]
        else:
            raise KeyError(f"no such column: {keys}")

    def __delitem__(self, key):
        """
        del table['a']  removes column 'a'
        del table[-3:] removes last 3 rows from all columns.
        """
        if isinstance(key, str) and key in self._columns:
            col = self._columns[key]
            mem.delete_column_reference(self.group, key, col.key)
            del self._columns[key]  # dereference the Column
        elif isinstance(key, slice):
            for col in self._columns.values():
                del col[key]
        else:
            raise NotImplemented()

    def copy(self):
        t = Table()
        for name, col in self._columns.items():
            t[name] = col
        return t

    def clear(self):
        for name in self.columns:
            self.__delitem__(name)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Table):
            return False
        if id(self) == id(__o):
            return True
        if len(self) != len(__o):
            return False
        if self.columns != __o.columns:
            return False
        for name, col in self._columns.items():
            if col != __o._columns[name]:
                return False
        return True

    def __add__(self,other):
        """
        enables concatenation for tables with the same column names.
        """
        c = self.copy()
        c += other
        return c
 
    def __iadd__(self,other):
        """
        enables extension with other tables with the same column names.
        """
        if not isinstance(other, Table):
            raise TypeError(f"no method for {type(other)}")
        if set(self.columns) != set(other.columns) or len(self.columns) != len(other.columns):
            raise ValueError("Columns names are not the same")
        for name, col in self._columns.items():
            other_col = other[name]
            col += other_col
        return self

    def __mul__(self,other):
        """
        enables repetition of a table
        Example: Table_x_10 = table * 10
        """
        if not isinstance(other, int):
            raise TypeError(f"can't multiply Table with {type(other)}")
        t = self.copy()
        for _ in range(1,other):
            t+=self
        return t

    def __imul__(self,other):
        """
        extends a table N times onto using itself as source.
        """
        if not isinstance(other, int):
            raise TypeError(f"can't multiply Table with {type(other)}")
        c = self.copy()
        for _ in range(other):
            self += c
        return self

    @classmethod
    def reload_saved_tables(cls,path=None):
        """
        Loads saved tables from disk.
        
        The default storage locations is:
        >>> from tablite.config import HDF5_Config
        >>> print(Config.H5_STORAGE)
        """
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
        """ Resets all stored tables. """
        mem.reset_storage()

    def add_rows(self, *args, **kwargs):
        """ its more efficient to add many rows at once. 
        
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
            if all(isinstance(i, (list, tuple, dict)) for i in args):
                if all(len(i) == len(self._columns) for i in args):
                    for arg in args:
                        if isinstance(arg, (list,tuple)):  # 2,3,5,6
                            for col,value in zip(self._columns.values(), arg):
                                col.append(value)
                        elif isinstance(arg, dict):  # 7,8
                            for k,v in arg.items():
                                col = self._columns[k]
                                col.append(v)
                        else:
                            raise TypeError(f"{arg}?")
            elif len(args) == len(self._columns):  # 1,4
                for col, value in zip(self._columns.values(), args):
                    col.append(value)
            else:
                raise ValueError(f"format not recognised: {args}")

        if kwargs:
            if isinstance(kwargs, dict):
                if all(isinstance(v, (list, tuple)) for v in kwargs.values()):
                    for k,v in kwargs.items():
                        col = self._columns[k]
                        col.extend(v)
                else:
                    for k,v in kwargs.items():
                        col = self._columns[k]
                        col.append(v)
            else:
                raise ValueError(f"format not recognised: {kwargs}")
        
        return

    def add_columns(self, *names):
        for name in names:
            self.__setitem__(name,[])

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

        t = self.copy()
        for name , col2 in other._columns.items():
            if name not in t.columns:  # fill with blanks
                t[name] = [None] * len(self)
            col1 = t[name]
            col1.extend(col2)

        for name, col in t._columns.items():
            if name not in other.columns:
                col.extend([None]*len(other))
        return t

    def to_ascii(self, blanks=None):
        """
        enables viewing in terminals
        returns the table as ascii string
        """
        widths = {}
        names = list(self.columns)
        if not names:
            return "Empty table"
        for name,col in self._columns.items():
            widths[name] = max([len(name)] + [len(str(v)) for v in col])

        def adjust(v, length):
            if v is None:
                return str(blanks).ljust(length)
            elif isinstance(v, str):
                return v.ljust(length)
            else:
                return str(v).rjust(length)

        s = []
        s.append("+" + "+".join(["=" * widths[n] for n in names]) + "+")
        s.append("|" + "|".join([n.center(widths[n], " ") for n in names]) + "|")
        # s.append("| " + "|".join([str(table.columns[n].dtype).center(widths[n], " ") for n in names]) + " |")
        s.append("+" + "+".join(["-" * widths[n] for n in names]) + "+")
        for row in self.rows:
            s.append("|" + "|".join([adjust(v, widths[n]) for v, n in zip(row, names)]) + "|")
        s.append("+" + "+".join(["=" * widths[h] for h in names]) + "+")
        return "\n".join(s)

    def show(self, *args, blanks=None):
        """
        accepted args:
          - slice

        """ 
        if args:
            for arg in args:
                if isinstance(arg, slice):
                    t = self[arg]
                    print(t.to_ascii(blanks))
                    
        elif len(self) < 20:
            print(self.to_ascii(blanks))
        else:
            t,n = Table(), len(self)
            t['#'] = [str(i) for i in range(7)] + ["..."] + [str(i) for i in range(n-7, n)]
            for name, col in self._columns.items():
                t[name] = [str(i) for i in col[:7]] + ["..."] + [str(i) for i in col[-7:]] 
            print(t.to_ascii(blanks))

    def index(self, *args):
        cols = []
        for arg in args:
            col = self._columns.get(arg, None)
            if col is not None:
                cols.append(col)
        if not cols:
            raise ValueError("no columns?")

        c = np.column_stack(cols)
        idx = defaultdict(set)
        for ix, key in enumerate(c):
            idx[tuple(key)].add(ix)
        return idx
    
    def copy_to_clipboard(self):
        """ copy data from a Table into clipboard. """
        try:
            s = ["\t".join([f"{name}" for name in self.columns])]
            for row in self.rows:
                s.append("\t".join((str(i) for i in row)))
            s = "\n".join(s)
            pyperclip.copy(s)
        except MemoryError:
            raise MemoryError("Cannot copy to clipboard. Select slice instead.")

    @staticmethod
    def copy_from_clipboard():
        """ copy data from clipboard into Table. """
        t = Table()
        txt = pyperclip.paste().split('\n')
        t.add_columns(*txt[0].split('\t'))

        for row in txt[1:]:
            data = row.split('\t')
            t.add_rows(data)    
        return t


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

    def clear(self):
        old_pages = mem.get_pages(self.group)
        self._len = mem.create_virtual_dataset(self.group, pages_before=old_pages, pages_after=[])

    def append(self, value):
        self.__setitem__(key=slice(self._len,None,None), value=[value])
        
    def insert(self, index, value):        
        old_pages = mem.get_pages(self.group)
        new_pages = old_pages[:]

        ix, start, _, page = old_pages.get_page_by_index(index)

        if mem.get_ref_count(page) == 1:
            new_page = page  # ref count match. Now let the page class do the insert.
            new_page.insert(index - start, value)
        else:
            data = page[:].tolist()
            data.insert(index-start,value)
            new_page = Page(data)  # copy the existing page so insert can be done below

        new_pages[ix] = new_page  # insert the changed page.
        self._len = mem.create_virtual_dataset(self.group, pages_before=old_pages, pages_after=new_pages)
    
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
                data = data.tolist()  
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
                key = self._len + key if key < 0 else key
                pages = mem.get_pages(self.group)
                ix,start,_,page = pages.get_page_by_index(key)
                if mem.get_ref_count(page) == 1:
                    page[key-start] = value
                else:
                    data = page[:].tolist()
                    data[key-start] = value
                    new_page = Page(data)
                    new_pages = pages[:]
                    new_pages[ix] = new_page
                    self._len = mem.create_virtual_dataset(self.group, pages_before=pages, pages_after=new_pages)
            else:
                raise IndexError("list assignment index out of range")

        elif isinstance(key, slice):
            start,stop,step = key.indices(self._len)
            if key.start == key.stop == None and key.step in (None,1): 
                # documentation: new = list(value)
                # example: L[:] = [1,2,3]
                before = mem.get_pages(self.group)
                if isinstance(value, Column):
                    after = mem.get_pages(value.group)
                elif isinstance(value, (list,tuple,np.ndarray)):
                    new_page = Page(value)
                    after = [new_page]
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.start != None and key.stop == key.step == None:   
                # documentation: new = old[:key.start] + list(value)
                # example: L[0:] = [1,2,3]
                before = mem.get_pages(self.group) 
                before_slice = before.getslice(0,start)

                if isinstance(value, Column):
                    after = before_slice + mem.get_pages(value.group)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    if not before_slice:
                        after = [Page(value)]
                    else:
                        last_page = before_slice[-1] 
                        if mem.get_ref_count(last_page) == 1:
                            last_page.extend(value)
                            after = before_slice
                        else:  # ref count > 1
                            new_page = Page(value)
                            after = before_slice + [new_page]
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.stop != None and key.start == key.step == None:  
                # documentation: new = list(value) + old[key.stop:] 
                # example: L[:3] = [1,2,3]
                before = mem.get_pages(self.group)
                before_slice = before.getslice(stop, self._len)
                if isinstance(value, Column):
                    after = mem.get_pages(value.group) + before_slice
                elif isinstance(value, (list,tuple, np.ndarray)):
                    new_page = Page(value)
                    after = [new_page] + before_slice
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
                
            elif key.step == None and key.start != None and key.stop != None:  # L[3:5] = [1,2,3]
                # documentation: new = old[:start] + list(values) + old[stop:] 
                
                stop = max(start,stop)  #  one of python's archaic rules.

                before = mem.get_pages(self.group)
                A, B = before.getslice(0,start), before.getslice(stop, self._len)
                if isinstance(value, Column):
                    after = A + mem.get_pages(value.group) + B
                elif isinstance(value, (list, tuple, np.ndarray)):
                    if value:
                        new_page = Page(value)
                        after = A + [new_page] + B  # new = old._getslice_(0,start) + list(value) + old._getslice_(stop,len(self.items))
                    else:
                        after = A + B
                else:
                    raise TypeError
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

            elif key.step != None:
                seq = range(start,stop,step)
                seq_size = len(seq)
                if len(value) > seq_size:
                    raise ValueError(f"attempt to assign sequence of size {len(value)} to extended slice of size {seq_size}")
                
                # documentation: See also test_slice_rules.py/MyList for details
                before = mem.get_pages(self.group)
                new = mem.get_data(self.group, slice(None)).tolist()  # new = old[:]  # cheap shallow pointer copy in case anything goes wrong.
                for new_index, position in zip(range(len(value)), seq):
                    new[position] = value[new_index]
                # all went well. No exceptions. Now update self.
                after = [Page(new)]  # This may seem redundant, but is in fact is good as the user may 
                # be cleaning up the dataset, so that we end up with a simple datatype instead of mixed.
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            else:
                raise KeyError(f"bad key: {key}")
        else:
            raise TypeError(f"bad key: {key}")

    def __delitem__(self, key):
        if isinstance(key, int):
            if -self._len-1 < key < self._len:
                before = mem.get_pages(self.group)
                after = before[:]
                ix,start,_,page = before.get_page_by_index(key)
                if mem.get_ref_count(page) == 1:
                    del page[key-start]
                else:
                    data = mem.get_data(page.group)
                    mask = np.ones(shape=data.shape)
                    new_data = np.compress(mask, data,axis=0)
                    after[ix] = Page(new_data)
            else:
                raise IndexError("list assignment index out of range")

            self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)

        elif isinstance(key, slice):
            start,stop,step = key.indices(self._len)
            before = mem.get_pages(self.group)
            if key.start == key.stop == None and key.step in (None,1):   # del L[:] == L.clear()
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=[])
            elif key.start != None and key.stop == key.step == None:   # del L[0:] 
                after = before.getslice(0, start)
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            elif key.stop != None and key.start == key.step == None:  # del L[:3] 
                after = before.getslice(stop, self._len)
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            elif key.step == None and key.start != None and key.stop != None:  # del L[3:5]
                after = before.getslice(0, start) + before.getslice(stop, self._len)
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            elif key.step != None:
                before = mem.get_pages(self.group)
                data = mem.get_data(self.group, slice(None))
                mask = np.ones(shape=data.shape)
                for i in range(start,stop,step):
                    mask[i] = 0
                new = np.compress(mask, data, axis=0)
                # all went well. No exceptions.
                after = [Page(new)]  # This may seem redundant, but is in fact is good as the user may 
                # be cleaning up the dataset, so that we end up with a simple datatype instead of mixed.
                self._len = mem.create_virtual_dataset(self.group, pages_before=before, pages_after=after)
            else:
                raise TypeError(f"bad key: {key}")
        else:
            raise TypeError(f"bad key: {key}")

    def __len__(self):
        return self._len

    def __eq__(self, other):
        if isinstance(other, (list,tuple)):
            A = self.__getitem__()
            return all(a==b for a,b in zip(A,other))
        
        if isinstance(other, Column):  # special case.
            if mem.get_pages(self.group) == mem.get_pages(other.group):
                return True  
                
        if isinstance(other, np.ndarray): 
            B = other
            A = self.__getitem__()
            return (A==B).all()
        else:
            raise TypeError
        
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
            return True
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

