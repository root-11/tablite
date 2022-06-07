import math
import pathlib
import random
import json
import time
import sys
import itertools
import operator
import warnings
import logging
import datetime as dt
from  multiprocessing import shared_memory

logging.getLogger('lml').propagate = False
logging.getLogger('pyexcel_io').propagate = False
logging.getLogger('pyexcel').propagate = False

import chardet
import pyexcel
import pyperclip
from tqdm import tqdm
import numpy as np
import h5py
import psutil
from mplite import TaskManager, Task

from collections import defaultdict
from itertools import chain, repeat

# from tablite.datatypes import DataTypes
# from tablite.file_reader_utils import detect_encoding, detect_seperator, split_by_sequence, text_escape
# from tablite.groupby_utils import Max, Min, Sum, First, Last, Count, CountUnique, Average, StandardDeviation, Median, Mode, GroupbyFunction


from tablite.memory_manager import MemoryManager, Page
from tablite.file_reader_utils import TextEscape
from tablite.utils import summary_statistics


mem = MemoryManager()

class Table(object):
    
    def __init__(self,key=None, save=False, _create=True, config=None) -> None:
        if key is None:
            key = mem.new_id('/table')
        elif not isinstance(key, str):
            raise TypeError
        self.key = key

        self.group = f"/table/{self.key}"
        self._columns = {}  # references for virtual datasets that behave like lists.
        if _create:
            if config is not None:
                if not isinstance(config, str):
                    raise TypeError("expected config as utf-8 encoded json")
            mem.create_table(key=self.group, save=save, config=config)  # attrs. 'columns'
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
        for key in list(self._columns):
            del self[key]
        mem.delete_table(self.group)

    def __str__(self):
        return f"Table({len(self._columns):,} columns, {len(self):,} rows)"

    def __repr__(self):
        return self.__str__()

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
    
    def __iter__(self):
        """
        Disabled. Users should use Table.rows or Table.columns
        """
        raise AttributeError("use Table.rows or Table.columns")

    def __len__(self):
        for v in self._columns.values():  
            return len(v)  # return on first key.
        return 0  # if there are no columns.

    def __setitem__(self, keys, values):
        if isinstance(keys, str): 
            if isinstance(values, (tuple,list,np.ndarray)):
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
            elif values is None:
                self._columns[keys] = Column(values)
            else:
                raise NotImplemented()
        elif isinstance(keys, tuple) and len(keys) == len(values):
            for key, value in zip(keys,values):
                self.__setitem__(key,value)
        else:
            raise NotImplemented()
    
    def __getitem__(self, *keys):
        """
        Enables selection of columns and rows
        Examples: 

            table['a']   # selects column 'a'
            table[:10]   # selects first 10 rows from all columns
            table['a','b', slice(3,20,2)]  # selects a slice from columns 'a' and 'b'
            table['b', 'a', 'a', 'c', 2:20:3]  # selects column 'b' and 'c' and 'a' twice for a slice.

        returns values in same order as selection.
        """
        if not isinstance(keys, tuple):
            keys = (keys, )
        if len(keys)==1 and all(isinstance(i,tuple) for i in keys):
            keys = keys[0]           
        
        cols = [c for c in keys if isinstance(c,str) and c in self._columns]
        cols = self.columns if not cols else cols
        slices = [i for i in keys if isinstance(i, slice)]
        if len(cols)==1:
            col = self._columns[cols[0]]
            if slices:
                return col[slices[0]]
            else:
                return col
        elif slices:
            slc = slices[0]
            t = Table()
            for name in cols:
                t[name] = self._columns[name][slc]
            return t
        else:
            t = Table()
            for name in cols:
                t[name] = self._columns[name]
            return t

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
            raise ValueError("Columns names are not the same. Use table.stack instead.")
        for name, col in self._columns.items():
            col += other[name]
        return self

    def __mul__(self,other):
        """
        enables repetition of a table
        Example: Table_x_10 = table * 10
        """
        if not isinstance(other, int):
            raise TypeError(f"can't multiply Table with {type(other)}")
        t = self.copy()
        for col in t._columns.values():
            col *= other
        return t

    def __imul__(self,other):
        """
        extends a table N times onto using itself as source.
        """
        if not isinstance(other, int):
            raise TypeError(f"can't multiply Table with {type(other)}")

        for col in self._columns.values():
            col *= other
        return self

    @classmethod
    def reload_saved_tables(cls,path=None):
        """
        Loads saved tables from a hdf5 storage.
        
        The default storage locations is:
        >>> from tablite.config import HDF5_Config
        >>> print(Config.H5_STORAGE)

        To import without changing the default location use:
        tables = reload_saved_tables("c:/another/location.hdf5)
        """
        tables = []
        if path is None:
            path = mem.path
        unsaved = 0
        with h5py.File(path, 'r+') as h5:
            if "/table" not in h5.keys():
                return []

            for table_key in h5["/table"].keys():
                dset = h5[f"/table/{table_key}"]
                if dset.attrs['saved'] is False:
                    unsaved += 1
                else:
                    t = Table.load(path, key=table_key)
                    tables.append(t)
        if unsaved:
            warnings.warn(f"Dropping {unsaved} tables from cache where save==False.")
        return tables

    @classmethod
    def load(cls, path, key):
        with h5py.File(path, 'r+') as h5:
            group = f"/table/{key}"
            dset = h5[group]
            saved = dset.attrs['saved']
            t = Table(key=key, save=saved, _create=False)
            columns = json.loads(dset.attrs['columns'])
            for col_name, column_key in columns.items():
                c = Column.load(key=column_key)
                ds2 = h5[f"/column/{column_key}"]
                c._len = ds2.len()
                t[col_name] = c
                
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
            self.__setitem__(name,None)

    def add_column(self,name, data=None):
        if not isinstance(name, str):
            raise TypeError()
        if name in self.columns:
            raise ValueError(f"{name} already in {self.columns}")
        if not data:
            pass
        self.__setitem__(name,data)

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

    def types(self):
        d = {}
        for name,col in self._columns.items():
            assert isinstance(col, Column)
            d[name] = col.types()
        return d

    def to_ascii(self, blanks=None, row_counts=None,split_after=None):
        """
        enables viewing in terminals
        returns the table as ascii string

        blanks: any stringable item.
        row_counts: declares the column with row counts, so it is presented as the first column.
        split_after: integer: inserts "..." to highlight split of rows
        """
        widths = {}
        column_types = {}
        names = list(self.columns)
        if not names:
            return "Empty table"
        for name,col in self._columns.items():
            
            types = col.types()
            if name == row_counts:
                column_types[name] = 'row'
            elif len(types) == 1:
                dt,n = types.popitem()
                column_types[name] = dt.__name__
            else:
                column_types[name] = 'mixed'
            dots = len("...") if split_after is not None else 0
            widths[name] = max([len(column_types[name]), len(name), dots] + [len(str(v)) if not isinstance(v,str) else len(str(v)) for v in col])

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
        s.append("|" + "|".join([column_types[n].center(widths[n], " ") for n in names]) + "|")
        s.append("+" + "+".join(["-" * widths[n] for n in names]) + "+")
        for ix, row in enumerate(self.rows):
            s.append("|" + "|".join([adjust(v, widths[n]) for v, n in zip(row, names)]) + "|")
            if ix == split_after:
                s.append("|" + "|".join([adjust("...", widths[n]) for _, n in zip(row, names)]) + "|")
                
        s.append("+" + "+".join(["=" * widths[h] for h in names]) + "+")
        return "\n".join(s)

    def show(self, *args, blanks=None):
        """
        accepted args:
          - slice

        """ 
        row_count_tags = ['#', '~', '*'] 
        cols = set(self.columns)
        for n,tag in itertools.product(range(1,6), row_count_tags):
            if n*tag not in cols:
                tag = n*tag
                break

        t = Table()
        split_after = None
        if args:
            for arg in args:
                if isinstance(arg, slice):
                    ro = range(*arg.indices(len(self)))
                    if len(ro)!=0:
                        t[tag] = [f"{i:,}" for i in ro]  # add rowcounts as first column.
                        for name,col in self._columns.items():
                            t[name] = col[arg]  # copy to match slices
                    else:
                        t.add_columns(*[tag] + self.columns)

        elif len(self) < 20:
            t[tag] = [f"{i:,}" for i in range(len(self))]  # add rowcounts to copy 
            for name,col in self._columns.items():
                t[name] = col

        else:  # take first and last 7 rows.
            n = len(self)
            split_after = 7
            t[tag] = [f"{i:,}" for i in range(7)] + [f"{i:,}" for i in range(n-7, n)]
            for name, col in self._columns.items():
                t[name] = [i for i in col[:7]] + [i for i in col[-7:]] 

        print(t.to_ascii(blanks=blanks,row_counts=tag, split_after=split_after))

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

    def to_json(self):
        d = {}
        for name, col in self._columns.items():
            d[name] = col[:].tolist()
        return json.dumps(d)

    @classmethod
    def from_json(cls, jsn):
        t = Table()
        for name, data in json.loads(jsn).items():
            if not isinstance(name, str):
                raise TypeError(f"expect {name} as a string")
            if not isinstance(data, list):
                raise TypeError(f"expected {data} as list")
            t[name] = data
        return t

    def to_hdf5(self, path):
        """
        creates a copy of the table as hdf5
        the hdf5 layout can be viewed using Table.inspect_h5_file(path/to.hdf5)
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        
        total = ":,".format(len(self.columns) * len(self))
        print(f"writing {total} records to {path}")

        with h5py.File(path, 'a') as f:
            with tqdm(total=len(self.columns), unit='columns') as pbar:
                n = 0
                for name, mc in self.columns.values():
                    f.create_dataset(name, data=mc[:])  # stored in hdf5 as '/name'
                    n += 1
                    pbar.update(n)
        print(f"writing {path} to HDF5 done")

    @classmethod
    def import_file(cls, path,  import_as, 
        newline='\n', text_qualifier=None,  delimiter=',', first_row_has_headers=True, columns=None, sheet=None, 
        start=0, limit=sys.maxsize, strip_leading_and_tailing_whitespace=True):
        """
        reads path and imports 1 or more tables as hdf5

        path: pathlib.Path or str
        import_as: 'csv','xlsx','txt'                               *123
        newline: newline character '\n', '\r\n' or b'\n', b'\r\n'   *13
        text_qualifier: character: " or '                           +13
        delimiter: character: typically ",", ";" or "|"             *1+3
        first_row_has_headers: boolean                              *123
        columns: dict with column names or indices and datatypes    *123
            {'A': int, 'B': str, 'C': float, D: datetime}
            Excess column names are ignored.

        sheet: sheet name to import (e.g. 'sheet_1')                 *2
            sheets not found excess names are ignored.
            filenames will be {path}+{sheet}.h5
        
        start: the first line to be read.
        limit: the number of lines to be read from start
        strip_leading_and_tailing_whitespace: bool: default True. 

        (*) required, (+) optional, (1) csv, (2) xlsx, (3) txt, (4) h5

        TABLES FROM IMPORTED FILES ARE IMMUTABLE.
        OTHER TABLES EXIST IN MEMORY MANAGERs CACHE IF USE DISK == True
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"expected pathlib.Path, got {type(path)}")
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
        if not isinstance(import_as, str):
            raise TypeError(f"import_as is expected to be str, not {type(import_as)}: {import_as}")
        reader = file_readers.get(import_as,None)
        if reader is None:
            raise ValueError(f"{import_as} is not in list of supported reader:\n{list(file_readers.keys())}")
        
        if not isinstance(strip_leading_and_tailing_whitespace, bool):
            raise TypeError()
        
        # At this point the import seems valid.
        # Now we check if the file already has been imported.
        config = {
            'import_as': import_as,
            'path': str(path),
            'filesize': path.stat().st_size,  # if file length changes - re-import.
            'delimiter': delimiter,
            'columns': columns, 
            'newline': newline,
            'first_row_has_headers': first_row_has_headers,
            'text_qualifier': text_qualifier,
            'sheet': sheet,
            'start': start,
            'limit': limit,
            'strip_leading_and_tailing_whitespace': strip_leading_and_tailing_whitespace
        }
        jsn_str = json.dumps(config)
        for table_key, jsnb in mem.get_imported_tables().items():
            if jsn_str == jsnb:
                return Table.load(mem.path, table_key)  # table already imported.
        # not returned yet? Then it's an import job:
        t = reader(**config)
        mem.set_config(t.group, jsn_str)
        if t.save is False:
            raise AttributeError("filereader should set table.save = True to avoid repeated imports")
        return t
    
    def index(self, *keys):
        """ 
        Returns index on *keys columns as d[(key tuple, )] = {index1, index2, ...} 
        """
        # keys = keys[0] if len(keys)==1 else keys
        # if len(keys) == 1 and keys[0] in self._columns:
        #     col = self._columns[keys[0]]
        #     idx = {(k,):set(v) for k,v in col.index().items()}
        # else:
        idx = defaultdict(set)
        tbl = self.__getitem__(*keys)
        g = tbl.rows if isinstance(tbl, Table) else iter(tbl)
        for ix, key in enumerate(g):
            if isinstance(key, list):
                key = tuple(key)
            else:
                key = (key,)
            idx[key].add(ix)
        return idx

    def filter(self, expressions, filter_type='all'):
        """
        enables filtering across columns for multiple criteria.
        
        expressions: 
        list of dicts:
        L = [
            {'column1':'A', 'criteria': "==", 'column2': 'B'}, 
            {'column1':'C', 'criteria': "!=", "value2": '4'},
            {'value1': 200, 'criteria': "<", column2: 'D' }
        ]

        accepted dictionary keys: 'column1', 'column2', 'criteria', 'value1', 'value2'

        filter_type: 'all' or 'any'
        """
        if not isinstance(expressions, list):
            raise TypeError

        for expression in expressions:
            if not isinstance(expression, dict):
                raise TypeError(f"invalid expression: {expression}")
            if not len(expression)==3:
                raise ValueError(f"expected 3 items, got {expression}")
            x = {'column1', 'column2', 'criteria', 'value1', 'value2'}
            if not set(expression.keys()).issubset(x):
                raise ValueError(f"got unknown key: {set(expression.keys()).difference(x)}")
            if expression['criteria'] not in filter_ops:
                raise ValueError(f"criteria missing from {expression}")

            c1 = expression.get('column1',None) 
            if c1 is not None and c1 not in self.columns: 
                raise ValueError(f"no such column: {c1}")
            v1 = expression.get('value1', None)
            if v1 is not None and c1 is not None:
                raise ValueError("filter can only take 1 left expr element. Got 2.")

            c2 = expression.get('column1',None) 
            if c2 is not None and c2 not in self.columns: 
                raise ValueError(f"no such column: {c2}")
            v2 = expression.get('value2',None)
            if v2 is not None and c1 is not None:
                raise ValueError("filter can only take 1 right expression element. Got 2.")
               
        if not isinstance(filter_type, str):
            raise TypeError()
        if not filter_type in {'all', 'any'}:
            raise ValueError(f"filter_type: {filter_type} not in ['all', 'any']")

        # the results are to be gathered here:
        arr = np.zeros(shape=(len(expressions), len(self)), dtype=bool)
        shm = shared_memory(create=True, size=arr.nbytes)
        result_array = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        
        # the task manager enables evaluation of a column per core,
        # which is assembled in the shared array.

        max_task_size = 1_000_000  # 1 million rows per column per core (best guess!)
        n_tasks = expressions * math.ceil( len(self) / max_task_size )  
        n_cpus = min(n_tasks, psutil.cpu_count())
        
        tasks = []
        for ix, expression in enumerate(expressions):
            blocksize = math.ceil(len(self) / max_task_size)
            for block in range(0, len(self), blocksize):
                config = {'table_key':self.key, 'expression':expression, 
                          'shm_name':shm.name, 'shm_index':ix, 'shm_shape': arr.shape, 
                          'slice_':slice(block, block+blocksize)}
                task = Task(f=filter_task, **config)
                tasks.append(task)

        tasks2 = []
        blocksize = math.ceil(len(self) / max_task_size)
        for block in range(0, len(self), blocksize):
            config = {
                'table_key': self.key,
                'true_key': mem.new_id('/table'),
                'false_key': mem.new_id('/table'),
                'shm_name': shm.name, 'shm_shape': arr.shape,
                'slice_': slice(block, block+blocksize,1)
            }
            task = Task(f=merge_task, **config)
            tasks2.append(task)

        with TaskManager(n_cpus) as tm: 
            # EVALUATE 
            errs = tm.execute(tasks)  # tm.execute returns the tasks with results, but we don't really care as the result is in the result array.
            if any(i is not None for i in errs):
                raise Exception(errs)
            # MERGE RESULTS
            errs = tm.execute(tasks2)  # tm.execute returns the tasks with results, but we don't really care as the result is in the result array.
            if any(i is not None for i in errs):
                raise Exception(errs)

        table_true, table_false = None, None
        for task in tasks2:
            tmp_true = Table.load(mem.path, key=task.kwargs['true_key'])
            tmp_false = Table.load(mem.path, key=task.kwargs['false_key'])
            if table_true is None:
                table_true = tmp_true
                table_false = tmp_false
            else:
                table_true += tmp_true
                table_false += tmp_false
            
        return table_true, table_false
    
    def sort_index(self, nan_value=float('inf'), **kwargs):  # TODO: This is slow single core code.
        """ 
        helper for methods `sort` and `is_sorted` 
        nan_value: value used to represent non-sortable values such as None and np.nan during sort.
        kwargs: sort criteria. See Table.sort()
        """
        if not isinstance(kwargs, dict):
            raise ValueError("Expected keyword arguments")
        if not kwargs:
            kwargs = {c: False for c in self.columns}
        
        for k, v in kwargs.items():
            if k not in self.columns:
                raise ValueError(f"no column {k}")
            if not isinstance(v, bool):
                raise ValueError(f"{k} was mapped to {v} - a non-boolean")

        rank = {i: tuple() for i in range(len(self))}
        for key in kwargs:
            unique_values = {v: 0 for v in self._columns[key] if v is not None}
            for r, v in enumerate(sorted(unique_values, reverse=kwargs[key])):
                unique_values[v] = r
            for ix, v in enumerate(self._columns[key]):
                rank[ix] += (unique_values.get(v, nan_value),)

        new_order = [(r, i) for i, r in rank.items()]  # tuples are listed and sort...
        new_order.sort()
        sorted_index = [i for r, i in new_order]  # new index is extracted.

        rank.clear()  # free memory.
        new_order.clear()
        return sorted_index

    def sort(self, nan_value=float('-inf'), **kwargs):  # TODO: This is slow single core code.
        """ Perform multi-pass sorting with precedence given order of column names.
        nan_value: value used to represent non-sortable values such as None and np.nan during sort.
        kwargs: keys: columns, values: 'reverse' as boolean.

        examples: 
        Table.sort('A'=False)  means sort by 'A' in ascending order.
        Table.sort('A'=True, 'B'=False) means sort 'A' in descending order, then (2nd priority) sort B in ascending order.
        """
        sorted_index = self.sort_index(nan_value, **kwargs)
        t = Table()
        for col_name, col in self._columns.items():
            t.add_column(col_name, data=[col[ix] for ix in sorted_index])
        return t

    def is_sorted(self, nan_value=float('inf'), **kwargs):  # TODO: This is slow single core code.
        """ Performs multi-pass sorting check with precedence given order of column names.
        nan_value: value used to represent non-sortable values such as None and np.nan during sort.
        **kwargs: sort criteria. See Table.sort()
        :return bool
        """
        sorted_index = self.sort_index(nan_value, **kwargs)
        if any(ix != i for ix, i in enumerate(sorted_index)):
            return False
        return True

    def all(self, **kwargs):  # TODO: This is slow single core code.
        """
        returns Table for rows where ALL kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you remember to add the ** in front of your dict?")
        if not all(k in self.columns for k in kwargs):
            raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in self.columns]}")

        ixs = None
        for k, v in kwargs.items():
            col = self.columns[k]
            if ixs is None:  # first header.
                if callable(v):
                    ix2 = {ix for ix, i in enumerate(col) if v(i)}
                else:
                    ix2 = {ix for ix, i in enumerate(col) if v == i}

            else:  # remaining headers.
                if callable(v):
                    ix2 = {ix for ix in ixs if v(col[ix])}
                else:
                    ix2 = {ix for ix in ixs if v == col[ix]}

            if not isinstance(ixs, set):
                ixs = ix2
            else:
                ixs = ixs.intersection(ix2)

            if not ixs:  # There are no matches.
                break

        t = Table()
        for col in tqdm(self.columns.values(), total=len(self.columns), desc="columns"):
            t.add_column(col.header, col.datatype, col.allow_empty, data=[col[ix] for ix in ixs])
        return t

    def any(self, **kwargs):  # TODO: This is slow single core code.
        """
        returns Table for rows where ANY kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you remember to add the ** in front of your dict?")

        ixs = set()
        for k, v in kwargs.items():
            col = self.columns[k]
            if callable(v):
                ix2 = {ix for ix, r in enumerate(col) if v(r)}
            else:
                ix2 = {ix for ix, r in enumerate(col) if v == r}
            ixs.update(ix2)

        t = Table()
        for col in tqdm(self.columns.values(), total=len(self.columns), desc="columns"):
            t.add_column(col.header, col.datatype, col.allow_empty, data=[col[ix] for ix in ixs])
        return t

    def groupby(self, keys, functions, pivot_on=None):  # TODO: This is slow single core code.
        """
        :param keys: headers for grouping
        :param functions: list of headers and functions.
        :return: GroupBy class
        Example usage:
            from tablite import Table
            t = Table()
            t.add_column('date', data=[1,1,1,2,2,2])
            t.add_column('sku', data=[1,2,3,1,2,3])
            t.add_column('qty', data=[4,5,4,5,3,7])
            from tablite import GroupBy, Sum
            g = t.groupby(keys=['sku'], functions=[('qty', Sum)])
            g.tablite.show()
        """
        g = GroupBy(keys=keys, functions=functions)
        g += self
        if pivot_on:
            g.pivot(pivot_on)
        return g.table()
    
    def _join_type_check(self, other, left_keys, right_keys, left_columns, right_columns):
        if not isinstance(other, Table):
            raise TypeError(f"other expected other to be type Table, not {type(other)}")

        if not isinstance(left_keys, list) and all(isinstance(k, str) for k in left_keys):
            raise TypeError(f"Expected keys as list of strings, not {type(left_keys)}")
        if not isinstance(right_keys, list) and all(isinstance(k, str) for k in right_keys):
            raise TypeError(f"Expected keys as list of strings, not {type(right_keys)}")

        if any(key not in self.columns for key in left_keys):
            raise ValueError(f"left key(s) not found: {[k for k in left_keys if k not in self.columns]}")
        if any(key not in other.columns for key in right_keys):
            raise ValueError(f"right key(s) not found: {[k for k in right_keys if k not in other.columns]}")

        if len(left_keys) != len(right_keys):
            raise ValueError(f"Keys do not have same length: \n{left_keys}, \n{right_keys}")

        for L, R in zip(left_keys, right_keys):
            Lcol, Rcol = self.columns[L], other.columns[R]
            if Lcol.datatype != Rcol.datatype:
                raise TypeError(f"{L} is {Lcol.datatype}, but {R} is {Rcol.datatype}")

        if not isinstance(left_columns, list) or not left_columns:
            raise TypeError("left_columns (list of strings) are required")
        if any(column not in self for column in left_columns):
            raise ValueError(f"Column not found: {[c for c in left_columns if c not in self.columns]}")

        if not isinstance(right_columns, list) or not right_columns:
            raise TypeError("right_columns (list or strings) are required")
        if any(column not in other for column in right_columns):
            raise ValueError(f"Column not found: {[c for c in right_columns if c not in other.columns]}")
        # Input is now guaranteed to be valid.

    def join(self, other, left_keys, right_keys, left_columns, right_columns, kind='inner'):
        """
        short-cut for all join functions.
        """
        kinds = {
            'inner':self.inner_join,
            'left':self.left_join,
            'outer':self.outer_join
        }
        if kind not in kinds:
            raise ValueError(f"join type unknown: {kind}")
        f = kinds.get(kind,None)
        return f(self,other,left_keys,right_keys,left_columns,right_columns)
    
    def left_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None):  # TODO: This is slow single core code.
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param left_columns: list of left columns to retain, if None, all are retained.
        :param right_columns: list of right columns to retain, if None, all are retained.
        :return: new Table
        Example:
        SQL:   SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
        Tablite: left_join = numbers.left_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])
        """
        if left_columns is None:
            left_columns = list(self.columns)
        if right_columns is None:
            right_columns = list(other.columns)

        self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        left_join = Table(use_disk=self._use_disk)
        for col_name in left_columns:
            col = self.columns[col_name]
            left_join.add_column(col_name, col.datatype, allow_empty=True)

        right_join_col_name = {}
        for col_name in right_columns:
            col = other.columns[col_name]
            revised_name = left_join.check_for_duplicate_header(col_name)
            right_join_col_name[revised_name] = col_name
            left_join.add_column(revised_name, col.datatype, allow_empty=True)

        left_ixs = range(len(self))
        right_idx = other.index(*right_keys)

        for left_ix in tqdm(left_ixs, total=len(left_ixs)):
            key = tuple(self[h][left_ix] for h in left_keys)
            right_ixs = right_idx.get(key, (None,))
            for right_ix in right_ixs:
                for col_name, column in left_join.columns.items():
                    if col_name in self:
                        column.append(self[col_name][left_ix])
                    elif col_name in right_join_col_name:
                        original_name = right_join_col_name[col_name]
                        if right_ix is not None:
                            column.append(other[original_name][right_ix])
                        else:
                            column.append(None)
                    else:
                        raise Exception('bad logic')
        return left_join

    def inner_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None):  # TODO: This is slow single core code.
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param left_columns: list of left columns to retain, if None, all are retained.
        :param right_columns: list of right columns to retain, if None, all are retained.
        :return: new Table
        Example:
        SQL:   SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
        Tablite: inner_join = numbers.inner_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])
        """
        if left_columns is None:
            left_columns = list(self.columns)
        if right_columns is None:
            right_columns = list(other.columns)

        self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        inner_join = Table(use_disk=self._use_disk)
        for col_name in left_columns:
            col = self.columns[col_name]
            inner_join.add_column(col_name, col.datatype, allow_empty=True)

        right_join_col_name = {}
        for col_name in right_columns:
            col = other.columns[col_name]
            revised_name = inner_join.check_for_duplicate_header(col_name)
            right_join_col_name[revised_name] = col_name
            inner_join.add_column(revised_name, col.datatype, allow_empty=True)

        key_union = set(self.filter(*left_keys)).intersection(set(other.filter(*right_keys)))

        left_ixs = self.index(*left_keys)
        right_ixs = other.index(*right_keys)

        for key in tqdm(sorted(key_union), total=len(key_union)):
            for left_ix in left_ixs.get(key, set()):
                for right_ix in right_ixs.get(key, set()):
                    for col_name, column in inner_join.columns.items():
                        if col_name in self:
                            column.append(self[col_name][left_ix])
                        else:  # col_name in right_join_col_name:
                            original_name = right_join_col_name[col_name]
                            column.append(other[original_name][right_ix])

        return inner_join

    def outer_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None):  # TODO: This is slow single core code.
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param left_columns: list of left columns to retain, if None, all are retained.
        :param right_columns: list of right columns to retain, if None, all are retained.
        :return: new Table
        Example:
        SQL:   SELECT number, letter FROM numbers OUTER JOIN letters ON numbers.colour == letters.color
        Tablite: outer_join = numbers.outer_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])
        """
        if left_columns is None:
            left_columns = list(self.columns)
        if right_columns is None:
            right_columns = list(other.columns)

        self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        outer_join = Table(use_disk=self._use_disk)
        for col_name in left_columns:
            col = self.columns[col_name]
            outer_join.add_column(col_name, col.datatype, allow_empty=True)

        right_join_col_name = {}
        for col_name in right_columns:
            col = other.columns[col_name]
            revised_name = outer_join.check_for_duplicate_header(col_name)
            right_join_col_name[revised_name] = col_name
            outer_join.add_column(revised_name, col.datatype, allow_empty=True)

        left_ixs = range(len(self))
        right_idx = other.index(*right_keys)
        right_keyset = set(right_idx)

        for left_ix in tqdm(left_ixs, total=left_ixs.stop, desc="left side outer join"):
            key = tuple(self[h][left_ix] for h in left_keys)
            right_ixs = right_idx.get(key, (None,))
            right_keyset.discard(key)
            for right_ix in right_ixs:
                for col_name, column in outer_join.columns.items():
                    if col_name in self:
                        column.append(self[col_name][left_ix])
                    elif col_name in right_join_col_name:
                        original_name = right_join_col_name[col_name]
                        if right_ix is not None:
                            column.append(other[original_name][right_ix])
                        else:
                            column.append(None)
                    else:
                        raise Exception('bad logic')

        for right_key in tqdm(right_keyset, total=len(right_keyset), desc="right side outer join"):
            for right_ix in right_idx[right_key]:
                for col_name, column in outer_join.columns.items():
                    if col_name in self:
                        column.append(None)
                    elif col_name in right_join_col_name:
                        original_name = right_join_col_name[col_name]
                        column.append(other[original_name][right_ix])
                    else:
                        raise Exception('bad logic')
        return outer_join

    def lookup(self, other, *criteria, all=True):  # TODO: This is slow single core code.
        """ function for looking up values in other according to criteria
        :param: other: Table
        :param: criteria: Each criteria must be a tuple with value comparisons in the form:
            (LEFT, OPERATOR, RIGHT)
        :param: all: boolean: True=ALL, False=Any
        OPERATOR must be a callable that returns a boolean
        LEFT must be a value that the OPERATOR can compare.
        RIGHT must be a value that the OPERATOR can compare.
        Examples:
              ('column A', "==", 'column B')  # comparison of two columns
              ('Date', "<", DataTypes.date(24,12) )  # value from column 'Date' is before 24/12.
              f = lambda L,R: all( ord(L) < ord(R) )  # uses custom function.
              ('text 1', f, 'text 2')
              value from column 'text 1' is compared with value from column 'text 2'
        """
        assert isinstance(self, Table)
        assert isinstance(other, Table)

        all = all
        any = not all

        def not_in(a, b):
            return not operator.contains(a, b)

        ops = {
            "in": operator.contains,
            "not in": not_in,
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "!=": operator.ne,
            "==": operator.eq,
        }

        table3 = Table(use_disk=self._use_disk)
        for name, col in chain(self.columns.items(), other.columns.items()):
            table3.add_column(name, col.datatype, allow_empty=True)

        functions, left_columns, right_columns = [], set(), set()

        for left, op, right in criteria:
            left_columns.add(left)
            right_columns.add(right)
            if callable(op):
                pass  # it's a custom function.
            else:
                op = ops.get(op, None)
                if not callable(op):
                    raise ValueError(f"{op} not a recognised operator for comparison.")

            functions.append((op, left, right))

        lru_cache = {}
        empty_row = tuple(None for _ in other.columns)

        for row1 in tqdm(self.rows, total=self.__len__()):
            row1_tup = tuple(v for v, name in zip(row1, self.columns) if name in left_columns)
            row1d = {name: value for name, value in zip(self.columns, row1) if name in left_columns}

            match_found = True if row1_tup in lru_cache else False

            if not match_found:  # search.
                for row2 in other.rows:
                    row2d = {name: value for name, value in zip(other.columns, row2) if name in right_columns}

                    evaluations = [op(row1d.get(left, left), row2d.get(right, right)) for op, left, right in functions]
                    # The evaluations above does a neat trick:
                    # as L is a dict, L.get(left, L) will return a value
                    # from the columns IF left is a column name. If it isn't
                    # the function will treat left as a value.
                    # The same applies to right.

                    if all and not False in evaluations:
                        match_found = True
                        lru_cache[row1_tup] = row2
                        break
                    elif any and True in evaluations:
                        match_found = True
                        lru_cache[row1_tup] = row2
                        break
                    else:
                        continue

            if not match_found:  # no match found.
                lru_cache[row1_tup] = empty_row

            new_row = row1 + lru_cache[row1_tup]

            table3.add_row(new_row)

        return table3
    
    def pivot_table(self, *args):
        raise NotImplementedError



class Column(object):
    def __init__(self, data=None, key=None) -> None:
        if key is None:
            key = mem.new_id('/column')
        elif not isinstance(key, str):
            raise TypeError
        self.key = key

        self.group = f"/column/{self.key}"
        self._len = 0
        if data is not None:
            self.extend(data)
    
    def __str__(self) -> str:
        return f"<{self.__class__.__name__}>({self._len} values | key={self.key})"

    def __repr__(self) -> str:
        return self.__str__()

    def types(self):
        """
        returns dict with datatype: frequency of occurrence
        """
        return mem.get_pages(self.group).get_types()

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
                new_page = page(data)  # create new page from copy
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
            new_page = Page(new_data)
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
            new_page = Page(data)
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
            return all(a==b for a,b in zip(self[:],other))
        
        elif isinstance(other, Column):  
            if mem.get_pages(self.group) == mem.get_pages(other.group):  # special case.
                return True  
            else:
                return (self[:] == other[:]).all()
        elif isinstance(other, np.ndarray): 
            return (self[:]==other).all()
        else:

            raise TypeError
        
    def copy(self):
        return Column(data=self)

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
        - histogram
        """
        return summary_statistics(self.histogram())

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
        old_pages = mem.get_pages(self.group)
        new_pages = old_pages * other
        self._len = mem.create_virtual_dataset(self.group, old_pages, new_pages)
        return self
    
    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError(f"a column can be repeated an integer number of times, not {type(other)} number of times")
        new = Column()
        old_pages = mem.get_pages(self.group)
        new_pages = old_pages * other
        new._len = mem.create_virtual_dataset(new.group, old_pages, new_pages)
        return new
    
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


def _in(a,b):
    """
    enables filter function 'in'
    """
    return a.decode('utf-8') in b.decode('utf-8')


filter_ops = {
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "<": operator.lt,
            "<=": operator.le,
            "!=": operator.ne,
            "in": _in
        }

filter_ops_from_text = {
    "gt": ">",
    "gteq": ">=",
    "eq": "==",
    "lt": "<",
    "lteq": "<=",
    "neq": "!=",
    "in": _in
}

def filter_task(table_key, expression, shm_name, shm_index, shm_shape, slice_):
    assert isinstance(table_key, str)  # 10 --> group = '/table/10'
    assert isinstance(expression, dict)
    assert len(expression)==3
    assert isinstance(shm_name, str)
    assert isinstance(shm_index, int)
    assert isinstance(shm_shape, tuple)
    assert isinstance(slice_,slice)
    c1 = expression.get('column1',None)
    c2 = expression.get('column2',None)
    c = expression.get('criteria',None)
    assert c in filter_ops
    f = filter_ops.get(c)
    assert callable(f)
    v1 = expression.get('value1',None)
    v2 = expression.get('value2',None)

    with h5py.File(mem.path, 'r') as h5:
        dset = h5[f'/table/{table_key}']
        columns = json.loads(dset.attrs['columns'])
        if c1 is not None:
            column_key = columns[c1]
            pages = mem.get_pages(f'/column/{column_key}')
            dset_A = pages.getslice(slice.start,slice.stop)
        else:  # v1 is active:
            dset_A = np.array([v1] * (slice.stop-slice.start))
        
        if c2 is not None:
            column_key = columns[c2]
            pages = mem.get_pages(f'/column/{column_key}')
            dset_B = pages.getslice(slice.start, slice.stop)
        else:  # v2 is active:
            dset_B = np.array([v2] * (slice.stop-slice.start))

        existing_shm = shared_memory.SharedMemory(name=shm_name)  # connect
        ra = np.ndarray(shm_shape, dtype=np.int64, buffer=existing_shm.buf)
        ra[shm_index] = f(dset_A,dset_B)
        existing_shm.close()  # disconnect
    return None


def merge_task(table_key, true_key, false_key, shm_name, shm_shape, slice_):
    raise NotImplementedError("wip")

def merge(source, mask, filter_type, slice_):
    """ PARALLEL TASK FUNCTION
    creates new tables from combining source and mask.
    """
    if not isinstance(source, dict):
        raise TypeError
    for L in source.values():
        if not isinstance(L, list):
            raise TypeError
        if not all(isinstance(sma, SharedMemoryAddress) for sma in L):
            raise TypeError

    if not isinstance(mask, SharedMemoryAddress):
        raise TypeError
    if not isinstance(filter_type, str) and filter_type in {'any', 'all'}:
        raise TypeError
    if not isinstance(slice_, slice):
        raise TypeError
    
    # 1. determine length of Falses and Trues
    f = any if filter_type == 'any' else all
    handle, mask = mask.to_shm() 
    if len(mask) == 1:
        true_mask = mask[0][slice_]
    else:
        true_mask = [f(c[i] for c in mask) for i in range(slice_.start, slice_.stop)]
    false_mask = np.invert(true_mask)

    t1 = Table.from_address(source)  # 2. load Table.from_shm(source)
    # 3. populate the tables
    
    true, false = Table(), Table()
    for name, mc in t1.columns.items():
        mc_unfiltered = np.array(mc[slice_])
        if any(true_mask):
            data = mc_unfiltered[true_mask]
            true.add_column(name, data)  # data = mc_unfiltered[new_mask]
        if any(false_mask):
            data = mc_unfiltered[false_mask]
            false.add_column(name, data)

    # 4. return table.to_shm()
    return true.address, false.address   


def excel_reader(path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table(s) from excel path

    **kwargs are excess arguments that are ignored.
    """
    book = pyexcel.get_book(file_name=str(path))

    if sheet is None:  # help the user.
        raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in book]}")
    elif sheet not in {ws.name for ws in book}:
        raise ValueError(f"sheet not found: {sheet}")
    
    if not (isinstance(start, int) and start >= 0):
        raise ValueError("expected start as an integer >=0")
    if not (isinstance(limit, int) and limit > 0):
        raise ValueError("expected limit as integer > 0")

    # import all sheets or a subset
    for ws in book:
        if ws.name != sheet:
            continue
        else:
            break
    assert ws.name == sheet, "sheet not found."

    config = {**kwargs, **{"first_row_has_headers":first_row_has_headers, "sheet":sheet, "columns":columns, 'start':start, 'limit':limit}}
    t = Table(save=True, config=json.dumps(config))
    for idx, column in enumerate(ws.columns(), 1):
        
        if first_row_has_headers:
            header, start_row_pos = str(column[0]), max(1, start)
        else:
            header, start_row_pos = f"_{idx}", max(0,start)

        if columns is not None:
            if header not in columns:
                continue

        t[header] = [v for v in column[start_row_pos:start_row_pos+limit]]
    return t


def ods_reader(path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize, **kwargs):
    """
    returns Table from .ODS

    **kwargs are excess arguments that are ignored.
    """
    sheets = pyexcel.get_book_dict(file_name=str(path))

    if sheet is None or sheet not in sheets:
        raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in sheets]}")
            
    data = sheets[sheet]
    for _ in range(len(data)):  # remove empty lines at the end of the data.
        if "" == "".join(str(i) for i in data[-1]):
            data = data[:-1]
        else:
            break
    
    if not (isinstance(start, int) and start >= 0):
        raise ValueError("expected start as an integer >=0")
    if not (isinstance(limit, int) and limit > 0):
        raise ValueError("expected limit as integer > 0")

    config = {**kwargs, **{"first_row_has_headers":first_row_has_headers, "sheet":sheet, "columns":columns, 'start':start, 'limit':limit}}
    t = Table(save=True, config=json.dumps(config))
    for ix, value in enumerate(data[0]):
        if first_row_has_headers:
            header, start_row_pos = str(value), 1
        else:
            header, start_row_pos = f"_{ix + 1}", 0

        if columns is not None:
            if header not in columns:
                continue    

        t[header] = [row[ix] for row in data[start_row_pos:start_row_pos+limit] if len(row) > ix]
    return t


def text_reader(path, newline='\n', text_qualifier=None, delimiter=',', first_row_has_headers=True, 
                columns=None, start=0, limit=sys.maxsize, 
                strip_leading_and_tailing_whitespace=True, **kwargs):
    """
    **kwargs are excess arguments that are ignored.
    """
    # define and specify tasks.
    path = pathlib.Path(path)
    file_length = path.stat().st_size  # 9,998,765,432 = 10Gb
    working_overhead = 5  # random guess. Calibrate as required.
    working_memory_required = file_length * working_overhead
    memory_usage_ceiling = 0.9
    if working_memory_required < psutil.virtual_memory().free:
        mem_per_cpu = math.ceil(working_memory_required / psutil.cpu_count())
    else:
        memory_ceiling = int(psutil.virtual_memory().total * memory_usage_ceiling)
        memory_used = psutil.virtual_memory().used
        available = memory_ceiling - memory_used  # 6,321,123,321 = 6 Gb
        mem_per_cpu = int(available / psutil.cpu_count())  # 790,140,415 = 0.8Gb/cpu
    mem_per_task = max(10_000_000, mem_per_cpu // working_overhead)  # min 10Mb, or 1 Gb / 10x = 100Mb
    n_tasks = math.ceil(file_length / mem_per_task)

    with path.open('rb') as fi:
        rawdata = fi.read(10000)
        encoding = chardet.detect(rawdata)['encoding']
    
    text_escape = TextEscape(delimiter=delimiter, qoute=text_qualifier, strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace)  # configure t.e.

    config = {
            "source":None,
            "destination":mem.path, 
            "table_key":None, 
            "columns":columns,
            "newline":newline, 
            "delimiter":delimiter, 
            "qoute":text_qualifier,
            "text_escape_openings":'', 
            "text_escape_closures":'',
            "encoding":encoding,
            "strip_leading_and_tailing_whitespace":strip_leading_and_tailing_whitespace
        }

    tasks = []
    with path.open('r', encoding=encoding) as fi:
        # task find chunk ...
        # Here is the problem in a nutshell:
        # --------------------------------------------------------
        # bs = "this is my \n text".encode('utf-16')
        # >>> bs
        # b'\xff\xfet\x00h\x00i\x00s\x00 \x00i\x00s\x00 \x00m\x00y\x00 \x00\n\x00 \x00t\x00e\x00x\x00t\x00'
        # >>> nl = "\n".encode('utf-16')
        # >>> nl in bs
        # False
        # >>> nl.decode('utf-16') in bs.decode('utf-16')
        # True
        # --------------------------------------------------------
        # This means we can't read the encoded stream to check if in contains a particular character.
        # We will need to decode it.
        # furthermore fi.tell() will not tell us which character we a looking at.
        # so the only option left is to read the file and split it in workable chunks.
        for line in fi:
            header_line = line
            line = line.rstrip('\n')
            break  # break on first
        fi.seek(0)
        headers = text_escape(line) # use t.e.
        if not columns:
            raise ValueError(f"No columns selected:\nAvailable columns: {headers}")

        if first_row_has_headers:    
            for name in columns:
                if name not in headers:
                    raise ValueError(f"column not found: {name}")
        else: # no headers.
            for index in columns:
                if index not in range(len(headers)):
                    raise IndexError(f"{index} out of range({len(headers)})")
            header_line = delimiter.join(str(i) for i in range(len(headers)))

        if not (isinstance(start, int) and start >= 0):
            raise ValueError("expected start as an integer >= 0")
        if not (isinstance(limit, int) and limit > 0):
            raise ValueError("expected limit as integer > 0")

        newlines = sum(1 for _ in fi)
        fi.seek(0)
        bytes_per_line = file_length / newlines
        lines_per_task = math.ceil(mem_per_task / bytes_per_line)

        if newlines <= start + (1 if first_row_has_headers else 0):  # Then start > end.
            t = Table()
            t.add_columns(*list(columns.keys()))
            t.save = True
            return t

        parts = []
        assert header_line != ""

        for ix, line in enumerate(fi, start=(-1 if first_row_has_headers else 0) ):
            if ix < start:
                # ix is -1 if the first row has headers, but header_line already has the first line.
                # ix is 0 if there are no headers, and if start is 0, the first row is added to parts.
                continue
            if ix >= start + limit:
                break

            parts.append(line)
            if ix!=0 and ix % lines_per_task == 0:
                p = path.parent / (path.stem + f'{ix}' + path.suffix)
                with p.open('w', encoding='utf-8') as fo:
                    parts.insert(0, header_line)
                    fo.write("".join(parts))
                parts.clear()
                tasks.append(Task( text_reader_task, **{**config, **{"source":str(p), "table_key":mem.new_id('/table'), 'encoding':'utf-8'}} ))

        if parts:  # any remaining parts at the end of the loop.
            p = path.parent / (path.stem + f'{ix}' + path.suffix)
            with p.open('w', encoding='utf-8') as fo:
                parts.insert(0, header_line)
                fo.write("".join(parts))
            parts.clear()
            config.update({"source":str(p), "table_key":mem.new_id('/table')})
            tasks.append(Task( text_reader_task, **{**config, **{"source":str(p), "table_key":mem.new_id('/table'), 'encoding':'utf-8'}} ))
    
    # execute the tasks
    with TaskManager(cpu_count=min(psutil.cpu_count(), n_tasks)) as tm:
        errors = tm.execute(tasks)   # I expects a list of None's if everything is ok.
        if any(errors):
            e = []
            for ix, err in enumerate(errors):
                if err is not None:
                    e.append("-" * 19)
                    e.append(f"Error in task {ix}:")
                    e.append(err)
            raise Exception("\n".join(e))
    
    # clean up the files
    for task in tasks:
        tmp = pathlib.Path(task.kwargs['source'])
        tmp.unlink()

    # consolidate the task results
    t = None
    for task in tasks:
        tmp = Table.load(path=mem.path, key=task.kwargs["table_key"])
        if t is None:
            t = tmp
        else:
            t += tmp
    t.save = True
    return t


def text_reader_task(source, destination, table_key, columns, 
    newline, delimiter=',', qoute='"', text_escape_openings='', text_escape_closures='', 
    strip_leading_and_tailing_whitespace=True, encoding='utf-8', timeout=10.0):
    """ PARALLEL TASK FUNCTION
    reads columnsname + path[start:limit] into hdf5.

    source: csv or txt file
    destination: available filename
    
    columns: column names or indices to import

    newline: '\r\n' or '\n'
    delimiter: ',' ';' or '|'
    text_escape_openings: str: default: "({[ 
    text_escape_closures: str: default: ]})" 
    strip_leading_and_tailing_whitespace: bool

    encoding: chardet encoding ('utf-8, 'ascii', ..., 'ISO-22022-CN')
    root: hdf5 root, cannot be the same as a column name.
    """
    if isinstance(source, str):
        source = pathlib.Path(source)
    if not isinstance(source, pathlib.Path):
        raise TypeError()
    if not source.exists():
        raise FileNotFoundError(f"File not found: {source}")

    if isinstance(destination, str):
        destination = pathlib.Path(destination)
    if not isinstance(destination, pathlib.Path):
        raise TypeError()

    if not isinstance(table_key, str):
        raise TypeError()

    if not isinstance(columns, dict):
        raise TypeError
    if not all(isinstance(name,str) for name in columns):
        raise ValueError()

    # declare CSV dialect.
    text_escape = TextEscape(text_escape_openings, text_escape_closures, qoute=qoute, delimiter=delimiter, 
                             strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace)

    with source.open('r', encoding=encoding) as fi:
        for line in fi:
            line = line.rstrip(newline)
            break  # break on first
        headers = text_escape(line)
        indices = {name: headers.index(name) for name in columns}        
        data = {h: [] for h in indices}
        text = fi.read()  # 1 IOP --> RAM.
        for line in text.split(newline):
            fields = text_escape(line)
            if fields == [""] or fields == []:
                break
            for header,index in indices.items():
                data[header].append(fields[index])

    # write out.
    t = 0.0
    start = time.process_time()
    while time.process_time() - start < timeout:
        try:    
            columns_refs = {}
            with h5py.File(destination, 'r+') as h5:
                for col_name, values in data.items():
                    new_page = Page(values)
                    dtype, shape = Page.layout([new_page])
                    
                    layout = h5py.VirtualLayout(shape=(shape,), dtype=dtype, maxshape=(None,), filename=destination)
                    
                    dset = h5[new_page.group]
                    vsource = h5py.VirtualSource(dset)
                    layout[0:len(dset)] = vsource
                    group_no = mem.new_id('/column')
                                        
                    h5.create_virtual_dataset(f"/column/{group_no}", layout=layout)
                    columns_refs[col_name] = group_no
                
                dset = h5.create_dataset(name=f"/table/{table_key}", dtype=h5py.Empty('f'))
                dset.attrs['columns'] = json.dumps(columns_refs)  
                dset.attrs['saved'] = True
            return
        except OSError:
            dt = random.randint(10,20)
            t+=dt
            time.sleep(dt/1000)
            
    raise OSError(f"couldn't write to disk (slept {t} msec")


file_readers = {
    'xlsx': excel_reader,
    'xls': excel_reader,
    'csv': text_reader,
    'txt': text_reader,
    'ods': ods_reader
}

