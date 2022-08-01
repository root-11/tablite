import math
import pathlib
import json
import sys
import itertools
import operator
import warnings
import logging

from collections import defaultdict
from multiprocessing import shared_memory

logging.getLogger('lml').propagate = False
logging.getLogger('pyexcel_io').propagate = False
logging.getLogger('pyexcel').propagate = False

log = logging.getLogger(__name__)


import chardet
import pyexcel
import pyperclip
from tqdm import tqdm
import numpy as np
import h5py
import psutil
from mplite import TaskManager, Task


PYTHON_EXIT = False  # exit handler so that Table.__del__ doesn't run into import error during exit.

def exiting():
    global PYTHON_EXIT
    PYTHON_EXIT = True

import atexit
atexit.register(exiting)


from tablite.memory_manager import MemoryManager, Page, Pages
from tablite.file_reader_utils import TextEscape, get_headers
from tablite.utils import summary_statistics, unique_name
from tablite import sortation
from tablite.groupby_utils import GroupBy, GroupbyFunction
from tablite.config import SINGLE_PROCESSING_LIMIT, TEMPDIR, H5_ENCODING
from tablite.datatypes import DataTypes


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
            mem.create_table(key=key, save=save, config=config)  # attrs. 'columns'
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
        if PYTHON_EXIT:
            return

        try:
            for key in list(self._columns):
                del self[key]
            mem.delete_table(self.group)
        except KeyError:
            log.info("Table.__del__ suppressed.")
        
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
            raise NotImplementedError()
    
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
                col_dset = h5[f"/column/{column_key}"]
                c._len = col_dset.attrs['length'] 
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
                dt, _ = types.popitem()
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
        if not self.columns:
            print("Empty Table")
            return

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
            split_after = 6
            t[tag] = [f"{i:,}" for i in range(7)] + [f"{i:,}" for i in range(n-7, n)]
            for name, col in self._columns.items():
                t[name] = [i for i in col[:7]] + [i for i in col[-7:]] 

        print(t.to_ascii(blanks=blanks,row_counts=tag, split_after=split_after))

    def _repr_html_(self):
        """ Ipython display compatible format
        https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display
        """
        start, end = "<div><table border=1>", "</table></div>"

        if not self.columns:
            return f"{start}<tr>Empty Table</tr>{end}"

        row_count_tags = ['#', '~', '*'] 
        cols = set(self.columns)
        for n,tag in itertools.product(range(1,6), row_count_tags):
            if n*tag not in cols:
                tag = n*tag
                break

        html = ["<tr>" + f"<th>{tag}</th>" +"".join( f"<th>{cn}</th>" for cn in self.columns) + "</tr>"]
        
        column_types = {}
        for name,col in self._columns.items():
            types = col.types()
            if len(types) == 1:
                dt, _ = types.popitem()
                column_types[name] = dt.__name__
            else:
                column_types[name] = 'mixed'

        html.append("<tr>" + f"<th>row</th>" +"".join( f"<th>{column_types[name]}</th>" for name in self.columns) + "</tr>")

        if len(self)<20:
            for ix, row in enumerate(self.rows):
                html.append( "<tr>" + f"<td>{ix}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
        else:
            t = Table()
            for name,col in self._columns.items():
                t[name] = [i for i in col[:7]] + [i for i in col[-7:]] 
            
            c = len(self)-7
            for ix, row in enumerate(t.rows):
                if ix < 7:
                    html.append( "<tr>" + f"<td>{ix}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
                if ix == 7: 
                    html.append( "<tr>" + f"<td>...</td>" + "".join(f"<td>...</td>" for _ in self._columns) + "</tr>")
                if ix >= 7:
                    html.append( "<tr>" + f"<td>{c}</td>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
                    c += 1

        return start + ''.join(html) + end

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

    def to_dict(self, row_count="row id", columns=None, slice_=None, start_on=1):
        """
        row_count: name of row counts. Default "row id". Use None to leave it out.
        columns: list of column names. Default is None == all columns.
        slice_: slice. Default is None == all rows.
        start_on: integer: first row (typically 0 or 1)
        """
        if slice_ is None:
            slice_  = slice(0, len(self))      
        assert isinstance(slice_, slice)
        
        if columns is None:
            columns = self.columns 
        if not isinstance(columns, list):
            raise TypeError("expected columns as list of strings")
        
        column_selection, own_cols = [], set(self.columns)
        for name in columns:
            if name in own_cols:
                column_selection.append(name)
            else:
                raise ValueError(f"column({name}) not found")
        
        cols = {}

        if row_count is not None:
            cols[row_count] = [i + start_on for i in range(*slice_.indices(len(self)))]

        for name in column_selection:
            new_name = unique_name(name, list_of_names=list(cols.keys()))
            col = self._columns[name]
            cols[new_name] = col[slice_].tolist()  # pure python objects. No numpy.
        d = {"columns": cols, "total_rows": len(self)}
        return d

    def as_json_serializable(self, row_count="row id", columns=None, slice_=None):
        args = row_count, columns, slice_
        d = self.to_dict(*args)
        for k,data in d['columns'].items():
            d['columns'][k] = [DataTypes.to_json(v) for v in data]  # deal with non-json datatypes.
        return d

    def to_json(self, *args, **kwargs):
        return json.dumps(self.as_json_serializable(*args, **kwargs))

    @classmethod
    def from_json(cls, jsn):
        d = json.loads(jsn)
        t = Table()
        for name, data in d['columns'].items():
            if not isinstance(name, str):
                raise TypeError(f"expect {name} as a string")
            if not isinstance(data, list):
                raise TypeError(f"expected {data} as list")
            t[name] = data
        return t

    def to_hdf5(self, path):
        """
        creates a copy of the table as hdf5
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

    def from_hdf5(self, path):
        """
        imports an exported hdf5 table.
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        
        t = Table()
        with h5py.File(path, 'r') as h5:
            for col_name in h5.keys():
                dset = h5[col_name]
                t[col_name] = dset[:]
        return t

    def to_sql(self):
        """
        generates ANSI-92 compliant SQL.
        """
        prefix = "Table"
        create_table = """CREATE TABLE {}{} ({})"""
        columns = []
        for name,col in self._columns.items():
            dtype = col.types()
            if len(dtype) == 1:
                dtype,_ = dtype.popitem()
                if dtype is int:
                    dtype = "INTEGER"
                elif dtype is float:
                    dtype = "REAL"
                else:
                    dtype = "TEXT"
            else:
                dtype = "TEXT"
            definition = f"{name} {dtype}"
            columns.append(definition)

        create_table = create_table.format(prefix, self.key, ", ".join(columns))
        
        # return create_table
        row_inserts = []
        for row in self.rows:
            row_inserts.append(str(tuple([i if i is not None else 'NULL' for i in row])))
        row_inserts = f"INSERT INTO {prefix}{self.key} VALUES " + ",".join(row_inserts) 
        return "begin; {}; {}; commit;".format(create_table, row_inserts)

    @classmethod
    def import_file(cls, path,  import_as, 
        newline='\n', text_qualifier=None,  delimiter=',', first_row_has_headers=True, columns=None, sheet=None, 
        start=0, limit=sys.maxsize, strip_leading_and_tailing_whitespace=True, encoding=None):
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
        encoding: str. Defaults to None (autodetect)

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
        if import_as.startswith("."):
            import_as = import_as[1:]
        reader = file_readers.get(import_as,None)
        if reader is None:
            raise ValueError(f"{import_as} is not in list of supported reader:\n{list(file_readers.keys())}")
        
        if not isinstance(strip_leading_and_tailing_whitespace, bool):
            raise TypeError()
        
        if columns is None:
            sample = get_headers(path)
            if import_as in {'csv', 'txt'}:
                columns = {k:'f' for k in sample[path.name][0]}
            elif sheet is not None:
                columns = sample[sheet][0]
            else:
                pass  # let it fail later.
        if not first_row_has_headers:
            columns = {str(i):'f' for i in range(len(columns))}

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
            'strip_leading_and_tailing_whitespace': strip_leading_and_tailing_whitespace,
            'encoding': encoding
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

            c2 = expression.get('column2',None) 
            if c2 is not None and c2 not in self.columns: 
                raise ValueError(f"no such column: {c2}")
            v2 = expression.get('value2', None)
            if v2 is not None and c2 is not None:
                raise ValueError("filter can only take 1 right expression element. Got 2.")
               
        if not isinstance(filter_type, str):
            raise TypeError()
        if not filter_type in {'all', 'any'}:
            raise ValueError(f"filter_type: {filter_type} not in ['all', 'any']")

        # the results are to be gathered here:
        arr = np.zeros(shape=(len(expressions), len(self)), dtype=bool)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        _ = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        
        # the task manager enables evaluation of a column per core,
        # which is assembled in the shared array.
        max_task_size = math.floor(SINGLE_PROCESSING_LIMIT / len(self.columns))  # 1 million fields per core (best guess!)
        
        filter_tasks = []
        for ix, expression in enumerate(expressions):
            for step in range(0, len(self), max_task_size):
                config = {'table_key':self.key, 'expression':expression, 
                          'shm_name':shm.name, 'shm_index':ix, 'shm_shape': arr.shape, 
                          'slice_':slice(step, min(step+max_task_size, len(self)))}
                task = Task(f=filter_evaluation_task, **config)
                filter_tasks.append(task)

        merge_tasks = []
        for step in range(0, len(self), max_task_size):
            config = {
                'table_key': self.key,
                'true_key': mem.new_id('/table'),
                'false_key': mem.new_id('/table'),
                'shm_name': shm.name, 'shm_shape': arr.shape,
                'slice_': slice(step, min(step+max_task_size,len(self)),1),
                'filter_type': filter_type
            }
            task = Task(f=filter_merge_task, **config)
            merge_tasks.append(task)

        n_cpus = min(max(len(filter_tasks),len(merge_tasks)), psutil.cpu_count())
        with TaskManager(n_cpus) as tm: 
            # EVALUATE 
            errs = tm.execute(filter_tasks)  # tm.execute returns the tasks with results, but we don't really care as the result is in the result array.
            if any(errs):
                raise Exception(errs)
            # MERGE RESULTS
            errs = tm.execute(merge_tasks)  # tm.execute returns the tasks with results, but we don't really care as the result is in the result array.
            if any(errs):
                raise Exception(errs)

        table_true, table_false = None, None
        for task in merge_tasks:
            tmp_true = Table.load(mem.path, key=task.kwargs['true_key'])
            if table_true is None:
                table_true = tmp_true
            elif len(tmp_true):
                table_true += tmp_true
            else:
                pass
                
            tmp_false = Table.load(mem.path, key=task.kwargs['false_key'])
            if table_false is None:
                table_false = tmp_false
            elif len(tmp_false):
                table_false += tmp_false
            else:
                pass
        return table_true, table_false
    
    def sort_index(self, sort_mode='excel', **kwargs):  
        """ 
        helper for methods `sort` and `is_sorted` 
        sort_mode: str: "alphanumeric", "unix", or, "excel"
        kwargs: sort criteria. See Table.sort()
        """
        logging.info(f"Table.sort_index running 1 core")  # TODO: This is single core code.

        if not isinstance(kwargs, dict):
            raise ValueError("Expected keyword arguments, did you forget the ** in front of your dict?")
        if not kwargs:
            kwargs = {c: False for c in self.columns}
        
        for k, v in kwargs.items():
            if k not in self.columns:
                raise ValueError(f"no column {k}")
            if not isinstance(v, bool):
                raise ValueError(f"{k} was mapped to {v} - a non-boolean")
        
        if sort_mode not in sortation.modes:
            raise ValueError(f"{sort_mode} not in list of sort_modes: {list(sortation.Sortable.modes.modes)}")

        rank = {i: tuple() for i in range(len(self))}  # create index and empty tuple for sortation.
        for key, reverse in tqdm(kwargs.items(), desc='creating sort index'):
            col = self._columns[key]
            assert isinstance(col, Column)
            ranks = sortation.rank(values=set(col[:].tolist()), reverse=reverse, mode=sort_mode)
            assert isinstance(ranks, dict)
            for ix, v in enumerate(col):
                rank[ix] += (ranks[v],)  # add tuple

        new_order = [(r, i) for i, r in rank.items()]  # tuples are listed and sort...
        rank.clear()  # free memory.
        new_order.sort()
        sorted_index = [i for _, i in new_order]  # new index is extracted.
        new_order.clear()
        return sorted_index

    def sort(self, sort_mode='excel', **kwargs):  
        """ Perform multi-pass sorting with precedence given order of column names.
        sort_mode: str: "alphanumeric", "unix", or, "excel"
        kwargs: 
            keys: columns, 
            values: 'reverse' as boolean.
            
        examples: 
        Table.sort('A'=False)  means sort by 'A' in ascending order.
        Table.sort('A'=True, 'B'=False) means sort 'A' in descending order, then (2nd priority) sort B in ascending order.
        """
        if len(self) * len(self.columns) < SINGLE_PROCESSING_LIMIT :  # the task is so small that multiprocessing doesn't make sense.
            sorted_index = self.sort_index(sort_mode=sort_mode, **kwargs)
            t = Table()
            for col_name, col in self._columns.items():  # this LOOP can be done with TaskManager
                data = list(col[:])
                t.add_column(col_name, data=[data[ix] for ix in sorted_index])
            return t
        else:
            arr = np.zeros(shape=(len(self), ), dtype=np.int64)
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)  # the co_processors will read this.
            sort_index = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            sort_index[:] = self.sort_index(sort_mode=sort_mode, **kwargs)

            tasks = []
            columns_refs = {}
            for name in self.columns:
                col = self[name]
                columns_refs[name] = d_key = mem.new_id('/column')
                tasks.append(Task(indexing_task, source_key=col.key, destination_key=d_key, shm_name_for_sort_index=shm.name, shape=arr.shape))

            with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
                errs = tm.execute(tasks)
                if any(errs):
                    msg = '\n'.join(errs)
                    raise Exception(f"multiprocessing error:{msg}")

            table_key = mem.new_id('/table')
            mem.create_table(key=table_key, columns=columns_refs)
            
            shm.close()
            shm.unlink()
            t = Table.load(path=mem.path, key=table_key)
            return t            

    def is_sorted(self, **kwargs):  
        """ Performs multi-pass sorting check with precedence given order of column names.
        nan_value: value used to represent non-sortable values such as None and np.nan during sort.
        **kwargs: sort criteria. See Table.sort()
        :return bool
        """
        logging.info(f"Table.is_sorted running 1 core")  # TODO: This is single core code.
        sorted_index = self.sort_index(**kwargs)
        if any(ix != i for ix, i in enumerate(sorted_index)):
            return False
        return True

    def _mp_compress(self, mask):
        """
        helper for `any` and `all` that performs compression of the table self according to mask
        using multiprocessing.
        """
        arr = np.zeros(shape=(len(self), ), dtype=bool)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)  # the co_processors will read this.
        compresssion_mask = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        compresssion_mask[:] = mask

        t = Table()
        tasks = []
        columns_refs = {}
        for name in self.columns:
            col = self[name]
            d_key = mem.new_id('/column')
            columns_refs[name] = d_key
            t = Task(compress_task, source_key=col.key, destination_key=d_key, shm_index_name=shm.name, shape=arr.shape)
            tasks.append(t)

        with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
            results = tm.execute(tasks)
            if any(r is not None for r in results):
                for r in results:
                    print(r)
                raise Exception("!")
        
        with h5py.File(mem.path, 'r+') as h5:
            table_key = mem.new_id('/table')
            dset = h5.create_dataset(name=f"/table/{table_key}", dtype=h5py.Empty('f'))
            dset.attrs['columns'] = json.dumps(columns_refs)  
            dset.attrs['saved'] = False
        
        shm.close()
        shm.unlink()
        t = Table.load(path=mem.path, key=table_key)
        return t

    def all(self, **kwargs):  
        """
        returns Table for rows where ALL kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you forget to add the ** in front of your dict?")
        if not all(k in self.columns for k in kwargs):
            raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in self.columns]}")

        ixs = None
        for k, v in kwargs.items():
            col = self._columns[k]
            if ixs is None:  # first header generates base set.
                if callable(v):
                    ix2 = {ix for ix, i in enumerate(col) if v(i)}
                else:
                    ix2 = {ix for ix, i in enumerate(col) if v == i}

            else:  # remaining headers reduce the base set.
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

        if len(self)*len(self.columns) < SINGLE_PROCESSING_LIMIT:
            t = Table()
            for col_name in self.columns:
                data = self[col_name]
                t[col_name] = [data[i] for i in ixs]
            return t
        else:
            mask =  np.array([True if i in ixs else False for i in range(len(self))],dtype=bool)
            return self._mp_compress(mask)

    def any(self, **kwargs):  
        """
        returns Table for rows where ANY kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you forget to add the ** in front of your dict?")

        ixs = set()
        for k, v in kwargs.items():
            col = self._columns[k]
            if callable(v):
                ix2 = {ix for ix, r in enumerate(col) if v(r)}
            else:
                ix2 = {ix for ix, r in enumerate(col) if v == r}
            ixs.update(ix2)

        if len(self) * len(self.columns) < SINGLE_PROCESSING_LIMIT:
            t = Table()
            for col_name in self.columns:
                data = self[col_name]
                t[col_name] = [data[i] for i in ixs]
            return t
        else:
            mask =  np.array([i in ixs for i in range(len(self))],dtype=bool)
            return self._mp_compress(mask)

    def groupby(self, keys, functions):  # TODO: This is slow single core code.
        """
        rows: column names for grouping as rows.
        columns: column names for grouping as columns.
        functions: list of column names and group functions (See GroupyBy)
        sum_on_rows: outputs group functions as extra rows if True, else as columns
        returns: table

        * NB: Column names can only be used once in rows & columns

        Example usage:
            from tablite import Table, GroupBy
            t = Table()
            t.add_column('date', data=[1,1,1,2,2,2])
            t.add_column('sku',  data=[1,2,3,1,2,3])
            t.add_column('qty',  data=[4,5,4,5,3,7])
            grp = t.groupby(rows=['sku'], functions=[('qty', GroupBy.Sum)])
            grp.show()
        """
        if len(set(keys)) != len(keys):
            duplicates = [k for k in keys if keys.count(k) > 1]
            s = "" if len(duplicates) > 1 else "s"
            raise ValueError(f"duplicate key{s} found across rows and columns: {duplicates}")

        if not isinstance(functions, list):
            raise TypeError(f"Expected functions to be a list of tuples. Got {type(functions)}")

        if not all(len(i) == 2 for i in functions):
            raise ValueError(f"Expected each tuple in functions to be of length 2. \nGot {functions}")

        if not all(isinstance(a, str) for a, _ in functions):
            L = [(a, type(a)) for a, _ in functions if not isinstance(a, str)]
            raise ValueError(f"Expected column names in functions to be strings. Found: {L}")

        if not all(issubclass(b, GroupbyFunction) and b in GroupBy.functions for _, b in functions):
            L = [b for _, b in functions if b not in GroupBy._functions]
            if len(L) == 1:
                singular = f"function {L[0]} is not in GroupBy.functions"
                raise ValueError(singular)
            else:
                plural = f"the functions {L} are not in GroupBy.functions"
                raise ValueError(plural)
        
        # 1. Aggregate data.
        aggregation_functions = defaultdict(dict)
        cols = keys + [col_name for col_name,_ in functions]
        seen,L = set(),[]
        for c in cols:
            if c not in seen:
                seen.add(c)
                L.append(c)
        for row in self.__getitem__(*L).rows:
            d = {col_name: value for col_name,value in zip(L, row)}
            key = tuple([d[k] for k in keys])
            agg_functions = aggregation_functions.get(key)
            if not agg_functions:
                aggregation_functions[key] = agg_functions =[(col_name, f()) for col_name, f in functions]
            for col_name, f in agg_functions:
                f.update(d[col_name])
        
        # 2. make dense table.
        cols = [[] for _ in cols]
        for key_tuple, funcs in aggregation_functions.items():
            for ix, key_value in enumerate(key_tuple):
                cols[ix].append(key_value)
            for ix, (_, f) in enumerate(funcs,start=len(keys)):
                cols[ix].append(f.value)
        
        new_names = keys + [f"{f.__name__}({col_name})" for col_name,f in functions]
        result = Table()
        for ix, (col_name, data) in enumerate(zip(new_names, cols)):
            revised_name = unique_name(col_name, result.columns)
            result[revised_name] = data            
        return result

    def pivot(self, rows, columns, functions, values_as_rows=True):
        if isinstance(rows, str):
            rows = [rows]
        if not all(isinstance(i,str) for i in rows)            :
            raise TypeError(f"Expected rows as a list of column names, not {[i for i in rows if not isinstance(i,str)]}")
        
        if isinstance(columns, str):
            columns = [columns]
        if not all(isinstance(i,str) for i in columns):
            raise TypeError(f"Expected columns as a list of column names, not {[i for i in columns if not isinstance(i, str)]}")

        if not isinstance(values_as_rows, bool):
            raise TypeError(f"expected sum_on_rows as boolean, not {type(values_as_rows)}")
        
        keys = rows + columns 
        assert isinstance(keys, list)

        grpby = self.groupby(keys, functions)

        if len(grpby) == 0:  # return empty table. This must be a test?
            return Table()
        
        # split keys to determine grid dimensions
        row_key_index = {}  
        col_key_index = {}

        r = len(rows)
        c = len(columns)
        g = len(functions)
        
        records = defaultdict(dict)

        for row in grpby.rows:
            row_key = tuple(row[:r])
            col_key = tuple(row[r:r+c])
            func_key = tuple(row[r+c:])
            
            if row_key not in row_key_index:
                row_key_index[row_key] = len(row_key_index)  # Y

            if col_key not in col_key_index:
                col_key_index[col_key] = len(col_key_index)  # X

            rix = row_key_index[row_key]
            cix = col_key_index[col_key]
            if cix in records:
                if rix in records[cix]:
                    raise ValueError("this should be empty.")
            records[cix][rix] = func_key
        
        result = Table()
        
        if values_as_rows:  # ---> leads to more rows.
            # first create all columns left to right

            n = r + 1  # rows keys + 1 col for function values.
            cols = [[] for _ in range(n)]
            for row, ix in row_key_index.items():
                for (col_name, f)  in functions:
                    cols[-1].append(f"{f.__name__}({col_name})")
                    for col_ix, v in enumerate(row):
                        cols[col_ix].append(v)

            for col_name, values in zip(rows + ["function"], cols):
                col_name = unique_name(col_name, result.columns)
                result[col_name] = values
            col_length = len(cols[0])
            cols.clear()
            
            # then populate the sparse matrix.
            for col_key, c in col_key_index.items():
                col_name = "(" + ",".join([f"{col_name}={value}" for col_name, value in zip(columns, col_key)]) + ")"
                col_name = unique_name(col_name, result.columns)
                L = [None for _ in range(col_length)]
                for r, funcs in records[c].items():
                    for ix, f in enumerate(funcs):
                        L[g*r+ix] = f
                result[col_name] = L
                
        else:  # ---> leads to more columns.
            n = r
            cols = [[] for _ in range(n)]
            for row in row_key_index:
                for col_ix, v in enumerate(row):
                    cols[col_ix].append(v)  # write key columns.
            
            for col_name, values in zip(rows, cols):
                result[col_name] = values
            
            col_length = len(row_key_index)

            # now populate the sparse matrix.
            for col_key, c in col_key_index.items():  # select column.
                cols, names = [],[]
                
                for f,v in zip(functions, func_key):
                    agg_col, func = f
                    col_name = f"{func.__name__}(" + ",".join([agg_col] + [f"{col_name}={value}" for col_name, value in zip(columns, col_key)]) + ")"
                    col_name = unique_name(col_name, result.columns)
                    names.append(col_name)
                    cols.append( [None for _ in range(col_length)] )
                for r, funcs in records[c].items():
                    for ix, f in enumerate(funcs):
                        cols[ix][r] = f
                for name,col in zip(names,cols):
                    result[name] = col

        return result

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
            Lcol, Rcol = self[L], other[R]
            if not set(Lcol.types()).intersection(set(Rcol.types())):
                raise TypeError(f"{L} is {Lcol.types()}, but {R} is {Rcol.types()}")

        if not isinstance(left_columns, list) or not left_columns:
            raise TypeError("left_columns (list of strings) are required")
        if any(column not in self.columns for column in left_columns):
            raise ValueError(f"Column not found: {[c for c in left_columns if c not in self.columns]}")

        if not isinstance(right_columns, list) or not right_columns:
            raise TypeError("right_columns (list or strings) are required")
        if any(column not in other.columns for column in right_columns):
            raise ValueError(f"Column not found: {[c for c in right_columns if c not in other.columns]}")
        # Input is now guaranteed to be valid.

    def join(self, other, left_keys, right_keys, left_columns, right_columns, kind='inner'):
        """
        short-cut for all join functions.
        kind: 'inner', 'left', 'outer', 'cross'
        """
        kinds = {
            'inner':self.inner_join,
            'left':self.left_join,
            'outer':self.outer_join,
            'cross': self.cross_join,
        }
        if kind not in kinds:
            raise ValueError(f"join type unknown: {kind}")
        f = kinds.get(kind,None)
        return f(self,other,left_keys,right_keys,left_columns,right_columns)
    
    def _sp_join(self, other, LEFT,RIGHT, left_columns, right_columns):
        """
        helper for single processing join
        """
        result = Table()
        for col_name in left_columns:
            col_data = self[col_name][:]
            result[col_name] = [col_data[k] if k is not None else None for k in LEFT]
        for col_name in right_columns:
            col_data = other[col_name][:]
            revised_name = unique_name(col_name, result.columns)
            result[revised_name] = [col_data[k] if k is not None else None for k in RIGHT]
        return result

    def _mp_join(self, other, LEFT,RIGHT, left_columns, right_columns):
        """ 
        helper for multiprocessing join
        """
        left_arr = np.zeros(shape=(len(LEFT)), dtype=np.int64)
        left_shm = shared_memory.SharedMemory(create=True, size=left_arr.nbytes)  # the co_processors will read this.
        left_index = np.ndarray(left_arr.shape, dtype=left_arr.dtype, buffer=left_shm.buf)
        left_index[:] = LEFT

        right_arr = np.zeros(shape=(len(RIGHT)), dtype=np.int64)
        right_shm = shared_memory.SharedMemory(create=True, size=right_arr.nbytes)  # the co_processors will read this.
        right_index = np.ndarray(right_arr.shape, dtype=right_arr.dtype, buffer=right_shm.buf)
        right_index[:] = RIGHT

        tasks = []
        columns_refs = {}
        for name in left_columns:
            col = self[name]
            columns_refs[name] = d_key = mem.new_id('/column')
            tasks.append(Task(indexing_task, source_key=col.key, destination_key=d_key, shm_name_for_sort_index=left_shm.name, shape=left_arr.shape))

        for name in right_columns:
            col = other[name]
            columns_refs[name] = d_key = mem.new_id('/column')
            tasks.append(Task(indexing_task, source_key=col.key, destination_key=d_key, shm_name_for_sort_index=right_shm.name, shape=right_arr.shape))

        with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
            results = tm.execute(tasks)
            
            if any(i is not None for i in results):
                for err in results:
                    if err is not None:
                        print(err)
                raise Exception("multiprocessing error.")
            
        with h5py.File(mem.path, 'r+') as h5:
            table_key = mem.new_id('/table')
            dset = h5.create_dataset(name=f"/table/{table_key}", dtype=h5py.Empty('f'))
            dset.attrs['columns'] = json.dumps(columns_refs)  
            dset.attrs['saved'] = False
        
        left_shm.close()
        left_shm.unlink()
        right_shm.close()
        right_shm.unlink()

        t = Table.load(path=mem.path, key=table_key)
        return t            

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

        left_index = self.index(*left_keys)
        right_index = other.index(*right_keys)
        LEFT,RIGHT = [],[]
        for left_key, left_ixs in left_index.items():
            right_ixs = right_index.get(left_key, (None,))
            for left_ix in left_ixs:
                for right_ix in right_ixs:
                    LEFT.append(left_ix)
                    RIGHT.append(right_ix)

        if len(LEFT) * len(left_columns + right_columns) < SINGLE_PROCESSING_LIMIT:
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns)
        else:  # use multi processing
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns)
            
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

        left_index = self.index(*left_keys)
        right_index = other.index(*right_keys)
        LEFT,RIGHT = [],[]
        for left_key, left_ixs in left_index.items():
            right_ixs = right_index.get(left_key, None)
            if right_ixs is None:
                continue
            for left_ix in left_ixs:
                for right_ix in right_ixs:
                    LEFT.append(left_ix)
                    RIGHT.append(right_ix)

        if len(LEFT) * len(left_columns + right_columns) < SINGLE_PROCESSING_LIMIT:
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns)       
        else:  # use multi processing
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns)

    def outer_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None):  # TODO: This is single core code.
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

        left_index = self.index(*left_keys)
        right_index = other.index(*right_keys)
        LEFT,RIGHT,RIGHT_UNUSED = [],[], set(right_index.keys())
        for left_key, left_ixs in left_index.items():
            right_ixs = right_index.get(left_key, (None,))
            for left_ix in left_ixs:
                for right_ix in right_ixs:
                    LEFT.append(left_ix)
                    RIGHT.append(right_ix)
                    RIGHT_UNUSED.discard(left_key)

        for right_key in RIGHT_UNUSED:
            for right_ix in right_index[right_key]:
                LEFT.append(None)
                RIGHT.append(right_ix)

        if len(LEFT) * len(left_columns + right_columns) < SINGLE_PROCESSING_LIMIT:
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns)
        else:  # use multi processing
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns)

    def cross_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None):
        """
        CROSS JOIN returns the Cartesian product of rows from tables in the join. 
        In other words, it will produce rows which combine each row from the first table 
        with each row from the second table
        """
        if left_columns is None:
            left_columns = list(self.columns)
        if right_columns is None:
            right_columns = list(other.columns)

        self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

        LEFT, RIGHT = zip(*itertools.product(range(len(self)), range(len(other))))
        if len(LEFT) < SINGLE_PROCESSING_LIMIT:
            return self._sp_join(other, LEFT, RIGHT, left_columns, right_columns)
        else:  # use multi processing
            return self._mp_join(other, LEFT, RIGHT, left_columns, right_columns)
        
    def lookup(self, other, *criteria, all=True):  # TODO: This is slow single core code.
        """ function for looking up values in `other` according to criteria in ascending order.
        :param: other: Table sorted in ascending search order.
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

        functions, left_criteria, right_criteria = [], set(), set()

        for left, op, right in criteria:
            left_criteria.add(left)
            right_criteria.add(right)
            if callable(op):
                pass  # it's a custom function.
            else:
                op = ops.get(op, None)
                if not callable(op):
                    raise ValueError(f"{op} not a recognised operator for comparison.")

            functions.append((op, left, right))
        left_columns = [n for n in left_criteria if n in self.columns]
        right_columns = [n for n in right_criteria if n in other.columns]

        results = []
        lru_cache = {}
        left = self.__getitem__(*left_columns)
        if isinstance(left, Column):
            tmp, left = left, Table()
            left[left_columns[0]] = tmp
        right = other.__getitem__(*right_columns)
        if isinstance(right, Column):
            tmp, right = right, Table()
            right[right_columns[0]] = tmp
        assert isinstance(left, Table)
        assert isinstance(right, Table)

        for row1 in tqdm(left.rows, total=self.__len__()):
            row1_tup = tuple(row1)
            row1d = {name: value for name, value in zip(left_columns, row1)}
            row1_hash = hash(row1_tup)

            match_found = True if row1_hash in lru_cache else False

            if not match_found:  # search.
                for row2ix, row2 in enumerate(right.rows):
                    row2d = {name: value for name, value in zip(right_columns, row2)}

                    evaluations = {op(row1d.get(left, left), row2d.get(right, right)) for op, left, right in functions}
                    # The evaluations above does a neat trick:
                    # as L is a dict, L.get(left, L) will return a value 
                    # from the columns IF left is a column name. If it isn't
                    # the function will treat left as a value.
                    # The same applies to right.
                    A = all and (False not in evaluations)
                    B = any and True in evaluations
                    if A or B:
                        match_found = True
                        lru_cache[row1_hash] = row2ix
                        break

            if not match_found:  # no match found.
                lru_cache[row1_hash] = None
            
            results.append(lru_cache[row1_hash])

        result = self.copy()
        if len(self) * len(other.columns) < SINGLE_PROCESSING_LIMIT:
            for col_name in other.columns:
                col_data = other[col_name][:]
                revised_name = unique_name(col_name, result.columns)
                result[revised_name] = [col_data[k] if k is not None else None for k in results]
            return result
        else:
            # 1. create shared memory array.
            right_arr = np.zeros(shape=(len(results)), dtype=np.int64)
            right_shm = shared_memory.SharedMemory(create=True, size=right_arr.nbytes)  # the co_processors will read this.
            right_index = np.ndarray(right_arr.shape, dtype=right_arr.dtype, buffer=right_shm.buf)
            right_index[:] = results
            # 2. create tasks
            tasks = []
            columns_refs = {}

            for name in other.columns:
                col = other[name]
                columns_refs[name] = d_key = mem.new_id('/column')
                tasks.append(Task(indexing_task, source_key=col.key, destination_key=d_key, shm_name_for_sort_index=right_shm.name, shape=right_arr.shape))

            # 3. let task manager handle the tasks
            with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
                errs = tm.execute(tasks)
                if any(errs):
                    raise Exception(f"multiprocessing error. {[e for e in errs if e]}")
            
            # 4. close the share memory and deallocate
            right_shm.close()
            right_shm.unlink()

            # 5. update the result table.
            with h5py.File(mem.path, 'r+') as h5:
                dset = h5[f"/table/{result.key}"]
                columns = dset.attrs['columns']
                columns.update(columns_refs)
                dset.attrs['columns'] = json.dumps(columns)  
                dset.attrs['saved'] = False
            
            # 6. reload the result table
            t = Table.load(path=mem.path, key=result.key)
            return t


class Column(object):
    def __init__(self, data=None, key=None) -> None:

        if key is None:
            self.key = mem.new_id('/column')
        else:
            self.key = key            
        self.group = f"/column/{self.key}"
        if key is None:
            self._len = 0
            if data is not None:
                self.extend(data)
        else:
            length, pages = mem.load_column_attrs(self.group)
            self._len = length
    
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
                    after = Pages([new_page])
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
                        after = Pages((Page(value),))
                    else:
                        last_page = before_slice[-1] 
                        if mem.get_ref_count(last_page) == 1:
                            last_page.extend(value)
                            after = before_slice
                        else:  # ref count > 1
                            new_page = Page(value)
                            after = before_slice + Pages([new_page])
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
                    after = Pages([new_page]) + before_slice
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
                        after = A + Pages([new_page]) + B  # new = old._getslice_(0,start) + list(value) + old._getslice_(stop,len(self.items))
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
                after = Pages([Page(new)])  # This may seem redundant, but is in fact is good as the user may 
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
                after = Pages([Page(new)])  # This may seem redundant, but is in fact is good as the user may 
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
        try:
            return np.unique(self.__getitem__())
        except TypeError:  # np arrays can't handle dtype='O':
            return np.array({i for i in self.__getitem__()})

    def histogram(self):
        """ 
        returns 2 arrays: unique elements and count of each element 
        
        example:
        >>> for item, counts in zip(self.histogram()):
        >>>     print(item,counts)
        """
        try:
            uarray, carray = np.unique(self.__getitem__(), return_counts=True)
        except TypeError:  # np arrays can't handle dtype='O':
            d = defaultdict(int)
            for i in self.__getitem__():
                d[i]+=1
            uarray, carray = [],[]
            for k,v in d.items():
                uarray.append(k), carray.append(v)
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
    return a.decode('utf-8') in b.decode('utf-8')  # TODO tested?


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

def filter_evaluation_task(table_key, expression, shm_name, shm_index, shm_shape, slice_):
    """
    multiprocessing tasks for evaluating Table.filter
    """
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

    columns = mem.mp_get_columns(table_key)
    if c1 is not None:
        column_key = columns[c1]
        dset_A = mem.get_data(f'/column/{column_key}', slice_)
    else:  # v1 is active:
        dset_A = np.array([v1] * (slice_.stop-slice_.start))
    
    if c2 is not None:
        column_key = columns[c2]
        dset_B = mem.get_data(f'/column/{column_key}', slice_)
    else:  # v2 is active:
        dset_B = np.array([v2] * (slice_.stop-slice_.start))

    existing_shm = shared_memory.SharedMemory(name=shm_name)  # connect
    result_array = np.ndarray(shm_shape, dtype=np.bool, buffer=existing_shm.buf)
    result_array[shm_index][slice_] = f(dset_A,dset_B)  # Evaluate
    existing_shm.close()  # disconnect


def filter_merge_task(table_key, true_key, false_key, shm_name, shm_shape, slice_, filter_type):
    """ 
    multiprocessing task for merging data after the filter task has been completed.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name)  # connect
    result_array = np.ndarray(shm_shape, dtype=np.bool, buffer=existing_shm.buf)
    mask_source = result_array

    if filter_type == 'any':
        true_mask = np.any(mask_source, axis=0)
    else:
        true_mask = np.all(mask_source, axis=0)
    true_mask = true_mask[slice_]
    false_mask = np.invert(true_mask)    
    
    # 2. load source
    columns = mem.mp_get_columns(table_key)

    true_columns, false_columns = {}, {}
    for col_name, column_key in columns.items():
        col = Column(key=column_key)
        slize = col[slice_]  # maybe use .tolist() ?
        true_values = slize[true_mask]
        if np.any(true_mask):
            true_columns[col_name] = mem.mp_write_column(true_values)
        false_values = slize[false_mask]
        if np.any(false_mask):
            false_columns[col_name] = mem.mp_write_column(false_values) 
        
    mem.mp_write_table(true_key, true_columns)
    mem.mp_write_table(false_key, false_columns)
    
    existing_shm.close()  # disconnect


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

    if columns is None:
        if first_row_has_headers:
            raise ValueError(f"no columns declared: \navailable columns: {[i[0] for i in ws.columns()]}")
        else:
            raise ValueError(f"no columns declared: \navailable columns: {[str(i) for i in range(len(ws.columns()))]}")

    config = {**kwargs, **{"first_row_has_headers":first_row_has_headers, "sheet":sheet, "columns":columns, 'start':start, 'limit':limit}}
    t = Table(save=True, config=json.dumps(config))
    for idx, column in enumerate(ws.columns()):
        
        if first_row_has_headers:
            header, start_row_pos = str(column[0]), max(1, start)
        else:
            header, start_row_pos = str(idx), max(0,start)

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
                strip_leading_and_tailing_whitespace=True, encoding=None, **kwargs):
    """
    **kwargs are excess arguments that are ignored.
    """
    # define and specify tasks.
    memory_usage_ceiling = 0.9
    free_memory = psutil.virtual_memory().free * memory_usage_ceiling
    free_memory_per_vcpu = free_memory / psutil.cpu_count()
    
    path = pathlib.Path(path)
       
    if encoding is None:
        with path.open('rb') as fi:
            rawdata = fi.read(10000)
            encoding = chardet.detect(rawdata)['encoding']       

    text_escape = TextEscape(delimiter=delimiter, qoute=text_qualifier, strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace)  # configure t.e.

    config = {
            "source":None,
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
        # task: find chunk ...
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

        if first_row_has_headers:
            if not columns:
                raise ValueError(f"No columns selected:\nAvailable columns: {headers}")
            for name in columns:
                if name not in headers:
                    raise ValueError(f"column not found: {name}")
        else: # no headers.
            valid_indices = [str(i) for i in range(len(headers))]
            if not columns:
                raise ValueError(f"No column index selected:\nAvailable columns: {valid_indices}")
            for index in columns:
                if index not in valid_indices:
                    raise IndexError(f"{index} not in valid range: {valid_indices}")
            header_line = delimiter.join(valid_indices) + '\n'

        if not (isinstance(start, int) and start >= 0):
            raise ValueError("expected start as an integer >= 0")
        if not (isinstance(limit, int) and limit > 0):
            raise ValueError("expected limit as integer > 0")

        try:
            newlines = sum(1 for _ in fi)  # log.info(f"{newlines} lines found.")
            fi.seek(0)
        except Exception as e:
            raise ValueError(f"file could not be read with encoding={encoding}\n{str(e)}")

        file_length = path.stat().st_size  # 9,998,765,432 = 10Gb
        bytes_per_line = math.ceil(file_length / newlines)
        working_overhead = 40  # MemoryError will occur if you get this wrong.
        
        total_workload = working_overhead * file_length
        cpu_count = psutil.cpu_count(logical=False) - 1
        memory_usage_ceiling = 0.9
        
        free_memory = int(psutil.virtual_memory().free * memory_usage_ceiling) - cpu_count * 20e6  # 20Mb per subproc.
        free_memory_per_vcpu = int(free_memory / cpu_count)  # 8 gb/ 16vCPU = 500Mb/vCPU
        
        if total_workload < free_memory_per_vcpu and total_workload < 10_000_000:  # < 1Mb --> use 1 vCPU
            lines_per_task = newlines
        else:  # total_workload > free_memory or total_workload > 10_000_000
            use_all_memory = free_memory_per_vcpu / (bytes_per_line * working_overhead)  # 500Mb/vCPU / (10 * 109 bytes / line ) = 458715 lines per task
            use_all_cores = newlines / (cpu_count)  # 8,000,000 lines / 16 vCPU = 500,000 lines per task
            lines_per_task = int(min(use_all_memory, use_all_cores))
        
        if not cpu_count * lines_per_task * bytes_per_line * working_overhead < free_memory:
            raise ValueError(f"{[cpu_count, lines_per_task , bytes_per_line , working_overhead , free_memory]}")
        assert newlines / lines_per_task >= 1
        
        if newlines <= start + (1 if first_row_has_headers else 0):  # Then start > end.
            t = Table()
            t.add_columns(*list(columns.keys()))
            t.save = True
            return t

        parts = []
        
        assert header_line != ""
        with tqdm(desc=f"splitting {path.name} for multiprocessing", total=newlines, unit="lines") as pbar:
            for ix, line in enumerate(fi, start=(-1 if first_row_has_headers else 0) ):
                if ix < start:
                    # ix is -1 if the first row has headers, but header_line already has the first line.
                    # ix is 0 if there are no headers, and if start is 0, the first row is added to parts.
                    continue
                if ix >= start + limit:
                    break

                parts.append(line)
                if ix!=0 and ix % lines_per_task == 0:
                    p = TEMPDIR / (path.stem + f'{ix}' + path.suffix)
                    with p.open('w', encoding=H5_ENCODING) as fo:
                        parts.insert(0, header_line)
                        fo.write("".join(parts))
                    pbar.update(len(parts))
                    parts.clear()
                    tasks.append( Task( text_reader_task, **{**config, **{"source":str(p), "table_key":mem.new_id('/table'), 'encoding':'utf-8'}} ) )
                    
            if parts:  # any remaining parts at the end of the loop.
                p = TEMPDIR / (path.stem + f'{ix}' + path.suffix)
                with p.open('w', encoding=H5_ENCODING) as fo:
                    parts.insert(0, header_line)
                    fo.write("".join(parts))
                pbar.update(len(parts))
                parts.clear()
                config.update({"source":str(p), "table_key":mem.new_id('/table')})
                tasks.append( Task( text_reader_task, **{**config, **{"source":str(p), "table_key":mem.new_id('/table'), 'encoding':'utf-8'}} ) )
        
        # execute the tasks
    # with TaskManager(cpu_count=min(cpu_count, len(tasks))) as tm:
    with TaskManager(cpu_count) as tm:
        errors = tm.execute(tasks)   # I expects a list of None's if everything is ok.
        
        # clean up the tmp source files, before raising any exception.
        for task in tasks:
            tmp = pathlib.Path(task.kwargs['source'])
            tmp.unlink()

        if any(errors):
            raise Exception("\n".join(e for e in errors if e))

    # consolidate the task results
    t = None
    for task in tasks:
        tmp = Table.load(path=mem.path, key=task.kwargs["table_key"])
        if t is None:
            t = tmp.copy()
        else:
            t += tmp
        tmp.save = False  # allow deletion of subproc tables.
    t.save = True
    return t


def text_reader_task(source, table_key, columns, 
    newline, delimiter=',', qoute='"', text_escape_openings='', text_escape_closures='', 
    strip_leading_and_tailing_whitespace=True, encoding='utf-8'):
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

    if not isinstance(table_key, str):
        raise TypeError()

    if not isinstance(columns, dict):
        raise TypeError
    if not all(isinstance(name,str) for name in columns):
        raise ValueError()

    # declare CSV dialect.
    text_escape = TextEscape(text_escape_openings, text_escape_closures, qoute=qoute, delimiter=delimiter, 
                             strip_leading_and_tailing_whitespace=strip_leading_and_tailing_whitespace)

    with source.open('r', encoding=encoding) as fi:  # --READ
        for line in fi:
            line = line.rstrip(newline)
            break  # break on first
        headers = text_escape(line)
        indices = {name: headers.index(name) for name in columns}        
        data = {h: [] for h in indices}
        for line in fi:  # 1 IOP --> RAM.
            fields = text_escape(line.rstrip('\n'))
            if fields == [""] or fields == []:
                break
            for header,index in indices.items():
                data[header].append(fields[index])
    # -- WRITE
    columns_refs = {}
    for col_name, values in data.items():
        values = DataTypes.guess(values)
        columns_refs[col_name] = mem.mp_write_column(values)
    mem.mp_write_table(table_key, columns=columns_refs)


file_readers = {
    'fods': excel_reader,
    'json': excel_reader,
    'html': excel_reader,
    'simple': excel_reader,
    'rst': excel_reader,
    'mediawiki': excel_reader,
    'xlsx': excel_reader,
    'xls': excel_reader,
    'xlsm': excel_reader,
    'csv': text_reader,
    'tsv': text_reader,
    'txt': text_reader,
    'ods': ods_reader
}


def indexing_task(source_key, destination_key, shm_name_for_sort_index, shape):
    """
    performs the creation of a column sorted by sort_index (shared memory object).
    source_key: column to read
    destination_key: column to write
    shm_name_for_sort_index: sort index' shm.name created by main.
    shape: shm array shape.

    *used by sort and all join functions.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name_for_sort_index)  # connect
    sort_index = np.ndarray(shape, dtype=np.int64, buffer=existing_shm.buf)

    data = mem.get_data(f'/column/{source_key}', slice(None)) # --- READ!
    values = [data[ix] for ix in sort_index]
    
    existing_shm.close()  # disconnect
    mem.mp_write_column(values, column_key=destination_key)  # --- WRITE!


def compress_task(source_key, destination_key, shm_index_name, shape):
    """
    compresses the source using boolean mask from shared memory

    source_key: column to read
    destination_key: column to write
    shm_name_for_sort_index: sort index' shm.name created by main.
    shape: shm array shape.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_index_name)  # connect
    mask = np.ndarray(shape, dtype=np.int64, buffer=existing_shm.buf)
    
    data = mem.get_data(f'/column/{source_key}', slice(None))  # --- READ!
    values = np.compress(mask, data)
    
    existing_shm.close()  # disconnect
    mem.mp_write_column(values, column_key=destination_key)  # --- WRITE!


