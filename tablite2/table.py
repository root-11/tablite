import math
import pathlib
from itertools import count
from collections import defaultdict
import json

import h5py  #https://stackoverflow.com/questions/27710245/is-there-an-analysis-speed-or-memory-usage-advantage-to-using-hdf5-for-large-arr?rq=1  
import chardet

from tablite2.settings import HDF5_IMPORT_ROOT, HDF5_CACHE_DIR, HDF5_CACHE_FILE, HDF5_TABLE_ROOT
from tablite2.column import Column, Page
from tablite2.task_manager import TaskManager, Task
from tablite2.utils import isiterable, normalize_slice, sha256sum
from tablite2.tasks.excel_reader import excel_reader,ods_reader
from tablite2.tasks.text_reader import text_reader, TextEscape, consolidate


class Table(object):
    tid = count(1)
    def __init__(self,key=None) -> None:
        self._columns = {}
        self.path = pathlib.Path(HDF5_CACHE_DIR) / HDF5_CACHE_FILE
        
        self.key = next(Table.tid) if key is None else key
        self.group = f"{HDF5_TABLE_ROOT}/Table-{self.key}"
        if key is not None:
            with h5py.File(self.path, mode='r') as h5:
                if key in h5[HDF5_TABLE_ROOT]:
                    dset = h5[self.group]
                    columns = json.loads(dset['columns'])
                    for name, column_id in columns.items():
                        self._columns[name] = Column(key=column_id)
        
    @property
    def columns(self):
        """
        returns the column names
        """
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
        
    def __getitem__(self, *keys):
        """
        table[text]  --> column   
        table[slice] --> n rows   
        table[int]  --> row
        table[args]  --> columns if text, rows if slice or int

        examples:

        table['a'] selects column 'a'
        table[:10] selects first 10 rows from all columns
        table['a','b', slice(3:20:2)] selects a slice from columns 'a' and 'b'
        table['b', 'a', 'a', 'c', 2:20:3]  selects column 'b' and 'c' and 'a' twice for a slice.
        """
        if isinstance(keys,str):  # it's a column
            return self._columns[keys]  # --> Column --> h5 virtual dataset
        elif isinstance(keys, slice):  # it's a slice of all columns.
            return {name:col[slice] for name, col in self._columns.items()}  
        elif isinstance(keys, int):  # it's a single row.
            return {name:col[keys] for name,col in self._columns.items()}
        elif isinstance(keys, tuple):  # it's a combination of columns (and slice)
            cols = [k for k in keys if k in self._columns]
            rows = [i for i in keys if isinstance(i,(slice,int))] 
            rows = slice(0,None,1) if not rows else rows[0]
            return {name: self._columns[name][rows] for name in cols}
        else:
            raise KeyError(keys)

    def __setitem__(self, key, value):
        """
        table[text] = Column(...)
        table[int]  = row
        table[slice] = rows
        table[args] ... depends: columns if text, rows if slice or int

        examples:

        table['a'] = [1,2,3]
        
        table[0] = {'a':44,'b':5.5,'c':60}  
        table[1] = (45,5.5,60)  
        table[2] = [45,5.5,60]

        table[:2] = {'a':[1,2], 'b':[1.1,2.2], 'c':[10,20]}   # dict w. lists/tuples
        table[:2] = [[1,2], [1.1,2.2], [10,20]]               # list of lists/tuples
        
        table['a','b'] =  [ [4, 5, 6], [1.1, 2.2, 3.3] ] --> key = ('A', 'B'), value = [ [4, 5, 6], [1.1, 2.2, 3.3] ]
        table['a','b', slice(2,None)] = [ [3], [4.4] ]
        table['a','b', 2:4] = [ [3, 2], [4.4, 5.5] ]
        """
        if isinstance(key, str):  
            self._columns[key] = Column(value)
        elif isinstance(key, (int, slice)) and len(value) == len(self._columns):
            if isinstance(value, dict):
                for name, col in self._columns.items():
                    col[key] = Column(value[name])
            elif isinstance(value, (list,tuple)):
                for ix, (name, col) in enumerate(self._columns.items()):
                    col[key] = Column(value[ix])
            else:
                raise NotImplementedError(f"{key}:{value}")            
        elif isinstance(key,tuple):  # it's a combination of columns (and slice)
            cols = [k for k in key if isinstance(k,str)]
            rows = [i for i in key if isinstance(i,(slice,int))] 
            
            rows = None if not rows else rows[0]  # rows = slice(0,None,1) if not rows else rows[0]
            if rows is None:  # it's an assignment.
                for ix, name in enumerate(cols):
                    if name in self._columns:
                        del self._columns[name]
                    if isinstance(value, dict):
                        data=value[name]
                    elif isinstance(value, (list,tuple)):
                        data=value[ix]
                    else:
                        raise TypeError(type(value))
                    self._columns[name] = Column(data)
            else:  # it's an update.
                for ix, name in enumerate(cols):
                    if name not in self._columns:
                        raise KeyError
                    c = self._columns[name]
                    if isinstance(value, dict):
                        data=value[name]
                    elif isinstance(value, (list,tuple)):
                        data=value[ix]
                    else:
                        raise TypeError(type(value))
                    c[rows] = data
        else:
            raise TypeError(f"Bad key type: {key}, expected str or tuple of strings")

    def add_row(*args,**kwargs):  
        pass
        # https://stackoverflow.com/questions/25655588/incremental-writes-to-hdf5-with-h5py?noredirect=1&lq=1
        # https://stackoverflow.com/questions/47072859/how-to-append-data-to-one-specific-dataset-in-a-hdf5-file-with-h5py

    def copy(self):
        """
        returns a copy of the table
        """
        t = Table()
        for name, c in self.columns.items():
            t[name] =  c.copy()
        return t

    def index(self, *keys):
        """ 
        Returns index on *keys columns as d[(key tuple, )] = {index1, index2, ...} 
        """
        if isinstance(keys, str) and keys in self._columns:
            col = self._columns[keys]
            assert isinstance(col, Column)
            return col.index()
        else:
            idx = defaultdict(set)
            generator = self.__getitem__(*keys)
            for ix, key in enumerate(generator.rows):
                idx[key].add(ix)
            return idx

    def __eq__(self,other):
        """
        enables comparison of self with other
        Example: TableA == TableB
        """
        if not isinstance(other, Table):
            raise TypeError(f"cannot compare {self.__class__.__name__} with {other.__class__.__name__}")
        # fast simple checks.
        try:  
            self.compare(other)
        except (TypeError, ValueError):
            return False

        if len(self) != len(other):
            return False

        # the longer check.
        for name, c in self.columns.items():
            c2 = other.columns[name]
            if c!=c2:  # exit at the earliest possible option.
                return False
        return True

    def compare(self,other):
        """
        compares the metadata of the two tables and raises on the first difference.
        """
        if not isinstance(other, Table):
            raise TypeError(f"cannot compare type {self.__class__.__name__} with {other.__class__.__name__}")
        for a, b in [[self, other], [other, self]]:  # check both dictionaries.
            for name, col in a.columns.items():
                if name not in b.columns:
                    raise ValueError(f"Column {name} not in other")
                col2 = b.columns[name]
                if col.dtype != col2.dtype:
                    raise ValueError(f"Column {name}.datatype different: {col.dtype}, {col2.dtype}")

    def __iadd__(self,other):
        """ 
        enables extension of self with data from other.
        Example: Table_1 += Table_2 
        """
        self.compare(other)
        for name,mc in self.columns.items():
            mc.extend(other.columns[name])
        return self

    def __add__(self,other):
        """
        returns the joint extension of self and other
        Example:  Table_3 = Table_1 + Table_2 
        """
        self.compare(other)
        t = self.copy()
        t += other
        return t

    def stack(self,other):  # TODO: Add tests.
        """
        returns the joint stack of tables
        Example:

        | Table A|  +  | Table B| = |  Table AB |
        | A| B| C|     | A| B| D|   | A| B| C| -|
                                    | A| B| -| D|
        """
        t = self.copy()
        for name,mc2 in other.columns.items():
            if name not in t.columns:
                t.add_column(name, data=[None] * len(self))
            mc = t.columns[name]
            mc.extend(mc2)
        for name, mc in t.columns.items():
            if name not in other.columns:
                mc.extend(data=[None]*len(other))
        return t

    def __mul__(self,other):
        """
        enables repetition of a table
        Example: Table_x_10 = table * 10
        """
        if not isinstance(other, int) and other > 0:
            raise TypeError(f"repetition of a table is only supported with positive integers")
        t = self.copy()
        for _ in range(1,other):  # from 1, because the copy is the first.
            t += self
        return t

    def sort(self, **kwargs):
        pass
    def join(self, other, left_keys, right_keys, left_columns, right_columns, join_type):
        pass
    def lookup(self, other, key, lookup_type):
        pass
    def groupby(self, keys, aggregates):
        pass
    def pivot(self, keys, aggregates, **kwargs):
        pass
    def view(self,*args,**kwargs):
        """
        Enables Excel type filtered view.
        """
        pass  # tbl.view( ('A','>',3), ('B','!=',None), ('A'*'B',">=",6), limit=50, sort_asc={'A':True, 'B':False})  
        # can be programmed as chained generators:
        # arg1 = (ix for ix,v in enumerate(self['A']) if v>3)
        # arg2 = (ix for ix in arg1 if self['B'][ix] != None)
        # arg3 = (ix for ix in arg2 if self['A'][ix] * self['B'] >= 6)
        # unsorted = [self[ix] for ix, _ in zip(arg3, range(limit))]  # self[ix] get's the row.
        # for key,asc in reversed(sort_asc.items():
        #   unsorted.sort(key,reversed=not asc)
        # t = self.copy(no_rows=True).extend(unsorted)
        # return t
    def show(self, *args, blanks=None, **kwargs):
        """
        sort AZ, ZA  <--- show only! the source doesn't change.
        unique values <--- applies on the column only.
        filter by condition [
            is empty, is not empty, 
            text {contains, does not contain, starts with, ends with, is exactly},
            date {is, is before, is after}
            value is {> >= < <= == != between, not between}
            formula (uses eval)
        ]
        filter by values [ unique values ]
        """  
        
        def to_ascii(table, blanks):
            """
            enables viewing in terminals
            returns the table as ascii string
            """
            widths = {}
            names = list(table.columns)
            for name,mc in table.columns.items():
                widths[name] = max([len(name), len(str(mc.dtype))] + [len(str(v)) for v in mc])

            def adjust(v, length):
                if v is None:
                    return str(blanks).ljust(length)
                elif isinstance(v, str):
                    return v.ljust(length)
                else:
                    return str(v).rjust(length)

            s = []
            s.append("+ " + "+".join(["=" * widths[n] for n in names]) + " +")
            s.append("| " + "|".join([n.center(widths[n], " ") for n in names]) + " |")
            s.append("| " + "|".join([str(table.columns[n].dtype).center(widths[n], " ") for n in names]) + " |")
            s.append("+ " + "+".join(["-" * widths[n] for n in names]) + " +")
            for row in table.rows:
                s.append("| " + "|".join([adjust(v, widths[n]) for v, n in zip(row, names)]) + " |")
            s.append("+ " + "+".join(["=" * widths[h] for h in names]) + " +")
            return "\n".join(s)
           
        slc = slice(0,min(len(self),20),1) if len(self) < 20 else None  # default slice 
        for arg in args:  # override by user defined slice (if provided)
            if isinstance(arg, slice):
                slc = slice(*normalize_slice(len(self), arg))
            break
        
        if slc:
            t = Table()
            t.add_column('#', data=[str(i) for i in range(slc.start, slc.stop, slc.step)])
            for n, mc in self.columns.items():
                t.add_column(n,data=[str(i) for i in mc[slc] ])
        else:
            t,n = Table(), len(self)
            t.add_column('#', data=[str(i) for i in range(7)] + ["..."] + [str(i) for i in range(n-7, n)])
            for name, mc in self.columns.items():
                data = [str(i) for i in mc[:7]] + ["..."] + [str(i) for i in mc[-7:]]
                t.add_column(name, data)
                
        print(to_ascii(t, blanks))

    @classmethod
    def import_file(cls, path,
        import_as, newline='\n', text_qualifier=None,
        delimiter=',', first_row_has_headers=True, columns=None, sheet=None):
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

        if not isinstance(import_as,str) and import_as in ['csv','txt','xlsx']:
            raise ValueError(f"{import_as} is not supported")
        
        # check the inputs.
        if import_as in {'xlsx'}:
            return excel_reader(path, sheet_name=sheet)
            
        if import_as in {'ods'}:
            return ods_reader(path, sheet_name=sheet)

        if import_as in {'csv', 'txt'}:
            h5 = pathlib.Path(str(path) + '.hdf5')
            if h5.exists():
                with h5py.File(h5,'r') as f:  # Create file, truncate if exists
                    stored_config = json.loads(f.attrs['config'])
            else:
                stored_config = {}

            file_length = path.stat().st_size  # 9,998,765,432 = 10Gb
            config = {
                'import_as': import_as,
                'path': str(path),
                'filesize': file_length,  # if this changes - re-import.
                'delimiter': delimiter,
                'columns': columns, 
                'newline': newline,
                'first_row_has_headers': first_row_has_headers,
                'text_qualifier': text_qualifier
            }

            skip = False
            for k,v in config.items():
                if stored_config.get(k,None) != v:
                    skip = False
                    break  # set skip to false and exit for loop.
                else:
                    skip = True
            if skip:
                print(f"file already imported as {h5}")  
                return Table.load_file(h5)  # <---- EXIT 1.

            # Ok. File doesn't exist, has been changed or it's a new import config.
            with path.open('rb') as fi:
                rawdata = fi.read(10000)
                encoding = chardet.detect(rawdata)['encoding']
            
            text_escape = TextEscape(delimiter=delimiter, qoute=text_qualifier)  # configure t.e.

            with path.open('r', encoding=encoding) as fi:
                for line in fi:
                    line = line.rstrip('\n')
                    break  # break on first
                headers = text_escape(line) # use t.e.
                
                if first_row_has_headers:    
                    for name in columns:
                        if name not in headers:
                            raise ValueError(f"column not found: {name}")
                else:
                    for index in columns:
                        if index not in range(len(headers)):
                            raise IndexError(f"{index} out of range({len(headers)})")

            with h5py.File(h5,'w') as f:  # Create file, truncate if exists
                f.attrs['config'] = json.dumps(config)

            with TaskManager() as tm:
                working_overhead = 5  # random guess. Calibrate as required.
                mem_per_cpu = tm.chunk_size_per_cpu(file_length * working_overhead)
                mem_per_task = mem_per_cpu // working_overhead  # 1 Gb / 10x = 100Mb
                n_tasks = math.ceil(file_length / mem_per_task)
                
                text_reader_task_config = {
                    "source":path, 
                    "destination":h5, 
                    "columns":columns, 
                    "newline":newline, 
                    "delimiter":delimiter, 
                    "first_row_has_headers":first_row_has_headers,
                    "qoute":text_qualifier,
                    "text_escape_openings":'', "text_escape_closures":'',
                    "start":None, "limit":mem_per_task,
                    "encoding":encoding
                }
                
                tasks = []
                for i in range(n_tasks):
                    # add task for each chunk for working
                    text_reader_task_config['start'] = i * mem_per_task
                    task = Task(f=text_reader, **text_reader_task_config)
                    tasks.append(task)
                
                tm.execute(tasks)
                # Merging chunks in hdf5 into single columns
                consolidate(h5)  # no need to task manager as this is done using
                # virtual layouts and virtual datasets.

            return Table.load_file(h5)  # <---- EXIT 2.
    @classmethod
    def load_file(cls, path):
        """
        enables loading of imported HDF5 file. 
        Import assumes that columns are in the HDF5 root as "/{column name}"

        :path: pathlib.Path
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"expected pathlib.Path, got {type(path)}")
        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")
        if not path.name.endswith(".hdf5"):
            raise TypeError(f"expected .hdf5 file, not {path.name}")
        
        # read the file and create managed columns
        # no need for task manager as this is just fetching metadata.
        t = Table()
        with h5py.File(path,'r') as f:  
            for name in f.keys():
                if name == HDF5_IMPORT_ROOT:
                    continue
                page = Page(path=path, route=f"/{name}")
                t[name] = Column(page)

        return t
    @classmethod
    def from_json(path):
        pass            
    @classmethod
    def from_tablite_cache(cls, path=None):
        pass  # reconstructs tables from tablite cache: returns dict with {table keys: Table(), .... }
        if path is None:
            path = pathlib.Path(HDF5_CACHE_DIR) / HDF5_CACHE_FILE
        
        with h5py.File(path,mode='r') as h5:
            tables = []
            for table_key in h5[HDF5_TABLE_ROOT]:
                table = Table(key=table_key)
                tables.append(table)
        return tables

