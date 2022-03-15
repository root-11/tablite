import hashlib
import math
import pathlib
import time
import weakref
import numpy as np
import h5py
import io
import traceback
import queue
import tqdm
import multiprocessing
import chardet
from graph import Graph
from itertools import count
from collections import deque
from multiprocessing import cpu_count, shared_memory
from ..tablite.file_reader_utils import text_escape
from ..tablite.datatypes import DataTypes

class TaskManager(object):
    def __init__(self) -> None:
        self.tq = multiprocessing.Queue()  # task queue for workers.
        self.rq = multiprocessing.Queue()  # result queue for workers.
        self.pool = []
        self.tasks = {}  # task register for progress tracking
        self.results = {}  # result register for progress tracking
    
    def add(self, task):
        if not isinstance(task, dict):
            raise TypeError
        if not 'id' in task:
            raise KeyError("expect task to have id, to preserve order")
        task_id = task['id']
        if task_id in self.tasks:
            raise KeyError(f"task {task_id} already in use.")
        self.tasks[task_id] = task
        self.tq.put(task)
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): # signature requires these, though I don't use them.
        self.stop()
        self.tasks.clear()
        self.results.clear()

    def start(self):
        self.pool = [Worker(name=str(i), tq=self.tq, rq=self.rq) for i in range(2)]
        for p in self.pool:
            p.start()
        while not all(p.is_alive() for p in self.pool):
            time.sleep(0.01)

    def execute(self):
        t = tqdm.tqdm(total=len(self.tasks), unit='task')
        while len(self.tasks) != len(self.results):
            try:
                result = self.rq.get_nowait()
                self.results[result['id']] = result
            except queue.Empty:
                time.sleep(0.01)
            t.update(len(self.results))
        t.close()
        
    def stop(self):
        self.tq.put("stop")
        while all(p.is_alive() for p in self.pool):
            time.sleep(0.01)
        print("all workers stopped")
        self.pool.clear()
  

class Worker(multiprocessing.Process):
    def __init__(self, name, tq, rq):
        super().__init__(group=None, target=self.update, name=name, daemon=False)
        self.exit = multiprocessing.Event()
        self.tq = tq  # workers task queue
        self.rq = rq  # workers result queue
        self._quit = False
        print(f"Worker-{self.name}: ready")
                
    def update(self):
        while True:
            try:
                task = self.tq.get_nowait()
            except queue.Empty:
                time.sleep(0.01)
                continue
            
            if task == "stop":
                print(f"Worker-{self.name}: stop signal received.")
                self.tq.put_nowait(task)  # this assures that everyone gets it.
                self.exit.set()
                break
            error = ""
            try:
                exec(task['script'])
            except Exception as e:
                f = io.StringIO()
                traceback.print_exc(limit=3, file=f)
                f.seek(0)
                error = f.read()
                f.close()

            self.rq.put({'id': task['id'], 'handled by': self.name, 'error': error})


class MemoryManager(object):
    registry = weakref.WeakValueDictionary()  # The weakref presents blocking of garbage collection.
    # Two usages:
    # {Object ID: Object} for all objects.
    # {sha256hash: Object} for DataBlocks (used to prevent duplication of data in memory.)
    lru_tracker = {}  # {DataBlockId: process_time, ...}
    map = Graph()  # Documents relations between Table, Column & Datablock.
    process_pool = None
    tasks = None

    @classmethod
    def __del__(cls):
        # Use `import gc; del TaskManager; gc.collect()` to delete the TaskManager class.
        # shm.close()
        # shm.unlink()
        pass

    @classmethod
    def register(cls, obj):  # used at __init__
        cls.registry[id(obj)] = obj
        cls.lru_tracker[id(obj)] = time.process_time()
    @classmethod
    def deregister(cls, obj):  # used at __del__
        cls.registry.pop(id(obj),None)
        cls.lru_tracker.pop(id(obj),None)
    @classmethod
    def link(cls,a,b):
        a = cls.registry[a] if isinstance(a, int) else a
        b = cls.registry[b] if isinstance(b, int) else b

        cls.map.add_edge(id(a),id(b))
        if isinstance(b, DataBlock):
            cls.map.add_node(id(b), b)  # keep the datablocks!
    @classmethod
    def unlink(cls, a,b):
        cls.map.del_edge(id(a), id(b))
        if isinstance(b, DataBlock):
            if cls.map.in_degree(id(b)) == 0:  # remove the datablock if in-degree == 0
                cls.map.del_node(id(b))  
    @classmethod
    def unlink_tree(cls, a):
        """
        removes `a` and descendents of `a` if descendant does not have other incoming edges.
        """
        nodes = deque([id(a)])
        while nodes:
            n1 = nodes.popleft()
            if cls.map.in_degree(n1) == 0:
                for n2 in cls.map.nodes(from_node=n1):
                    nodes.append(n2)
                cls.map.del_node(n1)
    @classmethod
    def get(cls, node_id):
        """
        maintains lru_tracker
        returns DataBlock
        """
        cls.lru_tracker[node_id] = time.process_time()  # keep the lru tracker up to date.
        return cls.map.node(node_id)
    @classmethod
    def inventory(cls):
        """
        returns printable overview of the registered tables, managed columns and datablocs.
        """
        c = count()
        node_count = len(cls.map.nodes())
        if node_count == 0:
            return "no nodes" 
        n = math.ceil(math.log10(node_count))+2
        L = []
        d = {id(obj): name for name,obj in globals().copy().items() if isinstance(obj, (Table))}

        for node_id in cls.map.nodes(in_degree=0):
            name = d.get(node_id, "Table")
            obj = cls.registry.get(node_id,None)
            if obj:
                columns = [] if obj is None else list(obj.columns.keys())
                L.append(f"{next(c)}|".zfill(n) + f" {name}, columns = {columns}, registry id: {node_id}")
                for name, mc in obj.columns.items():
                    L.append(f"{next(c)}|".zfill(n) + f" └─┬ {mc.__class__.__name__} \'{name}\', length = {len(mc)}, registry id: {id(mc)}")
                    for i, block_id in enumerate(mc.order):
                        block = cls.map.node(block_id)
                        L.append(f"{next(c)}|".zfill(n) + f"   └── {block.__class__.__name__}-{i}, length = {len(block)}, registry id: {block_id}")
        return "\n".join(L)


class DataBlock(object):  # DataBlocks are IMMUTABLE!
    def __init__(self, data):
        MemoryManager.register(self)
        self._on_disk = False
        
        if not isinstance(data, np.ndarray):
            raise TypeError("Expected a numpy array.")       

        self._shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        self._address = self._shm.name

        self._data = np.ndarray(data.shape, dtype=data.dtype, buffer=self._shm.buf)
        self._data = data[:] # copy the source data into the shm
        self._len = len(data)
        self._dtype = data.dtype
        # np in shared memory: address = psm_21467_46075
	    # h5 on disk: address = path/to/hdf5.h5/table_name/column_name/block_name
    @property
    def address(self):
        if self._shm is not None:
            return (self._data.shape, self._dtype, self._address)
        else:
            raise NotImplementedError("h5 isn't implemented yet.")
    @property
    def data(self):
        return self._data  # TODO: Needs to cope with data being on HDF5.
    @data.setter
    def data(self, value):
        raise AttributeError("DataBlock.data is immutable.")
    def __len__(self) -> int:
        return self._len
    def __iter__(self):
        raise AttributeError("Use vectorised functions on DataBlock.data instead of __iter__")
    def __del__(self):
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception as e:
                print(e)

        if self._on_disk:
            pass   # del file
        else:
            pass  # close shm
        MemoryManager.deregister(self)


def intercept(A,B):
    """
    A: range
    B: range
    returns range as intercept of ranges A and B.
    """
    assert isinstance(A, range)
    if A.step < 0: # turn the range around
        A = range(A.stop, A.start, abs(A.step))
    assert isinstance(B, range)
    if B.step < 0:  # turn the range around
        B = range(B.stop, B.start, abs(B.step))
    
    boundaries = [A.start, A.stop, B.start, B.stop]
    boundaries.sort()
    a,b,c,d = boundaries
    if [A.start, A.stop] in [[a,b],[c,d]]:
        return range(0) # then there is no intercept
    # else: The inner range (subset) is b,c, limited by the first shared step.
    A_start_steps = math.ceil((b - A.start) / A.step)
    A_start = A_start_steps * A.step + A.start

    B_start_steps = math.ceil((b - B.start) / B.step)
    B_start = B_start_steps * B.step + B.start

    if A.step == 1 or B.step == 1:
        start = max(A_start,B_start)
        step = B.step if A.step==1 else A.step
        end = c
    else:
        intersection = set(range(A_start, c, A.step)).intersection(set(range(B_start, c, B.step)))
        if not intersection:
            return range(0)
        start = min(intersection)
        end = max(c, max(intersection))
        intersection.remove(start)
        step = min(intersection) - start
    
    return range(start, end, step)


class ManagedColumn(object):  # Behaves like an immutable list.
    def __init__(self) -> None:
        MemoryManager.register(self)
        self.order = []  # strict order of datablocks.
        self.dtype = None
       
    def __len__(self):
        return sum(len(MemoryManager.get(block_id)) for block_id in self.order)

    def __del__(self):
        MemoryManager.unlink_tree(self)
        MemoryManager.deregister(self)

    def __iter__(self):
        for block_id in self.order:
            datablock = MemoryManager.get(block_id)
            assert isinstance(datablock, DataBlock)
            for value in datablock.data:
                yield value

    def _normalize_slice(self, item=None):
        """
        helper: transforms slice into range inputs
        returns start,stop,step
        """
        if item is None:
            item = slice(0, len(self), 1)
        assert isinstance(item, slice)
        
        stop = len(self) if item.stop is None else item.stop
        start = 0 if item.start is None else len(self) + item.start if item.start < 0 else item.start
        start, stop = min(start,stop), max(start,stop)
        step = 1 if item.step is None else item.step

        return start, stop, step
            
    def __getitem__(self, item):
        """
        returns a value or a ManagedColumn (slice).
        """
        if isinstance(item, slice):
            mc = ManagedColumn()  # to be returned.

            r = range(*self._normalize_slice(item))
            page_start = 0
            for block_id in self.order:
                if page_start > r.stop:
                    break
                block = MemoryManager.get(block_id)
                if page_start + len(block) < r.start:
                    page_start += len(block)
                    continue

                if r.step==1:
                    if r.start <= page_start and page_start + len(block) <= r.stop: # then we take the whole block.
                        mc.extend(block.data)
                        page_start += len(block)
                        continue
                    else:
                        pass # the block doesn't match.
                
                block_range = range(page_start, page_start+len(block))
                intercept_range = intercept(r,block_range)  # very effective!
                if len(intercept_range)==0:  # no match.
                    page_start += len(block)
                    continue

                x = {i for i in intercept_range}  # TODO: Candidate for TaskManager.
                mask = np.array([i in x for i in block_range])
                new_block = block.data[np.where(mask)]
                mc.extend(new_block)
                page_start += len(block)

            return mc
            
        elif isinstance(item, int):
            page_start = 0
            for block_id in self.order:
                block = MemoryManager.get(block_id)
                page_end = len(block)
                if page_start <= item < page_end:
                    ix = item-page_start
                    return block.data[ix]
        else:
            raise KeyError(f"{item}")

    def blocks(self):
        """ returns the address of all blocks. """
        return [MemoryManager.get(block_id).address for block_id in self.order]           

    def _dtype_check(self, other):
        assert isinstance(other, (np.ndarray, ManagedColumn))
        if self.dtype is None:
            self.dtype = other.dtype
        elif self.dtype != other.dtype:
            raise TypeError(f"the column expects {self.dtype}, but received {other.dtype}.")
        else:
            pass

    def extend(self, data):
        """
        extends ManagedColumn with data
        """
        if isinstance(data, ManagedColumn):  # It's data we've seen before.
            self._dtype_check(data)

            self.order.extend(data.order[:])
            for block_id in data.order:
                MemoryManager.link(self, block_id)
            
        else:  # It's supposedly new data.
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            self._dtype_check(data)

            m = hashlib.sha256()  # let's check if it really is new data...
            m.update(data.data.tobytes())
            sha256sign = m.hexdigest()
            if sha256sign in MemoryManager.registry:  # ... not new!
                block = MemoryManager.registry.get(sha256sign)
            else:  # ... it's new!
                block = DataBlock(data)
                MemoryManager.registry[sha256sign] = block
            # ok. solved. Now create links.
            self.order.append(id(block))
            MemoryManager.link(self, block)  # Add link from Column to DataBlock
    
    def append(self, value):
        """
        Disabled. Append items is slow. Use extend on a batch instead
        """
        raise AttributeError("Append items is slow. Use extend on a batch instead")
    

class Table(object):
    def __init__(self) -> None:
        MemoryManager.register(self)
        self.columns = {}
    
    def __len__(self) -> int:
        if not self.columns:
            return 0
        else:
            return max(len(mc) for mc in self.columns.values())
    
    def __del__(self):
        MemoryManager.unlink_tree(self)  # columns are automatically garbage collected.
        MemoryManager.deregister(self)

    def __getitem__(self, items):
        """
        Enables selection of columns and rows
        Examples: 

            table['a']   # selects column 'a'
            table[:10]   # selects first 10 rows from all columns
            table['a','b', slice(3:20:2)]  # selects a slice from columns 'a' and 'b'
            table['b', 'a', 'a', 'c', 2:20:3]  # selects column 'b' and 'c' and 'a' twice for a slice.

        returns values in same order as selection.
        """
        if isinstance(items, slice):
            names, slc = list(self.columns.keys()), items
        else:        
            names, slc = [], slice(len(self))
            for i in items:
                if isinstance(i,slice):
                    slc = i
                elif isinstance(i, str) and i in self.columns:
                    names.append(i)
                else:
                    raise KeyError(f"{i} is not a slice and not in column names")
        if not names:
            raise ValueError("No columns?")
        
        t = Table()
        for name in names:
            mc = self.columns[name]
            t.add_column(name, data=mc[slc])
        return t       

    def __delitem__(self, item):
        if isinstance(item, str):
            mc = self.columns[item]
            del self.columns[item]
            MemoryManager.unlink(self, mc)
            MemoryManager.unlink_tree(mc)
        elif isinstance(item, slice):
            raise AttributeError("Tables are immutable. Create a new table using filter or using an index")

    def del_column(self, name):  # alias for summetry to add_column
        self.__delitem__(name)
 
    def add_column(self, name, data):
        if name in self.columns:
            raise ValueError(f"name {name} already used")
        mc = ManagedColumn()
        mc.extend(data)
        self.columns[name] = mc
        MemoryManager.link(self, mc)  # Add link from Table to Column
    
    def __eq__(self, other) -> bool:  # TODO: Add tests for each condition.
        """
        enables comparison of self with other
        Example: TableA == TableB
        """
        if not isinstance(other, Table):
            a, b = self.__class__.__name__, other.__class__.__name__
            raise TypeError(f"cannot compare {a} with {b}")
        
        # fast simple checks.
        try:  
            self.compare(other)
        except (TypeError, ValueError):
            return False

        if len(self) != len(other):
            return False

        # the longer check.
        for name, mc in self.columns.items():
            mc2 = other.columns[name]
            if any(a!=b for a,b in zip(mc,mc2)):  # exit at the earliest possible option.
                return False
        return True

    def __iadd__(self, other):
        """ 
        enables extension of self with data from other.
        Example: Table_1 += Table_2 
        """
        self.compare(other)
        for name,mc in self.columns.items():
            mc.extend(other.columns[name])
        return self

    def __add__(self, other):
        """
        returns the joint extension of self and other
        Example:  Table_3 = Table_1 + Table_2 
        """
        self.compare(other)
        t = self.copy()
        for name,mc in other.columns.items():
            mc2 = t.columns[name]
            mc2.extend(mc)
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
                t.add_column(name, data=[None] * len(mc2))
            mc = t.columns[name]
            mc.extend(mc2)
        for name, mc in t.columns.items():
            if name not in other.columns:
                mc.extend(data=[None]*len(other))
        return t

    def __mul__(self, other):
        """
        enables repetition of a table
        Example: Table_x_10 = table * 10
        """
        if not isinstance(other, int):
            raise TypeError(f"repetition of a table is only supported with integers, not {type(other)}")
        t = self.copy()
        for _ in range(other-1):  # minus, because the copy is the first.
            t += self
        return t

    def compare(self,other):
        """
        compares the metadata of the two tables and raises on the first difference.
        """
        if not isinstance(other, Table):
            a, b = self.__class__.__name__, other.__class__.__name__
            raise TypeError(f"cannot compare type {b} with {a}")
        for a, b in [[self, other], [other, self]]:  # check both dictionaries.
            for name, col in a.columns.items():
                if name not in b.columns:
                    raise ValueError(f"Column {name} not in other")
                col2 = b.columns[name]
                if col.dtype != col2.dtype:
                    raise ValueError(f"Column {name}.datatype different: {col.dtype}, {col2.dtype}")
                # if col.allow_empty != col2.allow_empty:  // TODO!
                #     raise ValueError(f"Column {name}.allow_empty is different")

    def copy(self):
        """
        returns a copy of the table
        """
        t = Table()
        for name,mc in self.columns.items():
            t.add_column(name,mc)
        return t

    def rename_column(self, old, new):
        """
        renames existing column from old name to new name
        """
        if old not in self.columns:
            raise ValueError(f"'{old}' doesn't exist. See Table.columns ")
        if new in self.columns:
            raise ValueError(f"'{new}' is already in use.")

    def __iter__(self):
        """
        Disabled. Users should use Table.rows or Table.columns
        """
        raise AttributeError("use Table.rows or Table.columns")

    def __setitem__(self, key, value):
        raise TypeError(f"Use Table.add_column")

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

    def show(self, blanks=None, format='ascii'):
        """
        prints a _preview_ of the table.
        
        blanks: string to replace blanks (None is default) when shown.
        formats: 
          - 'ascii' --> ASCII (see also self.to_ascii)
          - 'md' --> markdown (see also self.to_markdown)
          - 'html' --> HTML (see also self.to_html)

        """
        converters = {
            'ascii': self.to_ascii,
            'md': self.to_markdown,
            'html': self.to_html
        }
        converter = converters.get(format, None)
        
        if converter is None:
            raise ValueError(f"format={format} not in known formats: {list(converters)}")

        if len(self) < 20:
            t = Table()
            t.add_column('#', data=[str(i) for i in range(len(self))])
            for n,mc in self.columns.items():
                t.add_column(n,data=[str(i) for i in mc])
            print(converter(t,blanks))

        else:
            t,mc,n = Table(), ManagedColumn(), len(self)
            data = [str(i) for i in range(7)] + ["..."] + [str(i) for i in range(n-7, n)]
            mc.extend(data)
            t.add_column('#', data=mc)
            for name, mc in self.columns.items():
                data = [str(i) for i in mc[:7]] + ["..."] + [str(i) for i in mc[-7:]]
                t.add_column(name, data)

        print(converter(t, blanks))

    @staticmethod
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
        # s.append("| " + "|".join([str(table.columns[n].allow_empty).center(widths[n], " ") for n in names]) + " |")
        s.append("+ " + "+".join(["-" * widths[n] for n in names]) + " +")
        for row in table.rows:
            s.append("| " + "|".join([adjust(v, widths[n]) for v, n in zip(row, names)]) + " |")
        s.append("+ " + "+".join(["=" * widths[h] for h in names]) + " +")
        return "\n".join(s)

    @staticmethod
    def to_markdown(table, blanks):
        widths = {}
        names = list(table.columns)
        for name, mc in table.columns.items():
            widths[name] = max([len(name)] + [len(str(i)) for i in mc])
        
        def adjust(v, length):
            if v is None:
                return str(blanks).ljust(length)
            elif isinstance(v, str):
                return v.ljust(length)
            else:
                return str(v).rjust(length)

        s = []
        s.append("| " + "|".join([n.center(widths[n], " ") for n in names]) + " |")
        s.append("| " + "|".join(["-" * widths[n] for n in names]) + " |")
        for row in table.rows:
            s.append("| " + "|".join([adjust(v, widths[n]) for v, n in zip(row, names)]) + " |")
        return "\n".join(s)

    @staticmethod
    def to_html(table, blanks):
        raise NotImplemented("coming soon!")
           
    @classmethod
    def import_file(cls, path, config):
        """
        reads path and imports 1 or more tables as hdf5
        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"expected pathlib.Path, got {type(path)}")
        if not path.exists():
            raise ValueError(f"file not found: {path}")

        if not config['import_as']:
            raise ValueError("import_as is empty")
            
        config = {
            "import_as": "csv",
            'newline': b'\n', 'text_qualifier':None,
            'delimiter':b',', 'first_row_headers':True,
            'columns': {}, 'chunk_size': 1_000_000
        }.update(config.copy())
        
        with TaskManager() as tm:
            pass

    def load_file(cls, path):
        raise NotImplementedError("coming soon!")


# FILE READER UTILS 2.0 ----------------------------
def text_escape(s, escape=b'"', sep=b';'):
    """ escapes text marks using a depth measure. """
    assert isinstance(s, bytes)
    word, words = [], tuple()
    in_esc_seq = False
    for ix, c in enumerate(s):
        if c == escape:
            if in_esc_seq:
                if ix+1 != len(s) and s[ix + 1] != sep:
                    word.append(c)
                    continue  # it's a fake escape.
                in_esc_seq = False
            else:
                in_esc_seq = True
            if word:
                words += (b"".join(word),)
                word.clear()
        elif c == sep and not in_esc_seq:
            if word:
                words += (b"".join(word),)
                word.clear()
        else:
            word.append(c)

    if word:
        if word:
            words += (b"".join(word),)
            word.clear()
    return words


def detect_seperator(bytes):
    """
    After reviewing the logic in the CSV sniffer, I concluded that all it
    really does is to look for a non-text character. As the separator is
    determined by the first line, which almost always is a line of headers,
    the text characters will be utf-8,16 or ascii letters plus white space.
    This leaves the characters ,;:| and \t as potential separators, with one
    exception: files that use whitespace as separator. My logic is therefore
    to (1) find the set of characters that intersect with ',;:|\t' which in
    practice is a single character, unless (2) it is empty whereby it must
    be whitespace.
    """
    seps = {b',', b'\t', b';', b':', b'|', b'\t'}.intersection(bytes)
    if not seps:
        if b" " in bytes:
            return b" "
    else:
        frq = [(bytes.count(i), i) for i in seps]
        frq.sort(reverse=True)  # most frequent first.
        return {k:v for k,v in frq}


def interactive_file_import():
    """
    interactive file import for commandline use.
    """
    def get_file():
        while 1:
            path = input("| file path > ")
            path = pathlib.Path(path)
            if not path.exists():
                print(f"| file not found > {path}")
                continue
            print(f"| importing > {p.absolute()}")
            r = input("| is this correct? [Y/N] >")
            if "Y" in r.upper():
                return p
    
    def csv_menu(path):
        config = {
            'newline': None,
            'text_qualifier':None,
            'delimiter':None,
            'first_row_headers':None,
            'columns': {}
        }
        with path.open('rb', encoding=encoding) as fi:
            # read first 5 rows
            s, new_line_counter = [], 5
            while new_line_counter > 0:
                c = fi.read(1)
                s.append(c)
                if c == '\n':
                    new_line_counter -= 1
        s = s.join()
        encoding = chardet.detect(s)
        print(f"| file encoding > {encoding}")
        config.update(encoding)
        if b'\r\n' in s:
            newline = b'\r\n'
        else:
            newline = b'\n'
        print(f"| newline > {newline} ")

        config['newline'] = newline

        for line in s.split(newline):
            print(line)
        
        config['first_row_headers'] = "y" in input("| first row are column names? > Y/N").lower()

        text_escape_char = input("| text escape character > ")
        config['text escape']= None if text_escape_char=="" else text_escape_char
        
        seps = detect_seperator(s)
        
        print(f"| delimiters detected > {seps}")

        delimiter = input("| delimiter ? >")
        config['delimiter'] = delimiter

        def preview(s, text_esc, delimit):
            array = []
            for line in s.split(newline):
                fields = text_escape(line, text_esc, delimit)
                array.append(fields)
            return array

        def is_done(array):
            print("| rotating the first 5 lines > ")
            print("\n".join(map(" | ".join, zip(*(array)))))
            answer = input("| does this look right? Y/N >")
            return answer.lower(), array
        
        array = preview(s,text_escape_char,delimiter)
        if "n" in is_done(array):
            print("| update keys and values. Enter blank key when done.")
            while 1:
                key = input("| key > ")
                if key == "":
                    array = preview(s,text_escape_char,delimiter)
                    if "y" in is_done(array):
                        break
                value = input("| value > ")
                config[key]= value

        cols = input(f"| select columns ? [all/some] > ")
        config['columns'] = {}
        if "a" in cols:
            pass  # empty dict means import all .
            for ix, colname in enumerate(array[0]):
                sample = [array[i][ix] for i in range(1,len(array))]
                datatype = DataTypes.guess(*sample)
                for dtype in DataTypes.types: # strict order.
                    if dtype in datatype:
                        break
                config['columns'][colname] = dtype

        else:
            print("| Enter columns to keep. Enter blank when done.")
            while 1: 
                key = input("| column name > ")
                ix = array[0].index(key)
                if ix == -1:
                    print(f"| {key} not found.")
                    continue
                sample = [array[i][ix] for i in range(1,6)]
                datatype = DataTypes.guess(*sample)
                print(f"| > {datatypes}")
                for dtype in DataTypes.types: # strict order.
                    if dtype in datatype:
                        break
                config['columns'][colname] = dtype
                while 1:
                    guess = input(f"| is {dtype} correct datatype ?\n| > Enter accepts / type name if different >")
                    if guess == "":
                        break
                    elif guess in [str(t) for t in DataTypes.types]:
                        config['columns'][colname] = dtype
                    else:
                        print(f"| {guess} > No such type.")
                
        print(f"| using config > \n{config}")
        return config       

    def get_config(p):
        assert isinstance(p, pathlib.Path)
        ext = path.name.split(".")[-1].lower()
        if ext == "csv":
            config = csv_menu(p)
        elif ext == "xlsx":
            config = xlsx_menu()
        elif ext == "txt":
            config = txt_menu(p)
        elif ext == '.h5':
            print("| {p} is already imported!")
        else:
            print(f"no method for .{ext}'s")
        
    try:
        p = get_file()
        config = get_config(p)
        new_p = Table.import_file(p,config)
        t = Table.load_file(new_p)
        return t
    except KeyboardInterrupt:
        return

    
# TESTS -----------------
def test_range_intercept():
    A = range(500,700,3)
    B = range(520,700,3)
    C = range(10,1000,30)

    assert intercept(A,C) == range(0)
    assert set(intercept(B,C)) == set(B).intersection(set(C))

    A = range(500_000, 700_000, 1)
    B = range(10, 10_000_000, 1000)

    assert set(intercept(A,B)) == set(A).intersection(set(B))

    A = range(500_000, 700_000, 1)
    B = range(10, 10_000_000, 1)

    assert set(intercept(A,B)) == set(A).intersection(set(B))


def test_basics():
    # creating a tablite incrementally is straight forward:
    table1 = Table()
    assert len(table1) == 0
    table1.add_column('A', data=[1,2,3])
    assert 'A' in table1.columns
    assert len(table1) == 3

    table1.add_column('B', data=['a','b','c'])
    assert 'B' in table1.columns
    assert len(table1) == 3

    table2 = table1.copy()

    table3 = table1 + table2
    assert len(table3) == len(table1) + len(table2)
    for row in table3.rows:
        print(row)

    tables = 3
    managed_columns_per_table = 2 
    datablocks = 2

    assert len(MemoryManager.map.nodes()) == tables + (tables * managed_columns_per_table) + datablocks
    assert len(MemoryManager.map.edges()) == tables * managed_columns_per_table + 8 - 2  # the -2 is because of double reference to 1 and 2 in Table3
    assert len(table1) + len(table2) + len(table3) == 3 + 3 + 6

    # delete table
    assert len(MemoryManager.map.nodes()) == 11, "3 tables, 6 managed columns and 2 datablocks"
    assert len(MemoryManager.map.edges()) == 12
    del table1  # removes 2 refs to ManagedColumns and 2 refs to DataBlocks
    assert len(MemoryManager.map.nodes()) == 8, "removed 1 table and 2 managed columns"
    assert len(MemoryManager.map.edges()) == 8 
    # delete column
    del table2['A']
    assert len(MemoryManager.map.nodes()) == 7, "removed 1 managed column reference"
    assert len(MemoryManager.map.edges()) == 6

    print(MemoryManager.inventory())

    del table3
    del table2
    assert len(MemoryManager.map.nodes()) == 0
    assert len(MemoryManager.map.edges()) == 0


def test_slicing():
    table1 = Table()
    base_data = list(range(10_000))
    table1.add_column('A', data=base_data)
    table1.add_column('B', data=[v*10 for v in base_data])
    table1.add_column('C', data=[-v for v in base_data])
    start = time.time()
    big_table = table1 * 100
    print(f"it took {time.time()-start} to extend a table to {len(big_table)} rows")
    start = time.time()
    _ = big_table.copy()
    print(f"it took {time.time()-start} to copy {len(big_table)} rows")
    
    a_preview = big_table['A', 'B', 1_000:900_000:700]
    for row in a_preview[3:15:3].rows:
        print(row)
    a_preview.show(format='ascii')
    a_preview.show(format='md')


def test_multiprocessing():
    # Create shared_memory array for workers to access.
    a = np.array([1, 1, 2, 3, 5, 8])
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]

    task = {
        'id':1,
        'address': shm.name, 'type': 'shm', 
        'dtype': a.dtype, 'shape': a.shape, 
        'script': f"""# from multiprocessing import shared_memory - is already imported.
existing_shm = shared_memory.SharedMemory(name='{shm.name}')
c = np.ndarray((6,), dtype=np.{a.dtype}, buffer=existing_shm.buf)
c[-1] = 888
existing_shm.close()
"""}

    tasks = [task]
    for i in range(4):
        task2 = task.copy()
        task2['id'] = 2+i
        task2['script'] = f"""existing_shm = shared_memory.SharedMemory(name='{shm.name}')
c = np.ndarray((6,), dtype=np.{a.dtype}, buffer=existing_shm.buf)
c[{i}] = 111+{i}  # DIFFERENT!
existing_shm.close()
time.sleep(0.1)  # Added delay to distribute the few tasks amongst the workers.
"""
        tasks.append(task2)
    
    with TaskManager() as tm:
        for task in tasks:
            tm.add(task)
        tm.execute()

        for v in tm.results.items():
            print(v)

    # Alternative "low level usage":
    # tm = TaskManager()
    # tm.add(task)
    # tm.start()
    # tm.execute()
    # tm.stop()
    print(b, f"assertion that b[-1] == 888 is {b[-1] == 888}")  
    print(b, f"assertion that b[0] == 111 is {b[0] == 111}")  
    
    shm.close()
    shm.unlink()

def test_file_importer():
    p = r"d:\remove_duplicates.csv"
    assert pathlib.Path(p).exists(), "?"
    config = {'encoding': 'ascii', 'delimiter': ',', 'text escape': '"'}
    Table.import_file(p,config)


# def fx2(address):
#     shape, dtype, name = address
#     existing_shm = shared_memory.SharedMemory(name=name)
#     c = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
#     result = 2*c
#     existing_shm.close()  # emphasising that shm is no longer used.
#     return result

# def test_shm():   # <---- This test was failing, because pool couldn't connect to shm.
#     table1 = Table()
#     assert len(table1) == 0
#     table1.add_column('A', data=[1,2,3])  # block1
#     table1['A'].extend(data=[4,4,8])  # block2
#     table1['A'].extend(data=[8,9,10])  # block3
#     assert [v for v in table1['A']] == [1,2,3,4,4,8,8,9,10]
#     blocks = table1['A'].blocks()

#     print(blocks)
#     result = MemoryManager.process_pool.map(fx2, blocks)
#     print(result)
    


# drop datablock to hdf5
# load datablack from hdf5

# import is read csv to hdf5.
# - one file = one hdf5 file.
# - one column = one hdf5 table.

# materialize table

# multiproc
# - create join as tasklist.

# memory limit
# set task manager memory limit relative to using psutil
# update LRU cache based on access.

if __name__ == "__main__":
    for k,v in globals().copy().items():
        if k.startswith("test") and callable(v):
            v()

