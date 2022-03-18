import hashlib
import math
import pathlib
import time
import weakref
import numpy as np
import h5py
import io
import os
import json
import traceback
import queue
from tqdm import tqdm, trange
import multiprocessing
import chardet
import psutil
from graph import Graph
from itertools import count
from abc import ABC
from collections import deque
from multiprocessing import cpu_count, shared_memory


class TaskManager(object):
    memory_usage_ceiling = 0.9  # 90%

    def __init__(self) -> None:
        self._memory = psutil.virtual_memory().available
        self._cpus = psutil.cpu_count()
        self._disk_space = psutil.disk_usage('/').free
        
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
        with tqdm(total=len(self.tasks), unit='task') as pbar:
            while len(self.tasks) != len(self.results):
                try:
                    result = self.rq.get_nowait()
                    self.results[result['id']] = result
                except queue.Empty:
                    time.sleep(0.01)
                pbar.update(len(self.results))
                
    def stop(self):
        self.tq.put("stop")
        while all(p.is_alive() for p in self.pool):
            time.sleep(0.01)
        print("all workers stopped")
        self.pool.clear()
  
    def chunk_size_per_cpu(self, working_memory_required):  # 39,683,483,123 = 39 Gb.
        if working_memory_required < psutil.virtual_memory().free:
            chunk_size = math.ceil(working_memory_required / self._cpus)
        else:
            memory_ceiling = int(psutil.virtual_memory().total * self.memory_usage_ceiling)
            memory_used = psutil.virtual_memory().used
            available = memory_ceiling - memory_used  # 6,321,123,321 = 6 Gb
            mem_per_cpu = int(available / self._cpus)  # 790,140,415 = 0.8Gb/cpu
            chunk_size = math.ceil((working_memory_required) / mem_per_cpu)
        return chunk_size

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
    cache_path = pathlib.Path(os.getcwd()) / 'tablite_cache.h5'
    
    # cache_file = h5py.File(cache_path, mode='a') # 'a': Read/write if exists, create otherwise
        
    @classmethod
    def reset(cls):
        """
        enables user to erase any cached hdf5 data.
        Useful for testing where the user wants a clean working directory.

        Example:
        # new test case:
        >>> import MemoryManager
        >>> MemoryManager.reset()
        >>> ... start on testcase ...
        """
        cls.cache_file = h5py.File(cls.cache_path, mode='w')  # 'w' Create file, truncate if exists
        cls.cache_file.close()
        for obj in list(cls.registry.values()):
            del obj

    @classmethod
    def __del__(cls):
        # Use `import gc; del MemoryManager; gc.collect()` to delete the MemoryManager class.
        # shm.close()
        # shm.unlink()
        cls.cache_file.unlink()  # no loss of data.

    @classmethod
    def register(cls, obj):  # used at __init__
        assert isinstance(obj, MemoryManagedObject)
        cls.registry[obj.mem_id] = obj
        cls.lru_tracker[obj.mem_id] = time.process_time()

    @classmethod
    def deregister(cls, obj):  # used at __del__
        assert isinstance(obj, MemoryManagedObject)
        cls.registry.pop(obj.mem_id, None)
        cls.lru_tracker.pop(obj.mem_id, None)

    @classmethod
    def link(cls, a, b):
        assert isinstance(a, MemoryManagedObject)
        assert isinstance(b, MemoryManagedObject)
        
        cls.map.add_edge(a.mem_id, b.mem_id)
        if isinstance(b, DataBlock):
            # as the registry is a weakref, I need a hard ref to the datablocks!
            cls.map.add_node(b.mem_id, b)  # <-- Hard ref.

    @classmethod
    def unlink(cls, a, b):
        assert isinstance(a, MemoryManagedObject)
        assert isinstance(b, MemoryManagedObject)

        cls.map.del_edge(a.mem_id, b.mem_id)
        if isinstance(b, DataBlock):
            if cls.map.in_degree(b.mem_id) == 0:  # remove the datablock if in-degree == 0
                cls.map.del_node(b.mem_id)

    @classmethod
    def unlink_tree(cls, a):
        """
        removes `a` and descendents of `a` if descendant does not have other incoming edges.
        """
        assert isinstance(a,MemoryManagedObject)
        
        nodes = deque([a.mem_id])
        while nodes:
            n1 = nodes.popleft()
            if cls.map.in_degree(n1) == 0:
                for n2 in cls.map.nodes(from_node=n1):
                    nodes.append(n2)
                cls.map.del_node(n1)  # removes all edges automatically.
    @classmethod
    def get(cls, mem_id):
        """
        maintains lru_tracker
        returns DataBlock
        """
        cls.lru_tracker[mem_id] = time.process_time()  # keep the lru tracker up to date.
        return cls.map.node(mem_id)
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
        d = {obj.mem_id: name for name,obj in globals().copy().items() if isinstance(obj, (Table))}

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


class MemoryManagedObject(ABC):
    """
    Base Class for Memory Managed Objects
    """
    _ids = count()
    def __init__(self, mem_id) -> None:
        self._mem_id = mem_id
        MemoryManager.register(self)
    @property
    def mem_id(self):
        return self._mem_id
    @mem_id.setter
    def mem_id(self,value):
        raise AttributeError("mem_id is immutable")
    def __del__(self):
        MemoryManager.deregister(self)


class DataBlock(MemoryManagedObject):  # DataBlocks are IMMUTABLE!
    hdf5 = 'hdf5'
    shm = 'shm'

    def __init__(self, mem_id, data=None, address=None):
        """
        mem_id: sha256sum of the datablock. Why? Because of storage.

        All datablocks are either imported or created at runtime.
        Imported datablocks reside in HDF and are immutable (otherwise you'd mess 
        with the initial state). They are stored in the users filetree.
        Datablocks created at runtime reside in the MemoryManager's 

        kwargs: (only one required)
        data: np.array
        address: tuple: 
            shared memory address: str: "psm_21467_46075"
            h5 address: tuple: ("path/to/hdf5.h5", "/table_name/column_name/sha256sum")
        """
        super().__init__(mem_id=mem_id)

        if (data is not None and address is None):
            self._type = self.shm
                        
            if not isinstance(data, np.ndarray):
                raise TypeError("Expected a numpy array.")       

            self._handle = shared_memory.SharedMemory(create=True, size=data.nbytes)
            self._address = self._handle.name  # Example: "psm_21467_46075"

            self._data = np.ndarray(data.shape, dtype=data.dtype, buffer=self._handle.buf)  
            self._data = data[:]  # copy the source data into the shm (source may be a np.view)
            self._len = len(data)
            self._dtype = data.dtype.name

        elif (address is not None and data is None):
            self._type = self.hdf5
            if not isinstance(address, tuple) or len(address)!=2:
                raise TypeError("Expected pathlib.Path and h5 dataset address")
            path, address = address
            # Address is expected as:
            # if import: ("path/to/hdf5.h5", "/table_name/column_name/sha256sum")
            # if use_disk: ("path/to/MemoryManagers/tmp/dir", "/sha256sum")

            if not isinstance(path, pathlib.Path):
                raise TypeError(f"expected pathlib.Path, not {type(path)}")
            if not path.exists():
                raise ValueError(f"file not found: {path}")
            if not isinstance(address,str):
                raise TypeError(f"expected address as str, but got {type(address)}")
            if not address.startswith('/'):
                raise ValueError(f"address doesn't start at root.")
            
            self._handle = h5py.File(path,'r')  # imported data is immutable.
            self._address = address

            self._data = self._handle[address]            
            self._len = len(self._data)
            self._dtype = self.data.dtype.name
        else:
            raise ValueError("Either address or data must be None")

    @property
    def use_disk(self):
        return self._type == self.hdf5
    
    def use_disk(self, value):
        if value is False:
            if self._type == self.shm:  
                return  # nothing to do. Already in shm mode.
            else:  # load from hdf5 to shm
                data = self._data[:]
                self._handle = shared_memory.SharedMemory(create=True, size=data.nbytes)
                self._address = self._handle.name
                self._data = np.ndarray(data.shape, dtype=data.dtype, buffer=self._handle.buf)
                self._data = data[:] # copy the source data into the shm
                self._type = self.shm
                return
        else:  # if value is True:
            if self._type == self.hdf5:
                return  # nothing to do. Already in HDF5 mode.
            # hdf5_name = f"{column_name}/{self.mem_id}"
            self._handle = h5py.File(MemoryManager.cache_path, 'a')
            self._address = f"/{self.sha256sum}"
            self._data = self._handle.create_dataset(self._address, data=self._data)
            
    @property
    def sha256sum(self):
        return self._mem_id
    @sha256sum.setter
    def sha256sum(self,value):
        raise AttributeError("sha256sum is immutable.")
    @property
    def address(self):
        return (self._data.shape, self._dtype, self._address)
    @property
    def data(self):
        return self._data[:]
    @data.setter
    def data(self, value):
        raise AttributeError("DataBlock.data is immutable.")
    def __len__(self) -> int:
        return self._len
    def __iter__(self):
        raise AttributeError("Use vectorised functions on DataBlock.data instead of __iter__")
    def __del__(self):
        if self._type == self.shm:
            self._handle.close()
            self._handle.unlink()
        elif self._type == self.hdf5:
            self._handle.close()
        super().__del__()


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


class ManagedColumn(MemoryManagedObject):  # Behaves like an immutable list.
    _ids = count()
    def __init__(self) -> None:
        super().__init__(mem_id=f"MC-{next(self._ids)}")

        self.order = []  # strict order of datablocks.
        self.dtype = None
       
    def __len__(self):
        return sum(len(MemoryManager.get(block_id)) for block_id in self.order)

    def __del__(self):
        MemoryManager.unlink_tree(self)
        super().__del__()

    def __iter__(self):
        for block_id in self.order:
            datablock = MemoryManager.get(block_id)
            assert isinstance(datablock, DataBlock)
            for value in datablock.data:
                yield value

    def _normalize_slice(self, item=None):  # There's an outdated version sitting in utils.py
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

    def blocks(self):  # NOT USED ANYWHERE.
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
                block = MemoryManager.get(block_id)
                MemoryManager.link(self, block)
            
        else:  # It's supposedly new data.
            if not isinstance(data, np.ndarray):
                data = np.array(data)

            self._dtype_check(data)

            m = hashlib.sha256()  # let's check if it really is new data...
            m.update(data.data.tobytes())
            sha256sum = m.hexdigest()
            if sha256sum in MemoryManager.registry:  # ... not new!
                block = MemoryManager.registry.get(sha256sum)
            else:  # ... it's new!
                block = DataBlock(mem_id=sha256sum, data=data)
                MemoryManager.registry[sha256sum] = block
            # ok. solved. Now create links.
            self.order.append(block.mem_id)
            MemoryManager.link(self, block)  # Add link from Column to DataBlock
    
    def append(self, value):
        """
        Disabled. Append items is slow. Use extend on a batch instead
        """
        raise AttributeError("Append items is slow. Use extend on a batch instead")
    

class Table(MemoryManagedObject):
    _ids = count()
    def __init__(self) -> None:
        super().__init__(mem_id=f"T-{next(self._ids)}")
        self.columns = {}
    
    def __len__(self) -> int:
        if not self.columns:
            return 0
        else:
            return max(len(mc) for mc in self.columns.values())
    
    def __del__(self):
        MemoryManager.unlink_tree(self)  # columns are automatically garbage collected.
        super().__del__()

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
    def import_file(cls, path, 
        import_as, newline=b'\n', text_qualifier=None,
        delimiter=b',', first_row_has_headers=True, columns=None, sheet=None):
        """
        TABLES FROM IMPORTED FILES ARE IMMUTABLE.
        TABLES CREATED FROM OTHER TABLES EXIST IN 
        MEMORY MANAGERs CACHE IF USE DISK == True

        reads path and imports 1 or more tables as hdf5

        path: pathlib.Path or str
        import_as: 'csv','xlsx','txt'                               *123
        newline: newline character '\n', '\r\n' or b'\n', b'\r\n'   *13
        text_qualifier: character: " or '                           +13
        delimiter: character: typically ",", ";" or "|"             *1+3
        first_row_has_headers: boolean                              *123
        columns: dict with column names and datatypes               *123
            {'A': int, 'B': str, 'C': float, D: datetime}
            columns not found excess column names are ignored. 
        sheet: sheet name to import (e.g. 'sheet_1')                 *2
            sheets not found excess names are ignored.
            filenames will be {path}+{sheet}.h5
        
        (*) required, (+) optional, (1) csv, (2) xlsx, (3) txt, (4) h5

        """
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"expected pathlib.Path, got {type(path)}")
        if not path.exists():
            raise ValueError(f"file not found: {path}")

        if not isinstance(import_as,str) and import_as in ['csv','txt','xlsx']:
            raise ValueError(f"{import_as} is not supported")
        
        # check the inputs.
        if import_as in {'xlsx'}:
            raise NotImplementedError("coming soon!")
            # 1. create a task for each the sheet.

        if import_as in {'csv', 'txt'}:
            # TODO: Check if file already has been imported.
            # TODO: Check if reimport is necessary.

            # Ok. File doesn't exist, has been changed or it's a new import config.
            with path.open('rb') as fi:
                if first_row_has_headers:
                    end = find_first(fi, 0, newline)
                    headers = fi.read(end).rstrip(newline)
                    headers = text_escape(headers, delimiter=delimiter)
                    indices = []
                    for name in columns:
                        if name not in headers:
                            raise ValueError(f"column not found: {name}")
                        indices.append(headers.index(name))
                else:
                    indices = [int(v) for v in columns]
                    headers = [str(v) for v in columns]
            
            config = {
                'import_as': import_as,
                'path': str(path),
                'filesize': file_length,  # if this changes - re-import.
                'delimiter': delimiter,
                'columns': columns, 
                'newline': newline,
                'first_row_has_headers': first_row_has_headers,
                'chunk_size': chunk_size,
                'text_qualifier': text_qualifier
            }

            h5 = pathlib.Path(str(path) + '.h5')
            with h5py.File(h5,'w') as f:  # Create file, truncate if exists
                root = f['/']
                root.attrs['config'] = json.dumps(config)

            with TaskManager() as tm:
                working_overhead = 3.1415  # random guess. TODO: Calibrate.
                file_length = path.stat().st_size  #9,998,765,432 = 10Gb
                chunk_size = tm.chunk_size_per_cpu(file_length*working_overhead)
                task_size = int(chunk_size / working_overhead)
                
                tasks = []
                for i in range(int(math.ceil(file_length/task_size))):
                    # add task for each chunk for working
                    script = f"""text_reader({str(path)}, delimiter={delimiter}, columns={str(columns)}, newline={newline}, first_row_has_headers={first_row_has_headers}, start={i}, limit={task_size})"""
                    task = {'id': 1, "script": script} 
                    tm.add(task)
                # add task for merging chunks in hdf5 into single columns
                

    @classmethod
    def inspect_h5_file(cls, path, group='/'):
        """
        enables inspection of contents of HDF5 file 
        path: str or pathlib.Path
        group: you can give a specific group, defaults to the root group
        """
        def descend_obj(obj,sep='  ', offset=''):
            """
            Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
            """
            if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
                if obj.attrs.keys():  
                    for k,v in obj.attrs.items():
                        print(offset, k,":",v)  # prints config
                for key in obj.keys():
                    print(offset, key,':',obj[key])  # prints groups
                    descend_obj(obj[key],sep=sep, offset=offset+sep)
            elif type(obj)==h5py._hl.dataset.Dataset:
                for key in obj.attrs.keys():
                    print(offset, key,':',obj.attrs[key])  # prints datasets.

        with h5py.File(path,'r') as f:
            print(f"{path} contents")
            descend_obj(f[group])
    @classmethod
    def load_file(cls, path):
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not isinstance(path, pathlib.Path):
            raise TypeError(f"expected pathlib.Path, got {type(path)}")
        if not path.exists():
            raise ValueError(f"file not found: {path}")
        if not path.name.endswith(".h5"):
            raise TypeError(f"expected .h5 file, not {path.name}")
        
        # read the file and create managed columns
        # no need for task manager as the job will be IO bound.
        t = Table()
        with h5py.File(path,'r+') as f:  # 'r+' in case the sha256sum is missing.
            root = f['/']
            for name in root.keys():
                h5_name = f"{root}{name}"
                sha256sum = f[h5_name].attrs.get('sha256sum',None)
                if sha256sum is None:
                    m = hashlib.sha256()  # let's check if it really is new data...
                    dset = f[h5_name]
                    step = 100_000
                    desc = f"Calculating missing sha256sum for {h5_name}: "
                    for i in trange(0,len(dset), step, desc=desc):
                        chunk = dset[i:i+step]
                        m.update(chunk.tobytes())
                    sha256sum = m.hexdigest()
                    f[h5_name].attrs['sha256sum'] = sha256sum
                
                mc = ManagedColumn()
                t.columns[name] = mc
                MemoryManager.link(t, mc)
                db = DataBlock(mem_id=sha256sum, address=(path, h5_name))
                mc.order.append(db)
                MemoryManager.link(mc, db)
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


# FILE READER UTILS 2.0 ----------------------------

class TextEscape(object):
    """
    enables parsing of CSV with respecting brackets and text marks.

    Example:
    text_escape = TextEscape()  # set up the instance.
    for line in somefile.readlines():
        list_of_words = text_escape(line)  # use the instance.
        ...
    """
    def __init__(self, openings=b'"({[', closures=b']})"', delimiter=b','):
        """
        As an example, the Danes and Germans use " for inches and ' for feet, 
        so we will see data that contains nail (75 x 4 mm, 3" x 3/12"), so 
        for this case ( and ) are valid escapes, but " and ' aren't.

        """
        if not isinstance(openings, bytes):
            raise TypeError(f"expected bytes, got {type(openings)}")
        if not isinstance(closures, bytes):
            raise TypeError(f"expected bytes, got {type(closures)}")
        if not isinstance(delimiter, bytes):
            raise TypeError(f"expected bytes, got {type(delimiter)}")
        
        self.delimiter = ord(delimiter)
        self.openings = {c for c in openings}
        self.closures = {c for c in closures}

    def __call__(self, s):
        
        words = []
        L = list(s)
        
        ix,depth = 0,0
        while ix < len(L):  # TODO: Compile some REGEX for this instead.
            c = L[ix]
            if depth == 0 and c == self.delimiter:
                word, L = L[:ix], L[ix+1:]
                words.append("".join(chr(c) for c in word).encode('utf-8'))
                ix = -1
            elif c in self.openings:
                depth += 1
            elif c in self.closures:
                depth -= 1
            else:
                pass
            ix += 1

        if L:
            words.append("".join(chr(c) for c in L).encode('utf-8'))
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

def find_first(fh, start, chars):
    """
    fh: filehandle (fh = pathlib.Path.open() )
    start: fh.seek(start) integer
    c: character to search for.

    as start + chunk_size may not equal the next newline index,
    start is read as a "soft start":
    +-------+
    x       |  
    |    y->+  if the 2nd start index is y, then I seek the 
    |       |  next newline character and start after that.
    """
    c, snippet_size = 0, 1000
    fh.seek(start)
    while 1:
        try:
            snippet = fh.read(snippet_size)
        except EOFError:
            return len(fh)
        ix = snippet.index(chars)
        if ix != -1:
            fh.seek(0)
            return start + c + ix
        c += snippet_size


def text_reader(path, columns, newline,
                text_escape_openings=b'"([{', text_escape_closures=b'}])"',
                delimiter=b',', 
                first_row_has_headers=True, start=None, limit=None):
    """
    reads columnsname + path[start:limit] into hdf5.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)

    if not isinstance(path, pathlib.Path):
        raise TypeError
    if not path.exists():
        raise ValueError(f"File not found: {path}")
    assert isinstance(delimiter, bytes)
    assert isinstance(columns, dict)

    text_escape = TextEscape(text_escape_openings, text_escape_closures, delimiter)

    with path.open('rb') as fi:
        if first_row_has_headers:
            end = find_first(fi, 0, newline)
            headers = fi.read(end).rstrip(newline)
            start = len(headers) if start ==0 else start  # special case for 1st slice.
            headers = text_escape(headers)
            
            indices = {name: headers.index(name) for name in columns}
        else:        
            indices = {name: int(name) for name in columns}

        if start != 0:  # find the true beginning.
            start = find_first(fi, start, newline) + len(newline)
        end = find_first(fi, start + limit, newline)  # find the true end.
        fi.seek(start)
        blob = fi.read(end-start)  # 1 hard iOps. Done.
        line_count = blob.count(newline) +1  # +1 because the last line will not have it's newline.

        data = {}
        for name, dtype in columns.items():
            data[name] = np.empty((line_count, ), dtype=dtype)

        for line_no, line in enumerate(blob.split(newline)):
            fields = text_escape(line)
            for name, ix in indices.items():
                value = fields[ix]  # should convert to dtype>?!
                data[name][line_no] = value 

    h5 = pathlib.Path(str(path) + '.h5')
    with h5py.File(h5, 'w-') as f:
        for name,arr in data.items():
            f.create_dataset(f"/{name}/{start}", data=arr)  
            # `start` declares the slice id which order will be used for sorting

    
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


def test_text_escape():
    s = b"this,is,a,,b,(comma,sep'd),text"
    te = TextEscape(delimiter=b',')
    L = te(s)
    assert L == [b"this", b"is", b"a", b"",b"b", b"(comma,sep'd)", b"text"]
    

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


def test_h5_inspection():
    filename = 'a.csv.h5'

    with h5py.File(filename, 'w') as f:
        print(f.name)

        print(list(f.keys()))

        config = {
            'import_as': 'csv',
            'newline': b'\r\n',
            'text_qualifier':b'"',
            'delimiter':b",",
            'first_row_headers':True,
            'columns': {"col1": 'i8', "col2": 'int64'}
        }
        
        f.attrs['config']=str(config)
        dset = f.create_dataset("col1", dtype='i8', data=[1,2,3,4,5,6])
        dset = f.create_dataset("col2", dtype='int64', data=[5,5,5,5,5,2**33])

    # Append to dataset
    # must have chunks=True and maxshape=(None,)
    with h5py.File(filename, 'a') as f:
        dset = f.create_dataset('/sha256sum', data=[2,5,6],chunks=True, maxshape=(None, ))
        print(dset[:])
        new_data = [3,8,4]
        new_length = len(dset) + len(new_data)
        dset.resize((new_length, ))
        dset[-len(new_data):] = new_data
        print(dset[:])

        print(list(f.keys()))

    Table.inspect_h5_file(filename)
    pathlib.Path(filename).unlink()  # cleanup.


def test_file_importer():
    p = r"d:\remove_duplicates.csv"
    assert pathlib.Path(p).exists(), "?"

    columns = {  # numpy type codes: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        b'SKU ID': 'i', # integer
        b'SKU description':'S', # np variable length str
        b'Shipped date' : 'S', #datetime
        b'Shipped time' : 'S', # integer to become time
        b'vendor case weight' : 'f'  # float
    }  
    config = {
        'delimiter': b',', 
        'text_escape_openings': b'"({[', 
        "text_escape_closures": b']})"', 
        "newline": b"\r\n",
        "columns": columns, 
        "first_row_has_headers": True
    }  
    text_reader(path=p, start=0, limit=10000, **config)
    p2 = pathlib.Path(str(p) + '.h5')
    Table.inspect_h5_file(p2)
    # Table.import_file(p, **config)
    p2.unlink()  # cleanup!



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
    for k,v in {k:v for k,v in sorted(globals().items()) if k.startswith('test') and callable(v)}.items():
        print(20 * "-" + k + "-" * 20)
        v()

