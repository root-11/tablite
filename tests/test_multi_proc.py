from graph import Graph
import math
import time
import weakref
from itertools import count
from collections import deque
import h5py
import numpy as np
from multiprocessing import cpu_count, shared_memory, Pool, freeze_support


_pool = None
def _get_pool():
    global _pool
    if _pool is None:
        _pool = Pool()
    return _pool


class TaskManager(object):
    registry = weakref.WeakValueDictionary()  # {Object ID: Object} The weakref presents blocking of garbage collection.
    lru_tracker = {}  # {DataBlockId: process_time, ...}
    map = Graph()  # Documents relations between Table, Column & Datablock.
    process_pool = None
    tasks = None

    @classmethod
    def get_pool(cls):
        cls.process_pool = _get_pool()

    @classmethod
    def __del__(cls):
        # Use `import gc; del TaskManager; gc.collect()` to delete the TaskManager class.
        # shm.close()
        # shm.unlink()
        if cls.process_pool is not None:
            cls.process_pool.close()

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
            cls.map.add_node(id(b), b)
    @classmethod
    def unlink(cls, a,b):
        cls.map.del_edge(id(a), id(b))
        if isinstance(b, DataBlock):
            if cls.map.in_degree(id(b)) == 0:
                cls.map.del_node(id(b))
    @classmethod
    def unlink_tree(cls, a):
        nodes = deque([id(a)])
        while nodes:
            n1 = nodes.popleft()
            if cls.map.in_degree(n1) == 0:
                for n2 in cls.map.nodes(from_node=n1):
                    nodes.append(n2)
                cls.map.del_node(n1)
    @classmethod
    def get(cls, node_id):
        cls.lru_tracker[node_id] = time.process_time()  # keep the lru tracker up to date.
        return cls.map.node(node_id)
    @classmethod
    def inventory(cls):
        """ returns printable overview of the registered tables, managed columns and datablocs. """
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


class DataBlock(object):
    def __init__(self, data):
        TaskManager.register(self)
        self._on_disk = False
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)

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
        return self._data  # TODO: Needs to cope with data not being loaded.
    @data.setter
    def data(self,value):
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
        TaskManager.deregister(self)


class ManagedColumn(object):  # Behaves like an immutable list.
    def __init__(self) -> None:
        TaskManager.register(self)
        self.order = []  # strict order of datablocks.

    def __len__(self):
        return sum(len(TaskManager.get(block_id)) for block_id in self.order)

    def __del__(self):
        TaskManager.unlink_tree(self)
        TaskManager.deregister(self)

    def __iter__(self):
        for block_id in self.order:
            datablock = TaskManager.get(block_id)
            assert isinstance(datablock, DataBlock)
            for value in datablock.data:
                yield value

    def blocks(self):
        return [TaskManager.get(block_id).address for block_id in self.order]           
        
    def extend(self, data):
        if isinstance(data, ManagedColumn):  # It's data we've seen before.
            self.order.extend(data.order[:])
            for block_id in data.order:
                TaskManager.link(self, block_id)
        else:  # It's new data.
            data = DataBlock(data)
            self.order.append(id(data))
            TaskManager.link(self, data)  # Add link from Column to DataBlock
        
    def append(self, value):
        raise AttributeError("Append items is slow. Use extend on a batch instead")
    

class Table(object):
    def __init__(self) -> None:
        TaskManager.register(self)
        self.columns = {}
    
    def __len__(self) -> int:
        if not self.columns:
            return 0
        else:
            return max(len(mc) for mc in self.columns.values())
    
    def __del__(self):
        TaskManager.unlink_tree(self)  # columns are automatically garbage collected.
        TaskManager.deregister(self)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.columns[item]
        elif isinstance(item, slice):
            raise NotImplementedError("coming!")
        else:
            raise NotImplemented(f"? {type(item)}")

    def __delitem__(self, item):
        if isinstance(item, str):
            mc = self.columns[item]
            del self.columns[item]
            TaskManager.unlink(self, mc)
            TaskManager.unlink_tree(mc)
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
        TaskManager.link(self, mc)  # Add link from Table to Column
        
    def __add__(self, other):
        if self.columns.keys() != other.columns.keys():
            raise ValueError("Columns are different")
        t = self.copy()
        for name,mc in other.columns.items():
            mc2 = t.columns[name]
            mc2.extend(mc)
        return t
    
    def copy(self):
        t = Table()
        for name,mc in self.columns.items():
            t.add_column(name,mc)
        return t


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

    tables = 3
    managed_columns_per_table = 2 
    datablocks = 2

    assert len(TaskManager.map.nodes()) == tables + (tables * managed_columns_per_table) + datablocks
    assert len(TaskManager.map.edges()) == tables * managed_columns_per_table + 8 - 2  # the -2 is because of double reference to 1 and 2 in Table3
    assert len(table1) + len(table2) + len(table3) == 3 + 3 + 6

    # delete table
    assert len(TaskManager.map.nodes()) == 11, "3 tables, 6 managed columns and 2 datablocks"
    assert len(TaskManager.map.edges()) == 12
    del table1  # removes 2 refs to ManagedColumns and 2 refs to DataBlocks
    assert len(TaskManager.map.nodes()) == 8, "removed 1 table and 2 managed columns"
    assert len(TaskManager.map.edges()) == 8 
    # delete column
    del table2['A']
    assert len(TaskManager.map.nodes()) == 7, "removed 1 managed column reference"
    assert len(TaskManager.map.edges()) == 6

    print(TaskManager.inventory())

    del table3
    del table2
    assert len(TaskManager.map.nodes()) == 0
    assert len(TaskManager.map.edges()) == 0

def fx2(address):
    shape, dtype, name = address
    existing_shm = shared_memory.SharedMemory(name=name)
    c = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    result = 2*c
    existing_shm.close()  # emphasising that shm is no longer used.
    return result

def test_shm():
    table1 = Table()
    assert len(table1) == 0
    table1.add_column('A', data=[1,2,3])  # block1
    table1['A'].extend(data=[4,4,8])  # block2
    table1['A'].extend(data=[8,9,10])  # block3
    assert [v for v in table1['A']] == [1,2,3,4,4,8,8,9,10]
    blocks = table1['A'].blocks()

    print(blocks)
    result = TaskManager.process_pool.map(fx2, blocks)
    print(result)
    


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
    # freeze_support()
    # for k,v in globals().copy().items():
    #     if k.startswith("test") and callable(v):
    #         v()
    TaskManager.get_pool()
    test_shm()
    TaskManager.process_pool.close()
    TaskManager.process_pool.join()
    TaskManager.process_pool = None
