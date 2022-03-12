from graph import Graph
import math
import time
import weakref
# from numpy import np
from itertools import count
from collections import deque


class TaskManager(object):
    registry = weakref.WeakValueDictionary()  # {Object ID: Object}
    lru_tracker = {}  # {DataBlockId: process_time, ...}
    map = Graph()  # Documents relations between Table, Column & Datablock.
    process_pool = None
    tasks = None
    
    def __del__(self):
        # + import gc; gc.collect()
        # shm.close()
        # shm.unlink()
        # pool.close()
        pass

    @classmethod
    def register(cls, obj):
        cls.registry[id(obj)] = obj
    @classmethod
    def remove(cls, obj) -> None:
        nodes = deque([id(obj)])
        while nodes:
            n1 = nodes.popleft()
            if cls.map.in_degree(n1) == 0:
                for n2 in cls.map.nodes(from_node=n1):
                    nodes.append(n2)
                cls.map.del_node(n1)
                cls.lru_tracker.pop(n1,None)
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
    def __getitem__(cls, node_id):
        cls.lru_tracker[node_id] = time.process_time()
        return cls.map.node(node_id)

    @classmethod
    def get(cls, node_id):
        cls.lru_tracker[node_id] = time.process_time()
        return cls.map.node(node_id)

    @classmethod
    def inventory(cls):
        c = count()
        node_count = len(cls.map.nodes())
        if node_count == 0:
            return "no nodes" 
        n = math.ceil(math.log10(node_count))+2
        L = []
        d = {id(obj): name for name,obj in globals().copy().items() if isinstance(obj, (Table))}

        for node_id in cls.map.nodes(in_degree=0):
            name = d.get(node_id, "")
            obj = cls.registry.get(node_id,None)
            if obj:
                columns = [] if obj is None else list(obj.columns.keys())
                L.append(f"{next(c)}|".zfill(n) + f" {name}, columns = {columns}")
                for name, mc in obj.columns.items():
                    L.append(f"{next(c)}|".zfill(n) + f" └─┬ {name} {mc.__class__.__name__}, length = {len(mc)}")
                    for i, block_id in enumerate(mc.order):
                        block = cls.map.node(block_id)
                        L.append(f"{next(c)}|".zfill(n) + f"   └── {i} {block.__class__.__name__}, length = {len(block)}")
        return "\n".join(L)


class DataBlock(object):
    def __init__(self, data):
        TaskManager.register(self)
        self._on_disk = False
        self._len = len(data)
        self._data = data  # hdf5 or shm 
        self._location = None  
    
    def __len__(self) -> int:
        return self._len
        
    def __del__(self):
        if self._on_disk:
            pass   # del file
        else:
            pass  # close shm


class ManagedColumn(object):  # Behaves like an immutable list.
    def __init__(self) -> None:
        TaskManager.register(self)
        self.order = []  # strict order of datablocks.

    def __len__(self):
        return sum(len(TaskManager.get(block_id)) for block_id in self.order)

    def __del__(self):
        TaskManager.remove(self)

    def __iter__(self):
        for block_id in self.order:
            datablock = TaskManager[block_id]
            for value in datablock:
                yield value

    def extend(self, data):
        if isinstance(data, ManagedColumn):  # It's data we've seen before.
            for block_id in data.order:
                self.order.append(block_id)
                TaskManager.link(self, block_id)
        else:  # It's new data.
            data = DataBlock(data)
            self.order.append(id(data))
            TaskManager.link(self, data)  # Add link from Column to DataBlock
        
    def append(self, value):
        raise AttributeError("Append is slow. Use extend instead")
    

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
        TaskManager.remove(self)

    def __delitem__(self, item):
        if isinstance(item, str):
            mc = self.columns[item]
            del self.columns[item]
            TaskManager.unlink(self, mc)
            TaskManager.remove(mc)

        elif isinstance(item, slice):
            raise NotImplementedError

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

print('done')

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
