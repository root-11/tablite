from graph import Graph
import math
import weakref
# from numpy import np
from itertools import count
from collections import deque
# from typing import List, Set, Dict, Tuple, Optional
# from typing import Callable, Iterator, Union, Optional
# from typing import Union, Any, Optional, cast


class TaskManager(object):
    registry = weakref.WeakValueDictionary()

    def __init__(self) -> None:
        self.reference_map = Graph()  # Documents relations between Table, Column & Datablock.
        self.lru_tracker = {}  
        self.process_pool=None
        self.tasks = []
        
    def __del__(self):
        # shm.close()
        # shm.unlink()
        # pool.close()
        pass

    def remove(self, node_id) -> None:
        g = self.reference_map
        nodes = deque([node_id])
        while nodes:
            n1 = nodes.popleft()
            if g.in_degree(n1) == 0:
                for n2 in g.nodes(from_node=n1):
                    nodes.append(n2)
                g.del_node(n1)

    def inventory(self):
        c = count()
        n = math.ceil(math.log10(len(self.reference_map.nodes())))+2
        L = []
        add = L.append
        d = {id(obj): name for name,obj in globals().copy().items() if isinstance(obj, (Table))}

        for node_id in self.reference_map.nodes(in_degree=0):
            name = d.get(node_id, "")
            obj = self.registry.get(node_id,None)
            if obj:
                columns = [] if obj is None else list(obj.columns.keys())
                add(f"{next(c)}|".zfill(n) + f" {name}, columns = {columns}")
                for name, mc in obj.columns.items():
                    add(f"{next(c)}|".zfill(n) + f" └─┬ {name} {mc.__class__.__name__}, length = {len(mc)}")
                    for i, block_id in enumerate(mc.order):
                        block = self.reference_map.node(block_id)
                        add(f"{next(c)}|".zfill(n) + f"   └── {i} {block.__class__.__name__}, length = {len(block)}")
        return "\n".join(L)

task_manager = TaskManager()
tmap = task_manager.reference_map


class DataBlock(object):
    def __init__(self, data):
        TaskManager.registry[id(self)] = self
        self._on_disk = False
        self._len = len(data)
        self._data = data  # in hdf5 or in memory
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
        self.order = []  # strict order of datablocks.
        TaskManager.registry[id(self)] = self

    def __len__(self):
        return sum(len(tmap.node(node_id)) for node_id in self.order)

    def __del__(self):
        task_manager.remove(id(self))

    def __iter__(self):
        for node_id in self.order:
            datablock = tmap.node(node_id)
            for value in datablock:
                yield value

    def extend(self, data):
        if isinstance(data, ManagedColumn):  # It's data we've seen before.
            for block_id in data.order:
                self.order.append(block_id)
                tmap.add_edge(id(self), block_id)
        else:  # It's new data.
            data = DataBlock(data)
            self.order.append(id(data))
            tmap.add_node(node_id=id(data), obj=data)  # Add the datablock to tmap.
            tmap.add_edge(node1=id(self), node2=id(data))  # Add link from Column to DataBlock
        
    def append(self, value):
        raise AttributeError("Append is slow. Use extend instead")
    

class Table(object):
    def __init__(self) -> None:
        self.columns = {}
        TaskManager.registry[id(self)] = self
    
    def __len__(self) -> int:
        if not self.columns:
            return 0
        else:
            return len(list(self.columns.values())[0])
    
    def __del__(self):
        task_manager.remove(id(self))

    def __delitem__(self, item):
        if isinstance(item, str):
            mc = self.columns[item]
            del self.columns[item]
            tmap.del_edge(id(self), id(mc))
            task_manager.remove(id(mc))

        elif isinstance(item, slice):
            raise NotImplementedError

    def del_column(self, name):
        self.__delitem__(name)
 
    def add_column(self, name, data):
        if name in self.columns:
            raise ValueError(f"name {name} already used")
        mc = ManagedColumn()
        mc.extend(data)
        self.columns[name] = mc
        tmap.add_edge(node1=id(self), node2=id(mc))  # Add link from Table to Column
        
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

assert len(tmap.nodes()) == tables + (tables * managed_columns_per_table) + datablocks
assert len(tmap.edges()) == tables * managed_columns_per_table + 8 - 2  # the -2 is because of double reference to 1 and 2 in Table3
assert len(table1) + len(table2) + len(table3) == 3 + 3 + 6



# delete table
assert len(tmap.nodes()) == 11, "3 tables, 6 managed columns and 2 datablocks"
assert len(tmap.edges()) == 12
del table1  # removes 2 refs to ManagedColumns and 2 refs to DataBlocks
assert len(tmap.nodes()) == 8, "removed 1 table and 2 managed columns"
assert len(tmap.edges()) == 8 
# delete column
del table2['A']
assert len(tmap.nodes()) == 7, "removed 1 managed column reference"
assert len(tmap.edges()) == 6

print(task_manager.inventory())

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
