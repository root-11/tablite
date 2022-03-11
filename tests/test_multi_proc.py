from graph import Graph
import math
# from numpy import np
from itertools import count
from typing import List, Set, Dict, Tuple, Optional
from typing import Callable, Iterator, Union, Optional
from typing import Union, Any, Optional, cast



class DataBlock(object):
    def __init__(self, data):
        self._on_disk = False
        self._len = len(data)
        self._data = data  # in hdf5 or in memory
        self._location = None
    def __len__(self) -> int:
        return self._len


class TaskManager(object):
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

    def inventory(self):
        c = count()
        n = math.ceil(math.log10(len(self.reference_map.nodes())))+1
        for node_id in self.reference_map.nodes(in_degree=0):
            obj = self.reference_map.node(node_id)
            print(f"{next(c)}|".zfill(n), node_id, ":", str(obj), "columns =", list(obj.columns.keys()))
            for name, mc in obj.columns.items():
                print(f"{next(c)}|".zfill(n), '└─┬', name, str(mc), "length =", len(mc))
                for i, block_id in enumerate(mc.order):
                    block = self.reference_map.node(block_id)
                    print(f"{next(c)}|".zfill(n), '  └──', i, str(block), "length =", len(block))

task_manager = TaskManager()
tmap = task_manager.reference_map


class ManagedColumn(object):  # Behaves like an immutable list.
    def __init__(self) -> None:
        self.order = []  # strict order of datablocks.
        tmap.add_node(node_id=id(self), obj=self)

    def __len__(self):
        return sum(len(tmap.node(node_id)) for node_id in self.order)

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
        tmap.add_node(node_id=id(self), obj=self)
    
    def __len__(self) -> int:
        if not self.columns:
            return 0
        else:
            return len(list(self.columns.values())[0])
        
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
# assert len(tmap.edges()) == tables * managed_columns_per_table + 8
assert len(table1) + len(table2) + len(table3) == 3 + 3 + 6

task_manager.inventory()


print('done')