import hashlib
from graph import Graph
import math
import time
import weakref
from itertools import count
from collections import deque
import h5py
import numpy as np
from multiprocessing import cpu_count, shared_memory


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
        transforms slice into range inputs
        returns start,stop,step
        """
        if item is None:
            item = slice(None, len(self), None)
        assert isinstance(item, slice)

        if item.stop < 0:
            start = len(self) + item.stop
            stop = len(self)
            step = 1 if item.step is None else item.step
        else:
            start = 0 if item.start is None else item.start
            stop = item.stop
            step = 1 if item.step is None else item.step
        return start, stop, step
            
    def __getitem__(self, item):
        """
        returns a slice (item) of a column.
        """
        mc = ManagedColumn()

        if isinstance(item, slice):
            r = range(*self._normalize_slice(item))
            page_start = 0
            for block_id in self.order:
                block = MemoryManager.get(block_id)

                if r.step==1 and page_start <= r.start and page_start+len(block) <= r.end:
                    mc.append(block_id)
                    continue
                
                block_range = range(page_start, page_start+len(block))
                intercept_range = intercept(r,block_range)  # very effective!
                if len(intercept_range)==0:
                    continue

                x = {i - page_start for i in intercept_range}
                mask = np.array([i in x for i in len(block)])
                new_block = block.data[np.where(mask)]
                mc.extend(new_block)

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
            raise KeyError

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
        
        try:
            self.compare(other)
        except (TypeError, ValueError):
            return False

        if len(self) != len(other):
            return False

        for name, mc in self.columns.items():
            mc2 = other.columns[name]
            if any(a!=b for a,b in zip(mc,mc2)):
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

    def select(self, *items):  # Was `filter` in Tablite < feature15
        """
        enables selection of existing columns with repetition

        Example:
        >>> Table.columns
        'a','b','c','d','e'

        >>> for row in Table.select('b', 'a', 'a', 'c', slice(2,20,2)):
        >>>    b,a,a,c = row ...

        returns values in same order as selection.
        """
        names, slc = [], slice(len(self))
        for i in items:
            if isinstance(i,slice):
                slc = i
            else:
                names.append(i)
        
        t = Table()
        for name in items:
            mc = self.columns[name]
            t.add_column(name, data=mc[slc])
        return t       


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

