import hashlib
import operator
import pyexcel
import math
import pathlib
import time
import weakref
import numpy as np
import h5py
import random
import re
import io
import os
import json
import traceback
import queue
import multiprocessing
import copy
from multiprocessing import shared_memory  # do not import SharedMemory as it is overwritten below.
from itertools import count, chain
from abc import ABC
from collections import defaultdict, deque

import chardet
import psutil
from tqdm import tqdm, trange
from graph import Graph

# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" <--- Do not enable this! Imports will fail.
HDF5_IMPORT_ROOT = "__h5_import"  # the hdf5 base name for imports. f.x. f['/__h5_import/column A']
MEMORY_MANAGER_CACHE_DIR = os.getcwd()
MEMORY_MANAGER_CACHE_FILE = "tablite_cache.hdf5"


# class TaskManager(object):
#     shared_memory_references = {}  # names: shm pointers.
#     shared_memory_reference_counter = defaultdict(int)  # tracker for the NAT protocol.

#     memory_usage_ceiling = 0.9  # 90%

#     def __init__(self,cores=None) -> None:    
#         """
#         cores: 0 < integer <= cpu count
#         """
#         self._cpus = min(psutil.cpu_count(), cores) if (isinstance(cores,int) and cores > 0) else psutil.cpu_count()
#         self._disk_space = psutil.disk_usage('/').free
#         self._memory = psutil.virtual_memory().available

#         self.tq = multiprocessing.Queue()  # task queue for workers.
#         self.rq = multiprocessing.Queue()  # result queue for workers.
#         self.pool = []                     # list of sub processes
#         self.pool_sigq = {}                # signal queue for each worker.
#         self.tasks = 0                     # counter for task tracking
        
#     def __enter__(self):
#         self.start()
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb): # signature requires these, though I don't use them.
#         self.stop()  # stop the workers.
        
#         # Clean up on exit.
#         for k,v in self.shared_memory_reference_counter.items():
#             if k in self.shared_memory_references and v == 0:
#                 del self.shared_memory_references[k]  # this unlinks the shared memory object,
#                 # which now can be GC'ed if no other variable points to it.
        
#     def start(self):
#         for i in range(self._cpus):  # create workers
#             name = str(i)
#             sigq = multiprocessing.Queue()  # we create one signal queue for each proc.
#             self.pool_sigq[name] = sigq
#             worker = Worker(name=name, tq=self.tq, rq=self.rq, sigq=sigq)
#             self.pool.append(worker)

#         with tqdm(total=self._cpus, unit="n", desc="workers ready") as pbar:
#             for p in self.pool:
#                 p.start()

#             while True:
#                 alive = sum(1 if p.is_alive() else 0 for p in self.pool)
#                 pbar.n = alive
#                 pbar.refresh()
#                 if alive < self._cpus:
#                     time.sleep(0.01)
#                 else:
#                     break  # all sub processes are alive. exit the setup loop.

#     def execute(self, tasks):
#         if isinstance(tasks, Task):
#             response = (tasks,)
#         if not isinstance(tasks, (list,tuple)) or not all([isinstance(i, Task) for i in tasks]):
#             raise TypeError

#         for t in tasks:
#             self.tq.put(t)
#             self.tasks += 1  # increment task counter.
        
#         results = []  
#         with tqdm(total=self.tasks, unit='task') as pbar:
#             while self.tasks != 0:
#                 try:
#                     response = self.rq.get_nowait()
                
#                     if isinstance(response, NATsignal): 
#                         if response.shm_name not in self.shared_memory_references:  # its a NOTIFY from a WORKER.
#                             # first create a hard ref to the memory object.
#                             self.shared_memory_references[response.shm_name] = SharedMemory(name=response.shm_name, create=False)
#                             self.shared_memory_reference_counter[response.shm_name] += 1
#                             # then send the ACKNOWLEDGEMENT directly to the WORKER.
#                             self.pool_sigq[response.worker_name].put(response)
#                         else:  # It's the second time we see the name so it's a TRANSFER COMPLETE
#                             self.shared_memory_reference_counter[response.shm_name] -= 1 
#                         # at this point we can be certain that the SHMs are in the main process.
#                         continue  # keep looping as there may be more.

#                     elif isinstance(response, Task):
#                         if response.exception:
#                             raise Exception(response.exception)

#                         self.tasks -= 1  # decrement task counter.
#                         pbar.set_description(response.f.__name__)
#                         results.append(response)
#                         pbar.update(1)
                    
#                 except queue.Empty:
#                     time.sleep(0.01)
#         return results 

#     def stop(self):
#         for _ in range(self._cpus):  # put enough stop messages for all workers.
#             self.tq.put("stop")

#         with tqdm(total=len(self.pool), unit="n", desc="workers stopping") as pbar:
#             while True:
#                 not_alive = sum(1 if not p.is_alive() else 0 for p in self.pool)
#                 pbar.n = not_alive
#                 pbar.refresh()
#                 if not_alive < self._cpus:
#                     time.sleep(0.01)
#                 else:
#                     break
#         self.pool.clear()

#         # clear the message queues.
#         while not self.tq.empty:  
#             _ = self.tq.get_nowait()  
#         while not self.rq.empty:
#             _ = self.rq.get_nowait()

#     def chunk_size_per_cpu(self, working_memory_required):  # 39,683,483,123 = 39 Gb.
#         if working_memory_required < psutil.virtual_memory().free:
#             mem_per_cpu = math.ceil(working_memory_required / self._cpus)
#         else:
#             memory_ceiling = int(psutil.virtual_memory().total * self.memory_usage_ceiling)
#             memory_used = psutil.virtual_memory().used
#             available = memory_ceiling - memory_used  # 6,321,123,321 = 6 Gb
#             mem_per_cpu = int(available / self._cpus)  # 790,140,415 = 0.8Gb/cpu
#         return mem_per_cpu


# class Worker(multiprocessing.Process):
#     def __init__(self, name, tq, rq, sigq):
#         super().__init__(group=None, target=self.update, name=name, daemon=False)
#         self.exit = multiprocessing.Event()
#         self.tq = tq  # workers task queue
#         self.rq = rq  # workers result queue
#         self.sigq = sigq  # worker signal reciept queue.
        
               
#     def update(self):
#         # this is global for the sub process only.
#         TaskManager.shared_memory_references

#         while True:
#             # first process any/all direct signals first.
#             while True:
#                 try:
#                     ack = self.sigq.get_nowait()   # receive acknowledgement of hard ref to SharedMemoryObject from SIGQ            
#                     shm = TaskManager.shared_memory_references.pop(ack.shm_name)  # pop the shm
#                     shm.close()  # assure closure of the shm.
#                     del TaskManager.shared_memory_reference_counter[ack.shm_name]
#                     self.rq.put(ack)  # respond to MAINs RQ that transfer is complete.
#                 except queue.Empty:
#                     break

#             # then deal with any tasks...
#             try:  
#                 task = self.tq.get_nowait()
#                 if task == "stop":
#                     self.tq.put_nowait(task)  # this assures that everyone gets the stop signal.
#                     self.exit.set()
#                     break
#                 elif isinstance(task, Task):
#                     task.execute()
                    
#                     for k,v in TaskManager.shared_memory_references.items():
#                         if k not in TaskManager.shared_memory_reference_counter:
#                             TaskManager.shared_memory_reference_counter[k] = 1
#                             self.rq.put(NATsignal(k, self.name))  # send Notify from subprocess to main
                        
#                     self.rq.put(task)

#                 else:
#                     raise Exception(f"What is {task}?")
#             except queue.Empty:
#                 time.sleep(0.01)
#                 continue


# class NATsignal(object):
#     def __init__(self, shm_name, worker_name):
#         """
#         shm_name: str: name from shared_memory.
#         worker_name: str: required by TaskManager for sending ACK message to worker.
#         """
#         self.shm_name = shm_name
#         self.worker_name = worker_name


# class SharedMemory(shared_memory.SharedMemory):
#     def __init__(self, name=None, create=False, size=0) -> None:
        
#         if name in TaskManager.shared_memory_references:
#             print(f'found SharedMemory({name}) in {__file__} in registry')
#             self = TaskManager.shared_memory_references[name]  # return from registry.
#         else:
#             try:
#                 super().__init__(name, create, size)
#                 TaskManager.shared_memory_references[self.name] = self  # add to registry. This blocks __del__ !  
#                 print(f"SharedMemory({self.name}) created.")
#             except FileNotFoundError:
#                 print(f"{name}!")
    
#     def __del__(self):
#         print(f"SharedMemory deleted... in {__name__}")
#         super().__del__()


# class Task(ABC):
#     """
#     Generic Task class for tasks.
#     """
#     ids = count(start=1)
#     def __init__(self, f, *args, **kwargs) -> None:
#         """
#         f: callable 
#         *args: arguments for f
#         **kwargs: keyword arguments for f.
#         """
#         if not callable(f):
#             raise TypeError
#         self.task_id = next(self.ids)
#         self.f = f
#         self.args = copy.deepcopy(args)  # deep copy is slow unless the data is shallow.
#         self.kwargs = copy.deepcopy(kwargs)
#         self.result = None
#         self.exception = None

#     def __str__(self) -> str:
#         if self.exception:
#             return f"Call to {self.f.__name__}(*{self.args}, **{self.kwargs}) --> Error: {self.exception}"
#         else:
#             return f"Call to {self.f.__name__}(*{self.args}, **{self.kwargs}) --> Result: {self.result}"

#     def execute(self):
#         """ The worker calls this function. """
#         try:
#             self.result = self.f(*self.args, **self.kwargs)
#         except Exception as e:
#             f = io.StringIO()
#             traceback.print_exc(limit=3, file=f)
#             f.seek(0)
#             error = f.read()
#             f.close()
#             self.exception = error


# class MemoryManager(object):
#     registry = weakref.WeakValueDictionary()  # The weakref presents blocking of garbage collection.
#     # Two usages:
#     # {Object ID: Object} for all objects.
#     # {sha256hash: Object} for DataBlocks (used to prevent duplication of data in memory.)
#     lru_tracker = {}  # {DataBlockId: process_time, ...}
#     map = Graph()  # Documents relations between Table, Column & Datablock.
#     process_pool = None
#     tasks = None
#     cache_path = pathlib.Path(MEMORY_MANAGER_CACHE_DIR) / MEMORY_MANAGER_CACHE_FILE
            
#     @classmethod
#     def reset(cls):
#         """
#         enables user to erase any cached hdf5 data.
#         Useful for testing where the user wants a clean working directory.

#         Example:
#         # new test case:
#         >>> import MemoryManager
#         >>> MemoryManager.reset()
#         >>> ... start on testcase ...
#         """
#         cls.cache_file = h5py.File(cls.cache_path, mode='w')  # 'w' Create file, truncate if exists
#         cls.cache_file.close()
#         for obj in list(cls.registry.values()):
#             del obj

#     @classmethod
#     def __del__(cls):
#         # Use `import gc; del MemoryManager; gc.collect()` to delete the MemoryManager class.
#         # shm.close()
#         # shm.unlink()
#         cls.cache_file.unlink()  # HDF5.

#     @classmethod
#     def register(cls, obj):  # used at __init__
#         assert isinstance(obj, MemoryManagedObject)
#         cls.registry[obj.mem_id] = obj
#         cls.lru_tracker[obj.mem_id] = time.process_time()

#     @classmethod
#     def deregister(cls, obj):  # used at __del__
#         assert isinstance(obj, MemoryManagedObject)
#         cls.registry.pop(obj.mem_id, None)
#         cls.lru_tracker.pop(obj.mem_id, None)

#     @classmethod
#     def link(cls, a, b):
#         assert isinstance(a, MemoryManagedObject)
#         assert isinstance(b, MemoryManagedObject)
        
#         cls.map.add_edge(a.mem_id, b.mem_id)
#         if isinstance(b, DataBlock):
#             # as the registry is a weakref, a hard ref to the datablocks is needed!
#             cls.map.add_node(b.mem_id, b)  # <-- Hard ref.

#     @classmethod
#     def unlink(cls, a, b):
#         assert isinstance(a, MemoryManagedObject)
#         assert isinstance(b, MemoryManagedObject)

#         cls.map.del_edge(a.mem_id, b.mem_id)
#         if isinstance(b, DataBlock):
#             if cls.map.in_degree(b.mem_id) == 0:  # remove the datablock if in-degree == 0
#                 cls.map.del_node(b.mem_id)

#     @classmethod
#     def unlink_tree(cls, a):
#         """
#         removes `a` and descendents of `a` if descendant does not have other incoming edges.
#         """
#         assert isinstance(a,MemoryManagedObject)
        
#         nodes = deque([a.mem_id])
#         while nodes:
#             n1 = nodes.popleft()
#             if cls.map.in_degree(n1) == 0:
#                 for n2 in cls.map.nodes(from_node=n1):
#                     nodes.append(n2)
#                 cls.map.del_node(n1)  # removes all edges automatically.
#     @classmethod
#     def get(cls, mem_id, default=None):
#         """
#         fetches datablock & maintains lru_tracker

#         mem_id: DataBlocks mem_id
#         returns: DataBlock
#         """
#         cls.lru_tracker[mem_id] = time.process_time()  # keep the lru tracker up to date.
#         n = cls.map.node(mem_id)
#         if n is None:
#             return default
#         else:
#             return n
#     @classmethod
#     def inventory(cls):
#         """
#         returns printable overview of the registered tables, managed columns and datablocs.
#         """
#         c = count()
#         node_count = len(cls.map.nodes())
#         if node_count == 0:
#             return "no nodes" 
#         n = math.ceil(math.log10(node_count))+2
#         L = []
#         d = {obj.mem_id: name for name,obj in globals().copy().items() if isinstance(obj, (Table))}

#         for node_id in cls.map.nodes(in_degree=0):
#             name = d.get(node_id, "Table")
#             obj = cls.registry.get(node_id,None)
#             if obj:
#                 columns = [] if obj is None else list(obj.columns.keys())
#                 L.append(f"{next(c)}|".zfill(n) + f" {name}, columns = {columns}, registry id: {node_id}")
#                 for name, mc in obj.columns.items():
#                     L.append(f"{next(c)}|".zfill(n) + f" └─┬ {mc.__class__.__name__} \'{name}\', length = {len(mc)}, registry id: {id(mc)}")
#                     for i, block_id in enumerate(mc.order):
#                         block = cls.map.node(block_id)
#                         L.append(f"{next(c)}|".zfill(n) + f"   └── {block.__class__.__name__}-{i}, length = {len(block)}, registry id: {block_id}")
#         return "\n".join(L)

# def isiterable(item):
#     """
#     Determines if an item is iterable.
#     """
#     # only valid way to check that a variable is iterable.
#     # https://stackoverflow.com/questions/1952464/in-python-how-do-i-determine-if-an-object-is-iterable
#     try:   
#         iter(item)
#         return True
#     except TypeError:
#         return False


# class SharedMemoryAddress(object):
#     """
#     Generic envelope for exchanging information about shared data between main and worker processes.
#     """
#     def __init__(self, mem_id, shape, dtype, path=None, hdf5_route=None, shm_name=None):
#         if not isinstance(shape, tuple) and all(isinstance(i, int) for i in shape):
#             raise TypeError

#         if isinstance(dtype, np.dtype):
#             dtype = dtype.name
#         if not isinstance(dtype, str) and dtype != "":
#             raise TypeError

#         if path is not None:
#             if isinstance(path, pathlib.Path):
#                 path = str(path)
#             if not isinstance(path, str):
#                 raise TypeError

#         if hdf5_route is not None and not isinstance(hdf5_route, str):
#                 raise TypeError

#         if shm_name is not None and not isinstance(shm_name, str):
#                 raise TypeError

#         if not any([path, hdf5_route, shm_name]):
#             raise ValueError("path and hdf5_route OR shm_name is required.")

#         self.mem_id = mem_id
#         self.shape = shape
#         self.dtype = dtype
#         self.path = path
#         self.hdf5_route = hdf5_route
#         self.shm_name = shm_name
    
#     def to_shm(self):
#         """
#         returns shm based numpy array from address.
#         """
#         handle = SharedMemory(name=self.shm_name)
#         data = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=handle.buf)
#         return handle, data


# class MemoryManagedObject(ABC):
#     """
#     Base Class for Memory Managed Objects
#     """
#     _ids = count()
#     def __init__(self, mem_id) -> None:
#         self._mem_id = mem_id
#         MemoryManager.register(self)
#     @property
#     def mem_id(self):
#         return self._mem_id
#     @mem_id.setter
#     def mem_id(self, value):
#         raise AttributeError("mem_id is immutable")
#     def __del__(self):
#         MemoryManager.deregister(self)


# class DataBlock(MemoryManagedObject):
#     """
#     Generic immutable class for holding data.
#     """
#     HDF5 = 'hdf5'
#     SHM = 'shm'

#     def __init__(self, mem_id, data=None, shm=None, shape=None, dtype=None, path=None, route=None):
#         """
#         mem_id: sha256sum of the datablock. Why? Because of storage.

#         init requires 1 of 3:
#         (1) data as np.array
#         (2) shm adress, np.shape, np.dtype
#         (3) path, hdf5 route

#         examples:

#         _1 = DataBlock(data=np.array([1,2,3,4]))
#         _2 = DataBlock(shm="psm_21467_46075", shape=(4,), dtype=np.int)
#         _3 = DataBlock(path="path/to/hdf5.h5", route="/table_name/column_name/sha256sum")
#                 """
#         super().__init__(mem_id=mem_id)

#         self._address_type = None
#         self._handle = None
#         self._data = None
#         self._len = None
#         self._dtype = None
#         self._path = None
#         self._hdf5_route = None
#         self._shm_name = None
        
#         if data is not None: 
#             self._from_data(data)
#         elif all(i is not None for i in [shm,shape,dtype]):
#             self._from_shm(shm, shape, dtype)
#         elif all(i is not None for i in [path, route]):
#             self._from_hdf5(path, route)
#         else:
#             raise ValueError("Either {data}, {shm,shape,dtype} or {path,route} must be None")

#     def _from_data(self, data):
#         """
#         enables init from data.
#         """
#         if not isinstance(data, np.ndarray):
#             raise TypeError("Expected a numpy array.")       

#         self._address_type = self.SHM
#         self._handle = SharedMemory(create=True, size=data.nbytes)  
#         self._data = np.ndarray(data.shape, dtype=data.dtype, buffer=self._handle.buf)  
#         self._data = data[:]  # copy the source data into the shm (source may be a np.view)
#         self._len = len(data)
#         self._dtype = data.dtype.name
#         self._path = None
#         self._hdf5_route = None
#         self._shm_name = self._handle.name  # Example: "psm_21467_46075"

#     def _from_shm(self, shm, shape, dtype):
#         """
#         enables init from shm address
#         """
#         if not isinstance(shm, str):
#             raise TypeError
#         self._address_type = self.SHM
#         self._handle = SharedMemory(name=shm)
#         self._data = np.ndarray(shape, dtype=dtype, buffer=self._handle.buf)
#         self._len = shape[0]
#         self._dtype = self._data.dtype.name
#         self._path = None
#         self._hdf5_route = None
#         self._shm_name = self._handle.name

#     def _from_hdf5(self, path,route):
#         """
#         enables init from hdf5 path and route.
#         """
#         if isinstance(path, str):
#             path = pathlib.Path(path)
#         if not isinstance(path, pathlib.Path):
#             raise TypeError(f"expected pathlib.Path, not {type(path)}")
#         if not path.exists():
#             raise FileNotFoundError(f"file not found: {path}")

#         if not isinstance(route,str):
#             raise TypeError(f"expected route as str, but got {type(route)}")
#         if not route.startswith('/'):
#             raise ValueError(f"route doesn't start at root.")

#         self._address_type = self.SHM
#         self._handle = h5py.File(path,'r')  # imported data is immutable.
#         self._data = self._handle[route] 
#         self._len = len(self._data)
#         self._dtype = self.data.dtype.name
#         self._path = path
#         self._hdf5_route = route
#         self._shm_name = None

#     def _shm_to_hdf5(self):
#         """ 
#         enables switch of existing data in shm to hdf5 in MemoryManager.cache_path
#         """
#         self._address_type = self.HDF5
#         self._handle = h5py.File(MemoryManager.cache_path, 'a')
#         self._hdf5_route = route = f"/{self.sha256sum}"

#         self._handle.create_dataset(route, data=self._data[:])
#         self._data = self._handle[route] 
#         self._len = len(self._data)
#         self._dtype = self.data.dtype.name
#         self._path = MemoryManager.cache_path
#         self._shm_name = None

#     @property
#     def use_disk(self):
#         return self._address_type == self.HDF5
    
#     def use_disk(self, value):
#         """
#         Enables the DataBlock to drop memory to disk (and reversely to memory)

#         value: boolean
#         """
#         if value is False:
#             if self._address_type == self.SHM:
#                 return  # nothing to do. Already in shm mode.
#             else:  # load from hdf5 to shm
#                 self._from_data(data=self._data[:])
#         else:  # if value is True:
#             if self._address_type == self.HDF5:
#                 return  # nothing to do. Already in HDF5 mode.
#             self._shm_to_hdf5()
            
#     @property
#     def sha256sum(self):
#         """
#         returns sha256sum (also the same as mem_id)
#         """
#         return self._mem_id

#     @sha256sum.setter
#     def sha256sum(self,value):
#         raise AttributeError("sha256sum is immutable.")

#     @property
#     def address(self):
#         """
#         enables the sending of information about the DataBlock to other processes.

#         returns SharedMemoryAddress
#         """
#         return SharedMemoryAddress(self.mem_id, self._data.shape, self._dtype, self._path, self._hdf5_route, self._shm_name)

#     @classmethod
#     def from_address(cls, shared_memory_address):
#         """
#         enables loading of the datablocks from a SharedMemoryAddress

#         shared_memory_address: class SharedMemoryAddress
#         """
#         if not isinstance(shared_memory_address, SharedMemoryAddress):
#             raise TypeError(f"expected SharedMemoryAddress, got {type(shared_memory_address)}")
#         sma = shared_memory_address
#         db = DataBlock(sma.mem_id, data=None, shm=sma.shm_name, shape=sma.shape, dtype=sma.dtype, path=sma.path, route=sma.hdf5_route)
#         return db

#     @property
#     def data(self):
#         return self._data[:]
    
#     @data.setter
#     def data(self, value):
#         raise AttributeError("DataBlock.data is immutable after init.")
    
#     def __len__(self) -> int:
#         return self._len
    
#     def __next__(self):
#         for value in self._data:
#             yield value

#     def __iter__(self):
#         return self

#     def __del__(self):
#         if self._address_type == self.SHM:
#             try:
#                 self._handle.close()
#             except AttributeError:
#                 print("handle was already closed.")
#         elif self._address_type == self.HDF5:
#             self._handle.close()
#         super().__del__()


# def intercept(A,B):
#     """
#     enables calculation of the intercept of two range objects.
#     Used to determine if a datablock contains a slice.
    
#     A: range
#     B: range
    
#     returns: range as intercept of ranges A and B.
#     """
#     if not isinstance(A, range):
#         raise TypeError
#     if A.step < 0: # turn the range around
#         A = range(A.stop, A.start, abs(A.step))

#     if not isinstance(B, range):
#         raise TypeError
#     if B.step < 0:  # turn the range around
#         B = range(B.stop, B.start, abs(B.step))
    
#     boundaries = [A.start, A.stop, B.start, B.stop]
#     boundaries.sort()
#     a,b,c,d = boundaries
#     if [A.start, A.stop] in [[a,b],[c,d]]:
#         return range(0) # then there is no intercept
#     # else: The inner range (subset) is b,c, limited by the first shared step.
#     A_start_steps = math.ceil((b - A.start) / A.step)
#     A_start = A_start_steps * A.step + A.start

#     B_start_steps = math.ceil((b - B.start) / B.step)
#     B_start = B_start_steps * B.step + B.start

#     if A.step == 1 or B.step == 1:
#         start = max(A_start,B_start)
#         step = B.step if A.step==1 else A.step
#         end = c
#     else:
#         intersection = set(range(A_start, c, A.step)).intersection(set(range(B_start, c, B.step)))
#         if not intersection:
#             return range(0)
#         start = min(intersection)
#         end = max(c, max(intersection))
#         intersection.remove(start)
#         step = min(intersection) - start
    
#     return range(start, end, step)


# def normalize_slice(length, item=None):  # There's an outdated version sitting in utils.py
#     """
#     helper: transforms slice into range inputs
#     returns start,stop,step
#     """
#     if item is None:
#         item = slice(0, length, 1)
#     assert isinstance(item, slice)
    
#     stop = length if item.stop is None else item.stop
#     start = 0 if item.start is None else length + item.start if item.start < 0 else item.start
#     start, stop = min(start,stop), max(start,stop)
#     step = 1 if item.step is None else item.step

#     return start, stop, step



# class ManagedColumn(MemoryManagedObject):  # Almost behaves like a list.
#     _ids = count()
#     def __init__(self) -> None:
#         super().__init__(mem_id=f"MC-{next(self._ids)}")

#         self.order = []  # strict order of datablocks.
#         self.dtype = None
    
#     def __str__(self) -> str:
#         n = self.__class__.__name__
#         return f"<{n} ({self._ids})> {len(self)} values ({self.dtype})"

#     def __repr__(self) -> str:
#         return self.__str__()

#     def __len__(self):
#         return sum(len(MemoryManager.get(block_id)) for block_id in self.order)

#     def __eq__(self, other: object) -> bool:
#         if isinstance(other, ManagedColumn):
#             return self.order == other.order
#         elif isiterable(other):
#             return not any(a!=b for a,b in zip(self,other))
#         else:
#             raise TypeError(f"can't compare {type(self)} with {type(other)}")

#     def __del__(self):
#         MemoryManager.unlink_tree(self)
#         super().__del__()

#     def __iter__(self):
#         for v in self.data:
#             yield v
    
#     @property
#     def data(self):
#         np_arrays = [MemoryManager.get(block_id).data for block_id in self.order]
#         return np.concatenate(np_arrays)

#     def __setitem__(self, key, value):
#         """
#         Enables update of values.

#         New User: __setitem__ is slow.
#         Old User: Don't use it then.
#         New User: But How am I supposed to update hundreds of values?
#         Old User: Don't update. Create a np.array and update that. Then replace the columns.
#         """
#         if isinstance(key,int):  # It's a single value update.
#             if key > len(self):
#                 raise IndexError(f"{key} > len")
#             if key < 0:
#                 key = len(self) + key
            
#             page_start = 0
#             for ix, block_id in enumerate(self.order):
#                 block = MemoryManager.get(block_id)
#                 page_end = page_start + len(block)
#                 if not page_start <= key < page_end:  
#                     page_start = page_end
#                 else:  # block found.
#                     data = np.array(block.data)
#                     offset_into_block = key - page_start
#                     data[offset_into_block] = value
#                     self._update_block(data,index=ix)
#                     break
                    
#         elif isinstance(key, slice) and isiterable(value):  # it's a slice update
#             r = range(*normalize_slice(length=len(self), item=key))
#             page_start = 0
#             for ix, block_id in enumerate(self.order):
#                 if page_start > r.stop:  # passed the last update.
#                     break

#                 block = MemoryManager.get(block_id)
#                 page_end = page_start + len(block)
                
#                 if page_end < r.start:  # way before the update starts.
#                     page_start = page_end
#                     continue

#                 itcpt = intercept(r, range(page_start, page_end, 1))
#                 if len(itcpt) == 0:
#                     continue
                
#                 new_values = np.array(block)
#                 for step_no, step in enumerate(itcpt, start=[page_start]):
#                     new_values[step] = value[step_no]
#                 self._update_block(new_values, index=ix)
#         else:
#             raise KeyError(f"can't update {key} with {value}")

#     def __getitem__(self, item):
#         """
#         returns a value or a ManagedColumn (slice).
#         """
#         if isinstance(item, slice):
#             mc = ManagedColumn()  # to be returned.

#             r = range(*normalize_slice(len(self), item))
#             page_start = 0
#             for block_id in self.order:
#                 if page_start > r.stop:
#                     break
#                 block = MemoryManager.get(block_id)
#                 if page_start + len(block) < r.start:
#                     page_start += len(block)
#                     continue

#                 if r.step==1:
#                     if r.start <= page_start and page_start + len(block) <= r.stop: # then we take the whole block.
#                         mc.extend(block.data)
#                         page_start += len(block)
#                         continue
#                     else:
#                         pass # the block doesn't match.
                
#                 block_range = range(page_start, page_start+len(block))
#                 intercept_range = intercept(r,block_range)  # very effective!
#                 if len(intercept_range)==0:  # no match.
#                     page_start += len(block)
#                     continue

#                 # x = {i for i in intercept_range}  # TODO: Candidate for TaskManager.
#                 # mask = np.array([i in x for i in block_range])
#                 # new_block = block.data[np.where(mask)]
#                 sr = slice(intercept_range.start, intercept_range.stop, intercept_range.step)
#                 new_block = block.data[sr]
#                 mc.extend(new_block)
#                 page_start += len(block)

#             return mc
            
#         elif isinstance(item, int):
#             page_start = 0
#             for block_id in self.order:
#                 block = MemoryManager.get(block_id)
#                 page_end = page_start + len(block)
#                 if page_start <= item < page_end:
#                     ix = item-page_start
#                     return block.data[ix]
#                 else:
#                     page_start = page_end
#         else:
#             raise KeyError(f"{item}")

#     @property
#     def address(self):
#         """
#         Returns list of data block addresses. See DataBlockAddress
#         """
#         L = []
#         for block_id in self.order:
#             datablock = MemoryManager.get(block_id)
#             assert isinstance(datablock, DataBlock)
#             L.append(datablock.address)
#         assert all(isinstance(i, SharedMemoryAddress) for i in L)
#         return L

#     @classmethod
#     def from_address_data(cls, address_data):
#         """
#         Creates ManagedColumn from list of data block addresses. 
#         """
#         if not isinstance(address_data, list):
#             raise TypeError
#         if not all(isinstance(i, SharedMemoryAddress) for i in address_data):
#             raise TypeError

#         mc = ManagedColumn()
#         for data_block_address in address_data:
#             mc.extend(data_block_address)
#         return mc

#     def _dtype_check(self, other):
#         assert isinstance(other, (np.ndarray, ManagedColumn))
#         if self.dtype is None:
#             self.dtype = other.dtype
#         elif self.dtype != other.dtype:
#             raise TypeError(f"the column expects {self.dtype}, but received {other.dtype}.")
#         else:
#             pass

#     def extend(self, data):
#         """
#         Extends ManagedColumn with data
#         """
#         if isinstance(data, ManagedColumn):  # It's data we've seen before.
#             self._dtype_check(data)

#             self.order.extend(data.order[:])
#             for block_id in data.order:
#                 block = MemoryManager.get(block_id)
#                 MemoryManager.link(self, block)
        
#         elif isinstance(data, DataBlock):  # It's from Table.load_file(...)
#             self.order.append(data.mem_id)
#             MemoryManager.link(self, data)  

#         elif isinstance(data, SharedMemoryAddress):
#             block = MemoryManager.get(mem_id=data.mem_id, default=DataBlock.from_address(data))  # get or create!

#             self.order.append(data.mem_id)
#             MemoryManager.link(self, block)  # Add link from Column to DataBlock
            
#         else:  # It's supposedly new data.
#             if not isinstance(data, np.ndarray):
#                 data = np.array(data)
#             self._update_block(data)
    
#     def _update_block(self, data, index=None):
#         """
#         Helper that enables update of managed blocks.
#         """
#         if not isinstance(data, np.ndarray):
#             raise TypeError
#         self._dtype_check(data)

#         m = hashlib.sha256()  # let's check if it really is new data...
#         m.update(data.data.tobytes())
#         sha256sum = m.hexdigest()
#         if sha256sum in MemoryManager.registry:  # ... not new!
#             block = MemoryManager.registry.get(sha256sum)
#         else:  # ... it's new!
#             block = DataBlock(mem_id=sha256sum, data=data)
#         # ok. solved. Now create links.
#         if index is None:
#             self.order.append(block.mem_id)
#         else:
#             old_block_id = self.order[index]
#             if self.order.count(old_block_id) == 1:  # check for unlinking.
#                 MemoryManager.unlink(self,old_block_id)
#             self.order[index] = block.mem_id
#         MemoryManager.link(self, block)  # Add link from Column to DataBlock

#     def append(self, value):
#         """
#         Disabled. Append items is slow. Use extend on a batch instead
#         """
#         raise AttributeError("Append items is slow. Use extend on a batch instead")
        

class Table(MemoryManagedObject):
    _ids = count()
    def __init__(self) -> None:
        super().__init__(mem_id=f"T-{next(self._ids)}")
        self.columns = {}
    
    # def __len__(self) -> int:
    #     if not self.columns:
    #         return 0
    #     else:
    #         return max(len(mc) for mc in self.columns.values())
    
    # def __del__(self):
    #     MemoryManager.unlink_tree(self)  # columns are automatically garbage collected.
    #     super().__del__()

    # def __getitem__(self, items):
    #     """
    #     Enables selection of columns and rows
    #     Examples: 

    #         table['a']   # selects column 'a'
    #         table[:10]   # selects first 10 rows from all columns
    #         table['a','b', slice(3:20:2)]  # selects a slice from columns 'a' and 'b'
    #         table['b', 'a', 'a', 'c', 2:20:3]  # selects column 'b' and 'c' and 'a' twice for a slice.

    #     returns values in same order as selection.
    #     """
    #     if isinstance(items, slice):
    #         names, slc = list(self.columns.keys()), items
    #     else:        
    #         names, slc = [], slice(len(self))
    #         for i in items:
    #             if isinstance(i,slice):
    #                 slc = i
    #             elif isinstance(i, str) and i in self.columns:
    #                 names.append(i)
    #             else:
    #                 raise KeyError(f"{i} is not a slice and not in column names")
    #     if not names:
    #         raise ValueError("No columns?")
        
    #     if len(names)==1 and names[0] in self.columns:
    #         return self.columns[names[0]]  # it's a single value
    #     else:  # it's a table query
    #         t = Table()
    #         for name in names:
    #             mc = self.columns[name]
    #             t.add_column(name, data=mc[slc])
    #         return t       

    # def __delitem__(self, item):
    #     if isinstance(item, str):
    #         mc = self.columns[item]
    #         del self.columns[item]
    #         MemoryManager.unlink(self, mc)
    #         MemoryManager.unlink_tree(mc)
    #     elif isinstance(item, slice):
    #         raise NotImplementedError("It might be smarter to create a new table using filter or using an index")  # TODO.
    #     else:
    #         raise TypeError(f"del using {type(item)}?")

    # def del_column(self, name):  # alias for symmetry to add_column
    #     self.__delitem__(name)
 
    # def add_column(self, name, data):
    #     if not isinstance(name, str):
    #         raise TypeError(f"expects column names to be str, not {type(name)}")
    #     if not isiterable(data):
    #         raise TypeError(f"data is plural of datum and means 'records'.\nExpected an iterable but got {type(data)}")

    #     mc = ManagedColumn()
    #     mc.extend(data)
    #     self.columns[name] = mc
    #     MemoryManager.link(self, mc)  # Add link from Table to Column
    
    # def __eq__(self, other) -> bool:  # TODO: Add tests for each condition.
    #     """
    #     enables comparison of self with other
    #     Example: TableA == TableB
    #     """
    #     if not isinstance(other, Table):
    #         a, b = self.__class__.__name__, other.__class__.__name__
    #         raise TypeError(f"cannot compare {a} with {b}")
        
    #     # fast simple checks.
    #     try:  
    #         self.compare(other)
    #     except (TypeError, ValueError):
    #         return False

    #     if len(self) != len(other):
    #         return False

    #     # the longer check.
    #     for name, mc in self.columns.items():
    #         mc2 = other.columns[name]
    #         if any(a!=b for a,b in zip(mc,mc2)):  # exit at the earliest possible option.
    #             return False
    #     return True

    # def __iadd__(self, other):
    #     """ 
    #     enables extension of self with data from other.
    #     Example: Table_1 += Table_2 
    #     """
    #     self.compare(other)
    #     for name,mc in self.columns.items():
    #         mc.extend(other.columns[name])
    #     return self
    # def __add__(self, other):
    #     """
    #     returns the joint extension of self and other
    #     Example:  Table_3 = Table_1 + Table_2 
    #     """
    #     self.compare(other)
    #     t = self.copy()
    #     t += other
    #     return t

    # def stack(self,other):  # TODO: Add tests.
    #     """
    #     returns the joint stack of tables
    #     Example:

    #     | Table A|  +  | Table B| = |  Table AB |
    #     | A| B| C|     | A| B| D|   | A| B| C| -|
    #                                 | A| B| -| D|
    #     """
    #     t = self.copy()
    #     for name,mc2 in other.columns.items():
    #         if name not in t.columns:
    #             t.add_column(name, data=[None] * len(self))
    #         mc = t.columns[name]
    #         mc.extend(mc2)
    #     for name, mc in t.columns.items():
    #         if name not in other.columns:
    #             mc.extend(data=[None]*len(other))
    #     return t

    # def __mul__(self, other):
    #     """
    #     enables repetition of a table
    #     Example: Table_x_10 = table * 10
    #     """
    #     if not isinstance(other, int) and other > 0:
    #         raise TypeError(f"repetition of a table is only supported with positive integers")
    #     t = self.copy()
    #     for _ in range(other-1):  # minus, because the copy is the first.
    #         t += self
    #     return t

    # def compare(self,other):
    #     """
    #     compares the metadata of the two tables and raises on the first difference.
    #     """
    #     if not isinstance(other, Table):
    #         a, b = self.__class__.__name__, other.__class__.__name__
    #         raise TypeError(f"cannot compare type {b} with {a}")
    #     for a, b in [[self, other], [other, self]]:  # check both dictionaries.
    #         for name, col in a.columns.items():
    #             if name not in b.columns:
    #                 raise ValueError(f"Column {name} not in other")
    #             col2 = b.columns[name]
    #             if col.dtype != col2.dtype:
    #                 raise ValueError(f"Column {name}.datatype different: {col.dtype}, {col2.dtype}")

    # def copy(self):
    #     """
    #     returns a copy of the table
    #     """
    #     t = Table()
    #     for name, mc in self.columns.items():
    #         t.add_column(name, mc)
    #     return t

    # def rename_column(self, old, new):
    #     """
    #     renames existing column from old name to new name
    #     """
    #     if old not in self.columns:
    #         raise ValueError(f"'{old}' doesn't exist. See Table.columns ")
    #     if new in self.columns:
    #         raise ValueError(f"'{new}' is already in use.")
    #     self.columns[new] = self.columns[old]
    #     del self.columns[old]

    # def __iter__(self):
    #     """
    #     Disabled. Users should use Table.rows or Table.columns
    #     """
    #     raise AttributeError("use Table.rows or Table.columns")

    # def __setitem__(self, key, value):
    #     if isinstance(key, str):  # it's a column name.
    #         pass
    #     elif isinstance(key, tuple) and all(isinstance(i,str) for i in key):
    #         pass  # it's a tuple of column names.
    #     else:
    #         raise TypeError(f"Bad key type: {key}, expected str or tuple of strings")

    #     if isinstance(key, str) and isiterable(value) and not isiterable(value[0]):
    #         pass  # it's a single assignment, like this:
    #         # table1['A'] = [1,2,3]
    #         # --> key = 'A', value = [1,2,3]
    #         self.add_column(key,value)

    #     elif isinstance(key, tuple) and isiterable(value) and all(isiterable(i) for i in value) and len(key)==len(value):
    #         pass  # it's a parallel assignment like this:
    #         # table1['A', 'B'] = [ [4, 5, 6], [1.1, 2.2, 3.3] ]
    #         # --> key = ('A', 'B'), value = [ [4, 5, 6], [1.1, 2.2, 3.3] ]
    #         for k,v in zip(key,value):
    #             self.add_column(k,v)
    #     else:
    #         raise ValueError(f"{value}")

    # @property
    # def rows(self):
    #     """
    #     enables iteration

    #     for row in Table.rows:
    #         print(row)
    #     """
    #     generators = [iter(mc) for mc in self.columns.values()]
    #     for _ in range(len(self)):
    #         yield [next(i) for i in generators]
    
    # def index(self, *keys):
    #     """ 
    #     Returns index on *keys columns as d[(key tuple, )] = {index1, index2, ...} 
    #     """
    #     idx = defaultdict(set)
    #     generator = self.__getitem__(*keys)
    #     for ix, key in enumerate(generator.rows):
    #         idx[key].add(ix)
    #     return idx

    # def filter(self, columns, filter_type='all'):
    #     """
    #     enables filtering across columns for multiple criteria.
        
    #     columns: 
    #         list of tuples [('A',"==", 4), ('B',">", 2), ('C', "!=", 'B')]
    #         list of dicts [{'column':'A', 'criteria': "==", 'value': 4}, {'column':'B', ....}]
    #     """
    #     if not isinstance(columns, list):
    #         raise TypeError

    #     for column in columns:
    #         if isinstance(column, dict):
    #             if not len(column)==3:
    #                 raise ValueError
    #             x = {'column', 'criteria', 'value1', 'value2'}
    #             if not set(column.keys()).issubset(x):
    #                 raise ValueError
    #             if column['criteria'] not in filter_ops:
    #                 raise ValueError

    #         elif isinstance(column, tuple):
    #             if not len(column)==3:
    #                 raise ValueError
    #             A,c,B = column
    #             if c not in filter_ops:
    #                 raise ValueError
    #             if isinstance(A, str) and A in self.columns:
    #                 pass

    #         else:
    #             raise TypeError
        
    #     if not isinstance(filter_type, str):
    #         raise TypeError
    #     if not filter_type in {'all', 'any'}:
    #         raise ValueError

    #     # 1. if dataset < 1_000_000 rows: do the job single proc.
                
    #     if len(columns)==1 and len(self) < 1_000_000:
    #         # The logic here is that filtering requires:
    #         # 1. the overhead to start a sub process.
    #         # 2. the time to filter.
    #         # Too few processes and the time increases.
    #         # Too many processes and the time increases.
    #         # The optimal result is based on the "ideal work block size"
    #         # of appx. 1M field evaluations.
    #         # If there are 3 columns and 6M rows, then 18M evaluations are
    #         # required. This leads to 18M/1M = 18 processes. If I have 64 cores
    #         # the optimal assignment is 18 cores.
    #         #
    #         # If, in contrast, there are 5 columns and 40,000 rows, then 200k 
    #         # only requires 1 core. Hereby starting a subprocesses is pointless.
    #         #
    #         # This assumption is rendered somewhat void if (!) the subprocesses 
    #         # can be idle in sleep mode and not require the startup overhead.
    #         pass  # TODO

    #     # the results are to be gathered here:
    #     arr = np.zeros(shape=(len(columns), len(self)), dtype='?')
    #     result_array = SharedMemory(create=True, size=arr.nbytes)
    #     result_address = SharedMemoryAddress(mem_id=1, shape=arr.shape, dtype=arr.dtype, shm_name=result_array.name)
        
    #     # the task manager enables evaluation of a column per core,
    #     # which is assembled in the shared array.
    #     with TaskManager(cores=1) as tm: 
    #         tasks = []
    #         for ix, column in enumerate(columns):
    #             if isinstance(column, dict):
    #                 A, criteria, B = column["column"], column["criteria"], column["value"]
    #             else:
    #                 A, criteria, B = column

    #             if A in self.columns:
    #                 mc = self.columns[A]
    #                 A = mc.address
    #             else:  # it's just a value.
    #                 pass

    #             if B in self.columns:
    #                 mc = self.columns[B]
    #                 B = mc.address
    #             else:  # it's just a value.
    #                 pass 

    #             if criteria not in filter_ops:
    #                 criteria = filter_ops_from_text.get(criteria)

    #             blocksize = math.ceil(len(self) / tm._cpus)
    #             for block in range(0, len(self), blocksize):
    #                 slc = slice(block, block+blocksize,1)
    #                 task = Task(filter, A, criteria, B, destination=result_address, destination_index=ix, slice_=slc)
    #                 tasks.append(task)

    #         _ = tm.execute(tasks)  # tm.execute returns the tasks with results, but we don't really care as the result is in the result array.

    #         # new blocks:
    #         blocksize = math.ceil(len(self) / (4*tm._cpus))
    #         tasks = []
    #         for block in range(0, len(self), blocksize):
    #             slc = slice(block, block+blocksize,1)
    #             # merge(source=self.address, mask=result_address, filter_type=filter_type, slice_=slc)
    #             task = Task(f=merge, source=self.address, mask=result_address, filter_type=filter_type, slice_=slc)
    #             tasks.append(task)
                
    #         results = tm.execute(tasks)  # tasks.result contain return the shm address
    #         results.sort(key=lambda x: x.task_id)

    #     table_true, table_false = None, None
    #     for task in results:
    #         true_address, false_address = task.result
    #         if table_true is None:
    #             table_true = Table.from_address(true_address)
    #             table_false = Table.from_address(false_address)
    #         else:
    #             table_true += Table.from_address(true_address)
    #             table_false += Table.from_address(false_address)
            
    #     return table_true, table_false
    
    # def sort_index(self, **kwargs):  # TODO: This is slow single core code.
    #     """ Helper for methods `sort` and `is_sorted` """
    #     if not isinstance(kwargs, dict):
    #         raise ValueError("Expected keyword arguments")
    #     if not kwargs:
    #         kwargs = {c: False for c in self.columns}
        
    #     for k, v in kwargs.items():
    #         if k not in self.columns:
    #             raise ValueError(f"no column {k}")
    #         if not isinstance(v, bool):
    #             raise ValueError(f"{k} was mapped to {v} - a non-boolean")
    #     none_substitute = float('-inf')

    #     rank = {i: tuple() for i in range(len(self))}
    #     for key in kwargs:
    #         unique_values = {v: 0 for v in self.columns[key] if v is not None}
    #         for r, v in enumerate(sorted(unique_values, reverse=kwargs[key])):
    #             unique_values[v] = r
    #         for ix, v in enumerate(self.columns[key]):
    #             rank[ix] += (unique_values.get(v, none_substitute),)

    #     new_order = [(r, i) for i, r in rank.items()]  # tuples are listed and sort...
    #     new_order.sort()
    #     sorted_index = [i for r, i in new_order]  # new index is extracted.

    #     rank.clear()  # free memory.
    #     new_order.clear()
    #     return sorted_index

    # def sort(self, **kwargs):  # TODO: This is slow single core code.
    #     """ Perform multi-pass sorting with precedence given order of column names.
    #     :param kwargs: keys: columns, values: 'reverse' as boolean.
    #     """
    #     sorted_index = self._sort_index(**kwargs)
    #     t = Table()
    #     for col_name, col in self.columns.items():
    #         t.add_column(col_name, data=[col[ix] for ix in sorted_index])
    #     return t

    # def is_sorted(self, **kwargs):  # TODO: This is slow single core code.
    #     """ Performs multi-pass sorting check with precedence given order of column names.
    #     :return bool
    #     """
    #     sorted_index = self._sort_index(**kwargs)
    #     if any(ix != i for ix, i in enumerate(sorted_index)):
    #         return False
    #     return True

    # def all(self, **kwargs):  # TODO: This is slow single core code.
    #     """
    #     returns Table for rows where ALL kwargs match
    #     :param kwargs: dictionary with headers and values / boolean callable
    #     """
    #     if not isinstance(kwargs, dict):
    #         raise TypeError("did you remember to add the ** in front of your dict?")
    #     if not all(k in self.columns for k in kwargs):
    #         raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in self.columns]}")

    #     ixs = None
    #     for k, v in kwargs.items():
    #         col = self.columns[k]
    #         if ixs is None:  # first header.
    #             if callable(v):
    #                 ix2 = {ix for ix, i in enumerate(col) if v(i)}
    #             else:
    #                 ix2 = {ix for ix, i in enumerate(col) if v == i}

    #         else:  # remaining headers.
    #             if callable(v):
    #                 ix2 = {ix for ix in ixs if v(col[ix])}
    #             else:
    #                 ix2 = {ix for ix in ixs if v == col[ix]}

    #         if not isinstance(ixs, set):
    #             ixs = ix2
    #         else:
    #             ixs = ixs.intersection(ix2)

    #         if not ixs:  # There are no matches.
    #             break

    #     t = Table()
    #     for col in tqdm(self.columns.values(), total=len(self.columns), desc="columns"):
    #         t.add_column(col.header, col.datatype, col.allow_empty, data=[col[ix] for ix in ixs])
    #     return t

    # def any(self, **kwargs):  # TODO: This is slow single core code.
    #     """
    #     returns Table for rows where ANY kwargs match
    #     :param kwargs: dictionary with headers and values / boolean callable
    #     """
    #     if not isinstance(kwargs, dict):
    #         raise TypeError("did you remember to add the ** in front of your dict?")

    #     ixs = set()
    #     for k, v in kwargs.items():
    #         col = self.columns[k]
    #         if callable(v):
    #             ix2 = {ix for ix, r in enumerate(col) if v(r)}
    #         else:
    #             ix2 = {ix for ix, r in enumerate(col) if v == r}
    #         ixs.update(ix2)

    #     t = Table()
    #     for col in tqdm(self.columns.values(), total=len(self.columns), desc="columns"):
    #         t.add_column(col.header, col.datatype, col.allow_empty, data=[col[ix] for ix in ixs])
    #     return t

    # def groupby(self, keys, functions, pivot_on=None):  # TODO: This is slow single core code.
    #     """
    #     :param keys: headers for grouping
    #     :param functions: list of headers and functions.
    #     :return: GroupBy class
    #     Example usage:
    #         from tablite import Table
    #         t = Table()
    #         t.add_column('date', data=[1,1,1,2,2,2])
    #         t.add_column('sku', data=[1,2,3,1,2,3])
    #         t.add_column('qty', data=[4,5,4,5,3,7])
    #         from tablite import GroupBy, Sum
    #         g = t.groupby(keys=['sku'], functions=[('qty', Sum)])
    #         g.tablite.show()
    #     """
    #     g = GroupBy(keys=keys, functions=functions)
    #     g += self
    #     if pivot_on:
    #         g.pivot(pivot_on)
    #     return g.table()
    
    # def _join_type_check(self, other, left_keys, right_keys, left_columns, right_columns):
    #     if not isinstance(other, Table):
    #         raise TypeError(f"other expected other to be type Table, not {type(other)}")

    #     if not isinstance(left_keys, list) and all(isinstance(k, str) for k in left_keys):
    #         raise TypeError(f"Expected keys as list of strings, not {type(left_keys)}")
    #     if not isinstance(right_keys, list) and all(isinstance(k, str) for k in right_keys):
    #         raise TypeError(f"Expected keys as list of strings, not {type(right_keys)}")

    #     if any(key not in self.columns for key in left_keys):
    #         raise ValueError(f"left key(s) not found: {[k for k in left_keys if k not in self.columns]}")
    #     if any(key not in other.columns for key in right_keys):
    #         raise ValueError(f"right key(s) not found: {[k for k in right_keys if k not in other.columns]}")

    #     if len(left_keys) != len(right_keys):
    #         raise ValueError(f"Keys do not have same length: \n{left_keys}, \n{right_keys}")

    #     for L, R in zip(left_keys, right_keys):
    #         Lcol, Rcol = self.columns[L], other.columns[R]
    #         if Lcol.datatype != Rcol.datatype:
    #             raise TypeError(f"{L} is {Lcol.datatype}, but {R} is {Rcol.datatype}")

    #     if not isinstance(left_columns, list) or not left_columns:
    #         raise TypeError("left_columns (list of strings) are required")
    #     if any(column not in self for column in left_columns):
    #         raise ValueError(f"Column not found: {[c for c in left_columns if c not in self.columns]}")

    #     if not isinstance(right_columns, list) or not right_columns:
    #         raise TypeError("right_columns (list or strings) are required")
    #     if any(column not in other for column in right_columns):
    #         raise ValueError(f"Column not found: {[c for c in right_columns if c not in other.columns]}")
    #     # Input is now guaranteed to be valid.

    # def join(self, other, left_keys, right_keys, left_columns, right_columns, kind='inner'):
    #     """
    #     short-cut for all join functions.
    #     """
    #     kinds = {
    #         'inner':self.inner_join,
    #         'left':self.left_join,
    #         'outer':self.outer_join
    #     }
    #     if kind not in kinds:
    #         raise ValueError(f"join type unknown: {kind}")
    #     f = kinds.get(kind,None)
    #     return f(self,other,left_keys,right_keys,left_columns,right_columns)
    
    # def left_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None):  # TODO: This is slow single core code.
    #     """
    #     :param other: self, other = (left, right)
    #     :param left_keys: list of keys for the join
    #     :param right_keys: list of keys for the join
    #     :param left_columns: list of left columns to retain, if None, all are retained.
    #     :param right_columns: list of right columns to retain, if None, all are retained.
    #     :return: new Table
    #     Example:
    #     SQL:   SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
    #     Tablite: left_join = numbers.left_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])
    #     """
    #     if left_columns is None:
    #         left_columns = list(self.columns)
    #     if right_columns is None:
    #         right_columns = list(other.columns)

    #     self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

    #     left_join = Table(use_disk=self._use_disk)
    #     for col_name in left_columns:
    #         col = self.columns[col_name]
    #         left_join.add_column(col_name, col.datatype, allow_empty=True)

    #     right_join_col_name = {}
    #     for col_name in right_columns:
    #         col = other.columns[col_name]
    #         revised_name = left_join.check_for_duplicate_header(col_name)
    #         right_join_col_name[revised_name] = col_name
    #         left_join.add_column(revised_name, col.datatype, allow_empty=True)

    #     left_ixs = range(len(self))
    #     right_idx = other.index(*right_keys)

    #     for left_ix in tqdm(left_ixs, total=len(left_ixs)):
    #         key = tuple(self[h][left_ix] for h in left_keys)
    #         right_ixs = right_idx.get(key, (None,))
    #         for right_ix in right_ixs:
    #             for col_name, column in left_join.columns.items():
    #                 if col_name in self:
    #                     column.append(self[col_name][left_ix])
    #                 elif col_name in right_join_col_name:
    #                     original_name = right_join_col_name[col_name]
    #                     if right_ix is not None:
    #                         column.append(other[original_name][right_ix])
    #                     else:
    #                         column.append(None)
    #                 else:
    #                     raise Exception('bad logic')
    #     return left_join

    # def inner_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None):  # TODO: This is slow single core code.
    #     """
    #     :param other: self, other = (left, right)
    #     :param left_keys: list of keys for the join
    #     :param right_keys: list of keys for the join
    #     :param left_columns: list of left columns to retain, if None, all are retained.
    #     :param right_columns: list of right columns to retain, if None, all are retained.
    #     :return: new Table
    #     Example:
    #     SQL:   SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
    #     Tablite: inner_join = numbers.inner_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])
    #     """
    #     if left_columns is None:
    #         left_columns = list(self.columns)
    #     if right_columns is None:
    #         right_columns = list(other.columns)

    #     self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

    #     inner_join = Table(use_disk=self._use_disk)
    #     for col_name in left_columns:
    #         col = self.columns[col_name]
    #         inner_join.add_column(col_name, col.datatype, allow_empty=True)

    #     right_join_col_name = {}
    #     for col_name in right_columns:
    #         col = other.columns[col_name]
    #         revised_name = inner_join.check_for_duplicate_header(col_name)
    #         right_join_col_name[revised_name] = col_name
    #         inner_join.add_column(revised_name, col.datatype, allow_empty=True)

    #     key_union = set(self.filter(*left_keys)).intersection(set(other.filter(*right_keys)))

    #     left_ixs = self.index(*left_keys)
    #     right_ixs = other.index(*right_keys)

    #     for key in tqdm(sorted(key_union), total=len(key_union)):
    #         for left_ix in left_ixs.get(key, set()):
    #             for right_ix in right_ixs.get(key, set()):
    #                 for col_name, column in inner_join.columns.items():
    #                     if col_name in self:
    #                         column.append(self[col_name][left_ix])
    #                     else:  # col_name in right_join_col_name:
    #                         original_name = right_join_col_name[col_name]
    #                         column.append(other[original_name][right_ix])

    #     return inner_join

    # def outer_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None):  # TODO: This is slow single core code.
    #     """
    #     :param other: self, other = (left, right)
    #     :param left_keys: list of keys for the join
    #     :param right_keys: list of keys for the join
    #     :param left_columns: list of left columns to retain, if None, all are retained.
    #     :param right_columns: list of right columns to retain, if None, all are retained.
    #     :return: new Table
    #     Example:
    #     SQL:   SELECT number, letter FROM numbers OUTER JOIN letters ON numbers.colour == letters.color
    #     Tablite: outer_join = numbers.outer_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])
    #     """
    #     if left_columns is None:
    #         left_columns = list(self.columns)
    #     if right_columns is None:
    #         right_columns = list(other.columns)

    #     self._join_type_check(other, left_keys, right_keys, left_columns, right_columns)  # raises if error

    #     outer_join = Table(use_disk=self._use_disk)
    #     for col_name in left_columns:
    #         col = self.columns[col_name]
    #         outer_join.add_column(col_name, col.datatype, allow_empty=True)

    #     right_join_col_name = {}
    #     for col_name in right_columns:
    #         col = other.columns[col_name]
    #         revised_name = outer_join.check_for_duplicate_header(col_name)
    #         right_join_col_name[revised_name] = col_name
    #         outer_join.add_column(revised_name, col.datatype, allow_empty=True)

    #     left_ixs = range(len(self))
    #     right_idx = other.index(*right_keys)
    #     right_keyset = set(right_idx)

    #     for left_ix in tqdm(left_ixs, total=left_ixs.stop, desc="left side outer join"):
    #         key = tuple(self[h][left_ix] for h in left_keys)
    #         right_ixs = right_idx.get(key, (None,))
    #         right_keyset.discard(key)
    #         for right_ix in right_ixs:
    #             for col_name, column in outer_join.columns.items():
    #                 if col_name in self:
    #                     column.append(self[col_name][left_ix])
    #                 elif col_name in right_join_col_name:
    #                     original_name = right_join_col_name[col_name]
    #                     if right_ix is not None:
    #                         column.append(other[original_name][right_ix])
    #                     else:
    #                         column.append(None)
    #                 else:
    #                     raise Exception('bad logic')

    #     for right_key in tqdm(right_keyset, total=len(right_keyset), desc="right side outer join"):
    #         for right_ix in right_idx[right_key]:
    #             for col_name, column in outer_join.columns.items():
    #                 if col_name in self:
    #                     column.append(None)
    #                 elif col_name in right_join_col_name:
    #                     original_name = right_join_col_name[col_name]
    #                     column.append(other[original_name][right_ix])
    #                 else:
    #                     raise Exception('bad logic')
    #     return outer_join

    # def lookup(self, other, *criteria, all=True):  # TODO: This is slow single core code.
    #     """ function for looking up values in other according to criteria
    #     :param: other: Table
    #     :param: criteria: Each criteria must be a tuple with value comparisons in the form:
    #         (LEFT, OPERATOR, RIGHT)
    #     :param: all: boolean: True=ALL, False=Any
    #     OPERATOR must be a callable that returns a boolean
    #     LEFT must be a value that the OPERATOR can compare.
    #     RIGHT must be a value that the OPERATOR can compare.
    #     Examples:
    #           ('column A', "==", 'column B')  # comparison of two columns
    #           ('Date', "<", DataTypes.date(24,12) )  # value from column 'Date' is before 24/12.
    #           f = lambda L,R: all( ord(L) < ord(R) )  # uses custom function.
    #           ('text 1', f, 'text 2')
    #           value from column 'text 1' is compared with value from column 'text 2'
    #     """
    #     assert isinstance(self, Table)
    #     assert isinstance(other, Table)

    #     all = all
    #     any = not all

    #     def not_in(a, b):
    #         return not operator.contains(a, b)

    #     ops = {
    #         "in": operator.contains,
    #         "not in": not_in,
    #         "<": operator.lt,
    #         "<=": operator.le,
    #         ">": operator.gt,
    #         ">=": operator.ge,
    #         "!=": operator.ne,
    #         "==": operator.eq,
    #     }

    #     table3 = Table(use_disk=self._use_disk)
    #     for name, col in chain(self.columns.items(), other.columns.items()):
    #         table3.add_column(name, col.datatype, allow_empty=True)

    #     functions, left_columns, right_columns = [], set(), set()

    #     for left, op, right in criteria:
    #         left_columns.add(left)
    #         right_columns.add(right)
    #         if callable(op):
    #             pass  # it's a custom function.
    #         else:
    #             op = ops.get(op, None)
    #             if not callable(op):
    #                 raise ValueError(f"{op} not a recognised operator for comparison.")

    #         functions.append((op, left, right))

    #     lru_cache = {}
    #     empty_row = tuple(None for _ in other.columns)

    #     for row1 in tqdm(self.rows, total=self.__len__()):
    #         row1_tup = tuple(v for v, name in zip(row1, self.columns) if name in left_columns)
    #         row1d = {name: value for name, value in zip(self.columns, row1) if name in left_columns}

    #         match_found = True if row1_tup in lru_cache else False

    #         if not match_found:  # search.
    #             for row2 in other.rows:
    #                 row2d = {name: value for name, value in zip(other.columns, row2) if name in right_columns}

    #                 evaluations = [op(row1d.get(left, left), row2d.get(right, right)) for op, left, right in functions]
    #                 # The evaluations above does a neat trick:
    #                 # as L is a dict, L.get(left, L) will return a value
    #                 # from the columns IF left is a column name. If it isn't
    #                 # the function will treat left as a value.
    #                 # The same applies to right.

    #                 if all and not False in evaluations:
    #                     match_found = True
    #                     lru_cache[row1_tup] = row2
    #                     break
    #                 elif any and True in evaluations:
    #                     match_found = True
    #                     lru_cache[row1_tup] = row2
    #                     break
    #                 else:
    #                     continue

    #         if not match_found:  # no match found.
    #             lru_cache[row1_tup] = empty_row

    #         new_row = row1 + lru_cache[row1_tup]

    #         table3.add_row(new_row)

    #     return table3
    
    # def pivot_table(self, *args):
    #     raise NotImplementedError

    # def show(self, *args, blanks=None, format='ascii'):
    #     """
    #     prints a _preview_ of the table.
        
    #     blanks: string to replace blanks (None is default) when shown.
    #     formats: 
    #       - 'ascii' --> ASCII (see also self.to_ascii)
    #       - 'md' --> markdown (see also self.to_markdown)
    #       - 'html' --> HTML (see also self.to_html)

    #     """
    #     converters = {
    #         'ascii': self.to_ascii,
    #         'md': self.to_markdown,
    #         'html': self.to_html
    #     }
    #     converter = converters.get(format, None)
        
    #     if converter is None:
    #         raise ValueError(f"format={format} not in known formats: {list(converters)}")
   
    #     slc = slice(0,min(len(self),20),1) if len(self) < 20 else None  # default slice 
    #     for arg in args:  # override by user defined slice (if provided)
    #         if isinstance(arg, slice):
    #             slc = slice(*normalize_slice(len(self), arg))
    #         break
        
    #     if slc:
    #         t = Table()
    #         t.add_column('#', data=[str(i) for i in range(slc.start, slc.stop, slc.step)])
    #         for n, mc in self.columns.items():
    #             t.add_column(n,data=[str(i) for i in mc[slc] ])
    #     else:
    #         t,n = Table(), len(self)
    #         t.add_column('#', data=[str(i) for i in range(7)] + ["..."] + [str(i) for i in range(n-7, n)])
    #         for name, mc in self.columns.items():
    #             data = [str(i) for i in mc[:7]] + ["..."] + [str(i) for i in mc[-7:]]
    #             t.add_column(name, data)

    #     print(converter(t, blanks))

    # @staticmethod
    # def to_ascii(table, blanks):
    #     """
    #     enables viewing in terminals
    #     returns the table as ascii string
    #     """
    #     widths = {}
    #     names = list(table.columns)
    #     for name,mc in table.columns.items():
    #         widths[name] = max([len(name), len(str(mc.dtype))] + [len(str(v)) for v in mc])

    #     def adjust(v, length):
    #         if v is None:
    #             return str(blanks).ljust(length)
    #         elif isinstance(v, str):
    #             return v.ljust(length)
    #         else:
    #             return str(v).rjust(length)

    #     s = []
    #     s.append("+ " + "+".join(["=" * widths[n] for n in names]) + " +")
    #     s.append("| " + "|".join([n.center(widths[n], " ") for n in names]) + " |")
    #     s.append("| " + "|".join([str(table.columns[n].dtype).center(widths[n], " ") for n in names]) + " |")
    #     s.append("+ " + "+".join(["-" * widths[n] for n in names]) + " +")
    #     for row in table.rows:
    #         s.append("| " + "|".join([adjust(v, widths[n]) for v, n in zip(row, names)]) + " |")
    #     s.append("+ " + "+".join(["=" * widths[h] for h in names]) + " +")
    #     return "\n".join(s)

    # @staticmethod
    # def to_markdown(table, blanks):
    #     widths = {}
    #     names = list(table.columns)
    #     for name, mc in table.columns.items():
    #         widths[name] = max([len(name)] + [len(str(i)) for i in mc])
        
    #     def adjust(v, length):
    #         if v is None:
    #             return str(blanks).ljust(length)
    #         elif isinstance(v, str):
    #             return v.ljust(length)
    #         else:
    #             return str(v).rjust(length)

    #     s = []
    #     s.append("| " + "|".join([n.center(widths[n], " ") for n in names]) + " |")
    #     s.append("| " + "|".join(["-" * widths[n] for n in names]) + " |")
    #     for row in table.rows:
    #         s.append("| " + "|".join([adjust(v, widths[n]) for v, n in zip(row, names)]) + " |")
    #     return "\n".join(s)

    # @staticmethod
    # def to_html(table, blanks):
    #     raise NotImplemented("coming soon!")
           
    # @classmethod
    # def import_file(cls, path, 
    #     import_as, newline='\n', text_qualifier=None,
    #     delimiter=',', first_row_has_headers=True, columns=None, sheet=None):
    #     """
    #     reads path and imports 1 or more tables as hdf5

    #     path: pathlib.Path or str
    #     import_as: 'csv','xlsx','txt'                               *123
    #     newline: newline character '\n', '\r\n' or b'\n', b'\r\n'   *13
    #     text_qualifier: character: " or '                           +13
    #     delimiter: character: typically ",", ";" or "|"             *1+3
    #     first_row_has_headers: boolean                              *123
    #     columns: dict with column names or indices and datatypes    *123
    #         {'A': int, 'B': str, 'C': float, D: datetime}
    #         Excess column names are ignored.

    #     sheet: sheet name to import (e.g. 'sheet_1')                 *2
    #         sheets not found excess names are ignored.
    #         filenames will be {path}+{sheet}.h5
        
    #     (*) required, (+) optional, (1) csv, (2) xlsx, (3) txt, (4) h5

    #     TABLES FROM IMPORTED FILES ARE IMMUTABLE.
    #     OTHER TABLES EXIST IN MEMORY MANAGERs CACHE IF USE DISK == True
    #     """
    #     if isinstance(path, str):
    #         path = pathlib.Path(path)
    #     if not isinstance(path, pathlib.Path):
    #         raise TypeError(f"expected pathlib.Path, got {type(path)}")
    #     if not path.exists():
    #         raise FileNotFoundError(f"file not found: {path}")

    #     if not isinstance(import_as,str) and import_as in ['csv','txt','xlsx']:
    #         raise ValueError(f"{import_as} is not supported")
        
    #     # check the inputs.
    #     if import_as in {'xlsx'}:
    #         return excel_reader(path, sheet_name=sheet)
            
    #     if import_as in {'ods'}:
    #         return ods_reader(path, sheet_name=sheet)

    #     if import_as in {'csv', 'txt'}:
    #         h5 = pathlib.Path(str(path) + '.hdf5')
    #         if h5.exists():
    #             with h5py.File(h5,'r') as f:  # Create file, truncate if exists
    #                 stored_config = json.loads(f.attrs['config'])
    #         else:
    #             stored_config = {}

    #         file_length = path.stat().st_size  # 9,998,765,432 = 10Gb
    #         config = {
    #             'import_as': import_as,
    #             'path': str(path),
    #             'filesize': file_length,  # if this changes - re-import.
    #             'delimiter': delimiter,
    #             'columns': columns, 
    #             'newline': newline,
    #             'first_row_has_headers': first_row_has_headers,
    #             'text_qualifier': text_qualifier
    #         }

    #         skip = False
    #         for k,v in config.items():
    #             if stored_config.get(k,None) != v:
    #                 skip = False
    #                 break  # set skip to false and exit for loop.
    #             else:
    #                 skip = True
    #         if skip:
    #             print(f"file already imported as {h5}")  
    #             return Table.load_file(h5)  # <---- EXIT 1.

    #         # Ok. File doesn't exist, has been changed or it's a new import config.
    #         with path.open('rb') as fi:
    #             rawdata = fi.read(10000)
    #             encoding = chardet.detect(rawdata)['encoding']
            
    #         text_escape = TextEscape(delimiter=delimiter, qoute=text_qualifier)  # configure t.e.

    #         with path.open('r', encoding=encoding) as fi:
    #             for line in fi:
    #                 line = line.rstrip('\n')
    #                 break  # break on first
    #             headers = text_escape(line) # use t.e.
                
    #             if first_row_has_headers:    
    #                 for name in columns:
    #                     if name not in headers:
    #                         raise ValueError(f"column not found: {name}")
    #             else:
    #                 for index in columns:
    #                     if index not in range(len(headers)):
    #                         raise IndexError(f"{index} out of range({len(headers)})")

    #         with h5py.File(h5,'w') as f:  # Create file, truncate if exists
    #             f.attrs['config'] = json.dumps(config)

    #         with TaskManager() as tm:
    #             working_overhead = 5  # random guess. Calibrate as required.
    #             mem_per_cpu = tm.chunk_size_per_cpu(file_length * working_overhead)
    #             mem_per_task = mem_per_cpu // working_overhead  # 1 Gb / 10x = 100Mb
    #             n_tasks = math.ceil(file_length / mem_per_task)
                
    #             tr_cfg = {
    #                 "source":path, 
    #                 "destination":h5, 
    #                 "columns":columns, 
    #                 "newline":newline, 
    #                 "delimiter":delimiter, 
    #                 "first_row_has_headers":first_row_has_headers,
    #                 "qoute":text_qualifier,
    #                 "text_escape_openings":'', "text_escape_closures":'',
    #                 "start":None, "limit":mem_per_task,
    #                 "encoding":encoding
    #             }
                
    #             tasks = []
    #             for i in range(n_tasks):
    #                 # add task for each chunk for working
    #                 tr_cfg['start'] = i * mem_per_task
    #                 task = Task(f=text_reader, **tr_cfg)
    #                 tasks.append(task)
                
    #             tm.execute(tasks)
    #             # Merging chunks in hdf5 into single columns
    #             consolidate(h5)  # no need to task manager as this is done using
    #             # virtual layouts and virtual datasets.

    #             # Finally: Calculate sha256sum.
    #             tasks = []
    #             for column_name in columns:
    #                 task = Task(f=sha256sum, **{"path":h5, "column_name":column_name})
    #                 tasks.append(task)
    #             tm.execute(tasks)
    #         return Table.load_file(h5)  # <---- EXIT 2.

    # @classmethod
    # def inspect_h5_file(cls, path, group='/'):
    #     """
    #     enables inspection of contents of HDF5 file 
    #     path: str or pathlib.Path
    #     group: you can give a specific group, defaults to the root: '/'
    #     """
    #     def descend_obj(obj,sep='  ', offset=''):
    #         """
    #         Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    #         """
    #         if type(obj) in [h5py._hl.group.Group,h5py._hl.files.File]:
    #             if obj.attrs.keys():  
    #                 for k,v in obj.attrs.items():
    #                     print(offset, k,":",v)  # prints config
    #             for key in obj.keys():
    #                 print(offset, key,':',obj[key])  # prints groups
    #                 descend_obj(obj[key],sep=sep, offset=offset+sep)
    #         elif type(obj)==h5py._hl.dataset.Dataset:
    #             for key in obj.attrs.keys():
    #                 print(offset, key,':',obj.attrs[key])  # prints datasets.

    #     with h5py.File(path,'r') as f:
    #         print(f"{path} contents")
    #         descend_obj(f[group])

    # @classmethod
    # def load_file(cls, path):
    #     """
    #     enables loading of imported HDF5 file. 
    #     Import assumes that columns are in the HDF5 root as "/{column name}"

    #     :path: pathlib.Path
    #     """
    #     if isinstance(path, str):
    #         path = pathlib.Path(path)
    #     if not isinstance(path, pathlib.Path):
    #         raise TypeError(f"expected pathlib.Path, got {type(path)}")
    #     if not path.exists():
    #         raise FileNotFoundError(f"file not found: {path}")
    #     if not path.name.endswith(".hdf5"):
    #         raise TypeError(f"expected .hdf5 file, not {path.name}")
        
    #     # read the file and create managed columns
    #     # no need for task manager as this is just fetching metadata.
    #     t = Table()
    #     with h5py.File(path,'r') as f:  # 'r+' in case the sha256sum is missing.
    #         for name in f.keys():
    #             if name == HDF5_IMPORT_ROOT:
    #                 continue
    #             sha256sum = f[f"/{name}"].attrs.get('sha256sum',None)
    #             if sha256sum is None:
    #                 raise ValueError("no sha256sum?")
    #             db = DataBlock(mem_id=sha256sum, path=path, route=f"/{name}")
    #             t.add_column(name=name, data=db)
    #     return t

    # def to_hdf5(self, path):
    #     """
    #     creates a copy of the table as hdf5
    #     the hdf5 layout can be viewed using Table.inspect_h5_file(path/to.hdf5)
    #     """
    #     if isinstance(path, str):
    #         path = pathlib.Path(path)
        
    #     total = ":,".format(len(self.columns) * len(self))
    #     print(f"writing {total} records to {path}")

    #     with h5py.File(path, 'a') as f:
    #         with tqdm(total=len(self.columns), unit='columns') as pbar:
    #             n = 0
    #             for name, mc in self.columns.values():
    #                 f.create_dataset(name, data=mc[:])  # stored in hdf5 as '/name'
    #                 n += 1
    #                 pbar.update(n)
    #     print(f"writing {path} to HDF5 done")

    # @property
    # def address(self):
    #     """
    #     Enables the main process to package a table as shared memory instructions.

    #     The data layout is:
    #     table = {
    #         column_name: [ addresses ],  where each address is digestible by ManagedColumn.
    #     }

    #     """
    #     return {name: mc.address for name, mc in self.columns.items()}

    # @classmethod
    # def from_address(cls, address_data):
    #     """
    #     Enables a worker to load the table from shared memory.
    #     """
    #     if not isinstance(address_data, dict):
    #         raise TypeError

    #     t = Table()
    #     for name, address_list in address_data.items():
    #         mc = ManagedColumn.from_address_data(address_list)
    #         t.columns[name] = mc
    #     return t
        

# FILE READER UTILS 2.0 ----------------------------

# class TextEscape(object):
#     """
#     enables parsing of CSV with respecting brackets and text marks.

#     Example:
#     text_escape = TextEscape()  # set up the instance.
#     for line in somefile.readlines():
#         list_of_words = text_escape(line)  # use the instance.
#         ...
#     """
#     def __init__(self, openings='({[', closures=']})', qoute='"', delimiter=','):
#         """
#         As an example, the Danes and Germans use " for inches and ' for feet, 
#         so we will see data that contains nail (75 x 4 mm, 3" x 3/12"), so 
#         for this case ( and ) are valid escapes, but " and ' aren't.

#         """
#         if openings is None:
#             pass
#         elif isinstance(openings, str):
#             self.openings = {c for c in openings}
#         else:
#             raise TypeError(f"expected str, got {type(openings)}")           

#         if closures is None:
#             pass
#         elif isinstance(closures, str):
#             self.closures = {c for c in closures}
#         else:
#             raise TypeError(f"expected str, got {type(closures)}")
    
#         if not isinstance(delimiter, str):
#             raise TypeError(f"expected str, got {type(delimiter)}")
#         self.delimiter = delimiter
#         self._delimiter_length = len(delimiter)
        
#         if qoute is None:
#             pass
#         elif qoute in openings or qoute in closures:
#             raise ValueError("It's a bad idea to have qoute character appears in openings or closures.")
#         else:
#             self.qoute = qoute
        
#         if not qoute:
#             self.c = self._call1
#         elif not openings + closures:
#             self.c = self._call2
#         else:
#             # TODO: The regex below needs to be constructed dynamically depending on the inputs.
#             self.re = re.compile("([\d\w\s\u4e00-\u9fff]+)(?=,|$)|((?<=\A)|(?<=,))(?=,|$)|(\(.+\)|\".+\")", "gmu") # <-- Disclaimer: Audrius wrote this.
#             self.c = self._call3

#     def __call__(self,s):
#         return self.c(s)
       
#     def _call1(self,s):  # just looks for delimiter.
#         return s.split(self.delimiter)

#     def _call2(self,s): # looks for qoutes.
#         words = []
#         qoute= False
#         ix = 0
#         while ix < len(s):  
#             c = s[ix]
#             if c == self.qoute:
#                 qoute = not qoute
#             if qoute:
#                 ix += 1
#                 continue
#             if c == self.delimiter:
#                 word, s = s[:ix], s[ix+self._delimiter_length:]
#                 words.append(word)
#                 ix = -1
#             ix+=1
#         if s:
#             words.append(s)
#         return words

#     def _call3(self, s):  # looks for qoutes, openings and closures.
#         return self.re.match(s)  # TODO - TEST!
#         # words = []
#         # qoute = False
#         # ix,depth = 0,0
#         # while ix < len(s):  
#         #     c = s[ix]

#         #     if c == self.qoute:
#         #         qoute = not qoute

#         #     if qoute:
#         #         ix+=1
#         #         continue

#         #     if depth == 0 and c == self.delimiter:
#         #         word, s = s[:ix], s[ix+self._delimiter_length:]
#         #         words.append(word)
#         #         ix = -1
#         #     elif c in self.openings:
#         #         depth += 1
#         #     elif c in self.closures:
#         #         depth -= 1
#         #     else:
#         #         pass
#         #     ix += 1

#         # if s:
#         #     words.append(s)
#         # return words


# def detect_seperator(text):
#     """
#     After reviewing the logic in the CSV sniffer, I concluded that all it
#     really does is to look for a non-text character. As the separator is
#     determined by the first line, which almost always is a line of headers,
#     the text characters will be utf-8,16 or ascii letters plus white space.
#     This leaves the characters ,;:| and \t as potential separators, with one
#     exception: files that use whitespace as separator. My logic is therefore
#     to (1) find the set of characters that intersect with ',;:|\t' which in
#     practice is a single character, unless (2) it is empty whereby it must
#     be whitespace.
#     """
#     seps = {',', '\t', ';', ':', '|'}.intersection(text)
#     if not seps:
#         if " " in text:
#             return " "
#     else:
#         frq = [(text.count(i), i) for i in seps]
#         frq.sort(reverse=True)  # most frequent first.
#         return {k:v for k,v in frq}


# def _in(a,b):
#     """
#     enables filter function 'in'
#     """
#     return a.decode('utf-8') in b.decode('utf-8')


# filter_ops = {
#             ">": operator.gt,
#             ">=": operator.ge,
#             "==": operator.eq,
#             "<": operator.lt,
#             "<=": operator.le,
#             "!=": operator.ne,
#             "in": _in
#         }

# filter_ops_from_text = {
#     "gt": ">",
#     "gteq": ">=",
#     "eq": "==",
#     "lt": "<",
#     "lteq": "<=",
#     "neq": "!=",
#     "in": _in
# }

# def filter(source1, criteria, source2, destination, destination_index, slice_):
#     """ PARALLEL TASK FUNCTION
#     source1: list of addresses
#     criteria: logical operator
#     source1: list of addresses
#     destination: shm address name.
#     destination_index: integer.
#     """    
#     # 1. access the data sources.
#     if isinstance(source1, list):
#         A = ManagedColumn()
#         for address in source1:
#             datablock = DataBlock.from_address(address)
#             A.extend(datablock)
#         sliceA = A[slice_]

#         A_is_data = True
#     else:
#         A_is_data = False  # A is value
    
#     if isinstance(source2, list):
#         B = ManagedColumn()
#         for address in source2:
#             datablock = DataBlock.from_address(address)
#             B.extend(datablock)
#         sliceB = B[slice_]

#         B_is_data = True
#     else:
#         B_is_data = False  # B is a value.

#     assert isinstance(destination, SharedMemoryAddress)
#     handle, data = destination.to_shm()  # the handle is required to sit idle as gc otherwise deletes it.
#     assert destination_index < len(data),  "len of data is the number of evaluations, so the destination index must be within this range."
    
#     # ir = range(*normalize_slice(length, slice_))
#     # di = destination_index
#     # if length_A is None:
#     #     if length_B is None:
#     #         result = criteria(source1,source2)
#     #         result = np.ndarray([result for _ in ir], dtype='bool')
#     #     else:  # length_B is not None
#     #         sliceA = np.array([source1] * length_B)
#     # else:
#     #     if length_B is None:
#     #         B = np.array([source2] * length_A)
#     #     else:  # A & B is not None
#     #         pass
    
#     if A_is_data and B_is_data:
#         result = eval(f"sliceA {criteria} sliceB")
#     if A_is_data or B_is_data:
#         if A_is_data:
#             sliceB = np.array([source2] * len(sliceA))
#         else:
#             sliceA = np.array([source1] * len(sliceB))
#     else:
#         v = criteria(source1,source2)
#         length = slice_.stop - slice_.start 
#         ir = range(*normalize_slice(length, slice_))
#         result = np.ndarray([v for _ in ir], dtype='bool')

#     if criteria == "in":
#         result = np.ndarray([criteria(a,b) for a, b in zip(sliceA, sliceB)], dtype='bool')
#     else:
#         result = eval(f"sliceA {criteria} sliceB")  # eval is evil .. blah blah blah... Eval delegates to optimized numpy functions.        

#     data[destination_index][slice_] = result


# def merge(source, mask, filter_type, slice_):
#     """ PARALLEL TASK FUNCTION
#     creates new tables from combining source and mask.
#     """
#     if not isinstance(source, dict):
#         raise TypeError
#     for L in source.values():
#         if not isinstance(L, list):
#             raise TypeError
#         if not all(isinstance(sma, SharedMemoryAddress) for sma in L):
#             raise TypeError

#     if not isinstance(mask, SharedMemoryAddress):
#         raise TypeError
#     if not isinstance(filter_type, str) and filter_type in {'any', 'all'}:
#         raise TypeError
#     if not isinstance(slice_, slice):
#         raise TypeError
    
#     # 1. determine length of Falses and Trues
#     f = any if filter_type == 'any' else all
#     handle, mask = mask.to_shm() 
#     if len(mask) == 1:
#         true_mask = mask[0][slice_]
#     else:
#         true_mask = [f(c[i] for c in mask) for i in range(slice_.start, slice_.stop)]
#     false_mask = np.invert(true_mask)

#     t1 = Table.from_address(source)  # 2. load Table.from_shm(source)
#     # 3. populate the tables
    
#     true, false = Table(), Table()
#     for name, mc in t1.columns.items():
#         mc_unfiltered = np.array(mc[slice_])
#         if any(true_mask):
#             data = mc_unfiltered[true_mask]
#             true.add_column(name, data)  # data = mc_unfiltered[new_mask]
#         if any(false_mask):
#             data = mc_unfiltered[false_mask]
#             false.add_column(name, data)

#     # 4. return table.to_shm()
#     return true.address, false.address   


# def text_reader(source, destination, columns, 
#                 newline, delimiter=',', first_row_has_headers=True, qoute='"',
#                 text_escape_openings='', text_escape_closures='',
#                 start=None, limit=None,
#                 encoding='utf-8'):
#     """ PARALLEL TASK FUNCTION
#     reads columnsname + path[start:limit] into hdf5.

#     source: csv or txt file
#     destination: available filename
    
#     columns: column names or indices to import

#     newline: '\r\n' or '\n'
#     delimiter: ',' ';' or '|'
#     first_row_has_headers: boolean
#     text_escape_openings: str: default: "({[ 
#     text_escape_closures: str: default: ]})" 

#     start: integer: The first newline after the start will be start of blob.
#     limit: integer: appx size of blob. The first newline after start of 
#                     blob + limit will be the real end.

#     encoding: chardet encoding ('utf-8, 'ascii', ..., 'ISO-22022-CN')
#     root: hdf5 root, cannot be the same as a column name.
#     """
#     if isinstance(source, str):
#         source = pathlib.Path(source)
#     if not isinstance(source, pathlib.Path):
#         raise TypeError
#     if not source.exists():
#         raise FileNotFoundError(f"File not found: {source}")

#     if isinstance(destination, str):
#         destination = pathlib.Path(destination)
#     if not isinstance(destination, pathlib.Path):
#         raise TypeError

#     if not isinstance(columns, dict):
#         raise TypeError
#     if not all(isinstance(name,str) for name in columns):
#         raise ValueError

#     root=HDF5_IMPORT_ROOT
    
#     # declare CSV dialect.
#     text_escape = TextEscape(text_escape_openings, text_escape_closures, qoute=qoute, delimiter=delimiter)

#     if first_row_has_headers:
#         with source.open('r', encoding=encoding) as fi:
#             for line in fi:
#                 line = line.rstrip('\n')
#                 break  # break on first
#         headers = text_escape(line)  
#         indices = {name: headers.index(name) for name in columns}
#     else:
#         indices = {name: int(name) for name in columns}

#     # find chunk:
#     # Here is the problem in a nutshell:
#     # --------------------------------------------------------
#     # bs = "this is my \n text".encode('utf-16')
#     # >>> bs
#     # b'\xff\xfet\x00h\x00i\x00s\x00 \x00i\x00s\x00 \x00m\x00y\x00 \x00\n\x00 \x00t\x00e\x00x\x00t\x00'
#     # >>> nl = "\n".encode('utf-16')
#     # >>> nl in bs
#     # False
#     # >>> nl.decode('utf-16') in bs.decode('utf-16')
#     # True
#     # --------------------------------------------------------
#     # This means we can't read the encoded stream to check if in contains a particular character.

#     # Fetch the decoded text:
#     with source.open('r', encoding=encoding) as fi:
#         fi.seek(0, 2)
#         filesize = fi.tell()
#         fi.seek(start)
#         text = fi.read(limit)
#         begin = text.index(newline)
#         text = text[begin+len(newline):]

#         snipsize = min(1000,limit)
#         while fi.tell() < filesize:
#             remainder = fi.read(snipsize)  # read with decoding
            
#             if newline not in remainder:  # decoded newline is in remainder
#                 text += remainder
#                 continue
#             ix = remainder.index(newline)
#             text += remainder[:ix]
#             break

#     # read rows with CSV reader.
#     data = {h: [] for h in indices}
#     for row in text.split(newline):
#         fields = text_escape(row)
#         if fields == [""] or fields == []:
#             break
#         for header,index in indices.items():
#             data[header].append(fields[index])

#     # turn rows into columns.    
#     for name, dtype in columns.items():
#         arr = np.array(data[name], dtype=dtype)
#         if arr.dtype == 'O':  # hdf5 doesn't like 'O' type
#             data[name] = np.array(arr[:], dtype='S')  
#         else:
#             data[name] = arr

#     # store as HDF5
#     for _ in range(100):  # overcome any IO blockings.
#         try:
#             with h5py.File(destination, 'a') as f:
#                 for name, arr in data.items():
#                     f.create_dataset(f"/{root}/{name}/{start}", data=arr)  # `start` declares the slice id which order will be used for sorting
#             return
#         except OSError as e:
#             time.sleep(random.randint(10,200)/1000)
#     raise TimeoutError("Couldn't connect to OS.")


# def consolidate(path):
#     """ PARALLEL TASK FUNCTION
#     enables consolidation of hdf5 imports from root into column named folders.
    
#     path: pathlib.Path
#     """
#     if not isinstance(path, pathlib.Path):
#         raise TypeError
#     if not path.exists():
#         raise FileNotFoundError(path)
    
#     root=HDF5_IMPORT_ROOT

#     with h5py.File(path, 'a') as f:
#         if root not in f.keys():
#             raise ValueError(f"hdf5 root={root} not in {f.keys()}")

#         lengths = defaultdict(int)
#         dtypes = defaultdict(set)  # necessary to track as data is dirty.
#         for col_name in f[f"/{root}"].keys():
#             for start in sorted(f[f"/{root}/{col_name}"].keys()):
#                 dset = f[f"/{root}/{col_name}/{start}"]
#                 lengths[col_name] += len(dset)
#                 dtypes[col_name].add(dset.dtype)
        
#         if len(set(lengths.values())) != 1:
#             d = {k:v for k,v in lengths.items()}
#             raise ValueError(f"assymmetric dataset: {d}")
#         for k,v in dtypes.items():
#             if len(v) != 1:
#                 L = list(dtypes[k])
#                 L.sort(key=lambda x: x.itemsize, reverse=True)
#                 dtypes[k] = L[0]  # np.bytes
#             else:
#                 dtypes[k] = v.pop()
        
#         for col_name in f[root].keys():
#             shape = (lengths[col_name], )
#             layout = h5py.VirtualLayout(shape=shape, dtype=dtypes[col_name])
#             a, b = 0, 0
#             for start in sorted(f[f"{root}/{col_name}"].keys()):
#                 dset = f[f"{root}/{col_name}/{start}"]
#                 b += len(dset)
#                 vsource = h5py.VirtualSource(dset)
#                 layout[a:b] = vsource
#                 a = b
#             f.create_virtual_dataset(f'/{col_name}', layout=layout)   


# def sha256sum(path, column_name):
#     """ PARALLEL TASK FUNCTION
#     calculates the sha256sum for a HDF5 column when given a path.
#     """
#     with h5py.File(path,'r') as f:  # 'r+' in case the sha256sum is missing.
#         m = hashlib.sha256()  # let's check if it really is new data...
#         dset = f[f"/{column_name}"]
#         step = 100_000
#         desc = f"Calculating missing sha256sum for {column_name}: "
#         for i in trange(0, len(dset), step, desc=desc):
#             chunk = dset[i:i+step]
#             m.update(chunk.tobytes())
#         sha256sum = m.hexdigest()
#         # f[f"/{column_name}"].attrs['sha256sum'] = sha256sum

#     for _ in range(100):  # overcome any IO blockings.
#         try:
#             with h5py.File(path, 'a') as f:
#                 f[f"/{column_name}"].attrs['sha256sum'] = sha256sum
#             return
#         except OSError as e:
#             time.sleep(random.randint(2,100)/1000)
#     raise TimeoutError("Couldn't connect to OS.")


# def excel_reader(path, has_headers=True, sheet_name=None, **kwargs):
#     """
#     returns Table(s) from excel path
#     """
#     if not isinstance(path, pathlib.Path):
#         raise ValueError(f"expected pathlib.Path, got {type(path)}")
#     book = pyexcel.get_book(file_name=str(path))

#     if sheet_name is None:  # help the user.
#         raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in book]}")
#     elif sheet_name not in {s.name for s in book}:
#         raise ValueError(f"sheet not found: {sheet_name}")

#     # import all sheets or a subset
#     for sheet in book:
#         if sheet.name != sheet_name:
#             continue
#         else:
#             break
    
#     t = Table()
#     for idx, column in enumerate(sheet.columns(), 1):
#         if has_headers:
#             header, start_row_pos = str(column[0]), 1
#         else:
#             header, start_row_pos = f"_{idx}", 0

#         dtypes = {type(v) for v in column[start_row_pos:]}
#         dtypes.discard(None)

#         if dtypes == {int, float}:
#             dtypes.remove(int)

#         if len(dtypes) == 1:
#             dtype = dtypes.pop()
#             data = [dtype(v) if not isinstance(v, dtype) else v for v in column[start_row_pos:]]
#         else:
#             dtype, data = str, [str(v) for v in column[start_row_pos:]]
#         t.add_column(header, data)
#     return t


# def ods_reader(path, has_headers=True, sheet_name=None, **kwargs):
#     """
#     returns Table from .ODS
#     """
#     if not isinstance(path, pathlib.Path):
#         raise ValueError(f"expected pathlib.Path, got {type(path)}")
#     sheets = pyexcel.get_book_dict(file_name=str(path))

#     if sheet_name is None or sheet_name not in sheets:
#         raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in sheets]}")
            
#     data = sheets[sheet_name]
#     for _ in range(len(data)):  # remove empty lines at the end of the data.
#         if "" == "".join(str(i) for i in data[-1]):
#             data = data[:-1]
#         else:
#             break

#     t = Table()
#     for ix, value in enumerate(data[0]):
#         if has_headers:
#             header, start_row_pos = str(value), 1
#         else:
#             header, start_row_pos = f"_{ix + 1}", 0

#         dtypes = set(type(row[ix]) for row in data[start_row_pos:] if len(row) > ix)
#         dtypes.discard(None)
#         if len(dtypes) == 1:
#             dtype = dtypes.pop()
#         elif dtypes == {float, int}:
#             dtype = float
#         else:
#             dtype = str
#         values = [dtype(row[ix]) for row in data[start_row_pos:] if len(row) > ix]
#         t.add_column(header, data=values)
#     return t


# TESTS -----------------
# def test_range_intercept():
#     A = range(500,700,3)
#     B = range(520,700,3)
#     C = range(10,1000,30)

#     assert intercept(A,C) == range(0)
#     assert set(intercept(B,C)) == set(B).intersection(set(C))

#     A = range(500_000, 700_000, 1)
#     B = range(10, 10_000_000, 1000)

#     assert set(intercept(A,B)) == set(A).intersection(set(B))

#     A = range(500_000, 700_000, 1)
#     B = range(10, 10_000_000, 1)

#     assert set(intercept(A,B)) == set(A).intersection(set(B))


# def test_text_escape():
#     # set up
#     text_escape = TextEscape(openings='({[', closures=']})', qoute='"', delimiter=',')
#     s = "this,is,a,,嗨,(comma,sep'd),\"text\""
#     # use
#     L = text_escape(s)
#     assert L == ["this", "is", "a", "","嗨", "(comma,sep'd)", "\"text\""]
    

# def test_basics():
#     # creating a tablite incrementally is straight forward:
#     table1 = Table()
#     assert len(table1) == 0
#     table1.add_column('A', data=[1,2,3])
#     assert 'A' in table1.columns
#     assert len(table1) == 3

#     table1.add_column('B', data=['a','b','c'])
#     assert 'B' in table1.columns
#     assert len(table1) == 3

#     table2 = table1.copy()

#     table3 = table1 + table2
#     assert len(table3) == len(table1) + len(table2)
#     for row in table3.rows:
#         print(row)

#     tables = 3
#     managed_columns_per_table = 2 
#     datablocks = 2

#     assert len(MemoryManager.map.nodes()) == tables + (tables * managed_columns_per_table) + datablocks
#     assert len(MemoryManager.map.edges()) == tables * managed_columns_per_table + 8 - 2  # the -2 is because of double reference to 1 and 2 in Table3
#     assert len(table1) + len(table2) + len(table3) == 3 + 3 + 6

#     # delete table
#     assert len(MemoryManager.map.nodes()) == 11, "3 tables, 6 managed columns and 2 datablocks"
#     assert len(MemoryManager.map.edges()) == 12
#     del table1  # removes 2 refs to ManagedColumns and 2 refs to DataBlocks
#     assert len(MemoryManager.map.nodes()) == 8, "removed 1 table and 2 managed columns"
#     assert len(MemoryManager.map.edges()) == 8 
#     # delete column
#     del table2['A']
#     assert len(MemoryManager.map.nodes()) == 7, "removed 1 managed column reference"
#     assert len(MemoryManager.map.edges()) == 6

#     print(MemoryManager.inventory())

#     del table3
#     del table2
#     assert len(MemoryManager.map.nodes()) == 0
#     assert len(MemoryManager.map.edges()) == 0

# def test_basics2():
#     """ testing immutability of dependencies """
#     table1 = Table()
#     table1.add_column('A', data=[1,2,3])
#     table1.add_column('B', data=['a','b','c'])
#     table2 = table1.copy()
    
#     # if a table is modified the old DataBlocks are replaced
#     table1['A', 'B'] = [ [4,5,6], ['q','w','e'] ]
#     assert table1['A'] == [4,5,6]  # a has now been derefenced from table1 datablocks.   
#     assert table2['A'] == [1,2,3]

#     col_a = table2['A']
#     assert isinstance(col_a, ManagedColumn)
#     assert col_a == [1,2,3]
    
#     table1 += table1
#     assert table1['A'] == [4, 5, 6, 4, 5, 6]
#     table1['A'][0] = 44
#     assert table1['A'] == [44, 5, 6, 4, 5, 6]
    

# def test_datatypes():
#     from datetime import datetime
#     now = datetime.now().replace(microsecond=0)
#     table4 = Table()
#     table4.add_column('A', data=[-1, 1])
#     table4.add_column('B', data=[None, 1])     # (1)
#     table4.add_column('C', data=[-1.1, 1.1])
#     table4.add_column('D', data=["", "1"])     # (2)
#     table4.add_column('E', data=[None, "1"])   # (1,2)
#     table4.add_column('F', data=[False, True])
#     table4.add_column('G', data=[now, now])
#     table4.add_column('H', data=[now.date(), now.date()])
#     table4.add_column('I', data=[now.time(), now.time()])
#     # (1) with `allow_empty=True` `None` is permitted.
#     # (2) Empty string is not a None, when datatype is string.
#     table4.show()

#     for name in 'ABCDEFGHI':
#         dt = []
#         for v in table4[name]:
#             dt.append(type(v))
#         print(name, dt)
#     # + test for use_disk=True
#     table4

# def test_add_data():
#     from tablite import Table

#     t = Table()
#     t.add_column('row', int)
#     t.add_column('A', int)
#     t.add_column('B', int)
#     t.add_column('C', int)
#     t.add_row(1, 1, 2, 3)  # individual values
#     t.add_row([2, 1, 2, 3])  # list of values
#     t.add_row((3, 1, 2, 3))  # tuple of values
#     t.add_row(*(4, 1, 2, 3))  # unpacked tuple
#     t.add_row(row=5, A=1, B=2, C=3)   # keyword - args
#     t.add_row(**{'row': 6, 'A': 1, 'B': 2, 'C': 3})  # dict / json.
#     t.add_row((7, 1, 2, 3), (8, 4, 5, 6))  # two (or more) tuples.
#     t.add_row([9, 1, 2, 3], [10, 4, 5, 6])  # two or more lists
#     t.add_row({'row': 11, 'A': 1, 'B': 2, 'C': 3},
#               {'row': 12, 'A': 4, 'B': 5, 'C': 6})  # two (or more) dicts as args.
#     t.add_row(*[{'row': 13, 'A': 1, 'B': 2, 'C': 3},
#                 {'row': 14, 'A': 1, 'B': 2, 'C': 3}])  # list of dicts.

# def test_plotly():
#     from tablite import Table

#     t = Table()
#     t.add_column('a', data=[1, 2, 8, 3, 4, 6, 5, 7, 9])
#     t.add_column('b', data=[10, 100, 3, 4, 16, -1, 10, 10, 10])

#     t.show(slice(5))  # first 5 rows only.
#     try:
#         import plotly.graph_objects as go
#     except ImportError:
#         return
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(y=t['a']))  # <-- get column 'a' from Table t
#     fig.add_trace(go.Bar(y=t['b']))  #     <-- get column 'b' from Table t
#     fig.update_layout(title = 'Hello Figure')
#     fig.show()


# def test_slicing():
#     table1 = Table()
#     base_data = list(range(10_000))
#     table1.add_column('A', data=base_data)
#     table1.add_column('B', data=[v*10 for v in base_data])
#     table1.add_column('C', data=[-v for v in base_data])
#     start = time.time()
#     big_table = table1 * 10_000  # = 100_000_000
#     print(f"it took {time.time()-start} to extend a table to {len(big_table)} rows")
#     start = time.time()
#     _ = big_table.copy()
#     print(f"it took {time.time()-start} to copy {len(big_table)} rows")
    
#     a_preview = big_table['A', 'B', 1_000:900_000:700]
#     for row in a_preview[3:15:3].rows:
#         print(row)
#     a_preview.show(format='ascii')
    

# # PARALLEL TASK FUNCTION
# def mem_test_job(shm_name, dtype, shape, index, value):
#     """
#     function for TaskManager for test_multiprocessing
#     """
#     existing_shm = shared_memory.SharedMemory(name=shm_name)
#     c = np.ndarray((6,), dtype=dtype, buffer=existing_shm.buf)
#     c[index] = value
#     existing_shm.close()
#     time.sleep(0.1)

# def test_multiprocessing():
#     # Create shared_memory array for workers to access.
#     a = np.array([1, 1, 2, 3, 5, 8])
#     shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
#     b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
#     b[:] = a[:]

#     task = Task(f=mem_test_job, shm_name=shm.name, dtype=a.dtype, shape=a.shape, index=-1, value=888)

#     tasks = [task]
#     for i in range(4):
#         task = Task(f=mem_test_job, shm_name=shm.name, dtype=a.dtype, shape=a.shape, index=i, value=111+i)
#         tasks.append(task)
        
#     with TaskManager() as tm:
#         # Alternative "low level usage" instead of using `with` is:
#         # tm = TaskManager()
#         # tm.start()
#         # results = tm.execute(task)  # returns Tasks with attribute T.result populated.
#         # tm.stop()
#         results = tm.execute(tasks)

#         for v in results:
#             print(str(v))

#     print(b, f"assertion that b[-1] == 888 is {b[-1] == 888}")  
#     print(b, f"assertion that b[0] == 111 is {b[0] == 111}")  
    
#     shm.close()
#     shm.unlink()


# def test_h5_inspection():
#     filename = 'a.csv.h5'

#     with h5py.File(filename, 'w') as f:
#         print(f.name)

#         print(list(f.keys()))

#         config = {
#             'import_as': 'csv',
#             'newline': b'\r\n',
#             'text_qualifier':b'"',
#             'delimiter':b",",
#             'first_row_headers':True,
#             'columns': {"col1": 'i8', "col2": 'int64'}
#         }
        
#         f.attrs['config']=str(config)
#         dset = f.create_dataset("col1", dtype='i8', data=[1,2,3,4,5,6])
#         dset = f.create_dataset("col2", dtype='int64', data=[5,5,5,5,5,2**33])

#     # Append to dataset
#     # must have chunks=True and maxshape=(None,)
#     with h5py.File(filename, 'a') as f:
#         dset = f.create_dataset('/sha256sum', data=[2,5,6],chunks=True, maxshape=(None, ))
#         print(dset[:])
#         new_data = [3,8,4]
#         new_length = len(dset) + len(new_data)
#         dset.resize((new_length, ))
#         dset[-len(new_data):] = new_data
#         print(dset[:])

#         print(list(f.keys()))

#     Table.inspect_h5_file(filename)
#     pathlib.Path(filename).unlink()  # cleanup.


# def test_file_importer():
#     BIG_PATH = r"d:\remove_duplicates.csv"
#     assert pathlib.Path(BIG_PATH).exists(), "?"
#     BIG_HDF5 = pathlib.Path(BIG_PATH + '.hdf5')
#     if BIG_HDF5.exists():
#         BIG_HDF5.unlink()

#     columns = {  # numpy type codes: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
#         'SKU ID': 'i', # integer
#         'SKU description':'S', # np variable length str
#         'Shipped date' : 'S', #datetime
#         'Shipped time' : 'S', # integer to become time
#         'vendor case weight' : 'f'  # float
#     }  
#     config = {
#         'delimiter': ',', 
#         "qoute": '"',
#         "newline": "\n",
#         "columns": columns, 
#         "first_row_has_headers": True,
#         "encoding": "ascii"
#     }  

#     # single processing.
#     start, limit = 0, 10_000
#     for _ in range(4):
#         text_reader(source=BIG_PATH, destination=BIG_HDF5, start=start, limit=limit, **config)
#         start = start + limit
#         limit += 10_000

#     consolidate(BIG_HDF5)
    
#     Table.inspect_h5_file(BIG_HDF5)
#     BIG_HDF5.unlink()  # cleanup!


# def test_file_importer_multiproc():

#     BIG_HDF5
#     if BIG_HDF5.exists():
#         BIG_HDF5.unlink()

#     columns = {  # numpy type codes: https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
#         'SKU ID': 'i', # integer
#         'SKU description':'S', # np variable length str
#         'Shipped date' : 'S', #datetime
#         'Shipped time' : 'S', # integer to become time
#         'vendor case weight' : 'f'  # float
#     }  

#     # now use multiprocessing
#     start = time.time()
#     t1 = Table.import_file(BIG_PATH, import_as='csv', columns=columns, delimiter=',', text_qualifier=None, newline='\n', first_row_has_headers=True)
#     end = time.time()
#     print(f"import took {round(end-start, 4)} secs.")

#     start = time.time()
#     t2 = Table.load_file(BIG_HDF5)
#     end = time.time()
#     print(f"reloading an imported table took {round(end-start, 4)} secs.")
#     t1.show()
#     print("-"*120)
#     t2.show()

#     # re-import bypass check
#     start = time.time()
#     t3 = Table.import_file(BIG_PATH, import_as='csv', columns=columns, delimiter=',', text_qualifier=None, newline='\n', first_row_has_headers=True)
#     end = time.time()
#     print(f"reloading an already imported table took {round(end-start, 4)} secs.")

#     t3.show(slice(3,100,17))


# def cpu_intense_task_with_shared_memory(n):
#     """ Task for multiprocess bug 82300 """
#     # create shared memory object
#     arr = np.array(list(range(n)))
#     shm = SharedMemory(create=True, size=arr.nbytes)
#     datablock = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
#     datablock[:] = arr[:]  # copy the data.
#     # disconnect from the task.
#     return shm.name, datablock.shape


# def test_multiprocess_bug_82300():
#     n = 7

#     tasks =[ Task(f=cpu_intense_task_with_shared_memory, n=10**i) for i in range(n) ]
    
#     with TaskManager(cores=2) as tm:  # start sub procs by using the context manager.
#         results = tm.execute(tasks)
#         results.sort(key=lambda x: x.task_id)

#         # collect evidence that it worked.
#         assert len(results) == len(tasks)

#         result_names, arrays = set(), []
#         total = 0 
#         for r in results:
#             result_name, shape = r.result
#             result_names.add(result_name)
#             shm = tm.shared_memory_references[result_name]
#             data = np.ndarray(shape, dtype=int, buffer=shm.buf)
#             total += data.shape[0]  # get the data from the workers.
            
#             arrays.append(data)

#         tm_names = set(tm.shared_memory_references.keys())
#         assert result_names == tm_names, (result_names, tm_names)
#         assert total == sum(10**i for i in range(n)), total
#     # stop all subprocs by exiting the context mgr.

#     # check the data is still around.
#     assert sum(len(arr) for arr in arrays) == total


# def test_filter():
#     t3 = Table.load_file(BIG_HDF5)
#     t3_true, t3_false = t3.filter(columns=[('vendor case weight', ">", 2.0)])
#     assert t3.columns == t3_true.columns == t3_false.columns
#     assert len(t3_true) != 0
#     assert len(t3_false) != 0
#     assert len(t3_true) + len(t3_false) == len(t3)



# DATATYPES.
# drop data to disk.    
# Sort
# Groupby & Pivot table.
# excel reader!
# Join - create join as tasklist.
# replace missing values.


# memory limit
# set task manager memory limit relative to using psutil
# update LRU cache based on access.

# GLOBAL_CLEANUP = False
# BIG_PATH = r"d:\remove_duplicates2.csv"
# BIG_FILE = pathlib.Path(BIG_PATH)

# BIG_HDF5 = pathlib.Path(str(BIG_FILE) + '.hdf5')

# if __name__ == "__main__":
#     # test_datatypes()
#     test_filter()
#     # test_multiprocess_bug_82300()
    
#     test_file_importer_multiproc()  # now the imported file is available for other tests.

#     # for k,v in {k:v for k,v in sorted(globals().items()) if k.startswith('test') and callable(v)}.items():
#     #     if k == "test_file_importer_multiproc":
#     #         continue
#     #     print(20 * "-" + k + "-" * 20)
#     #     v()

#     if GLOBAL_CLEANUP:
#         try:
#             BIG_HDF5.unlink()  # cleanup!
#         except PermissionError:
#             pass


