import multiprocessing
from multiprocessing import shared_memory, cpu_count
from tqdm import tqdm
import time
import queue
from abc import ABC
import copy
from itertools import count
import io
import numpy as np
import traceback
from collections import defaultdict


class TaskManager(object):
    shared_memory_references = {}  
    shared_memory_reference_counter = defaultdict(int)  # tracker for the NAT protocol.

    def __init__(self) -> None:    
        self._cpus = cpu_count()
        self.tq = multiprocessing.Queue()  # task queue for workers.
        self.rq = multiprocessing.Queue()  # result queue for workers.
        self.pool = []                     # list of sub processes
        self.pool_sigq = {}                # signal queue for each worker.
        self.tasks = 0                     # counter for task tracking
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): # signature requires these, though I don't use them.
        self.stop()  # stop the workers.
        
        # Clean up on exit.
        for k,v in self.shared_memory_reference_counter.items():
            if k in self.shared_memory_references and v == 0:
                del self.shared_memory_references[k]  # this unlinks the shared memory object,
                # which now can be GC'ed if no other variable points to it.
        
    def start(self):
        for i in range(self._cpus):  # create workers
            name = str(i)
            sigq = multiprocessing.Queue()  # we create one signal queue for each proc.
            self.pool_sigq[name] = sigq
            worker = Worker(name=name, tq=self.tq, rq=self.rq, sigq=sigq)
            self.pool.append(worker)

        with tqdm(total=self._cpus, unit="n", desc="workers ready") as pbar:
            for p in self.pool:
                p.start()

            while True:
                alive = sum(1 if p.is_alive() else 0 for p in self.pool)
                pbar.n = alive
                pbar.refresh()
                if alive < self._cpus:
                    time.sleep(0.01)
                else:
                    break  # all sub processes are alive. exit the setup loop.

    def execute(self, tasks):
        if isinstance(tasks, Task):
            task = (tasks,)
        if not isinstance(tasks, (list,tuple)) or not all([isinstance(i, Task) for i in tasks]):
            raise TypeError

        for t in tasks:
            self.tq.put(t)
            self.tasks += 1  # increment task counter.
        
        results = []  
        with tqdm(total=self.tasks, unit='task') as pbar:
            while self.tasks != 0:
                try:
                    task = self.rq.get_nowait()
                
                    if isinstance(task, NATsignal): 
                        if task.shm_name not in self.shared_memory_references:  # its a NOTIFY from a WORKER.
                            # first create a hard ref to the memory object.
                            self.shared_memory_references[task.shm_name] = TrackedSharedMemory(name=task.shm_name, create=False)
                            self.shared_memory_reference_counter[task.shm_name] += 1
                            # then send the ACKNOWLEDGEMENT directly to the WORKER.
                            self.pool_sigq[task.worker_name].put(task)
                        else:  # It's the second time we see the name so it's a TRANSFER COMPLETE
                            self.shared_memory_reference_counter[task.shm_name] -= 1 
                        # at this point we can be certain that the SHMs are in the main process.
                        continue  # keep looping as there may be more.

                    elif isinstance(task, Task):
                        if task.exception:
                            raise Exception(task.exception)

                        self.tasks -= 1  # decrement task counter.
                        pbar.set_description(task.f.__name__)
                        results.append(task)
                        pbar.update(1)
                    
                except queue.Empty:
                    time.sleep(0.01)
        return results 

    def stop(self):
        for _ in range(self._cpus):  # put enough stop messages for all workers.
            self.tq.put("stop")

        with tqdm(total=len(self.pool), unit="n", desc="workers stopping") as pbar:
            while True:
                not_alive = sum(1 if not p.is_alive() else 0 for p in self.pool)
                pbar.n = not_alive
                pbar.refresh()
                if not_alive < self._cpus:
                    time.sleep(0.01)
                else:
                    break
        self.pool.clear()

        # clear the message queues.
        while not self.tq.empty:  
            _ = self.tq.get_nowait()  
        while not self.rq.empty:
            _ = self.rq.get_nowait()
        
  
class Worker(multiprocessing.Process):
    def __init__(self, name, tq, rq, sigq):
        super().__init__(group=None, target=self.update, name=name, daemon=False)
        self.exit = multiprocessing.Event()
        self.tq = tq  # workers task queue
        self.rq = rq  # workers result queue
        self.sigq = sigq  # worker signal reciept queue.
        
               
    def update(self):
        # this is global for the sub process only.
        TaskManager.shared_memory_references  

        while True:
            # first process any/all direct signals first.
            while True:
                try:
                    ack = self.sigq.get_nowait()   # receive acknowledgement of hard ref to SharedMemoryObject from SIGQ            
                    shm = TaskManager.shared_memory_references.pop(ack.shm_name)  # pop the shm
                    shm.close()  # assure closure of the shm.
                    del TaskManager.shared_memory_reference_counter[ack.shm_name]
                    self.rq.put(ack)  # respond to MAINs RQ that transfer is complete.
                except queue.Empty:
                    break

            # then deal with any tasks...
            try:  
                task = self.tq.get_nowait()
                if task == "stop":
                    self.tq.put_nowait(task)  # this assures that everyone gets the stop signal.
                    self.exit.set()
                    break
                elif isinstance(task, Task):
                    task.execute()
                    
                    for k,v in TaskManager.shared_memory_references.items():
                        if k not in TaskManager.shared_memory_reference_counter:
                            TaskManager.shared_memory_reference_counter[k] = 1
                            self.rq.put(NATsignal(k, self.name))  # send Notify from subprocess to main
                        
                    self.rq.put(task)

                else:
                    raise Exception(f"What is {task}?")
            except queue.Empty:
                time.sleep(0.01)
                continue


class NATsignal(object):
    def __init__(self, shm_name, worker_name):
        """
        shm_name: str: name from shared_memory.
        worker_name: str: required by TaskManager for sending ACK message to worker.
        """
        self.shm_name = shm_name
        self.worker_name = worker_name


class TrackedSharedMemory(shared_memory.SharedMemory):
    def __init__(self, name=None, create=False, size=0) -> None:
        if name in TaskManager.shared_memory_references:
            return TaskManager.shared_memory_references[name]  # return from registry.
        else:
            super().__init__(name, create, size)
            TaskManager.shared_memory_references[self.name] = self  # add to registry. This blocks __del__ !  


class Task(ABC):
    """
    Generic Task class for tasks.
    """
    ids = count(start=1)
    def __init__(self, f, *args, **kwargs) -> None:
        """
        f: callable 
        *args: arguments for f
        **kwargs: keyword arguments for f.
        """
        if not callable(f):
            raise TypeError
        self.task_id = next(self.ids)
        self.f = f
        self.args = copy.deepcopy(args)  # deep copy is slow unless the data is shallow.
        self.kwargs = copy.deepcopy(kwargs)
        self.result = None
        self.exception = None

    def __str__(self) -> str:
        if self.exception:
            return f"Call to {self.f.__name__}(*{self.args}, **{self.kwargs}) --> Error: {self.exception}"
        else:
            return f"Call to {self.f.__name__}(*{self.args}, **{self.kwargs}) --> Result: {self.result}"

    def execute(self):
        """ The worker calls this function. """
        try:
            self.result = self.f(*self.args, **self.kwargs)
        except Exception as e:
            f = io.StringIO()
            traceback.print_exc(limit=3, file=f)
            f.seek(0)
            error = f.read()
            f.close()
            self.exception = error


def cpu_intense_task_with_shared_memory(n):
    # create shared memory object
    arr = np.array(list(range(n)))
    shm = TrackedSharedMemory(create=True, size=arr.nbytes)
    datablock = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    datablock[:] = arr[:]  # copy the data.
    # disconnect from the task.
    return shm.name, datablock.shape


if __name__ == "__main__":
    """ test... """
    n = 9

    tasks =[ Task(f=cpu_intense_task_with_shared_memory, n=10**i) for i in range(n) ]
    
    with TaskManager() as tm:  # start sub procs by using the context manager.
        results = tm.execute(tasks)
        results.sort(key=lambda x: x.task_id)

        # collect evidence that it worked.
        assert len(results) == len(tasks)

        result_names, arrays = set(), []
        total = 0 
        for r in results:
            result_name, shape = r.result
            result_names.add(result_name)
            shm = tm.shared_memory_references[result_name]
            data = np.ndarray(shape, dtype=int, buffer=shm.buf)
            total += data.shape[0]  # get the data from the workers.
            
            arrays.append(data)

        tm_names = set(tm.shared_memory_references.keys())
        assert result_names == tm_names, (result_names, tm_names)
        assert total == sum(10**i for i in range(n)), total
    # stop all subprocs by exiting the context mgr.

    # check the data is still around.
    assert sum(len(arr) for arr in arrays) == total

