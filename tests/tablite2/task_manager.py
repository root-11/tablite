import copy
import math
import io
import traceback
import multiprocessing
import queue
import time
import atexit
from itertools import count
from abc import ABC

import psutil
from tqdm import tqdm


POOL = []  # Global pool of workers.


def stop_remaining_workers():
    global POOL
    for worker in POOL:  # put enough stop messages for all workers.
        worker.tq.put("stop")

    with tqdm(total=len(POOL), unit="n", desc="workers stopping") as pbar:
        while True:
            not_alive = sum(1 if not p.is_alive() else 0 for p in POOL)
            pbar.n = not_alive
            pbar.refresh()
            if not_alive < len(POOL):
                time.sleep(0.01)
            else:
                break
    POOL.clear()

atexit.register(stop_remaining_workers)


class TaskManager(object):
    def __init__(self,cores=None) -> None:
        self._cpus = min(psutil.cpu_count(), cores) if (isinstance(cores,int) and cores > 0) else psutil.cpu_count()
        self._disk_space = psutil.disk_usage('/').free
        self._memory = psutil.virtual_memory().available

        self.tq = multiprocessing.Queue()  # task queue for workers.
        self.rq = multiprocessing.Queue()  # result queue for workers.
        self.pool_sigq = {}                # signal queue for each worker.
        self.tasks = 0                     # counter for task tracking
        global POOL
        self.pool = POOL                     # list of sub processes
        
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): # signature requires these, though I don't use them.
        self.stop()  # stop the workers.
    
    def start(self):
        if len(self.pool) == self._cpus and all(p.is_alive() for p in self.pool):
            return
        else:
            self.pool.clear()
            for i in range(self._cpus):  # create workers
                name = str(i)
                worker = Worker(name=name, tq=self.tq, rq=self.rq)
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
            response = (tasks,)
        if not isinstance(tasks, (list,tuple)) or not all([isinstance(i, Task) for i in tasks]):
            raise TypeError

        for t in tasks:
            self.tq.put(t)
            self.tasks += 1  # increment task counter.
        
        results = []  
        with tqdm(total=self.tasks, unit='task') as pbar:
            while self.tasks != 0:
                try:
                    response = self.rq.get_nowait()
                    if response.exception:
                        raise Exception(response.exception)

                    self.tasks -= 1  # decrement task counter.
                    pbar.set_description(response.f.__name__)
                    results.append(response)
                    pbar.update(1)
                    
                except queue.Empty:
                    time.sleep(0.01)
        return results

    def halt(self):
        stop_remaining_workers()
            
    def chunk_size_per_cpu(self, working_memory_required):  # 39,683,483,123 = 39 Gb.
        if working_memory_required < psutil.virtual_memory().free:
            mem_per_cpu = math.ceil(working_memory_required / self._cpus)
        else:
            memory_ceiling = int(psutil.virtual_memory().total * self.memory_usage_ceiling)
            memory_used = psutil.virtual_memory().used
            available = memory_ceiling - memory_used  # 6,321,123,321 = 6 Gb
            mem_per_cpu = int(available / self._cpus)  # 790,140,415 = 0.8Gb/cpu
        return mem_per_cpu


class Worker(multiprocessing.Process):
    def __init__(self, name, tq, rq):
        super().__init__(group=None, target=self.update, name=name, daemon=False)
        self.exit = multiprocessing.Event()
        self.tq = tq  # workers task queue
        self.rq = rq  # workers result queue       
               
    def update(self):
        while True:
            # then deal with any tasks...
            try:  
                task = self.tq.get_nowait()
                if task == "stop":
                    self.tq.put_nowait(task)  # this assures that everyone gets the stop signal.
                    self.exit.set()
                    break
                elif isinstance(task, Task):
                    task.execute()
                    self.rq.put(task)
                else:
                    raise Exception(f"What is {task}?")
            except queue.Empty:
                time.sleep(0.01)
                continue


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

