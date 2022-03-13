import io
import traceback
import queue
import time
import tqdm
import multiprocessing
from multiprocessing import shared_memory
import numpy as np


class TaskManager(object):
    def __init__(self) -> None:
        self.tq = multiprocessing.Queue()  # task queue for workers.
        self.rq = multiprocessing.Queue()  # result queue for workers.
        self.pool = [Worker(name=str(i), tq=self.tq, rq=self.rq) for i in range(2)]
        for p in self.pool:
            p.start()
        while not all(p.is_alive() for p in self.pool):
            time.sleep(0.01)
        
        self.tasks = {}  # task register for progress tracking
        self.results = {}  # result register for progress tracking
    
    def add(self, task):
        if not isinstance(task, dict):
            raise TypeError
        if not 'id' in task:
            raise KeyError("expect task to have id, to preserve order")
        if task['id'] in self.tasks:
            raise KeyError(f"task {task['id']} already in use.")
        self.tasks[id] = task
        self.tq.put(task)
    
    def execute(self):
        t = tqdm.tqdm(total=len(self.tasks))
        while len(self.tasks) != len(self.results):
            try:
                result = self.rq.get_nowait()
                self.results[result['id']] = result
            except queue.Empty:
                time.sleep(0.01)
            t.update(len(self.results))
        print("all tasks accounted for")
        t.close()
    
    def stop(self):
        self.tq.put("stop")
        while all(p.is_alive() for p in self.pool):
            time.sleep(0.01)
        print("all workers stopped")


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
                time.sleep(0.1)
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


if __name__ == "__main__":

    # Create shared_memory array for workers to access.
    a = np.array([1, 1, 2, 3, 5, 8])
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]

    task = {
        'id':1,
        'address': shm.name, 'type': 'shm', 
        'dtype': a.dtype, 'shape': a.shape, 
        'script': f"""# from multiprocssing import shared_memory - is already imported.
existing_shm = shared_memory.SharedMemory(name='{shm.name}')
c = np.ndarray((6,), dtype=np.{a.dtype}, buffer=existing_shm.buf)
c[-1] = 888
existing_shm.close()
"""}

    tm = TaskManager()
    tm.add(task)
    tm.execute()
    tm.stop()
    print(b, f"assertion that b[-1] == 888 is {b[-1] == 888}")  
    
    shm.close()
    shm.unlink()
    for k,v in tm.results.items():
        print(k,v)

