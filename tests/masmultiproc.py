import queue
import time
import multiprocessing

class Scheduler(object):
    def __init__(self) -> None:
        self.tq = multiprocessing.Queue()
        self.rq = multiprocessing.Queue()
        self.logq = multiprocessing.Queue()
        self.pool = [Worker(name=str(i), tq=self.tq, rq=self.rq, logq=self.logq) for i in range(2)]
        for p in self.pool:
            p.start()
            print(p.name, p.is_alive())
        while not all(p.is_alive() for p in self.pool):
            time.sleep(0.01)
    
    def stop(self):
        self.tq.put("stop")
        while all(p.is_alive() for p in self.pool):
            time.sleep(0.01)
            

class Worker(multiprocessing.Process):
    def __init__(self, name, tq, rq, logq):
        super().__init__(group=None, target=self.update, name=name, daemon=False)
        self.exit = multiprocessing.Event()
        self.tq = tq
        self.rq = rq
        self.logq = logq
        self._quit = False
        self.logq.put(f"{self.name}: ready")
                
    def update(self):
        while not self._quit:
            try:
                task = self.tq.get_nowait()
            except queue.Empty:
                time.sleep(0.1)
                continue
            
            if task == "stop":
                self.logq.put(f"{self.name}: stop signal received.")
                self.tq.put_nowait(task)  # this assures that everyone gets it.
                self._quit = True
                self.exit.set()
                break
            try:
                result = eval(task)
            except Exception as e:
                result = str(e)

            if result:
                self.rq.put(result)
            

if __name__ == "__main__":
    tm = Scheduler()
    counter = 10_000
    for i in range(counter):
        task = f"{i}*{i}"
        tm.tq.put(task)
    
    start = time.time()
    result = 0
    while 1:
        try:
            print(tm.logq.get_nowait())
        except queue.Empty:
            pass
        try:
            result = tm.rq.get_nowait()
            print(result)
        except queue.Empty:
            pass
        if not isinstance(result, int):
            break
        if result > 1_000_000:
            break
        if time.time() - start > 10:
            break

    print("done")
    tm.stop()
    print("finalized")


