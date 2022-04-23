from multiprocessing import shared_memory
import numpy as np
import time
from tablite2.task_manager import Task, TaskManager, REF_COUNT


# PARALLEL TASK FUNCTION
def syncman_job(i, ref_count):
    print(i)
    try:
        ref_count[i] = __name__
        time.sleep(0.1)
        print("worked!")
    except Exception:
        print("can't get global")
        return

    
# PARALLEL TASK FUNCTION
def mem_test_job(shm_name, dtype, shape, index, value):
    """
    function for TaskManager for test_multiprocessing
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    c = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    c[index] = value
    existing_shm.close()
    time.sleep(0.1)


def test_multiprocessing():
    # Create shared_memory array for workers to access.
    a = np.array([1, 1, 2, 3, 5, 8])
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    b[:] = a[:]

    task = Task(f=mem_test_job, shm_name=shm.name, dtype=a.dtype, shape=a.shape, index=-1, value=888)

    tasks = [task]
    for i in range(4):
        task = Task(f=mem_test_job, shm_name=shm.name, dtype=a.dtype, shape=a.shape, index=i, value=111+i)
        tasks.append(task)
        
    with TaskManager() as tm:
        # Alternative "low level usage" instead of using `with` is:
        # tm = TaskManager()
        # tm.start()
        # results = tm.execute(task)  # returns Tasks with attribute T.result populated.
        # tm.stop()
        results = tm.execute(tasks)

        for v in results:
            print(str(v))

    print(b, f"assertion that b[-1] == 888 is {b[-1] == 888}")  
    print(b, f"assertion that b[0] == 111 is {b[0] == 111}")  
    
    shm.close()
    shm.unlink()
    
    with TaskManager() as tm:
        tasks = [Task(f=syncman_job, i=i, ref_count=REF_COUNT) for i in range(20)]
        results = tm.execute(tasks)
    
    

if __name__ == "__main__":
    for k,v in {k:v for k,v in sorted(globals().items()) if k.startswith('test') and callable(v)}.items():
        print(20 * "-" + k + "-" * 20)
        v()

