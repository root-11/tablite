import h5py
import numpy as np
import shutil
import pathlib
import tempfile
import time
from mplite import Task, TaskManager


dataset_size = int(1e6)


def write(n):
    while True:
        try:
            data = [f"{i}\n" for i in range(dataset_size)]
            with open(n,'w') as fo:
                fo.write("".join(data))
            return
        except MemoryError:
            time.sleep(0.01)
        except OSError:
            time.sleep(0.01)
        

def h5write(n, group=1):
    while True:
        try:
            with h5py.File(n,'a') as h5:
                data = np.array([i for i in range(dataset_size)])
                h5.create_dataset(name=f"/{group}", data=data, dtype=data.dtype, maxshape=(None, ))
            return
        except MemoryError:
            time.sleep(0.01)
        except OSError:
            time.sleep(0.01)


def parallel_write_test():      
    folder = pathlib.Path(tempfile.gettempdir()) / 'tablite_os_test'
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    if not folder.exists():
        folder.mkdir()
    
    # parallel different files    
    start = time.perf_counter()
    tasks = []
    for fileno in range(0,50):
        path = folder / f"test_file{fileno}.txt"
        tasks.append(Task(write, str(path)))
    
    with TaskManager() as tm:
        results = tm.execute(tasks)
        assert all(i is None for i in results), [print(r) for r in results]
    end = time.perf_counter()
    print(f'parallel took: {end-start}')

    # serial different files.
    start2 = time.perf_counter()
    for fileno in range(51,100):
        path = folder / f"test_file{fileno}.txt"
        write(str(path))
    end2 = time.perf_counter()
    print(f"serial took {end2-start2} ")

    # parallel same hdf file.    
    start3 = time.perf_counter()
    p = folder / "test_file.h5"
    tasks = []
    for group in range(100,150):
        tasks.append( Task(h5write, str(p), group))
    with TaskManager() as tm:
        results = tm.execute(tasks)
        assert all(i is None for i in results), [print(r) for r in results]
    end3 = time.perf_counter()
    print(f"parallel same hdf file {end3-start3} ")

    # parallel multiple hdf files.    
    start4 = time.perf_counter()
    
    tasks = []
    for fileno in range(150,200):
        path = folder / f"test_file{fileno}.h5"
        tasks.append( Task(h5write, str(path)))
    with TaskManager() as tm:
        results = tm.execute(tasks)
        assert all(i is None for i in results), [print(r) for r in results]
    end4 = time.perf_counter()
    print(f"parallel multiple hdf files {end4-start4} ")

    # python os_test.py
    # 100%|██████████████████████████████████████| 50/50 [00:02<002<00:00, 22.57tasks/s]
    # parallel took: 2.6400329999996757
    # serial took 8.967031500000303                                                                                                      0:00, 22.57tasks/s]
    # 100%|██████████████████████████████████████| 50/50 [00:04<004<00:00, 10.36tasks/s]
    # parallel same hdf file 5.3461384000002                                                                                             0:00, 10.36tasks/s]
    # 100%|██████████████████████████████████████| 50/50 [00:01<001<00:00, 33.80tasks/s]                                                                                                            0:00, 33.80tasks/s]
    # parallel multiple hdf files 2.0312580999998318

    # conclusion: parallel write ....
    # 1. to multiple HDF5 files is at least as fast as parallel write.
    # 2. to same HDF% file requires delay of 2.6x

    shutil.rmtree(folder, ignore_errors=True)  # cleanup!

if __name__ == "__main__":
    parallel_write_test()

