import time
import numpy as np
import pathlib
from multiprocessing import Pool

def w(i):
    file = pathlib.Path(r"D:\Newport_Data_DEC-MAR.txt")
    start = time.process_time()
    try:
        x = np.loadtxt(fname=file, delimiter="\t", dtype=bytes, skiprows=1, usecols=i, encoding='ascii')
    except Exception as e:
        print(e)
        x = [str(e)]
    end = time.process_time()
    print('\tcol:', i, len(x), "rows", x[:10], end-start)


if __name__ == "__main__":
    file = pathlib.Path(r"D:\Newport_Data_DEC-MAR.txt")
    with file.open('r', encoding='ascii') as fi:
        header_row = fi.read(3000)
        print(header_row)
        
        header_row = header_row.split('\n')[0]
        headers = header_row.split('\t')
        print(headers)

    start = time.time()
    pool = Pool(processes=4)
    print(pool.map(w, range(len(headers))))
    pool.close()
    pool.join()
    end = time.time()
    print(f"total duration: {end-start} secs.")

