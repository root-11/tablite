import time
from multiprocessing import Pool


def cube(x):
    print(f"start process {x}")
    result = x * x * x
    time.sleep(1)
    print(f"end process {x}")
    return result


if __name__ == "__main__":
    pool = Pool(processes=4)
    print(pool.map(cube, range(5)))
    pool.close()
    pool.join()