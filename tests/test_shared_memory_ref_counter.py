import time
from multiprocessing import Process, Manager, Pool


def f(pid, d, l):    
    nap = d['sleep']
    d['sleep'] = 0.1
    print(pid, "will nap", nap)
    time.sleep(nap)
    print(pid, "woke up", nap)
    l.reverse()


def test1():
    manager = Manager()
    manager.__enter__()
    d = manager.dict()
    l = manager.list(range(10))

    d['sleep'] = 2
    p = Process(target=f, args=(1, d, l))
    p.start()
    time.sleep(0.1)
    p2 = Process(target=f, args=(2, d, l))
    p2.start()

    p.join()
    p2.join()

    print(d)
    print(l)
    manager.__exit__(None,None,None)

def f2(d,x):
    d[x] = x*x

def test2():  # <--- The better way to solve pooled tasks.
    with Pool(8) as pool:
        with Manager() as mgr:
            d = mgr.dict()
            tasks = [(d,x) for x in range(4000)]
            result = pool.starmap(f2, tasks)
    assert len(result)


if __name__ == '__main__':
    test1()
    test2()