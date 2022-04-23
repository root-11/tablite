import time
from multiprocessing import Process, Manager

def f(pid, d, l):    
    nap = d['sleep']
    d['sleep'] = 0.1
    print(pid, "will nap", nap)
    time.sleep(nap)
    print(pid, "woke up", nap)
    l.reverse()


if __name__ == '__main__':
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