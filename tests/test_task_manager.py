from tablite.task_manager import TaskManager,Task
import time


def test_alpha():
    args = list(range(10)) * 5
    start = time.time()
    with TaskManager() as tm:
        tasks = [Task(f, args=(arg/10,), kwargs={'hello': arg}) for arg in args]
        results = tm.execute(tasks)
    end = time.time()
    print(f"did nothing for {end-start} seconds, producing {results} result")

def f(*args, **kwargs):
    print(args, kwargs)
    time.sleep(args[0])
    return args[0]


if __name__ == "__main__":
    test_alpha()