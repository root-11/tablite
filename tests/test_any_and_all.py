from tablite import Table
import random
from random import randint
import time


def test_any():
    t = Table()
    t["a"] = [1, 2, 3, 4]
    t["b"] = [10, 20, 30, 40]

    def f(x):
        return x == 4

    def g(x):
        return x < 20

    t2 = t.any(**{"a": f, "b": g})
    assert [r for r in t2.rows] == [[1, 10], [4, 40]]

    t2 = t.any(a=f, b=g)
    assert [r for r in t2.rows] == [[1, 10], [4, 40]]

    def h(x):
        return x >= 2

    def i(x):
        return x <= 30

    t2 = t.all(a=h, b=i)
    assert [r for r in t2.rows] == [[2, 20], [3, 30]]


def test_filter_profile():
    random.seed(5432)
    t = Table()
    t["1"] = list(range(1000))
    t["2"] = list(randint(1, 20) for _ in range(1000))
    t["3"] = list(str(i) for i in range(1000))
    t["4"] = list(randint(1, 20) for _ in range(1000))
    t["5"] = list(str(i) for i in range(1000))
    t["6"] = list(randint(1, 20) for _ in range(1000))
    t["7"] = list(str(i) for i in range(1000))
    t["8"] = list(randint(1, 20) for _ in range(1000))
    t["9"] = list(str(i) for i in range(1000))
    t["10"] = list(range(1000))
    t["11"] = list(range(1000))
    t["12"] = list(range(1000))
    t["13"] = list(range(1000))
    t["14"] = list(range(1000))

    start = time.process_time()
    t2 = t.all(**{"2": lambda x: x > 4, "4": lambda x: x > 5, "6": lambda x: x > 6, "8": lambda x: x > 7})
    end = time.process_time()
    assert 259 == len(t2)
    assert end - start < 2, "this shouldn't take 2 seconds."


def test_drop_na():
    t = Table()
    t["a"] = [1, 2, 3, None]
    t["b"] = [1, 2, None, None]
    t["c"] = [1, 2, 3, 4]
    t["d"] = [10, 20, 30, 40]
    t2 = t.drop(None)
    assert len(t2) == 2
    t3 = t.drop(30, 40)
    assert len(t3) == 2
    assert t2 == t3

    try:
        t.drop()
        assert False, "this should raise as it is unknown what to drop."
    except ValueError:
        assert True

    t4 = t.drop(None, None, None)
    assert t4 == t2
