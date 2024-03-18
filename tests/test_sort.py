from tablite import Table
from datetime import datetime
from numpy import datetime64
from tablite.sort_utils import unix_sort

def test_sort():
    t = Table(columns={"A": [4, 3, 2, 1], "B": [2, 2, 1, 1], "C": ["a", "d", "c", "b"]})

    L = list(t["A"])
    L.sort(reverse=False)
    t.sort({"A": False})
    assert t["A"] == L

    t.sort({"A": True})
    L.sort(reverse=True)
    assert t["A"] == L

    t.sort({"B": False, "A": True})
    assert t["B"] == [1, 1, 2, 2]
    assert t["A"] == [2, 1, 4, 3]
    t.sort({"C": False})
    assert t["C"] == ["a", "b", "c", "d"]
    assert t["A"] == [4, 1, 2, 3]
    assert t["B"] == [2, 1, 1, 2]
    t.sort({"C": True})
    assert t["C"] == ["d", "c", "b", "a"]
    assert t["A"] == [3, 2, 1, 4]


def test_sorted():
    t = Table(columns={"A": [4, 3, 2, 1], "B": [2, 2, 1, 1], "C": ["a", "d", "c", "b"]})

    t2 = t.sorted({"A": False})
    assert t2 is not t
    assert t2 != t
    t.sort({"A": False})
    assert t2 == t
    assert t2 is not t


def test_sort_multiple_datatypes():
    t = Table(columns={"A": [None, True, 2.0, 3, 4, "5"], "B": [0, 1, 2, 3, 4, 5]})
    t.sort(mapping={"A": False})
    assert t["A"] == [2.0, 3, 4, "5", True, None]


def test_sort_datetime():
    t = Table(columns={"A": [datetime(2019, 2, 2, 12, 12, 12), datetime.now()], "B": [2, 2]})
    t.sort({"A": False})

    t = Table(columns={"A": [datetime64("2005"), datetime64(datetime.now())], "B": [2, 2]})
    t.sort({"A": False})


def test_unix_sort():
    d = unix_sort([True, True, True, 0, 1, 1.0, False, 2])
    assert False in d
    assert d[False] == 0
    
    assert True in d
    assert d[True] == 3
    assert 0 in d
    assert d[0] == 4
    
    assert 1 in d
    assert d[1] == 5
    
    assert 1.0 in d
    assert d[1.0] == 6
    
    assert 2 in d
    assert d[2] == 7