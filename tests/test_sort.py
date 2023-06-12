from tablite import Table


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
