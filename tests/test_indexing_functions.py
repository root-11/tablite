from tablite import Table
import pytest


@pytest.fixture(autouse=True)  # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield


def test_reindex():
    t = Table()
    t["#"] = ["c", "b", "a"]
    t["n"] = [3, 2, 1]
    t_sorted = t.reindex(index=[2, 1, 0])
    assert list(t_sorted["n"]) == [1, 2, 3]


def test_no_args():
    t = Table()
    t["1"] = [1, 1, 2, 2, 3, 3, 4, 4]
    t["2"] = [4, 4, 3, 3, 2, 2, 1, 1]
    t2 = t.drop_duplicates()
    assert t2["1"] == [1, 2, 3, 4]
    assert t2["2"] == [4, 3, 2, 1]


def test_args():
    t = Table()
    t["1"] = [1, 1, 2, 2, 3, 3, 4, 5]
    t["2"] = [4, 4, 3, 3, 2, 2, 1, 1]
    t2 = t.drop_duplicates("1")
    assert t2["1"] == [1, 2, 3, 4, 5]
    assert t2["2"] == [4, 3, 2, 1, 1]


def test_longer_table():
    t = Table()
    base = list(range(100))
    t["1"] = base * 10000
    t["2"] = base * 10000
    t2 = t.drop_duplicates()
    assert t2["1"] == base
