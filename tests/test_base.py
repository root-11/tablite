from tablite.base import BaseTable, Column, Page
from mplite import TaskManager, Task
from tablite.config import Config
import numpy as np
from pathlib import Path
import math
import os
import gc

import logging

log = logging.getLogger("base")
log.setLevel(logging.DEBUG)
console = logging.StreamHandler()
console.setLevel(level=logging.DEBUG)
formatter = logging.Formatter("%(levelname)s : %(message)s")
console.setFormatter(formatter)
log.addHandler(console)


def test_basics():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t = BaseTable(columns=data)
    assert t == data
    assert t.items() == data.items()

    a = t["A"]  # selects column 'a'
    assert isinstance(a, Column)
    b = t[3]  # selects row 3 as a tuple.
    assert isinstance(b, tuple)
    c = t[:10]  # selects first 10 rows from all columns
    assert isinstance(c, BaseTable)
    assert len(c) == 10
    assert c.items() == {k: v[:10] for k, v in data.items()}.items()
    d = t["A", "B", slice(3, 20, 2)]  # selects a slice from columns 'a' and 'b'
    assert isinstance(d, BaseTable)
    assert len(d) == 9
    assert d.items() == {k: v[3:20:2] for k, v in data.items() if k in ("A", "B")}.items()

    e = t["B", "A"]  # selects column 'b' and 'c' and 'a' twice for a slice.
    assert list(e.columns) == ["B", "A"], "order not maintained."

    x = t["A"]
    assert isinstance(x, Column)
    assert x == list(A)
    assert x == np.array(A)
    assert x == tuple(A)


def test_empty_table():
    t2 = BaseTable()
    assert isinstance(t2, BaseTable)
    t2["A"] = []
    assert len(t2) == 0
    c = t2["A"]
    assert isinstance(c, Column)
    assert len(c) == 0


def test_page_size():
    original_value = Config.PAGE_SIZE

    Config.PAGE_SIZE = 10
    assert Config.PAGE_SIZE == 10
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    # during __setitem__ Column.paginate will use config.
    t = BaseTable(columns=data)
    assert t == data
    assert t.items() == data.items()

    x = t["A"]
    assert isinstance(x, Column)
    assert len(x.pages) == math.ceil(len(A) / Config.PAGE_SIZE)
    Config.PAGE_SIZE = 7
    t2 = BaseTable(columns=data)
    assert t == t2
    assert not t != t2
    x2 = t2["A"]
    assert isinstance(x2, Column)
    assert len(x2.pages) == math.ceil(len(A) / Config.PAGE_SIZE)

    Config.reset()
    assert Config.PAGE_SIZE == original_value


def test_cleaup():
    A = list(range(1, 10))
    B = [i * 10 for i in A]

    data = {"A": A, "B": B}

    t = BaseTable(columns=data)
    assert isinstance(t, BaseTable)
    _folder = t._pid_dir

    del t
    import gc

    while gc.collect() > 0:
        pass

    assert _folder.exists()  # should be there until sigint.


def save_and_load():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t = BaseTable(columns=data)
    assert isinstance(t, BaseTable)
    my_folder = Path(Config.workdir) / "data"
    my_folder.mkdir(exist_ok=True)
    my_file = my_folder / "my_first_file.tpz"
    t.save(my_file)
    assert my_file.exists()
    assert os.path.getsize(my_file) > 0

    del t
    import gc

    while gc.collect() > 0:
        pass

    assert my_file.exists()
    t2 = BaseTable.load(my_file)

    t3 = BaseTable(columns=data)
    assert t2 == t3
    assert t2.path.parent == t3.path.parent, "t2 must be loaded into same PID dir as t3"

    del t2
    while gc.collect() > 0:
        pass

    assert my_file.exists(), "deleting t2 MUST not affect the file saved by the user."
    t4 = BaseTable.load(my_file)
    assert t4 == t3

    os.remove(my_file)
    os.rmdir(my_folder)


def test_copy():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}

    t1 = BaseTable(columns=data)
    assert isinstance(t1, BaseTable)

    t2 = t1.copy()
    for name, t2column in t2.columns.items():
        t1column = t1.columns[name]
        assert id(t2column) != id(t1column)
        for p2, p1 in zip(t2column.pages, t1column.pages):
            assert id(p2) == id(p1), "columns can point to the same pages."


def test_speed():
    log.setLevel(logging.INFO)
    A = list(range(1, 10_000_000))
    B = [i * 10 for i in A]
    data = {"A": A, "B": B}
    t = BaseTable(columns=data)
    import time
    import random

    loops = 100
    random.seed(42)
    start = time.time()
    for i in range(loops):
        a, b = random.randint(1, len(A)), random.randint(1, len(A))
        a, b = min(a, b), max(a, b)
        block = t["A"][a:b]  # pure numpy.
        assert len(block) == b - a
    end = time.time()
    numpy_array = end - start
    print(f"numpy array: {numpy_array}")

    random.seed(42)
    start = time.time()
    for i in range(loops):
        a, b = random.randint(1, len(A)), random.randint(1, len(A))
        a, b = min(a, b), max(a, b)
        block = t["A"][a:b].tolist()  # python.
        assert len(block) == b - a
    end = time.time()
    python_list = end - start
    print(f"python list: {python_list}")

    assert numpy_array < python_list, "something is wrong numpy should be faster."


def test_immutability_of_pages():
    A = list(range(1, 21))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = BaseTable(columns=data)
    change = t["A"][7:3:-1]
    t["A"][3:7] = change


def test_slice_functions():
    Config.PAGE_SIZE = 3
    t = BaseTable(columns={"A": np.array([1, 2, 3, 4])})
    L = t["A"]
    assert list(L[:]) == [1, 2, 3, 4]  # slice(None,None,None)
    assert list(L[:0]) == []  # slice(None,0,None)
    assert list(L[0:]) == [1, 2, 3, 4]  # slice(0,None,None)

    assert list(L[:2]) == [1, 2]  # slice(None,2,None)

    assert list(L[-1:]) == [4]  # slice(-1,None,None)
    assert list(L[-1::1]) == [4]
    assert list(L[-1:None:1]) == [4]

    # assert list(L[-1:4:1]) == [4]  # slice(-1,4,1)
    # assert list(L[-1:0:-1]) == [4, 3, 2]  # slice(-1,0,-1) -->  slice(4,0,-1) --> for i in range(4,0,-1)
    # assert list(L[-1:0:1]) == []  # slice(-1,0,1) --> slice(4,0,-1) --> for i in range(4,0,1)
    # assert list(L[-3:-1:1]) == [2, 3]

    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    t = BaseTable(columns={"A": np.array(data)})
    L = t["A"]
    assert list(L[-1:0:-1]) == data[-1:0:-1]
    assert list(L[-1:0:1]) == data[-1:0:1]

    data = list(range(100))
    t2 = BaseTable(columns={"A": np.array(data)})
    L = t2["A"]
    assert list(L[51:40:-1]) == data[51:40:-1]
    Config.reset()


def test_various():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = BaseTable(columns=data)
    t *= 2
    assert len(t) == len(A) * 2
    x = t["A"]
    assert len(x) == len(A) * 2

    t2 = t.copy()
    t2 += t
    assert len(t2) == len(t) * 2
    assert len(t2["A"]) == len(t["A"]) * 2
    assert len(t2["B"]) == len(t["B"]) * 2
    assert len(t2["C"]) == len(t["C"]) * 2

    assert t != t2

    orphaned_column = t["A"].copy()
    orphaned_column_2 = orphaned_column * 2
    t3 = BaseTable()
    t3["A"] = orphaned_column_2

    orphaned_column.replace(mapping={2: 20, 3: 30, 4: 40})
    z = set(orphaned_column[:].tolist())
    assert {2, 3, 4}.isdisjoint(z)
    assert {20, 30, 40}.issubset(z)

    t4 = t + t2
    assert len(t4) == len(t) + len(t2)


def test_types():
    # SINGLE TYPES.
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = BaseTable(columns=data)

    c = t["A"]
    assert c.types() == {int: len(A)}

    assert t.types() == {"A": {int: len(A)}, "B": {int: len(A)}, "C": {int: len(A)}}

    # MIXED DATATYPES
    A = list(range(5))
    B = list("abcde")
    data = {"A": A, "B": B}
    t = BaseTable(columns=data)
    typ1 = t.types()
    expected = {"A": {int: 5}, "B": {str: 5}}
    assert typ1 == expected
    more = {"A": B, "B": A}
    t += BaseTable(columns=more)
    typ2 = t.types()
    expected2 = {"A": {int: 5, str: 5}, "B": {str: 5, int: 5}}
    assert typ2 == expected2


def test_table_row_functions():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = BaseTable(columns=data)

    t2 = BaseTable()
    t2.add_column("A")
    t2.add_columns("B", "C")
    t2.add_rows(
        **{
            "A": [i + max(A) for i in A],
            "B": [i + max(A) for i in A],
            "C": [i + max(C) for i in C],
        }
    )
    t3 = t2.stack(t)
    assert len(t3) == len(t2) + len(t)

    t4 = BaseTable(columns={"B": [-1, -2], "D": [0, 1]})
    t5 = t2.stack(t4)
    assert len(t5) == len(t2) + len(t4)
    assert list(t5.columns) == ["A", "B", "C", "D"]


def test_remove_all():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = BaseTable(columns=data)

    c = t["A"]
    c.remove_all(3)
    A.remove(3)
    assert list(c) == A
    c.remove_all(4, 5, 6)
    A = [i for i in A if i not in {4, 5, 6}]
    assert list(c) == A


def test_replace():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = BaseTable(columns=data)

    c = t["A"]
    c.replace({3: 30, 4: 40})
    assert list(c) == [i if i not in {3, 4} else i * 10 for i in A]


def test_display_options():
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = BaseTable(columns=data)

    d = t.display_dict()
    assert d == {
        "#": [" 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"],
        "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "B": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "C": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    }

    txt = t.to_ascii()
    expected = """\
+==+==+===+====+
|# |A | B | C  |
+--+--+---+----+
| 0| 1| 10| 100|
| 1| 2| 20| 200|
| 2| 3| 30| 300|
| 3| 4| 40| 400|
| 4| 5| 50| 500|
| 5| 6| 60| 600|
| 6| 7| 70| 700|
| 7| 8| 80| 800|
| 8| 9| 90| 900|
| 9|10|100|1000|
+==+==+===+====+"""
    assert txt == expected
    txt2 = t.to_ascii(dtype=True)
    expected2 = """\
+===+===+===+====+
| # | A | B | C  |
|row|int|int|int |
+---+---+---+----+
| 0 |  1| 10| 100|
| 1 |  2| 20| 200|
| 2 |  3| 30| 300|
| 3 |  4| 40| 400|
| 4 |  5| 50| 500|
| 5 |  6| 60| 600|
| 6 |  7| 70| 700|
| 7 |  8| 80| 800|
| 8 |  9| 90| 900|
| 9 | 10|100|1000|
+===+===+===+====+"""
    assert txt2 == expected2

    html = t._repr_html_()

    expected3 = """<div><table border=1>\
<tr><th>#</th><th>A</th><th>B</th><th>C</th></tr>\
<tr><th> 0</th><th>1</th><th>10</th><th>100</th></tr>\
<tr><th> 1</th><th>2</th><th>20</th><th>200</th></tr>\
<tr><th> 2</th><th>3</th><th>30</th><th>300</th></tr>\
<tr><th> 3</th><th>4</th><th>40</th><th>400</th></tr>\
<tr><th> 4</th><th>5</th><th>50</th><th>500</th></tr>\
<tr><th> 5</th><th>6</th><th>60</th><th>600</th></tr>\
<tr><th> 6</th><th>7</th><th>70</th><th>700</th></tr>\
<tr><th> 7</th><th>8</th><th>80</th><th>800</th></tr>\
<tr><th> 8</th><th>9</th><th>90</th><th>900</th></tr>\
<tr><th> 9</th><th>10</th><th>100</th><th>1000</th></tr>\
</table></div>"""
    assert html == expected3

    A = list(range(1, 51))
    B = [i * 10 for i in A]
    C = [i * 1000 for i in B]
    data = {"A": A, "B": B, "C": C}
    t2 = BaseTable(columns=data)
    d2 = t2.display_dict()
    assert d2 == {
        "#": [" 0", " 1", " 2", " 3", " 4", " 5", " 6", "...", "43", "44", "45", "46", "47", "48", "49"],
        "A": [1, 2, 3, 4, 5, 6, 7, "...", 44, 45, 46, 47, 48, 49, 50],
        "B": [10, 20, 30, 40, 50, 60, 70, "...", 440, 450, 460, 470, 480, 490, 500],
        "C": [
            10000,
            20000,
            30000,
            40000,
            50000,
            60000,
            70000,
            "...",
            440000,
            450000,
            460000,
            470000,
            480000,
            490000,
            500000,
        ],
    }


def test_index():
    A = [1, 1, 2, 2, 3, 3, 4]
    B = list("asdfghjkl"[: len(A)])
    data = {"A": A, "B": B}
    t = BaseTable(columns=data)
    t.index("A")
    t.index("A", "B")


def test_to_dict():
    # to_dict and as_json_serializable
    A = list(range(1, 11))
    B = [i * 10 for i in A]
    C = [i * 10 for i in B]
    data = {"A": A, "B": B, "C": C}
    t = BaseTable(columns=data)
    assert t.to_dict() == data


def test_unique():
    t = BaseTable({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert np.all(t["A"].unique() == np.array([1, 2, 3, 4]))


def test_unique_multiple_dtypes():
    t = BaseTable({"A": [0, 1.0, True, None, False, "0", "1"] * 2})
    u = list(t["A"].unique())
    assert all(i in u for i in [0, 1.0, True, None, False, "0", "1"])
    assert len(u) == 7


def test_histogram():
    t = BaseTable({"A": [1, 2, 3, 4, 5] * 3})
    u, c = t["A"].histogram()
    assert u == [1, 2, 3, 4, 5]
    assert c == [3] * len(u)


def test_histogram_multiple_dtypes():
    t = BaseTable({"A": [0, 1, True, False, None, "0", "1"] * 2})
    u, c = t["A"].histogram()
    assert u == [0, 1, True, False, None, "0", "1"]
    assert c == [2] * len(u)


def test_count():
    t = BaseTable({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert t["A"].count(2) == 3


def test_count_multiple_dtypes():
    t = BaseTable({"A": [0, 1.0, True, False, None, "0"] * 2})
    for v in t["A"]:
        assert t["A"].count(v) == 2


def test_contains():
    t = BaseTable({"A": [1, 1, 1, 2, 2, 2, 3, 3, 4]})
    assert 3 in t["A"]
    assert 7 not in t["A"]


def test_contains_multiple_dtypes():
    t = BaseTable({"A": [0, 1.0, True, False, None, "0"] * 2})
    for v in t["A"]:
        assert v in t["A"]


def test_get_by_indices_one_page():
    data = list("abcdefg")
    t = BaseTable({"A": data})
    col = t["A"]
    assert isinstance(col, Column)
    indices = np.array([6, 3, 4, 1, 2])
    values = col.get_by_indices(indices)
    expected = [data[i] for i in indices]
    assert np.all(values == expected)


def test_get_by_indices_multiple_pages():
    old_cfg = Config.PAGE_SIZE
    Config.PAGE_SIZE = 5

    data = [i for i in range(23)]
    t = BaseTable({"A": data})
    col = t["A"]
    assert isinstance(col, Column)
    indices = np.array(data[3:22:4])
    values = col.get_by_indices(indices)
    expected = [data[i] for i in indices]
    assert np.all(values == expected)

    Config.PAGE_SIZE = old_cfg


def test_get_by_indices():
    old_cfg = Config.PAGE_SIZE
    Config.PAGE_SIZE = 50

    t = BaseTable({"A": [str(i) for i in range(100)]})
    col = t["A"]
    assert isinstance(col, Column)
    indices = np.array([3, 4, 5, 6, 56, 57, 58, 3, 8, 9, 10, 59])
    values = col.get_by_indices(indices)
    expected = [str(i) for i in indices]
    assert np.all(values == expected)

    Config.PAGE_SIZE = old_cfg


def fn_foo_table(tbl):
    return tbl


def test_page_refcount():
    table = BaseTable({"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]})

    assert all(Page.refcounts.get(p.path, 0) == 1 for p in table["A"].pages), "Refcount expected to be 1"
    assert all(Page.refcounts.get(p.path, 0) == 1 for p in table["B"].pages), "Refcount expected to be 1"

    with TaskManager(1, error_mode="exception") as tm:
        """ this will cause deep table copy by copying table from main process -> child process -> main process """
        tasks = [Task(fn_foo_table, table)]

        result_table, *_ = tm.execute(tasks)

    assert all(Page.refcounts.get(p.path, 0) == 2 for p in table["A"].pages), "Refcount expected to be 2"
    assert all(Page.refcounts.get(p.path, 0) == 2 for p in table["B"].pages), "Refcount expected to be 2"

    del result_table # deleting the table should reduce the refcounts for all pages
    gc.collect()

    assert all(Page.refcounts.get(p.path, 0) == 1 for p in table["A"].pages), "Refcount expected to be 1"
    assert all(Page.refcounts.get(p.path, 0) == 1 for p in table["B"].pages), "Refcount expected to be 1"

    table.show() # make sure table is not corrupt

    a_pages = [p.path for p in table["A"].pages]
    b_pages = [p.path for p in table["B"].pages]

    del tm, tasks, table # deleting the table should reduce the refcounts for all pages
    gc.collect()

    assert all(p not in Page.refcounts for p in a_pages), "There should be no more pages left"
    assert all(p not in Page.refcounts for p in b_pages), "There should be no more pages left"

    assert all(p.exists() == False for p in a_pages), "Pages should be deleted"
    assert all(p.exists() == False for p in b_pages), "Pages should be deleted"