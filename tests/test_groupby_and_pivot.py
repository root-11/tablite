from tablite.core import Table
from tablite.groupbys import GroupBy as gb
from random import seed, choice
import numpy as np
import pytest


@pytest.fixture(autouse=True)  # this resets the HDF5 file for every test.
def refresh():
    yield


def test_groupby():
    t = Table()
    for c in "abcde":
        t.add_column(c, data=[i for i in range(5)])

    # we want to add two new columns using the functions:
    def f1(a, b, c):
        return a + b + c + 1

    def f2(b, c, d):
        return b * c * d

    # and we want to compute two new columns 'f' and 'g':
    t.add_columns("f", "g")

    # we can now use the filter, to iterate over the tablite:
    f, g = [], []
    for row in t["a", "b", "c", "d"].rows:
        a, b, c, d = row

        # ... and add the values to the two new columns
        f.append(f1(a, b, c))
        g.append(f2(b, c, d))
    t["f"].extend(np.array(f))
    t["g"].extend(np.array(g))

    assert len(t) == 5
    assert list(t.columns) == list("abcdefg")

    t += t
    t.show()

    t2 = t.groupby(
        keys=["a", "b"],
        functions=[
            ("f", gb.max),
            ("f", gb.min),
            ("f", gb.sum),
            ("f", gb.product),
            ("f", gb.first),
            ("f", gb.last),
            ("f", gb.count),
            ("f", gb.count_unique),
            ("f", gb.avg),
            ("f", gb.stdev),
            ("a", gb.stdev),
            ("f", gb.median),
            ("f", gb.mode),
            ("g", gb.median),
        ],
    )
    t2.show()
    # fmt: off
    # +===+===+===+======+======+======+==========+========+=======+========+==============+==========+====================+====================+=========+=======+=========+  # noqa
    # | # | a | b |Max(f)|Min(f)|Sum(f)|Product(f)|First(f)|Last(f)|Count(f)|CountUnique(f)|Average(f)|StandardDeviation(f)|StandardDeviation(a)|Median(f)|Mode(f)|Median(g)|  # noqa
    # |row|int|int| int  | int  | int  |   int    |  int   |  int  |  int   |     int      |  float   |       float        |       float        |   int   |  int  |   int   |  # noqa
    # +---+---+---+------+------+------+----------+--------+-------+--------+--------------+----------+--------------------+--------------------+---------+-------+---------+  # noqa
    # |0  |  0|  0|     1|     1|     2|         1|       1|      1|       2|             1|       1.0|                 0.0|                 0.0|        1|      1|        0|  # noqa
    # |1  |  1|  1|     4|     4|     8|        16|       4|      4|       2|             1|       4.0|                 0.0|                 0.0|        4|      4|        1|  # noqa
    # |2  |  2|  2|     7|     7|    14|        49|       7|      7|       2|             1|       7.0|                 0.0|                 0.0|        7|      7|        8|  # noqa
    # |3  |  3|  3|    10|    10|    20|       100|      10|     10|       2|             1|      10.0|                 0.0|                 0.0|       10|     10|       27|  # noqa
    # |4  |  4|  4|    13|    13|    26|       169|      13|     13|       2|             1|      13.0|                 0.0|                 0.0|       13|     13|       64|  # noqa
    # +===+===+===+======+======+======+==========+========+=======+========+==============+==========+====================+====================+=========+=======+=========+  # noqa
    # fmt: on
    assert list(t2.rows) == [
        [0, 0, 1, 1, 2, 1, 1, 1, 2, 1, 1.0, 0.0, 0.0, 1, 1, 0],
        [1, 1, 4, 4, 8, 16, 4, 4, 2, 1, 4.0, 0.0, 0.0, 4, 4, 1],
        [2, 2, 7, 7, 14, 49, 7, 7, 2, 1, 7.0, 0.0, 0.0, 7, 7, 8],
        [3, 3, 10, 10, 20, 100, 10, 10, 2, 1, 10.0, 0.0, 0.0, 10, 10, 27],
        [4, 4, 13, 13, 26, 169, 13, 13, 2, 1, 13.0, 0.0, 0.0, 13, 13, 64],
    ]


def test_groupby_missing_args():
    t = Table()
    t.add_column("A", data=[1, 1, 2, 2, 3, 3] * 2)
    t.add_column("B", data=[1, 2, 3, 4, 5, 6] * 2)
    try:
        _ = t.groupby(keys=[], functions=[])  # value error. DONE.
        assert False
    except Exception as e:
        assert type(e).__name__ == "ValueError"
        assert type(e).__module__ == "nimpy"
        assert True

    g0 = t.groupby(keys=[], functions=[("A", gb.sum)])
    # +==+======+
    # |# |Sum(A)|
    # +--+------+
    # | 0|    24|
    # +==+======+
    assert g0["Sum(A)"] == [sum(t["A"])]

    g1 = t.groupby(keys=["A"], functions=[])  # list of unique values
    assert g1["A"] == [1, 2, 3]

    g2 = t.groupby(keys=["A", "B"], functions=[])  # list of unique values, grouped by longest combination.
    assert g2["A"] == [1, 1, 2, 2, 3, 3]
    assert g2["B"] == [1, 2, 3, 4, 5, 6]

    g3 = t.groupby(keys=["A"], functions=[("A", gb.count)])  # A key (unique values) and count hereof.
    assert g3["A"] == [1, 2, 3]
    assert g3["Count(A)"] == [4, 4, 4]


def test_groupby_w_pivot():
    t = Table()
    t.add_column("A", data=[1, 1, 2, 2, 3, 3] * 2)
    t.add_column("B", data=[1, 2, 3, 4, 5, 6] * 2)
    t.add_column("C", data=[6, 5, 4, 3, 2, 1] * 2)

    t.show()
    # +=====+=====+=====+
    # |  A  |  B  |  C  |
    # | int | int | int |
    # +-----+-----+-----+
    # |    1|    1|    6|
    # |    1|    2|    5|
    # |    2|    3|    4|
    # |    2|    4|    3|
    # |    3|    5|    2|
    # |    3|    6|    1|
    # |    1|    1|    6|
    # |    1|    2|    5|
    # |    2|    3|    4|
    # |    2|    4|    3|
    # |    3|    5|    2|
    # |    3|    6|    1|
    # +=====+=====+=====+

    g = t.groupby(keys=["A", "C"], functions=[("B", gb.sum)])
    g.show()
    # +===+===+===+======+
    # | # | A | C |Sum(B)|
    # |row|int|int| int  |
    # +---+---+---+------+
    # |0  |  1|  6|     2|
    # |1  |  1|  5|     4|
    # |2  |  2|  4|     6|
    # |3  |  2|  3|     8|
    # |4  |  3|  2|    10|
    # |5  |  3|  1|    12|
    # +===+===+===+======+

    g2 = t.groupby(keys=[], functions=[("B", gb.sum)])
    g2.show()
    # +==+======+
    # |# |Sum(B)|
    # +--+------+
    # | 0|    42|
    # +==+======+

    t2 = t.pivot(rows=["C"], columns=["A"], functions=[("B", gb.sum)])
    t2.show()
    # +===+===+========+=====+=====+=====+
    # | # | C |function|(A=1)|(A=2)|(A=3)|
    # |row|int|  str   |mixed|mixed|mixed|
    # +---+---+--------+-----+-----+-----+
    # |0  |  6|Sum(B)  |    2|None |None |
    # |1  |  5|Sum(B)  |    4|None |None |
    # |2  |  4|Sum(B)  |None |    6|None |
    # |3  |  3|Sum(B)  |None |    8|None |
    # |4  |  2|Sum(B)  |None |None |   10|
    # |5  |  1|Sum(B)  |None |None |   12|
    # +===+===+========+=====+=====+=====+
    assert len(t2) == 6 and len(t2.columns) == 4 + 1

    t3 = t.pivot(rows=["C"], columns=["A"], functions=[("B", gb.sum)], values_as_rows=False)
    t3.show()
    # +===+===+==========+==========+==========+
    # | # | C |Sum(B,A=1)|Sum(B,A=2)|Sum(B,A=3)|
    # |row|int|  mixed   |  mixed   |  mixed   |
    # +---+---+----------+----------+----------+
    # |0  |  6|         2|None      |None      |
    # |1  |  5|         4|None      |None      |
    # |2  |  4|None      |         6|None      |
    # |3  |  3|None      |         8|None      |
    # |4  |  2|None      |None      |        10|
    # |5  |  1|None      |None      |        12|
    # +===+===+==========+==========+==========+
    assert len(t3) == 6 and len(t3.columns) == 4
    t4 = t.pivot(rows=["A"], columns=["C"], functions=[("B", gb.sum)])
    t4.show()
    # +===+===+========+=====+=====+=====+=====+=====+=====+
    # | # | A |function|(C=6)|(C=5)|(C=4)|(C=3)|(C=2)|(C=1)|
    # |row|int|  str   |mixed|mixed|mixed|mixed|mixed|mixed|
    # +---+---+--------+-----+-----+-----+-----+-----+-----+
    # |0  |  1|Sum(B)  |    2|    4|None |None |None |None |
    # |1  |  2|Sum(B)  |None |None |    6|    8|None |None |
    # |2  |  3|Sum(B)  |None |None |None |None |   10|   12|
    # +===+===+========+=====+=====+=====+=====+=====+=====+
    assert len(t4) == 3 and len(t4.columns) == 8

    t5 = t.pivot(rows=["A"], columns=["C"], functions=[("B", gb.sum)], values_as_rows=False)
    t5.show()
    # +===+===+==========+==========+==========+==========+==========+==========+
    # | # | A |Sum(B,C=6)|Sum(B,C=5)|Sum(B,C=4)|Sum(B,C=3)|Sum(B,C=2)|Sum(B,C=1)|
    # |row|int|  mixed   |  mixed   |  mixed   |  mixed   |  mixed   |  mixed   |
    # +---+---+----------+----------+----------+----------+----------+----------+
    # |0  |  1|         2|         4|None      |None      |None      |None      |
    # |1  |  2|None      |None      |         6|         8|None      |None      |
    # |2  |  3|None      |None      |None      |None      |        10|        12|
    # +===+===+==========+==========+==========+==========+==========+==========+
    assert len(t5) == 3 and len(t5.columns) == 7

    g6 = t.groupby(keys=["C", "A"], functions=[("B", gb.sum), ("B", gb.count)])
    g6.show()
    # +===+===+===+======+========+
    # | # | C | A |Sum(B)|Count(B)|
    # |row|int|int| int  |  int   |
    # +---+---+---+------+--------+
    # |0  |  6|  1|     2|       2|
    # |1  |  5|  1|     4|       2|
    # |2  |  4|  2|     6|       2|
    # |3  |  3|  2|     8|       2|
    # |4  |  2|  3|    10|       2|
    # |5  |  1|  3|    12|       2|
    # +===+===+===+======+========+
    assert len(g6) == 6 and len(g6.columns) == 4
    assert all(v == 2 for v in g6["Count(B)"])
    assert g6["Sum(B)"] == [2, 4, 6, 8, 10, 12]

    t6 = t.pivot(rows=["C"], columns=["A"], functions=[("B", gb.sum), ("B", gb.count)])
    t6.show()
    # +===+===+========+=====+=====+=====+
    # | # | C |function|(A=1)|(A=2)|(A=3)|
    # |row|int|  str   |mixed|mixed|mixed|
    # +---+---+--------+-----+-----+-----+
    # |0  |  6|Sum(B)  |    2|None |None |
    # |1  |  6|Count(B)|    2|None |None |
    # |2  |  5|Sum(B)  |    4|None |None |
    # |3  |  5|Count(B)|    2|None |None |
    # |4  |  4|Sum(B)  |None |    6|None |
    # |5  |  4|Count(B)|None |    2|None |
    # |6  |  3|Sum(B)  |None |    8|None |
    # |7  |  3|Count(B)|None |    2|None |
    # |8  |  2|Sum(B)  |None |None |   10|
    # |9  |  2|Count(B)|None |None |    2|
    # |10 |  1|Sum(B)  |None |None |   12|
    # |11 |  1|Count(B)|None |None |    2|
    # +===+===+========+=====+=====+=====+
    assert len(t6) == 12 and len(t6.columns) == 5
    assert set(t6["function"]) == {"Sum(B)", "Count(B)"}
    assert t6["C"] == [6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]
    assert t6["(A=1)"] == [2, 2, 4, 2] + [None] * 8
    assert t6["(A=2)"] == [None] * 4 + [6, 2, 8, 2] + [None] * 4
    assert t6["(A=3)"] == [None] * 8 + [10, 2, 12, 2]

    t7 = t.pivot(rows=["C"], columns=["A"], functions=[("B", gb.sum), ("B", gb.count)], values_as_rows=False)
    t7.show()
    # +===+===+==========+============+==========+============+==========+============+
    # | # | C |Sum(B,A=1)|Count(B,A=1)|Sum(B,A=2)|Count(B,A=2)|Sum(B,A=3)|Count(B,A=3)|
    # |row|int|  mixed   |   mixed    |  mixed   |   mixed    |  mixed   |   mixed    |
    # +---+---+----------+------------+----------+------------+----------+------------+
    # |0  |  6|         2|           2|None      |None        |None      |None        |
    # |1  |  5|         4|           2|None      |None        |None      |None        |
    # |2  |  4|None      |None        |         6|           2|None      |None        |
    # |3  |  3|None      |None        |         8|           2|None      |None        |
    # |4  |  2|None      |None        |None      |None        |        10|           2|
    # |5  |  1|None      |None        |None      |None        |        12|           2|
    # +===+===+==========+============+==========+============+==========+============+
    assert len(t7) == 6 and len(t7.columns) == 7


def test_reverse_pivot():
    """example code from the readme as "reversing a pivot tablite"."""
    seed(11)

    records = 9
    t = Table()
    t.add_column("record id", data=[i for i in range(records)])
    for column in [f"4.{i}.a" for i in range(5)]:
        t.add_column(column, data=[choice(["a", "h", "e", None]) for i in range(records)])

    print("\nshowing raw data:")
    t.show()
    # +=========+=====+=====+=====+=====+=====+
    # |record id|4.0.a|4.1.a|4.2.a|4.3.a|4.4.a|
    # |   int   | str | str | str | str | str |
    # +---------+-----+-----+-----+-----+-----+
    # |        0|None |e    |a    |h    |e    |
    # |        1|None |h    |a    |e    |e    |
    # |        2|None |a    |h    |None |h    |
    # |        3|h    |a    |h    |a    |e    |
    # |        4|h    |None |a    |a    |a    |
    # |        5|None |None |None |None |a    |
    # |        6|h    |h    |e    |e    |a    |
    # |        7|a    |a    |None |None |None |
    # |        8|None |a    |h    |a    |a    |
    # +=========+=====+=====+=====+=====+=====+

    reconstructed = Table()
    reconstructed.add_column("record id")
    reconstructed.add_column("4.x")
    reconstructed.add_column("ahe")

    records = t["record id"]
    for name in t.columns:
        if not name.startswith("4."):
            continue
        column = t[name]
        for index, entry in enumerate(column):
            new_row = records[index], name, entry  # record id, 4.x, ahe
            reconstructed.add_rows(new_row)

    print("\nshowing reversed pivot of the raw data:")
    reconstructed.show()
    # +===+=========+=====+=====+
    # | # |record id| 4.x | ahe |
    # |row|   int   | str |mixed|
    # +---+---------+-----+-----+
    # |0  |        0|4.0.a|None |
    # |1  |        1|4.0.a|None |
    # |2  |        2|4.0.a|None |
    # |3  |        3|4.0.a|h    |
    # |4  |        4|4.0.a|h    |
    # |5  |        5|4.0.a|None |
    # |6  |        6|4.0.a|h    |
    # |...|...      |...  |...  |
    # |38 |        2|4.4.a|h    |
    # |39 |        3|4.4.a|e    |
    # |40 |        4|4.4.a|a    |
    # |41 |        5|4.4.a|a    |
    # |42 |        6|4.4.a|a    |
    # |43 |        7|4.4.a|None |
    # |44 |        8|4.4.a|a    |
    # +===+=========+=====+=====+

    g = reconstructed.groupby(keys=["4.x", "ahe"], functions=[("ahe", gb.count)])
    print("\nshowing basic groupby of the reversed pivot")
    g.show()
    # +===+=====+=====+==========+
    # | # | 4.x | ahe |Count(ahe)|
    # |row| str |mixed|   int    |
    # +---+-----+-----+----------+
    # |0  |4.0.a|None |         5|
    # |1  |4.0.a|h    |         3|
    # |2  |4.0.a|a    |         1|
    # |3  |4.1.a|e    |         1|
    # |4  |4.1.a|h    |         2|
    # |5  |4.1.a|a    |         4|
    # |6  |4.1.a|None |         2|
    # |7  |4.2.a|a    |         3|
    # |8  |4.2.a|h    |         3|
    # |9  |4.2.a|None |         2|
    # |10 |4.2.a|e    |         1|
    # |11 |4.3.a|h    |         1|
    # |12 |4.3.a|e    |         2|
    # |13 |4.3.a|None |         3|
    # |14 |4.3.a|a    |         3|
    # |15 |4.4.a|e    |         3|
    # |16 |4.4.a|h    |         1|
    # |17 |4.4.a|a    |         4|
    # |18 |4.4.a|None |         1|
    # +===+=====+=====+==========+
    assert len(g) == 19 and len(g.columns) == 3

    t2 = reconstructed.pivot(rows=["4.x"], columns=["ahe"], functions=[("ahe", gb.count)], values_as_rows=False)
    print("\nshowing the wanted output:")
    t2.show()
    # +===+=====+===================+================+================+================+
    # | # | 4.x |Count(ahe,ahe=None)|Count(ahe,ahe=h)|Count(ahe,ahe=a)|Count(ahe,ahe=e)|
    # |row| str |        int        |      int       |      int       |     mixed      |
    # +---+-----+-------------------+----------------+----------------+----------------+
    # |0  |4.0.a|                  5|               3|               1|None            |
    # |1  |4.1.a|                  2|               2|               4|               1|
    # |2  |4.2.a|                  2|               3|               3|               1|
    # |3  |4.3.a|                  3|               1|               3|               2|
    # |4  |4.4.a|                  1|               1|               4|               3|
    # +===+=====+===================+================+================+================+
    assert len(t2) == 5 and len(t2.columns) == 5
    assert t2["Count(ahe,ahe=e)"][0] is None

def test_groupby_funcs():
    # ======== MEDIAN ==========

    def createTable(valArr):
        return Table({
            'k': [1 for _ in valArr],
            'v': valArr,
        })
    
    t = createTable([1, 2, 3, 4, 5]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [3]

    t = createTable([1, 2, 3, 6, 7, 8]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [4.5]

    t = createTable([3]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [3]

    t = createTable([3, 3]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [3]

    t = createTable([3, 3, 3]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [3]
    
    t = createTable([3, 3, 6, 6, 9, 9]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [6]

    t = createTable([3, 3, 3, 9, 9, 9]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [6]

    t = createTable([-1, -1, 0, 1, 1]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [0]

    t = createTable([-1, -1, 0, 0, 1, 1]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [0]

    t = createTable([5, 4, 6, 3, 7, 2, 8, 1, 9]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [5]

    t = createTable([i / 10 for i in range(10)]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [0.45]

    t = createTable([i / 10 for i in range(1, 10)]).groupby(keys=['k'], functions=[('v', gb.median)])
    assert t["Median(v)"] == [0.5]
    # ==========================

    # ========== MAX ===========
    t = createTable([-2, -1, 0, 1, 2, 3]).groupby(keys=['k'], functions=[('v', gb.max)])
    assert t["Max(v)"] == [3]
    # ==========================

    # ========== MIN ===========
    t = createTable([-2, -1, 0, 1, 2, 3]).groupby(keys=['k'], functions=[('v', gb.min)])
    assert t["Min(v)"] == [-2]
    # ==========================

    # ========== SUM ===========
    t = createTable([-2, -1, 0, 1, 2, 3]).groupby(keys=['k'], functions=[('v', gb.sum)])
    assert t["Sum(v)"] == [3]
    # ==========================

    # ======== PRODUCT =========
    L = [1, 2, 3, 4, 5]
    x = 1
    for i in L:
        x *= i
    t = createTable([1, 2, 3, 4, 5]).groupby(keys=['k'], functions=[('v', gb.product)])
    assert t["Product(v)"] == [x]
    # ==========================

