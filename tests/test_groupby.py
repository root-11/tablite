from tablite.core import Table  #, GroupBy

# gb = GroupBy


def test_groupby():
    t = Table()
    for c in 'abcde':
        t.add_column(header=c, datatype=int, allow_empty=False, data=[i for i in range(5)])

    # we want to add two new columns using the functions:
    def f1(a, b, c):
        return a + b + c + 1

    def f2(b, c, d):
        return b * c * d

    # and we want to compute two new columns 'f' and 'g':
    t.add_column(header='f', datatype=int, allow_empty=False)
    t.add_column(header='g', datatype=int, allow_empty=True)

    # we can now use the filter, to iterate over the tablite:
    for row in t.filter('a', 'b', 'c', 'd'):
        a, b, c, d = row

        # ... and add the values to the two new columns
        t['f'].append(f1(a, b, c))
        t['g'].append(f2(b, c, d))

    assert len(t) == 5
    assert list(t.columns) == list('abcdefg')
    t.show()

    g = GroupBy(keys=['a', 'b'],
                functions=[('f', gb.max),
                           ('f', gb.min),
                           ('f', gb.sum),
                           ('f', gb.first),
                           ('f', gb.last),
                           ('f', gb.count),
                           ('f', gb.count_unique),
                           ('f', gb.avg),
                           ('f', gb.stdev),
                           ('a', gb.stdev),
                           ('f', gb.median),
                           ('f', gb.mode),
                           ('g', gb.median)])
    t2 = t + t
    assert len(t2) == 2 * len(t)
    t2.show()

    g += t2

    assert list(g.rows) == [
        (0, 0, 1, 1, 2, 1, 1, 2, 1, 1.0, 0.0, 0.0, 1, 1, 0),
        (1, 1, 4, 4, 8, 4, 4, 2, 1, 4.0, 0.0, 0.0, 4, 4, 1),
        (2, 2, 7, 7, 14, 7, 7, 2, 1, 7.0, 0.0, 0.0, 7, 7, 8),
        (3, 3, 10, 10, 20, 10, 10, 2, 1, 10.0, 0.0, 0.0, 10, 10, 27),
        (4, 4, 13, 13, 26, 13, 13, 2, 1, 13.0, 0.0, 0.0, 13, 13, 64)
    ]

    g.table.show()

    g2 = GroupBy(keys=['a', 'b'], functions=[('f', gb.max), ('f', g.sum)])
    g2 += t + t + t

    g2.table.show()

    pivot_table = g2.pivot('b')

    pivot_table.show()


def test_groupby_02():
    t = Table()
    t.add_column('A', int, data=[1, 1, 2, 2, 3, 3] * 2)
    t.add_column('B', int, data=[1, 2, 3, 4, 5, 6] * 2)
    t.add_column('C', int, data=[6, 5, 4, 3, 2, 1] * 2)

    t.show()
    # +=====+=====+=====+
    # |  A  |  B  |  C  |
    # | int | int | int |
    # |False|False|False|
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

    g = t.groupby(keys=['A', 'C'], functions=[('B', gb.sum)])
    g.table.show()
    t2 = g.pivot('A')

    t2.show()
    # +=====+==========+==========+==========+
    # |  C  |Sum(B,A=1)|Sum(B,A=2)|Sum(B,A=3)|
    # | int |   int    |   int    |   int    |
    # |False|   True   |   True   |   True   |
    # +-----+----------+----------+----------+
    # |    5|         4|      None|      None|
    # |    6|         2|      None|      None|
    # |    3|      None|         8|      None|
    # |    4|      None|         6|      None|
    # |    1|      None|      None|        12|
    # |    2|      None|      None|        10|
    # +=====+==========+==========+==========+

    assert len(t2) == 6 and len(t2.columns) == 4


def test_ttopi():
    """ example code from the readme as "reversing a pivot tablite". """
    from random import seed, choice
    seed(11)

    records = 9
    t = Table()
    t.add_column('record id', int, allow_empty=False, data=[i for i in range(records)])
    for column in [f"4.{i}.a" for i in range(5)]:
        t.add_column(column, str, allow_empty=True, data=[choice(['a', 'h', 'e', None]) for i in range(records)])

    print("\nshowing raw data:")
    t.show()
    # +=====+=====+
    # |  A  |  B  |
    # | str | str |
    # |False|False|
    # +-----+-----+
    # |4.2.a|e    |
    # |4.3.a|h    |
    # |4.2.a|h    |
    # |4.2.a|e    |
    # |4.3.a|e    |
    # |4.3.a|e    |
    # |4.1.a|e    |
    # |4.1.a|a    |
    # |4.3.a|e    |
    # |4.2.a|a    |
    # |4.3.a|e    |
    # |4.3.a|a    |
    # |4.1.a|a    |
    # |4.1.a|a    |
    # |4.2.a|a    |
    # |4.2.a|a    |
    # |4.1.a|e    |
    # |4.1.a|a    |
    # |4.3.a|h    |
    # |4.3.a|h    |
    # |4.3.a|h    |
    # |4.1.a|e    |
    # +=====+=====+

    # wanted output:
    # +=====+=====
    # |  A  | Count(A,B=a) | Count(A,B=h) | Count(A,B=e) |
    # | str |     int      |     int      |     int      |
    # |False|    False     |    False     |    False     |
    # +-----+--------------+--------------+--------------+
    # |4.1.a|            3 |            0 |            3 |
    # |4.2.a|            3 |            1 |            2 |
    # |4.3.a|            1 |            4 |            4 |
    # +=====+==============+==============+==============+

    reverse_pivot = Table()
    records = t['record id']
    reverse_pivot.add_column('record id', records.datatype, allow_empty=False)
    reverse_pivot.add_column('4.x', str, allow_empty=False)
    reverse_pivot.add_column('ahe', str, allow_empty=True)

    for name in t.columns:
        if not name.startswith('4.'):
            continue
        column = t[name]
        for index, entry in enumerate(column):
            new_row = records[index], name, entry  # record id, 4.x, ahe
            reverse_pivot.add_row(new_row)

    print("\nshowing reversed pivot of the raw data:")
    reverse_pivot.show()

    g = reverse_pivot.groupby(['4.x', 'ahe'], functions=[('ahe', GroupBy.count)])
    print("\nshowing basic groupby of the reversed pivot")
    g.table.show()
    t2 = g.pivot('ahe')
    print("\nshowing the wanted output:")
    t2.show()
