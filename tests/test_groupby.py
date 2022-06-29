from tablite.core import Table, GroupBy

gb = GroupBy


def test_groupby():
    t = Table()
    for c in 'abcde':
        t.add_column(c,data=[i for i in range(5)])

    # we want to add two new columns using the functions:
    def f1(a, b, c):
        return a + b + c + 1

    def f2(b, c, d):
        return b * c * d

    # and we want to compute two new columns 'f' and 'g':
    t.add_columns('f', 'g')

    # we can now use the filter, to iterate over the tablite:
    for row in t['a', 'b', 'c', 'd'].rows:
        a, b, c, d = row

        # ... and add the values to the two new columns
        t['f'].append(f1(a, b, c))
        t['g'].append(f2(b, c, d))

    assert len(t) == 5
    assert list(t.columns) == list('abcdefg')
    
    t+=t
    t.show()

    t2 = t.groupby(keys=['a', 'b'],
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
    t2.show()

    assert list(t2.rows) == [
        [0, 0,  1,  1,  2,  1,  1, 2, 1,  1.0, 0.0, 0.0,  1,  1,  0],
        [1, 1,  4,  4,  8,  4,  4, 2, 1,  4.0, 0.0, 0.0,  4,  4,  1],
        [2, 2,  7,  7, 14,  7,  7, 2, 1,  7.0, 0.0, 0.0,  7,  7,  8],
        [3, 3, 10, 10, 20, 10, 10, 2, 1, 10.0, 0.0, 0.0, 10, 10, 27],
        [4, 4, 13, 13, 26, 13, 13, 2, 1, 13.0, 0.0, 0.0, 13, 13, 64]
    ]
  

def test_groupby_w_pivot():
    t = Table()
    t.add_column('A', data=[1, 1, 2, 2, 3, 3] * 2)
    t.add_column('B', data=[1, 2, 3, 4, 5, 6] * 2)
    t.add_column('C', data=[6, 5, 4, 3, 2, 1] * 2)

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

    g = t.groupby(keys=['A', 'C'], functions=[('B', gb.sum)])
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

    t2 = t.pivot(rows=['C'], columns=['A'], functions=[('B', gb.sum)])
    for row in t2.rows:
        print(row)
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
    assert len(t2) == 6 and len(t2.columns) == 4+1

    t3 = t.pivot(rows=['C'], columns=['A'], functions=[('B', gb.sum)], values_as_rows=False)
    t3.show()

    t4 = t.pivot(rows=['A'], columns=['C'], functions=[('B', gb.sum)])
    t4.show()

    t5 = t.pivot(rows=['C'], columns=['A'], functions=[('B', gb.sum), ('B', gb.count)])
    t5.show()


def test_reverse_pivot():
    """ example code from the readme as "reversing a pivot tablite". """
    from random import seed, choice
    seed(11)

    records = 9
    t = Table()
    t.add_column('record id', data=[i for i in range(records)])
    for column in [f"4.{i}.a" for i in range(5)]:
        t.add_column(column, data=[choice(['a', 'h', 'e', None]) for i in range(records)])

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
    reverse_pivot.add_column('record id')
    reverse_pivot.add_column('4.x')
    reverse_pivot.add_column('ahe')

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
