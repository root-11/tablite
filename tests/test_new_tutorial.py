import math
import time as cputime
import pytest
import random
import pathlib
import numpy as np
from datetime import datetime, date, time, timedelta
from tablite import Table, DataTypes, GroupBy, get_headers


@pytest.fixture(autouse=True) # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield


def test_the_basics():

    # THE BASICS
    # -----------------------------------------
    # there are three ways to create tables:

    # one column at a time (faster)
    t = Table()
    t['A'] = [1,2,3]
    t['B'] = ['a','b','c']
    
    # all columns at once (slower)
    t2 = Table()
    t2.add_columns('A','B')
    t2.add_rows((1,'a'),(2,'b'),(3,'c'))
    assert t==t2

    # load data:
    path = pathlib.Path('tests/data/book1.csv')
    t3 = Table.import_file(path)
    
    # to view any table use .show(). Note that show gives either first and last 7 rows or the whole table if it is less than 20 rows.
    t3.show()
    # +===+===+===========+===========+===========+===========+===========+
    # | # | a |     b     |     c     |     d     |     e     |     f     |
    # |row|str|    str    |    str    |    str    |    str    |    str    |
    # +---+---+-----------+-----------+-----------+-----------+-----------+
    # |0  |1  |0.060606061|0.090909091|0.121212121|0.151515152|0.181818182|
    # |1  |2  |0.121212121|0.242424242|0.484848485|0.96969697 |1.939393939|
    # |2  |3  |0.242424242|0.484848485|0.96969697 |1.939393939|3.878787879|
    # |3  |4  |0.484848485|0.96969697 |1.939393939|3.878787879|7.757575758|
    # |4  |5  |0.96969697 |1.939393939|3.878787879|7.757575758|15.51515152|
    # |5  |6  |1.939393939|3.878787879|7.757575758|15.51515152|31.03030303|
    # |6  |7  |3.878787879|7.757575758|15.51515152|31.03030303|62.06060606|
    # |...|...|...        |...        |...        |...        |...        |
    # |38 |39 |16659267088|33318534175|66637068350|1.33274E+11|2.66548E+11|
    # |39 |40 |33318534175|66637068350|1.33274E+11|2.66548E+11|5.33097E+11|
    # |40 |41 |66637068350|1.33274E+11|2.66548E+11|5.33097E+11|1.06619E+12|
    # |41 |42 |1.33274E+11|2.66548E+11|5.33097E+11|1.06619E+12|2.13239E+12|
    # |42 |43 |2.66548E+11|5.33097E+11|1.06619E+12|2.13239E+12|4.26477E+12|
    # |43 |44 |5.33097E+11|1.06619E+12|2.13239E+12|4.26477E+12|8.52954E+12|
    # |44 |45 |1.06619E+12|2.13239E+12|4.26477E+12|8.52954E+12|1.70591E+13|
    # +===+===+===========+===========+===========+===========+===========+

    # should you however want to select the headers instead of importing everything
    # (which maybe timeconsuming), simply use get_headers(path)
    sample = get_headers(path)
    headers = sample.get(path.name)[0]
    print(headers)

    # to extend a table by adding columns, use t[new] = [new values]
    t['C'] = [4,5,6]
    # but make sure the column has the same length as the rest of the table!

    t.show()
    # +===+===+===+
    # | A | B | C |
    # |int|str|int|
    # +---+---+---+
    # |  1|a  |  4|
    # |  2|b  |  5|
    # |  3|c  |  6|
    # +===+===+===+

    # should you want to mix datatypes, tablite will not complain:
    # What you put in ...
    t4 = Table()
    L = [
        -1,0,1,  # regular integers
        -12345678909876543211234567890987654321,  # very very large integer
        None,  # null values 
        "one", "",  # strings
        True,False,  # booleans
        float('inf'), 0.01,  # floats
        date(2000,1,1),   # date
        datetime(2002,2,3,23,0,4,6660),  # datetime
        time(12,12,12),  # time
        timedelta(days=3, seconds=5678)  # timedelta
    ]

    t4['mixed'] = L
    # ... is exactly what you get out:
    assert t4['mixed'] == L
    # [-1, 0, 1, 
    # -12345678909876543211234567890987654321, 
    # None, 
    # 'one', '', 
    # True, True, 
    # inf, 0.01, 
    # datetime.date(2000, 1, 1), 
    # datetime.datetime(2002, 2, 3, 23, 0, 4, 6660), 
    # datetime.time(12, 12, 12), 
    # datetime.timedelta(days=3, seconds=5678)
    # ]

    print(t4['mixed'])
    # <Column>(16 values | key=25)
    
    # to view the datatypes in a column, use Column.types()
    type_dict = t4['mixed'].types()
    for k,v in type_dict.items():
        print(k,v)
    # <class 'NoneType'> 1
    # <class 'bool'> 2
    # <class 'int'> 4
    # <class 'float'> 3
    # <class 'str'> 2
    # <class 'datetime.datetime'> 1
    # <class 'datetime.date'> 1
    # <class 'datetime.time'> 1
    # <class 'datetime.timedelta'> 1
    
    # you may notice that all datatypes in t3 are str. To convert to the most probable
    # datatype used the datatype modules .guess function on each column
    t3['a'] = DataTypes.guess(t3['a'])
    # You can also convert the datatype using a list comprehension
    t3['b'] = [float(v) for v in t3['b']]
    t3.show()


    # APPEND
    # -----------------------------------------
    
    # to append one table to another, use + or += 
    print('length before:', len(t3))  # length before: 45
    t5 = t3 + t3  
    print('length after +', len(t5))  # length after + 90
    t5 += t3 
    print('length after +=', len(t5))  # length after += 135

    # if you need a lot of numbers for a test, you can repeat a table using * and *=
    t5 *= 1_000
    print('length after +=', len(t5))  # length after += 135000

    # if your are in doubt whether your tables will be the same you can use .stack(other)
    assert t.columns != t2.columns  # compares list of column names.
    t6 = t.stack(t2)
    t6.show()
    # +===+===+=====+
    # | A | B |  C  |
    # |int|str|mixed|
    # +---+---+-----+
    # |  1|a  |    4|
    # |  2|b  |    5|
    # |  3|c  |    6|
    # |  1|a  |None |
    # |  2|b  |None |
    # |  3|c  |None |
    # +===+===+=====+

    # As you can see above, t6['C'] is padded with "None" where t2 was missing the columns.
    
    # if you need a more detailed view of the columns you can iterate:
    for name in t.columns:
        col_from_t = t[name]
        if name in t2.columns:
            col_from_t2 = t2[name]
            print(name, col_from_t == col_from_t2)
        else:
            print(name, "not in t2")
    # prints:
    # A True
    # B True
    # C not in t2

    # to make a copy of a table, use table.copy()
    t3_copy = t3.copy()
    assert t3_copy == t3
    # you can also perform multi criteria selections using getitem [ ... ]
    t3_slice = t3['a','b','d', 5:25:5]
    t3_slice.show()
    # +===+===========+===========+
    # | a |     b     |     d     |
    # |str|    str    |    str    |
    # +---+-----------+-----------+
    # |6  |1.939393939|7.757575758|
    # |11 |62.06060606|248.2424242|
    # |16 |1985.939394|7943.757576|
    # |21 |63550.06061|254200.2424|
    # +===+===========+===========+

    #deleting items also works the same way:
    del t3_slice[1:3]  # delete row number 2 & 3 
    t3_slice.show()
    # +===+===========+===========+
    # | a |     b     |     d     |
    # |str|    str    |    str    |
    # +---+-----------+-----------+
    # |6  |1.939393939|7.757575758|
    # |21 |63550.06061|254200.2424|
    # +===+===========+===========+

    # to wipe a table, use .clear:
    t3_slice.clear()
    t3_slice.show()
    # prints "Empty table"

    # SAVE
    # -----------------------------------------
    # tablite uses HDF5 as the backend storage because it is fast.
    # this means you can make a table persistent using .save
    t5.save = True
    key = t5.key
    del t5
    stored_tables = Table.reload_saved_tables()
    old_t5 = [t for t in stored_tables if t.key == key][0]
    print("the t5 table had", len(old_t5), "rows")  # the t5 table had 135000 rows

    # to clear out all stored tables, use .reset_storage
    Table.reset_storage()
    assert Table.reload_saved_tables() == []
    # this can be useful when writing tests!

def test_filter():
    # FILTER
    # -----------------------------------------
    # in this example we will reload book1.csv:
    t = Table.import_file('tests/data/book1.csv')
    # i can now perform a number of actions:

    # filter Col A > value

    # filter Col A > Col B

    # filter with multiple criteria
    

    # SORT
    # -----------------------------------------
def test_sort():
    table = Table()
    table.add_column('A', data=[ 1, None, 8, 3, 4, 6,  5,  7,  9])
    table.add_column('B', data=[10,'100', 1, 1, 1, 1, 10, 10, 10])
    table.add_column('C', data=[ 0,    1, 0, 1, 0, 1,  0,  1,  0])

    sort_order = {'B': False, 'C': False, 'A': False}
    assert not table.is_sorted(**sort_order)

    sorted_table = table.sort(**sort_order)

    assert list(sorted_table.rows) == [
        [4, 1, 0],
        [8, 1, 0],
        [3, 1, 1],
        [6, 1, 1],
        [1, 10, 0],
        [5, 10, 0],
        [9, 10, 0],
        [7, 10, 1],
        [None, "100", 1]  # multi type sort "excel style"
    ]

    assert list(sorted_table['A', 'B', slice(4, 8)].rows) == [[1, 10], [5, 10], [9, 10], [7, 10]]

    assert sorted_table.is_sorted(**sort_order)

def test_sort_parallel():
    table = Table()
    n = math.ceil(1_000_000 / (9*3))
    table.add_column('A', data=[ 1, None, 8,   3, 4, 6,  5,  7,  9]*n)
    table.add_column('B', data=[10,  100, 1, "1", 1, 1, 10, 10, 10]*n)
    table.add_column('C', data=[ 0,    1, 0,   1, 0, 1,  0,  1,  0]*n)
    table.show()

    start = cputime.time()
    sort_order = {'B': False, 'C': False, 'A': False}
    sorted_table = table.sort(**sort_order)  # sorts 1M values.
    print("table sorting took ", round(cputime.time() - start,3), "secs")
    sorted_table.show()
    assert set(sorted_table['A']) == set(table['A'])
    assert set(sorted_table['B']) == set(table['B']) == {1,"1",10,100}
    assert set(sorted_table['C']) == set(table['C']) == {0,1}

    z1, ts1 = set(),[]
    for i in table.rows:
        t = tuple(i)
        if t in z1:
            continue
        else:
            z1.add(t)
            ts1.append(t)

    z2,ts2 = set(),[]
    for i in sorted_table.rows:
        t = tuple(i)
        if t in z2:
            continue
        else:
            z2.add(t)
            ts2.append(t)

    assert z1==z2
    assert ts1 != ts2

    assert ts2 == [
        (4, 1, 0), 
        (8, 1, 0), 
        (6, 1, 1), 
        (1, 10, 0), 
        (5, 10, 0), 
        (9, 10, 0), 
        (7, 10, 1), 
        (None, 100, 1), 
        (3, '1', 1)
    ]  # correct "excel"-style sort. Differs from previous test as the last '1' is text!


    # GROUPBY
    # -----------------------------------------
def test_group_by_logic():
    table = Table()
    n = math.ceil(1_000_000 / (9*3))
    table.add_column('A', data=[ 1, None, 8,   3, 4, 6,  5,  7,  9]*n)
    table.add_column('B', data=[10,  100, 1, "1", 1, 1, 10, 10, 10]*n)
    table.add_column('C', data=[ 0,    1, 0,   1, 0, 1,  0,  1,  0]*n)
    table.show()

    gb = GroupBy
    grpby = table.groupby(keys=['C', 'B'], functions=[('A', gb.count)])
    grpby.show()

    # JOIN
    # -----------------------------------------
def test_join_logic():
    numbers = Table()
    numbers.add_column('number', data=[      1,      2,       3,       4,   None])
    numbers.add_column('colour', data=['black', 'blue', 'white', 'white', 'blue'])

    letters = Table()
    letters.add_column('letter', data=[  'a',     'b',      'c',     'd',   None])
    letters.add_column('color', data=['blue', 'white', 'orange', 'white', 'blue'])

    # left join
    # SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
    left_join = numbers.left_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])
    left_join.show()
    # +======+======+
    # |number|letter|
    # | int  | str  |
    # | True | True |
    # +------+------+
    # |     1|None  |
    # |     2|a     |
    # |     2|None  |
    # |     3|b     |
    # |     3|d     |
    # |     4|b     |
    # |     4|d     |
    # |None  |a     |
    # |None  |None  |
    # +======+======+
    expected = [[1, None], [2, 'a'], [2, None], [None, 'a'], [None, None], [3, 'b'], [3, 'd'], [4, 'b'], [4, 'd']]
    assert expected == [r for r in left_join.rows]

    # inner join
    # SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
    inner_join = numbers.inner_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])
    inner_join.show()
    # +======+======+
    # |number|letter|
    # | int  | str  |
    # | True | True |
    # +------+------+
    # |     2|a     |
    # |     2|None  |
    # |None  |a     |
    # |None  |None  |
    # |     3|b     |
    # |     3|d     |
    # |     4|b     |
    # |     4|d     |
    # +======+======+
    assert [i for i in inner_join['number']] == [2, 2, None, None, 3, 3, 4, 4]
    assert [i for i in inner_join['letter']] == ['a', None, 'a', None, 'b', 'd', 'b', 'd']

    # outer join
    # SELECT number, letter FROM numbers OUTER JOIN letters ON numbers.colour == letters.color
    outer_join = numbers.outer_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])
    outer_join.show()
    # +======+======+
    # |number|letter|
    # | int  | str  |
    # | True | True |
    # +------+------+
    # |     1|None  |
    # |     2|a     |
    # |     2|None  |
    # |     3|b     |
    # |     3|d     |
    # |     4|b     |
    # |     4|d     |
    # |None  |a     |
    # |None  |None  |
    # |None  |c     |
    # +======+======+
    expected = [[1, None], [2, 'a'], [2, None], [None, 'a'], [None, None], [3, 'b'], [3, 'd'], [4, 'b'], [4, 'd'], [None, 'c']]
    assert expected == [r for r in outer_join.rows]

    assert left_join != inner_join
    assert inner_join != outer_join
    assert left_join != outer_join

    # LOOKUP
    # -----------------------------------------
def test_lookup_logic():
    friends = Table()
    friends.add_column("name", data=['Alice', 'Betty', 'Charlie', 'Dorethy', 'Edward', 'Fred'])
    friends.add_column("stop", data=['Downtown-1', 'Downtown-2', 'Hillside View', 'Hillside Crescent', 'Downtown-2', 'Chicago'])
    friends.show()

    random.seed(11)
    table_size = 40

    times = [DataTypes.time(random.randint(21, 23), random.randint(0, 59)) for i in range(table_size)]
    stops = ['Stadium', 'Hillside', 'Hillside View', 'Hillside Crescent', 'Downtown-1', 'Downtown-2',
             'Central station'] * 2 + [f'Random Road-{i}' for i in range(table_size)]
    route = [random.choice([1, 2, 3]) for i in stops]

    bustable = Table()
    bustable.add_column("time", data=times)
    bustable.add_column("stop", data=stops[:table_size])
    bustable.add_column("route", data=route[:table_size])

    bustable.sort(**{'time': False})

    print("Departures from Concert Hall towards ...")
    bustable.show(slice(0,10))

    lookup_1 = friends.lookup(bustable, (DataTypes.time(21, 10), "<=", 'time'), ('stop', "==", 'stop'))
    lookup1_sorted = lookup_1.sort(**{'time': True, 'name':False, "sort_mode":'unix'})
    lookup1_sorted.show()

    expected = [
        ['Dorethy', 'Hillside Crescent', time(23, 54), 'Hillside Crescent', 1], 
        ['Alice', 'Downtown-1', time(23, 12), 'Downtown-1', 3], 
        ['Charlie', 'Hillside View', time(22, 28), 'Hillside View', 1], 
        ['Betty', 'Downtown-2', time(21, 51), 'Downtown-2', 1], 
        ['Edward', 'Downtown-2', time(21, 51), 'Downtown-2', 1], 
        ['Fred', 'Chicago', None, None, None]
        ]
    assert expected == [r for r in lookup1_sorted.rows]

    # USER DEFINED FUNCTIONS
    # -----------------------------------------

    # REPLACE MISSING VALUES
    # -----------------------------------------

