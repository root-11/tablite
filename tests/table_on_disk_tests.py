from table import *
from datetime import date, time, datetime
import zlib


def test_basic_table():
    # creating a table incrementally is straight forward:
    table = Table(use_disk=True)
    table.use_disk = True
    table.use_disk = False
    table.use_disk = True

    table.add_column('A', int, False)
    assert 'A' in table

    table.add_column('B', str, allow_empty=False)
    assert 'B' in table

    # appending rows is easy:
    table.add_row((1, 'hello'))
    table.add_row((2, 'world'))

    # converting to and from json is easy:
    table_as_json = table.to_json()
    table2 = Table.from_json(table_as_json)

    zipped = zlib.compress(table_as_json.encode())
    a, b = len(zipped), len(table_as_json)
    print("zipping reduces to", a, "from", b, "bytes, e.g.", round(100 * a / b, 0), "% of original")

    # copying is easy:
    table3 = table.copy()

    # and checking for headers is simple:
    assert 'A' in table
    assert 'Z' not in table

    # comparisons are straight forward:
    assert table == table2 == table3

    # even if you only want to check metadata:
    table.compare(table3)  # will raise exception if they're different.

    # append is easy as + also work:
    table3x2 = table3 + table3
    assert len(table3x2) == len(table3) * 2

    # and so does +=
    table3x2 += table3
    assert len(table3x2) == len(table3) * 3

    # type verification is included:
    try:
        table.columns['A'][0] = 'Hallo'
        assert False, "A TypeError should have been raised."
    except TypeError:
        assert True

    # updating values is familiar to any user who likes a list:
    assert 'A' in table.columns
    assert isinstance(table.columns['A'], (StoredList,list))
    last_row = -1
    table['A'][last_row] = 44
    table['B'][last_row] = "Hallo"

    assert table != table2

    # if you try to loop and forget the direction, Table will tell you
    try:
        for row in table:  # wont pass
            assert False, "not possible. Use for row in table.rows or for column in table.columns"
    except AttributeError:
        assert True

    _ = [table2.add_row(row) for row in table.rows]

    before = [r for r in table2.rows]
    assert before == [(1, 'hello'), (2, 'world'), (1, 'hello'), (44, 'Hallo')]

    # as is filtering for ALL that match:
    filter_1 = lambda x: 'llo' in x
    filter_2 = lambda x: x > 3

    after = table2.all(**{'B': filter_1, 'A': filter_2})

    assert list(after.rows) == [(44, 'Hallo')]

    # as is filtering or for ANY that match:
    after = table2.any(**{'B': filter_1, 'A': filter_2})

    assert list(after.rows) == [(1, 'hello'), (1, 'hello'), (44, 'Hallo')]

    # Imagine a table with columns a,b,c,d,e (all integers) like this:
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

    # we can now use the filter, to iterate over the table:
    for row in t.filter('a', 'b', 'c', 'd'):
        a, b, c, d = row

        # ... and add the values to the two new columns
        t['f'].append(f1(a, b, c))
        t['g'].append(f2(b, c, d))

    assert len(t) == 5
    assert list(t.columns) == list('abcdefg')
    t.show()

    # slicing is easy:
    table_chunk = table2[2:4]
    assert isinstance(table_chunk, Table)

    # we will handle duplicate names gracefully.
    table2.add_column('B', int, allow_empty=True)
    assert set(table2.columns) == {'A', 'B', 'B_1'}

    # you can delete a column as key...
    del table2['B_1']
    assert set(table2.columns) == {'A', 'B'}

    # adding a computed column is easy:
    table.add_column('new column', str, allow_empty=False, data=[f"{r}" for r in table.rows])

    # part of or the whole table is easy:
    table.show()

    table.show('A', slice(0, 1))

    # updating a column with a function is easy:
    f = lambda x: x * 10
    table['A'] = [f(r) for r in table['A']]

    # using regular indexing will also work.
    for ix, r in enumerate(table['A']):
        table['A'][ix] = r * 10

    # and it will tell you if you're not allowed:
    try:
        f = lambda x: f"'{x} as text'"
        table['A'] = [f(r) for r in table['A']]
        assert False, "The line above must raise a TypeError"
    except TypeError as error:
        print("The error is:", str(error))

    # works with all datatypes:
    now = datetime.now()

    table4 = Table()
    table4.add_column('A', int, allow_empty=False, data=[-1, 1])
    table4.add_column('A', int, allow_empty=True, data=[None, 1])  # None!
    table4.add_column('A', float, False, data=[-1.1, 1.1])
    table4.add_column('A', str, False, data=["", "1"])  # Empty string is not a None, when dtype is str!
    table4.add_column('A', str, True, data=[None, "1"])  # Empty string is not a None, when dtype is str!
    table4.add_column('A', bool, False, data=[False, True])
    table4.add_column('A', datetime, False, data=[now, now])
    table4.add_column('A', date, False, data=[now.date(), now.date()])
    table4.add_column('A', time, False, data=[now.time(), now.time()])

    table4_json = table4.to_json()
    table5 = Table.from_json(table4_json)

    # .. to json and back.
    assert table4 == table5

    # And finally: I can add metadata:
    table5.metadata['db_mapping'] = {'A': 'customers.customer_name',
                                     'A_2': 'product.sku',
                                     'A_4': 'locations.sender'}

    # which also jsonifies without fuzz.
    table5_json = table5.to_json()
    table5_from_json = Table.from_json(table5_json)
    assert table5 == table5_from_json


def test_lookup_functions():  # doing lookups is supported by indexing:
    table6 = Table(use_disk=True)
    table6.add_column('A', str, data=['Alice', 'Bob', 'Bob', 'Ben', 'Charlie', 'Ben', 'Albert'])
    table6.add_column('B', str, data=['Alison', 'Marley', 'Dylan', 'Affleck', 'Hepburn', 'Barnes', 'Einstein'])

    index = table6.index('A')  # single key.
    assert index[('Bob',)] == {1, 2}

    index2 = table6.index('A', 'B')  # multiple keys.
    assert index2[('Bob', 'Dylan')] == {2}


def test_sql_joins():  # a couple of examples with SQL join:
    numbers = Table(use_disk=True)
    numbers.add_column('number', int, allow_empty=True, data=[1, 2, 3, 4, None])
    numbers.add_column('colour', str, data=['black', 'blue', 'white', 'white', 'blue'])

    letters = Table(use_disk=True)
    letters.add_column('letter', str, allow_empty=True, data=['a', 'b', 'c', 'd', None])
    letters.add_column('color', str, data=['blue', 'white', 'orange', 'white', 'blue'])

    # left join
    # SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
    left_join = numbers.left_join(letters, left_keys=['colour'], right_keys=['color'], columns=['number', 'letter'])
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
    assert [i for i in left_join['number']] == [1, 2, 2, 3, 3, 4, 4, None, None]
    assert [i for i in left_join['letter']] == [None, 'a', None, 'b', 'd', 'b', 'd', 'a', None]

    # inner join
    # SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
    inner_join = numbers.inner_join(letters, left_keys=['colour'], right_keys=['color'], columns=['number', 'letter'])
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
    outer_join = numbers.outer_join(letters, left_keys=['colour'], right_keys=['color'], columns=['number', 'letter'])
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
    assert [i for i in outer_join['number']] == [1, 2, 2, 3, 3, 4, 4, None, None, None]
    assert [i for i in outer_join['letter']] == [None, 'a', None, 'b', 'd', 'b', 'd', 'a', None, 'c']

    assert left_join != inner_join
    assert inner_join != outer_join
    assert left_join != outer_join


def test_sortation():  # Sortation

    table7 = Table(use_disk=True)
    table7.add_column('A', int, data=[1, None, 8, 3, 4, 6, 5, 7, 9], allow_empty=True)
    table7.add_column('B', int, data=[10, 100, 1, 1, 1, 1, 10, 10, 10])
    table7.add_column('C', int, data=[0, 1, 0, 1, 0, 1, 0, 1, 0])

    assert not table7.is_sorted()

    sort_order = {'B': False, 'C': False, 'A': False}

    table7.sort(**sort_order)

    assert list(table7.rows) == [
        (4, 1, 0),
        (8, 1, 0),
        (3, 1, 1),
        (6, 1, 1),
        (1, 10, 0),
        (5, 10, 0),
        (9, 10, 0),
        (7, 10, 1),
        (None, 100, 1)
    ]

    assert list(table7.filter('A', 'B', slice(4, 8))) == [(1, 10), (5, 10), (9, 10), (7, 10)]

    assert table7.is_sorted(**sort_order)

