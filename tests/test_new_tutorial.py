import pytest
import numpy as np
from datetime import datetime, date, time, timedelta
from tablite import Table, DataTypes


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
    dtype = 'f' # float
    t3 = Table.import_file('tests/data/book1.csv', import_as='csv', columns={k:dtype for k in 'abcdef'})
    
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
    t4['mixed'] = [
        -1,0,1,  # regular integers
        -12345678909876543211234567890987654321,  # very very large integer
        None,np.nan,  # null values 
        "one", "",  # strings
        True,False,  # booleans
        float('inf'), 0.01,  # floats
        date(2000,1,1),   # date
        datetime(2002,2,3,23,0,4,6660),  # datetime
        time(12,12,12),  # time
        timedelta(days=3, seconds=5678)  # timedelta
    ]
    # ... is exactly what you get out:
    print(list(t4['mixed']))
    # [-1, 0, 1, 
    # -12345678909876543211234567890987654321, 
    # None, nan, 
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
    t = Table.import_file('tests/data/book1.csv', import_as='csv', columns={k:'f' for k in 'abcdef'})
    # i can now perform a number of actions:

    # filter Col A > value

    # filter Col A > Col B

    # filter with multiple criteria
    

    # SORT
    # -----------------------------------------

    # GROUPBY
    # -----------------------------------------

    # JOIN
    # -----------------------------------------

    # USER DEFINED FUNCTIONS
    # -----------------------------------------

    # REPLACE MISSING VALUES
    # -----------------------------------------

    # THE BASICS
    # -----------------------------------------