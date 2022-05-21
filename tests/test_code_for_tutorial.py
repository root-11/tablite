def test_basic_table():
    # creating a tablite incrementally is straight forward:
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
            assert False, "not possible. Use for row in tablite.rows or for column in tablite.columns"
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

    # Imagine a tablite with columns a,b,c,d,e (all integers) like this:
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

    # part of or the whole tablite is easy:
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


def test_recreate_readme_comparison():  # TODO: Use cputils for getting the memory footprint.
    try:
        import os
        import psutil
    except ImportError:
        return
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss
    from time import process_time

    from tablite import Table

    digits = 1_000_000

    records = Table()
    records.add_column('method', str)
    records.add_column('memory', int)
    records.add_column('time', float)

    records.add_row(('python', baseline_memory, 0.0))

    # Let's now use the common and convenient "row" based format:

    start = process_time()
    L = []
    for _ in range(digits):
        L.append(tuple([11 for _ in range(10)]))
    end = process_time()

    # go and check taskmanagers memory usage.
    # At this point we're using ~154.2 Mb to store 1 million lists with 10 items.
    records.add_row(('1e6 lists w. 10 integers', process.memory_info().rss - baseline_memory, round(end-start,4)))

    L.clear()

    # Let's now use a columnar format instead:
    start = process_time()
    L = [[11 for i in range(digits)] for _ in range(10)]
    end = process_time()

    # go and check taskmanagers memory usage.
    # at this point we're using ~98.2 Mb to store 10 lists with 1 million items.
    records.add_row(('10 lists with 1e6 integers', process.memory_info().rss - baseline_memory, round(end-start,4)))
    L.clear()

    # We've thereby saved 50 Mb by avoiding the overhead from managing 1 million lists.

    # Q: But why didn't I just use an array? It would have even lower memory footprint.
    # A: First, array's don't handle None's and we get that frequently in dirty csv data.
    # Second, Table needs even less memory.

    # Let's start with an array:

    import array
    start = process_time()
    L = [array.array('i', [11 for _ in range(digits)]) for _ in range(10)]
    end = process_time()
    # go and check taskmanagers memory usage.
    # at this point we're using 60.0 Mb to store 10 lists with 1 million integers.

    records.add_row(('10 lists with 1e6 integers in arrays', process.memory_info().rss - baseline_memory, round(end-start,4)))
    L.clear()

    # Now let's use Table:

    start = process_time()
    t = Table()
    for i in range(10):
        t.add_column(str(i), int, allow_empty=False, data=[11 for _ in range(digits)])
    end = process_time()

    records.add_row(('Table with 10 columns with 1e6 integers', process.memory_info().rss - baseline_memory, round(end-start,4)))

    # go and check taskmanagers memory usage.
    # At this point we're using  97.5 Mb to store 10 columns with 1 million integers.

    # Next we'll use the api `use_stored_lists` to drop to disk:
    start = process_time()
    t.use_disk = True
    end = process_time()
    records.add_row(('Table on disk with 10 columns with 1e6 integers', process.memory_info().rss - baseline_memory, round(end-start,4)))

    # go and check taskmanagers memory usage.
    # At this point we're using  24.5 Mb to store 10 columns with 1 million integers.
    # Only the metadata remains in pythons memory.

    records.show()
