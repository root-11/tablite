# Tablite 

[![Build Status](https://travis-ci.org/root-11/tablite.svg?branch=master)](https://travis-ci.org/root-11/tablite)
[![Code coverage](https://codecov.io/gh/root-11/tablite/branch/master/graph/badge.svg)](https://codecov.io/gh/root-11/tablite)
[![Downloads](https://pepy.tech/badge/tablite)](https://pepy.tech/project/tablite)
[![Downloads](https://pepy.tech/badge/tablite/month)](https://pepy.tech/project/tablite/month)


We're all tired of reinventing the wheel when we need to process a bit of data.

- Pandas has a huge memory overhead when the datatypes are messy (hint: They are!).
- Numpy has become a language of it's own. It just doesn't seem pythonic anymore.
- Arrows [isn't ready](https://arrow.apache.org/docs/python/dataset.html).
- SQLite is great but just too slow, particularly on disk.
- Protobuffer is just overkill for storing data when I still need to implement all the analytics after that.

So what do we do? We write a custom built class for the problem at hand and
discover that we've just spent 3 hours doing something that should have taken
20 minutes. No more please!

### Enter: [Tablite](https://pypi.org/project/tablite/)
A python library for tables that does everything you need in 89kB.

Install: `pip install tablite`  
Usage:  `>>> import table`  

- it handles all datatypes: `str`, `float`, `bool`, `int`, `date`, `datetime`, `time`.
- you can select: 
  - all rows in a column as `table['A']` 
  - rows across all columns as `table[4:8]`
  - or a slice as `list(table.filter('A', 'B', slice(4,8)))`.
- you to update with `table['A'][2] = new value`
- you can store or send data using json, by: 
  - dumping to json: `json_str = table.to_json()`, or 
  - you can load it with `Table.from_json(json_str)`.
- Type checking is automatic when you append or replace values. 
- it checks if the header name is already in use.
- you can add any type of metadata to the table as `table(some_key='some_value')` or as `table.metadata['some key'] = 'some value'`.
- you can ask `column_xyz in Table.colums` ?
- load from files with `tables = list(Table.from_file('this.csv'))` which has automatic datatype detection
- perform inner, outer & left sql join between tables as simple as `table_1.inner_join(table2, keys=['A', 'B'])` 
- summarise using `table.groupby( ... )` 
- create pivot tables using `groupby.pivot( ... )`


Here are some examples:

```
# 1. Create the table.
table = Table()  # in memory
table_on_disk = Table(use_disk=True)  # on disk in Tempfile.gettempdir()

# 2. Add a column
table.add_column('A', int, False)

# 3. check that the column is there.
assert 'A' in table

# 4. Add another column that doesn't tolerate None's
table.add_column('B', str, allow_empty=False)

# 5. appending a couple of rows:
table.add_row((1, 'hello'))
table.add_row((2, 'world'))

# 6. converting to json is easy:
table_as_json = table.to_json()

# 7. loading from json is easy:
table2 = Table.from_json(table_as_json)

# 8. for storing in a database or on disk, I recommend to zip the json.
zipped = zlib.compress(table_as_json.encode())

# 9. copying is easy:
table3 = table.copy()

# 10. comparing tables are straight forward:
assert table == table2 == table3

# 11. comparing metadata is also straight forward 
# (for example if you want to append the one table to the other)
table.compare(table3)  # will raise exception if they're different.

# 12. The plus operator `+` also works:
table3x2 = table3 + table3
assert len(table3x2) == len(table3) * 2

# 13. and so does plus-equal: +=
table3x2 += table3
assert len(table3x2) == len(table3) * 3

# 14. updating values is familiar to any user who likes a list:
assert 'A' in table.columns
assert isinstance(table.columns['A'], list)
last_row = -1
table['A'][last_row] = 44
table['B'][last_row] = "Hallo"

# 15. type verification is included, and complaints if you're doing it wrong:
try:
    table.columns['A'][0] = 'Hallo'
    assert False, "A TypeError should have been raised."
except TypeError:
    assert True

# 16. slicing is easy:
table_chunk = table2[2:4]
assert isinstance(table_chunk, Table)

# 17. we will handle duplicate names gracefully.
table2.add_column('B', int, allow_empty=True)
assert set(table2.columns) == {'A', 'B', 'B_1'}

# 18. you can delete a column as key...
del table2['B_1']
assert set(table2.columns) == {'A', 'B'}

# 19. adding a computed column is easy:
table.add_column('new column', str, allow_empty=False, data=[f"{r}" for r in table.rows])

# 20. iterating over the rows is easy, but if you forget `.rows` will remind you.
try:
    for row in table: # <-- wont pass needs: table.rows
        assert False, "not possible. Use for row in table.rows or for column in table.columns"
except AttributeError:
    assert True

print(table)
for row in table.rows:
    print(row)

# but if you just want to view it interactively (or a slice of it), use:
table.show()

+ =====+=====+============= +
|   A  |  B  |  new column  |
|  int | str |     str      |
| False|False|    False     |
+ -----+-----+------------- +
|     1|hello| (1, 'hello') |
|    44|Hallo|(44, 'Hallo') |
+ =====+=====+============= +

table.show('A', slice(0,1))

+ ===== +
|   A   |
|  int  |
| False |
+ ----- +
|     1 |
+ ===== +


# 21 .updating a column with a function is easy:
f = lambda x: x * 10
table['A'] = [f(r) for r in table['A']]

# 22. using regular indexing will also work.
for ix, r in enumerate(table['A']):
    table['A'][ix] = r * 10

# 23. and it will tell you if you're not allowed:
try:
    f = lambda x: f"'{x} as text'"
    table['A'] = [f(r) for r in table['A']]
    assert False, "The line above must raise a TypeError"
except TypeError as error:
    print("The error is:", str(error))

# 24. works with all datatypes ...
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

# ...to and from json:
table4_json = table4.to_json()
table5 = Table.from_json(table4_json)

assert table4 == table5

# 25. doing lookups is supported by indexing:
table6 = Table()
table6.add_column('A', str, data=['Alice', 'Bob', 'Bob', 'Ben', 'Charlie', 'Ben', 'Albert'])
table6.add_column('B', str, data=['Alison', 'Marley', 'Dylan', 'Affleck', 'Hepburn', 'Barnes', 'Einstein'])

index = table6.index('A')  # single key.
assert index[('Bob',)] == {1, 2}

index2 = table6.index('A', 'B')  # multiple keys.
assert index2[('Bob', 'Dylan')] == {2}

# 26. And finally: You can add metadata until the cows come home:
table5.metadata['db_mapping'] = {'A': 'customers.customer_name',
                                 'A_2': 'product.sku',
                                 'A_4': 'locations.sender'}

table5_json = table5.to_json()
table5_from_json = Table.from_json(table5_json)
assert table5 == table5_from_json

```

### Okay, great. How do I load data?

Easy. Use `filereader`. Here's an example:

```
from pathlib import Path
from table import filereader

for filename in ['data.csv', 'data.xlsx', 'data.txt', 'data.tsv', 'data.ods']:
    path = Path(filename)
    for table in filereader(path):
        assert isinstance(t, Table)
        ...
```

table.filereader currently accepts the following formats:  

`csv, tsv, txt, xls, xlsx, xlsm, ods, zip, log.`

And should you have some wicked format like:  

```19-Sep 02:59:47.153 web_id821 LOG 62.13.11.127 [3] (USER_N) ProcessScannedItem() : Scan[35572] LineNo 201636 scanned 1 of product 2332```

you can provide a split_sequence as a keyword:

```
table = filereader('web.log', split_sequence `" ", " ", " ", " "," [", "] (", ") ", " : ", "LineNo ", " scanned ", "of "`)
```

I've included all formats in the test suite that are publicly available from 
the alan turing institute, dateutils and csv reader. 

### Cool. Does it play well with plotly?

Yes. Here's an example you can copy paste:
```
from table import Table

t = Table()
t.add_column('a', int, data=[1, 2, 8, 3, 4, 6, 5, 7, 9], allow_empty=True)
t.add_column('b', int, data=[10, 100, 3, 4, 16, -1, 10, 10, 10])

t[:5].show()

    +=====+=====+
    |  a  |  b  |
    | int | int |
    | True|False|
    +-----+-----+
    |    1|   10|
    |    2|  100|
    |    8|    3|
    |    3|    4|
    |    4|   16|
    +=====+=====+

import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(y=t['a']))
fig.add_trace(go.Bar(y=t['b']))
fig.update_layout(title = 'Hello Figure')
fig.show()
```

![new plot](images/newplot.png)


### But what do I do if I'm about to run out of memory?

```
import table
table.new_tables_use_disk = True  # Sets global property for all new tables.

path = 'zip_full_of_large_csvs.zip'  # 314 Gb unzipped.
tables = filereader(path)  # uses 11 Mb of memory to manage 314 Gb of data.

```

Let's do a comparison:

```
from table import StoredList, Table

digits = 1_000_000

# go and check taskmanagers memory usage. 
# At this point were using ~20.3 Mb ram to python started.

# Let's now use the common and convenient "row" based format:

L = []
for _ in range(digits):
    L.append(tuple([11 for _ in range(10)]))

# go and check taskmanagers memory usage.
# At this point we're using ~154.2 Mb to store 1 million lists with 10 items. 
L.clear() 

# Let's now use a columnar format instead:
L = [[11 for i in range(digits)] for _ in range(10)]

# go and check taskmanagers memory usage.
# at this point we're using ~98.2 Mb to store 10 lists with 1 million items.
L.clear() 

```
We've thereby saved 50 Mb by avoiding the overhead from managing 1 million lists.

    Python alone: 20.3 Mb
    1,000,000 lists of 10 values: 154 Mb ram with 134 Mb for the lists and data.
    10 lists of 1,000,000 values: 98.2 Mb ram with 78 Mb for lists and data.
    Saved: 100% - (78 Mb / 134 Mb) = 44%. 

Q: But why didn't I just use an array? It would have even lower memory footprint.  
A: First, array's don't handle None's and we get that frequently in dirty csv data.  
Second, Table needs even less memory.

Let's start with an array:
```
import array
L = [array.array('i', [11 for _ in range(digits)]) for _ in range(10)]
# go and check taskmanagers memory usage.
# at this point we're using 60.0 Mb to store 10 lists with 1 million integers.
L.clear() 
```

Now let's use Table:

```
t = Table()
for i in range(10):
    t.add_column(str(i), int, allow_empty=False, data=[11 for _ in range(digits)])

# go and check taskmanagers memory usage.
# At this point we're using  97.5 Mb to store 10 columns with 1 million integers.

# Next we'll use the api `use_stored_lists` to drop to disk:
t.use_disk = True

# go and check taskmanagers memory usage.
# At this point we're using  24.5 Mb to store 10 columns with 1 million integers.
# Only the metadata remains in pythons memory.
```

Conclusion a drop from 154.2 Mb to 24.5 Mb working memory using tables in
1 line of code: `t.use_disk = True`.

If you want all your tables to be on disk, set the class variable `Table.new_tables_use_disk = True`.  
If you want a single table to be on disk, use `t = Table(use_disk=True)`.  
If you want an existing table to drop to disk use: `t.use_disk = True`


----------------

### At this point you know it's a solid bread and butter table. 
### No surprises. Now onto the features that will save a lot of time.


### Iteration

**_Iteration_** supports for loops and list comprehension at the speed of c: 

Just use `[r for r in table.rows]`, or:

    for row in table.rows:
        row ...

Here's a more practical use case:

(1) Imagine a table with columns a,b,c,d,e (all integers) like this:
```
t = Table()
_ = [t.add_column(header=c, datatype=int, allow_empty=False, data=[i for i in range(5)]) for c in 'abcde']
```

(2) we want to add two new columns using the functions:
```
def f1(a,b,c): 
    return a+b+c+1
def f2(b,c,d): 
    return b*c*d
```

(3) and we want to compute two new columns `f` and `g`:
```
t.add_column(header='f', datatype=int, allow_empty=False)
t.add_column(header='g', datatype=int, allow_empty=True)
```

(4) we can now use the filter, to iterate over the table, and add the values to the two new columns:
```
for row in t.filter('a', 'b', 'c', 'd'):
    a, b, c, d = row

    t['f'].append(f1(a, b, c))
    t['g'].append(f2(b, c, d))

assert len(t) == 5
assert list(t.columns) == list('abcdefg')

```

### Sort


**_Sort_** supports multi-column sort as simple as `table.sort(**{'A': False, 'B': True, 'C': False})`

Here's an example:

```
table7 = Table()
table7.add_column('A', int, data=[1, None, 8, 3, 4, 6, 5, 7, 9], allow_empty=True)
table7.add_column('B', int, data=[10, 100, 1, 1, 1, 1, 10, 10, 10])
table7.add_column('C', int, data=[0, 1, 0, 1, 0, 1, 0, 1, 0])

table7.sort(**{'B': False, 'C': False, 'A': False})

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
```
This takes us to filter.


### Filter


**_Filter_** allows selection of particular rows like: 

    for row in table.filter('b', 'a', 'a', 'c')
        b,a,a,c = row

So and if you only want a slice, for example column A and B for rows 4-8 from 
table7, you'd do it like this: `list(table7.filter('A', 'B', slice(4, 8)))`.

Hereby you'd get:
```
assert list(table7.filter('A', 'B', slice(4, 8))) == [
    (1, 10), 
    (5, 10), 
    (9, 10), 
    (7, 10)
]
```
As you can see, the table is sorted first by column `B` in ascending order, then
by column 'C' and finally by column 'A'. Note that `None` is handled as `float('-inf')`


### Create Index / Indices


**_Index_** supports multi-key indexing using args: `table.index('B','C')`. 

This gives you a dictionary with the key as a tuple and the indices as a set, e.g. 

    indices = {
        (1, 44): {2,3,33,35}
        (2, 44): {4,5,32}
    }



### search using ALL and ANY

**_All_** allows copy of a table where "all" criteria match.

This allows you to use custom functions like this:

```
before = [r for r in table2.rows]
assert before == [(1, 'hello'), (2, 'world'), (1, 'hello'), (44, 'Hallo')]

# as is filtering for ALL that match:
filter_1 = lambda x: 'llo' in x
filter_2 = lambda x: x > 3

after = table2.all(**{'B': filter_1, 'A': filter_2})

assert list(after.rows) == [(44, 'Hallo')]
```

**_Any_** works like `all` except it retrieves where "any" criteria match instead of all.

```
after = table2.any(**{'B': filter_1, 'A': filter_2})

assert list(after.rows) == [(1, 'hello'), (1, 'hello'), (44, 'Hallo')]
```

----------------

### SQL join operations


**_SQL JOINs_** are supported out of the box. 

Here are a couple of examples:

```
# We start with creating two tables:
left = Table()
left.add_column('number', int, allow_empty=True, data=[1, 2, 3, 4, None])
left.add_column('colour', str, data=['black', 'blue', 'white', 'white', 'blue'])

right = Table()
right.add_column('letter', str, allow_empty=True, data=['a', 'b,', 'c', 'd', None])
right.add_column('colour', str, data=['blue', 'white', 'orange', 'white', 'blue'])
```

**Left join** would in SQL be:
`SELECT number, letter FROM left LEFT JOIN right on left.colour == right.colour`

with table it's:
```
left_join = left.left_join(right, keys=['colour'], columns=['number', 'letter'])
```

**Inner join** would in SQL be:
`SELECT number, letter FROM left JOIN right ON left.colour == right.colour`

with table it's
```
inner_join = left.inner_join(right, keys=['colour'],  columns=['number','letter'])
```


**Outer join** would in SQL be:
`SELECT number, letter FROM left OUTER JOIN right ON left.colour == right.colour`

with table it's:

```
outer_join = left.outer_join(right, keys=['colour'], columns=['number','letter'])
```

----------------

### Groupby and Pivot tables


**_GroupBy_** operations are supported using the GroupBy class.

It allows summarising the data for all of the functions below:

```
g = GroupBy(keys=['a', 'b'],  # <-- Group by these columns
            functions=[('f', Max),  # <-- find the max on column `f` for each group.
                       ('f', Min),
                       ('f', Sum),
                       ('f', First),
                       ('f', Last),
                       ('f', Count),
                       ('f', CountUnique),
                       ('f', Average),
                       ('f', StandardDeviation),
                       ('a', StandardDeviation),
                       ('f', Median),
                       ('f', Mode),
                       ('g', Median)])
t2 = t + t
assert len(t2) == 2 * len(t)
t2.show()

+ =====+=====+=====+=====+=====+=====+===== +
|   a  |  b  |  c  |  d  |  e  |  f  |  g   |
|  int | int | int | int | int | int | int  |
| False|False|False|False|False|False| True |
+ -----+-----+-----+-----+-----+-----+----- +
|     0|    0|    0|    0|    0|    1|    0 |
|     1|    1|    1|    1|    1|    4|    1 |
|     2|    2|    2|    2|    2|    7|    8 |
|     3|    3|    3|    3|    3|   10|   27 |
|     4|    4|    4|    4|    4|   13|   64 |
|     0|    0|    0|    0|    0|    1|    0 |
|     1|    1|    1|    1|    1|    4|    1 |
|     2|    2|    2|    2|    2|    7|    8 |
|     3|    3|    3|    3|    3|   10|   27 |
|     4|    4|    4|    4|    4|   13|   64 |
+ =====+=====+=====+=====+=====+=====+===== +

g += t2

assert list(g.rows) == [
    (0, 0, 1, 1, 2, 1, 1, 2, 1, 1.0, 0.0, 0.0, 1, 1, 0),
    (1, 1, 4, 4, 8, 4, 4, 2, 1, 4.0, 0.0, 0.0, 4, 4, 1),
    (2, 2, 7, 7, 14, 7, 7, 2, 1, 7.0, 0.0, 0.0, 7, 7, 8),
    (3, 3, 10, 10, 20, 10, 10, 2, 1, 10.0, 0.0, 0.0, 10, 10, 27),
    (4, 4, 13, 13, 26, 13, 13, 2, 1, 13.0, 0.0, 0.0, 13, 13, 64)
]

g.table.show()

+ =====+=====+======+======+======+========+=======+========+==============+==========+====================+====================+=========+=======+========= +
|   a  |  b  |Max(f)|Min(f)|Sum(f)|First(f)|Last(f)|Count(f)|CountUnique(f)|Average(f)|StandardDeviation(f)|StandardDeviation(a)|Median(f)|Mode(f)|Median(g) |
|  int | int | int  | int  | int  |  int   |  int  |  int   |     int      |  float   |       float        |       float        |   int   |  int  |   int    |
| False|False| True | True | True |  True  |  True |  True  |     True     |   True   |        True        |        True        |   True  |  True |   True   |
+ -----+-----+------+------+------+--------+-------+--------+--------------+----------+--------------------+--------------------+---------+-------+--------- +
|     0|    0|     1|     1|     2|       1|      1|       2|             1|       1.0|                 0.0|                 0.0|        1|      1|        0 |
|     1|    1|     4|     4|     8|       4|      4|       2|             1|       4.0|                 0.0|                 0.0|        4|      4|        1 |
|     2|    2|     7|     7|    14|       7|      7|       2|             1|       7.0|                 0.0|                 0.0|        7|      7|        8 |
|     3|    3|    10|    10|    20|      10|     10|       2|             1|      10.0|                 0.0|                 0.0|       10|     10|       27 |
|     4|    4|    13|    13|    26|      13|     13|       2|             1|      13.0|                 0.0|                 0.0|       13|     13|       64 |
+ =====+=====+======+======+======+========+=======+========+==============+==========+====================+====================+=========+=======+========= +

```
 
Note that groupby is instantiated on it's own, without any data, and then
data is added using `+=` ? That's because I wanted the GroupBy class to be
friendly to updates that otherwise might run out of memory. Here's the case:

(1) Imagine you have a large number of files that you want to summarize.

For this you first need a groupby operation:
```
g = GroupBy(keys=['a','b'], functions=[('c', StandardDeviation), ('d', Average)])
```

(2) now you can just iterate over the files and not having to worry about 
the memory footprint, as each table is consumed by the groupby function:
```
files = Path(__file__).parent / 'archive'
assert files.isdir()  
for file in files.iterdir():
    json_str = json.loads(file.read())
    table = Table.from_json(json)
    g += table
```

(3) Once all files have been summarized, you can read the results using
Pythons friendly for loop:
```
for a, b, stdev, avg in g.rows:
     # ... do something ...
```

### Did I say pivot table? Yes.


**Pivot Table** included in the groupby? Yes. You can pivot the groupby on any
column that is used for grouping. Here's a simple example:

```
g2 = GroupBy(keys=['a', 'b'], functions=[('f', Max), ('f', Sum)])
g2 += t + t + t

g2.table.show()

+=====+=====+======+======+
|  a  |  b  |Max(f)|Sum(f)|
| int | int | int  | int  |
|False|False| True | True |
+-----+-----+------+------+
|    0|    0|     1|     3|
|    1|    1|     4|    12|
|    2|    2|     7|    21|
|    3|    3|    10|    30|
|    4|    4|    13|    39|
+=====+=====+======+======+

pivot_table = g2.pivot(columns=['b'])

pivot_table.show()

+=====+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+
|  a  |Max(f,b=0)|Sum(f,b=0)|Max(f,b=1)|Sum(f,b=1)|Max(f,b=2)|Sum(f,b=2)|Max(f,b=3)|Sum(f,b=3)|Max(f,b=4)|Sum(f,b=4)|
| int |   int    |   int    |   int    |   int    |   int    |   int    |   int    |   int    |   int    |   int    |
|False|   True   |   True   |   True   |   True   |   True   |   True   |   True   |   True   |   True   |   True   |
+-----+----------+----------+----------+----------+----------+----------+----------+----------+----------+----------+
|    0|         1|         3|      None|      None|      None|      None|      None|      None|      None|      None|
|    1|      None|      None|         4|        12|      None|      None|      None|      None|      None|      None|
|    2|      None|      None|      None|      None|         7|        21|      None|      None|      None|      None|
|    3|      None|      None|      None|      None|      None|      None|        10|        30|      None|      None|
|    4|      None|      None|      None|      None|      None|      None|      None|      None|        13|        39|
+=====+==========+==========+==========+==========+==========+==========+==========+==========+==========+==========+
```

---------------------

### Conclusions

This concludes the mega-tutorial to `tablite`. There's nothing more to it.
But oh boy it'll save a lot of time.

Here's a summary of features:

- Everything a list can do, plus data type checking.
- import csv*, tsv, txt, xls, xlsx, xlsm, ods, zip and log using `Table.from_file(...)`
- import multiple files use `filereader`.
- Move fluently between disk and ram using `t.use_disk = True/False`
- Iterate over rows or columns
- Create multikey index, sort, use filter, any and all to select.
- Perform multikey joins with other tables.
- Perform groupby and reorganise data as a pivot table with max, min, sum, first, last, count, unique, average, st.deviation, median and mode
- Update tables with += which automatically sorts out the columns - even if they're not in perfect order.
- Calculate out-of-memory summaries using += on groupby, f.x. groupby += t1
  



