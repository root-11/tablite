# Tablite 

![Build status](https://github.com/root-11/tablite/actions/workflows/python-package.yml/badge.svg)
[![Code coverage](https://codecov.io/gh/root-11/tablite/branch/master/graph/badge.svg)](https://codecov.io/gh/root-11/tablite)
[![Downloads](https://pepy.tech/badge/tablite)](https://pepy.tech/project/tablite)
[![Downloads](https://pepy.tech/badge/tablite/month)](https://pepy.tech/project/tablite/month)

--------------

# Overview


We're all tired of reinventing the wheel when we need to process a bit of data.

- Pandas has a huge memory overhead when the datatypes are messy (hint: They are!).
- Numpy has become a language of it's own. It just doesn't seem pythonic anymore.
- Arrows [isn't ready](https://arrow.apache.org/docs/python/dataset.html).
- SQLite is great but just too slow, particularly on disk.
- Protobuffer is just overkill for storing data when I still need to implement all the analytics after that.

So what do we do? We write a custom built class for the problem at hand and
discover that we've just spent 3 hours doing something that should have taken
20 minutes. No more please!

### Solution: [Tablite](https://pypi.org/project/tablite/)
A python library for tables that does everything you need in < 200 kB.

Install: `pip install tablite`  
Usage:  `>>> from tablite import Table`  


`Table` is multiprocessing enabled by default and ...

- behaves like a dict with lists: `my_table[column name] = [... data ...]`
- handles all python datatypes natively: `str`, `float`, `bool`, `int`, `date`, `datetime`, `time`, `timedelta` and `None` 
- uses HDF5 as storage which is faster than mmap'ed files for the average case. 10,000,000 integers python will use < 1 Mb RAM instead of 133.7 Mb (1M lists with 10 integers).

An instance of a table allows you to:

- get rows in a column as `mytable['A']`
- get rows across all columns as `mytable[4:8]`
- slice as `mytable['A', 'B', slice(4,8) ]`.
- update individual values with `mytable['A'][2] = new value`
- update many values even faster with list comprehensions such as: `mytable['A'] = [ f(x) for x in mytable['A'] if x % 2 != 0 ]`

You can:

- Use `Table.import_file` to import csv*, tsv, txt, xls, xlsx, xlsm, ods, zip and logs. There is automatic type detection (see [tutorial.ipynb](https://github.com/root-11/tablite/blob/master/tutorial.ipynb))

- To peek into any supported file use `get_headers` which shows the first 10 rows.
- Use `mytable.rows` and `mytable.columns` to iterate over rows or columns.
- Create multi-key `.index` for quick lookups.
- Perform multi-key `.sort`, 
- Filter using `.any` and `.all` to select specific rows.
- use multi-key `.lookup` and `.join` to find data across tables.
- Perform `.groupby` and reorganise data as a `.pivot` table with max, min, sum, first, last, count, unique, average, st.deviation, median and mode
- Append / concatenate tables with `+=` which automatically sorts out the columns - even if they're not in perfect order.
- Should you tables be similar but not the identical you can use `.stack` to "stack" tables on top of each other.

You can store or send data using json, by: 
- dumping to json: `json_str = table.to_json()`, or 
- you can load it with `Table.from_json(json_str)`.- 


**One-liners**

- loop over rows: `[ row for row in table.rows ]`
- loop over columns: `[ table[col_name] for col_name in table.columns ]`
- slice: myslice = `table['A', 'B', slice(0,None,15)]` 
- join: `left_join = numbers.left_join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'])` 
- lookup: `travel_plan = friends.lookup(bustable, (DataTypes.time(21, 10), "<=", 'time'), ('stop', "==", 'stop'))`
- groupby: `group_by = table.groupby(keys=['C', 'B'], functions=[('A', gb.count)])` 
- pivot table `my_pivot = t.pivot(rows=['C'], columns=['A'], functions=[('B', gb.sum), ('B', gb.count)], values_as_rows=False)`
- index: `indices = old_table.index(*old_table.columns)` 
- sort: `lookup1_sorted = lookup_1.sort(**{'time': True, 'name':False, "sort_mode":'unix'})`
- filter: `true,false = unfiltered.filter( [{"column1": 'a', "criteria":">=", 'value2':3}, ... more criteria ... ], filter_type='all' )`
- any: `even = mytable.any('A': lambda x : x%2==0, 'B': lambda x > 0)`
- all: `even = mytable.all('A': lambda x : x%2==0, 'B': lambda x > 0)`


# Tutorial

To learn more see [tutorial.ipynb](https://github.com/root-11/tablite/blob/master/tutorial.ipynb)

# API 

To read the detailed documentation see [tablite](https://root-11.github.io/tablite/index.html)

# Credits

- Martynas Kaunas - GroupBy functionality.
- Audrius Kulikajevas - Edge case testing / various bugs.

