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
A python library for tables that does everything you need in 200kB.

Install: `pip install tablite`  
Usage:  `>>> from tablite import Table`  

- it handles all datatypes: `str`, `float`, `bool`, `int`, `date`, `datetime`, `time` and type checking is automatic when you append or replace values. 
- Move fluently between disk and ram using `t.use_disk = True/False` For 10,000,000 integers python will use 4.2Mb RAM instead of 133.7 Mb.
- it can import csv*, tsv, txt, xls, xlsx, xlsm, ods, zip and log using `Table.from_file(...)`
- `file_reader` is a generator of tables, so it doesn't take up memory until the tables are consumed.
- Iterate over rows or columns with `for row in table.rows` or `for column in table.columns`.
- Create multikey index, sort, use filter, any and all to select.
- Lookup between tables using custom functions.
- Perform multikey joins with other tables.
- Perform groupby and reorganise data as a pivot table with max, min, sum, first, last, count, unique, average, st.deviation, median and mode
- Update tables with += which automatically sorts out the columns - even if they're not in perfect order.
- Calculate out-of-memory summaries using += on groupby, f.x. groupby += t1
- you can select: 
  - all rows in a column as `table['A']` 
  - rows across all columns as `table[4:8]`
  - or a slice as `list(table.filter('A', 'B', slice(4,8)))`.
- you to update with `table['A'][2] = new value`
- you can store or send data using json, by: 
  - dumping to json: `json_str = table.to_json()`, or 
  - you can load it with `Table.from_json(json_str)`.- 
- it automatically deduplicates header names that already are in use.
- you can add any type of metadata to the table as `table(some_key='some_value')` or as `table.metadata['some key'] = 'some value'`.
- you can ask `column_xyz in Table.columns` ?
- load from files with `tables = list(Table.from_file('this.csv'))` which has automatic datatype detection
- perform inner, outer & left sql join between tables as simple as `table_1.inner_join(table2, keys=['A', 'B'])` 
- summarise using `table.groupby( ... )` 
- create pivot tables using `groupby.pivot( ... )`
- perform multi-criteria lookup in tables using `table1.lookup(table2, criteria=.....`
- And everything else a python list can do, plus data type checking.

# Tutorial

To learn more see [tutorial.ipynb](https://github.com/root-11/tablite/blob/master/tutorial.ipynb)

# API 

To read the detailed documentation see [tablite](https://root-11.github.io/tablite/index.html)


