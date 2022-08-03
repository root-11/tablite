# Tablite

![Build status](https://github.com/root-11/tablite/actions/workflows/python-package.yml/badge.svg)
[![Code coverage](https://codecov.io/gh/root-11/tablite/branch/master/graph/badge.svg)](https://codecov.io/gh/root-11/tablite)
[![Downloads](https://pepy.tech/badge/tablite)](https://pepy.tech/project/tablite)
[![Downloads](https://pepy.tech/badge/tablite/month)](https://pepy.tech/project/tablite)

--------------

## Overview

|Tablite 2022.7 features |  |
|---|---|
|**Even smaller memory footprint.**<br>Tablite uses HDF5 as a backend with strong abstraction, so that copy/append/repetition of data is handled in pages. This is imperative for incremental data processing such as in the image on the right, where 43M rows are processed in 208 steps.<br>Tablite achieves this by storing all data in `/tmp/tablite.hdf5` so if your OS sits on SSD it will benefit from high IOPS, and permit slices of 9,000,000,000 rows in less than a second.<br>**Multiprocessing enabled by default**<br>Tablite has multiprocessing is implemented for bypassing the GIL on all major operations. <br>CSV import is tested with 96M fields that are imported and type-mapped to native python types in 120 secs.<br>**All algorithms have been reworked to respect memory limits**<br>Tablite respects the limits of free memory by tagging the free memory and defining task size before each memory intensive task is initiated (join, groupby, data import, etc)<br>**100% support for all python datatypes**<br>Tablite uses datatype mapping to HDF5 native types where possible and uses type mapping for non-native types such as timedelta, None, date, time… e.g. what you put in, is what you get out.<br>**Light weight** - Tablite is ~200 kB.|![incremental dataprocessing](../../blob/master/images/incremental_dataprocessing.png?raw=true)<br>[click for full size image](../../blob/master/images//incremental_dataprocessing.svg) <br> ![1bn rows](../../blob/master/images/1TB_test.png?raw=true)|


## Installation

[Tablite](https://pypi.org/project/tablite/)

Install: `pip install tablite`  
Usage:  `>>> from tablite import Table`  

## General overview

A tablite `Table` is multiprocessing enabled by default and ...

- behaves like a dict with lists: `my_table[column name] = [... data ...]`
- handles all python datatypes natively: `str`, `float`, `bool`, `int`, `date`, `datetime`, `time`, `timedelta` and `None`
- uses HDF5 as storage which is faster than mmap'ed files for the average case. 10,000,000 integers python will use < 1 Mb RAM instead of 133.7 Mb (1M lists with 10 integers). The example below shows data from `tests/test_filereader_time.py` with 1 terabyte of data:

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

One-liners

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

## Tutorial

To learn more see the [tutorial.ipynb](https://github.com/root-11/tablite/blob/master/tutorial.ipynb) (Jupyter notebook)

## Credits

- Martynas Kaunas - GroupBy functionality.
- Audrius Kulikajevas - Edge case testing / various bugs.
