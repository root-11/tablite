# Tablite

![Build status](https://github.com/root-11/tablite/actions/workflows/python-test.yml/badge.svg)
[![codecov](https://codecov.io/gh/root-11/tablite/branch/master/graph/badge.svg?token=A0QEWGO9R6)](https://codecov.io/gh/root-11/tablite)
[![Downloads](https://pepy.tech/badge/tablite)](https://pepy.tech/project/tablite)
[![Downloads](https://pepy.tech/badge/tablite/month)](https://pepy.tech/project/tablite)
[![PyPI version](https://badge.fury.io/py/tablite.svg)](https://badge.fury.io/py/tablite)

--------------

## Contents

- [introduction](#introduction)
- [installation](#installation)
- [feature overview](#feature_overview)
- [tutorial](#tutorial)
- [latest updates](#latest_updates)
- [credits](#credits)

## <a name="introduction"></a>Introduction 

`Tablite` seeks to be the go-to library for manipulating tabular data with an api that is as close in syntax to pure python as possible. 


### Even smaller memory footprint

Tablite uses HDF5 as a backend with strong abstraction, so that copy, append & repetition of data is handled in pages. This is imperative for [incremental data processing](https://raw.githubusercontent.com/root-11/tablite/74e7b44cfc314950b7a769316cb48d67cce725d0/images/incremental_dataprocessing.svg).

Tablite tests [for memory footprint](https://github.com/root-11/tablite/blob/master/tests/test_memory_footprint.py). One test compares the memory footprint of 10,000,000 integers where `tablite` will use < 1 Mb RAM in contrast to python which will require around 133.7 Mb of RAM (1M lists with 10 integers). Tablite also tests to assure that working with [1Tb of data](https://github.com/root-11/tablite/blob/9bb6e572538a85aee31ef8a4a60c0945a6f857a4/tests/test_filereader_performance.py#L104) is tolerable.

Tablite achieves this by using `HDF5` as storage which is faster than mmap'ed files for the average case \[[1](https://stackoverflow.com/questions/27710245/is-there-an-analysis-speed-or-memory-usage-advantage-to-using-hdf5-for-large-arr), [2](https://github.com/root-11/root-11.github.io/blob/master/content/short_intro_to_hdf5.ipynb) \] and stores all data in `/tmp/tablite.hdf5` so if your OS (windows/linux/mac) sits on a SSD it will benefit from high IOPS and permit slices of [9,000,000,000 rows in less than a second](https://github.com/root-11/tablite/blob/master/images/1TB_test.png?raw=true).

### Multiprocessing enabled by default

Tablite uses multiprocessing for bypassing the GIL on all major operations. CSV import is [tested with 96M fields](https://github.com/root-11/tablite/blob/master/tests/test_filereader_time.py) that are imported and type-mapped to native python types in 120 secs.

### All algorithms have been reworked to respect memory limits

Tablite respects the limits of free memory by tagging the free memory and defining task size before each memory intensive task is initiated (join, groupby, data import, etc)

### 100% support for all python datatypes

Tablite wants to make it easy for you to work with data. `tablite.Table's` behave like a dict with lists:

`my_table[column name] = [... data ...]`.

Tablite uses datatype mapping to HDF5 native types where possible and uses type mapping for non-native types such as timedelta, None, date, time… e.g. what you put in, is what you get out. This is inspired by [bank python](https://calpaterson.com/bank-python.html).

### Light weight

Tablite is ~200 kB.

### Helpful

Tablite wants you to be productive, so a number of helpers are available. 

- `Table.import_file` to import csv*, tsv, txt, xls, xlsx, xlsm, ods, zip and logs. There is automatic type detection (see [tutorial.ipynb](https://nbviewer.org/github/root-11/tablite/blob/master/tutorial.ipynb) )
- To peek into any supported file use `get_headers` which shows the first 10 rows.
- Use `mytable.rows` and `mytable.columns` to iterate over rows or columns.
- Create multi-key `.index` for quick lookups.
- Perform multi-key `.sort`,
- Filter using `.any` and `.all` to select specific rows.
- use multi-key `.lookup` and `.join` to find data across tables.
- Perform `.groupby` and reorganise data as a `.pivot` table with max, min, sum, first, last, count, unique, average, st.deviation, median and mode
- Append / concatenate tables with `+=` which automatically sorts out the columns - even if they're not in perfect order.
- Should you tables be similar but not the identical you can use `.stack` to "stack" tables on top of each other

If you're still missing something add it to the [wishlist](https://github.com/root-11/tablite/issues)


---------------

## <a name="installation"></a>Installation

Get it from pypi: [Tablite](https://pypi.org/project/tablite/) [![PyPI version](https://badge.fury.io/py/tablite.svg)](https://badge.fury.io/py/tablite)

Install: `pip install tablite`  
Usage:  `>>> from tablite import Table`  

## <a name="feature_overview"></a>Feature overview

|want to...| this way... |
|---|---|
|loop over rows| `[ row for row in table.rows ]`|
|loop over columns| `[ table[col_name] for col_name in table.columns ]`|
|slice | `myslice = table['A', 'B', slice(0,None,15)]`|
|get column by name | `my_table['A']` |
|get row by index | `my_table[9_000_000_001]` |
|value update| `mytable['A'][2] = new value` |
|update w. list comprehension | `mytable['A'] = [ x*x for x in mytable['A'] if x % 2 != 0 ]`|
|join| `a_join = numbers.join(letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter'], kind='left')`|
| lookup| `travel_plan = friends.lookup(bustable, (DataTypes.time(21, 10), "<=", 'time'), ('stop', "==", 'stop'))`|
| groupby| `group_by = table.groupby(keys=['C', 'B'], functions=[('A', gb.count)])`|
| pivot table | `my_pivot = t.pivot(rows=['C'], columns=['A'], functions=[('B', gb.sum), ('B', gb.count)], values_as_rows=False)`|
| index| `indices = old_table.index(*old_table.columns)`|
| sort| `lookup1_sorted = lookup_1.sort(**{'time': True, 'name':False, "sort_mode":'unix'})`|
| filter    | `true, false = unfiltered.filter( [{"column1": 'a', "criteria":">=", 'value2':3}, ... more criteria ... ], filter_type='all' )`|
| find any  | `any_even_rows = mytable.any('A': lambda x : x%2==0, 'B': lambda x > 0)`|
| find all  | `all_even_rows = mytable.all('A': lambda x : x%2==0, 'B': lambda x > 0)`|
| to json   | `json_str = my_table.to_json()`|
| from json | `Table.from_json(json_str)`|


## <a name="tutorial"></a>Tutorial

To learn more see the [tutorial.ipynb](https://github.com/root-11/tablite/blob/master/tutorial.ipynb) (Jupyter notebook)


## <a name="latest_updates"></a>Latest updates

See [changelog.md](https://github.com/root-11/tablite/blob/master/changelog.md)


## <a name="credits"></a>Credits

- Martynas Kaunas - GroupBy functionality.
- Audrius Kulikajevas - Edge case testing / various bugs, Jupyter notebook integration.
- Sergej Sinkarenko - various bugs.
- Ovidijus Grigas - various bugs, documentation.
- Lori Cooper - spell checking.
