# Changelog

| Version    | Change                                              |
|:-----------|-----------------------------------------------------|
|2023.6.1| Major change of the backend processes. Speed up of ~6x. For more see the [release notes](https://github.com/root-11/tablite/releases/tag/2023.6.1) |
| 2022.11.19 | Fixed some memory leaks. |
| 2022.11.18 | `copy`, `filter`, `sort`, `any`, `all` methods now properly respects the table subclass.<br>Filter for tables with under `SINGLE_PROCESSING_LIMIT` rows will run on same process to reduce overhead.<br>Errors within child processes now properly propagate to parent.<br>`Table.reset_storage(include_imports=True)` now allows the user to reset the storage but exclude any imported files by setting `include_imports=False` during `Table.reset(...)`.<br>Bug: A column with `1,None,2` would be written to csv & tsv as `"1,None,2"`. Now it is written `"1,,2"` where None means absent.<br>Fix mp `join` producing mismatched columns lengths when different table lengths are used as an input or when join product is longer than the input table. |
| 2022.11.17 | `Table.load` now properly subclassess the table instead of always resulting in `tablite.Table`.<br>`Table.from_*` methods now respect subclassess, fixed some `from_*` methods which were instance methods and not class methods.<br>Fixed `Table.from_dict` only accepting `list` and `tuple` but not `tablite.Column` which is an equally valid type.<br>Fix `lookup` parity in single process and multiple process outputs.<br>Fix an issue with multiprocess `lookup` where no matches would throw instead of producing `None`.<br>Fix an issue with filtering an empty table. |
| 2022.11.16 | Changed `join` to process 1M rows per task to avoid potential OOM on lower memory systems.<br> Added `mp_merge_columns` to `MemoryManager` that merges column pages into a single column.<br>Fix `join` parity in single process and multiple process outputs.<br>Fix an issue with multiprocess `join` where no matches would throw instead of producing `None`. |
| 2022.11.15 | Bump `mplite` to avoid deadlock issues OS kill the process. |
| 2022.11.14 | Improve locking mechanism to allow retries when opening file as the previous solution could cause deadlocks when running multiple threads. |
| 2022.11.13 | Fix an issue with copying empty pages. |
| 2022.11.12 | Tablite now is now able to create it's own temporary directory. |
| 2022.11.11 | `text_reader` tqdm tracks the entire process now. <br> `text_reader` properly respects free memory in *nix based systems. <br> `text_reader` no longer discriminates against hyperthreaded cores.
| 2022.11.10 | `get_headers` now uses plain `openpyxl` instead of `pyexcel` wrapper to speed up fetch times ~10x on certain files. |
| 2022.11.9 | `get_headers` can fail safe on unrecognized characters. |
| 2022.11.8 | Fix a bug with task size calculation on single core systems. |
| 2022.11.7 | Added `TABLITE_TMPDIR` environment variable for setting tablite work directory. <br> Characters that fail to be read text reader due to improper encoding will be skipped. <br> Fixed an issue where single column text files with no column delimiters would be imported as empty tables. |
| 2022.11.6 | Date inference fix |
| 2022.11.5 | Fixed negative slicing issues |
| 2022.11.4 | Transpose API changes: <br> `table.transpose(...)` was renamed to `table.pivot_transpose(...)` <br> new `table.transpose()` and `table.T` were added, it's functionality acts similarly to `numpy.T`, the column headers are used the first row in the table when transposing. |
| 2022.11.3 | Bugfix for non-ascii encoded strings during `t.add_rows(...)` |
| 2022.11.2 | As `utf-8` is ascii compatible, the file reader utils selects `utf-8` instead of `ascii` as a default. |
| 2022.11.1 | bugfix in `datatypes.infer()` where 1 was inferred as int, not float. |
| 2022.11.0 | New table features: <br>`Table.diff(other, columns=...)`, <br>`table.remove_duplicates_rows()`, <br>`table.drop_na(*arg)`,<br>`table.replace(target,replacement)`,<br> `table.imputation(sources, targets, methods=...)`, <br>`table.to_pandas()` and `Table.from_pandas(pd.DataFrame)`,<br>`table.to_dict(columns, slice)`, <br>`Table.from_dict()`,<br>`table.transpose(columns, keep, ...)`,<br> New column features: <br> `Column.count(item)`, <br>`Column[:]` is guaranteed to return a python list.<br>`Column.to_numpy(slice)` returns `np.ndarray`. <br> new `tools` library: `from tablite import tools` with: <br> `date_range(start,end)`, <br>`xround(value, multiple, up=None)`, and, <br> `guess` as short-cut for `Datatypes.guess(...)`.<br> bugfixes: <br> `__eq__` was updated but missed `__ne__`.<br>`in` operator in filter would crash if datatypes were not strings. |
| 2022.10.11 | filter now accepts any expression (str) that can be compiled by pythons compiler |
| 2022.10.11 | Bugfix for `.any` and `.all`. The code now executes much faster|
| 2022.10.10 | Bugfix for `Table.import_file`: `import_as` has been removed from keywords.|
| 2022.10.10 | All Table functions now have tqdm progressbar. |
| 2022.10.10 | More robust calculation for task size for multiprocessing. |
| 2022.10.10 | Dependency update: mplite==1.2.0 is now required. |
| 2022.10.9 | Bugfix for `Table.import_file`: <br>files with duplicate header names would only have last duplicate name imported.<br>Now the headers are made unique using `name_x` where x is a number.|
| 2022.10.8 | Bugfix for groupby: <br>Where keys are empty error should have been raised.<br>Where there are no functions, unique keypairs are returned.|
| 2022.10.7 | Bugfix for Column.statistics() for an empty column |
| 2022.10.6 | Bugfix for `__setitem__`: tbl['a'] = [] is now seen as `tbl.add_column('a')`<br>Bugfix for `__getitem__`: calling a missing key raises keyerror. |
| 2022.10.5 | Bugfix for summary statistics. |
| 2022.10.4 | Bugfix for join shortcut. |
| 2022.10.3 | Bugfix for DataTypes where bool was evaluated wrongly |
| 2022.10.0 | Added ability to reindex in `table.reindex(index=[0,1...,n,n-1])` |
| 2022.9.0 | Added ability to store python objects ([example](https://github.com/root-11/tablite/blob/master/tests/test_api_basics.py#L111)).<br>Added warning when user iterates over non-rectangular dataset.|
| 2022.8.0 | Added `table.export(path)` which exports tablite Tables to file format given by the file extension. For example `my_table.export('example.xlsx')`.<br>supported formats are: `json`, `html`, `xlsx`, `xls`, `csv`, `tsv`, `txt`, `ods` and `sql`.| 
| 2022.7.8 | Added ability to forward `tqdm` progressbar into `Table.import_file(..., tqdm=your_tqdm)`, so that Jupyter notebook can use it in `display`-methods. |
| 2022.7.7 | Added method `Table.to_sql()` for export to ANSI-92 SQL engines<br>Bugfix on to_json for `timedelta`. <br>Jupyter notebook provides nice view using `Table._repr_html_()` <br>JS-users can use `.as_json_serializable` where suitable. |
| 2022.7.6 | get_headers now takes argument `(path, linecount=10)` |
| 2022.7.5 | added helper `Table.as_json_serializable` as Jupyterkernel compat. |
| 2022.7.4 | adder helper `Table.to_dict`, and updated `Table.to_json` |
| 2022.7.3 | table.to_json now takes kwargs: `row_count`, `columns`, `slice_`, `start_on` |
| 2022.7.2 | documentation update. |
| 2022.7.1 | minor bugfix. |
| 2022.7.0 | BREAKING CHANGES<br>- Tablite now uses HDF5 as backend. <br>- Has multiprocessing enabled by default. <br>- Is 20x faster. <br>- Completely new API. |
| 2022.6.0 | `DataTypes.guess([list of strings])` returns the best matching python datatype. |
