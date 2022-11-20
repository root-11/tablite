# Changelog

| Version    | Change                                              |
|:-----------|-----------------------------------------------------|
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