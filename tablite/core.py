import sys
import logging
import numpy as np
from pathlib import Path

from tqdm import tqdm as _tqdm

from tablite.base import Table as BaseTable
from tablite.base import Column  # noqa
from tablite.utils import type_check
from tablite import import_utils
from tablite import export_utils
from tablite import redux
from tablite import reindex as _reindex
from tablite import joins
from tablite import lookup
from tablite import sortation
from tablite import groupbys
from tablite import pivots
from tablite import imputation
from tablite import diff


logging.getLogger("lml").propagate = False
logging.getLogger("pyexcel_io").propagate = False
logging.getLogger("pyexcel").propagate = False

log = logging.getLogger(__name__)


class Table(BaseTable):
    def __init__(self, columns=None, headers=None, rows=None, _path=None) -> None:
        """creates Table

        Args:
            EITHER:
                columns (dict, optional): dict with column names as keys, values as lists.
                Example: t = Table(columns={"a": [1, 2], "b": [3, 4]})
            OR
                headers (list of strings, optional): list of column names.
                rows (list of tuples or lists, optional): values for columns
                Example: t = Table(headers=["a", "b"], rows=[[1,3], [2,4]])
        """
        super().__init__(columns, headers, rows, _path)

    @classmethod
    def from_file(
        cls,
        path,
        columns=None,
        first_row_has_headers=True,
        encoding=None,
        start=0,
        limit=sys.maxsize,
        sheet=None,
        guess_datatypes=True,
        newline="\n",
        text_qualifier=None,
        delimiter=None,
        strip_leading_and_tailing_whitespace=True,
        text_escape_openings="",
        text_escape_closures="",
        tqdm=_tqdm,
    ):
        """
        reads path and imports 1 or more tables

        REQUIRED
        --------
        path: pathlib.Path or str
            selection of filereader uses path.suffix.
            See `filereaders`.

        OPTIONAL
        --------
        columns:
            None: (default) All columns will be imported.
            List: only column names from list will be imported (if present in file)
                  e.g. ['A', 'B', 'C', 'D']

                  datatype is detected using Datatypes.guess(...)
                  You can try it out with:
                  >> from tablite.datatypes import DataTypes
                  >> DataTypes.guess(['001','100'])
                  [1,100]

                  if the format cannot be achieved the read type is kept.
            Excess column names are ignored.

            HINT: To get the head of file use:
            >>> from tablite.tools import head
            >>> head = head(path)

        first_row_has_headers: boolean
            True: (default) first row is used as column names.
            False: integers are used as column names.

        encoding: str. Defaults to None (autodetect using n bytes).
            n is declared in filereader_utils as ENCODING_GUESS_BYTES

        start: the first line to be read (default: 0)

        limit: the number of lines to be read from start (default sys.maxint ~ 2**63)

        OPTIONAL FOR EXCEL AND ODS READERS
        ----------------------------------

        sheet: sheet name to import  (applicable to excel- and ods-reader only)
            e.g. 'sheet_1'
            sheets not found excess names are ignored.

        OPTIONAL FOR TEXT READERS
        -------------------------
        guess_datatype: bool
            True: (default) datatypes are guessed using DataTypes.guess(...)
            False: all data is imported as strings.

        newline: newline character (applicable to text_reader only)
            str: '\n' (default) or '\r\n'

        text_qualifier: character (applicable to text_reader only)
            None: No text qualifier is used.
            str: " or '

        delimiter: character (applicable to text_reader only)
            None: file suffix is used to determine field delimiter:
                .txt: "|"
                .csv: ",",
                .ssv: ";"
                .tsv: "\t" (tab)

        strip_leading_and_tailing_whitespace: bool:
            True: default

        text_escape_openings: (applicable to text_reader only)
            None: default
            str: list of characters such as ([{

        text_escape_closures: (applicable to text_reader only)
            None: default
            str: list of characters such as }])

        """
        if isinstance(path, str):
            path = Path(path)
        type_check(path, Path)

        if not path.exists():
            raise FileNotFoundError(f"file not found: {path}")

        if not isinstance(start, int) or not 0 <= start <= sys.maxsize:
            raise ValueError(f"start {start} not in range(0,{sys.maxsize})")

        if not isinstance(limit, int) or not 0 < limit <= sys.maxsize:
            raise ValueError(f"limit {limit} not in range(0,{sys.maxsize})")

        if not isinstance(first_row_has_headers, bool):
            raise TypeError("first_row_has_headers is not bool")

        import_as = path.suffix
        if import_as.startswith("."):
            import_as = import_as[1:]

        reader = import_utils.file_readers.get(import_as, None)
        if reader is None:
            raise ValueError(f"{import_as} is not in supported format: {import_utils.valid_readers}")

        additional_configs = {"tqdm": tqdm}
        if reader == import_utils.text_reader:
            # here we inject tqdm, if tqdm is not provided, use generic iterator
            # fmt:off
            config = (path, columns, first_row_has_headers, encoding, start, limit, newline,
                      guess_datatypes, text_qualifier, strip_leading_and_tailing_whitespace,
                      delimiter, text_escape_openings, text_escape_closures)
            # fmt:on

        elif reader == import_utils.from_html:
            config = (path,)
        elif reader == import_utils.from_hdf5:
            config = (path,)

        elif reader == import_utils.excel_reader:
            # config = path, first_row_has_headers, sheet, columns, start, limit
            config = (
                str(path),
                first_row_has_headers,
                sheet,
                columns,
                start,
                limit,
            )  # if file length changes - re-import.

        if reader == import_utils.ods_reader:
            # path, first_row_has_headers=True, sheet=None, columns=None, start=0, limit=sys.maxsize,
            config = (
                str(path),
                first_row_has_headers,
                sheet,
                columns,
                start,
                limit,
            )  # if file length changes - re-import.

        # At this point the import config seems valid.
        # Now we check if the file already has been imported.

        # publish the settings
        return reader(cls, *config, **additional_configs)

    @classmethod
    def from_pandas(cls, df):
        """
        Creates Table using pd.to_dict('list')

        similar to:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6]})
        >>> df
            a  b
            0  1  4
            1  2  5
            2  3  6
        >>> df.to_dict('list')
        {'a': [1, 2, 3], 'b': [4, 5, 6]}

        >>> t = Table.from_dict(df.to_dict('list))
        >>> t.show()
            +===+===+===+
            | # | a | b |
            |row|int|int|
            +---+---+---+
            | 0 |  1|  4|
            | 1 |  2|  5|
            | 2 |  3|  6|
            +===+===+===+
        """
        return import_utils.from_pandas(cls, df)

    @classmethod
    def from_hdf5(cls, path):
        """
        imports an exported hdf5 table.
        """
        return import_utils.from_hdf5(cls, path)

    @classmethod
    def from_json(cls, jsn):
        """
        Imports table exported using .to_json
        """
        return import_utils.from_json(cls, jsn)

    def to_hdf5(self, path):
        """
        creates a copy of the table as hdf5
        """
        export_utils.to_hdf5(self, path)

    def to_pandas(self):
        """
        returns pandas.DataFrame
        """
        return export_utils.to_pandas(self)

    def to_sql(self, name):
        """
        generates ANSI-92 compliant SQL.
        """
        return export_utils.to_sql(self, name)  # remove after update to test suite.

    def to_json(self):
        """
        returns JSON
        """
        return export_utils.to_json(self)

    def to_xlsx(self, path):
        """
        exports table to path
        """
        export_utils.path_suffix_check(path, ".xlsx")
        export_utils.excel_writer(self, path)

    def to_ods(self, path):
        """
        exports table to path
        """
        export_utils.path_suffix_check(path, ".ods")
        export_utils.excel_writer(self, path)

    def to_csv(self, path):
        """
        exports table to path
        """
        export_utils.path_suffix_check(path, ".csv")
        export_utils.text_writer(self, path)

    def to_tsv(self, path):
        """
        exports table to path
        """
        export_utils.path_suffix_check(path, ".tsv")
        export_utils.text_writer(self, path)

    def to_text(self, path):
        """
        exports table to path
        """
        export_utils.path_suffix_check(path, ".txt")
        export_utils.text_writer(self, path)

    def to_html(self, path):
        """
        exports table to path
        """
        export_utils.path_suffix_check(path, ".html")
        export_utils.to_html(self, path)

    def expression(self, expression):
        """
        filters based on an expression, such as:

            "all((A==B, C!=4, 200<D))"

        which is interpreted using python's compiler to:

            def _f(A,B,C,D):
                return all((A==B, C!=4, 200<D))
        """
        return redux._filter_using_expression(self, expression)

    def filter(self, expressions, filter_type="all", tqdm=_tqdm):
        """
        enables filtering across columns for multiple criteria.

        expressions:

            str: Expression that can be compiled and executed row by row.
                exampLe: "all((A==B and C!=4 and 200<D))"

            list of dicts: (example):

                L = [
                    {'column1':'A', 'criteria': "==", 'column2': 'B'},
                    {'column1':'C', 'criteria': "!=", "value2": '4'},
                    {'value1': 200, 'criteria': "<", column2: 'D' }
                ]

            accepted dictionary keys: 'column1', 'column2', 'criteria', 'value1', 'value2'

        filter_type: 'all' or 'any'
        """
        return redux.filter(self, expressions, filter_type, tqdm)

    def sort_index(self, sort_mode="excel", tqdm=_tqdm, pbar=None, **kwargs):
        """
        helper for methods `sort` and `is_sorted`

        param: sort_mode: str: "alphanumeric", "unix", or, "excel" (default)
        param: **kwargs: sort criteria. See Table.sort()
        """
        return sortation.sort_index(self, sort_mode, tqdm=tqdm, pbar=pbar, **kwargs)

    def reindex(self, index):
        """
        index: list of integers that declare sort order.

        Examples:

            Table:  ['a','b','c','d','e','f','g','h']
            index:  [0,2,4,6]
            result: ['b','d','f','h']

            Table:  ['a','b','c','d','e','f','g','h']
            index:  [0,2,4,6,1,3,5,7]
            result: ['a','c','e','g','b','d','f','h']

        """
        if isinstance(index, list):
            index = np.array(index)
        return _reindex.reindex(self, index)

    def drop_duplicates(self, *args):
        """
        removes duplicate rows based on column names

        args: (optional) column_names
        if no args, all columns are used.
        """
        if not args:
            args = self.columns
        index = self.unique_index(*args)
        return self.reindex(index)

    def sort(self, mapping, sort_mode="excel", tqdm=_tqdm, pbar: _tqdm = None):
        """Perform multi-pass sorting with precedence given order of column names.

        Args:
            mapping (dict): keys as columns,
                            values as boolean for 'reverse'
            sort_mode: str: "alphanumeric", "unix", or, "excel"

        Returns:
            None: Table.sort is sorted inplace

        Examples:
        Table.sort(mappinp={A':False}) means sort by 'A' in ascending order.
        Table.sort(mapping={'A':True, 'B':False}) means sort 'A' in descending order, then (2nd priority)
        sort B in ascending order.
        """
        new = sortation.sort(self, mapping, sort_mode, tqdm=tqdm, pbar=pbar)
        self.columns = new.columns

    def sorted(self, mapping, sort_mode="excel", tqdm=_tqdm, pbar: _tqdm = None):
        """See sort.
        Sorted returns a new table in contrast to "sort", which is in-place.

        Returns:
            Table.
        """
        return sortation.sort(self, mapping, sort_mode, tqdm=tqdm, pbar=pbar)

    def is_sorted(self, mapping, sort_mode="excel"):
        """Performs multi-pass sorting check with precedence given order of column names.
        **kwargs: optional: sort criteria. See Table.sort()
        :return bool
        """
        return sortation.is_sorted(self, mapping, sort_mode)

    def any(self, **kwargs):
        """
        returns Table for rows where ANY kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        return redux.filter_any(self, **kwargs)

    def all(self, **kwargs):
        """
        returns Table for rows where ALL kwargs match
        :param kwargs: dictionary with headers and values / boolean callable

        Examples:

            t = Table()
            t['a'] = [1,2,3,4]
            t['b'] = [10,20,30,40]

            def f(x):
                return x == 4
            def g(x):
                return x < 20

            t2 = t.any( **{"a":f, "b":g})
            assert [r for r in t2.rows] == [[1, 10], [4, 40]]

            t2 = t.any(a=f,b=g)
            assert [r for r in t2.rows] == [[1, 10], [4, 40]]

            def h(x):
                return x>=2

            def i(x):
                return x<=30

            t2 = t.all(a=h,b=i)
            assert [r for r in t2.rows] == [[2,20], [3, 30]]


        """
        return redux.filter_all(self, **kwargs)

    def drop(self, *args):
        """
        removes all rows where args are present.

        Exmaple:
        >>> t = Table()
        >>> t['A'] = [1,2,3,None]
        >>> t['B'] = [None,2,3,4]
        >>> t2 = t.drop(None)
        >>> t2['A'][:], t2['B'][:]
        ([2,3], [2,3])

        """
        if not args:
            raise ValueError("What to drop? None? np.nan? ")
        return redux.drop(self, *args)

    def replace(self, mapping, columns=None):
        """replaces all mapped keys with values from named columns

        Args:
            mapping (dict): keys are targets for replacement,
                            values are replacements.
            columns (list or str, optional): target columns.
                Defaults to None (all columns)

        Raises:
            ValueError: _description_
        """
        if columns is None:
            columns = list(self.columns)
        if not isinstance(columns, list) and columns in self.columns:
            columns = [columns]
        type_check(columns, list)
        for n in columns:
            if n not in self.columns:
                raise ValueError(f"column not found: {n}")

        for name in columns:
            col = self.columns[name]
            col.replace(mapping)

    def groupby(self, keys, functions, tqdm=_tqdm, pbar=None):
        """
        keys: column names for grouping.
        functions: [optional] list of column names and group functions (See GroupyBy class)
        returns: table

        Example:

        t = Table()
        t.add_column('A', data=[1, 1, 2, 2, 3, 3] * 2)
        t.add_column('B', data=[1, 2, 3, 4, 5, 6] * 2)
        t.add_column('C', data=[6, 5, 4, 3, 2, 1] * 2)

        t.show()
        # +=====+=====+=====+
        # |  A  |  B  |  C  |
        # | int | int | int |
        # +-----+-----+-----+
        # |    1|    1|    6|
        # |    1|    2|    5|
        # |    2|    3|    4|
        # |    2|    4|    3|
        # |    3|    5|    2|
        # |    3|    6|    1|
        # |    1|    1|    6|
        # |    1|    2|    5|
        # |    2|    3|    4|
        # |    2|    4|    3|
        # |    3|    5|    2|
        # |    3|    6|    1|
        # +=====+=====+=====+

        g = t.groupby(keys=['A', 'C'], functions=[('B', gb.sum)])
        g.show()
        # +===+===+===+======+
        # | # | A | C |Sum(B)|
        # |row|int|int| int  |
        # +---+---+---+------+
        # |0  |  1|  6|     2|
        # |1  |  1|  5|     4|
        # |2  |  2|  4|     6|
        # |3  |  2|  3|     8|
        # |4  |  3|  2|    10|
        # |5  |  3|  1|    12|
        # +===+===+===+======+

        Cheat sheet:

        # list of unique values
        >>> g1 = t.groupby(keys=['A'], functions=[])
        >>> g1['A'][:]
        [1,2,3]

        # alternatively:
        >>> t['A'].unique()
        [1,2,3]

        # list of unique values, grouped by longest combination.
        >>> g2 = t.groupby(keys=['A', 'B'], functions=[])
        >>> g2['A'][:], g2['B'][:]
        ([1,1,2,2,3,3], [1,2,3,4,5,6])

        # alternatively:
        >>> list(zip(*t.index('A', 'B').keys()))
        [(1,1,2,2,3,3) (1,2,3,4,5,6)]

        # A key (unique values) and count hereof.
        >>> g3 = t.groupby(keys=['A'], functions=[('A', gb.count)])
        >>> g3['A'][:], g3['Count(A)'][:]
        ([1,2,3], [4,4,4])

        # alternatively:
        >>> t['A'].histogram()
        ([1,2,3], [4,4,4])

        for more exmaples see:
            https://github.com/root-11/tablite/blob/master/tests/test_groupby.py

        """
        return groupbys.groupby(self, keys, functions, tqdm=tqdm, pbar=pbar)

    def pivot(self, rows, columns, functions, values_as_rows=True, tqdm=_tqdm, pbar=None):
        """
        param: rows: column names to keep as rows
        param: columns: column names to keep as columns
        param: functions: aggregation functions from the Groupby class as

        example:

        t.show()
        # +=====+=====+=====+
        # |  A  |  B  |  C  |
        # | int | int | int |
        # +-----+-----+-----+
        # |    1|    1|    6|
        # |    1|    2|    5|
        # |    2|    3|    4|
        # |    2|    4|    3|
        # |    3|    5|    2|
        # |    3|    6|    1|
        # |    1|    1|    6|
        # |    1|    2|    5|
        # |    2|    3|    4|
        # |    2|    4|    3|
        # |    3|    5|    2|
        # |    3|    6|    1|
        # +=====+=====+=====+

        t2 = t.pivot(rows=['C'], columns=['A'], functions=[('B', gb.sum)])
        t2.show()
        # +===+===+========+=====+=====+=====+
        # | # | C |function|(A=1)|(A=2)|(A=3)|
        # |row|int|  str   |mixed|mixed|mixed|
        # +---+---+--------+-----+-----+-----+
        # |0  |  6|Sum(B)  |    2|None |None |
        # |1  |  5|Sum(B)  |    4|None |None |
        # |2  |  4|Sum(B)  |None |    6|None |
        # |3  |  3|Sum(B)  |None |    8|None |
        # |4  |  2|Sum(B)  |None |None |   10|
        # |5  |  1|Sum(B)  |None |None |   12|
        # +===+===+========+=====+=====+=====+

        """
        return pivots.pivot(self, rows, columns, functions, values_as_rows, tqdm=tqdm, pbar=pbar)

    def join(self, other, left_keys, right_keys, left_columns, right_columns, kind="inner", tqdm=_tqdm, pbar=None):
        """
        short-cut for all join functions.
        kind: 'inner', 'left', 'outer', 'cross'
        """
        kinds = {
            "inner": self.inner_join,
            "left": self.left_join,
            "outer": self.outer_join,
            "cross": self.cross_join,
        }
        if kind not in kinds:
            raise ValueError(f"join type unknown: {kind}")
        f = kinds.get(kind, None)
        return f(other, left_keys, right_keys, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def left_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param left_columns: list of left columns to retain, if None, all are retained.
        :param right_columns: list of right columns to retain, if None, all are retained.
        :return: new Table
        Example:
        SQL:   SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
        Tablite: left_join = numbers.left_join(
            letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter']
        )
        """
        return joins.left_join(self, other, left_keys, right_keys, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def inner_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param left_columns: list of left columns to retain, if None, all are retained.
        :param right_columns: list of right columns to retain, if None, all are retained.
        :return: new Table
        Example:
        SQL:   SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
        Tablite: inner_join = numbers.inner_join(
            letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter']
            )
        """
        return joins.inner_join(self, other, left_keys, right_keys, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def outer_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param left_columns: list of left columns to retain, if None, all are retained.
        :param right_columns: list of right columns to retain, if None, all are retained.
        :return: new Table
        Example:
        SQL:   SELECT number, letter FROM numbers OUTER JOIN letters ON numbers.colour == letters.color
        Tablite: outer_join = numbers.outer_join(
            letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter']
            )
        """
        return joins.outer_join(self, other, left_keys, right_keys, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def cross_join(self, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
        """
        CROSS JOIN returns the Cartesian product of rows from tables in the join.
        In other words, it will produce rows which combine each row from the first table
        with each row from the second table
        """
        return joins.cross_join(self, other, left_keys, right_keys, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    def lookup(self, other, *criteria, all=True, tqdm=_tqdm):
        """function for looking up values in `other` according to criteria in ascending order.
        :param: other: Table sorted in ascending search order.
        :param: criteria: Each criteria must be a tuple with value comparisons in the form:
            (LEFT, OPERATOR, RIGHT)
        :param: all: boolean: True=ALL, False=Any

        OPERATOR must be a callable that returns a boolean
        LEFT must be a value that the OPERATOR can compare.
        RIGHT must be a value that the OPERATOR can compare.

        Examples:
              ('column A', "==", 'column B')  # comparison of two columns
              ('Date', "<", DataTypes.date(24,12) )  # value from column 'Date' is before 24/12.
              f = lambda L,R: all( ord(L) < ord(R) )  # uses custom function.
              ('text 1', f, 'text 2')
              value from column 'text 1' is compared with value from column 'text 2'
        """
        return lookup.lookup(self, other, *criteria, all=all, tqdm=tqdm)

    def replace_missing_values(self, *args, **kwargs):
        raise AttributeError("See imputation")

    def imputation(self, targets, missing=None, method="carry forward", sources=None, tqdm=_tqdm):
        """
        In statistics, imputation is the process of replacing missing data with substituted values.

        See more: https://en.wikipedia.org/wiki/Imputation_(statistics)

        Args:
            table (Table): source table.

            targets (str or list of strings): column names to find and
                replace missing values

            missing (any): value to be replaced

            method (str): method to be used for replacement. Options:

                'carry forward':
                    takes the previous value, and carries forward into fields
                    where values are missing.
                    +: quick. Realistic on time series.
                    -: Can produce strange outliers.

                'mean':
                    calculates the column mean (exclude `missing`) and copies
                    the mean in as replacement.
                    +: quick
                    -: doesn't work on text. Causes data set to drift towards the mean.

                'mode':
                    calculates the column mode (exclude `missing`) and copies
                    the mean in as replacement.
                    +: quick
                    -: most frequent value becomes over-represented in the sample

                'nearest neighbour':
                    calculates normalised distance between items in source columns
                    selects nearest neighbour and copies value as replacement.
                    +: works for any datatype.
                    -: computationally intensive (e.g. slow)

            sources (list of strings): NEAREST NEIGHBOUR ONLY
                column names to be used during imputation.
                if None or empty, all columns will be used.

        Returns:
            table: table with replaced values.
        """
        return imputation.imputation(self, targets, missing, method, sources, tqdm=tqdm)

    def transpose(self, tqdm=_tqdm):
        return pivots.transpose(self, tqdm)

    def pivot_transpose(self, columns, keep=None, column_name="transpose", value_name="value", tqdm=_tqdm):
        """Transpose a selection of columns to rows.

        Args:
            columns (list of column names): column names to transpose
            keep (list of column names): column names to keep (repeat)

        Returns:
            Table: with columns transposed to rows

        Example:
            transpose columns 1,2 and 3 and transpose the remaining columns, except `sum`.

        Input:

        | col1 | col2 | col3 | sun | mon | tue | ... | sat | sum  |
        |------|------|------|-----|-----|-----|-----|-----|------|
        | 1234 | 2345 | 3456 | 456 | 567 |     | ... |     | 1023 |
        | 1244 | 2445 | 4456 |     |   7 |     | ... |     |    7 |
        | ...  |      |      |     |     |     |     |     |      |

        t.transpose(keep=[col1, col2, col3], transpose=[sun,mon,tue,wed,thu,fri,sat])`

        Output:

        |col1| col2| col3| transpose| value|
        |----|-----|-----|----------|------|
        |1234| 2345| 3456| sun      |   456|
        |1234| 2345| 3456| mon      |   567|
        |1244| 2445| 4456| mon      |     7|

        """
        return pivots.pivot_transpose(self, columns, keep, column_name, value_name, tqdm=tqdm)

    def diff(self, other, columns=None):
        """compares table self with table other

        Args:
            self (Table): Table
            other (Table): Table
            columns (List, optional): list of column names to include in comparison. Defaults to None.

        Returns:
            Table: diff of self and other with diff in columns 1st and 2nd.
        """
        return diff.diff(self, other, columns)
