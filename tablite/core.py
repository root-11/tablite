import json
import zipfile
import operator
import pyexcel
import pyperclip

from collections import defaultdict
from itertools import count, chain
from pathlib import Path
from tempfile import gettempdir

from tablite.datatypes import DataTypes
from tablite.file_reader_utils import detect_encoding, detect_seperator, split_by_sequence, text_escape
from tablite.groupby_utils import Max, Min, Sum, First, Last, Count, CountUnique, Average, StandardDeviation, Median, \
    Mode, GroupbyFunction

from tablite.columns import StoredColumn, InMemoryColumn
from tablite.stored_list import tempfile


class Table(object):
    new_tables_use_disk = False

    """ The main workhorse for data processing. """
    def __init__(self, **kwargs):
        self.columns = {}
        self._use_disk = kwargs.pop('use_disk', self.new_tables_use_disk)
        self.metadata = {**kwargs}

    @property
    def use_disk(self):
        return self._use_disk

    @use_disk.setter
    def use_disk(self, value):
        if not isinstance(value, bool):
            raise TypeError(str(value))
        if self._use_disk == value:
            return

        self._use_disk = value
        if value is True:
            C = StoredColumn
        else:
            C = InMemoryColumn

        for col_name, column in self.columns.items():
            self.columns[col_name] = C(col_name, column.datatype, column.allow_empty, data=column)

    def __eq__(self, other):
        if not isinstance(other, Table):
            a, b = self.__class__.__name__, other.__class__.__name__
            raise TypeError(f"cannot compare {a} with {b}")
        if self.metadata != other.metadata:
            return False
        if not all(a == b for a, b in zip(self.columns.values(), other.columns.values())):
            return False
        return True

    def __len__(self):
        """ returns length of longest column."""
        if self.columns.values():
            return max(len(c) for c in self.columns.values())
        else:
            return 0

    def __bool__(self):
        return any(self.columns)

    def __copy__(self):
        t = Table(use_disk=self._use_disk)
        for col in self.columns.values():
            t.add_column(col.header, col.datatype, col.allow_empty, data=col[:])
        t.metadata = self.metadata.copy()
        return t

    def __repr__(self):
        m = self.metadata.copy()
        m['use_disk'] = self._use_disk
        kwargs = ", ".join(f"{k}={v}" for k,v in sorted(m.items()))
        return f"{self.__class__.__name__}({kwargs})"

    def __str__(self):
        variation = ""
        lengths = {k: len(v) for k, v in self.columns.items()}
        if len(set(lengths.values())) != 1:
            longest_col = max(lengths.values())
            variation = f"(except {', '.join([f'{k}({v})' for k, v in lengths.items() if v < longest_col])})"
        return f"{self.__class__.__name__}() # {len(self.columns)} columns x {len(self)} rows {variation}"

    def copy_to_clipboard(self):
        """ copy data from a Table into clipboard. """
        try:
            s = ["\t".join([f"{name}" for name in self.columns])]
            for row in self.rows:
                s.append("\t".join((str(i) for i in row)))
            s = "\n".join(s)
            pyperclip.copy(s)
        except MemoryError:
            raise MemoryError("Cannot copy to clipboard. Select slice instead.")

    @staticmethod
    def copy_from_clipboard():
        """ copy data from clipboard into Table. """
        tmpfile = tempfile(suffix='.csv')
        with open(tmpfile, 'w') as fo:
            fo.writelines(pyperclip.paste())
        g = Table.from_file(tmpfile)
        t = list(g)[0]
        del t.metadata['filename']
        return t

    def show(self, *items, blanks=None, row_count=True, metadata=False):
        """ shows the tablite.
        param: items: column names

        DEFAULT                   EXAMPLE

        t.show()                  t.show('A', 'C', slice(4), blanks="-", metadata=True)
        +=====+=====+=====+       +=====+=====+
        |  A  |  B  |  C  |       |  A  |  C  |
        | int | str | str |       | int | str |
        |False|False| True|       |False| True|
        +-----+-----+-----+       +-----+-----+
        |    0|0x   |None |       |    0|-    |
        |    1|1x   |1    |       |    1|1    |
        |    2|2x   |None |       |    2|-    |
        |    3|3x   |3    |       |    3|3    |
        |    4|4x   |None |       +=====+=====+
        |    5|5x   |5    |       (showing 4 of 10 rows)
        |    6|6x   |None |       metadata:
        |    7|7x   |7    |          filename d:\test_data.csv
        |    8|8x   |None |
        |    9|9x   |9    |
        +=====+=====+=====+
        showing all 10 rows

            Table.show('A','C', blanks="", metadata=True

        param: blanks: string to replace blanks (None is default) when shown.
        param: row_count: bool: shows rowcount at the end.
        param: metadata: bool: displays metadata at the end.
        :returns None. Output is printed to stdout.
        """
        if any(not isinstance(i, (str, slice)) for i in items):
            raise SyntaxError(f"unexpected input: {[not isinstance(i, (str, slice)) for i in items]}")

        slices = [i for i in items if isinstance(i, slice)]
        if len(slices) > 2:
            raise SyntaxError("1 > slices")
        if not slices:
            slc = slice(0, len(self), None)
        else:
            slc = slices[0]
        assert isinstance(slc, slice)

        headers = [i for i in items if isinstance(i, str)]
        if any(h not in self.columns for h in headers):
            raise ValueError(f"column not found: {[h for h in headers if h not in self.columns]}")
        if not headers:
            headers = list(self.columns)

        # starting to produce output
        c_lens = {}
        for h in headers:
            col = self.columns[h]
            assert isinstance(col, (InMemoryColumn, StoredColumn))
            c_lens[h] = max(
                [len(col.header), len(str(col.datatype.__name__)), len(str(False))] + [len(str(v)) for v in col[slc]])

        def adjust(v, length):
            if v is None:
                return str(blanks).ljust(length)
            elif isinstance(v, str):
                return v.ljust(length)
            else:
                return str(v).rjust(length)
        rows = 0
        print("+", "+".join(["=" * c_lens[h] for h in headers]), "+", sep="")
        print("|", "|".join([h.center(c_lens[h], " ") for h in headers]), "|", sep="")
        print("|", "|".join([self.columns[h].datatype.__name__.center(c_lens[h], " ") for h in headers]), "|", sep="")
        print("|", "|".join([str(self.columns[h].allow_empty).center(c_lens[h], " ") for h in headers]), "|", sep="")
        print("+", "+".join(["-" * c_lens[h] for h in headers]), "+", sep="")
        for row in self.filter(*tuple(headers) + (slc,)):
            print("|", "|".join([adjust(v, c_lens[h]) for v, h in zip(row, headers)]), "|", sep="")
            rows += 1
        print("+", "+".join(["=" * c_lens[h] for h in headers]), "+", sep="")

        if row_count:
            if rows != len(self):
                print(f"(showing {rows} of {len(self)} rows)")
            elif len(self) > 0:
                print(f"showing all {len(self)} rows")
            else:
                print("no rows")
        if metadata:
            print("metadata:")
            for k, v in self.metadata.items():
                print("  ", k, v)

    def copy(self):
        return self.__copy__()

    def to_json(self):
        return json.dumps({
            'metadata': self.metadata,
            'columns': [c.to_json() for c in self.columns.values()]
        })

    @classmethod
    def from_json(cls, json_):
        t = Table()
        data = json.loads(json_)
        t.metadata = data['metadata']
        for c in data['columns']:
            if cls.new_tables_use_disk:
                col = StoredColumn.from_json(c)
            else:
                col = InMemoryColumn.from_json(c)
            col.header = t.check_for_duplicate_header(col.header)
            t.columns[col.header] = col
        return t

    @classmethod
    def from_file(cls, path, **kwargs):
        """ reads path and returns 1 or more tables.
        Use `list(Table.from_file(...))` to obtain all tables """
        for table in file_reader(path, **kwargs):
            yield table

    def copy_columns_only(self):
        """creates a new table with metadata but without the records"""
        t = Table()
        for col in self.columns.values():
            t.add_column(col.header, col.datatype, col.allow_empty, data=[])
        t.metadata = self.metadata.copy()
        return t

    def check_for_duplicate_header(self, header):
        """ Helper used to detect duplicate headers.
        :return valid header name
        """
        assert isinstance(header, str)
        if not header:
            header = 'None'
        new_header = header
        counter = count(start=1)
        while new_header in self.columns:
            new_header = f"{header}_{next(counter)}"  # valid attr names must be ascii.
        return new_header

    def rename_column(self, header, new_name):
        """
        :param header: current header name
        :param new_name: new name
        :return: None.
        """
        if new_name != self.check_for_duplicate_header(new_name):
            raise ValueError(f"header name {new_name} is already in use.")

        order = list(self.columns)
        d = {}
        for name in order:
            if name == header:
                d[new_name] = self.columns[name]
            else:
                d[name] = self.columns[name]
        self.columns = d

    def add_column(self, header, datatype, allow_empty=False, data=None):
        """
        :param header: str name of column
        :param datatype: from: int, str, float, bool, date, datetime, time
        :param allow_empty: bool
        :param data: list of values of given datatype.
        """
        assert isinstance(header, str)
        header = self.check_for_duplicate_header(header)
        if self._use_disk is False:
            self.columns[header] = InMemoryColumn(header, datatype, allow_empty, data=data)
        else:
            self.columns[header] = StoredColumn(header, datatype, allow_empty, data=data)

    def add_row(self, *args, **kwargs):
        """ Adds row(s) to the tablite.
        :param args: see below
        :param kwargs: see below
        :return: None

        Example:

            t = Table()
            t.add_column('A', int)
            t.add_column('B', int)
            t.add_column('C', int)

        The following examples are all valid and append the row (1,2,3) to the tablite.

            t.add_row(1,2,3)
            t.add_row([1,2,3])
            t.add_row((1,2,3))
            t.add_row(*(1,2,3))
            t.add_row(A=1, B=2, C=3)
            t.add_row(**{'A':1, 'B':2, 'C':3})

        The following examples add two rows to the tablite

            t.add_row((1,2,3), (4,5,6))
            t.add_row([1,2,3], [4,5,6])
            t.add_row({'A':1, 'B':2, 'C':3}, {'A':4, 'B':5, 'C':6}) # two (or more) dicts as args.
            t.add_row([{'A':1, 'B':2, 'C':3}, {'A':1, 'B':2, 'C':3}]) # list of dicts.

        """
        if args:
            if not any(isinstance(i, (list, tuple, dict)) for i in args):
                if len(args) == len(self.columns):
                    args = (args, )
                elif len(args) < len(self.columns):
                    raise TypeError(f"{args} doesn't match the number of columns. Are values missing?")
                elif len(args) > len(self.columns):
                    raise TypeError(f"{args} doesn't match the number of columns. Too many values?")
                else:
                    raise TypeError(f"{args} doesn't match the format of the tablite.")

            for arg in args:
                if len(arg) != len(self.columns):
                    raise ValueError(f"expected {len(self.columns)} columns, not {len(arg)}: {arg}")

                if isinstance(arg, (list, tuple)):
                    for value, col in zip(arg, self.columns.values()):
                        col.append(value)

                elif isinstance(arg, dict):
                    for k, value in arg.items():
                        col = self.columns.get(k, None)
                        if col is None:
                            raise ValueError(f"column {k} unknown: {list(self.columns)}")
                        assert isinstance(col, (InMemoryColumn, StoredColumn))
                        col.append(value)
                else:
                    raise TypeError(f"no handler for {type(arg)}s: {arg}")

        if kwargs:
            if len(kwargs) < len(self.columns):
                missing = [k for k in kwargs if k not in self.columns]
                raise ValueError(f"expected {len(self.columns)} columns, not {len(kwargs)}: Missing columns: {missing}")
            elif len(kwargs) > len(self.columns):
                excess = [k for k in kwargs if k not in self.columns]
                raise ValueError(f"expected {len(self.columns)} columns, not {len(kwargs)}: Excess columns: {excess}")
            else:
                pass  # looks alright.

            for k, value in kwargs.items():
                col = self.columns.get(k, None)
                if col is None:
                    raise ValueError(f"column {k} unknown: {list(self.columns)}")
                assert isinstance(col, (InMemoryColumn, StoredColumn))
                col.append(value)
            return

    def __contains__(self, item):
        return item in self.columns

    def __iter__(self):
        raise AttributeError("use Table.rows or Table.columns")

    def _slice(self, item=None):
        """ transforms a slice into start,stop,step"""
        if not item:
            item = slice(None, len(self), None)
        else:
            assert isinstance(item, slice)

        if item.stop < 0:
            start = len(self) + item.stop
            stop = len(self)
            step = 1 if item.step is None else item.step
        else:
            start = 0 if item.start is None else item.start
            stop = item.stop
            step = 1 if item.step is None else item.step
        return start, stop, step

    def __getitem__(self, item):
        """ returns rows as a tuple """
        if isinstance(item, int):
            item = slice(item, item + 1, 1)
        if isinstance(item, slice):
            t = Table(use_disk=self._use_disk)
            for col in self.columns.values():
                t.add_column(col.header, col.datatype, col.allow_empty, col[item])
            return t
        else:
            return self.columns[item]

    def __setitem__(self, key, value):
        if key in self.columns and isinstance(value, list):
            c = self.columns[key]
            c.clear()
            for v in value:
                c.append(v)
        else:
            raise TypeError(f"Use add_column to add_column: {key}")

    def __delitem__(self, key):
        """ delete column as key """
        if key in self.columns:
            del self.columns[key]
        else:
            raise KeyError(f"key not found")

    def __setattr__(self, name, value):
        if isinstance(name, str) and hasattr(self, name):
            if name in self.columns and isinstance(value, list):
                col = self.columns[name]
                col.replace(value)
                return
        super().__setattr__(name, value)

    def compare(self, other):
        """ compares the metadata of two tables."""
        if not isinstance(other, Table):
            a, b = self.__class__.__name__, other.__class__.__name__
            raise TypeError(f"cannot compare type {b} with {a}")

        if self.metadata != other.metadata:
            raise ValueError("tables have different metadata.")
        for a, b in [[self, other], [other, self]]:  # check both dictionaries.
            for name, col in a.columns.items():
                if name not in b.columns:
                    raise ValueError(f"Column {name} not in other")
                col2 = b.columns[name]
                if col.datatype != col2.datatype:
                    raise ValueError(f"Column {name}.datatype different: {col.datatype}, {col2.datatype}")
                if col.allow_empty != col2.allow_empty:
                    raise ValueError(f"Column {name}.allow_empty is different")
        return True

    def __iadd__(self, other):
        """ enables Table_1 += Table_2 """
        self.compare(other)
        for h, col in self.columns.items():
            c2 = other.columns[h]
            col.extend(c2[:])
        return self

    def __add__(self, other):
        """ enables Table_3 = Table_1 + Table_2 """
        self.compare(other)
        cp = self.copy()
        for h, col in cp.columns.items():
            c2 = other.columns[h]
            col.extend(c2[:])
        return cp

    @property
    def rows(self):
        """ enables iteration

        for row in tablite.rows:
            print(row)

        """
        for ix in range(len(self)):
            yield tuple(c[ix] if ix < len(c) else None for c in self.columns.values())

    def index(self, *args):
        """ Creates index on *args columns as d[(key tuple, )] = {index1, index2, ...} """
        idx = defaultdict(set)
        for ix, key in enumerate(self.filter(*args)):
            idx[key].add(ix)
        return idx

    def _sort_index(self, **kwargs):
        """ Helper for methods `sort` and `is_sorted` """
        if not isinstance(kwargs, dict):
            raise ValueError("Expected keyword arguments")
        if not kwargs:
            kwargs = {c: False for c in self.columns}

        for k, v in kwargs.items():
            if k not in self.columns:
                raise ValueError(f"no column {k}")
            if not isinstance(v, bool):
                raise ValueError(f"{k} was mapped to {v} - a non-boolean")
        none_substitute = float('-inf')

        rank = {i: tuple() for i in range(len(self))}
        for key in kwargs:
            unique_values = {v: 0 for v in self.columns[key] if v is not None}
            for r, v in enumerate(sorted(unique_values, reverse=kwargs[key])):
                unique_values[v] = r
            for ix, v in enumerate(self.columns[key]):
                rank[ix] += (unique_values.get(v, none_substitute),)

        new_order = [(r, i) for i, r in rank.items()]  # tuples are listed and sort...
        new_order.sort()
        sorted_index = [i for r, i in new_order]  # new index is extracted.

        rank.clear()  # free memory.
        new_order.clear()

        return sorted_index

    def sort(self, **kwargs):
        """ Perform multi-pass sorting with precedence given order of column names.
        :param kwargs: keys: columns, values: 'reverse' as boolean.
        """
        sorted_index = self._sort_index(**kwargs)
        for col_name, col in self.columns.items():
            assert isinstance(col, (StoredColumn, InMemoryColumn))
            col.replace(values=[col[ix] for ix in sorted_index])

    def is_sorted(self, **kwargs):
        """ Performs multi-pass sorting check with precedence given order of column names.
        :return bool
        """
        sorted_index = self._sort_index(**kwargs)
        if any(ix != i for ix, i in enumerate(sorted_index)):
            return False
        return True

    def filter(self, *items):
        """ enables iteration on a limited number of headers:

        >>> tablite.columns
        'a','b','c','d','e'

        for row in tablite.filter('b', 'a', 'a', 'c'):
            b,a,a,c = row ...

        returns values in same order as headers. """
        if any(not isinstance(i, (str, slice)) for i in items):
            raise SyntaxError(f"unexpected input: {[not isinstance(i, (str, slice)) for i in items]}")

        slices = [i for i in items if isinstance(i, slice)]
        if len(slices) > 2:
            raise SyntaxError("1 > slices")

        if not slices:
            slc = slice(None, len(self), None)
        else:
            slc = slices[0]
        assert isinstance(slc, slice)

        headers = [i for i in items if isinstance(i, str)]
        if any(h not in self.columns for h in headers):
            raise ValueError(f"column not found: {[h for h in headers if h not in self.columns]}")

        start, stop, step = DataTypes.infer_range_from_slice(slc, len(self))
        if step > 0 and start > stop:  # this wont work for range.
            return
        if step < 0 and start < stop:  # this wont work for range.
            return

        L = [self.columns[h] for h in headers]
        for ix in range(start, stop, step):
            item = tuple(c[ix] if ix < len(c) else None for c in L)
            yield item

    def all(self, **kwargs):
        """
        returns Table for rows where ALL kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you remember to add the ** in front of your dict?")
        if not all(k in self.columns for k in kwargs):
            raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in self.columns]}")

        ixs = None
        for k, v in kwargs.items():
            col = self.columns[k]
            if ixs is None:  # first header.
                if callable(v):
                    ix2 = {ix for ix, i in enumerate(col) if v(i)}
                else:
                    ix2 = {ix for ix, i in enumerate(col) if v == i}

            else:  # remaining headers.
                if callable(v):
                    ix2 = {ix for ix in ixs if v(col[ix])}
                else:
                    ix2 = {ix for ix in ixs if v == col[ix]}

            if not isinstance(ixs, set):
                ixs = ix2
            else:
                ixs = ixs.intersection(ix2)

            if not ixs:  # There are no matches.
                break

        t = Table(use_disk=self._use_disk)
        for col in self.columns.values():
            t.add_column(col.header, col.datatype, col.allow_empty, data=[col[ix] for ix in ixs])
        return t

    def any(self, **kwargs):
        """
        returns Table for rows where ANY kwargs match
        :param kwargs: dictionary with headers and values / boolean callable
        """
        if not isinstance(kwargs, dict):
            raise TypeError("did you remember to add the ** in front of your dict?")

        ixs = set()
        for k, v in kwargs.items():
            col = self.columns[k]
            if callable(v):
                ix2 = {ix for ix, r in enumerate(col) if v(r)}
            else:
                ix2 = {ix for ix, r in enumerate(col) if v == r}
            ixs.update(ix2)

        t = Table(use_disk=self._use_disk)
        for col in self.columns.values():
            t.add_column(col.header, col.datatype, col.allow_empty, data=[col[ix] for ix in ixs])
        return t

    def _join_type_check(self, other, left_keys, right_keys, columns):
        if not isinstance(other, Table):
            raise TypeError(f"other expected other to be type Table, not {type(other)}")

        if len(left_keys) != len(right_keys):
            raise ValueError(f"Keys do not have same length: \n{left_keys}, \n{right_keys}")

        if not isinstance(left_keys, list) and all(isinstance(k, str) for k in left_keys):
            raise TypeError(f"Expected keys as list of strings, not {type(left_keys)}")
        if not isinstance(right_keys, list) and all(isinstance(k, str) for k in right_keys):
            raise TypeError(f"Expected keys as list of strings, not {type(right_keys)}")

        if any(key not in self.columns for key in left_keys):
            raise ValueError(f"left key(s) not found: {[k for k in left_keys if k not in self.columns]}")
        if any(key not in other.columns for key in right_keys):
            raise ValueError(f"right key(s) not found: {[k for k in right_keys if k not in other.columns]}")
        for L, R in zip(left_keys, right_keys):
            Lcol, Rcol = self.columns[L], other.columns[R]
            if Lcol.datatype != Rcol.datatype:
                raise TypeError(f"{L} is {Lcol.datatype}, but {R} is {Rcol.datatype}")

        if columns is None:
            pass
        else:
            union = list(self.columns) + list(other.columns)
            if any(column not in union for column in columns):
                raise ValueError(f"Column not found: {[column for column in columns if column not in union]}")

    def left_join(self, other, left_keys, right_keys, columns=None):
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param columns: list of columns to retain, if None, all are retained.
        :return: new Table

        Example:
        SQL:   SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
        Tablite: left_join = numbers.left_join(letters, left_keys=['colour'], right_keys=['color'], columns=['number', 'letter'])
        """
        if columns is None:
            columns = list(self.columns) + list(other.columns)

        self._join_type_check(other, left_keys, right_keys, columns)  # raises if error

        left_join = Table(use_disk=self._use_disk)
        for col_name in columns:
            if col_name in self.columns:
                col = self.columns[col_name]
            elif col_name in other.columns:
                col = other.columns[col_name]
            else:
                raise ValueError(f"column name '{col_name}' not in any table.")
            left_join.add_column(col_name, col.datatype, allow_empty=True)

        left_ixs = range(len(self))
        right_idx = other.index(*right_keys)

        for left_ix in left_ixs:
            key = tuple(self[h][left_ix] for h in left_keys)
            right_ixs = right_idx.get(key, (None,))
            for right_ix in right_ixs:
                for col_name, column in left_join.columns.items():
                    if col_name in self:
                        column.append(self[col_name][left_ix])
                    elif col_name in other:
                        if right_ix is not None:
                            column.append(other[col_name][right_ix])
                        else:
                            column.append(None)
                    else:
                        raise Exception('bad logic')
        return left_join

    def inner_join(self, other, left_keys, right_keys, columns=None):
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param columns: list of columns to retain, if None, all are retained.
        :return: new Table

        Example:
        SQL:   SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
        Tablite: inner_join = numbers.inner_join(letters, left_keys=['colour'], right_keys=['color'], columns=['number', 'letter'])
        """
        if columns is None:
            columns = list(self.columns) + list(other.columns)

        self._join_type_check(other, left_keys, right_keys, columns)  # raises if error

        inner_join = Table(use_disk=self._use_disk)
        for col_name in columns:
            if col_name in self.columns:
                col = self.columns[col_name]
            elif col_name in other.columns:
                col = other.columns[col_name]
            else:
                raise ValueError(f"column name '{col_name}' not in any table.")
            inner_join.add_column(col_name, col.datatype, allow_empty=True)

        key_union = set(self.filter(*left_keys)).intersection(set(other.filter(*right_keys)))

        left_ixs = self.index(*left_keys)
        right_ixs = other.index(*right_keys)

        for key in sorted(key_union):
            for left_ix in left_ixs.get(key, set()):
                for right_ix in right_ixs.get(key, set()):
                    for col_name, column in inner_join.columns.items():
                        if col_name in self:
                            column.append(self[col_name][left_ix])
                        elif col_name in other:
                            column.append(other[col_name][right_ix])
                        else:
                            raise ValueError(f"column {col_name} not found. Duplicate names?")
        return inner_join

    def outer_join(self, other, left_keys, right_keys, columns=None):
        """
        :param other: self, other = (left, right)
        :param left_keys: list of keys for the join
        :param right_keys: list of keys for the join
        :param columns: list of columns to retain, if None, all are retained.
        :return: new Table

        Example:
        SQL:   SELECT number, letter FROM numbers OUTER JOIN letters ON numbers.colour == letters.color
        Tablite: outer_join = numbers.outer_join(letters, left_keys=['colour'], right_keys=['color'], columns=['number', 'letter'])
        """
        if columns is None:
            columns = list(self.columns) + list(other.columns)

        self._join_type_check(other, left_keys, right_keys, columns)  # raises if error

        outer_join = Table(use_disk=self._use_disk)
        for col_name in columns:
            if col_name in self.columns:
                col = self.columns[col_name]
            elif col_name in other.columns:
                col = other.columns[col_name]
            else:
                raise ValueError(f"column name '{col_name}' not in any table.")
            outer_join.add_column(col_name, col.datatype, allow_empty=True)

        left_ixs = range(len(self))
        right_idx = other.index(*right_keys)
        right_keyset = set(right_idx)

        for left_ix in left_ixs:
            key = tuple(self[h][left_ix] for h in left_keys)
            right_ixs = right_idx.get(key, (None,))
            right_keyset.discard(key)
            for right_ix in right_ixs:
                for col_name, column in outer_join.columns.items():
                    if col_name in self:
                        column.append(self[col_name][left_ix])
                    elif col_name in other:
                        if right_ix is not None:
                            column.append(other[col_name][right_ix])
                        else:
                            column.append(None)
                    else:
                        raise Exception('bad logic')

        for right_key in right_keyset:
            for right_ix in right_idx[right_key]:
                for col_name, column in outer_join.columns.items():
                    if col_name in self:
                        column.append(None)
                    elif col_name in other:
                        column.append(other[col_name][right_ix])
                    else:
                        raise Exception('bad logic')
        return outer_join

    def groupby(self, keys, functions):
        """
        :param keys: headers for grouping
        :param functions: list of headers and functions.
        :return: GroupBy class

        Example usage:
            from tablite import Table

            t = Table()
            t.add_column('date', int, allow_empty=False, data=[1,1,1,2,2,2])
            t.add_column('sku', int, allow_empty=False, data=[1,2,3,1,2,3])
            t.add_column('qty', int, allow_empty=False, data=[4,5,4,5,3,7])

            from tablite import GroupBy, Sum

            g = t.groupby(keys=['sku'], functions=[('qty', Sum)])
            g.tablite.show()

        """
        g = GroupBy(keys=keys, functions=functions)
        g += self
        return g

    def lookup(self, other, *criteria):
        """ function for looking up values in other according to criteria
        :param: other: Table
        :param: criteria: Each criteria must be a tuple with value comparisons in the form:
            (LEFT, OPERATOR, RIGHT)

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
        assert isinstance(self, Table)
        assert isinstance(other, Table)
        ops = {
            "in": operator.contains,
            "not in": lambda left, right: not left.contains(right),
            "<": operator.lt,
            "<=": operator.le,
            ">": operator.gt,
            ">=": operator.ge,
            "!=": operator.ne,
            "==": operator.eq,
        }

        table3 = Table(use_disk=self._use_disk)
        for name, col in chain(self.columns.items(), other.columns.items()):
            table3.add_column(name, col.datatype, allow_empty=True)

        functions = []
        columns = set(chain(self.columns, other.columns))
        columns_used = set()
        for left, op, right in criteria:
            if left in columns: columns_used.add(left)
            if right in columns: columns_used.add(right)
            if not callable(op):
                op = ops.get(op, None)
                if not callable(op):
                    raise ValueError(f"{op} not a recognised operator for comparison.")

            functions.append(lambda L, R: op(L.get(left, left), R.get(right, right)))
            # The lambda function above does a neat trick:
            # as L is a dict, L.get(left, L) will return a value
            # from the columns IF left is a column name. If it isn't
            # the function will treat left as a value.
            # The same applies to right.

        lru_cache = {}
        empty_row = tuple(None for _ in other.columns)

        for row1 in self.rows:
            row1_tup = tuple(v for v, name in zip(row1, columns_used) if name in self.columns)
            row1d = {name: value for name, value in zip(self.columns, row1) if name in columns_used}

            match_found = True if row1_tup in lru_cache else False

            if not match_found:  # search.
                for row2 in other.rows:
                    row2d = {name: value for name, value in zip(other.columns, row2) if name in columns_used}

                    if all(f(row1d, row2d) for f in functions):  # match found!
                        lru_cache[row1_tup] = row2
                        match_found = True
                        break

            if not match_found:  # no match found.
                lru_cache[row1_tup] = empty_row

            new_row = row1 + lru_cache[row1_tup]

            table3.add_row(new_row)

        return table3


class GroupBy(object):
    max = Max  # shortcuts to avoid having to type a long list of imports.
    min = Min
    sum = Sum
    first = First
    last = Last
    count = Count
    count_unique = CountUnique
    avg = Average
    stdev = StandardDeviation
    median = Median
    mode = Mode

    _functions = [
        Max, Min, Sum, First, Last,
        Count, CountUnique,
        Average, StandardDeviation, Median, Mode
    ]
    _function_names = {f.__name__: f for f in _functions}

    def __init__(self, keys, functions):
        """
        :param keys: headers for grouping
        :param functions: list of headers and functions.
        :return: None.

        Example usage:
        --------------------
        from tablite import Table

        t = Table()
        t.add_column('date', int, allow_empty=False, data=[1,1,1,2,2,2])
        t.add_column('sku', int, allow_empty=False, data=[1,2,3,1,2,3])
        t.add_column('qty', int, allow_empty=False, data=[4,5,4,5,3,7])

        from tablite import GroupBy, Sum

        g = GroupBy(keys=['sku'], functions=[('qty', Sum)])
        g += t
        g.tablite.show()

        """
        if not isinstance(keys, list):
            raise TypeError(f"Expected keys as a list of header names, not {type(keys)}")

        if len(set(keys)) != len(keys):
            duplicates = [k for k in keys if keys.count(k) > 1]
            s = "" if len(duplicates) > 1 else "s"
            raise ValueError(f"duplicate key{s} found: {duplicates}")

        self.keys = keys

        if not isinstance(functions, list):
            raise TypeError(f"Expected functions to be a list of tuples. Got {type(functions)}")

        if not all(len(i) == 2 for i in functions):
            raise ValueError(f"Expected each tuple in functions to be of length 2. \nGot {functions}")

        if not all(isinstance(a, str) for a, b in functions):
            L = [(a, type(a)) for a, b in functions if not isinstance(a, str)]
            raise ValueError(f"Expected header names in functions to be strings. Found: {L}")

        if not all(issubclass(b, GroupbyFunction) and b in GroupBy._functions for a, b in functions):
            L = [b for a, b in functions if b not in GroupBy._functions]
            if len(L) == 1:
                singular = f"function {L[0]} is not in GroupBy.functions"
                raise ValueError(singular)
            else:
                plural = f"the functions {L} are not in GroupBy.functions"
                raise ValueError(plural)

        self.groupby_functions = functions  # list with header name and function name

        self._output = None   # class Table.
        self._required_headers = None  # headers for reading input.
        self.aggregation_functions = defaultdict(list)  # key: [list of groupby functions]
        self._function_classes = []  # initiated functions.

        # Order is preserved so that this is doable:
        # for header, function, function_instances in zip(self.groupby_functions, self.function_classes) ....

    def _setup(self, table):
        """ helper to setup the group functions """
        self._output = Table()
        self._required_headers = self.keys + [h for h, fn in self.groupby_functions]

        for h in self.keys:
            col = table[h]
            self._output.add_column(header=h, datatype=col.datatype, allow_empty=True)  # add column for keys

        self._function_classes = []
        for h, fn in self.groupby_functions:
            col = table[h]
            assert isinstance(col, (StoredColumn, InMemoryColumn))
            f_instance = fn(col.datatype)
            assert isinstance(f_instance, GroupbyFunction)
            self._function_classes.append(f_instance)

            function_name = f"{fn.__name__}({h})"
            self._output.add_column(header=function_name, datatype=f_instance.datatype, allow_empty=True)  # add column for fn's.

    def __iadd__(self, other):
        """
        To view results use `for row in self.rows`
        To add more data use self += new data (Table)
        """
        assert isinstance(other, Table)
        if self._output is None:
            self._setup(other)
        else:
            self._output.compare(other)  # this will raise if there are problems

        for row in other.filter(*self._required_headers):
            d = {h: v for h, v in zip(self._required_headers, row)}
            key = tuple([d[k] for k in self.keys])
            functions = self.aggregation_functions.get(key)
            if not functions:
                functions = [fn.__class__(fn.datatype) for fn in self._function_classes]
                self.aggregation_functions[key] = functions

            for (h, fn), f in zip(self.groupby_functions, functions):
                f.update(d[h])
        return self

    def _generate_table(self):
        """ helper that generates the result for .tablite and .rows """
        for key, functions in self.aggregation_functions.items():
            row = key + tuple(fn.value for fn in functions)
            self._output.add_row(row)
        self.aggregation_functions.clear()  # hereby we only create the tablite once.
        self._output.sort(**{k: False for k in self.keys})

    @property
    def table(self):
        """ returns Table """
        if self._output is None:
            return None

        if self.aggregation_functions:
            self._generate_table()

        assert isinstance(self._output, Table)
        return self._output

    @property
    def rows(self):
        """ returns iterator for Groupby.rows """
        if self._output is None:
            return None

        if self.aggregation_functions:
            self._generate_table()

        assert isinstance(self._output, Table)
        for row in self._output.rows:
            yield row

    def pivot(self, *args):
        """ pivots the groupby so that `columns` become new columns.

        :param args: column names
        :return: New Table

        Example:
        t = Table()
        t.add_column('A', int, data=[1, 1, 2, 2, 3, 3] * 2)
        t.add_column('B', int, data=[1, 2, 3, 4, 5, 6] * 2)
        t.add_column('C', int, data=[6, 5, 4, 3, 2, 1] * 2)

        t.show()
        +=====+=====+=====+
        |  A  |  B  |  C  |
        | int | int | int |
        |False|False|False|
        +-----+-----+-----+
        |    1|    1|    6|
        |    1|    2|    5|
        |    2|    3|    4|
        |    2|    4|    3|
        |    3|    5|    2|
        |    3|    6|    1|
        |    1|    1|    6|
        |    1|    2|    5|
        |    2|    3|    4|
        |    2|    4|    3|
        |    3|    5|    2|
        |    3|    6|    1|
        +=====+=====+=====+

        g = t.groupby(keys=['A', 'C'], functions=[('B', Sum)])

        t2 = g.pivot('A')

        t2.show()
        +=====+==========+==========+==========+
        |  C  |Sum(B,A=1)|Sum(B,A=2)|Sum(B,A=3)|
        | int |   int    |   int    |   int    |
        |False|   True   |   True   |   True   |
        +-----+----------+----------+----------+
        |    5|         4|      None|      None|
        |    6|         2|      None|      None|
        |    3|      None|         8|      None|
        |    4|      None|         6|      None|
        |    1|      None|      None|        12|
        |    2|      None|      None|        10|
        +=====+==========+==========+==========+
        """
        columns = args
        if not all(isinstance(i, str) for i in args):
            raise TypeError(f"column name not str: {[i for i in columns if not isinstance(i,str)]}")

        if self._output is None:
            return None

        if self.aggregation_functions:
            self._generate_table()

        assert isinstance(self._output, Table)
        if any(i not in self._output.columns for i in columns):
            raise ValueError(f"column not found in groupby: {[i not in self._output.columns for i in columns]}")

        sort_order = {k: False for k in self.keys}
        if not self._output.is_sorted(**sort_order):
            self._output.sort(**sort_order)

        t = Table()
        for col_name, col in self._output.columns.items():  # add vertical groups.
            if col_name in self.keys and col_name not in columns:
                t.add_column(col_name, col.datatype, allow_empty=False)

        tup_length = 0
        for column_key in self._output.filter(*columns):  # add horizontal groups.
            col_name = ",".join(f"{h}={v}" for h, v in zip(columns, column_key))  # expressed "a=0,b=3" in column name "Sum(g, a=0,b=3)"

            for (header, function), function_instances in zip(self.groupby_functions, self._function_classes):
                new_column_name = f"{function.__name__}({header},{col_name})"
                if new_column_name not in t.columns:  # it's could be duplicate key value.
                    t.add_column(new_column_name, datatype=function_instances.datatype, allow_empty=True)
                    tup_length += 1
                else:
                    pass  # it's a duplicate.

        # add rows.
        key_index = {k: i for i, k in enumerate(self._output.columns)}
        old_v_keys = tuple(None for k in self.keys if k not in columns)

        for row in self._output.rows:
            v_keys = tuple(row[key_index[k]] for k in self.keys if k not in columns)
            if v_keys != old_v_keys:
                t.add_row(v_keys + tuple(None for i in range(tup_length)))
                old_v_keys = v_keys

            function_values = [v for h, v in zip(self._output.columns, row) if h not in self.keys]

            col_name = ",".join(f"{h}={row[key_index[h]]}" for h in columns)
            for (header, function), fi in zip(self.groupby_functions, function_values):
                column_key = f"{function.__name__}({header},{col_name})"
                t[column_key][-1] = fi

        return t


def text_reader(path, split_sequence=None, sep=None, has_headers=True):
    """ txt, tab & csv reader """
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")

    # detect newline format
    windows = '\n'
    unix = '\r\n'

    encoding = detect_encoding(path)  # detect encoding

    if split_sequence is None and sep is None:  #
        sep = detect_seperator(path, encoding)

    t = Table()
    t.metadata['filename'] = path.name
    n_columns = None
    with path.open('r', encoding=encoding) as fi:
        for line in fi:
            end = windows if line.endswith(windows) else unix
            # this is more robust if the file was concatenated by a non-programmer, than doing it once only.

            line = line.rstrip(end)
            line = line.lstrip('\ufeff')  # utf-8-sig byte order mark.

            if split_sequence:
                values = split_by_sequence(line, split_sequence)
            elif line.count('"') >= 2 or line.count("'") >= 2:
                values = text_escape(line, sep=sep)
            else:
                values = tuple((i.lstrip().rstrip() for i in line.split(sep)))

            if not t.columns:
                for idx, v in enumerate(values, 1):
                    if not has_headers:
                        t.add_column(f"_{idx}", datatype=str, allow_empty=True)
                    else:
                        header = v.rstrip(" ").lstrip(" ")
                        t.add_column(header, datatype=str, allow_empty=True)
                n_columns = len(values)

                if not has_headers:  # first line is our first row
                    t.add_row(values)
            else:
                while n_columns > len(values):  # this makes the reader more robust.
                    values += ('',)
                t.add_row(values)
    yield t


def excel_reader(path, has_headers=True, sheet_names=None):
    """  returns Table(s) from excel path """
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")
    book = pyexcel.get_book(file_name=str(path))

    # import all sheets or a subset
    sheets = [s for s in book if sheet_names is None or s.name in sheet_names]

    for sheet in sheets:
        if len(sheet) == 0:
            continue

        t = Table()
        t.metadata['sheet_name'] = sheet.name
        t.metadata['filename'] = path.name
        for idx, column in enumerate(sheet.columns(), 1):
            if has_headers:
                header, start_row_pos = str(column[0]), 1
            else:
                header, start_row_pos = f"_{idx}", 0

            dtypes = {type(v) for v in column[start_row_pos:]}
            allow_empty = True if None in dtypes else False
            dtypes.discard(None)

            if dtypes == {int, float}:
                dtypes.remove(int)

            if len(dtypes) == 1:
                dtype = dtypes.pop()
                data = [dtype(v) if not isinstance(v, dtype) else v for v in column[start_row_pos:]]
            else:
                dtype, data = str, [str(v) for v in column[start_row_pos:]]
            t.add_column(header, dtype, allow_empty, data)
        yield t


def ods_reader(path, has_headers=True):
    """  returns Table from .ODS """
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")
    sheets = pyexcel.get_book_dict(file_name=str(path))

    for sheet_name, data in sheets.items():
        if all((row == [] for row in data)):  # no data.
            continue
        for i in range(len(data)):  # remove empty lines at the end of the data.
            if "" == "".join(str(i) for i in data[-1]):
                data = data[:-1]
            else:
                break

        table = Table(filename=path.name)
        table.metadata['filename'] = path.name
        table.metadata['sheet_name'] = sheet_name

        for ix, value in enumerate(data[0]):
            if has_headers:
                header, start_row_pos = str(value), 1
            else:
                header, start_row_pos = f"_{ix + 1}", 0

            dtypes = set(type(row[ix]) for row in data[start_row_pos:] if len(row) > ix)
            allow_empty = None in dtypes
            dtypes.discard(None)
            if len(dtypes) == 1:
                dtype = dtypes.pop()
            elif dtypes == {float, int}:
                dtype = float
            else:
                dtype = str
            values = [dtype(row[ix]) for row in data[start_row_pos:] if len(row) > ix]
            table.add_column(header, dtype, allow_empty, data=values)
        yield table


def zip_reader(path):
    """ reads zip files and unpacks anything it can read."""
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")

    temp_dir_path = gettempdir()
    tempdir = Path(temp_dir_path)

    with zipfile.ZipFile(path, 'r') as zipf:

        for name in zipf.namelist():

            zipf.extract(name, temp_dir_path)

            p = tempdir / name
            try:
                tables = file_reader(p)
                for table in tables:
                    yield table
            except Exception as e:  # unknown file type.
                print(f'reading\n  {p}\nresulted in the error:')
                print(str(e))
                continue

            p.unlink()


def log_reader(path, has_headers=True):
    """ returns Table from log files (txt)"""
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")
    for line in path.open()[:10]:
        print(repr(line))
    print("please declare separators. Blank return means 'done'.")
    split_sequence = []
    while True:
        response = input(">")
        if response == "":
            break
        print("got", repr(response))
        split_sequence.append(response)
    table = text_reader(path, split_sequence=split_sequence, has_headers=has_headers)
    return table


def file_reader(path, **kwargs):
    """
    :param path: pathlib.Path object with extension as:
        .csv, .tsv, .txt, .xls, .xlsx, .xlsm, .ods, .zip, .log

        .zip is automatically flattened

    :param kwargs: dictionary options:
        'sep': False or single character
        'split_sequence': list of characters

    :return: generator of Tables.
        to get the tablite in one line.

        >>> list(file_reader(abc.csv)[0]

        use the following for Excel and Zips:
        >>> for tablite in file_reader(filename):
                ...
    """
    assert isinstance(path, Path)
    extension = path.name.split(".")[-1]
    if extension not in readers:
        raise TypeError(f"Filetype for {path.name} not recognised.")
    reader, default_kwargs = readers[extension]
    kwargs = {**default_kwargs, **kwargs}

    for table in reader(path, **kwargs):
        assert isinstance(table, Table), "programmer returned something else than a Table"
        find_format(table)
        yield table


def find_format(table):
    """ common function for harmonizing formats AFTER import. """
    assert isinstance(table, Table)

    for col_name, column in table.columns.items():
        assert isinstance(column, (StoredColumn, InMemoryColumn))
        column.allow_empty = any(v in DataTypes.nones for v in column)

        values = [v for v in column if v not in DataTypes.nones]
        assert isinstance(column, (StoredColumn, InMemoryColumn))
        values.sort()

        works = []
        if not values:
            works.append((0, DataTypes.str))
        else:
            for dtype in DataTypes.types:  # try all datatypes.
                last_value = None
                c = 0
                for v in values:
                    if v != last_value:  # no need to repeat duplicates.
                        try:
                            DataTypes.infer(v, dtype)  # handles None gracefully.
                        except (ValueError, TypeError):
                            break
                        last_value = v
                    c += 1

                works.append((c, dtype))
                if c == len(values):
                    break  # we have a complete match for the simplest
                    # data format for all values. No need to do more work.

        for c, dtype in works:
            if c == len(values):
                values.clear()
                if table.use_disk:
                    c2 = StoredColumn
                else:
                    c2 = InMemoryColumn

                new_column = c2(column.header, dtype, column.allow_empty)
                for v in column:
                    new_column.append(DataTypes.infer(v, dtype) if v not in DataTypes.nones else None)
                column.clear()
                table.columns[col_name] = new_column
                break


readers = {
    'csv': [text_reader, {}],
    'tsv': [text_reader, {}],
    'txt': [text_reader, {}],
    'xls': [excel_reader, {}],
    'xlsx': [excel_reader, {}],
    'xlsm': [excel_reader, {}],
    'ods': [ods_reader, {}],
    'zip': [zip_reader, {}],
    'log': [log_reader, {'sep': False}]
}
