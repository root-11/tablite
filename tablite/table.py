import json
from collections import defaultdict
from itertools import count

import pyperclip

from tablite import StoredColumn, Column, windows_tempfile, DataTypes
from tablite.file_readers import file_reader
from tablite.groupby import GroupBy


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
            C = Column

        for col_name, column in self.columns.items():
            self.columns[col_name] = C(col_name, column.datatype, column.allow_empty, data=column)

    def __eq__(self, other):
        if not isinstance(other, Table):
            a, b = self.__class__.__name__, other.__class__.__name__
            raise TypeError(f"cannot compare {a} with {b}")
        if self.metadata != other.metadata:
            return False
        if any(a != b for a, b in zip(self.columns.values(), other.columns.values())):
            return False
        return True

    def __len__(self):
        """ returns length of longest column."""
        return max(len(c) for c in self.columns.values())

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
        tempfile = windows_tempfile(suffix='.csv')
        with open(tempfile, 'w') as fo:
            fo.writelines(pyperclip.paste())
        g = Table.from_file(tempfile)
        t = list(g)[0]
        del t.metadata['filename']
        return t

    def show(self, *items, blanks=None):
        """ shows the tablite.
        param: items: column names, slice.
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
            assert isinstance(col, (Column, StoredColumn))
            c_lens[h] = max(
                [len(col.header), len(str(col.datatype.__name__)), len(str(False))] + [len(str(v)) for v in col[slc]])

        def adjust(v, length):
            if v is None:
                return str(blanks).ljust(length)
            elif isinstance(v, str):
                return v.ljust(length)
            else:
                return str(v).rjust(length)

        print("+", "+".join(["=" * c_lens[h] for h in headers]), "+", sep="")
        print("|", "|".join([h.center(c_lens[h], " ") for h in headers]), "|", sep="")
        print("|", "|".join([self.columns[h].datatype.__name__.center(c_lens[h], " ") for h in headers]), "|", sep="")
        print("|", "|".join([str(self.columns[h].allow_empty).center(c_lens[h], " ") for h in headers]), "|", sep="")
        print("+", "+".join(["-" * c_lens[h] for h in headers]), "+", sep="")
        for row in self.filter(*tuple(headers) + (slc,)):
            print("|", "|".join([adjust(v, c_lens[h]) for v, h in zip(row, headers)]), "|", sep="")
        print("+", "+".join(["=" * c_lens[h] for h in headers]), "+", sep="")

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
            col = Column.from_json(c)
            col.header = t.check_for_duplicate_header(col.header)
            t.columns[col.header] = col
        return t

    @classmethod
    def from_file(cls, path, **kwargs):
        """ reads path and returns 1 or more tables.
        Use `list(Table.from_file(...))` to obtain all tables """
        for table in file_reader(path, **kwargs):
            yield table

    @staticmethod
    def copy_columns_only(table):
        """creates a new tablite without the data"""
        assert isinstance(table, Table)
        t = Table()
        for col in table.columns.values():
            t.add_column(col.header, col.datatype, col.allow_empty, data=[])
        t.metadata = table.metadata.copy()
        return t

    def check_for_duplicate_header(self, header):
        assert isinstance(header, str)
        if not header:
            header = 'None'
        new_header = header
        counter = count(start=1)
        while new_header in self.columns:
            new_header = f"{header}_{next(counter)}"  # valid attr names must be ascii.
        return new_header

    def add_column(self, header, datatype, allow_empty=False, data=None):
        assert isinstance(header, str)
        header = self.check_for_duplicate_header(header)
        if self._use_disk is False:
            self.columns[header] = Column(header, datatype, allow_empty, data=data)
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
                        assert isinstance(col, (Column, StoredColumn))
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
                assert isinstance(col, (Column, StoredColumn))
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
            item = tuple(c[ix] if ix < len(c) else None for c in self.columns.values())
            yield item

    def index(self, *args):
        """ Creates index on *args columns as d[(key tuple, )] = {index1, index2, ...} """
        idx = defaultdict(set)
        for ix, key in enumerate(self.filter(*args)):
            idx[key].add(ix)
        return idx

    def _sort_index(self, **kwargs):
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
            assert isinstance(col, (StoredColumn, Column))
            col.replace(values=[col[ix] for ix in sorted_index])

    def is_sorted(self, **kwargs):
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

        sss = DataTypes.infer_range_from_slice(slc, len(self))
        if sss is None:
            return

        L = [self.columns[h] for h in headers]
        for ix in range(*sss):
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
        :return: new tablite

        Example:
        SQL:   SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
        Table: left_join = numbers.left_join(letters, left_keys=['colour'], right_keys=['color'], columns=['number', 'letter'])
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
                raise ValueError(f"column name '{col_name}' not in any tablite.")
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
        :return: new tablite

        Example:
        SQL:   SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
        Table: inner_join = numbers.inner_join(letters, left_keys=['colour'], right_keys=['color'], columns=['number', 'letter'])
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
                raise ValueError(f"column name '{col_name}' not in any tablite.")
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
        :return: new tablite

        Example:
        SQL:   SELECT number, letter FROM numbers OUTER JOIN letters ON numbers.colour == letters.color
        Table: outer_join = numbers.outer_join(letters, left_keys=['colour'], right_keys=['color'], columns=['number', 'letter'])
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
                raise ValueError(f"column name '{col_name}' not in any tablite.")
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