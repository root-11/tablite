from collections import defaultdict

from tablite import Max, Min, Sum, First, Last, Count, CountUnique, Average, StandardDeviation, Median, Mode, \
    GroupbyFunction, Table, Column


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
        self.data = defaultdict(list)  # key: [list of groupby functions]
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
            assert isinstance(col, Column)
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
            functions = self.data.get(key)
            if not functions:
                functions = [fn.__class__(fn.datatype) for fn in self._function_classes]
                self.data[key] = functions

            for (h, fn), f in zip(self.groupby_functions, functions):
                f.update(d[h])
        return self

    def _generate_table(self):
        """ helper that generates the result for .tablite and .rows """
        for key, functions in self.data.items():
            row = key + tuple(fn.value for fn in functions)
            self._output.add_row(row)
        self.data.clear()  # hereby we only create the tablite once.
        self._output.sort(**{k: False for k in self.keys})

    @property
    def table(self):
        """ returns Table """
        if self._output is None:
            return None

        if self.data:
            self._generate_table()

        assert isinstance(self._output, Table)
        return self._output

    @property
    def rows(self):
        """ returns iterator for Groupby.rows """
        if self._output is None:
            return None

        if self.data:
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

        if self.data:
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