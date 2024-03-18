from tablite.utils import sub_cls_check, type_check
from tablite.base import BaseTable
from tablite.config import Config
from tablite.datatypes import DataTypes
from pathlib import Path


from tqdm import tqdm as _tqdm


def to_sql(table, name):
    """
    generates ANSI-92 compliant SQL.

    args:
        name (str): name of SQL table.
    """
    sub_cls_check(table, BaseTable)
    type_check(name, str)

    prefix = name
    name = "T1"
    create_table = """CREATE TABLE {} ({})"""
    columns = []
    for name, col in table.columns.items():
        dtype = col.types()
        if len(dtype) == 1:
            dtype, _ = dtype.popitem()
            if dtype is int:
                dtype = "INTEGER"
            elif dtype is float:
                dtype = "REAL"
            else:
                dtype = "TEXT"
        else:
            dtype = "TEXT"
        definition = f"{name} {dtype}"
        columns.append(definition)

    create_table = create_table.format(prefix, ", ".join(columns))

    # return create_table
    row_inserts = []
    for row in table.rows:
        row_inserts.append(str(tuple([i if i is not None else "NULL" for i in row])))
    row_inserts = f"INSERT INTO {prefix} VALUES " + ",".join(row_inserts)
    return "begin; {}; {}; commit;".format(create_table, row_inserts)


def to_pandas(table):
    """
    returns pandas.DataFrame
    """
    sub_cls_check(table, BaseTable)
    try:
        return pd.DataFrame(table.to_dict())  # noqa
    except ImportError:
        import pandas as pd  # noqa
    return pd.DataFrame(table.to_dict())  # noqa


def to_hdf5(table, path):
    # fmt: off
    """
    creates a copy of the table as hdf5

    Note that some loss of type information is to be expected in columns of mixed type:
    >>> t.show(dtype=True)
    +===+===+=====+=====+====+=====+=====+===================+==========+========+===============+===+=========================+=====+===+
    | # | A |  B  |  C  | D  |  E  |  F  |         G         |    H     |   I    |       J       | K |            L            |  M  | O |
    |row|int|mixed|float|str |mixed| bool|      datetime     |   date   |  time  |   timedelta   |str|           int           |float|int|
    +---+---+-----+-----+----+-----+-----+-------------------+----------+--------+---------------+---+-------------------------+-----+---+
    | 0 | -1|None | -1.1|    |None |False|2023-06-09 09:12:06|2023-06-09|09:12:06| 1 day, 0:00:00|b  |-100000000000000000000000|  inf| 11|
    | 1 |  1|    1|  1.1|1000|1    | True|2023-06-09 09:12:06|2023-06-09|09:12:06|2 days, 0:06:40|嗨  | 100000000000000000000000| -inf|-11|
    +===+===+=====+=====+====+=====+=====+===================+==========+========+===============+===+=========================+=====+===+
    >>> t.to_hdf5(filename)
    >>> t2 = Table.from_hdf5(filename)
    >>> t2.show(dtype=True)
    +===+===+=====+=====+=====+=====+=====+===================+===================+========+===============+===+=========================+=====+===+
    | # | A |  B  |  C  |  D  |  E  |  F  |         G         |         H         |   I    |       J       | K |            L            |  M  | O |
    |row|int|mixed|float|mixed|mixed| bool|      datetime     |      datetime     |  time  |      str      |str|           int           |float|int|
    +---+---+-----+-----+-----+-----+-----+-------------------+-------------------+--------+---------------+---+-------------------------+-----+---+
    | 0 | -1|None | -1.1|None |None |False|2023-06-09 09:12:06|2023-06-09 00:00:00|09:12:06|1 day, 0:00:00 |b  |-100000000000000000000000|  inf| 11|
    | 1 |  1|    1|  1.1| 1000|    1| True|2023-06-09 09:12:06|2023-06-09 00:00:00|09:12:06|2 days, 0:06:40|嗨  | 100000000000000000000000| -inf|-11|
    +===+===+=====+=====+=====+=====+=====+===================+===================+========+===============+===+=========================+=====+===+
    """
    # fmt: in
    import h5py

    sub_cls_check(table, BaseTable)
    type_check(path, Path)

    total = f"{len(table.columns) * len(table):,}"  # noqa
    print(f"writing {total} records to {path}", end="")

    with h5py.File(path, "w") as f:
        n = 0
        for name, col in table.items():
            try:
                f.create_dataset(name, data=col[:])  # stored in hdf5 as '/name'
            except TypeError:
                f.create_dataset(name, data=[str(i) for i in col[:]])  # stored in hdf5 as '/name'
            n += 1
    print("... done")


def excel_writer(table, path):
    """
    writer for excel files.

    This can create xlsx files beyond Excels.
    If you're using pyexcel to read the data, you'll see the data is there.
    If you're using Excel, Excel will stop loading after 1,048,576 rows.

    See pyexcel for more details:
    http://docs.pyexcel.org/
    """
    import pyexcel

    sub_cls_check(table, BaseTable)
    type_check(path, Path)

    def gen(table):  # local helper
        yield table.columns
        for row in table.rows:
            yield row

    data = list(gen(table))
    if path.suffix in [".xls", ".ods"]:
        data = [
            [str(v) if (isinstance(v, (int, float)) and abs(v) > 2**32 - 1) else DataTypes.to_json(v) for v in row]
            for row in data
        ]

    pyexcel.save_as(array=data, dest_file_name=str(path))


def to_json(table, *args, **kwargs):
    import json

    sub_cls_check(table, BaseTable)
    return json.dumps(table.as_json_serializable())


def path_suffix_check(path, kind):
    if not path.suffix == kind:
        raise ValueError(f"Suffix mismatch: Expected {kind}, got {path.suffix} in {path.name}")
    if not path.parent.exists():
        raise FileNotFoundError(f"directory {path.parent} not found.")


def text_writer(table, path, tqdm=_tqdm):
    """exports table to csv, tsv or txt dependening on path suffix.
    follows the JSON norm. text escape is ON for all strings.

    Note:
    ----------------------
    If the delimiter is present in a string when the string is exported,
    text-escape is required, as the format otherwise is corrupted.
    When the file is being written, it is unknown whether any string in
    a column contrains the delimiter. As text escaping the few strings
    that may contain the delimiter would lead to an assymmetric format,
    the safer guess is to text escape all strings.
    """
    sub_cls_check(table, BaseTable)
    type_check(path, Path)

    def txt(value):  # helper for text writer
        if value is None:
            return ""  # A column with 1,None,2 must be "1,,2".
        elif isinstance(value, str):
            # if not (value.startswith('"') and value.endswith('"')):
            #     return f'"{value}"'  # this must be escape: "the quick fox, jumped over the comma"
            # else:
            return value  # this would for example be an empty string: ""
        else:
            return str(DataTypes.to_json(value))  # this handles datetimes, timedelta, etc.

    delimiters = {".csv": ",", ".tsv": "\t", ".txt": "|"}
    delimiter = delimiters.get(path.suffix)

    with path.open("w", encoding="utf-8") as fo:
        fo.write(delimiter.join(c for c in table.columns) + "\n")
        for row in tqdm(table.rows, total=len(table), disable=Config.TQDM_DISABLE):
            fo.write(delimiter.join(txt(c) for c in row) + "\n")


def sql_writer(table, path):
    type_check(table, BaseTable)
    type_check(path, Path)
    with path.open("w", encoding="utf-8") as fo:
        fo.write(to_sql(table))


def json_writer(table, path):
    type_check(table, BaseTable)
    type_check(path, Path)
    with path.open("w") as fo:
        fo.write(to_json(table))


def to_html(table, path):
    type_check(table, BaseTable)
    type_check(path, Path)
    with path.open("w", encoding="utf-8") as fo:
        fo.write(table._repr_html_(slice(0, len(table))))
