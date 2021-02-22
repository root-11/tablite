import tempfile
import zipfile
from pathlib import Path

import pyexcel

from tablite import Table, StoredColumn, Column, DataTypes


def split_by_sequence(text, sequence):
    """ helper to split text according to a split sequence. """
    chunks = tuple()
    for element in sequence:
        idx = text.find(element)
        if idx < 0:
            raise ValueError(f"'{element}' not in row")
        chunk, text = text[:idx], text[len(element) + idx:]
        chunks += (chunk,)
    chunks += (text,)  # the remaining text.
    return chunks


encodings = [
    'utf-32',
    'utf-16',
    'ascii',
    'utf-8',
    'windows-1252',
    'utf-7',
]


def detect_encoding(path):
    """ helper that automatically detects encoding from files. """
    assert isinstance(path, Path)
    for encoding in encodings:
        try:
            snippet = path.open('r', encoding=encoding).read(100)
            if snippet.startswith('ï»¿'):
                return 'utf-8-sig'
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            pass
    raise UnicodeDecodeError


def detect_seperator(path, encoding):
    """
    :param path: pathlib.Path objects
    :param encoding: file encoding.
    :return: 1 character.
    """
    # After reviewing the logic in the CSV sniffer, I concluded that all it
    # really does is to look for a non-text character. As the separator is
    # determined by the first line, which almost always is a line of headers,
    # the text characters will be utf-8,16 or ascii letters plus white space.
    # This leaves the characters ,;:| and \t as potential separators, with one
    # exception: files that use whitespace as separator. My logic is therefore
    # to (1) find the set of characters that intersect with ',;:|\t' which in
    # practice is a single character, unless (2) it is empty whereby it must
    # be whitespace.
    text = ""
    for line in path.open('r', encoding=encoding):  # pick the first line only.
        text = line
        break
    seps = {',', '\t', ';', ':', '|'}.intersection(text)
    if not seps:
        if " " in text:
            return " "
        else:
            raise ValueError("separator not detected")
    if len(seps) == 1:
        return seps.pop()
    else:
        frq = [(text.count(i), i) for i in seps]
        frq.sort(reverse=True)  # most frequent first.
        return frq[0][-1]


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
                    values += ('', )
                t.add_row(values)
    yield t


def text_escape(s, escape='"', sep=';'):
    """ escapes text marks using a depth measure. """
    assert isinstance(s, str)
    word, words = [], tuple()
    in_esc_seq = False
    for ix, c in enumerate(s):
        if c == escape:
            if in_esc_seq:
                if ix+1 != len(s) and s[ix + 1] != sep:
                    word.append(c)
                    continue  # it's a fake escape.
                in_esc_seq = False
            else:
                in_esc_seq = True
            if word:
                words += ("".join(word),)
                word.clear()
        elif c == sep and not in_esc_seq:
            if word:
                words += ("".join(word),)
                word.clear()
        else:
            word.append(c)

    if word:
        if word:
            words += ("".join(word),)
            word.clear()
    return words


def excel_reader(path, has_headers=True):
    """  returns Table(s) from excel path """
    if not isinstance(path, Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")
    sheets = pyexcel.get_book(file_name=str(path))

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
        if data == [[], []]:  # no data.
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
                header, start_row_pos = f"_{ix+1}", 0

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

    with tempfile.TemporaryDirectory() as temp_dir_path:
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
                    print(f'reading {p} resulted in the error:')
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


def find_format(table):
    """ common function for harmonizing formats AFTER import. """
    assert isinstance(table, Table)

    for col_name, column in table.columns.items():
        assert isinstance(column, (StoredColumn, Column))
        column.allow_empty = any(v in DataTypes.nones for v in column)

        values = [v for v in column if v not in DataTypes.nones]
        assert isinstance(column, (StoredColumn, Column))
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
                    c2 = Column

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