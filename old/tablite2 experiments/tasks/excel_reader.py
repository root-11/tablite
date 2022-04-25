import pathlib
import pyexcel 


def excel_reader(path, has_headers=True, sheet_name=None, **kwargs):
    """
    returns Table(s) from excel path
    """
    if not isinstance(path, pathlib.Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")
    book = pyexcel.get_book(file_name=str(path))

    if sheet_name is None:  # help the user.
        raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in book]}")
    elif sheet_name not in {s.name for s in book}:
        raise ValueError(f"sheet not found: {sheet_name}")

    # import all sheets or a subset
    for sheet in book:
        if sheet.name != sheet_name:
            continue
        else:
            break
    
    raise NotImplementedError("GO STRAIGHT TO HDF5 instead.")
    t = Table()
    for idx, column in enumerate(sheet.columns(), 1):
        if has_headers:
            header, start_row_pos = str(column[0]), 1
        else:
            header, start_row_pos = f"_{idx}", 0

        dtypes = {type(v) for v in column[start_row_pos:]}
        dtypes.discard(None)

        if dtypes == {int, float}:
            dtypes.remove(int)

        if len(dtypes) == 1:
            dtype = dtypes.pop()
            data = [dtype(v) if not isinstance(v, dtype) else v for v in column[start_row_pos:]]
        else:
            dtype, data = str, [str(v) for v in column[start_row_pos:]]
        t.add_column(header, data)
    return t


def ods_reader(path, has_headers=True, sheet_name=None, **kwargs):
    """
    returns Table from .ODS
    """
    if not isinstance(path, pathlib.Path):
        raise ValueError(f"expected pathlib.Path, got {type(path)}")
    sheets = pyexcel.get_book_dict(file_name=str(path))

    if sheet_name is None or sheet_name not in sheets:
        raise ValueError(f"No sheet_name declared: \navailable sheets:\n{[s.name for s in sheets]}")
            
    data = sheets[sheet_name]
    for _ in range(len(data)):  # remove empty lines at the end of the data.
        if "" == "".join(str(i) for i in data[-1]):
            data = data[:-1]
        else:
            break

    raise NotImplementedError("GO STRAIGHT TO HDF5 instead.")
    t = Table()
    for ix, value in enumerate(data[0]):
        if has_headers:
            header, start_row_pos = str(value), 1
        else:
            header, start_row_pos = f"_{ix + 1}", 0

        dtypes = set(type(row[ix]) for row in data[start_row_pos:] if len(row) > ix)
        dtypes.discard(None)
        if len(dtypes) == 1:
            dtype = dtypes.pop()
        elif dtypes == {float, int}:
            dtype = float
        else:
            dtype = str
        values = [dtype(row[ix]) for row in data[start_row_pos:] if len(row) > ix]
        t.add_column(header, data=values)
    return t

