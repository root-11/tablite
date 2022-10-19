from tablite import Table
from pathlib import Path
import pytest


@pytest.fixture(autouse=True) # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield


def test_defaults():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv")
    d = tbl.to_dict()  # defaults.
    assert d['columns']
    for name in tbl.columns:
        assert name in d['columns']
        assert d['columns'][name] == tbl[name]
    assert d['total_rows'] == len(tbl)

    assert "row id" in d['columns']
    assert "row id" not in tbl.columns


def test_no_row_count():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv")
    d = tbl.to_dict(row_count=None)  

    assert len(d['columns']) == len(tbl.columns)
    for name in tbl.columns:
        assert name in d['columns']
        assert d['columns'][name] == tbl[name]
    assert d['total_rows'] == len(tbl)


def test_limited_columns_and_slice():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv")
    columns=["Id","Client","Product"]
    slcs = slice(3,100,9)
    d = tbl.to_dict(columns=columns, slice_=slcs, row_count='rid')
    
    assert len(d['columns']) == len(columns) + 1  # 1 for row id.
    L = list(range(*slcs.indices(len(tbl))))
    for _, data in d['columns'].items():
         assert len(data) == len(L)
    assert d['columns']['rid'] == [1 + i for i in L]  # start_on = 1
    
    assert d['total_rows'] == len(tbl)


def test_defaults_to_json_and_back():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv")
    s = tbl.to_json()  # just use defaults.
    tbl2 = Table.from_json(s)
    assert all(c in tbl2.columns for c in tbl.columns)
    assert len(tbl2.columns) == len(tbl.columns) + 1
    
