from datetime import datetime
from tablite import Table
from pathlib import Path
import pytest


def test_defaults():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv", import_as='csv')
    d = tbl.to_dict()  # defaults.
    assert d['columns']
    for name in tbl.columns:
        assert name in d['columns']
        assert d['columns'][name] == tbl[name]
    assert d['total_rows'] == len(tbl)

    assert "row id" in d['columns']
    assert "row id" not in tbl.columns

def test_no_row_count():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv", import_as='csv')
    d = tbl.to_dict(row_count=None)  

    assert len(d['columns']) == len(tbl.columns)
    for name in tbl.columns:
        assert name in d['columns']
        assert d['columns'][name] == tbl[name]
    assert d['total_rows'] == len(tbl)
    
def test_limited_columns_and_slice():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv", import_as='csv')
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
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv", import_as='csv')
    s = tbl.to_json()  # just use defaults.
    tbl2 = Table.from_json(s)
    assert all(c in tbl2.columns for c in tbl.columns)
    assert len(tbl2.columns) == len(tbl.columns) + 1
    
def test_column_descriptors():
    types1 = {
        'Date': {datetime.time: 1997},
        'Id': {int: 1997},
        'Client': {int: 1997},
        'Product': {int: 1997},
        'Qty': {int: 1997}
    }

    desc1_actual = Table.get_column_descriptors(types1)
    desc1_expected = {'Date': 'time', 'Id': 'int', 'Client': 'int', 'Product': 'int', 'Qty': 'int'}
    
    assert desc1_actual == desc1_expected # test single types

    types2 = {
        'Date': {datetime.time: 1997},
        'Id': {int: 1997},
        'Client': {int: 997, str: 1000},
        'Product': {int: 1997},
        'Qty': {int: 1997}
    }

    desc2_actual = Table.get_column_descriptors(types2, False)
    desc2_expected = {'Date': 'time', 'Id': 'int', 'Client': 'mixed', 'Product': 'int', 'Qty': 'int'}
    
    assert desc2_actual == desc2_expected # test mixed types without mixed support disabled

    desc3_actual = Table.get_column_descriptors(types2, True)
    desc3_expected = {'Date': 'time', 'Id': 'int', 'Client': ['int', 'str'], 'Product': 'int', 'Qty': 'int'}

    assert desc3_actual == desc3_expected # test mixed types with mixed enabled enabled
    