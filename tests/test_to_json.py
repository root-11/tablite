from tablite import Table
from pathlib import Path


def test_defaults():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv")
    assert len(tbl) > 0, "data wasn't loaded"
    d = tbl.to_dict()  # defaults.
    for name in tbl.columns:
        assert name in d
        assert d[name] == tbl[name]


def test_no_row_count():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv")
    d = tbl.to_dict()

    for name in tbl.columns:
        assert name in d
        assert d[name] == tbl[name]

    jsn_d = tbl.as_json_serializable()
    assert isinstance(jsn_d, dict)
    for k, v in d.items():
        len(jsn_d["columns"][k]) == len(v)

    assert jsn_d["total_rows"] == len(tbl)


def test_limited_columns_and_slice():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv")
    columns = ["Id", "Client", "Product"]
    slcs = slice(3, 100, 9)
    d = tbl.to_dict(columns=columns, slice_=slcs)

    L = list(range(*slcs.indices(len(tbl))))
    for _, data in d.items():
        assert len(data) == len(L)


def test_defaults_to_json_and_back():
    tbl = Table.import_file(Path(__file__).parent / "data" / "ols.csv")
    s = tbl.to_json()  # just use defaults.
    tbl2 = Table.from_json(s)
    assert all(c in tbl2.columns for c in tbl.columns)
    assert len(tbl2.columns) == len(tbl.columns) + 1
