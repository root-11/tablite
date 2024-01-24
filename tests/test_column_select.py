from tablite import Table
from tablite.config import Config
from datetime import date, time, datetime

def test_column_select_naming_1():
    tbl = Table(columns={
        "A ": [0, 1, 2]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A ", "type": "int", "allow_empty": False, "rename": None}
    ])

    assert list(select_1_pass.columns.keys()) == ['A ']
    assert list(select_1_fail.columns.keys()) == ['A ', 'reject_reason']

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A ']) == [0, 1, 2]


def test_column_select_naming_2():
    tbl = Table(columns={
        "A ": [0, 1, 2]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A ", "type": "int", "allow_empty": False, "rename": "A"}
    ])

    assert list(select_1_pass.columns.keys()) == ['A']
    assert list(select_1_fail.columns.keys()) == ['A ', 'reject_reason']

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A']) == [0, 1, 2]


def test_column_select_naming_3():
    tbl = Table(columns={
        "A ": [0, 1, 2]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A ", "type": "int", "allow_empty": False, "rename": "  A  "}
    ])

    assert list(select_1_pass.columns.keys()) == ['A']
    assert list(select_1_fail.columns.keys()) == ['A ', 'reject_reason']

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A']) == [0, 1, 2]


def test_column_select_naming_4():
    tbl = Table(columns={
        "A ": [0, 1, 2]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A ", "type": "int", "allow_empty": False}
    ])

    assert list(select_1_pass.columns.keys()) == ['A ']
    assert list(select_1_fail.columns.keys()) == ['A ', 'reject_reason']

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A ']) == [0, 1, 2]


def test_column_select_naming_5():
    tbl = Table(columns={
        "A ": [0, 1, 2]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A ", "type": "int", "allow_empty": False, "rename": ""}
    ])

    assert list(select_1_pass.columns.keys()) == ['A ']
    assert list(select_1_fail.columns.keys()) == ['A ', 'reject_reason']

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A ']) == [0, 1, 2]


def test_column_select_naming_6():
    tbl = Table(columns={
        "A ": [0, 1, 2]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A ", "type": "int", "allow_empty": False, "rename": ""}
    ])

    assert list(select_1_pass.columns.keys()) == ['A ']
    assert list(select_1_fail.columns.keys()) == ['A ', 'reject_reason']

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A ']) == [0, 1, 2]


def test_allow_empty_1():
    tbl = Table(columns={
        "A": [0, None, 2]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": False, "rename": ""}
    ])

    assert len(select_1_pass) == 2
    assert list(select_1_pass['A']) == [0, 2]

    assert len(select_1_fail) == 1
    assert list(select_1_fail['A']) == [None]


def test_allow_empty_2():
    tbl = Table(columns={
        "A": [0, None, 2]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": True, "rename": ""}
    ])

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A']) == [0, None, 2]

    assert len(select_1_fail) == 0
    assert list(select_1_fail['A']) == []


def test_sp_1():
    original_mode = Config.MULTIPROCESSING_MODE
    original_page = Config.PAGE_SIZE

    Config.MULTIPROCESSING_MODE = Config.FALSE
    Config.PAGE_SIZE = 2

    try:
        tbl = Table(columns={
            "A": ["0", None, "2"],
            "B": ["3", None, "4"]
        })

        select_1_pass, select_1_fail = tbl.column_select(cols=[
            {"column": "A", "type": "int", "allow_empty": True},
            {"column": "B", "type": "int", "allow_empty": True}
        ])

        assert len(select_1_pass) == 3
        assert list(select_1_pass['A']) == [0, None, 2]

        assert len(select_1_pass) == 3
        assert list(select_1_pass['B']) == [3, None, 4]

        assert len(select_1_fail) == 0
        assert list(select_1_fail['A']) == []
    finally:
        Config.MULTIPROCESSING_MODE = original_mode
        Config.PAGE_SIZE = original_page


def test_mp_1():
    original_mode = Config.MULTIPROCESSING_MODE
    original_page = Config.PAGE_SIZE

    Config.MULTIPROCESSING_MODE = Config.FORCE
    Config.PAGE_SIZE = 2

    try:
        tbl = Table(columns={
            "A": ["0", None, "2"],
            "B": ["3", None, "4"]
        })

        select_1_pass, select_1_fail = tbl.column_select(cols=[
            {"column": "A", "type": "int", "allow_empty": True},
            {"column": "B", "type": "int", "allow_empty": True}
        ])

        assert len(select_1_pass) == 3
        assert list(select_1_pass['A']) == [0, None, 2]
        assert list(select_1_pass['B']) == [3, None, 4]

        assert len(select_1_fail) == 0
        assert list(select_1_fail['A']) == []
    finally:
        Config.MULTIPROCESSING_MODE = original_mode
        Config.PAGE_SIZE = original_page


def test_casting_1():
    tbl = Table(columns={
        "A": ["0", None, "2"]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": True, "rename": ""}
    ])

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A']) == [0, None, 2]

    assert len(select_1_fail) == 0
    assert list(select_1_fail['A']) == []


def test_casting_2():
    tbl = Table(columns={
        "A": ["0", None, "2"]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "float", "allow_empty": True, "rename": ""}
    ])

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A']) == [0.0, None, 2.0]

    assert len(select_1_fail) == 0
    assert list(select_1_fail['A']) == []


def test_casting_3():
    tbl = Table(columns={
        "A": ["0", None, "2", "c"]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": True, "rename": ""}
    ])

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A']) == [0, None, 2]

    assert len(select_1_fail) == 1
    assert list(select_1_fail['A']) == ["c"]


def test_casting_4():
    tbl = Table(columns={
        "A": [0.7, None, "2", "c"]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": True, "rename": ""}
    ])

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A']) == [0, None, 2]

    assert len(select_1_fail) == 1
    assert list(select_1_fail['A']) == ["c"]


def test_casting_5():
    tbl = Table(columns={
        "A": [0, None, 2, "c"]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "float", "allow_empty": True, "rename": ""}
    ])

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A']) == [0.0, None, 2.0]

    assert len(select_1_fail) == 1
    assert list(select_1_fail['A']) == ["c"]


def test_casting_6():
    """ shouldn't perform any iterations, because we accept empties and inp type matches out type """
    tbl = Table(columns={
        "A": [0, None, 2]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": True, "rename": "AA"}
    ])

    assert len(select_1_pass) == 3
    assert list(select_1_pass['AA']) == [0, None, 2]

    assert len(select_1_fail) == 0
    assert list(select_1_fail['A']) == []


def test_casting_7():
    """ shouldn't perform type checks per row for A column, because we accept empties and inp type matches out type """
    tbl = Table(columns={
        "A": [0, None, 2],
        "B": ["3", "4", "5"]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": True, "rename": "AA"},
        {"column": "B", "type": "int", "allow_empty": True, "rename": "BB"}
    ])

    assert len(select_1_pass) == 3
    assert list(select_1_pass['AA']) == [0, None, 2]
    assert list(select_1_pass['BB']) == [3, 4, 5]

    assert len(select_1_fail) == 0
    assert list(select_1_fail['A']) == []


def test_casting_8():
    tbl = Table(columns={
        "A": [],
        "B": []
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": True, "rename": "AA"},
        {"column": "B", "type": "int", "allow_empty": True, "rename": "BB"}
    ])

    assert len(select_1_pass) == 0
    assert list(select_1_pass['AA']) == []
    assert list(select_1_pass['BB']) == []

    assert len(select_1_fail) == 0
    assert list(select_1_fail['A']) == []


def test_casting_9():
    tbl = Table(columns={
        "A": [None],
        "B": [None]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": True, "rename": "AA"},
        {"column": "B", "type": "int", "allow_empty": True, "rename": "BB"}
    ])

    assert len(select_1_pass) == 1
    assert list(select_1_pass['AA']) == [None]
    assert list(select_1_pass['BB']) == [None]

    assert len(select_1_fail) == 0
    assert list(select_1_fail['A']) == []

def test_casting_10():
    """ python parser can't parse floating point as int """
    tbl = Table(columns={
        "A": ["0.4", None, "2.9"]
    })

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": "A", "type": "int", "allow_empty": True, "rename": ""}
    ])

    assert len(select_1_pass) == 3
    assert list(select_1_pass['A']) == [0, None, 2]

    assert len(select_1_fail) == 0
    assert list(select_1_fail['A']) == []


def test_casting_bool_1():
    col_name = "bool"
    input_values = [True, False]

    tbl = Table({col_name: input_values})

    types = ["bool", "int", "float", "str", "date", "time", "datetime"]
    results = [
        [True, False],
        [1, 0],
        [1.0, 0.0],
        ["True", "False"],
        [date(1970, 1, 2), date(1970, 1, 1)],
        [time(0, 0, 1), time(0, 0, 0)],
        [datetime(1970, 1, 1, 0, 0, 1), datetime(1970, 1, 1, 0, 0, 0)]
    ]

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": col_name, "type": t, "allow_empty": False, "rename": t}
        for t in types
    ])

    assert len(select_1_pass) == 2, select_1_pass.show()
    assert len(select_1_fail) == 0, select_1_fail.show()
    assert list(select_1_pass.columns.keys()) == types
    
    for true, expect in zip((list(l) for l in select_1_pass.columns.values()), results):
        assert true == expect

def test_casting_int_1():
    col_name = "int"
    input_values = [1, 0]

    tbl = Table({col_name: input_values})

    types = ["bool", "int", "float", "str", "date", "time", "datetime"]
    results = [
        [True, False],
        [1, 0],
        [1.0, 0.0],
        ["1", "0"],
        [date(1970, 1, 2), date(1970, 1, 1)],
        [time(0, 0, 1), time(0, 0, 0)],
        [datetime(1970, 1, 1, 0, 0, 1), datetime(1970, 1, 1, 0, 0, 0)]
    ]

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": col_name, "type": t, "allow_empty": False, "rename": t}
        for t in types
    ])

    assert len(select_1_pass) == 2, select_1_pass.show()
    assert len(select_1_fail) == 0, select_1_fail.show()
    assert list(select_1_pass.columns.keys()) == types
    
    for true, expect in zip((list(l) for l in select_1_pass.columns.values()), results):
        assert true == expect

def test_casting_float_1():
    col_name = "float"
    input_values = [1.0, 0.0]

    tbl = Table({col_name: input_values})

    types = ["bool", "int", "float", "str", "date", "time", "datetime"]
    results = [
        [True, False],
        [1, 0],
        [1.0, 0.0],
        ["1.0", "0.0"],
        [date(1970, 1, 2), date(1970, 1, 1)],
        [time(0, 0, 1), time(0, 0, 0)],
        [datetime(1970, 1, 1, 0, 0, 1), datetime(1970, 1, 1, 0, 0, 0)]
    ]

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": col_name, "type": t, "allow_empty": False, "rename": t}
        for t in types
    ])

    assert len(select_1_pass) == 2, select_1_pass.show()
    assert len(select_1_fail) == 0, select_1_fail.show()
    assert list(select_1_pass.columns.keys()) == types
    
    for true, expect in zip((list(l) for l in select_1_pass.columns.values()), results):
        assert true == expect

def test_casting_str_1():
    col_name = "str"
    input_values = ["1.0", "0.0"]

    tbl = Table({col_name: input_values})

    types = ["int", "float", "str"]
    results = [
        [1, 0],
        [1.0, 0.0],
        ["1.0", "0.0"]
    ]

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": col_name, "type": t, "allow_empty": False, "rename": t}
        for t in types
    ])

    assert len(select_1_pass) == 2, select_1_pass.show()
    assert len(select_1_fail) == 0, select_1_fail.show()
    assert list(select_1_pass.columns.keys()) == types
    
    for true, expect in zip((list(l) for l in select_1_pass.columns.values()), results):
        assert true == expect

def test_casting_date_1():
    col_name = "date"
    input_values = [date(2000, 1, 1), date(2000, 1, 2)]

    tbl = Table({col_name: input_values})

    types = ["bool", "int", "float", "str", "date", "datetime"]
    results = [
        [True, True],
        [946684800, 946771200],
        [946684800.0, 946771200.0],
        ["2000-01-01", "2000-01-02"],
        [date(2000, 1, 1), date(2000, 1, 2)],
        [datetime(2000, 1, 1), datetime(2000, 1, 2)],
    ]

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": col_name, "type": t, "allow_empty": False, "rename": t}
        for t in types
    ])

    assert len(select_1_pass) == 2, select_1_pass.show()
    assert len(select_1_fail) == 0, select_1_fail.show()
    assert list(select_1_pass.columns.keys()) == types
    
    for true, expect in zip((list(l) for l in select_1_pass.columns.values()), results):
        assert true == expect

def test_casting_time_1():
    col_name = "time"
    input_values = [time(0, 0, 1), time(0, 0, 0)]

    tbl = Table({col_name: input_values})

    types = ["bool", "int", "float", "str", "time"]
    results = [
        [True, False],
        [1, 0],
        [1.0, 0.0],
        ["00:00:01", "00:00:00"],
        [time(0, 0, 1), time(0, 0, 0)]
    ]

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": col_name, "type": t, "allow_empty": False, "rename": t}
        for t in types
    ])

    assert len(select_1_pass) == 2, select_1_pass.show()
    assert len(select_1_fail) == 0, select_1_fail.show()
    assert list(select_1_pass.columns.keys()) == types
    
    for true, expect in zip((list(l) for l in select_1_pass.columns.values()), results):
        assert true == expect

def test_casting_datetime_1():
    col_name = "datetime"
    input_values = [datetime(2000, 1, 1), datetime(2000, 1, 2)]

    tbl = Table({col_name: input_values})

    types = ["bool", "int", "float", "str", "date", "datetime"]
    results = [
        [True, True],
        [946684800, 946771200],
        [946684800.0, 946771200.0],
        ["2000-01-01 00:00:00", "2000-01-02 00:00:00"],
        [date(2000, 1, 1), date(2000, 1, 2)],
        [datetime(2000, 1, 1), datetime(2000, 1, 2)],
    ]

    select_1_pass, select_1_fail = tbl.column_select(cols=[
        {"column": col_name, "type": t, "allow_empty": False, "rename": t}
        for t in types
    ])

    assert len(select_1_pass) == 2, select_1_pass.show()
    assert len(select_1_fail) == 0, select_1_fail.show()
    assert list(select_1_pass.columns.keys()) == types
    
    for true, expect in zip((list(l) for l in select_1_pass.columns.values()), results):
        assert true == expect

if __name__ == "__main__":
    test_casting_time_1()