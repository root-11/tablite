import sqlite3
from tablite import Table
from tablite.config import Config
import pytest


@pytest.fixture(autouse=True)  # this resets the HDF5 file for every test.
def refresh():
    try:
        yield
    finally:
        Config.PROCESSING_PRIORITY = Config.AUTO


def do_left_join():
    """joining a table on itself. Wierd but possible."""
    numbers = Table()
    numbers.add_column("number", data=[1, 2, 3, 4, None])
    numbers.add_column("colour", data=["black", "blue", "white", "white", "blue"])

    numbers2 = Table()
    numbers2.add_column("number", data=[1, 2, 3, 4, None, 6, 6, 6, 6, 6, 6])
    numbers2.add_column(
        "colour",
        data=["black", "blue", "white", "white", "blue", "purple", "purple", "purple", "purple", "purple", "purple"],
    )

    left_join = numbers.left_join(numbers2, left_keys=["colour"], right_keys=["colour"])
    left_join.show()

    assert list(left_join.rows) == [
        [1, "black", 1, "black"],
        [2, "blue", 2, "blue"],
        [2, "blue", None, "blue"],
        [None, "blue", 2, "blue"],
        [None, "blue", None, "blue"],
        [3, "white", 3, "white"],
        [3, "white", 4, "white"],
        [4, "white", 3, "white"],
        [4, "white", 4, "white"],
    ]


def test_left_join_sp():
    Config.MULTIPROCESSING_MODE = Config.FALSE
    do_left_join()
    Config.MULTIPROCESSING_MODE = Config.reset()


def test_left_join_mp():
    Config.MULTIPROCESSING_MODE = Config.FORCE
    do_left_join()
    Config.MULTIPROCESSING_MODE = Config.reset()


def do_left_join2():
    """joining a table on itself. Wierd but possible."""
    numbers = Table()
    numbers.add_column("number", data=[1, 2, 3, 4, None])
    numbers.add_column("colour", data=["black", "blue", "white", "white", "blue"])

    left_join = numbers.left_join(
        numbers,
        left_keys=["colour"],
        right_keys=["colour"],
        left_columns=["colour", "number"],
        right_columns=["number", "colour"],
    )
    left_join.show()

    assert list(left_join.rows) == [
        ["black", 1, 1, "black"],
        ["blue", 2, 2, "blue"],
        ["blue", 2, None, "blue"],
        ["blue", None, 2, "blue"],
        ["blue", None, None, "blue"],
        ["white", 3, 3, "white"],
        ["white", 3, 4, "white"],
        ["white", 4, 3, "white"],
        ["white", 4, 4, "white"],
    ]


def test_left_join2_sp():
    Config.MULTIPROCESSING_MODE = Config.FALSE
    do_left_join2()
    Config.MULTIPROCESSING_MODE = Config.reset()


def test_left_join2_mp():
    Config.MULTIPROCESSING_MODE = Config.FALSE
    do_left_join2()
    Config.MULTIPROCESSING_MODE = Config.reset()


def _join_left(pairs_1, pairs_2, pairs_ans, column_1, column_2):
    """
    SELECT tbl1.number, tbl1.color, tbl2.number, tbl2.color
      FROM `tbl2`
      LEFT JOIN `tbl2`
        ON tbl1.color = tbl2.color;
    """
    numbers_1 = Table()
    numbers_1.add_column("number", data=[p[0] for p in pairs_1])
    numbers_1.add_column("colour", data=[p[1] for p in pairs_1])

    numbers_2 = Table()
    numbers_2.add_column("number", data=[p[0] for p in pairs_2])
    numbers_2.add_column("colour", data=[p[1] for p in pairs_2])

    left_join = numbers_1.left_join(
        numbers_2,
        left_keys=[column_1],
        right_keys=[column_2],
        left_columns=["number", "colour"],
        right_columns=["number", "colour"],
    )

    assert len(pairs_ans) == len(left_join)
    for a, b in zip(sorted(pairs_ans, key=lambda x: str(x)), sorted(list(left_join.rows), key=lambda x: str(x))):
        assert a == tuple(b)


def do_same_join_1():
    """FIDDLE: http://sqlfiddle.com/#!9/7dd756/7"""

    pairs_1 = [
        (1, "black"),
        (2, "blue"),
        (2, "blue"),
        (3, "white"),
        (3, "white"),
        (4, "white"),
        (4, "white"),
        (None, "blue"),
        (None, "blue"),
    ]
    pairs_2 = [
        (1, "black"),
        (2, "blue"),
        (None, "blue"),
        (3, "white"),
        (4, "white"),
        (3, "white"),
        (4, "white"),
        (2, "blue"),
        (None, "blue"),
    ]
    pairs_ans = [
        (1, "black", 1, "black"),
        (2, "blue", 2, "blue"),
        (2, "blue", 2, "blue"),
        (2, "blue", None, "blue"),
        (2, "blue", None, "blue"),
        (3, "white", 3, "white"),
        (3, "white", 3, "white"),
        (3, "white", 4, "white"),
        (3, "white", 4, "white"),
        (3, "white", 3, "white"),
        (3, "white", 3, "white"),
        (3, "white", 4, "white"),
        (3, "white", 4, "white"),
        (2, "blue", 2, "blue"),
        (2, "blue", 2, "blue"),
        (2, "blue", None, "blue"),
        (2, "blue", None, "blue"),
        (None, "blue", 2, "blue"),
        (None, "blue", 2, "blue"),
        (None, "blue", None, "blue"),
        (None, "blue", None, "blue"),
        (4, "white", 3, "white"),
        (4, "white", 3, "white"),
        (4, "white", 4, "white"),
        (4, "white", 4, "white"),
        (4, "white", 3, "white"),
        (4, "white", 3, "white"),
        (4, "white", 4, "white"),
        (4, "white", 4, "white"),
        (None, "blue", 2, "blue"),
        (None, "blue", 2, "blue"),
        (None, "blue", None, "blue"),
        (None, "blue", None, "blue"),
    ]

    _join_left(pairs_1, pairs_2, pairs_ans, "colour", "colour")


def test_same_join_1_sp():
    Config.MULTIPROCESSING_MODE = Config.FALSE
    do_same_join_1()
    Config.MULTIPROCESSING_MODE = Config.reset()


def test_same_join_1_mp():
    Config.MULTIPROCESSING_MODE = Config.FORCE
    do_same_join_1()
    Config.MULTIPROCESSING_MODE = Config.reset()


def do_left_join_2():
    """FIDDLE: http://sqlfiddle.com/#!9/986b2a/3"""

    pairs_1 = [(1, "black"), (2, "blue"), (3, "white"), (4, "white"), (None, "blue")]
    pairs_ans = [
        (1, "black", 1, "black"),
        (2, "blue", 2, "blue"),
        (None, "blue", 2, "blue"),
        (3, "white", 3, "white"),
        (4, "white", 3, "white"),
        (3, "white", 4, "white"),
        (4, "white", 4, "white"),
        (2, "blue", None, "blue"),
        (None, "blue", None, "blue"),
    ]
    _join_left(pairs_1, pairs_1, pairs_ans, "colour", "colour")


def test_left_join_2_sp():
    Config.MULTIPROCESSING_MODE = Config.FALSE
    do_left_join_2()
    Config.MULTIPROCESSING_MODE = Config.reset()


def test_left_join_2_mp():
    Config.MULTIPROCESSING_MODE = Config.FORCE
    do_left_join_2()
    Config.MULTIPROCESSING_MODE = Config.reset()


# https://en.wikipedia.org/wiki/Join_(SQL)#Inner_join
def do_wiki_joins():
    employees = Table()
    employees["last name"] = ["Rafferty", "Jones", "Heisenberg", "Robinson", "Smith", "Williams"]
    employees["department"] = [31, 33, 33, 34, 34, None]
    employees.show()

    sql = employees.to_sql(name="department")

    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.executescript(sql)
    result = cur.execute(f"select * from department;").fetchall()
    assert len(result) == 6

    departments = Table()
    departments["id"] = [31, 33, 34, 35]
    departments["name"] = ["Sales", "Engineering", "Clerical", "Marketing"]
    departments.show()

    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE department(
        DepartmentID INT PRIMARY KEY NOT NULL,
        DepartmentName VARCHAR(20));
        """
    )
    cur.execute(
        """
        CREATE TABLE employee (
            LastName VARCHAR(20),
            DepartmentID INT REFERENCES department(DepartmentID)
        );
        """
    )

    cur.execute(
        """
    INSERT INTO department
    VALUES (31, 'Sales'),
        (33, 'Engineering'),
        (34, 'Clerical'),
        (35, 'Marketing');
    """
    )

    cur.execute(
        """
    INSERT INTO employee
    VALUES ('Rafferty', 31),
        ('Jones', 33),
        ('Heisenberg', 33),
        ('Robinson', 34),
        ('Smith', 34),
        ('Williams', NULL);
    """
    )

    sql_result = cur.execute("""SELECT * FROM employee CROSS JOIN department;""").fetchall()
    # L = [
    #     ("Rafferty", 31, 31, "Sales"),
    #     ("Rafferty", 31, 33, "Engineering"),
    #     ("Rafferty", 31, 34, "Clerical"),
    #     ("Rafferty", 31, 35, "Marketing"),
    #     ("Jones", 33, 31, "Sales"),
    #     ("Jones", 33, 33, "Engineering"),
    #     ("Jones", 33, 34, "Clerical"),
    #     ("Jones", 33, 35, "Marketing"),
    #     ("Heisenberg", 33, 31, "Sales"),
    #     ("Heisenberg", 33, 33, "Engineering"),
    #     ("Heisenberg", 33, 34, "Clerical"),
    #     ("Heisenberg", 33, 35, "Marketing"),
    #     ("Robinson", 34, 31, "Sales"),
    #     ("Robinson", 34, 33, "Engineering"),
    #     ("Robinson", 34, 34, "Clerical"),
    #     ("Robinson", 34, 35, "Marketing"),
    #     ("Smith", 34, 31, "Sales"),
    #     ("Smith", 34, 33, "Engineering"),
    #     ("Smith", 34, 34, "Clerical"),
    #     ("Smith", 34, 35, "Marketing"),
    #     ("Williams", None, 31, "Sales"),
    #     ("Williams", None, 33, "Engineering"),
    #     ("Williams", None, 34, "Clerical"),
    #     ("Williams", None, 35, "Marketing"),
    # ]
    tbl_result = [
        tuple(row) for row in employees.cross_join(departments, left_keys=["department"], right_keys=["id"]).rows
    ]

    # Definition: A cross join produces a cartesian product between the two tables,
    # returning all possible combinations of all rows. It has no on clause because
    # you're just joining everything to everything.
    assert set(sql_result) == set(tbl_result)
    assert len(sql_result) == len(tbl_result)

    # inner join
    sql_result = cur.execute(
        """
    SELECT employee.LastName, employee.DepartmentID, department.DepartmentName
    FROM employee
    INNER JOIN department ON
    employee.DepartmentID = department.DepartmentID;
    """
    ).fetchall()
    # L = [
    #     ("Rafferty", 31, "Sales"),
    #     ("Jones", 33, "Engineering"),
    #     ("Heisenberg", 33, "Engineering"),
    #     ("Robinson", 34, "Clerical"),
    #     ("Smith", 34, "Clerical"),
    # ]
    tbl_result = [
        tuple(row)
        for row in employees.inner_join(
            departments, ["department"], ["id"], left_columns=["last name"], right_columns=["id", "name"]
        ).rows
    ]

    assert sql_result == tbl_result

    # left outer join
    sql_result = cur.execute(
        """
    SELECT *
    FROM employee
    LEFT OUTER JOIN department ON employee.DepartmentID = department.DepartmentID;
    """
    ).fetchall()
    # L = [
    #     ("Rafferty", 31, 31, "Sales"),
    #     ("Jones", 33, 33, "Engineering"),
    #     ("Heisenberg", 33, 33, "Engineering"),
    #     ("Robinson", 34, 34, "Clerical"),
    #     ("Smith", 34, 34, "Clerical"),
    #     ("Williams", None, None, None),
    # ]
    tbl_join = employees.left_join(departments, ["department"], ["id"])
    tbl_join.show()
    tbl_result = [tuple(row) for row in tbl_join.rows]

    assert sql_result == tbl_result

    # Right outer join
    try:
        sql_result = cur.execute(
            """
        SELECT *
        FROM employee RIGHT OUTER JOIN department
        ON employee.DepartmentID = department.DepartmentID;
        """
        ).fetchall()

    except sqlite3.OperationalError:
        sql_result = None  # sqlite3.OperationalError: RIGHT and FULL OUTER JOINs are not currently supported

    # left join where L and R are swopped.
    tbl_result = [tuple(row) for row in departments.left_join(employees, ["id"], ["department"]).rows]
    assert tbl_result == [
        (31, "Sales", "Rafferty", 31),
        (33, "Engineering", "Jones", 33),
        (33, "Engineering", "Heisenberg", 33),
        (34, "Clerical", "Robinson", 34),
        (34, "Clerical", "Smith", 34),
        (35, "Marketing", None, None),
    ]
    # ^-- this result is from here:  https://en.wikipedia.org/wiki/Join_(SQL)#Right_outer_join

    # Full outer join
    try:
        sql_result = cur.execute(
            """
        SELECT *
        FROM employee FULL OUTER JOIN department
        ON employee.DepartmentID = department.DepartmentID;
        """
        ).fetchall()

    except sqlite3.OperationalError:
        sql_result = None  # sqlite3.OperationalError: RIGHT and FULL OUTER JOINs are not currently supported

    tbl_result = [
        tuple(row) for row in employees.outer_join(departments, left_keys=["department"], right_keys=["id"]).rows
    ]
    assert tbl_result == [
        ("Rafferty", 31, 31, "Sales"),
        ("Jones", 33, 33, "Engineering"),
        ("Heisenberg", 33, 33, "Engineering"),
        ("Robinson", 34, 34, "Clerical"),
        ("Smith", 34, 34, "Clerical"),
        ("Williams", None, None, None),
        (None, None, 35, "Marketing"),
    ]  # <-- result from https://en.wikipedia.org/wiki/Join_(SQL)#Full_outer_join


def test_wiki_joins_sp():
    Config.MULTIPROCESSING_MODE = Config.FALSE
    do_wiki_joins()
    Config.MULTIPROCESSING_MODE = Config.reset()


def test_wiki_joins_mp():
    Config.MULTIPROCESSING_MODE = Config.FORCE
    do_wiki_joins()
    Config.MULTIPROCESSING_MODE = Config.reset()

def test_join_with_key_merge():
    a = Table(columns={'A': [1,2,3,None,5], 'B':[10,20,None,40,50]})    
    b = Table(columns={'C': [1,2,3,6,7], 'D':[11,12,13,16,17]})
    c = a.join(b,left_keys=['A'], right_keys=['C'], left_columns=['A', 'B'], right_columns=['C','D'], kind="outer")
    
    # +==+====+====+====+====+
    # |# | A  | B  | C  | D  |
    # +--+----+----+----+----+
    # | 0|   1|  10|   1|  11|
    # | 1|   2|  20|   2|  12|
    # | 2|   3|None|   3|  13|
    # | 3|None|  40|None|None|
    # | 4|   5|  50|None|None|
    # | 5|None|None|   6|  16|
    # | 6|None|None|   7|  17|
    # +==+====+====+====+====+
    assert c["A"] == [1,2,3,None,5,None,None]
    assert c['C'] == [1,2,3,None,None,6,7]

    d = c.copy().merge("A", "C", new="E", criteria=[v != None for v in c['A']])
    assert "A" not in d.columns
    assert "C" not in d.columns
    assert d["E"] == [1,2,3,None,5,6,7]


    e = a.join(b,left_keys=['A'], right_keys=['C'], left_columns=['A', 'B'], right_columns=['C','D'], kind="outer", merge_keys=True)
    assert e["A"] == [1,2,3,None,5,6,7]
    

def test_left_join_with_key_merge():
    a = Table(columns={'SKU_ID':[1,2,4], "A": [10,20,30], "B": [40,50,60]})
    b = Table(columns={'SKU_ID':[1,1,3], 'C':[11,22,33], 'D':[44,55,66]})

    # LEFT
    c1a = a.left_join(b, ["SKU_ID"], ["SKU_ID"], merge_keys=False)
    assert isinstance(c1a, Table)  # nothing changes in the table
    assert "SKU_ID_1" in c1a.columns

    c1b = a.left_join(b, ["SKU_ID"], ["SKU_ID"], merge_keys=True)
    assert c1b["SKU_ID"] == [1,1,2,4]
    assert "SKU_ID_1" not in c1b.columns

    assert c1a["A"] == c1b["A"]

    # OUTER
    c2a = a.outer_join(b, ["SKU_ID"], ["SKU_ID"], merge_keys=True)
    assert c2a["SKU_ID"] == [1,1,2,4,3]
    assert "SKU_ID_1" not in c2a.columns
    
    c2b = a.outer_join(b, ["SKU_ID"], ["SKU_ID"], merge_keys=False)
    assert "SKU_ID_1" in c2b.columns

    assert c2a["A"] == c2b["A"]

    # INNER
    c3a = a.inner_join(b, ["SKU_ID"], ["SKU_ID"], merge_keys=True)
    assert isinstance(c3a, Table) 
    assert "SKU_ID_1" not in c3a.columns

    c3b = a.inner_join(b, ["SKU_ID"], ["SKU_ID"], merge_keys=False)
    assert "SKU_ID_1" in c3b.columns

    assert c3a["A"] == c3b["A"]

    # CROSS
    c4a = a.cross_join(b, ["SKU_ID"], ["SKU_ID"], merge_keys=True)
    assert isinstance(c4a, Table)
    assert "SKU_ID_1" not in c4a.columns

    c4b = a.cross_join(b, ["SKU_ID"], ["SKU_ID"], merge_keys=False)
    assert "SKU_ID_1" in c4b.columns
    
    assert c4a["A"] == c4b["A"]
