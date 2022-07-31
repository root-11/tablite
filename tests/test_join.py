import sqlite3
from tablite import Table
import pytest


@pytest.fixture(autouse=True) # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield


def test_left_join():
    """ joining a table on itself. Wierd but possible. """
    numbers = Table()
    numbers.add_column('number', data=[1, 2, 3, 4, None])
    numbers.add_column('colour', data=['black', 'blue', 'white', 'white', 'blue'])

    left_join = numbers.left_join(numbers, left_keys=['colour'], right_keys=['colour'])
    left_join.show()

    assert list(left_join.rows) == [
        [1, 'black', 1, 'black'],
        [2, 'blue', 2, 'blue'],
        [2, 'blue', None, 'blue'],
        [None, 'blue', 2, 'blue'],
        [None, 'blue', None, 'blue'],
        [3, 'white', 3, 'white'],
        [3, 'white', 4, 'white'],
        [4, 'white', 3, 'white'],
        [4, 'white', 4, 'white'],
    ]


def test_left_join2():
    """ joining a table on itself. Wierd but possible. """
    numbers = Table()
    numbers.add_column('number', data=[1, 2, 3, 4, None])
    numbers.add_column('colour', data=['black', 'blue', 'white', 'white', 'blue'])

    left_join = numbers.left_join(numbers, left_keys=['colour'], right_keys=['colour'], left_columns=['colour', 'number'], right_columns=['number', 'colour'])
    left_join.show()

    assert list(left_join.rows) == [
        ['black', 1, 1, 'black'],
        ['blue', 2, 2, 'blue'],
        ['blue', 2, None, 'blue'],
        ['blue', None, 2, 'blue'],
        ['blue', None, None, 'blue'],
        ['white', 3, 3, 'white'],
        ['white', 3, 4, 'white'],
        ['white', 4, 3, 'white'],
        ['white', 4, 4, 'white'],
    ]

def _join_left(pairs_1, pairs_2, pairs_ans, column_1, column_2):
    """
    SELECT tbl1.number, tbl1.color, tbl2.number, tbl2.color
      FROM `tbl2`
      LEFT JOIN `tbl2`
        ON tbl1.color = tbl2.color;
    """
    numbers_1 = Table()
    numbers_1.add_column('number', data=[p[0] for p in pairs_1])
    numbers_1.add_column('colour', data=[p[1] for p in pairs_1])
    
    numbers_2 = Table()
    numbers_2.add_column('number', data=[p[0] for p in pairs_2])
    numbers_2.add_column('colour', data=[p[1] for p in pairs_2])

    left_join = numbers_1.left_join(numbers_2, left_keys=[column_1], right_keys=[column_2], left_columns=['number','colour'], right_columns=['number','colour'])

    assert len(pairs_ans) == len(left_join)
    for a, b in zip(sorted(pairs_ans, key=lambda x: str(x)), sorted(list(left_join.rows), key=lambda x: str(x))):
        assert a == tuple(b)


def test_same_join_1():
    """ FIDDLE: http://sqlfiddle.com/#!9/7dd756/7 """

    pairs_1 = [
        (1, 'black'),
        (2, 'blue'),
        (2, 'blue'),
        (3, 'white'),
        (3, 'white'),
        (4, 'white'),
        (4, 'white'),
        (None, 'blue'),
        (None, 'blue')
    ]
    pairs_2 = [
        (1, 'black'),
        (2, 'blue'),
        (None, 'blue'),
        (3, 'white'),
        (4, 'white'),
        (3, 'white'),
        (4, 'white'),
        (2, 'blue'),
        (None, 'blue')
    ]
    pairs_ans = [
        (1, 'black', 1, 'black'),
        (2, 'blue', 2, 'blue'),
        (2, 'blue', 2, 'blue'),
        (2, 'blue', None, 'blue'),
        (2, 'blue', None, 'blue'),
        (3, 'white', 3, 'white'),
        (3, 'white', 3, 'white'),
        (3, 'white', 4, 'white'),
        (3, 'white', 4, 'white'),
        (3, 'white', 3, 'white'),
        (3, 'white', 3, 'white'),
        (3, 'white', 4, 'white'),
        (3, 'white', 4, 'white'),
        (2, 'blue', 2, 'blue'),
        (2, 'blue', 2, 'blue'),
        (2, 'blue', None, 'blue'),
        (2, 'blue', None, 'blue'),
        (None, 'blue', 2, 'blue'),
        (None, 'blue', 2, 'blue'),
        (None, 'blue', None, 'blue'),
        (None, 'blue', None, 'blue'),
        (4, 'white', 3, 'white'),
        (4, 'white', 3, 'white'),
        (4, 'white', 4, 'white'),
        (4, 'white', 4, 'white'),
        (4, 'white', 3, 'white'),
        (4, 'white', 3, 'white'),
        (4, 'white', 4, 'white'),
        (4, 'white', 4, 'white'),
        (None, 'blue', 2, 'blue'),
        (None, 'blue', 2, 'blue'),
        (None, 'blue', None, 'blue'),
        (None, 'blue', None, 'blue'),
    ]

    _join_left(pairs_1, pairs_2, pairs_ans, 'colour', 'colour')


def test_left_join_2():
    """ FIDDLE: http://sqlfiddle.com/#!9/986b2a/3 """

    pairs_1 = [(1, 'black'), (2, 'blue'), (3, 'white'), (4, 'white'), (None, 'blue')]
    pairs_ans = [
        (1, 'black', 1, 'black'),
        (2, 'blue', 2, 'blue'),
        (None, 'blue', 2, 'blue'),
        (3, 'white', 3, 'white'),
        (4, 'white', 3, 'white'),
        (3, 'white', 4, 'white'),
        (4, 'white', 4, 'white'),
        (2, 'blue', None, 'blue'),
        (None, 'blue', None, 'blue'),
    ]
    _join_left(pairs_1, pairs_1, pairs_ans, 'colour', 'colour')



# https://en.wikipedia.org/wiki/Join_(SQL)#Inner_join
def test_wiki_joins():
    employees = Table()
    employees['last name'] = ["Rafferty", "Jones", "Heisenberg", "Robinson", "Smith", "Williams"]
    employees['department'] = [31,33,33,34,34,None]
    employees.show()

    sql = employees.to_sql()

    con = sqlite3.connect(':memory:')
    cur = con.cursor()
    cur.executescript(sql)
    result = cur.execute(f"select * from Table{employees.key};").fetchall()
    assert len(result) == 6
    
    departments = Table()
    departments['id'] = [31,33,34,35]
    departments['name'] = ['Sales', 'Engineering', 'Clerical', 'Marketing']
    departments.show()

    con = sqlite3.connect(':memory:')
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE department(
        DepartmentID INT PRIMARY KEY NOT NULL,
        DepartmentName VARCHAR(20));
        """)
    cur.execute("""
        CREATE TABLE employee (
            LastName VARCHAR(20),
            DepartmentID INT REFERENCES department(DepartmentID)
        );
        """)

    cur.execute("""
    INSERT INTO department
    VALUES (31, 'Sales'),
        (33, 'Engineering'),
        (34, 'Clerical'),
        (35, 'Marketing');
    """)

    cur.execute("""
    INSERT INTO employee
    VALUES ('Rafferty', 31),
        ('Jones', 33),
        ('Heisenberg', 33),
        ('Robinson', 34),
        ('Smith', 34),
        ('Williams', NULL);
    """)

    sql_result = cur.execute("""SELECT * FROM employee CROSS JOIN department;""").fetchall()
    # [('Rafferty', 31, 31, 'Sales'), ('Rafferty', 31, 33, 'Engineering'), ('Rafferty', 31, 34, 'Clerical'), ('Rafferty', 31, 35, 'Marketing'), ('Jones', 33, 31, 'Sales'), ('Jones', 33, 33, 'Engineering'), ('Jones', 33, 34, 'Clerical'), ('Jones', 33, 35, 'Marketing'), ('Heisenberg', 33, 31, 'Sales'), ('Heisenberg', 33, 33, 'Engineering'), ('Heisenberg', 33, 34, 'Clerical'), ('Heisenberg', 33, 35, 'Marketing'), ('Robinson', 34, 31, 'Sales'), ('Robinson', 34, 33, 'Engineering'), ('Robinson', 34, 34, 'Clerical'), ('Robinson', 34, 35, 'Marketing'), ('Smith', 34, 31, 'Sales'), ('Smith', 34, 33, 'Engineering'), ('Smith', 34, 34, 'Clerical'), ('Smith', 34, 35, 'Marketing'), ('Williams', None, 31, 'Sales'), ('Williams', None, 33, 'Engineering'), ('Williams', None, 34, 'Clerical'), ('Williams', None, 35, 'Marketing')]
    tbl_result = [tuple(row) for row in employees.cross_join(departments, left_keys=['department'], right_keys=['id']).rows]
    
    # Definition: A cross join produces a cartesian product between the two tables, returning all possible combinations of all rows. It has no on clause because you're just joining everything to everything.
    assert sql_result == tbl_result

    # inner join
    sql_result = cur.execute("""
    SELECT employee.LastName, employee.DepartmentID, department.DepartmentName 
    FROM employee 
    INNER JOIN department ON
    employee.DepartmentID = department.DepartmentID;
    """).fetchall()
    # [('Rafferty', 31, 'Sales'), ('Jones', 33, 'Engineering'), ('Heisenberg', 33, 'Engineering'), ('Robinson', 34, 'Clerical'), ('Smith', 34, 'Clerical')]
    tbl_result = [tuple(row) for row in employees.inner_join(departments, ['department'], ['id'], left_columns=['last name'], right_columns=['id', 'name']).rows]

    assert sql_result == tbl_result

    # left outer join
    sql_result = cur.execute("""
    SELECT *
    FROM employee 
    LEFT OUTER JOIN department ON employee.DepartmentID = department.DepartmentID;
    """).fetchall()
    # [('Rafferty', 31, 31, 'Sales'), ('Jones', 33, 33, 'Engineering'), ('Heisenberg', 33, 33, 'Engineering'), ('Robinson', 34, 34, 'Clerical'), ('Smith', 34, 34, 'Clerical'), ('Williams', None, None, None)]
    tbl_result = [tuple(row) for row in employees.left_join(departments, ['department'], ['id']).rows]

    assert sql_result == tbl_result

    # Right outer join
    try:
        sql_result = cur.execute("""
        SELECT *
        FROM employee RIGHT OUTER JOIN department
        ON employee.DepartmentID = department.DepartmentID;
        """).fetchall()

    except sqlite3.OperationalError:
        sql_result = None  # sqlite3.OperationalError: RIGHT and FULL OUTER JOINs are not currently supported

    # left join where L and R are swopped.
    tbl_result = [tuple(row) for row in departments.left_join(employees, ['id'], ['department']).rows]
    assert tbl_result == [(31, 'Sales', 'Rafferty', 31), (33, 'Engineering', 'Jones', 33), (33, 'Engineering', 'Heisenberg', 33), (34, 'Clerical', 'Robinson', 34), (34, 'Clerical', 'Smith', 34), (35, 'Marketing', None, None)]
    # ^-- this result is from here:  https://en.wikipedia.org/wiki/Join_(SQL)#Right_outer_join

    # Full outer join
    try:
        sql_result = cur.execute("""
        SELECT *
        FROM employee FULL OUTER JOIN department
        ON employee.DepartmentID = department.DepartmentID;
        """).fetchall()

    except sqlite3.OperationalError:
        sql_result = None  # sqlite3.OperationalError: RIGHT and FULL OUTER JOINs are not currently supported

    tbl_result = [tuple(row) for row in employees.outer_join(departments, left_keys=['department'], right_keys=['id']).rows]
    assert tbl_result == [
        ('Rafferty', 31, 31, 'Sales'), 
        ('Jones', 33, 33, 'Engineering'), 
        ('Heisenberg', 33, 33, 'Engineering'),
        ('Robinson', 34, 34, 'Clerical'),
        ('Smith', 34, 34, 'Clerical'), 
        ('Williams', None, None, None),
        (None, None, 35, 'Marketing')
    ]  # <-- result from https://en.wikipedia.org/wiki/Join_(SQL)#Full_outer_join
