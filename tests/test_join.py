from tablite import Table


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
    numbers_1.add_column('number', int, allow_empty=True)
    numbers_1.add_column('colour', str)
    for row in pairs_1:
        numbers_1.add_row(row)

    numbers_2 = Table()
    numbers_2.add_column('number', int, allow_empty=True)
    numbers_2.add_column('colour', str)
    for row in pairs_2:
        numbers_2.add_row(row)

    left_join = numbers_1.left_join(numbers_2, left_keys=[column_1], right_keys=[column_2], left_columns=['number','colour'], right_columns=['number','colour'])

    assert len(pairs_ans) == len(left_join)
    for a, b in zip(sorted(pairs_ans, key=lambda x: str(x)), sorted(list(left_join.rows), key=lambda x: str(x))):
        assert a == b


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