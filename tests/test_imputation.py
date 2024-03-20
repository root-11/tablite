from tablite import Table


def test_replace_missing_values_00():
    sample = [[1, 2, 3], [1, 2, None], [5, 5, 5], [6, 6, 6]]

    t = Table()
    t.add_columns(*list("abc"))
    for row in sample:
        t.add_rows(row)

    expected = [r[:] for r in sample]
    expected[1][-1] = 3
    result = t.imputation(sources=["a", "b"], targets=["c"], method="nearest neighbour", missing={None})
    assert [r for r in result.rows] == expected
    result = t.imputation(sources=["a", "b"], targets=["c"], method="nearest neighbour")  # missing not declared.
    assert [r for r in result.rows] == expected

def test_replace_missing_values_345():
    t = Table({
        "a": [0, 1, None, 3, 0],
        "b": ["4", 5, 6, 7, 4]
    })

    result = t.imputation(sources=["a", "b"], targets=["a"], method="nearest neighbour", missing={None})
    assert result["a"][2] == 3


def test_nearest_neighbour_multiple_missing():
    sample = [[1, 2, 3], [1, 2, None], [5, 5, 5], [5, 5, "NULL"], [6, 6, 6], [6, -1, 6]]

    t = Table()
    t.add_columns(*list("abc"))
    for row in sample:
        t.add_rows(row)

    result = t.imputation(sources=["a", "b"], targets=["c"], method="nearest neighbour", missing={None, "NULL", -1})

    expected = [[1, 2, 3], [1, 2, 3], [5, 5, 5], [5, 5, 5], [6, 6, 6], [6, -1, 6]]  # only repair column C using A and B
    assert [r for r in result.rows] == expected

    result = t.imputation(sources=["a", "b", "c"], targets=["b", "c"], method="nearest neighbour", missing={None, "NULL", -1})
    expected = [[1, 2, 3], [1, 2, 3], [5, 5, 5], [5, 5, 5], [6, 6, 6], [6, 6, 6]]  # only repair column C using A and B


def test_replace_missing_values_01():
    sample = [[1, 2, 3], [1, 2, 3], [1, None, None], [5, 5, 5], [6, 6, 6]]

    t = Table()
    t.add_columns(*list("abc"))
    for row in sample:
        t.add_rows(row)

    expected = [r[:] for r in sample]
    expected[2][1] = 2
    expected[2][2] = 3

    result = t.imputation(sources=["a", "b"], targets=["b", "c"], method="nearest neighbour", missing=None)
    assert [r for r in result.rows] == expected

    result = t.imputation(sources=["a", "b", "c"], targets=["a", "b", "c"], method="nearest neighbour", missing=None)
    assert [r for r in result.rows] == expected


def test_replace_missing_values_02():
    sample = [[1, 2, None], [5, 5, 5], [6, 6, 6]]

    t = Table()
    t.add_columns(*list("abc"))
    for row in sample:
        t.add_rows(row)

    expected = [r[:] for r in sample]
    expected[0][-1] = 5
    result = t.imputation(targets=["c"], sources=["a", "b"], method="nearest neighbour")
    assert [r for r in result.rows] == expected

    result = t.imputation(targets=["c"], method="nearest neighbour")
    assert [r for r in result.rows] == expected


def test_replace_missing_values_02b():
    sample = [[1, 2, None], [5, 5, None], [5, 5, 5], [6, 6, 6]]

    t = Table()
    t.add_columns(*list("abc"))
    for row in sample:
        t.add_rows(row)

    expected = [r[:] for r in sample]
    expected[0][-1] = 5
    expected[1][-1] = 5

    result = t.imputation(targets=["c"], method="nearest neighbour")
    assert [r for r in result.rows] == expected


def test_replace_missing_values_02_same_row_twice():
    sample = [[1, 2, None], [1, 2, None], [5, 5, None], [5, 5, 5], [6, 6, 6]]

    t = Table()
    t.add_columns(*list("abc"))
    for row in sample:
        t.add_rows(row)

    expected = [r[:] for r in sample]
    expected[0][-1] = 5
    expected[1][-1] = 5
    expected[2][-1] = 5

    result = t.imputation(targets=["c"], method="nearest neighbour")
    assert [r for r in result.rows] == expected


def test_replace_missing_values_05():
    sample = [
        [None, 1, 2, 3], 
        [0, None, 2, 3], 
        [0, 1, None, 3], 
        [0, 1, 2, None]
    ]

    cols = [str(i) for i, _ in enumerate(sample[0])]
    t = Table()
    t.add_columns(*cols)
    for row in sample:
        t.add_rows(row)

    result = t.imputation(targets=cols, missing=[None], method="nearest neighbour", sources=cols)
    result = result.imputation(targets=cols, method="nearest neighbour", sources=cols)

    expected = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]

    assert [r for r in result.rows] == expected


def test_replace_missing_values_06():
    sample = [[None, None, 2, 3], [0, None, None, 3], [0, 1, None, None], [0, 1, 2, None]]

    cols = [str(i) for i, _ in enumerate(sample[0])]
    t = Table()
    t.add_columns(*cols)
    for row in sample:
        t.add_rows(row)

    result = t.imputation(targets=cols, method="nearest neighbour", sources=cols)
    result = result.imputation(targets=cols, method="nearest neighbour", sources=cols)

    expected = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]

    assert [r for r in result.rows] == expected


def test_replace_missing_values_07():
    sample = [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, None]]

    cols = [str(i) for i, _ in enumerate(sample[0])]
    t = Table()
    t.add_columns(*cols)
    for row in sample:
        t.add_rows(row)

    result = t.imputation(targets=cols, method="nearest neighbour", sources=cols)

    expected = [[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 2]]
    # special case: columns 1 to 5 are unique, so they become indices.
    # the missing value is then subtituted with first match.
    assert [r for r in result.rows] == expected


def test_replace_missing_values_09():
    sample = [["1", 2, True, "this"], ["2", 2, False, "that"], ["3", 1, True, "that"]]

    cols = [str(i) for i, _ in enumerate(sample[0])]
    t = Table()
    t.add_columns(*cols)
    for row in sample:
        t.add_rows(row)

    result = t.imputation(targets=cols, method="nearest neighbour", sources=cols)

    assert [r for r in result.rows] == sample


def test_replace_missing_values_10():
    sample = [["1", 2, False, "this"], ["2", 2, False, "that"], ["3", 1, True, "that"], ["4", 1, True, None]]

    cols = [str(i) for i, _ in enumerate(sample[0])]
    t = Table()
    t.add_columns(*cols)
    for row in sample:
        t.add_rows(row)

    result = t.imputation(targets=cols, method="nearest neighbour", sources=cols)

    expected = [["1", 2, False, "this"], ["2", 2, False, "that"], ["3", 1, True, "that"], ["4", 1, True, "that"]]

    assert [r for r in result.rows] == expected


def test_replace_missing_values_03():
    sample = [
        [1, 2, 1, 1, 1],
        [2, 1, 4, 3, 1],
        [2, 3, 2, None, 1],
        [3, 3, 4, 4, 3],
        [3, 1, 2, 1, 1],
        [4, 3, 3, 3, 1],
        [2, 2, 4, 2, 1],
        [1, 4, 2, None, 3],
        [4, 4, 2, 3, 4],
        [4, 2, 1, 1, 2],
        [4, 4, 4, 1, 1],
        [3, 3, 4, 1, None],
    ]

    t = Table()
    t.add_columns(*list("abcde"))
    for row in sample:
        t.add_rows(row)

    result = t.imputation(targets=["d", "e"], method="nearest neighbour", sources=list("abc"))

    expected = [
        [1, 2, 1, 1, 1],
        [2, 1, 4, 3, 1],
        [2, 3, 2, 1, 1],
        [3, 3, 4, 4, 3],
        [3, 1, 2, 1, 1],
        [4, 3, 3, 3, 1],
        [2, 2, 4, 2, 1],
        [1, 4, 2, 1, 3],
        [4, 4, 2, 3, 4],
        [4, 2, 1, 1, 2],
        [4, 4, 4, 1, 1],
        [3, 3, 4, 1, 3],
    ]

    assert [r for r in result.rows] == expected

    dtypes = result.types()


def test_replace_missing_values_04():
    sample = [
        [5, None, 9, None, 5, 9, 3, 3, 9, None, 8, 6, 5, 4, 2, 8, 9, 3, 9, 9],
        [7, None, None, 4, 7, None, 8, 8, 4, 1, 4, 1, 1, 1, 2, 5, 4, 10, 4, 7],
        [2, 6, 8, 1, 2, 6, 2, 4, 3, 5, 2, 9, 9, 2, None, 3, 8, 7, None, 1],
        [None, 8, 9, 8, 5, 1, 9, 6, 10, 9, 9, 3, 2, 4, 10, 10, 2, 10, 6, None],
        [6, 7, 5, 2, 10, None, None, 4, 3, None, 4, 3, 9, 10, None, 9, 9, 2, None, 2],
        [9, 5, 6, 5, 5, None, 2, 5, 6, 5, 8, 7, 2, 3, 6, 3, 1, 1, 4, 2],
        [9, 7, None, 4, 3, 1, 6, 7, 4, 2, 1, 4, 8, 4, 4, 5, 5, 9, 9, 1],
        [5, 5, 10, 6, 9, 10, 1, 1, 9, 4, 2, None, None, 2, 5, 7, 5, 4, 7, 3],
        [5, 1, 4, 10, 6, 8, 9, 3, 5, 10, 4, 4, 3, 4, None, 2, 10, 7, 4, 1],
        [2, 4, 9, 9, 2, 8, 9, 8, 9, 9, 3, 8, 4, 5, 9, 4, 2, 4, 6, 2],
        [4, 3, 2, 6, 7, 7, 10, None, 10, 1, None, 7, 7, 9, 4, 5, 1, 7, 8, None],
        [None, 7, 1, 7, 8, 5, 4, 6, 7, 7, 2, 9, 6, 2, 1, 4, 1, 4, 2, 4],
    ]

    cols = [str(i) for i, _ in enumerate(sample[0])]
    t = Table()
    t.add_columns(*cols)
    for row in sample:
        t.add_rows(row)

    result = t.imputation(targets=cols, method="nearest neighbour", sources=cols)

    expected = [
        [5, 4, 9, 9, 5, 9, 3, 3, 9, 9, 8, 6, 5, 4, 2, 8, 9, 3, 9, 9],
        [7, 7, 9, 4, 7, 1, 8, 8, 4, 1, 4, 1, 1, 1, 2, 5, 4, 10, 4, 7],
        [2, 6, 8, 1, 2, 6, 2, 4, 3, 5, 2, 9, 9, 2, 9, 3, 8, 7, 6, 1],
        [2, 8, 9, 8, 5, 1, 9, 6, 10, 9, 9, 3, 2, 4, 10, 10, 2, 10, 6, 2],
        [6, 7, 5, 2, 10, 6, 2, 4, 3, 5, 4, 3, 9, 10, 6, 9, 9, 2, 4, 2],
        [9, 5, 6, 5, 5, 8, 2, 5, 6, 5, 8, 7, 2, 3, 6, 3, 1, 1, 4, 2],
        [9, 7, 9, 4, 3, 1, 6, 7, 4, 2, 1, 4, 8, 4, 4, 5, 5, 9, 9, 1],
        [5, 5, 10, 6, 9, 10, 1, 1, 9, 4, 2, 8, 4, 2, 5, 7, 5, 4, 7, 3],
        [5, 1, 4, 10, 6, 8, 9, 3, 5, 10, 4, 4, 3, 4, 9, 2, 10, 7, 4, 1],
        [2, 4, 9, 9, 2, 8, 9, 8, 9, 9, 3, 8, 4, 5, 9, 4, 2, 4, 6, 2],
        [4, 3, 2, 6, 7, 7, 10, 8, 10, 1, 3, 7, 7, 9, 4, 5, 1, 7, 8, 2],
        [2, 7, 1, 7, 8, 5, 4, 6, 7, 7, 2, 9, 6, 2, 1, 4, 1, 4, 2, 4],
    ]

    assert [r for r in result.rows] == expected

def create_dtypes_fragmented_table():
    sample = [
        [1, 2, 1, 1, 1],
        [2, 1, 4, 3, 1],
        [2, 3, 2, None, 1],
        [3, 3, 4, 4, 3],
        [3, 1, 2, 1, 1],
        [4, 3, 3, 3, 1],
        [2, 2, 4, 2, 1],
        [1, 4, 2, None, 3],
        [4, 4, 2, 3, 4],
        [4, 2, 1, 1, 2],
        [4, 4, 4, 1, 1],
        [3, 3, 4, 1, None],
    ]

    t = Table()
    t.add_columns(*list("abcde"))
    for row in sample:
        t.add_rows(row)

    return t

def test_imputation_dtypes_01():
    t = create_dtypes_fragmented_table()

    result = t.imputation(targets=["d", "e"], method="nearest neighbour", sources=list("abc"))

    dtypes = result.types()

    assert dtypes["d"] == {int: 12}
    assert dtypes["e"] == {int: 12}


def test_imputation_dtypes_02():
    t = create_dtypes_fragmented_table()

    result = t.imputation(targets=["d", "e"], method="carry forward", sources=list("abc"))

    dtypes = result.types()

    assert dtypes["d"] == {int: 12}
    assert dtypes["e"] == {int: 12}


def test_imputation_dtypes_03():
    t = create_dtypes_fragmented_table()

    result = t.imputation(targets=["d", "e"], method="mode", sources=list("abc"))

    dtypes = result.types()

    assert dtypes["d"] == {int: 12}
    assert dtypes["e"] == {int: 12}


def test_imputation_dtypes_04():
    t = create_dtypes_fragmented_table()

    result = t.imputation(targets=["d", "e"], method="mean", sources=list("abc"))

    dtypes = result.types()

    assert dtypes["d"] == {int: 10, float: 2}
    assert dtypes["e"] == {int: 11, float: 1}

def create_dtypes_solid_table():
    return Table({
        'a': [1, 2, 2, 3, 3, 4, 2, 1, 4, 4, 4, 3],
        'b': [2, 1, 3, 3, 1, 3, 2, 4, 4, 2, 4, 3],
        'c': [1, 4, 2, 4, 2, 3, 4, 2, 2, 1, 4, 4],
        'd': [1, 3, None, 4, 1, 3, 2, None, 3, 1, 1, 1],
        'e': [1, 1, 1, 3, 1, 1, 1, 3, 4, 2, 1, None]
    })

def test_imputation_dtypes_05():
    t = create_dtypes_solid_table()

    result = t.imputation(targets=["d", "e"], method="nearest neighbour", sources=list("abc"))

    dtypes = result.types()

    assert dtypes["d"] == {int: 12}
    assert dtypes["e"] == {int: 12}


def test_imputation_dtypes_06():
    t = create_dtypes_solid_table()

    result = t.imputation(targets=["d", "e"], method="carry forward", sources=list("abc"))

    dtypes = result.types()

    assert dtypes["d"] == {int: 12}
    assert dtypes["e"] == {int: 12}


def test_imputation_dtypes_07():
    t = create_dtypes_solid_table()

    result = t.imputation(targets=["d", "e"], method="mode", sources=list("abc"))

    dtypes = result.types()

    assert dtypes["d"] == {int: 12}
    assert dtypes["e"] == {int: 12}


def test_imputation_dtypes_08():
    t = create_dtypes_solid_table()

    result = t.imputation(targets=["d", "e"], method="mean", sources=list("abc"))

    dtypes = result.types()

    assert dtypes["d"] == {int: 10, float: 2}
    assert dtypes["e"] == {int: 11, float: 1}


if __name__=="__main__":
    test_replace_missing_values_00()
    test_replace_missing_values_345()
    test_nearest_neighbour_multiple_missing()
    test_replace_missing_values_01()
    test_replace_missing_values_02()
    test_replace_missing_values_02b()