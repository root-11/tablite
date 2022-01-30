from tablite.columns import StoredColumn, InMemoryColumn, CommonColumn


def test_in_memory_column_basics():
    # creating a column remains easy:
    c = InMemoryColumn('A', int, False)

    # so does adding values:
    c.append(44)
    c.append(44)
    assert len(c) == 2

    # and converting to and from json
    d = c.to_json()
    c2 = InMemoryColumn.from_json(d)
    assert len(c2) == 2

    # comparing columns is easy:
    assert c == c2
    assert c != InMemoryColumn('A', str, False)


def test_stored_column_basics():
    # creating a column remains easy:
    c = StoredColumn('A', int, False)

    # so does adding values:
    c.append(44)
    c.append(44)
    assert len(c) == 2

    # and converting to and from json
    d = c.to_json()
    c2 = StoredColumn.from_json(d)
    assert len(c2) == 2

    # comparing columns is easy:
    assert c == c2
    assert c != StoredColumn('A', str, False)

