from tablite import Table


def test_diff():
    s1 = [1,2,3,4,5,1,2,3,4,5]
    s2 = [2,2,2,2,3,3,3,3,4,4]
    s3 = [1,1,1,1,1,1,1]

    t1 = Table.from_dict({'A': s1, 'B': s2})
    t2 = Table.from_dict({'A': s3 + s1 + s3, 'B': s3 + s2 + s3})

    t3 = t1.diff(t2)
    t3.show(slice(None))
    assert t3['1st'].count('=') == len(s1)
    assert len(t3['1st']) == len(t2)
    assert t3['1st'].count('-') == len(t2) - len(t1)


